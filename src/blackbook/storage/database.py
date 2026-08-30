"""SQLite-backed storage for BlackBook.

A single :class:`Database` wraps one SQLite connection and exposes the
persistence operations the rest of the system needs: source/document/chunk
upserts (with content-hash change detection), FTS5 lexical search, and basic
entity/relationship/case persistence used by the knowledge graph and case
context tools.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from blackbook.storage import migrations
from blackbook.storage.models import (
    Case,
    CaseObservation,
    Chunk,
    Document,
    Entity,
    Relationship,
    Source,
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _escape_like(term: str) -> str:
    """Escape LIKE wildcards so user input can't broaden a substring match.

    Backslash is the ESCAPE character in the queries that call this.
    """
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class Database:
    """Wraps a SQLite connection for BlackBook persistence."""

    def __init__(self, path: str | Path, echo: bool = False):
        self.path = str(path)
        self._echo = echo
        # check_same_thread=False so the MCP server thread can share it; the
        # server is single-writer by design.
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        # Wait (up to 5s) for a transient lock instead of failing immediately.
        # WAL already lets readers and a single writer coexist; this covers the
        # brief window when another instance (e.g. a stdio server the editor
        # spawned, or a concurrent CLI ingest) is mid-commit — including the
        # idempotent migrate() write below — so a second instance can start
        # against the same DB rather than crashing with "database is locked".
        self.conn.execute("PRAGMA busy_timeout = 5000")
        migrations.migrate(self.conn)

    # -- lifecycle --------------------------------------------------------

    @contextmanager
    def session(self) -> Iterator["Database"]:
        try:
            yield self
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- sources ----------------------------------------------------------

    def upsert_source(self, src: Source) -> None:
        self.conn.execute(
            """
            INSERT INTO sources(source_id, name, authority, enabled, source_type, url, meta)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id) DO UPDATE SET
                name=excluded.name,
                authority=excluded.authority,
                enabled=excluded.enabled,
                source_type=excluded.source_type,
                url=excluded.url,
                meta=excluded.meta
            """,
            (
                src.source_id,
                src.name,
                src.authority,
                int(src.enabled),
                src.source_type,
                src.url,
                json.dumps(src.meta),
            ),
        )

    def get_source(self, source_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM sources WHERE source_id = ?", (source_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_sources(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM sources ORDER BY source_id").fetchall()
        return [dict(r) for r in rows]

    # -- documents --------------------------------------------------------

    def upsert_document(self, doc: Document) -> int:
        """Insert or update a document, returning its ``doc_id``.

        Change detection is by ``content_hash``: if the hash is unchanged the
        row is left alone and the existing ``doc_id`` is returned.
        """
        existing = self.conn.execute(
            "SELECT doc_id, content_hash FROM documents WHERE source_id = ? AND external_id = ?",
            (doc.source_id, doc.external_id),
        ).fetchone()
        if existing and existing["content_hash"] == doc.content_hash:
            return int(existing["doc_id"])

        self.conn.execute(
            """
            INSERT INTO documents(source_id, external_id, title, url, path,
                                  content_hash, metadata, categories)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id, external_id) DO UPDATE SET
                title=excluded.title,
                url=excluded.url,
                path=excluded.path,
                content_hash=excluded.content_hash,
                metadata=excluded.metadata,
                categories=excluded.categories,
                updated_at=datetime('now')
            """,
            (
                doc.source_id,
                doc.external_id,
                doc.title,
                doc.url,
                doc.path,
                doc.content_hash,
                json.dumps(doc.metadata),
                json.dumps(doc.categories),
            ),
        )
        row = self.conn.execute(
            "SELECT doc_id FROM documents WHERE source_id = ? AND external_id = ?",
            (doc.source_id, doc.external_id),
        ).fetchone()
        assert row is not None
        return int(row["doc_id"])

    def get_document(self, doc_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_document_by_external(self, source_id: str, external_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM documents WHERE source_id = ? AND external_id = ?",
            (source_id, external_id),
        ).fetchone()
        return dict(row) if row else None

    def iter_documents(self, source_ids: list[str] | None = None) -> Iterator[dict]:
        """Yield every document row, optionally scoped to ``source_ids``.

        Used by the knowledge-graph builder to walk the corpus. Ordered by
        ``doc_id`` for a stable, reproducible build.
        """
        if source_ids:
            placeholders = ",".join("?" for _ in source_ids)
            rows = self.conn.execute(
                f"SELECT * FROM documents WHERE source_id IN ({placeholders}) "
                "ORDER BY doc_id",
                source_ids,
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM documents ORDER BY doc_id"
            ).fetchall()
        for r in rows:
            yield dict(r)

    def document_chunks(self, doc_id: int) -> list[dict]:
        """Return a document's chunks (id, ordinal, text, section_path, page)."""
        rows = self.conn.execute(
            "SELECT chunk_id, ordinal, text, section_path, page "
            "FROM chunks WHERE doc_id = ? ORDER BY ordinal, chunk_id",
            (doc_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- chunks -----------------------------------------------------------

    def replace_chunks(self, doc_id: int, chunks: Iterable[Chunk]) -> list[int]:
        """Replace all chunks for a document, returning the new chunk_ids."""
        self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        ids: list[int] = []
        for c in chunks:
            cur = self.conn.execute(
                """
                INSERT INTO chunks(doc_id, ordinal, text, section_path, page,
                                   token_estimate, content_hash, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    c.ordinal,
                    c.text,
                    json.dumps(c.section_path),
                    c.page,
                    c.token_estimate,
                    c.content_hash,
                    json.dumps(c.metadata),
                ),
            )
            ids.append(int(cur.lastrowid))
        return ids

    def get_chunk(self, chunk_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return dict(row) if row else None

    def chunk_count(self, doc_id: int | None = None) -> int:
        if doc_id is None:
            row = self.conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        return int(row["n"])

    # -- lexical search ---------------------------------------------------

    def fts_search(
        self,
        query: str,
        source_ids: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """FTS5 BM25 search over chunks.

        Returns chunk rows joined with document/source metadata and the FTS5
        ``bm25`` rank (lower is better; we negate it so higher is better).
        """
        # The FTS5 MATCH must be the first WHERE condition (it binds `query`).
        conditions = ["chunks_fts MATCH ?"]
        params: list[Any] = [query]
        if source_ids:
            placeholders = ",".join("?" for _ in source_ids)
            conditions.append(f"d.source_id IN ({placeholders})")
            params.extend(source_ids)
        where = "WHERE " + " AND ".join(conditions)
        sql = f"""
            SELECT
                c.chunk_id,
                c.doc_id,
                c.ordinal,
                c.text,
                c.section_path,
                c.page,
                c.token_estimate,
                c.metadata AS chunk_metadata,
                d.source_id,
                d.external_id,
                d.title,
                d.url,
                d.path,
                d.metadata AS doc_metadata,
                d.categories,
                s.name AS source_name,
                s.authority AS source_authority,
                bm25(chunks_fts) AS bm25
            FROM chunks_fts
            JOIN chunks c ON c.chunk_id = chunks_fts.rowid
            JOIN documents d ON d.doc_id = c.doc_id
            JOIN sources s ON s.source_id = d.source_id
            {where}
            ORDER BY bm25 ASC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # -- embeddings (Phase 3, optional semantic layer) --------------------

    def upsert_embedding(
        self, chunk_id: int, model: str, dim: int, vector: bytes
    ) -> None:
        """Store (or replace) the dense vector for a chunk.

        ``vector`` is raw float32 little-endian bytes of length ``dim*4``. The
        model name is recorded so stale vectors (from a different model) can be
        detected and re-embedded.
        """
        self.conn.execute(
            """
            INSERT INTO chunk_embeddings(chunk_id, model, dim, vector)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                model=excluded.model,
                dim=excluded.dim,
                vector=excluded.vector,
                created_at=datetime('now')
            """,
            (int(chunk_id), model, int(dim), sqlite3.Binary(vector)),
        )

    def delete_embeddings(
        self, model: str | None = None, source_ids: list[str] | None = None
    ) -> int:
        """Delete stored embeddings.

        ``model`` scopes to a single model; ``source_ids`` scopes to chunks
        belonging to those sources. With neither, every embedding is removed.
        """
        conditions: list[str] = []
        params: list[Any] = []
        if model is not None:
            conditions.append("model = ?")
            params.append(model)
        if source_ids:
            placeholders = ",".join("?" for _ in source_ids)
            # Restrict to embeddings whose chunk's document is in these sources.
            conditions.append(
                "chunk_id IN ("
                "SELECT c.chunk_id FROM chunks c "
                "JOIN documents d ON d.doc_id = c.doc_id "
                f"WHERE d.source_id IN ({placeholders}))"
            )
            params.extend(source_ids)
        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        cur = self.conn.execute(f"DELETE FROM chunk_embeddings{where}", params)
        return int(cur.rowcount or 0)

    def embedding_count(self, model: str | None = None) -> int:
        if model is None:
            row = self.conn.execute(
                "SELECT COUNT(*) AS n FROM chunk_embeddings"
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) AS n FROM chunk_embeddings WHERE model = ?",
                (model,),
            ).fetchone()
        return int(row["n"])

    def iter_chunks_missing_embeddings(
        self, model: str, source_ids: list[str] | None = None, batch: int = 256
    ) -> Iterator[tuple[int, str]]:
        """Yield ``(chunk_id, text)`` for chunks lacking a current-model vector.

        A chunk needs (re)embedding when it has no row in ``chunk_embeddings``
        for ``model``. Yields in batches to bound memory on large corpora.
        """
        conditions = [
            "NOT EXISTS (SELECT 1 FROM chunk_embeddings e "
            "WHERE e.chunk_id = c.chunk_id AND e.model = ?)"
        ]
        params: list[Any] = [model]
        if source_ids:
            placeholders = ",".join("?" for _ in source_ids)
            conditions.append(f"d.source_id IN ({placeholders})")
            params.extend(source_ids)
        where = " AND ".join(conditions)
        sql = f"""
            SELECT c.chunk_id, c.text
            FROM chunks c
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE {where}
            ORDER BY c.chunk_id
        """
        cur = self.conn.execute(sql, params)
        while True:
            rows = cur.fetchmany(batch)
            if not rows:
                break
            for r in rows:
                yield int(r["chunk_id"]), r["text"]

    def load_embeddings(
        self, model: str, source_ids: list[str] | None = None
    ) -> tuple[list[int], list[bytes]]:
        """Load all stored vectors for ``model`` (optionally source-filtered).

        Returns parallel lists of ``chunk_ids`` and raw ``vector`` blobs. The
        caller decodes the blobs into a matrix. Kept deliberately dumb so the
        storage layer has no numpy dependency.
        """
        conditions = ["e.model = ?"]
        params: list[Any] = [model]
        if source_ids:
            placeholders = ",".join("?" for _ in source_ids)
            conditions.append(f"d.source_id IN ({placeholders})")
            params.extend(source_ids)
        where = " AND ".join(conditions)
        sql = f"""
            SELECT e.chunk_id, e.vector
            FROM chunk_embeddings e
            JOIN chunks c ON c.chunk_id = e.chunk_id
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE {where}
            ORDER BY e.chunk_id
        """
        rows = self.conn.execute(sql, params).fetchall()
        ids = [int(r["chunk_id"]) for r in rows]
        vecs = [bytes(r["vector"]) for r in rows]
        return ids, vecs

    def hydrate_chunks(self, chunk_ids: list[int]) -> dict[int, dict]:
        """Return chunk+document+source metadata for the given chunk_ids.

        Used by semantic retrieval to build full hits (the vector index only
        knows chunk_ids). Missing ids are simply absent from the result.
        """
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        sql = f"""
            SELECT
                c.chunk_id,
                c.doc_id,
                c.ordinal,
                c.text,
                c.section_path,
                c.page,
                c.token_estimate,
                c.metadata AS chunk_metadata,
                d.source_id,
                d.external_id,
                d.title,
                d.url,
                d.path,
                d.metadata AS doc_metadata,
                d.categories,
                s.name AS source_name,
                s.authority AS source_authority
            FROM chunks c
            JOIN documents d ON d.doc_id = c.doc_id
            JOIN sources s ON s.source_id = d.source_id
            WHERE c.chunk_id IN ({placeholders})
        """
        rows = self.conn.execute(sql, chunk_ids).fetchall()
        return {int(r["chunk_id"]): dict(r) for r in rows}

    # -- entities / relationships -----------------------------------------
    def upsert_entity(self, e: Entity) -> int:
        self.conn.execute(
            """
            INSERT INTO entities(name, entity_type, description, meta)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(name, entity_type) DO UPDATE SET
                description=excluded.description,
                meta=excluded.meta
            """,
            (e.name, e.entity_type, e.description, json.dumps(e.meta)),
        )
        row = self.conn.execute(
            "SELECT entity_id FROM entities WHERE name = ? AND entity_type = ?",
            (e.name, e.entity_type),
        ).fetchone()
        assert row is not None
        return int(row["entity_id"])

    def get_entity(self, name: str, entity_type: str | None = None) -> dict | None:
        if entity_type is None:
            row = self.conn.execute(
                "SELECT * FROM entities WHERE name = ? LIMIT 1", (name,)
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM entities WHERE name = ? AND entity_type = ?",
                (name, entity_type),
            ).fetchone()
        return dict(row) if row else None

    def add_relationship(self, r: Relationship) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO relationships(subject_id, predicate, object_id,
                                      evidence_doc_id, confidence, inferred)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                r.subject_id,
                r.predicate,
                r.object_id,
                r.evidence_doc_id,
                r.confidence,
                int(r.inferred),
            ),
        )
        return int(cur.lastrowid)

    def get_entity_by_id(self, entity_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM entities WHERE entity_id = ?", (entity_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_entities(self, entity_type: str | None = None) -> list[dict]:
        if entity_type is None:
            rows = self.conn.execute(
                "SELECT * FROM entities ORDER BY entity_type, name"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM entities WHERE entity_type = ? ORDER BY name",
                (entity_type,),
            ).fetchall()
        return [dict(r) for r in rows]

    def find_entities(self, name_like: str, entity_type: str | None = None) -> list[dict]:
        """Case-insensitive substring match on entity name.

        ``name_like`` is matched as a LIKE pattern with the wildcards escaped so
        user-supplied ``%``/``_`` cannot broaden the search.
        """
        pattern = "%" + _escape_like(name_like) + "%"
        if entity_type is None:
            rows = self.conn.execute(
                "SELECT * FROM entities WHERE name LIKE ? ESCAPE '\\' "
                "ORDER BY entity_type, name",
                (pattern,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM entities WHERE name LIKE ? ESCAPE '\\' "
                "AND entity_type = ? ORDER BY name",
                (pattern, entity_type),
            ).fetchall()
        return [dict(r) for r in rows]

    def entity_relationships(
        self, entity_id: int, predicate: str | None = None
    ) -> list[dict]:
        """Return every edge touching ``entity_id``, joined to the neighbour and
        (when present) the evidence document + its source.

        Each row carries ``direction`` (``out`` when the entity is the subject,
        ``in`` when it is the object), the neighbour's id/name/type, and the
        evidence provenance so callers can cite a real document — never a
        fabricated reference. Optionally filtered to a single ``predicate``.
        """
        # Outgoing: this entity is the subject; neighbour is the object.
        # Incoming: this entity is the object; neighbour is the subject.
        sql = """
            SELECT r.rel_id, r.predicate, 'out' AS direction,
                   r.confidence, r.inferred, r.evidence_doc_id,
                   o.entity_id AS other_id, o.name AS other_name,
                   o.entity_type AS other_type, o.description AS other_description,
                   d.title AS evidence_title, d.url AS evidence_url,
                   d.external_id AS evidence_external_id,
                   s.source_id AS evidence_source_id, s.name AS evidence_source_name,
                   s.authority AS evidence_authority
            FROM relationships r
            JOIN entities o ON o.entity_id = r.object_id
            LEFT JOIN documents d ON d.doc_id = r.evidence_doc_id
            LEFT JOIN sources s ON s.source_id = d.source_id
            WHERE r.subject_id = ?
            UNION ALL
            SELECT r.rel_id, r.predicate, 'in' AS direction,
                   r.confidence, r.inferred, r.evidence_doc_id,
                   sub.entity_id AS other_id, sub.name AS other_name,
                   sub.entity_type AS other_type, sub.description AS other_description,
                   d.title AS evidence_title, d.url AS evidence_url,
                   d.external_id AS evidence_external_id,
                   s.source_id AS evidence_source_id, s.name AS evidence_source_name,
                   s.authority AS evidence_authority
            FROM relationships r
            JOIN entities sub ON sub.entity_id = r.subject_id
            LEFT JOIN documents d ON d.doc_id = r.evidence_doc_id
            LEFT JOIN sources s ON s.source_id = d.source_id
            WHERE r.object_id = ?
        """
        rows = self.conn.execute(sql, (entity_id, entity_id)).fetchall()
        out = [dict(r) for r in rows]
        if predicate is not None:
            out = [r for r in out if r["predicate"] == predicate]
        # Highest-confidence edges first, then stable by rel_id.
        out.sort(key=lambda r: (-float(r["confidence"]), int(r["rel_id"])))
        return out

    def clear_graph(self) -> None:
        """Remove all graph entities and relationships (idempotent rebuild).

        Cases/observations are intentionally left untouched — they are a
        separate, user-authored layer, not derived from ingestion.
        """
        self.conn.execute("DELETE FROM relationships")
        self.conn.execute("DELETE FROM entities")

    # -- cases ------------------------------------------------------------

    def upsert_case(self, case: Case) -> int:
        self.conn.execute(
            """
            INSERT INTO cases(name, target, platform, meta)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                target=excluded.target,
                platform=excluded.platform,
                meta=excluded.meta,
                updated_at=datetime('now')
            """,
            (case.name, case.target, case.platform, json.dumps(case.meta)),
        )
        row = self.conn.execute(
            "SELECT case_id FROM cases WHERE name = ?", (case.name,)
        ).fetchone()
        assert row is not None
        return int(row["case_id"])

    def get_case(self, name: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM cases WHERE name = ?", (name,)
        ).fetchone()
        return dict(row) if row else None

    def add_observation(self, obs: CaseObservation) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO case_observations(case_id, kind, text, status)
            VALUES (?, ?, ?, ?)
            """,
            (obs.case_id, obs.kind, obs.text, obs.status),
        )
        return int(cur.lastrowid)

    def list_observations(self, case_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM case_observations WHERE case_id = ? ORDER BY created_at, obs_id",
            (case_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_cases(self) -> list[dict]:
        """All cases, most-recently-updated first, with observation counts."""
        rows = self.conn.execute(
            """
            SELECT c.*, COUNT(o.obs_id) AS observation_count
            FROM cases c
            LEFT JOIN case_observations o ON o.case_id = c.case_id
            GROUP BY c.case_id
            ORDER BY c.updated_at DESC, c.case_id DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def get_observation(self, obs_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM case_observations WHERE obs_id = ?", (obs_id,)
        ).fetchone()
        return dict(row) if row else None

    def set_observation_status(self, obs_id: int, status: str) -> bool:
        """Update an observation's status; True when a row was changed."""
        cur = self.conn.execute(
            "UPDATE case_observations SET status = ? WHERE obs_id = ?",
            (status, obs_id),
        )
        return cur.rowcount > 0

    # -- stats / maintenance ----------------------------------------------

    def counts(self) -> dict[str, int]:
        def one(sql: str) -> int:
            row = self.conn.execute(sql).fetchone()
            return int(row[0])

        return {
            "sources": one("SELECT COUNT(*) FROM sources"),
            "documents": one("SELECT COUNT(*) FROM documents"),
            "chunks": one("SELECT COUNT(*) FROM chunks"),
            "embeddings": one("SELECT COUNT(*) FROM chunk_embeddings"),
            "entities": one("SELECT COUNT(*) FROM entities"),
            "relationships": one("SELECT COUNT(*) FROM relationships"),
            "cases": one("SELECT COUNT(*) FROM cases"),
        }

    def rebuild_fts(self) -> None:
        """Rebuild the FTS index from the chunks table."""
        self.conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")

    def optimize_fts(self) -> None:
        self.conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
