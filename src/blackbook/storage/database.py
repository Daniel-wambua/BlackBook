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
import threading
import time
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
    QueryLogEntry,
    Relationship,
    Source,
)


# The ``meta`` key holding the record of the last knowledge-graph build. Read by
# the incremental rebuild to tell a graph that already matches the corpus from
# one that needs rebuilding. Dropped by clear_graph() with the graph itself.
_GRAPH_BUILD_KEY = "graph_build"


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
        # Serializes write sessions. Under the HTTP transport, FastMCP runs
        # sync tools in a threadpool, so concurrent requests share this one
        # connection; without the lock one request's rollback could abort
        # another's in-flight session (commit/rollback is connection-wide).
        # RLock so an accidental nested session() in the same thread can't
        # self-deadlock; reads need no lock — WAL readers don't block the writer.
        self._session_lock = threading.RLock()
        # Cached counts for the poll-shaped readouts (/health and the landing
        # page), which a monitor or a browser may fetch on a timer. Holds
        # (monotonic_timestamp, counts) or None. Cleared by any write this
        # process commits; see counts_cached().
        self._counts_cache: tuple[float, dict[str, int]] | None = None
        # check_same_thread=False so the MCP server thread can share it; the
        # server is single-writer by design.
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        # Allow a short wait when another process is finishing a write. This is
        # important because the CLI ingestion process and stdio MCP server may
        # legitimately share the same SQLite database. WAL already lets readers
        # and a single writer coexist; the timeout covers the brief window when
        # another instance (e.g. a stdio server the editor spawned, or a
        # concurrent CLI ingest) is mid-commit — including the idempotent
        # migrate() write below — so a second instance can start against the
        # same DB rather than crashing with "database is locked".
        self.conn.execute("PRAGMA busy_timeout = 10000")
        self.conn.execute("PRAGMA journal_mode = WAL")
        migrations.migrate(self.conn)
        # migrations.migrate writes schema metadata but intentionally does not
        # own the connection lifecycle. Commit it here so a long-lived MCP
        # server cannot hold the migration transaction open and block a later
        # CLI invocation such as `blackbook ingest`.
        self.conn.commit()

    # -- lifecycle --------------------------------------------------------

    @contextmanager
    def session(self) -> Iterator["Database"]:
        with self._session_lock:
            try:
                yield self
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
            # Only after a committed write: anything we just changed must show
            # up in the next counts_cached() call, not after the TTL. A rolled
            # back session changed nothing, so it leaves the cache alone.
            self._counts_cache = None

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

    def mark_source_fetched(self, source_id: str, version: str | None = None) -> int:
        """Stamp a source as fetched now, at ``version`` when one is known.

        Returns the number of rows changed, which is 0 for a source that was
        never registered. That is a no-op rather than an insert on purpose: the
        caller (ingest) registers the source immediately before running it, so
        a missing row means something is wrong upstream, and inventing a row
        with a placeholder name here would hide that rather than show it.

        ``version`` is only written when the caller has one. A source with no
        revision marker (a website crawl) leaves whatever is stored alone, so a
        best-effort run cannot erase a commit recorded by an earlier one.

        The timestamp comes from SQLite's clock, so every row in this schema
        agrees on what "now" is, and it is UTC like every other timestamp here.
        """
        with self.session():
            if version is None:
                cur = self.conn.execute(
                    "UPDATE sources SET last_fetched = datetime('now') "
                    "WHERE source_id = ?",
                    (source_id,),
                )
            else:
                cur = self.conn.execute(
                    "UPDATE sources SET last_fetched = datetime('now'), version = ? "
                    "WHERE source_id = ?",
                    (version, source_id),
                )
        return int(cur.rowcount)

    def source_index_counts(self) -> dict[str, dict[str, int]]:
        """Return indexed document/chunk counts grouped by source."""
        rows = self.conn.execute(
            """
            SELECT s.source_id,
                   COUNT(DISTINCT d.doc_id) AS documents,
                   COUNT(c.chunk_id) AS chunks
            FROM sources s
            LEFT JOIN documents d ON d.source_id = s.source_id
            LEFT JOIN chunks c ON c.doc_id = d.doc_id
            GROUP BY s.source_id
            """
        ).fetchall()
        return {
            str(row["source_id"]): {
                "documents": int(row["documents"]),
                "chunks": int(row["chunks"]),
            }
            for row in rows
        }

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

    def refresh_document_citation(
        self,
        source_id: str,
        external_id: str,
        *,
        title: str,
        url: str | None,
        path: str | None,
        categories: list[str],
    ) -> None:
        """Update citation metadata (title/url/path/categories) on a document
        whose content is otherwise unchanged — e.g. a source re-publishes its
        pages under new permalinks. Chunks and their embeddings are untouched,
        so this never triggers re-embedding or new chunk ids.
        """
        self.conn.execute(
            """
            UPDATE documents
            SET title = ?, url = ?, path = ?, categories = ?, updated_at = datetime('now')
            WHERE source_id = ? AND external_id = ?
            """,
            (
                title,
                url,
                path,
                json.dumps(categories),
                source_id,
                external_id,
            ),
        )

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

        ``None`` means every document; an empty list means none. Used by the
        knowledge-graph builder to walk the corpus. Ordered by ``doc_id`` for
        a stable, reproducible build.
        """
        if source_ids is not None and not source_ids:
            return
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

    def chunk_hashes(
        self,
        source_ids: list[str] | None = None,
        exclude_source_ids: list[str] | None = None,
        exclude_doc_id: int | None = None,
    ) -> set[str]:
        """Return normalized chunk hashes, optionally scoped by source."""
        conditions: list[str] = []
        params: list[object] = []
        if source_ids:
            placeholders = ",".join("?" for _ in source_ids)
            conditions.append(f"d.source_id IN ({placeholders})")
            params.extend(source_ids)
        if exclude_source_ids:
            placeholders = ",".join("?" for _ in exclude_source_ids)
            conditions.append(f"d.source_id NOT IN ({placeholders})")
            params.extend(exclude_source_ids)
        if exclude_doc_id is not None:
            conditions.append("c.doc_id != ?")
            params.append(exclude_doc_id)
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self.conn.execute(
            "SELECT c.content_hash FROM chunks c "
            "JOIN documents d ON d.doc_id = c.doc_id" + where,
            params,
        ).fetchall()
        return {str(row["content_hash"]) for row in rows}

    # -- chunks -----------------------------------------------------------

    def _bump_embeddings_version(self) -> None:
        """Invalidate in-process embedding caches (retrieval/semantic.py).

        Bumped by every change to the chunk set or the embeddings themselves.
        Semantic retrieval keys its matrix cache on this counter rather than
        the embedding *count*, so a delete+add that leaves the count unchanged
        still invalidates stale vectors.
        """
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES('embeddings_version', '1') "
            "ON CONFLICT(key) DO UPDATE SET "
            "value = CAST(CAST(meta.value AS INTEGER) + 1 AS TEXT)"
        )

    def embeddings_version(self) -> int:
        """Monotonic counter of chunk/embedding changes; starts at 0."""
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = 'embeddings_version'"
        ).fetchone()
        return int(row["value"]) if row else 0

    def replace_chunks(self, doc_id: int, chunks: Iterable[Chunk]) -> list[int]:
        """Replace all chunks for a document, returning the new chunk_ids."""
        self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        # The DELETE cascade-removes this document's embeddings, so cached
        # vector matrices are stale even before the new chunks are inserted.
        self._bump_embeddings_version()
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

    # FTS5 column weights for ``bm25()``, in the virtual table's declared column
    # order: (title, section, text). A term matching a document's *title* is far
    # stronger evidence of relevance than the same term buried in body prose, so
    # title and heading matches are weighted above the body. Unweighted
    # ``bm25()`` treats all three columns identically, which let long body
    # chunks outrank documents actually *about* the query term.
    BM25_WEIGHTS: tuple[float, float, float] = (5.0, 2.0, 1.0)

    def fts_search(
        self,
        query: str,
        source_ids: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
        platform: str | None = None,
        categories: list[str] | None = None,
        bm25_weights: tuple[float, ...] | None = None,
    ) -> list[dict]:
        """FTS5 BM25 search over chunks.

        Returns chunk rows joined with document/source metadata and the FTS5
        ``bm25`` rank (lower is better; we negate it so higher is better).

        ``bm25_weights`` overrides :attr:`BM25_WEIGHTS` for this call. Weights
        change only the *ranking*, not the sign convention: SQLite's ``bm25()``
        still returns smaller-is-better values, so the caller's negation and
        normalization are unaffected.

        Ties on ``bm25`` are broken by ``chunk_id``. This matters more than it
        looks: a term present in *every* document (IDF 0, e.g. a single-document
        corpus or a very common term) scores exactly 0.0 for every chunk, and
        without a tie-break the order of those rows is whatever the FTS index
        happens to yield -- unstable across index rebuilds and sensitive to
        query planning. ``chunk_id`` is stable and follows ingest order.

        ``platform`` and ``categories`` are *hard* filters over the document's
        category tags (case-insensitive): a hit that doesn't carry the tag is
        excluded, not merely down-ranked.
        """
        # The FTS5 MATCH must be the first WHERE condition (it binds `query`).
        # ``source_ids is None`` means "no filter"; an empty list means "no
        # sources are in scope" and matches nothing (never widens to all).
        if source_ids is not None and not source_ids:
            return []
        conditions = ["chunks_fts MATCH ?"]
        params: list[Any] = [query]
        if source_ids:
            placeholders = ",".join("?" for _ in source_ids)
            conditions.append(f"d.source_id IN ({placeholders})")
            params.extend(source_ids)
        if platform:
            conditions.append(
                "EXISTS (SELECT 1 FROM json_each(d.categories) je "
                "WHERE lower(je.value) = ?)"
            )
            params.append(platform.lower())
        cats = [c.lower() for c in (categories or []) if c.strip()]
        if cats:
            placeholders = ",".join("?" for _ in cats)
            conditions.append(
                "EXISTS (SELECT 1 FROM json_each(d.categories) je "
                f"WHERE lower(je.value) IN ({placeholders}))"
            )
            params.extend(cats)
        where = "WHERE " + " AND ".join(conditions)
        # ``bm25()``'s weight arguments are bound as literals rather than as
        # placeholders (SQLite evaluates FTS5 auxiliary-function arguments at
        # prepare time). ``float()`` coerces each weight, so nothing but a
        # numeric literal can reach the SQL text.
        weights = bm25_weights if bm25_weights is not None else self.BM25_WEIGHTS
        weight_sql = ", ".join(f"{float(w):g}" for w in weights)
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
                bm25(chunks_fts, {weight_sql}) AS bm25
            FROM chunks_fts
            JOIN chunks c ON c.chunk_id = chunks_fts.rowid
            JOIN documents d ON d.doc_id = c.doc_id
            JOIN sources s ON s.source_id = d.source_id
            {where}
            ORDER BY bm25 ASC, c.chunk_id ASC
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
        self._bump_embeddings_version()

    def delete_embeddings(
        self, model: str | None = None, source_ids: list[str] | None = None
    ) -> int:
        """Delete stored embeddings.

        ``model`` scopes to a single model; ``source_ids`` scopes to chunks
        belonging to those sources. With neither, every embedding is removed.
        An empty ``source_ids`` list matches nothing (never widens to all).
        """
        if source_ids is not None and not source_ids:
            return 0
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
        if cur.rowcount:
            self._bump_embeddings_version()
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
        An empty ``source_ids`` list matches nothing (never widens to all).
        """
        if source_ids is not None and not source_ids:
            return
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
        storage layer has no numpy dependency. An empty ``source_ids`` list
        matches nothing (never widens to all).
        """
        if source_ids is not None and not source_ids:
            return [], []
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
                                      evidence_doc_id, confidence, inferred, support)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r.subject_id,
                r.predicate,
                r.object_id,
                r.evidence_doc_id,
                r.confidence,
                int(r.inferred),
                int(r.support),
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

    def entity_count(self) -> int:
        """Number of graph entities.

        ``counts()`` answers this too, but it recomputes a ``COUNT(*)`` over the
        chunk table as part of the same dict, so it is the wrong call for a
        reader that only needs to know whether a graph exists at all. This is a
        count over the entities table alone.
        """
        return int(self.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0])

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
                   r.confidence, r.inferred, r.evidence_doc_id, r.support,
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
                   r.confidence, r.inferred, r.evidence_doc_id, r.support,
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
        # Highest-confidence edges first. ``support`` breaks ties: a co-occurrence
        # claim backed by 400 documents is a stronger signal than the same claim
        # backed by one, and confidence alone cannot separate them because every
        # co-occurrence edge carries the same tier. rel_id keeps the order stable.
        out.sort(
            key=lambda r: (
                -float(r["confidence"]),
                -int(r["support"]),
                int(r["rel_id"]),
            )
        )
        return out

    def clear_graph(self) -> None:
        """Remove all graph entities and relationships (idempotent rebuild).

        Cases/observations are intentionally left untouched — they are a
        separate, user-authored layer, not derived from ingestion. So is the
        ``document_graph`` term cache, which stays valid: what it holds is a
        function of the documents, not of the graph. The build record is
        dropped, because the graph it described no longer exists.
        """
        self.conn.execute("DELETE FROM relationships")
        self.conn.execute("DELETE FROM entities")
        self.conn.execute("DELETE FROM meta WHERE key = ?", (_GRAPH_BUILD_KEY,))

    def has_graph(self) -> bool:
        """True when at least one relationship is stored.

        Used by the incremental rebuild to tell a graph that is current from one
        that was cleared out from under it, which no document fingerprint can
        show.
        """
        return self.conn.execute("SELECT 1 FROM relationships LIMIT 1").fetchone() is not None

    def relationship_counts_by_predicate(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT predicate, COUNT(*) FROM relationships GROUP BY predicate"
        ).fetchall()
        return {str(r[0]): int(r[1]) for r in rows}

    # -- knowledge-graph term cache ---------------------------------------

    def document_graph_cache(self) -> dict[int, tuple[str, str]]:
        """doc_id -> (fingerprint, terms JSON), for the incremental rebuild."""
        rows = self.conn.execute(
            "SELECT doc_id, fingerprint, terms FROM document_graph"
        ).fetchall()
        return {int(r[0]): (str(r[1]), str(r[2])) for r in rows}

    def replace_document_graph(
        self,
        rows: Iterable[tuple[int, str, str]],
        source_ids: list[str] | None = None,
    ) -> None:
        """Store the term cache for the documents just processed.

        Scoped to ``source_ids`` when the rebuild was scoped, so a rebuild of one
        source cannot discard the cache for the others.
        """
        if source_ids:
            placeholders = ",".join("?" for _ in source_ids)
            self.conn.execute(
                "DELETE FROM document_graph WHERE doc_id IN "
                f"(SELECT doc_id FROM documents WHERE source_id IN ({placeholders}))",
                tuple(source_ids),
            )
        else:
            self.conn.execute("DELETE FROM document_graph")
        self.conn.executemany(
            "INSERT INTO document_graph(doc_id, fingerprint, terms) VALUES (?, ?, ?)",
            list(rows),
        )

    def get_graph_build(self) -> dict | None:
        """The record of the last graph build, or None if there is none."""
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = ?", (_GRAPH_BUILD_KEY,)
        ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(row[0])
        except (TypeError, ValueError):  # pragma: no cover - only a corrupt row
            return None
        return payload if isinstance(payload, dict) else None

    def set_graph_build(self, payload: dict) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (_GRAPH_BUILD_KEY, json.dumps(payload)),
        )

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

    # -- query log ---------------------------------------------------------

    def log_query(self, entry: QueryLogEntry, max_entries: int = 0) -> int:
        """Append one query-log entry and return its id.

        Runs in a ``session()`` so the insert takes the same write lock as every
        other writer and commits immediately: a long-lived MCP server that left
        this transaction open would block a concurrent CLI ingest, which is the
        failure the session lock exists to prevent.

        ``max_entries`` above zero prunes the log to that many newest rows after
        inserting. The prune is a single indexed range delete, and it runs here
        rather than on a schedule because the insert is the only moment the log
        can grow. The count that gates it is over a table the bound keeps small.
        """
        with self.session():
            cur = self.conn.execute(
                """
                INSERT INTO query_log(
                    tool, query, mode, sources, result_count,
                    top_score, latency_ms, backend, degraded
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.tool,
                    entry.query,
                    entry.mode,
                    json.dumps(entry.sources),
                    entry.result_count,
                    entry.top_score,
                    entry.latency_ms,
                    entry.backend,
                    1 if entry.degraded else 0,
                ),
            )
            query_id = int(cur.lastrowid)
            if max_entries > 0:
                self._prune_query_log(max_entries)
        return query_id

    def _prune_query_log(self, max_entries: int) -> int:
        """Drop all but the ``max_entries`` newest rows. Returns rows removed.

        Deletes *below* the oldest id of the newest ``max_entries``, so the
        statement is an indexed range scan on the primary key rather than a set
        difference over the table. The bound is exclusive: ``<=`` would take the
        cut row as well and leave one entry fewer than asked for. Caller holds
        the session.
        """
        count = int(self.conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0])
        if count <= max_entries:
            return 0
        cur = self.conn.execute(
            """
            DELETE FROM query_log WHERE query_id < (
                SELECT MIN(query_id) FROM (
                    SELECT query_id FROM query_log
                    ORDER BY query_id DESC LIMIT ?
                )
            )
            """,
            (max_entries,),
        )
        return int(cur.rowcount)

    def list_queries(
        self, limit: int = 50, empty_only: bool = False
    ) -> list[dict]:
        """Recent query-log entries, newest first.

        ``empty_only`` keeps only entries that returned nothing, which is the
        view worth reading: those are the phrasings the corpus could not answer.
        """
        where = "WHERE result_count = 0" if empty_only else ""
        rows = self.conn.execute(
            f"""
            SELECT * FROM query_log {where}
            ORDER BY query_id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def query_log_stats(self) -> dict:
        """Aggregate query-log counts, or zeros when nothing is logged.

        ``empty_rate`` is computed over logged entries with a result count, so
        an empty log reports 0.0 rather than a meaningless division.
        """
        row = self.conn.execute(
            """
            SELECT COUNT(*)                                   AS total,
                   COALESCE(SUM(result_count = 0), 0)         AS empty,
                   MIN(created_at)                            AS first_at,
                   MAX(created_at)                            AS last_at,
                   AVG(latency_ms)                            AS avg_latency_ms
            FROM query_log
            """
        ).fetchone()
        total = int(row["total"] or 0)
        empty = int(row["empty"] or 0)
        return {
            "total": total,
            "empty": empty,
            "empty_rate": (empty / total) if total else 0.0,
            "first_at": row["first_at"],
            "last_at": row["last_at"],
            "avg_latency_ms": float(row["avg_latency_ms"] or 0.0),
        }

    def clear_query_log(self) -> int:
        """Delete the whole query log. Returns rows removed."""
        with self.session():
            cur = self.conn.execute("DELETE FROM query_log")
        return int(cur.rowcount)

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

    # A cached count reading is good for this long. See counts_cached().
    COUNTS_TTL_SECONDS = 5.0

    def counts_cached(self, max_age: float | None = None) -> dict[str, int]:
        """Counts from a short-lived cache, for the meta readouts.

        ``counts()`` scans the chunk table, which is the largest in the schema:
        about 4ms against a half-million-chunk corpus. That is nothing once and
        wasteful per request on ``/health``, which a monitor may poll on a
        timer, and on the landing page, which a browser retries. Those are
        status readouts, not something a decision hangs on, so a bounded
        five-second staleness is a fair trade for not re-counting half a
        million rows on every poll.

        Two things keep it honest. Any write *this* process commits clears the
        cache outright, so a server that ingests reports the new numbers on its
        very next call. A write from *another* process (a CLI ingest while the
        server is up) is picked up within ``max_age``, because nothing here can
        observe it. Where the exact number matters, call ``counts()``.
        """
        max_age = self.COUNTS_TTL_SECONDS if max_age is None else max_age
        cached = self._counts_cache
        if cached is not None and (time.monotonic() - cached[0]) < max_age:
            return dict(cached[1])
        fresh = self.counts()
        self._counts_cache = (time.monotonic(), fresh)
        return dict(fresh)

    def rebuild_fts(self) -> None:
        """Rebuild the FTS index from the chunks table."""
        self.conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")

    def optimize_fts(self) -> None:
        self.conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
