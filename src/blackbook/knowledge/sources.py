"""Source resolution: turn a citation back into the exact source excerpt.

``knowledge_search`` returns references; ``knowledge_source`` resolves them to
the smallest useful excerpt. This is what makes citations *verifiable* — the
returned text always comes from an actually-indexed chunk, never from memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from blackbook.storage.database import Database


@dataclass
class SourceExcerpt:
    chunk_id: int
    doc_id: int
    title: str
    source_id: str
    source_name: str
    authority: str
    text: str
    section_path: list[str] = field(default_factory=list)
    url: str | None = None
    path: str | None = None
    page: int | None = None
    ordinal: int = 0


def get_chunk_excerpt(db: Database, chunk_id: int) -> SourceExcerpt | None:
    row = db.conn.execute(
        """
        SELECT c.chunk_id, c.doc_id, c.ordinal, c.text, c.section_path, c.page,
               d.title, d.url, d.path, d.source_id, s.name AS source_name, s.authority
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
        JOIN sources s ON s.source_id = d.source_id
        WHERE c.chunk_id = ?
        """,
        (chunk_id,),
    ).fetchone()
    return _row_to_excerpt(row) if row else None


def list_document_chunks(db: Database, doc_id: int) -> list[SourceExcerpt]:
    rows = db.conn.execute(
        """
        SELECT c.chunk_id, c.doc_id, c.ordinal, c.text, c.section_path, c.page,
               d.title, d.url, d.path, d.source_id, s.name AS source_name, s.authority
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
        JOIN sources s ON s.source_id = d.source_id
        WHERE c.doc_id = ?
        ORDER BY c.ordinal
        """,
        (doc_id,),
    ).fetchall()
    return [_row_to_excerpt(r) for r in rows]


def find_document(
    db: Database, *, source_id: str | None = None, external_id: str | None = None,
    doc_id: int | None = None, title_like: str | None = None,
) -> dict | None:
    if doc_id is not None:
        return db.get_document(doc_id)
    if source_id and external_id:
        return db.get_document_by_external(source_id, external_id)
    if title_like:
        row = db.conn.execute(
            "SELECT * FROM documents WHERE title LIKE ? ORDER BY title LIMIT 1",
            (f"%{title_like}%",),
        ).fetchone()
        return dict(row) if row else None
    return None


def _row_to_excerpt(row) -> SourceExcerpt:
    import json

    section = row["section_path"]
    if isinstance(section, str):
        try:
            section = json.loads(section)
        except Exception:
            section = []
    return SourceExcerpt(
        chunk_id=int(row["chunk_id"]),
        doc_id=int(row["doc_id"]),
        ordinal=int(row["ordinal"]),
        text=row["text"],
        section_path=section or [],
        page=row["page"],
        title=row["title"],
        url=row["url"],
        path=row["path"],
        source_id=row["source_id"],
        source_name=row["source_name"],
        authority=row["authority"],
    )
