"""Lexical retrieval over SQLite FTS5.

This is the default, always-available retrieval path. It uses FTS5's BM25
ranking and produces a snippet for each hit. Query text is normalized and
escaped so arbitrary user input cannot inject FTS5 query syntax.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from blackbook.storage.database import Database

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+\-\.]*")


def normalize_query(query: str) -> str:
    """Normalize a free-text query into a safe FTS5 MATCH expression.

    We extract word tokens and OR-join quoted terms so that:
      * arbitrary punctuation can't break the FTS5 parser
      * multi-word queries behave as "any of these terms, ranked by BM25"
    """
    tokens = _TOKEN_RE.findall(query.lower())
    if not tokens:
        return '""'
    # Deduplicate while preserving order, then quote each token.
    seen: dict[str, None] = {}
    for t in tokens:
        seen[t] = None
    return " OR ".join(f'"{t}"' for t in seen)


@dataclass
class LexicalHit:
    chunk_id: int
    doc_id: int
    text: str
    title: str
    source_id: str
    source_name: str
    authority: str
    bm25: float
    score: float = 0.0
    section_path: list[str] = field(default_factory=list)
    url: str | None = None
    path: str | None = None
    page: int | None = None
    snippet: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class LexicalRetriever:
    def __init__(self, db: Database):
        self.db = db

    def search(
        self,
        query: str,
        *,
        source_ids: list[str] | None = None,
        limit: int = 50,
        platform: str | None = None,
        categories: list[str] | None = None,
    ) -> list[LexicalHit]:
        match = normalize_query(query)
        if match == '""':
            return []
        rows = self.db.fts_search(
            match,
            source_ids=source_ids,
            limit=limit,
            platform=platform,
            categories=categories,
        )
        hits: list[LexicalHit] = []
        for r in rows:
            # FTS5 bm25() returns 0 for a non-match and increasingly *negative*
            # values as a chunk matches better (lower is better). Convert to a
            # bounded score in [0, 1) that is monotonically *increasing* in match
            # strength, so it orders the same way bm25 does and is directly
            # comparable to the semantic cosine score during merge/rerank.
            bm25 = float(r["bm25"])
            strength = max(0.0, -bm25)
            score = strength / (1.0 + strength)
            hits.append(
                LexicalHit(
                    chunk_id=int(r["chunk_id"]),
                    doc_id=int(r["doc_id"]),
                    text=r["text"],
                    title=r["title"],
                    source_id=r["source_id"],
                    source_name=r["source_name"],
                    authority=r["source_authority"],
                    bm25=bm25,
                    score=score,
                    section_path=_json_list(r.get("section_path")),
                    url=r.get("url"),
                    path=r.get("path"),
                    page=r.get("page"),
                    snippet=_make_snippet(r["text"], query),
                    metadata={
                        "categories": _json_list(r.get("categories")),
                        "date": _doc_date(r.get("doc_metadata")),
                    },
                )
            )
        return hits


def _doc_date(val: Any) -> str | None:
    """Pull a document date (YYYY-MM-DD) from the document metadata blob.

    0xdf writeups carry ``metadata.date``; other sources usually have none,
    in which case the hit simply has no date and recency never applies.
    """
    import json

    if not val:
        return None
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except Exception:
            return None
    if isinstance(val, dict):
        d = val.get("date")
        if isinstance(d, str) and len(d) >= 10:
            return d[:10]
    return None


def _json_list(val: Any) -> list[str]:
    import json

    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def _make_snippet(text: str, query: str, width: int = 400) -> str:
    """Return a short window around the first query-term match."""
    tokens = set(_TOKEN_RE.findall(query.lower()))
    lower = text.lower()
    pos = -1
    for t in tokens:
        i = lower.find(t)
        if i != -1 and (pos == -1 or i < pos):
            pos = i
    if pos == -1:
        return text[:width].strip() + ("…" if len(text) > width else "")
    start = max(0, pos - width // 3)
    end = min(len(text), start + width)
    snippet = text[start:end].strip()
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet += "…"
    return snippet
