"""Hybrid retrieval facade.

Callers use :class:`HybridRetriever.search` regardless of which backends are
enabled. In this phase only the lexical (FTS5) backend is active; the semantic
backend is optional and, when enabled, its results are merged and re-ranked
alongside lexical hits. Keeping this facade means adding embeddings later does
not change any caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from blackbook.config import Settings
from blackbook.retrieval.lexical import LexicalHit, LexicalRetriever
from blackbook.retrieval.reranker import rerank
from blackbook.storage.database import Database

SearchMode = Literal["keyword", "semantic", "hybrid", "case_similarity", "technique"]


@dataclass
class SearchResult:
    """A single, deduplicated, reranked result ready for presentation."""

    chunk_id: int
    doc_id: int
    title: str
    source_id: str
    source_name: str
    authority: str
    score: float
    snippet: str
    text: str
    section_path: list[str]
    url: str | None = None
    path: str | None = None
    page: int | None = None

    @classmethod
    def from_hit(cls, h: LexicalHit) -> "SearchResult":
        return cls(
            chunk_id=h.chunk_id,
            doc_id=h.doc_id,
            title=h.title,
            source_id=h.source_id,
            source_name=h.source_name,
            authority=h.authority,
            score=h.score,
            snippet=h.snippet,
            text=h.text,
            section_path=h.section_path,
            url=h.url,
            path=h.path,
            page=h.page,
        )


class HybridRetriever:
    def __init__(self, db: Database, settings: Settings):
        self.db = db
        self.settings = settings
        self.lexical = LexicalRetriever(db)
        self._semantic = None  # lazily constructed if enabled

    def search(
        self,
        query: str,
        *,
        mode: SearchMode = "hybrid",
        source_ids: list[str] | None = None,
        platform: str | None = None,
        categories: list[str] | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        cfg = self.settings.retrieval
        limit = min(limit or cfg.default_limit, cfg.max_limit)

        # Candidate pool is larger than the final limit so reranking has room.
        pool_size = max(limit * 3, 30)

        lexical_hits: list[LexicalHit] = []
        if mode in ("keyword", "hybrid", "technique", "case_similarity"):
            lexical_hits = self.lexical.search(query, source_ids=source_ids, limit=pool_size)

        semantic_hits: list[LexicalHit] = []
        if mode in ("semantic", "hybrid") and self.settings.embeddings.enabled:
            semantic_hits = self._semantic_search(query, source_ids=source_ids, limit=pool_size)

        merged = self._merge(lexical_hits, semantic_hits)
        ranked = rerank(
            merged,
            query=query,
            limit=limit,
            per_document_cap=cfg.per_document_cap,
            platform=platform,
            categories=categories,
            mode=mode,
        )
        return [SearchResult.from_hit(h) for h in ranked]

    # -- semantic (optional, Phase 3) ---------------------------------------

    def _semantic_search(
        self, query: str, *, source_ids: list[str] | None, limit: int
    ) -> list[LexicalHit]:
        if self._semantic is None:
            try:
                from blackbook.retrieval.semantic import SemanticRetriever

                self._semantic = SemanticRetriever(self.db, self.settings)
            except Exception:
                # Semantic backend unavailable; degrade gracefully to lexical.
                return []
        return self._semantic.search(query, source_ids=source_ids, limit=limit)

    @staticmethod
    def _merge(a: list[LexicalHit], b: list[LexicalHit]) -> list[LexicalHit]:
        """Merge two hit lists, keeping the higher score per chunk_id."""
        best: dict[int, LexicalHit] = {}
        for h in list(a) + list(b):
            cur = best.get(h.chunk_id)
            if cur is None or h.score > cur.score:
                best[h.chunk_id] = h
        return list(best.values())
