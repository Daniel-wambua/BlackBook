"""Hybrid retrieval facade.

Callers use :class:`HybridRetriever.search` regardless of which backends are
enabled. The lexical (FTS5) backend is always available; the semantic backend
is optional and, when enabled *and* populated, its results are merged and
re-ranked alongside lexical hits. Keeping this facade means adding embeddings
later does not change any caller.

Two invariants this module is responsible for:

* **No silent empty.** ``mode="semantic"`` degrades to lexical when the
  semantic backend is unavailable or holds no vectors, and records that
  degradation in :class:`SearchDiagnostics`. It never returns zero results
  merely because embeddings are switched off.
* **No silent substitution.** A caller that asked for semantic and got lexical
  can find out, both from the diagnostics object and from the ``note`` the MCP
  layer surfaces. See :meth:`HybridRetriever.search`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from blackbook.config import Settings
from blackbook.retrieval.lexical import LexicalHit, LexicalRetriever
from blackbook.retrieval.reranker import rerank
from blackbook.storage.database import Database

SearchMode = Literal["keyword", "semantic", "hybrid", "case_similarity", "technique"]

# Modes that always consult the lexical backbone.
_LEXICAL_MODES = ("keyword", "hybrid", "technique", "case_similarity")


@dataclass
class SearchDiagnostics:
    """Which retrieval backends a single search actually used.

    Filled in place by :meth:`HybridRetriever.search` when the caller passes
    one. It is a caller-owned out-parameter rather than part of the return
    value so the result stays a plain list (existing callers are unaffected)
    and so concurrent requests sharing one retriever cannot race on it.
    """

    mode: str = ""
    # Backends that were actually executed, and those that returned >= 1 hit.
    backends_queried: list[str] = field(default_factory=list)
    backends_contributed: list[str] = field(default_factory=list)
    semantic_requested: bool = False
    semantic_enabled: bool = False
    semantic_contributed: bool = False
    # True when semantic retrieval was asked for but did not contribute —
    # either because it is disabled, its dependency is missing, or the vector
    # index is empty. The results are lexical-only in that case.
    degraded: bool = False
    # Populated only when a degradation happened, so the common path pays no
    # extra COUNT(*).
    semantic_vectors: int = 0
    semantic_error: str | None = None

    @property
    def backend(self) -> str:
        """A compact human-readable summary of what produced the results."""
        if not self.backends_contributed:
            return "none"
        return "+".join(self.backends_contributed)


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
        self._semantic_error: str | None = None  # why construction failed

    def search(
        self,
        query: str,
        *,
        mode: SearchMode = "hybrid",
        source_ids: list[str] | None = None,
        platform: str | None = None,
        categories: list[str] | None = None,
        techniques: list[str] | None = None,
        limit: int | None = None,
        diagnostics: SearchDiagnostics | None = None,
    ) -> list[SearchResult]:
        cfg = self.settings.retrieval
        limit = min(limit or cfg.default_limit, cfg.max_limit)

        # Candidate pool is larger than the final limit so reranking has room.
        pool_size = max(limit * 3, 30)

        semantic_requested = mode in ("semantic", "hybrid")
        semantic_enabled = bool(self.settings.embeddings.enabled)

        # Semantic first: mode="semantic" needs to know whether it delivered
        # before deciding to fall back to the lexical backbone.
        semantic_hits: list[LexicalHit] = []
        if semantic_requested and semantic_enabled:
            semantic_hits = self._semantic_search(
                query, source_ids=source_ids, limit=pool_size
            )

        lexical_backbone = mode in _LEXICAL_MODES
        if mode == "semantic" and not semantic_hits:
            # Either embeddings are disabled, the extra is missing, or the
            # vector table is empty. Degrade to lexical rather than returning
            # an unexplained empty list.
            lexical_backbone = True

        lexical_hits: list[LexicalHit] = []
        if lexical_backbone:
            lexical_hits = self.lexical.search(
                query,
                source_ids=source_ids,
                limit=pool_size,
                platform=platform,
                categories=categories,
            )

        degraded = semantic_requested and not semantic_hits
        if diagnostics is not None:
            queried: list[str] = []
            if lexical_backbone:
                queried.append("lexical")
            if semantic_requested and semantic_enabled:
                queried.append("semantic")
            contributed: list[str] = []
            if lexical_hits:
                contributed.append("lexical")
            if semantic_hits:
                contributed.append("semantic")
            diagnostics.mode = mode
            diagnostics.backends_queried = queried
            diagnostics.backends_contributed = contributed
            diagnostics.semantic_requested = semantic_requested
            diagnostics.semantic_enabled = semantic_enabled
            diagnostics.semantic_contributed = bool(semantic_hits)
            diagnostics.degraded = degraded
            diagnostics.semantic_error = self._semantic_error
            if degraded:
                try:
                    diagnostics.semantic_vectors = self.db.embedding_count()
                except Exception:  # pragma: no cover - diagnostics must not raise
                    diagnostics.semantic_vectors = 0

        merged = self._merge(lexical_hits, semantic_hits)
        ranked = rerank(
            merged,
            query=query,
            limit=limit,
            per_document_cap=cfg.per_document_cap,
            per_source_cap=cfg.per_source_cap,
            platform=platform,
            categories=categories,
            techniques=techniques,
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
                self._semantic_error = None
            except Exception as exc:
                # Semantic backend unavailable; degrade gracefully to lexical.
                # Record *why* so a caller asking for semantic mode can be told
                # instead of quietly receiving lexical results.
                self._semantic_error = f"{type(exc).__name__}: {exc}"
                return []
        try:
            return self._semantic.search(query, source_ids=source_ids, limit=limit)
        except Exception as exc:
            self._semantic_error = f"{type(exc).__name__}: {exc}"
            return []

    @staticmethod
    def _merge(a: list[LexicalHit], b: list[LexicalHit]) -> list[LexicalHit]:
        """Merge two hit lists, keeping the higher score per chunk_id."""
        best: dict[int, LexicalHit] = {}
        for h in list(a) + list(b):
            cur = best.get(h.chunk_id)
            if cur is None or h.score > cur.score:
                best[h.chunk_id] = h
        return list(best.values())
