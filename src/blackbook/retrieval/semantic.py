"""Semantic retrieval over local dense embeddings (Phase 3, optional).

This is the module :mod:`blackbook.retrieval.hybrid` imports lazily. It is only
constructed when ``embeddings.enabled`` is true; if the ``[semantic]`` extra is
missing, construction raises :class:`~blackbook.embeddings.EmbeddingsUnavailable`
and the hybrid facade degrades to lexical-only.

Retrieval is a **brute-force flat cosine index**: every stored chunk vector is
compared against the query vector with a single matrix multiply. Because all
vectors are L2-normalized at ingest time, cosine similarity is just the dot
product. For a single-file knowledge base of this size (tens of thousands of
chunks) a flat scan is fast (a few tens of milliseconds) and needs no extra
index structure or dependency. Swapping in an ANN index later would not change
this module's public surface.

The returned type is :class:`~blackbook.retrieval.lexical.LexicalHit` — the same
type lexical search returns — so the hybrid merge/rerank pipeline treats both
backends uniformly.
"""

from __future__ import annotations

import logging

from blackbook.config import Settings
from blackbook.embeddings import Embedder
from blackbook.retrieval.lexical import (
    LexicalHit,
    _doc_date,
    _json_list,
    _make_snippet,
)
from blackbook.storage.database import Database

log = logging.getLogger(__name__)


class SemanticRetriever:
    """Dense-vector retrieval using a local sentence-transformers model.

    Constructed lazily by the hybrid facade. Loading the model happens here in
    ``__init__``; if the semantic extra is unavailable the underlying
    :class:`Embedder` raises and the hybrid facade catches it.
    """

    def __init__(self, db: Database, settings: Settings, embedder: Embedder | None = None):
        self.db = db
        self.settings = settings
        # Constructing the Embedder loads the model and raises
        # EmbeddingsUnavailable if the extra is missing — hybrid.py catches it.
        # An embedder may be injected (to reuse an already-loaded model, or a
        # deterministic fake in tests) instead of loading one here.
        self.embedder = embedder or Embedder(
            settings.embeddings.model,
            device=settings.embeddings.device,
            batch_size=settings.embeddings.batch_size,
        )
        self.model_name = self.embedder.model_name
        # Cache of decoded vector matrices keyed by the source-filter signature.
        # Each entry is (embeddings_version_at_load, chunk_ids, matrix). The
        # version counter (bumped on *any* chunk/embedding change, see
        # Database._bump_embeddings_version) invalidates the cache after a
        # re-embed — and after a delete+add that leaves the row count unchanged,
        # which the old count-based guard silently missed.
        self._cache: dict[tuple[str, ...] | None, tuple[int, list[int], object]] = {}

    # -- public API --------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        source_ids: list[str] | None = None,
        limit: int = 50,
    ) -> list[LexicalHit]:
        query = (query or "").strip()
        if not query:
            return []
        # An empty source list means "no sources in scope" — never widen to all.
        if source_ids is not None and not source_ids:
            return []

        ids, matrix = self._matrix(source_ids)
        if not ids:
            return []

        np = self.embedder._np
        q = self.embedder.encode_one(query)  # (dim,), unit-normalized
        # Cosine similarity == dot product for unit vectors.
        sims = matrix @ q  # (N,)

        # Top-`limit` by similarity without a full sort of the whole corpus.
        k = min(limit, len(ids))
        if k <= 0:
            return []
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]  # order the k winners by score desc

        top_ids = [ids[int(i)] for i in top]
        meta = self.db.hydrate_chunks(top_ids)

        hits: list[LexicalHit] = []
        for i in top:
            cid = ids[int(i)]
            row = meta.get(cid)
            if row is None:
                # Vector without a live chunk (should not happen thanks to the
                # FK cascade, but never fabricate a hit for a missing chunk).
                continue
            cos = float(sims[int(i)])
            # Map cosine (~[-1, 1]) into a non-negative relevance. Irrelevant
            # matches sit near/below 0 and are clamped; relevant ones keep their
            # ordering. This keeps the scale comparable to lexical scores so the
            # reranker can combine both fairly.
            score = max(0.0, cos)
            hits.append(
                LexicalHit(
                    chunk_id=cid,
                    doc_id=int(row["doc_id"]),
                    text=row["text"],
                    title=row["title"],
                    source_id=row["source_id"],
                    source_name=row["source_name"],
                    authority=row["source_authority"],
                    bm25=0.0,  # not a lexical hit
                    score=score,
                    section_path=_json_list(row.get("section_path")),
                    url=row.get("url"),
                    path=row.get("path"),
                    page=row.get("page"),
                    snippet=_make_snippet(row["text"], query),
                    metadata={
                        "categories": _json_list(row.get("categories")),
                        "date": _doc_date(row.get("doc_metadata")),
                        "retrieval": "semantic",
                        "cosine": cos,
                    },
                )
            )
        return hits

    # -- internals ---------------------------------------------------------

    def _matrix(self, source_ids: list[str] | None):
        """Return ``(chunk_ids, matrix)`` for the given source filter.

        Results are cached per source-filter signature and invalidated when the
        embeddings version changes (any insert/delete/re-chunk bumps it), so
        results never go stale within a long-lived server process.
        """
        sig: tuple[str, ...] | None = (
            tuple(sorted(source_ids)) if source_ids is not None else None
        )
        version = self.db.embeddings_version()
        cached = self._cache.get(sig)
        if cached is not None and cached[0] == version:
            return cached[1], cached[2]

        ids, blobs = self.db.load_embeddings(self.model_name, source_ids=source_ids)
        matrix, kept = self.embedder.matrix_from_blobs(blobs)
        # Keep chunk_ids aligned with the rows that survived decoding (a stale
        # vector of the wrong dimensionality is dropped by matrix_from_blobs).
        kept_ids = [ids[i] for i in kept]
        self._cache[sig] = (version, kept_ids, matrix)
        return kept_ids, matrix
