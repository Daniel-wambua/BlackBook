"""Semantic retrieval over local dense embeddings (Phase 3, optional).

This is the module :mod:`blackbook.retrieval.hybrid` imports lazily. It is only
constructed when ``embeddings.enabled`` is true; if the ``[semantic]`` extra is
missing, construction raises :class:`~blackbook.embeddings.EmbeddingsUnavailable`
and the hybrid facade degrades to lexical-only.

Retrieval is a **brute-force flat cosine index**: every stored chunk vector is
compared against the query vector with a single matrix multiply. Vectors are
L2-normalized at ingest time, so cosine similarity is just the dot product. No
index structure is built and no extra dependency is required.

The price of that simplicity is linear in the corpus, and it is not small at
full scale. The decoded matrix is ``chunks * dim * 4`` bytes, so the full
corpus (~5.3e5 chunks at dim 384) is roughly 800 MB resident, re-read from
SQLite on every cache miss. Enabling embeddings here is therefore a deliberate,
expensive operation rather than a free upgrade: ``blackbook embed`` must first
encode the whole corpus (hours of CPU on a laptop), and the server then holds
that matrix in memory for the life of the process. Two consequences are handled
explicitly below:

* the matrix cache is **bounded** (LRU by entry count and a total byte budget),
  so a caller that alternates source filters cannot multiply the resident cost
  of the index by the number of distinct filters it happens to use;
* a matrix above :attr:`SemanticRetriever.MATRIX_WARN_BYTES` logs one warning
  naming the measured size, so the cost is visible instead of being discovered
  as swap pressure.

The returned type is :class:`~blackbook.retrieval.lexical.LexicalHit` — the same
type lexical search returns — so the hybrid merge/rerank pipeline treats both
backends uniformly. :meth:`SemanticRetriever._matrix` is the only place that
knows how candidates are produced, which is the seam an ANN index would replace
without changing this module's public surface.
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

    # The decoded-matrix cache is bounded two ways. Entry count caps the damage
    # from a caller that cycles through many distinct source filters; the byte
    # budget caps it when a single entry is itself enormous (the full-corpus
    # matrix is ~800 MB at the documented corpus size). Oldest entries are
    # evicted first. An entry larger than the whole budget is not cached at all
    # — it is still served, just recomputed on the next call.
    CACHE_MAX_ENTRIES = 4
    CACHE_MAX_BYTES = 512 * 1024 * 1024
    # Above this, one warning is logged per (version, size) pair naming the
    # measured resident cost, so the flat scan's price is never a surprise.
    MATRIX_WARN_BYTES = 256 * 1024 * 1024

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
        # Each entry is (embeddings_version_at_load, chunk_ids, matrix, nbytes).
        # The version counter (bumped on *any* chunk/embedding change, see
        # Database._bump_embeddings_version) invalidates the cache after a
        # re-embed — and after a delete+add that leaves the row count unchanged,
        # which the old count-based guard silently missed. Insertion order is
        # the LRU order (dicts preserve it), so eviction pops from the front.
        self._cache: dict[tuple[str, ...] | None, tuple[int, list[int], object, int]] = {}
        self._cache_bytes = 0
        # (version, nbytes) pairs already warned about, to log the scale warning
        # once per corpus size rather than on every cache miss.
        self._warned: set[tuple[int, int]] = set()

    # -- public API --------------------------------------------------------

    def describe(self) -> dict[str, object]:
        """Cheap readiness summary of the vector index for this model.

        Deliberately does not touch the matrix cache or the embedder's encode
        path, so it is safe to call from a status/inspection tool: it costs one
        ``COUNT(*)``. ``bytes`` is the resident cost the flat scan would pay,
        and ``over_warn_bytes`` flags that it exceeds ``MATRIX_WARN_BYTES``.
        """
        vectors = self.db.embedding_count(self.model_name)
        dim = int(getattr(self.embedder, "dim", 0) or 0)
        nbytes = vectors * dim * 4  # float32
        return {
            "model": self.model_name,
            "vectors": vectors,
            "dim": dim,
            "bytes": nbytes,
            "over_warn_bytes": nbytes > self.MATRIX_WARN_BYTES,
        }

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
        results never go stale within a long-lived server process. The cache is
        bounded by :attr:`CACHE_MAX_ENTRIES` and :attr:`CACHE_MAX_BYTES`; the
        most recently used entry is moved to the back so a hot filter survives
        an eviction sweep.
        """
        sig: tuple[str, ...] | None = (
            tuple(sorted(source_ids)) if source_ids is not None else None
        )
        version = self.db.embeddings_version()
        nbytes = 0
        cached = self._cache.get(sig)
        if cached is not None and cached[0] == version:
            # Refresh LRU position on a hit (cheap: delete + reinsert).
            del self._cache[sig]
            self._cache[sig] = cached
            return cached[1], cached[2]

        ids, blobs = self.db.load_embeddings(self.model_name, source_ids=source_ids)
        matrix, kept = self.embedder.matrix_from_blobs(blobs)
        # Keep chunk_ids aligned with the rows that survived decoding (a stale
        # vector of the wrong dimensionality is dropped by matrix_from_blobs).
        kept_ids = [ids[i] for i in kept]
        nbytes = int(getattr(matrix, "nbytes", 0) or 0)

        self._warn_if_large(version, nbytes, len(kept_ids))
        self._store(sig, version, kept_ids, matrix, nbytes)
        return kept_ids, matrix

    def _warn_if_large(self, version: int, nbytes: int, rows: int) -> None:
        """Log the flat scan's resident cost once per (version, size)."""
        if nbytes <= self.MATRIX_WARN_BYTES:
            return
        key = (version, nbytes)
        if key in self._warned:
            return
        self._warned.add(key)
        log.warning(
            "Semantic flat index holds %d vectors (%.0f MB resident) for model %s; "
            "every cache miss re-reads that from SQLite and the scan is O(n) per "
            "query. This is expected at full corpus scale, but it is the point at "
            "which an ANN index becomes worth its dependency.",
            rows,
            nbytes / (1024 * 1024),
            self.model_name,
        )

    def _store(
        self,
        sig: tuple[str, ...] | None,
        version: int,
        kept_ids: list[int],
        matrix: object,
        nbytes: int,
    ) -> None:
        """Insert into the bounded cache, evicting oldest entries as needed."""
        if nbytes > self.CACHE_MAX_BYTES:
            # Bigger than the entire budget: serve it, but do not pin it.
            return
        previous = self._cache.pop(sig, None)
        if previous is not None:
            self._cache_bytes -= previous[3]
        while self._cache and (
            len(self._cache) >= self.CACHE_MAX_ENTRIES
            or self._cache_bytes + nbytes > self.CACHE_MAX_BYTES
        ):
            # Plain dicts keep insertion order, so the first key is the oldest.
            # (dict.popitem takes no arguments; that is the OrderedDict API.)
            evicted = self._cache.pop(next(iter(self._cache)))
            self._cache_bytes -= evicted[3]
        self._cache[sig] = (version, kept_ids, matrix, nbytes)
        self._cache_bytes += nbytes
