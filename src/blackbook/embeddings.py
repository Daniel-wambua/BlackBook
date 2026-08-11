"""Local embedding model wrapper (Phase 3, optional).

This module isolates the only hard dependency on the ``[semantic]`` extra
(``sentence-transformers`` + ``numpy``). Nothing here is imported at process
start; the hybrid retriever and the CLI embed command construct an
:class:`Embedder` lazily, and construction raises :class:`EmbeddingsUnavailable`
with an actionable message when the extra is not installed. Callers catch that
and degrade to lexical-only retrieval — the system is never broken by the
absence of embeddings.

Design choices:

* **Local only.** The model is a local sentence-transformers checkpoint. No
  network embedding APIs are ever called; the first run downloads the model
  weights to the HuggingFace cache and subsequent runs are offline-capable.
* **Unit-normalized vectors.** Every vector is L2-normalized, so cosine
  similarity reduces to a dot product. This keeps the retrieval math simple and
  lets us store vectors and compare with a single matrix multiply.
* **float32 blobs.** Vectors are serialized as little-endian float32 bytes for
  compact, dependency-free storage in SQLite.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

log = logging.getLogger(__name__)


class EmbeddingsUnavailable(RuntimeError):
    """Raised when the semantic extra (sentence-transformers) is not installed.

    The message tells the operator exactly how to enable it. Callers that want
    graceful degradation should catch this and fall back to lexical search.
    """


# The serialized vector dtype. Kept as a module constant so storage and
# retrieval agree without importing numpy just for the name.
VECTOR_DTYPE = "<f4"  # little-endian float32


def _require_numpy():
    try:
        import numpy as np  # noqa: F401

        return np
    except Exception as e:  # pragma: no cover - environment dependent
        raise EmbeddingsUnavailable(
            "numpy is required for semantic search. Install the semantic extra: "
            'pip install "blackbook-mcp[semantic]"'
        ) from e


class Embedder:
    """Wraps a local sentence-transformers model.

    Construction loads the model, which can be slow on first use (weights are
    downloaded and cached). Construction is therefore lazy at the call sites and
    the instance is reused for the life of the process.
    """

    def __init__(self, model_name: str, *, device: str = "cpu", batch_size: int = 32):
        self.model_name = model_name
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self._np = _require_numpy()
        self._model = self._load_model(model_name, device)
        # Probe dimensionality up front so storage can record it and retrieval
        # can validate stored vectors against the live model.
        dim = self._model.get_sentence_embedding_dimension()
        self.dim = int(dim)

    # -- construction helpers ---------------------------------------------

    @staticmethod
    def _load_model(model_name: str, device: str):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise EmbeddingsUnavailable(
                "sentence-transformers is not installed. Enable semantic search "
                'with: pip install "blackbook-mcp[semantic]"'
            ) from e
        log.info("loading embedding model %s on %s", model_name, device)
        return SentenceTransformer(model_name, device=device)

    # -- encoding ----------------------------------------------------------

    def encode(self, texts: Sequence[str]) -> "np.ndarray":
        """Encode a batch of texts into an (N, dim) float32, L2-normalized array."""
        np = self._np
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        vecs = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_one(self, text: str) -> "np.ndarray":
        """Encode a single text into a (dim,) float32, L2-normalized vector."""
        return self.encode([text])[0]

    # -- serialization -----------------------------------------------------

    def to_bytes(self, vector: "np.ndarray") -> bytes:
        """Serialize a 1-D vector to little-endian float32 bytes."""
        np = self._np
        return np.asarray(vector, dtype=np.float32).astype(VECTOR_DTYPE, copy=False).tobytes()

    def matrix_from_blobs(self, blobs: Sequence[bytes]) -> "np.ndarray":
        """Decode a list of float32 blobs into an (N, dim) matrix.

        Rows whose length does not match ``dim`` (e.g. a stale vector from a
        model with a different dimensionality) are skipped defensively; the
        parallel index of kept rows is returned so callers can align chunk_ids.
        """
        np = self._np
        rows = []
        kept: list[int] = []
        expected = self.dim * 4  # float32 == 4 bytes
        for i, b in enumerate(blobs):
            if len(b) != expected:
                continue
            rows.append(np.frombuffer(b, dtype=VECTOR_DTYPE))
            kept.append(i)
        if not rows:
            return np.empty((0, self.dim), dtype=np.float32), kept
        return np.vstack(rows).astype(np.float32, copy=False), kept


def try_build_embedder(settings) -> "Embedder | None":
    """Construct an :class:`Embedder` from settings, or return ``None``.

    Returns ``None`` (never raises) when embeddings are disabled in config or
    when the semantic extra is unavailable — the graceful-degradation path used
    by retrieval. Callers that *want* the error (e.g. the CLI embed command)
    should construct :class:`Embedder` directly.
    """
    if not settings.embeddings.enabled:
        return None
    try:
        return Embedder(
            settings.embeddings.model,
            device=settings.embeddings.device,
            batch_size=settings.embeddings.batch_size,
        )
    except EmbeddingsUnavailable as e:
        log.warning("semantic search unavailable, using lexical only: %s", e)
        return None


def embed_missing_chunks(
    db,
    embedder: "Embedder",
    *,
    source_ids: list[str] | None = None,
    on_progress=None,
) -> int:
    """Embed every chunk that lacks a current-model vector. Returns the count.

    Shared by the ingestion hook and the ``blackbook embed`` CLI command. The
    set of missing chunks is materialized up front (draining the storage cursor)
    so that upserting embeddings mid-scan cannot disturb the ``NOT EXISTS``
    query that selected them. Encoding runs in batches of ``embedder.batch_size``
    to amortize model overhead; each batch is committed in its own session so an
    interruption leaves a consistent, resumable partial state.

    ``on_progress(done, total)`` is called after each committed batch if given.
    """
    model = embedder.model_name
    bs = embedder.batch_size
    pending = list(db.iter_chunks_missing_embeddings(model, source_ids=source_ids))
    total = len(pending)
    embedded = 0
    for start in range(0, total, bs):
        batch = pending[start : start + bs]
        vecs = embedder.encode([text for _, text in batch])
        with db.session():
            for (cid, _), vec in zip(batch, vecs):
                db.upsert_embedding(cid, model, embedder.dim, embedder.to_bytes(vec))
        embedded += len(batch)
        if on_progress is not None:
            on_progress(embedded, total)
    return embedded
