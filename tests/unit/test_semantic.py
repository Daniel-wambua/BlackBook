"""Phase 3 semantic-layer tests.

These exercise the embedding storage, the ``SemanticRetriever`` cosine ranking,
the hybrid merge, and graceful degradation — all **without downloading a model**.
A deterministic :class:`FakeEmbedder` produces normalized bag-of-words vectors
that reproduce real cosine geometry (texts sharing tokens score higher), so the
retrieval math is tested for real while CI stays offline and fast.

A separate class of tests uses the *actual* local model but skips cleanly when
the ``[semantic]`` extra (sentence-transformers) is not installed.
"""

import re

import numpy as np
import pytest

from blackbook.config import DatabaseConfig, Settings
from blackbook.embeddings import VECTOR_DTYPE, embed_missing_chunks
from blackbook.retrieval.hybrid import HybridRetriever
from blackbook.retrieval.semantic import SemanticRetriever

_WORD = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Model-free embedder with the same surface as :class:`Embedder`.

    Encodes text as an L2-normalized hashing bag-of-words. This is deterministic
    and has genuine cosine geometry: two texts that share tokens land on
    overlapping dimensions and score a higher dot product, so ranking assertions
    are meaningful without a real model.
    """

    def __init__(self, model_name="fake-bow-64", dim=64, batch_size=8):
        self.model_name = model_name
        self.dim = dim
        self.batch_size = batch_size
        self.device = "cpu"
        self._np = np

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in _WORD.findall(text.lower()):
            v[hash(tok) % self.dim] += 1.0
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        return v

    def encode(self, texts):
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.vstack([self._vec(t) for t in texts]).astype(np.float32)

    def encode_one(self, text):
        return self._vec(text)

    def to_bytes(self, vector):
        return np.asarray(vector, dtype=np.float32).astype(VECTOR_DTYPE, copy=False).tobytes()

    def matrix_from_blobs(self, blobs):
        rows, kept = [], []
        expected = self.dim * 4
        for i, b in enumerate(blobs):
            if len(b) != expected:
                continue
            rows.append(np.frombuffer(b, dtype=VECTOR_DTYPE))
            kept.append(i)
        if not rows:
            return np.empty((0, self.dim), dtype=np.float32), kept
        return np.vstack(rows).astype(np.float32, copy=False), kept


def make_settings(tmp_path, enabled=True):
    s = Settings(home=tmp_path, database=DatabaseConfig(path=str(tmp_path / "d.db")))
    s.embeddings.enabled = enabled
    s.embeddings.model = "fake-bow-64"
    return s


def _embed_all(db, embedder):
    return embed_missing_chunks(db, embedder)


# --------------------------------------------------------------------------
# Embedding storage
# --------------------------------------------------------------------------

def test_embedding_roundtrip_and_counts(seeded_db):
    emb = FakeEmbedder()
    v = emb.encode_one("kerberoasting service tickets")
    blob = emb.to_bytes(v)
    assert len(blob) == emb.dim * 4

    with seeded_db.session():
        seeded_db.upsert_embedding(1, emb.model_name, emb.dim, blob)
    assert seeded_db.embedding_count(emb.model_name) == 1
    assert seeded_db.embedding_count("other-model") == 0

    ids, blobs = seeded_db.load_embeddings(emb.model_name)
    assert ids == [1]
    got = np.frombuffer(blobs[0], dtype=VECTOR_DTYPE)
    assert np.allclose(got, v, atol=1e-6)


def test_upsert_embedding_replaces(seeded_db):
    emb = FakeEmbedder()
    with seeded_db.session():
        seeded_db.upsert_embedding(1, emb.model_name, emb.dim, emb.to_bytes(emb.encode_one("a")))
        seeded_db.upsert_embedding(1, emb.model_name, emb.dim, emb.to_bytes(emb.encode_one("b")))
    assert seeded_db.embedding_count(emb.model_name) == 1  # replaced, not duplicated


def test_iter_missing_embeddings(seeded_db):
    emb = FakeEmbedder()
    missing = list(seeded_db.iter_chunks_missing_embeddings(emb.model_name))
    assert len(missing) == 3  # all seeded chunks
    # After embedding chunk 1, it drops out of the missing set.
    with seeded_db.session():
        seeded_db.upsert_embedding(1, emb.model_name, emb.dim, emb.to_bytes(emb.encode_one("x")))
    missing2 = [cid for cid, _ in seeded_db.iter_chunks_missing_embeddings(emb.model_name)]
    assert 1 not in missing2 and len(missing2) == 2


def test_embedding_fk_cascade(seeded_db):
    """Re-chunking a document drops its stale vectors via the FK cascade."""
    emb = FakeEmbedder()
    _embed_all(seeded_db, emb)
    assert seeded_db.embedding_count(emb.model_name) == 3
    # Replace chunks of doc 1 (the Kerberoasting doc) -> its 2 vectors cascade away.
    from blackbook.storage import Chunk
    from blackbook.storage.database import sha256_text

    with seeded_db.session():
        seeded_db.replace_chunks(
            1,
            [Chunk(doc_id=1, ordinal=0, text="new", section_path=[], token_estimate=1,
                   content_hash=sha256_text("new"))],
        )
    assert seeded_db.embedding_count(emb.model_name) == 1  # only doc 2's vector remains


def test_scoped_delete_embeddings(seeded_db):
    emb = FakeEmbedder()
    _embed_all(seeded_db, emb)
    removed = seeded_db.delete_embeddings(emb.model_name, source_ids=["0xdf"])
    assert removed == 1
    assert seeded_db.embedding_count(emb.model_name) == 2
    ids0, _ = seeded_db.load_embeddings(emb.model_name, source_ids=["0xdf"])
    assert ids0 == []


def test_hydrate_chunks(seeded_db):
    meta = seeded_db.hydrate_chunks([1, 2, 999])
    assert set(meta) == {1, 2}  # missing id simply absent
    assert meta[1]["title"] == "Kerberoasting"
    assert meta[1]["source_id"] == "hacktricks"


# --------------------------------------------------------------------------
# embed_missing_chunks driver
# --------------------------------------------------------------------------

def test_embed_missing_chunks_idempotent(seeded_db):
    emb = FakeEmbedder(batch_size=2)
    n = embed_missing_chunks(seeded_db, emb)
    assert n == 3
    assert embed_missing_chunks(seeded_db, emb) == 0  # nothing left


def test_embed_missing_chunks_source_scoped(seeded_db):
    emb = FakeEmbedder()
    n = embed_missing_chunks(seeded_db, emb, source_ids=["0xdf"])
    assert n == 1
    assert seeded_db.embedding_count(emb.model_name) == 1


# --------------------------------------------------------------------------
# SemanticRetriever
# --------------------------------------------------------------------------

def test_semantic_ranking(tmp_path, seeded_db):
    emb = FakeEmbedder()
    _embed_all(seeded_db, emb)
    settings = make_settings(tmp_path)
    sr = SemanticRetriever(seeded_db, settings, embedder=emb)

    hits = sr.search("kerberoasting service tickets SPN", limit=5)
    assert hits
    # Every hit is real and fully hydrated.
    for h in hits:
        assert seeded_db.get_chunk(h.chunk_id) is not None
        assert h.metadata["retrieval"] == "semantic"
        assert h.bm25 == 0.0
        assert h.snippet
        assert 0.0 <= h.score <= 1.0
        assert "cosine" in h.metadata
    # The chunk that literally shares the most tokens ranks first.
    assert hits[0].chunk_id == 1


def test_semantic_source_filter(tmp_path, seeded_db):
    emb = FakeEmbedder()
    _embed_all(seeded_db, emb)
    settings = make_settings(tmp_path)
    sr = SemanticRetriever(seeded_db, settings, embedder=emb)
    hits = sr.search("kerberoast", source_ids=["0xdf"], limit=5)
    assert hits and all(h.source_id == "0xdf" for h in hits)


def test_semantic_empty_query(tmp_path, seeded_db):
    emb = FakeEmbedder()
    _embed_all(seeded_db, emb)
    settings = make_settings(tmp_path)
    sr = SemanticRetriever(seeded_db, settings, embedder=emb)
    assert sr.search("") == []
    assert sr.search("   ") == []


def test_semantic_no_embeddings_returns_empty(tmp_path, seeded_db):
    """With the model configured but nothing embedded, search yields nothing."""
    emb = FakeEmbedder()
    settings = make_settings(tmp_path)
    sr = SemanticRetriever(seeded_db, settings, embedder=emb)
    assert sr.search("kerberoast") == []


def test_semantic_cache_invalidates_on_delete(tmp_path, seeded_db):
    emb = FakeEmbedder()
    _embed_all(seeded_db, emb)
    settings = make_settings(tmp_path)
    sr = SemanticRetriever(seeded_db, settings, embedder=emb)

    ids_a, mat_a = sr._matrix(None)
    ids_b, mat_b = sr._matrix(None)
    assert mat_a is mat_b  # cached: same object

    with seeded_db.session():
        seeded_db.delete_embeddings(emb.model_name)
    ids_c, mat_c = sr._matrix(None)
    assert ids_c == [] and mat_c is not mat_a  # invalidated


def test_semantic_cache_invalidates_on_same_count_change(tmp_path, seeded_db):
    """Replace one vector: embedding *count* is unchanged, so the old
    count-based cache guard kept the stale matrix. The version counter must
    still invalidate it."""
    emb = FakeEmbedder()
    _embed_all(seeded_db, emb)
    settings = make_settings(tmp_path)
    sr = SemanticRetriever(seeded_db, settings, embedder=emb)

    ids_a, mat_a = sr._matrix(None)
    assert len(ids_a) == 3

    with seeded_db.session():
        seeded_db.upsert_embedding(
            ids_a[0],
            emb.model_name,
            emb.dim,
            emb.to_bytes(emb.encode_one("totally different tokens xyzzy plugh")),
        )
    # The count is unchanged — only the version moved.
    assert seeded_db.embedding_count(emb.model_name) == 3
    assert seeded_db.embeddings_version() > 0

    ids_b, mat_b = sr._matrix(None)
    assert mat_b is not mat_a  # invalidated despite the same count
    assert ids_b == ids_a


def test_semantic_cache_invalidates_on_rechunk(tmp_path, seeded_db):
    """Re-chunking a document cascade-deletes its vectors; the cache must not
    keep serving the old matrix even though no embeddings were touched."""
    from blackbook.storage import Chunk

    emb = FakeEmbedder()
    _embed_all(seeded_db, emb)
    sr = SemanticRetriever(seeded_db, make_settings(tmp_path), embedder=emb)

    ids_a, mat_a = sr._matrix(None)
    assert len(ids_a) == 3

    with seeded_db.session():
        seeded_db.replace_chunks(
            1,
            [Chunk(doc_id=1, ordinal=0, text="rechunked body", section_path=[],
                   token_estimate=2, content_hash="rechunked")],
        )
    ids_b, mat_b = sr._matrix(None)
    assert mat_b is not mat_a
    # Doc 1 owned chunks 1-2; only doc 2's vector (chunk 3) survives.
    assert ids_b == [3]


# --------------------------------------------------------------------------
# Hybrid merge + graceful degradation
# --------------------------------------------------------------------------

def test_hybrid_merge_prefers_higher_score():
    from blackbook.retrieval.lexical import LexicalHit

    def hit(cid, score, source="lex"):
        return LexicalHit(chunk_id=cid, doc_id=1, text="t", title="T",
                          source_id="s", source_name="S", authority="trusted",
                          bm25=0.0, score=score, metadata={"retrieval": source})

    a = [hit(1, 0.2, "lex"), hit(2, 0.9, "lex")]
    b = [hit(1, 0.8, "sem"), hit(3, 0.5, "sem")]
    merged = {h.chunk_id: h for h in HybridRetriever._merge(a, b)}
    assert merged[1].score == 0.8 and merged[1].metadata["retrieval"] == "sem"
    assert merged[2].score == 0.9
    assert merged[3].score == 0.5


def test_hybrid_uses_semantic_branch(tmp_path, seeded_db, monkeypatch):
    """A query with no lexical keyword overlap still surfaces the right chunk
    through the semantic branch of the hybrid merge."""
    emb = FakeEmbedder()
    _embed_all(seeded_db, emb)
    settings = make_settings(tmp_path)
    r = HybridRetriever(seeded_db, settings)
    # Inject the fake-embedder-backed semantic retriever (no model download).
    r._semantic = SemanticRetriever(seeded_db, settings, embedder=emb)

    # "SPN" only appears in chunk 1's text lexically, but semantic bag-of-words
    # will also match; assert the top hit is a real kerberoasting chunk.
    res = r.search("kerberoasting SPN service tickets", limit=5)
    assert res
    assert res[0].title == "Kerberoasting"


def test_hybrid_graceful_degradation_when_semantic_unavailable(tmp_path, seeded_db, monkeypatch):
    """If constructing the semantic backend raises, hybrid falls back to lexical."""
    settings = make_settings(tmp_path)
    r = HybridRetriever(seeded_db, settings)

    import blackbook.retrieval.semantic as sem

    def boom(*a, **k):
        raise RuntimeError("no sentence-transformers here")

    monkeypatch.setattr(sem, "SemanticRetriever", boom)
    # Lexical still works; no exception bubbles up.
    res = r.search("kerberoasting", limit=5)
    assert res and res[0].title in ("Kerberoasting", "HTB: Forest")


# --------------------------------------------------------------------------
# Real-model smoke (skips when the extra is absent)
# --------------------------------------------------------------------------

def _has_semantic_extra():
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_semantic_extra(), reason="semantic extra not installed")
def test_real_model_semantic_ranking(tmp_path, seeded_db):
    """With the actual local model, a paraphrased query (no shared keyword)
    still ranks the kerberoasting chunk above the unrelated one."""
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from blackbook.embeddings import Embedder, EmbeddingsUnavailable

    settings = Settings(home=tmp_path, database=DatabaseConfig(path=str(tmp_path / "d.db")))
    settings.embeddings.enabled = True
    model = settings.embeddings.model
    try:
        embedder = Embedder(model, device="cpu", batch_size=8)
    except EmbeddingsUnavailable:
        pytest.skip("model weights unavailable offline")

    embed_missing_chunks(seeded_db, embedder)
    sr = SemanticRetriever(seeded_db, settings, embedder=embedder)
    # No literal token overlap with "kerberoast"/"SPN".
    hits = sr.search("crack service account credentials in active directory", limit=3)
    assert hits
    assert hits[0].source_id in ("hacktricks", "0xdf")
    assert hits[0].metadata["cosine"] > hits[-1].metadata["cosine"]
