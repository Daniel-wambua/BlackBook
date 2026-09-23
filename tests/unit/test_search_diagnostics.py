"""Regression tests for the retrieval facade's reporting contract.

Two things are pinned here:

* **A1, no silent empty.** ``mode="semantic"`` used to return ``[]`` with no
  explanation whenever embeddings were disabled (the default) or the vector
  table was empty - the mode was simply absent from the lexical branch's
  condition, so nothing ran at all. It now degrades to lexical and says so.
* **A2, no silent substitution.** ``SearchOutput`` reports which backend
  actually produced the results, so a lexical-only ``mode="hybrid"`` call is
  distinguishable from a genuine merged one.

Also covers the BM25 column weighting (title/section above body prose), which
must reorder hits without ever changing which chunks match.

Finally it pins the A3 cost contract: the flat-index matrix cache is bounded,
an oversized entry is served but not pinned, and the index reports its own
readiness so "semantic is enabled but doing nothing" is a visible state.
"""

import logging

from blackbook.config import DatabaseConfig, Settings
from blackbook.mcp.schemas import KnowledgeSourceInput, SearchInput
from blackbook.mcp.tools import KnowledgeTools
from blackbook.retrieval import HybridRetriever, SearchDiagnostics
from blackbook.retrieval.semantic import SemanticRetriever
from blackbook.storage import Chunk, Document, Source
from blackbook.storage.database import Database, sha256_text

from tests.unit.test_semantic import FakeEmbedder


def make_settings(tmp_path, enabled=False):
    s = Settings(home=tmp_path, database=DatabaseConfig(path=str(tmp_path / "d.db")))
    s.embeddings.enabled = enabled
    s.embeddings.model = "fake-bow-64"
    return s


# ---------------------------------------------------------------------------
# A1: a semantic request can never come back empty and unexplained
# ---------------------------------------------------------------------------


def test_semantic_mode_degrades_to_lexical_when_disabled(tmp_path, seeded_db):
    """The default install has embeddings off; mode="semantic" must still work."""
    r = HybridRetriever(seeded_db, make_settings(tmp_path, enabled=False))
    diag = SearchDiagnostics()

    results = r.search("kerberoasting", mode="semantic", limit=5, diagnostics=diag)

    # The regression: this used to be [] with nothing run at all.
    assert results, "semantic mode returned nothing with no error"
    assert diag.backends_queried == ["lexical"]
    assert diag.backends_contributed == ["lexical"]
    assert diag.semantic_requested is True
    assert diag.semantic_enabled is False
    assert diag.degraded is True
    assert diag.backend == "lexical"


def test_semantic_mode_degrades_when_index_empty(tmp_path, seeded_db):
    """Embeddings on but zero vectors stored: degrade, and report the count."""
    settings = make_settings(tmp_path, enabled=True)
    r = HybridRetriever(seeded_db, settings)
    # Inject a model-free semantic backend; the DB genuinely holds no vectors.
    r._semantic = SemanticRetriever(seeded_db, settings, embedder=FakeEmbedder())
    diag = SearchDiagnostics()

    results = r.search("kerberoasting", mode="semantic", limit=5, diagnostics=diag)

    assert results
    assert diag.semantic_enabled is True
    assert diag.semantic_contributed is False
    assert diag.degraded is True
    assert diag.semantic_vectors == 0
    # Semantic ran and returned nothing; only lexical contributed.
    assert diag.backends_queried == ["lexical", "semantic"]
    assert diag.backends_contributed == ["lexical"]


def test_hybrid_reports_lexical_only_when_semantic_off(tmp_path, seeded_db):
    """A hybrid call with no semantic backend is not silently a hybrid."""
    r = HybridRetriever(seeded_db, make_settings(tmp_path, enabled=False))
    diag = SearchDiagnostics()

    assert r.search("kerberoasting", mode="hybrid", limit=5, diagnostics=diag)

    assert diag.backend == "lexical"
    assert diag.degraded is True  # semantic was requested but did not run


def test_keyword_mode_is_not_reported_as_degraded(tmp_path, seeded_db):
    """Keyword mode never asked for semantic, so it is not degraded."""
    r = HybridRetriever(seeded_db, make_settings(tmp_path, enabled=False))
    diag = SearchDiagnostics()

    assert r.search("kerberoasting", mode="keyword", limit=5, diagnostics=diag)

    assert diag.backend == "lexical"
    assert diag.semantic_requested is False
    assert diag.degraded is False


def test_hybrid_reports_both_backends_when_semantic_contributes(tmp_path, seeded_db):
    from blackbook.embeddings import embed_missing_chunks

    settings = make_settings(tmp_path, enabled=True)
    emb = FakeEmbedder()
    embed_missing_chunks(seeded_db, emb)
    r = HybridRetriever(seeded_db, settings)
    r._semantic = SemanticRetriever(seeded_db, settings, embedder=emb)
    diag = SearchDiagnostics()

    assert r.search("kerberoasting SPN tickets", mode="hybrid", limit=5, diagnostics=diag)

    assert diag.backends_queried == ["lexical", "semantic"]
    assert diag.backends_contributed == ["lexical", "semantic"]
    assert diag.backend == "lexical+semantic"
    assert diag.degraded is False


def test_diagnostics_are_optional(tmp_path, seeded_db):
    """The out-parameter is additive: omitting it changes nothing."""
    r = HybridRetriever(seeded_db, make_settings(tmp_path))
    assert r.search("kerberoasting", mode="semantic", limit=5)


# ---------------------------------------------------------------------------
# A2: the MCP layer surfaces the backend and explains a degradation
# ---------------------------------------------------------------------------


def _tools(tmp_path, seeded_db, enabled=False):
    return KnowledgeTools(seeded_db, make_settings(tmp_path, enabled=enabled))


def test_search_output_reports_backend(tmp_path, seeded_db):
    out = _tools(tmp_path, seeded_db).knowledge_search(
        SearchInput(query="kerberoasting", mode="keyword")
    )
    assert out.backend == "lexical"
    assert out.degraded is False


def test_search_output_explains_disabled_semantic(tmp_path, seeded_db):
    out = _tools(tmp_path, seeded_db).knowledge_search(
        SearchInput(query="kerberoasting", mode="semantic")
    )
    assert out.count >= 1           # degraded, not empty
    assert out.backend == "lexical"
    assert out.degraded is True
    assert "lexical" in out.note
    assert "embeddings.enabled" in out.note


def test_search_output_explains_empty_vector_index(tmp_path, seeded_db):
    tools = _tools(tmp_path, seeded_db, enabled=True)
    tools.settings.embeddings.model = "fake-bow-64"
    tools.retriever._semantic = SemanticRetriever(
        seeded_db, tools.settings, embedder=FakeEmbedder()
    )
    out = tools.knowledge_search(SearchInput(query="kerberoasting", mode="semantic"))
    assert out.count >= 1
    assert out.degraded is True
    assert "blackbook embed" in out.note


def test_search_output_explains_unavailable_semantic_backend(tmp_path, seeded_db, monkeypatch):
    """A missing [semantic] extra is reported, not swallowed."""
    import blackbook.retrieval.semantic as sem

    def boom(*a, **k):
        raise RuntimeError("no sentence-transformers here")

    monkeypatch.setattr(sem, "SemanticRetriever", boom)
    tools = _tools(tmp_path, seeded_db, enabled=True)
    out = tools.knowledge_search(SearchInput(query="kerberoasting", mode="semantic"))
    assert out.count >= 1
    assert out.degraded is True
    assert "could not start" in out.note
    assert "sentence-transformers" in out.note


# ---------------------------------------------------------------------------
# A3: the semantic index is honest about its own cost and readiness
# ---------------------------------------------------------------------------


def _semantic(tmp_path, seeded_db, embed=True, **kwargs):
    """A SemanticRetriever over the seeded corpus, model-free."""
    from blackbook.embeddings import embed_missing_chunks

    settings = make_settings(tmp_path, enabled=True)
    emb = FakeEmbedder()
    if embed:
        embed_missing_chunks(seeded_db, emb)
    return SemanticRetriever(seeded_db, settings, embedder=emb), emb


def test_semantic_status_reports_disabled(tmp_path, seeded_db):
    """Off by default: the status says so instead of leaving it to be guessed."""
    out = _tools(tmp_path, seeded_db, enabled=False).knowledge_sources(
        KnowledgeSourceInput()
    )
    assert out.semantic is not None
    assert out.semantic.enabled is False
    assert out.semantic.ready is False
    assert "embeddings.enabled" in out.semantic.note


def test_semantic_status_reports_enabled_but_empty(tmp_path, seeded_db):
    """The second, quieter way semantic is dead: on, but no vectors stored."""
    out = _tools(tmp_path, seeded_db, enabled=True).knowledge_sources(
        KnowledgeSourceInput()
    )
    assert out.semantic.enabled is True
    assert out.semantic.vectors == 0
    assert out.semantic.ready is False
    assert "blackbook embed" in out.semantic.note


def test_semantic_status_ready_when_vectors_exist(tmp_path, seeded_db):
    from blackbook.embeddings import embed_missing_chunks

    tools = _tools(tmp_path, seeded_db, enabled=True)
    embed_missing_chunks(seeded_db, FakeEmbedder())

    out = tools.knowledge_sources(KnowledgeSourceInput())

    assert out.semantic.vectors == 3
    assert out.semantic.ready is True
    assert out.semantic.note.startswith("3 vectors")


def test_semantic_status_omitted_for_single_source_lookup(tmp_path, seeded_db):
    """A per-source question is not a corpus-wide one."""
    out = _tools(tmp_path, seeded_db).knowledge_sources(
        KnowledgeSourceInput(source="hacktricks")
    )
    assert out.semantic is None


def test_semantic_status_costs_no_model_load(tmp_path, seeded_db, monkeypatch):
    """Reporting readiness must not import/load the embedding model."""
    import blackbook.retrieval.semantic as sem

    def boom(*a, **k):
        raise AssertionError("knowledge_sources constructed an Embedder")

    monkeypatch.setattr(sem, "SemanticRetriever", boom)
    out = _tools(tmp_path, seeded_db, enabled=True).knowledge_sources(
        KnowledgeSourceInput()
    )
    assert out.semantic is not None and out.semantic.ready is False


def test_describe_reports_index_shape(tmp_path, seeded_db, monkeypatch):
    sr, emb = _semantic(tmp_path, seeded_db)

    info = sr.describe()

    assert info["model"] == emb.model_name
    assert info["vectors"] == 3
    assert info["dim"] == emb.dim
    assert info["bytes"] == 3 * emb.dim * 4
    assert info["over_warn_bytes"] is False
    # A status call must not pull the matrix into memory.
    assert sr._cache == {}

    monkeypatch.setattr(SemanticRetriever, "MATRIX_WARN_BYTES", 1)
    assert sr.describe()["over_warn_bytes"] is True


def test_matrix_cache_is_bounded_by_entry_count(tmp_path, seeded_db, monkeypatch):
    """Alternating source filters must not multiply the resident index."""
    sr, _ = _semantic(tmp_path, seeded_db)
    monkeypatch.setattr(SemanticRetriever, "CACHE_MAX_ENTRIES", 1)

    ids_a, mat_a = sr._matrix(["hacktricks"])
    ids_b, _ = sr._matrix(["0xdf"])

    # Two distinct filters, one resident matrix: the older entry was evicted.
    assert len(sr._cache) == 1
    assert set(sr._cache) == {("0xdf",)}
    assert ids_a and ids_b

    # The evicted filter is recomputed rather than served stale.
    ids_a2, mat_a2 = sr._matrix(["hacktricks"])
    assert ids_a2 == ids_a
    assert mat_a2 is not mat_a


def test_matrix_cache_is_bounded_by_bytes(tmp_path, seeded_db, monkeypatch):
    """An entry larger than the whole budget is served but never pinned."""
    sr, _ = _semantic(tmp_path, seeded_db)
    monkeypatch.setattr(SemanticRetriever, "CACHE_MAX_BYTES", 1)

    ids, matrix = sr._matrix(None)

    assert ids and matrix is not None
    assert sr._cache == {}
    assert sr._cache_bytes == 0


def test_matrix_cache_hit_survives_eviction_pressure(tmp_path, seeded_db, monkeypatch):
    """A hit refreshes LRU position, so a hot filter is not evicted by a cold one."""
    sr, _ = _semantic(tmp_path, seeded_db)
    monkeypatch.setattr(SemanticRetriever, "CACHE_MAX_ENTRIES", 2)

    sr._matrix(["hacktricks"])
    _, mat_hot = sr._matrix(["hacktricks"])  # refresh LRU position
    sr._matrix(["0xdf"])
    _, mat_hot_again = sr._matrix(["hacktricks"])

    assert mat_hot_again is mat_hot  # still cached


def test_large_matrix_logs_one_warning_per_size(tmp_path, seeded_db, monkeypatch, caplog):
    """The flat scan's resident cost is named once, not on every miss."""
    sr, emb = _semantic(tmp_path, seeded_db)
    monkeypatch.setattr(SemanticRetriever, "MATRIX_WARN_BYTES", 1)
    # Never cache, so every call is a miss and reaches the warning check.
    monkeypatch.setattr(SemanticRetriever, "CACHE_MAX_BYTES", 1)

    with caplog.at_level(logging.WARNING, logger="blackbook.retrieval.semantic"):
        sr._matrix(None)
        sr._matrix(None)  # same size, same version: no repeat
        # Re-embed: the version moves, so the new size is reported again.
        from blackbook.embeddings import embed_missing_chunks

        with seeded_db.session():
            seeded_db.delete_embeddings(emb.model_name)
        embed_missing_chunks(seeded_db, emb)
        sr._matrix(None)

    warnings = [r.getMessage() for r in caplog.records if "flat index holds" in r.getMessage()]
    assert len(warnings) == 2, warnings
    assert "3 vectors" in warnings[0]


def test_missing_embeddings_never_reach_the_matrix(tmp_path, seeded_db):
    """A chunk with no vector is simply absent from the flat index."""
    sr, emb = _semantic(tmp_path, seeded_db, embed=False)
    assert sr.search("kerberoasting") == []
    assert sr.describe()["vectors"] == 0


# ---------------------------------------------------------------------------
# A4: BM25 column weights
# ---------------------------------------------------------------------------


def _weight_corpus(db):
    """Two docs: one *about* the term, one that merely repeats it in the body."""
    with db.session():
        db.upsert_source(Source(source_id="s", name="S", authority="trusted"))
        titled = db.upsert_document(
            Document(
                source_id="s",
                external_id="titled.md",
                title="Kerberoasting",
                content_hash=sha256_text("titled"),
            )
        )
        body = db.upsert_document(
            Document(
                source_id="s",
                external_id="body.md",
                title="Weekly misc notes",
                content_hash=sha256_text("body"),
            )
        )
        db.replace_chunks(
            titled,
            [
                Chunk(
                    doc_id=titled,
                    ordinal=0,
                    text="A short page whose heading names the topic and says little else.",
                    section_path=["Intro"],
                    token_estimate=12,
                    content_hash=sha256_text("tc"),
                )
            ],
        )
        db.replace_chunks(
            body,
            [
                Chunk(
                    doc_id=body,
                    ordinal=0,
                    text=" ".join(["kerberoasting"] * 30),
                    section_path=["Misc"],
                    token_estimate=30,
                    content_hash=sha256_text("bc"),
                )
            ],
        )
    return titled, body


def test_fts_column_weights_reorder_without_filtering(tmp_path):
    db = Database(tmp_path / "w.db")
    try:
        titled, body = _weight_corpus(db)
        query = "kerberoasting"

        default = db.fts_search(query, limit=10)
        title_heavy = db.fts_search(query, limit=10, bm25_weights=(1000.0, 1.0, 1.0))
        body_heavy = db.fts_search(query, limit=10, bm25_weights=(0.01, 0.01, 1.0))

        assert len(default) == 2
        # Weights change ranking only; the matched set is identical.
        ids = lambda rows: {r["chunk_id"] for r in rows}
        assert ids(default) == ids(title_heavy) == ids(body_heavy)

        assert title_heavy[0]["doc_id"] == titled
        assert body_heavy[0]["doc_id"] == body
        # Best-first ordering is preserved under weighting (bm25 ascending).
        assert title_heavy[0]["bm25"] < title_heavy[1]["bm25"]
    finally:
        db.close()


def test_default_weights_are_the_declared_constant(tmp_path):
    db = Database(tmp_path / "w.db")
    try:
        _weight_corpus(db)
        assert Database.BM25_WEIGHTS == (5.0, 2.0, 1.0)
        default = db.fts_search("kerberoasting", limit=10)
        explicit = db.fts_search(
            "kerberoasting", limit=10, bm25_weights=Database.BM25_WEIGHTS
        )
        assert [r["bm25"] for r in default] == [r["bm25"] for r in explicit]
        # And the weights really are in play: unweighted bm25 gives other values.
        unweighted = db.fts_search("kerberoasting", limit=10, bm25_weights=(1.0, 1.0, 1.0))
        assert [r["bm25"] for r in default] != [r["bm25"] for r in unweighted]
    finally:
        db.close()


# ---------------------------------------------------------------------------
# A5: `doctor` reports which backends a search can actually use
# ---------------------------------------------------------------------------


def _doctor_stdout(tmp_path, monkeypatch, *, enabled, embed=False):
    """Run ``blackbook doctor`` over a home-scoped corpus; return its output.

    The database is built at ``<home>/data.db`` rather than using the ``db``
    fixture, because that is the path the CLI resolves from the config it is
    handed: a fixture database elsewhere would leave doctor reading an empty
    one and the check under test would never see the vectors.
    """
    from typer.testing import CliRunner

    from blackbook.cli.main import app

    monkeypatch.setenv("BLACKBOOK_HOME", str(tmp_path))
    config = tmp_path / "config.yaml"
    config.write_text(
        f"home: {tmp_path}\n"
        f"embeddings:\n  enabled: {str(enabled).lower()}\n  model: fake-bow-64\n"
        "sources:\n  - id: hacktricks\n    name: HackTricks\n"
        "    type: git\n    url: https://example.test/r.git\n"
    )
    monkeypatch.setenv("BLACKBOOK_CONFIG", str(config))
    # The check table is wider than a default 80-column capture.
    monkeypatch.setenv("COLUMNS", "200")

    database = Database(tmp_path / "data.db")
    with database.session():
        database.upsert_source(Source(source_id="hacktricks", name="HackTricks"))
        doc = database.upsert_document(
            Document(
                source_id="hacktricks",
                external_id="ad/kerberoasting.md",
                title="Kerberoasting",
                content_hash=sha256_text("kerberoasting doc"),
            )
        )
        # Three chunks, so the reported vector count reads as a plural.
        database.replace_chunks(
            doc,
            [
                Chunk(
                    doc_id=doc,
                    ordinal=i,
                    text=f"Kerberoasting requests SPN service tickets ({i}).",
                    section_path=["AD"],
                    token_estimate=8,
                    content_hash=sha256_text(f"kc{i}"),
                )
                for i in range(3)
            ],
        )
    if embed:
        from blackbook.embeddings import embed_missing_chunks

        embed_missing_chunks(database, FakeEmbedder())
    database.close()

    return CliRunner().invoke(app, ["doctor"]).stdout


def test_doctor_reports_lexical_only_when_semantic_disabled(tmp_path, monkeypatch):
    """The default install: doctor must say the semantic backend is absent."""
    out = _doctor_stdout(tmp_path, monkeypatch, enabled=False)
    assert "search diagnostics" in out
    assert "lexical only (semantic disabled)" in out


def test_doctor_warns_when_semantic_is_on_but_inert(tmp_path, monkeypatch):
    """On with no vectors is a degradation, and the row must be actionable."""
    out = _doctor_stdout(tmp_path, monkeypatch, enabled=True)
    assert "search diagnostics" in out
    assert "degrade to lexical" in out
    assert "blackbook embed" in out


def test_doctor_reports_both_backends_when_vectors_exist(tmp_path, monkeypatch):
    """Vectors present: the row names the pair, so a healthy install is legible."""
    out = _doctor_stdout(tmp_path, monkeypatch, enabled=True, embed=True)
    assert "search diagnostics" in out
    assert "lexical + semantic (3 vectors)" in out
