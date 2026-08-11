"""Quality tests for the evaluation suite itself.

These lock in the invariants the benchmark exists to protect:

* the pure ranking metrics compute what they claim;
* the benchmark corpus builds deterministically and every gold label points at
  a document that actually exists in it (a typo fails loudly here, not silently
  as a zero score);
* the *real* retriever, run over the benchmark, produces intact citations and
  meets ranking floors, with the mode-specific ordering the reranker promises;
* the lexical BM25->score mapping stays monotonically increasing in match
  strength — a regression guard for the inverted-score bug the benchmark caught.
"""

from __future__ import annotations

from blackbook.config import Settings
from blackbook.eval import (
    BENCHMARK_DOCS,
    EVAL_SOURCES,
    GOLD_QUERIES,
    build_eval_corpus,
    run_evaluation,
)
from blackbook.eval.metrics import (
    hit_rate,
    mrr,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from blackbook.retrieval.lexical import LexicalRetriever
from blackbook.storage import Database


# --------------------------------------------------------------------------- #
# Pure metrics                                                                 #
# --------------------------------------------------------------------------- #
def test_metrics_basic():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, {"a", "c"}, 4) == 1.0
    assert recall_at_k(ranked, {"a", "z"}, 4) == 0.5
    assert recall_at_k(ranked, {"c"}, 2) == 0.0  # c is outside top-2
    assert precision_at_k(ranked, {"a", "b"}, 2) == 1.0
    assert reciprocal_rank(ranked, {"c"}) == 1.0 / 3.0
    assert reciprocal_rank(ranked, {"z"}) == 0.0


def test_metrics_empty_inputs():
    assert recall_at_k([], {"a"}, 5) == 0.0
    assert recall_at_k(["a"], set(), 5) == 0.0  # no relevant -> nothing to recall
    assert precision_at_k([], {"a"}, 5) == 0.0
    assert mrr([]) == 0.0
    assert hit_rate([], 5) == 0.0
    assert percentile([], 95) == 0.0
    assert percentile([7.0], 50) == 7.0


def test_percentile_interpolates():
    data = [0.0, 10.0]
    assert percentile(data, 50) == 5.0
    assert percentile(data, 0) == 0.0
    assert percentile(data, 100) == 10.0


# --------------------------------------------------------------------------- #
# Corpus determinism + goldset consistency                                    #
# --------------------------------------------------------------------------- #
def test_corpus_build_is_deterministic(tmp_path):
    db1 = Database(tmp_path / "a.db")
    db2 = Database(tmp_path / "b.db")
    try:
        c1 = build_eval_corpus(db1)
        c2 = build_eval_corpus(db2)
        # Two independent builds agree exactly.
        assert c1 == c2
        assert c1["sources"] == len(EVAL_SOURCES)
        assert c1["documents"] == len(BENCHMARK_DOCS)
        assert c1["chunks"] > c1["documents"] > 0
        # Idempotent: re-running against the same DB is stable (hash-guarded).
        assert build_eval_corpus(db1) == c1
    finally:
        db1.close()
        db2.close()


def test_gold_labels_exist_in_corpus():
    ext_ids = {d.external_id for d in BENCHMARK_DOCS}
    assert ext_ids, "corpus is empty"
    for gq in GOLD_QUERIES:
        assert gq.relevant, f"{gq.qid} has no relevant labels"
        for rel in gq.relevant:
            assert rel in ext_ids, f"{gq.qid} references unknown doc {rel!r}"


# --------------------------------------------------------------------------- #
# End-to-end: real retriever over the benchmark                               #
# --------------------------------------------------------------------------- #
def _report(tmp_path):
    db = Database(tmp_path / "eval.db")
    build_eval_corpus(db)
    return db, run_evaluation(db, Settings(), k=5)


def test_citation_integrity_is_perfect(tmp_path):
    db, report = _report(tmp_path)
    try:
        # The forbidden invariant: every returned chunk resolves to real text.
        assert report.citations_total > 0
        assert report.citations_resolved == report.citations_total
        assert report.citation_integrity == 1.0
    finally:
        db.close()


def test_ranking_meets_floors(tmp_path):
    db, report = _report(tmp_path)
    try:
        assert report.mrr >= 0.8
        assert report.mean_recall_at_k >= 0.8
        assert report.hit_rate >= 0.8
        assert report.latency_p95_ms >= report.latency_p50_ms >= 0.0
    finally:
        db.close()


def test_mode_specific_ordering(tmp_path):
    """technique -> reference first; case_similarity -> writeup first."""
    db, report = _report(tmp_path)
    try:
        by_qid = {qr.qid: qr for qr in report.query_results}
        tech = by_qid["tech-kerberoast"]
        assert tech.ranked[0] == "ref/kerberoasting.md"
        case = by_qid["case-forest"]
        assert case.ranked[0] == "wu/htb-forest.md"
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# Regression guard: lexical BM25 -> score monotonicity                        #
# --------------------------------------------------------------------------- #
def test_lexical_score_monotonic_in_match_strength(tmp_path):
    """Stronger BM25 (more negative) must map to a *higher* score in [0, 1).

    Pins the fix for the inverted-score bug: the reranker sorts by score
    descending, so the mapping has to increase with match strength or the
    ranking silently reverses.
    """
    db = Database(tmp_path / "lex.db")
    try:
        build_eval_corpus(db)
        hits = LexicalRetriever(db).search(
            "kerberoasting service principal name TGS crack offline", limit=15
        )
        assert len(hits) >= 2
        for h in hits:
            assert 0.0 <= h.score < 1.0
            strength = max(0.0, -h.bm25)
            assert abs(h.score - strength / (1.0 + strength)) < 1e-9
        # DB returns best-first (bm25 ascending); scores must be non-increasing
        # and track strength in the same direction.
        for a, b in zip(hits, hits[1:]):
            assert a.score >= b.score
            if a.bm25 != b.bm25:
                assert (a.bm25 < b.bm25) == (a.score > b.score)
    finally:
        db.close()
