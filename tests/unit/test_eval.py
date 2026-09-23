"""Quality tests for the evaluation suite itself.

These lock in the invariants the benchmark exists to protect:

* the pure ranking metrics compute what they claim;
* the benchmark corpus builds deterministically and every gold label points at
  a document that actually exists in it (a typo fails loudly here, not silently
  as a zero score);
* the *real* retriever, run over the benchmark, produces intact citations and
  meets ranking floors, with the mode-specific ordering the reranker promises;
* the benchmark can still *fail* — the gold set covers every mode, carries
  genuine distractors and unanswerable queries, and at least one relevant
  document is still beaten by a distractor somewhere;
* unanswerable queries are scored on the precision side (score separation from
  answered queries) and excluded from the ranking aggregates;
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
        for rel in gq.relevant:
            assert rel in ext_ids, f"{gq.qid} references unknown doc {rel!r}"
        # An unanswerable query is defined by having *nothing* to retrieve, and
        # every other query must name at least one correct answer.
        if gq.unanswerable:
            assert not gq.relevant, (
                f"{gq.qid} is marked unanswerable but labels {gq.relevant}"
            )
        else:
            assert gq.relevant, f"{gq.qid} has no relevant labels"


def test_goldset_exercises_every_mode_and_stays_discriminating():
    """The gold set must be capable of failing, and of scoring each mode.

    An earlier version of this file had 14 single-answer queries and every one
    scored a perfect 1.0 — recall@k, MRR and hit-rate were all saturated, so no
    ranking regression could ever have been detected. These assertions keep that
    from recurring by pinning the properties that make the benchmark informative
    rather than the scores it happens to produce.
    """
    modes = {gq.mode for gq in GOLD_QUERIES}
    assert {"keyword", "technique", "case_similarity"} <= modes

    unanswerable = [gq for gq in GOLD_QUERIES if gq.unanswerable]
    assert len(unanswerable) >= 3, "negatives are what test the precision side"

    multi = [gq for gq in GOLD_QUERIES if len(gq.relevant) > 1]
    assert multi, "at least one query must have more than one correct answer"

    # Genuine distractors: documents no query labels as an answer. Without them
    # every query would be a single-document lookup and recall could not vary.
    labelled = {rel for gq in GOLD_QUERIES for rel in gq.relevant}
    distractors = {d.external_id for d in BENCHMARK_DOCS} - labelled
    assert len(distractors) >= 3, sorted(distractors)

    # Two different qids must not ask the same thing, or one is dead weight.
    qids = [gq.qid for gq in GOLD_QUERIES]
    assert len(qids) == len(set(qids))


def test_benchmark_is_not_saturated(tmp_path):
    """Some relevant document must be beaten by a distractor somewhere.

    This is the anti-saturation guard: it fails when the benchmark has stopped
    being able to detect a ranking regression. The right response is to add
    harder corpus material (or a harder query), never to delete this test — a
    benchmark where everything scores 1.0 measures nothing, which is exactly
    the state this suite was in before.
    """
    db, report = _report(tmp_path)
    try:
        answered = [qr for qr in report.query_results if not qr.unanswerable]
        missed = [
            qr.qid
            for qr in answered
            if qr.recall_at_k < 1.0 or qr.reciprocal_rank < 1.0
        ]
        assert missed, (
            "every gold query now scores perfectly - the corpus needs harder "
            "distractors for the benchmark to discriminate again"
        )
    finally:
        db.close()


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
    """Regression floors, set below the measured values with real headroom.

    Measured on this corpus (lexical-only, deterministic): MRR 0.906, mean
    recall@5 0.979, hit-rate@5 1.00. The floors are not set at the measured
    numbers on purpose — they are the point below which the shipped retriever
    has genuinely regressed, not a licence to keep the score where it is.
    """
    db, report = _report(tmp_path)
    try:
        assert report.mrr >= 0.8
        assert report.mean_recall_at_k >= 0.8
        assert report.hit_rate >= 0.8
        assert report.latency_p95_ms >= report.latency_p50_ms >= 0.0
        # Ranking aggregates must describe answered queries only: including the
        # unanswerable ones would add a constant 0.0 per negative and report a
        # lower score for a reason that says nothing about ranking.
        assert report.unanswerable_count == sum(
            1 for qr in report.query_results if qr.unanswerable
        )
        assert report.unanswerable_count >= 3
    finally:
        db.close()


def test_unanswerable_queries_are_scored_and_separated(tmp_path):
    """A query the corpus cannot answer must not look like an answered one.

    Recall cannot express this — an unanswerable query has no relevant document
    to rank first, so it scores 0.0 by construction and says nothing. What is
    asserted instead is the *separation*: every answered query's best chunk
    scored higher than every unanswerable query's best chunk.

    The margin is thin and that is the honest result. Measured: the worst
    answered query is `par-metadata-creds` at 0.8175 (a paraphrase that never
    names SSRF), while the strongest negative is `neg-cloud-iam` at 0.7931 —
    it shares the token "rotation" with a golden-ticket page that says "Rotating
    the krbtgt password". The FTS5 query is OR-joined, so a single shared
    generic token is enough to drag in an unrelated document; negatives exist to
    make that visible rather than to pretend it does not happen.
    """
    db, report = _report(tmp_path)
    try:
        assert report.unanswerable_count >= 3
        assert report.negative_separation > 0, (
            f"an unanswerable query scored at or above the weakest answered one "
            f"(negative max {report.unanswerable_top_score_max:.4f} vs "
            f"answerable min {report.answerable_top_score_min:.4f})"
        )
        # Floor with headroom under the measured 0.0243, not at it.
        assert report.negative_separation >= 0.02

        # The strongest negative must lose to the *median* answered query by a
        # comfortable margin — a single thin comparison could be coincidence.
        answered = sorted(
            qr.top_score for qr in report.query_results if not qr.unanswerable
        )
        median = answered[len(answered) // 2]
        assert median - report.unanswerable_top_score_max > 0.05

        # And a negative that matches nothing at all returns nothing, rather
        # than padding the result list to look helpful.
        wireless = next(qr for qr in report.query_results if qr.qid == "neg-wireless")
        assert wireless.documents_returned == 0
        assert wireless.top_score == 0.0
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


def test_mode_bonus_picks_the_writeup_for_a_near_duplicate(tmp_path):
    """The mode bonus is observable: same subject, different winner per mode.

    ``wu/htb-blackfield.md`` and ``ref/asrep-roasting.md`` cover the same ground,
    as do ``wu/pg-hawat.md`` and ``ref/ssrf.md``. For a descriptive query both
    stay in scope, and case_similarity must put the writeup first.

    The two pairs fail differently in technique mode, and the difference is the
    point:

    * blackfield/asrep is a real *flip*. The reference page scores higher on its
      own merits (0.898 vs 0.835), so the writeup wins case_similarity only
      because of the mode bonus, and technique restores the reference page.
    * hawat/ssrf is not a flip and never was. The writeup scores higher to begin
      with (0.932 vs 0.763) because the query describes the *case* more closely
      than the reference page. It therefore wins in both modes, and asserting
      otherwise would be asserting that a mode bonus can demote a document that
      legitimately matched better.
    """
    db = Database(tmp_path / "mode.db")
    build_eval_corpus(db)
    try:
        from blackbook.mcp.tools import KnowledgeTools

        tools = KnowledgeTools(db, Settings())

        def top(query, mode):
            hits = tools.retriever.search(query, mode=mode, limit=5)
            return db.get_document(hits[0].doc_id)["external_id"] if hits else None

        def base(query, external_id):
            for h in tools.retriever.search(query, mode="keyword", limit=30):
                if db.get_document(h.doc_id)["external_id"] == external_id:
                    return h.score
            return 0.0

        pairs = [
            (
                "active directory kerberos pre-authentication disabled backup "
                "operator secrets dump",
                "ref/asrep-roasting.md",
                "wu/htb-blackfield.md",
                True,  # the reference page leads on base score: a true flip
            ),
            (
                "web application fetches a user supplied url reaching an "
                "internal service",
                "ref/ssrf.md",
                "wu/pg-hawat.md",
                False,  # the writeup already leads on base score
            ),
        ]
        for query, ref, wu, reference_leads in pairs:
            ref_base, wu_base = base(query, ref), base(query, wu)
            assert (ref_base > wu_base) is reference_leads, (query, ref_base, wu_base)

            # The mode's promise holds either way: a case query returns the case.
            assert top(query, "case_similarity") == wu, query
            # And technique returns the reference page only when it is not
            # already behind on the merits.
            expected = ref if reference_leads else wu
            assert top(query, "technique") == expected, query
    finally:
        db.close()


def test_mode_bonus_cannot_flip_a_dominant_base_score(tmp_path):
    """Pin *where* the mode bonus stops working, measured rather than assumed.

    The reranker's final score is ``base * authority * (1 + bonus)`` with
    ``_MODE_BONUS = 0.3``, so a writeup overtakes the reference page only when
    its base score is at least ~1/1.3 = 0.77 of the reference page's. This test
    states that boundary as a property of the shipped constants, which is what
    makes the case_similarity gold queries meaningful: they are phrased to land
    inside the band.

    It also records the reason a bare technique query cannot be flipped — the
    FTS index does not stem, so "kerberoasting" does not match a body that says
    "kerberoast", and the writeup's base score is 0.0. No mode bonus can rescue
    a document that did not match at all.
    """
    from blackbook.retrieval import reranker

    db = Database(tmp_path / "bonus.db")
    build_eval_corpus(db)
    try:
        from blackbook.mcp.tools import KnowledgeTools

        tools = KnowledgeTools(db, Settings())
        threshold = 1.0 / (1.0 + reranker._MODE_BONUS)

        def base(query, external_id):
            """Mode-neutral base score for one document under a query."""
            for h in tools.retriever.search(query, mode="keyword", limit=30):
                if db.get_document(h.doc_id)["external_id"] == external_id:
                    return h.score
            return 0.0

        within = [
            (
                "active directory kerberos pre-authentication disabled backup "
                "operator secrets dump",
                "ref/asrep-roasting.md",
                "wu/htb-blackfield.md",
            ),
            (
                "active directory as-rep roasting to domain admin foothold windows",
                "ref/asrep-roasting.md",
                "wu/htb-forest.md",
            ),
            (
                "web application fetches a user supplied url reaching an internal service",
                "ref/ssrf.md",
                "wu/pg-hawat.md",
            ),
        ]
        for query, ref, wu in within:
            ratio = base(query, wu) / base(query, ref)
            assert ratio >= threshold, (query, ratio, threshold)

        # And the case the bonus cannot win: the writeup did not match the term.
        assert base("kerberoasting", "wu/htb-sizzle.md") == 0.0
        assert base("kerberoasting", "ref/kerberoasting.md") > 0.0
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
