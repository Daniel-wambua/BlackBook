"""The evaluation harness.

:func:`run_evaluation` drives the real retriever over a populated benchmark
database and produces an :class:`EvalReport`. It measures four things, all from
real behaviour — nothing is asserted or thresholded here (thresholds live in the
tests and the CLI exit code):

* **ranking quality** — document-level recall@k and reciprocal rank per query,
  aggregated to MRR / hit-rate / mean-recall;
* **citation integrity** — every chunk the retriever returns is resolved back
  through the same primitive :func:`knowledge_source` uses; a chunk that fails to
  resolve to real text would be a hallucinated citation (the invariant we forbid);
* **refusal behaviour** — queries the corpus cannot answer must not look like
  well-answered ones. Recall cannot express this (an unanswerable query has no
  relevant document to rank first), so these are scored on their *top* score and
  reported as a separation from the weakest answered query;
* **latency** — per-query wall-clock, aggregated to p50 / p95.

The retriever is the exact instance the MCP tools use (``KnowledgeTools.retriever``),
so a regression in the shipped path is what gets measured.

Unanswerable queries are excluded from the MRR / hit-rate / recall aggregates.
Including them would not be neutral: with an empty relevant set they score 0.0 by
construction, so they would only dilute the aggregate by a constant, and they
would drag the reported numbers down for a reason that says nothing about
ranking. Their own signal is :attr:`EvalReport.negative_separation`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from blackbook.config import Settings
from blackbook.eval.goldset import GOLD_QUERIES, GoldQuery
from blackbook.eval.metrics import (
    hit_rate,
    mrr,
    percentile,
    recall_at_k,
    reciprocal_rank,
)
from blackbook.knowledge.sources import get_chunk_excerpt
from blackbook.mcp.tools import KnowledgeTools
from blackbook.storage.database import Database

DEFAULT_K = 5


@dataclass
class QueryResult:
    """Per-query outcome."""

    qid: str
    query: str
    mode: str
    ranked: tuple[str, ...]  # document external_ids, best-first, deduplicated
    relevant: tuple[str, ...]
    recall_at_k: float
    reciprocal_rank: float
    hit: bool
    documents_returned: int
    citations_checked: int
    citations_resolved: int
    latency_ms: float
    # Score of the best-ranked chunk. For an answered query this is how strongly
    # anything matched; for an unanswerable one it is how confidently the
    # retriever answered a question the corpus cannot answer.
    top_score: float = 0.0
    unanswerable: bool = False

    def as_dict(self) -> dict:
        return {
            "qid": self.qid,
            "query": self.query,
            "mode": self.mode,
            "ranked": list(self.ranked),
            "relevant": list(self.relevant),
            "recall_at_k": round(self.recall_at_k, 4),
            "reciprocal_rank": round(self.reciprocal_rank, 4),
            "hit": self.hit,
            "unanswerable": self.unanswerable,
            "top_score": round(self.top_score, 4),
            "documents_returned": self.documents_returned,
            "citations_checked": self.citations_checked,
            "citations_resolved": self.citations_resolved,
            "latency_ms": round(self.latency_ms, 3),
        }


@dataclass
class EvalReport:
    """Aggregate evaluation results.

    Ranking aggregates (``mrr`` / ``hit_rate`` / ``mean_recall_at_k``) cover the
    answerable queries only. ``unanswerable_count`` and the two score bounds
    describe the refusal behaviour separately.
    """

    k: int
    query_results: list[QueryResult] = field(default_factory=list)
    mrr: float = 0.0
    hit_rate: float = 0.0
    mean_recall_at_k: float = 0.0
    citations_total: int = 0
    citations_resolved: int = 0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    unanswerable_count: int = 0
    # Highest top-score among unanswerable queries, and the lowest among
    # answered ones. A useful retriever keeps the former comfortably below the
    # latter; see `negative_separation`.
    unanswerable_top_score_max: float = 0.0
    answerable_top_score_min: float = 0.0

    @property
    def citation_integrity(self) -> float:
        """Fraction of returned citations that resolved to real indexed text.

        1.0 when no citations were produced (vacuously intact) — the CLI treats
        this the same as "no violations".
        """
        if self.citations_total == 0:
            return 1.0
        return self.citations_resolved / self.citations_total

    @property
    def negative_separation(self) -> float:
        """How far the weakest answered query outranks the confident nonsense.

        ``answerable_top_score_min - unanswerable_top_score_max``. Positive means
        every query with a real answer scored higher than every query without
        one, which is the property that lets a caller use the score as a signal
        at all. 0.0 when there are no unanswerable queries (nothing to separate
        from), which callers must treat as "not measured" rather than "passed".
        """
        if self.unanswerable_count == 0:
            return 0.0
        return self.answerable_top_score_min - self.unanswerable_top_score_max

    def as_dict(self) -> dict:
        return {
            "k": self.k,
            "queries": len(self.query_results),
            "mrr": round(self.mrr, 4),
            "hit_rate": round(self.hit_rate, 4),
            "mean_recall_at_k": round(self.mean_recall_at_k, 4),
            "citation_integrity": round(self.citation_integrity, 6),
            "citations_total": self.citations_total,
            "citations_resolved": self.citations_resolved,
            "latency_p50_ms": round(self.latency_p50_ms, 3),
            "latency_p95_ms": round(self.latency_p95_ms, 3),
            "unanswerable_count": self.unanswerable_count,
            "unanswerable_top_score_max": round(self.unanswerable_top_score_max, 4),
            "answerable_top_score_min": round(self.answerable_top_score_min, 4),
            "negative_separation": round(self.negative_separation, 4),
            "per_query": [qr.as_dict() for qr in self.query_results],
        }


def run_evaluation(
    db: Database,
    settings: Settings,
    *,
    queries: list[GoldQuery] | None = None,
    k: int = DEFAULT_K,
) -> EvalReport:
    """Run the gold queries over ``db`` and return an :class:`EvalReport`.

    ``db`` must already be populated (see
    :func:`blackbook.eval.corpus.build_eval_corpus`). ``db`` is treated as
    read-only here — the harness only searches and resolves.
    """
    tools = KnowledgeTools(db, settings)
    gold = queries if queries is not None else GOLD_QUERIES

    # Retrieve enough chunks that reranking's per-document cap still yields at
    # least k distinct documents to score against.
    chunk_limit = max(k * 3, 15)

    ext_cache: dict[int, str | None] = {}

    def external_id(doc_id: int) -> str | None:
        if doc_id not in ext_cache:
            row = db.get_document(doc_id)
            ext_cache[doc_id] = row["external_id"] if row else None
        return ext_cache[doc_id]

    results: list[QueryResult] = []
    latencies: list[float] = []
    cit_total = 0
    cit_ok = 0

    for gq in gold:
        t0 = time.perf_counter()
        hits = tools.retriever.search(
            gq.query,
            mode=gq.mode,
            source_ids=None,
            platform=gq.platform,
            categories=list(gq.categories) if gq.categories else None,
            limit=chunk_limit,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)

        ranked: list[str] = []
        seen: set[str] = set()
        q_checked = 0
        q_resolved = 0
        for h in hits:
            ext = external_id(h.doc_id)
            if ext and ext not in seen:
                seen.add(ext)
                ranked.append(ext)
            # Citation integrity: resolve exactly as knowledge_source would.
            q_checked += 1
            ex = get_chunk_excerpt(db, h.chunk_id)
            if ex is not None and ex.text and ex.chunk_id == h.chunk_id:
                q_resolved += 1

        cit_total += q_checked
        cit_ok += q_resolved

        rec = recall_at_k(ranked, gq.relevant, k)
        rr = reciprocal_rank(ranked, gq.relevant)
        results.append(
            QueryResult(
                qid=gq.qid,
                query=gq.query,
                mode=gq.mode,
                ranked=tuple(ranked),
                relevant=gq.relevant,
                recall_at_k=rec,
                reciprocal_rank=rr,
                hit=rr > 0.0,
                documents_returned=len(ranked),
                citations_checked=q_checked,
                citations_resolved=q_resolved,
                latency_ms=latency_ms,
                top_score=float(hits[0].score) if hits else 0.0,
                unanswerable=gq.unanswerable,
            )
        )

    # Ranking aggregates cover answered queries only. An unanswerable query has
    # an empty relevant set, so it would contribute a hard 0.0 to every one of
    # these — a constant dilution that measures the label, not the ranking.
    answered = [qr for qr in results if not qr.unanswerable]
    unanswerable = [qr for qr in results if qr.unanswerable]
    pairs = [(qr.ranked, qr.relevant) for qr in answered]
    mean_recall = (
        sum(qr.recall_at_k for qr in answered) / len(answered) if answered else 0.0
    )
    return EvalReport(
        k=k,
        query_results=results,
        mrr=mrr(pairs),
        hit_rate=hit_rate(pairs, k),
        mean_recall_at_k=mean_recall,
        citations_total=cit_total,
        citations_resolved=cit_ok,
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
        unanswerable_count=len(unanswerable),
        unanswerable_top_score_max=max(
            (qr.top_score for qr in unanswerable), default=0.0
        ),
        answerable_top_score_min=min((qr.top_score for qr in answered), default=0.0),
    )
