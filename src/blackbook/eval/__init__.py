"""BlackBook evaluation suite.

A self-contained, offline benchmark for the retrieval stack. It builds a small
*curated* corpus of clearly-labelled synthetic security documents, indexes them
through the **real** chunker and storage layer, and measures the real retriever:

* **citation integrity** — every returned ``ref.chunk_id`` must resolve back to
  real indexed text (the "NO HALLUCINATED CITATIONS" invariant, measured);
* **ranking quality** — recall@k and MRR against a labelled gold query set;
* **latency** — per-query wall-clock, so a performance regression is visible.

Nothing here fetches or executes anything. The benchmark corpus is authored,
not scraped, and is written to an isolated database (the CLI uses an in-memory
one) so it never touches the user's real index. It is a measuring instrument,
not a knowledge source: its documents exist to exercise retrieval mechanics.
"""

from __future__ import annotations

from blackbook.eval.corpus import BENCHMARK_DOCS, EVAL_SOURCES, build_eval_corpus
from blackbook.eval.goldset import GOLD_QUERIES, GoldQuery
from blackbook.eval.harness import EvalReport, QueryResult, run_evaluation
from blackbook.eval.metrics import (
    hit_rate,
    mrr,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "BENCHMARK_DOCS",
    "EVAL_SOURCES",
    "build_eval_corpus",
    "GOLD_QUERIES",
    "GoldQuery",
    "EvalReport",
    "QueryResult",
    "run_evaluation",
    "hit_rate",
    "mrr",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
