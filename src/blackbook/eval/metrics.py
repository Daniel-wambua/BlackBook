"""Ranking-quality metrics.

Pure functions over ranked result lists. Each takes ``ranked`` — the ordered
list of *retrieved* item identifiers (best first) — and ``relevant`` — the set
of identifiers that are known-relevant for the query. No I/O, no globals; every
function is deterministic and independently testable.

Identifiers are opaque (we use document ``external_id`` in the harness), so
these functions never touch the database or the retriever.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def _relevant_set(relevant: Iterable[str]) -> set[str]:
    return set(relevant)


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the relevant items that appear in the top ``k`` results.

    Returns 0.0 when there are no relevant items (nothing to recall) so an
    empty gold label never inflates the score.
    """
    rel = _relevant_set(relevant)
    if not rel:
        return 0.0
    top = list(ranked)[: max(0, k)]
    hits = sum(1 for item in rel if item in top)
    return hits / len(rel)


def precision_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top ``k`` results that are relevant.

    Denominator is ``k`` (or the number of results if fewer), so padding the
    list with irrelevant items is correctly penalised.
    """
    rel = _relevant_set(relevant)
    top = list(ranked)[: max(0, k)]
    if not top:
        return 0.0
    hits = sum(1 for item in top if item in rel)
    return hits / len(top)


def reciprocal_rank(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    """Reciprocal of the 1-based rank of the first relevant hit; 0 if none."""
    rel = _relevant_set(relevant)
    for idx, item in enumerate(ranked, start=1):
        if item in rel:
            return 1.0 / idx
    return 0.0


def mrr(rankings: Iterable[tuple[Sequence[str], Iterable[str]]]) -> float:
    """Mean reciprocal rank over ``(ranked, relevant)`` pairs.

    Returns 0.0 for an empty set of queries.
    """
    pairs = list(rankings)
    if not pairs:
        return 0.0
    total = sum(reciprocal_rank(ranked, relevant) for ranked, relevant in pairs)
    return total / len(pairs)


def hit_rate(rankings: Iterable[tuple[Sequence[str], Iterable[str]]], k: int) -> float:
    """Fraction of queries with at least one relevant item in the top ``k``.

    Returns 0.0 for an empty set of queries.
    """
    pairs = list(rankings)
    if not pairs:
        return 0.0
    hits = 0
    for ranked, relevant in pairs:
        rel = _relevant_set(relevant)
        top = list(ranked)[: max(0, k)]
        if any(item in rel for item in top):
            hits += 1
    return hits / len(pairs)


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile (``pct`` in [0, 100]); 0.0 if empty.

    Used by the harness for latency p50/p95. Implemented here so the eval
    package has no numpy dependency and the number is reproducible.
    """
    data = sorted(values)
    if not data:
        return 0.0
    if len(data) == 1:
        return float(data[0])
    rank = (pct / 100.0) * (len(data) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(data) - 1)
    frac = rank - lo
    return float(data[lo] + (data[hi] - data[lo]) * frac)
