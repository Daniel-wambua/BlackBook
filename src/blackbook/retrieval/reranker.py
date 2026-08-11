"""Reranking and source-diversity.

Raw lexical/semantic scores are combined with metadata signals (source
authority, platform/category match, section relevance), then reordered to
avoid returning many near-identical chunks from a single document. The goal is
*diverse, high-value evidence*, not a dump of the top N rows from one page.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from blackbook.knowledge.vocab import extract_terms, is_writeup_category
from blackbook.retrieval.dedup import shingle_overlap
from blackbook.retrieval.lexical import LexicalHit

# Authority weight: trusted official material outranks unknown/user content.
_AUTHORITY_WEIGHT = {
    "official": 1.0,
    "trusted": 0.9,
    "user": 0.75,
    "unknown": 0.6,
}

# How strongly an intent mode nudges a matching hit. Large enough to reorder
# genuinely on-intent hits above generic keyword matches, small enough that a
# far weaker base hit can't leapfrog a much stronger one on mode alone.
_MODE_BONUS = 0.3


def _authority_factor(authority: str) -> float:
    return _AUTHORITY_WEIGHT.get(authority, 0.6)


def _is_technique_hit(hit: LexicalHit) -> bool:
    """True when a controlled technique term names the hit's title/heading.

    Body text is deliberately ignored: a chunk that merely *mentions* a
    technique in passing is not necessarily *about* it, whereas a technique in
    the title or a section breadcrumb is a strong topical signal.
    """
    hay = hit.title + " " + " ".join(hit.section_path)
    return bool(extract_terms(hay)["technique"])


def _keyword_overlap(query: str, hit: LexicalHit) -> float:
    """Fraction of query tokens that appear in the hit title/section (0..1)."""
    import re

    tokens = set(re.findall(r"[A-Za-z0-9][A-Za-z0-9_+\-\.]*", query.lower()))
    if not tokens:
        return 0.0
    hay = (hit.title + " " + " ".join(hit.section_path)).lower()
    matched = sum(1 for t in tokens if t in hay)
    return matched / len(tokens)


def rerank(
    hits: Iterable[LexicalHit],
    *,
    query: str,
    limit: int,
    per_document_cap: int = 2,
    platform: str | None = None,
    categories: list[str] | None = None,
    mode: str | None = None,
) -> list[LexicalHit]:
    """Score, filter, and diversify hits.

    Final score = base_score * authority * (1 + keyword_overlap + metadata +
    intent-mode bonus). Then apply a per-document cap so one page can't dominate
    the result set.

    ``mode`` biases the ranking toward the caller's intent without excluding
    anything: ``case_similarity`` favours hands-on writeups/case studies,
    ``technique`` favours pages whose title/heading name a known technique. Both
    are additive nudges layered on the same base evidence score, so a strong
    generic hit still surfaces — it is merely outranked by an equally strong
    on-intent one.
    """
    platform = (platform or "").lower() or None
    cat_filter = {c.lower() for c in (categories or [])}

    scored: list[LexicalHit] = []
    for h in hits:
        base = h.score
        auth = _authority_factor(h.authority)
        overlap = _keyword_overlap(query, h)
        bonus = 0.25 * overlap

        # Platform/category match bonus.
        hit_cats = h.metadata.get("categories", [])
        meta_cats = {str(c).lower() for c in hit_cats}
        if platform and platform in meta_cats:
            bonus += 0.15
        if cat_filter and meta_cats & cat_filter:
            bonus += 0.15

        # Intent-mode bonus.
        if mode == "case_similarity" and is_writeup_category(list(hit_cats)):
            bonus += _MODE_BONUS
        elif mode == "technique" and _is_technique_hit(h):
            bonus += _MODE_BONUS

        final = base * auth * (1.0 + bonus)
        scored.append(replace(h, score=final))

    # Sort by score descending.
    scored.sort(key=lambda x: x.score, reverse=True)

    # Source-diversity: per-document cap + cross-document near-dedup.
    out: list[LexicalHit] = []
    per_doc: dict[int, int] = {}
    for h in scored:
        used = per_doc.get(h.doc_id, 0)
        if used >= per_document_cap:
            continue
        # Skip a hit that is a near-duplicate of an already-selected one,
        # even if it comes from a different document.
        if any(shingle_overlap(h.text, kept.text) >= 0.9 for kept in out):
            continue
        per_doc[h.doc_id] = used + 1
        out.append(h)
        if len(out) >= limit:
            break
    return out
