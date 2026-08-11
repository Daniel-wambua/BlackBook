"""Near-duplicate detection for chunks.

Exact dedup (identical content hash) happens in the pipeline. This module adds
*near*-duplicate detection so that two chunks that are essentially the same
text — common when HackTricks material and a PDF overlap — are recognized even
if whitespace/casing/punctuation differ slightly.

Two tiers:

* ``normalized_hash`` — exact match after aggressive normalization (cheap,
  catches re-formatted copies).
* ``shingle_overlap`` — Jaccard similarity over word shingles, for graded
  "is this basically the same content?" checks.
"""

from __future__ import annotations

import hashlib
import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    """Aggressively normalize text: lowercase, keep only alnum tokens."""
    return " ".join(_TOKEN_RE.findall(text.lower()))


def normalized_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def shingles(text: str, n: int = 3) -> set[str]:
    """Word n-gram shingles of the normalized text."""
    tokens = normalize(text).split()
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def shingle_overlap(a: str, b: str, n: int = 3) -> float:
    """Jaccard similarity (0..1) over word shingles of two texts."""
    sa, sb = shingles(a, n), shingles(b, n)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def is_near_duplicate(a: str, b: str, *, threshold: float = 0.85) -> bool:
    """True if two texts are near-duplicates.

    Fast path: identical normalized hash. Otherwise shingle similarity.
    """
    if normalized_hash(a) == normalized_hash(b):
        return True
    return shingle_overlap(a, b) >= threshold
