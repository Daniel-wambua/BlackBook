"""Structural analysis of PDF text.

Uses pypdf's ``visitor_text`` hook to capture per-text-run font name and size,
which lets us detect headings (larger font), code blocks (monospaced font), and
emphasis with reasonable confidence — without a heavyweight layout engine.

Everything here is *heuristic* and is therefore tagged ``inferred`` by the
caller; we only claim structure when the signal is strong enough.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Common monospaced font substrings (case-insensitive).
_MONO_HINTS = ("mono", "courier", "consol", "menlo", "dejavu sans mono", "typewriter", "fixedsys", "code")

# A numbered-section heading like "3.2 Kerberoasting" or "4. Services".
_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)[.\s]\s+(\S.{1,90})$")


@dataclass
class TextRun:
    text: str
    font: str = ""
    size: float = 0.0


@dataclass
class PageContent:
    """Text of one PDF page plus the detected structural blocks."""

    page: int
    text: str
    runs: list[TextRun] = field(default_factory=list)
    # Detected structure (all inferred)
    headings: list[str] = field(default_factory=list)
    code_blocks: list[str] = field(default_factory=list)


def is_monospace(font: str) -> bool:
    f = font.lower()
    return any(h in f for h in _MONO_HINTS)


def collect_page_runs(page) -> tuple[str, list[TextRun]]:
    """Extract plain text and per-run font/size using pypdf visitor hooks.

    Returns the page's plain text and a list of :class:`TextRun`.
    """
    runs: list[TextRun] = []

    def visitor(text, cm, tm, font_dict, font_size):
        if not text or not text.strip():
            return
        font = ""
        try:
            if font_dict:
                base = font_dict.get("/BaseFont")
                if base:
                    font = str(base).lstrip("/")
        except Exception:
            font = ""
        runs.append(TextRun(text=text, font=font, size=float(font_size or 0.0)))

    plain = page.extract_text(visitor_text=visitor) or ""
    return plain, runs


def _median_size(runs: list[TextRun]) -> float:
    sizes = sorted(r.size for r in runs if r.size > 0)
    if not sizes:
        return 0.0
    return sizes[len(sizes) // 2]


def detect_headings(runs: list[TextRun], body_size: float) -> list[str]:
    """Detect headings on a page from font size and numbering patterns.

    A run is a heading candidate if its font is notably larger than the body
    median, or it matches a numbered-section pattern. Returns the heading text
    lines (deduped, order-preserved).
    """
    headings: list[str] = []
    seen: set[str] = set()
    # 1) Font-size-based headings
    if body_size > 0:
        threshold = body_size * 1.15
        cur: list[str] = []
        for r in runs:
            txt = r.text.strip()
            if not txt:
                continue
            if r.size >= threshold and len(txt) <= 100:
                cur.append(txt)
            else:
                if cur:
                    h = " ".join(cur).strip()
                    if h and h not in seen:
                        seen.add(h)
                        headings.append(h)
                    cur = []
        if cur:
            h = " ".join(cur).strip()
            if h and h not in seen:
                seen.add(h)
                headings.append(h)
    # 2) Numbered-section headings (even at body size)
    for r in runs:
        for line in r.text.splitlines():
            line = line.strip()
            m = _NUMBERED_HEADING_RE.match(line)
            if m and line not in seen:
                seen.add(line)
                headings.append(line)
    return headings


def detect_code_blocks(runs: list[TextRun]) -> list[str]:
    """Group consecutive monospaced runs into code blocks."""
    blocks: list[str] = []
    cur: list[str] = []
    for r in runs:
        if r.font and is_monospace(r.font):
            cur.append(r.text)
        else:
            if cur:
                block = "".join(cur).strip()
                if len(block) > 2:
                    blocks.append(block)
                cur = []
    if cur:
        block = "".join(cur).strip()
        if len(block) > 2:
            blocks.append(block)
    return blocks


def analyze_page(page, page_number: int) -> PageContent:
    """Full structural analysis of a single PDF page."""
    try:
        plain, runs = collect_page_runs(page)
    except Exception:
        # A corrupt page yields empty content, not a crash.
        return PageContent(page=page_number, text="")
    body = _median_size(runs)
    headings = detect_headings(runs, body)
    code_blocks = detect_code_blocks(runs)
    return PageContent(
        page=page_number,
        text=plain,
        runs=runs,
        headings=headings,
        code_blocks=code_blocks,
    )
