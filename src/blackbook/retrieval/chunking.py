"""Semantic chunking.

Documents are split on meaningful boundaries — markdown headings, fenced code
blocks, and paragraph breaks — never blind character counts. Every chunk
retains a ``section_path`` (the heading breadcrumb leading to it), so a
retrieved chunk can be traced back to its exact location in the source
document and cited precisely.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Rough tokens-per-word heuristic; good enough for token budgeting.
_TOKENS_PER_WORD = 1.33

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^```")


def estimate_tokens(text: str) -> int:
    words = len(text.split())
    return max(1, int(words * _TOKENS_PER_WORD))


@dataclass
class RawChunk:
    """A chunk produced by the chunker, before persistence."""

    text: str
    section_path: list[str] = field(default_factory=list)
    ordinal: int = 0
    page: int | None = None
    kind: str = "text"  # text | code | table | heading


def _flush(
    out: list[RawChunk],
    buf: list[str],
    section_path: list[str],
    page: int | None,
    kind: str,
) -> None:
    text = "\n".join(buf).strip()
    if not text:
        return
    out.append(
        RawChunk(text=text, section_path=list(section_path), ordinal=len(out), page=page, kind=kind)
    )


def _split_long(buf: list[str], max_tokens: int) -> list[list[str]]:
    """Split a paragraph buffer that exceeds ``max_tokens`` into smaller ones
    on blank-line boundaries, keeping order. Any group that is still too large
    (a single long paragraph with no blank lines) is hard-split by word count.
    """
    groups: list[list[str]] = []
    cur: list[str] = []
    cur_tokens = 0
    for line in buf:
        t = estimate_tokens(line)
        if cur and cur_tokens + t > max_tokens and not line.strip():
            # blank-line boundary
            groups.append(cur)
            cur = []
            cur_tokens = 0
            continue
        cur.append(line)
        cur_tokens += t
    if cur:
        groups.append(cur)

    # Hard-split any group that still exceeds the budget.
    out: list[list[str]] = []
    max_words = max(1, int(max_tokens / _TOKENS_PER_WORD))
    for grp in groups:
        if estimate_tokens("\n".join(grp)) <= max_tokens:
            out.append(grp)
            continue
        words = "\n".join(grp).split()
        for i in range(0, len(words), max_words):
            out.append([" ".join(words[i : i + max_words])])
    return out


def chunk_markdown(
    text: str,
    *,
    title_path: list[str] | None = None,
    max_tokens: int = 400,
) -> list[RawChunk]:
    """Chunk a markdown document on heading/code/paragraph boundaries.

    ``title_path`` seeds the breadcrumb (e.g. the document title) so chunks
    that appear before the first heading still carry provenance.
    """
    chunks: list[RawChunk] = []
    # heading stack: list[(level, title)] -> section_path derived from it
    section_stack: list[tuple[int, str]] = []
    base_path = list(title_path or [])

    buf: list[str] = []
    buf_kind = "text"
    in_fence = False
    fence_lang = ""

    def current_section() -> list[str]:
        return base_path + [t for (_lvl, t) in section_stack]

    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Fenced code block handling (toggled on ``` even mid-stream).
        if _FENCE_RE.match(stripped):
            if not in_fence:
                # close any open text buffer, then start the code buffer
                _flush(chunks, buf, current_section(), None, buf_kind)
                buf = []
                in_fence = True
                fence_lang = stripped[3:].strip()
                buf.append(line)
                buf_kind = "code"
            else:
                # closing fence
                buf.append(line)
                _flush(chunks, buf, current_section(), None, "code")
                buf = []
                in_fence = False
                buf_kind = "text"
                fence_lang = ""
            i += 1
            continue

        if in_fence:
            buf.append(line)
            i += 1
            continue

        # Heading
        m = _HEADING_RE.match(line)
        if m:
            _flush(chunks, buf, current_section(), None, buf_kind)
            buf = []
            level = len(m.group(1))
            title = m.group(2).strip()
            # pop stack to this level
            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()
            section_stack.append((level, title))
            # The heading itself is captured as the start of the next chunk.
            buf.append(line)
            buf_kind = "text"
            i += 1
            continue

        # Blank line -> paragraph boundary
        if not stripped:
            if buf:
                # Emit buffer (splitting if it's oversized)
                text_now = "\n".join(buf)
                if estimate_tokens(text_now) > max_tokens:
                    for grp in _split_long(buf, max_tokens):
                        _flush(chunks, grp, current_section(), None, buf_kind)
                else:
                    _flush(chunks, buf, current_section(), None, buf_kind)
                buf = []
            i += 1
            continue

        buf.append(line)
        i += 1

    # flush trailing buffer
    if buf:
        text_now = "\n".join(buf)
        if estimate_tokens(text_now) > max_tokens:
            for grp in _split_long(buf, max_tokens):
                _flush(chunks, grp, current_section(), None, buf_kind)
        else:
            _flush(chunks, buf, current_section(), None, buf_kind)

    return chunks


def chunk_plain_pages(
    pages: list[str],
    *,
    title_path: list[str] | None = None,
    max_tokens: int = 400,
) -> list[RawChunk]:
    """Chunk a list of per-page plain-text strings (e.g. from a PDF).

    Pages are kept intact where possible and split on paragraph boundaries
    when they exceed ``max_tokens``. Each chunk records its page number.
    """
    chunks: list[RawChunk] = []
    base_path = list(title_path or [])
    for page_num, page_text in enumerate(pages, start=1):
        paras = re.split(r"\n\s*\n", page_text)
        buf: list[str] = []
        cur_tokens = 0
        for para in paras:
            para = para.strip()
            if not para:
                continue
            t = estimate_tokens(para)
            if buf and cur_tokens + t > max_tokens:
                _flush(chunks, buf, base_path, page_num, "text")
                buf = []
                cur_tokens = 0
            buf.append(para)
            cur_tokens += t
        _flush(chunks, buf, base_path, page_num, "text")
    return chunks


def chunk_structured_pages(
    pages: list,  # list of PageContent-like objects with .text/.headings/.code_blocks/.page
    *,
    title_path: list[str] | None = None,
    max_tokens: int = 400,
) -> list[RawChunk]:
    """Chunk structurally-analyzed PDF pages.

    Headings update the running ``section_path`` breadcrumb, code blocks are
    emitted as intact ``code`` chunks, and body text is split on paragraph
    boundaries. Every chunk records its page number so citations resolve to a
    precise ``(page, section)``.
    """
    chunks: list[RawChunk] = []
    base_path = list(title_path or [])
    current_section: list[str] = []  # most recent heading, persists across pages

    def section() -> list[str]:
        return base_path + current_section

    for page in pages:
        page_num = page.page
        # Advance the section from this page's headings (first heading wins).
        if getattr(page, "headings", None):
            # Use the deepest/first heading as the current section context.
            current_section = [page.headings[0]]

        code_set = set(getattr(page, "code_blocks", []) or [])
        body = page.text or ""
        paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]

        buf: list[str] = []
        cur_tokens = 0
        for para in paras:
            t = estimate_tokens(para)
            if buf and cur_tokens + t > max_tokens:
                _flush(chunks, buf, section(), page_num, "text")
                buf = []
                cur_tokens = 0
            buf.append(para)
            cur_tokens += t
        _flush(chunks, buf, section(), page_num, "text")

        # Emit code blocks as intact chunks tagged to the current section.
        for block in code_set:
            if not block:
                continue
            _flush(chunks, ["```\n" + block + "\n```"], section(), page_num, "code")
    return chunks
