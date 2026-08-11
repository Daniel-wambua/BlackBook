"""0xdf (https://0xdf.gitlab.io/) ingestion adapter.

0xdf publishes detailed HTB/CTF writeups as a Jekyll blog. Each post carries a
rich, consistent structure that we exploit for structured metadata:

* ``og:title`` (e.g. "HTB: Helix") -> machine name + source kind
* ``article:published_time`` -> date
* ``description`` -> the author's own summary of the attack chain
* ``.htb-card`` -> difficulty, OS, release/retire dates, creator
* H2/H3 headings -> the attack-chain narrative ("Recon", "Shell as X", ...)

Fields that cannot be extracted confidently are left ``None`` (nullable
metadata) and any inferred value is marked ``inferred=True`` in the metadata,
per BlackBook's provenance rules.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Iterator

import httpx
from bs4 import BeautifulSoup

from blackbook.config import SourceConfig
from blackbook.ingestion.base import ParsedDocument, SourceAdapter
from blackbook.knowledge.vocab import extract_signals
from blackbook.retrieval.chunking import RawChunk
from blackbook.utils.paths import safe_join

log = logging.getLogger(__name__)

_POST_URL_RE = re.compile(r"^/(\d{4})/(\d{2})/(\d{2})/([^/]+?)\.html$")
_TITLE_PREFIX_RE = re.compile(r"^(HTB|PG|OSCP|VulnHub|TryHackMe|HackTheBox)\s*[:\-]\s*", re.I)


class ZeroDFAdapter(SourceAdapter):
    """Ingests 0xdf writeups."""

    source_id = "0xdf"

    def __init__(self, config: SourceConfig, raw_dir: str | None = None):
        super().__init__(config, raw_dir)
        self._client: httpx.Client | None = None
        self.base_url = (config.url or "https://0xdf.gitlab.io/").rstrip("/")

    def _workdir(self) -> Path:
        assert self.raw_dir, "raw_dir is required"
        return Path(self.raw_dir) / "0xdf"

    # -- fetching ----------------------------------------------------------

    def fetch(self, force: bool = False) -> None:
        """Download the index and cache each post page locally.

        We cache pages under ``raw/0xdf/pages/`` and only re-fetch a post when
        it is new or ``force`` is set, so repeat ingestion is cheap and polite.
        """
        workdir = self._workdir()
        pages_dir = workdir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        self._client = httpx.Client(timeout=30.0, follow_redirects=True)

        index_html = self._get(self.base_url + "/")
        post_urls = self._extract_post_urls(index_html)
        max_files = self.config.max_files
        if max_files is not None:
            post_urls = post_urls[:max_files]
        log.info("0xdf: discovered %d posts", len(post_urls))

        # Record the URL list so iter_documents can walk it.
        (workdir / "index_urls.json").write_text("\n".join(post_urls))

        for url_path in post_urls:
            cache_path = self._cache_path(url_path)
            if cache_path.is_file() and not force:
                continue
            try:
                html = self._get(self.base_url + url_path)
                cache_path.write_text(html, encoding="utf-8")
            except Exception as e:
                log.warning("0xdf: failed to fetch %s: %s", url_path, e)

    def _get(self, url: str) -> str:
        assert self._client is not None
        r = self._client.get(url)
        r.raise_for_status()
        return r.text

    @staticmethod
    def _extract_post_urls(index_html: str) -> list[str]:
        soup = BeautifulSoup(index_html, "lxml")
        seen: dict[str, None] = {}
        for a in soup.find_all("a", href=True):
            href = a["href"]
            m = _POST_URL_RE.match(href)
            if m:
                seen[href] = None  # dedup, preserve order
        return list(seen.keys())

    def _cache_path(self, url_path: str) -> Path:
        # url_path like /2026/08/08/htb-helix.html -> pages/htb-helix.html
        slug = url_path.rstrip("/").rsplit("/", 1)[-1]
        return safe_join(self._workdir() / "pages", slug)

    # -- parsing -----------------------------------------------------------

    def iter_documents(self) -> Iterator[ParsedDocument]:
        workdir = self._workdir()
        pages_dir = workdir / "pages"
        if not pages_dir.is_dir():
            log.error("0xdf pages not fetched; run fetch() first")
            return
        for path in sorted(pages_dir.glob("*.html")):
            try:
                doc = self._parse_post(path)
                if doc is not None:
                    yield doc
            except Exception as e:
                log.warning("0xdf: failed to parse %s: %s", path, e)

    def _parse_post(self, path: Path) -> ParsedDocument | None:
        html = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "lxml")

        title_tag = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "name"})
        raw_title = title_tag["content"].strip() if title_tag and title_tag.get("content") else path.stem

        # Machine name and source kind from "HTB: Helix" style titles.
        machine_name, kind = self._split_title(raw_title)

        # Published date
        date = None
        date_tag = soup.find("meta", property="article:published_time")
        if date_tag and date_tag.get("content"):
            date = date_tag["content"].split("T")[0]

        # Author summary (og:description) — a high-signal abstract.
        summary = None
        desc_tag = soup.find("meta", property="og:description") or soup.find("meta", attrs={"name": "description"})
        if desc_tag and desc_tag.get("content"):
            summary = desc_tag["content"].strip()

        # Canonical URL
        url = None
        url_tag = soup.find("meta", property="og:url")
        if url_tag and url_tag.get("content"):
            url = url_tag["content"].strip()

        # Structured card metadata (difficulty, OS, dates, creator).
        card = self._parse_card(soup)

        # Body + section narrative.
        body = soup.select_one("#postBody") or soup.select_one(".post-content")
        chunks, sections = self._body_to_chunks(body, base_path=[raw_title]) if body else ([], [])

        # Assemble full text for hashing / change detection.
        text_parts = [p.text for p in chunks]
        full_text = "\n\n".join(text_parts)

        # Services / technologies: light, confidence-tagged heuristic scan.
        services, techniques, tools = self._extract_signals(full_text)

        metadata: dict[str, Any] = {
            "machine_name": machine_name,
            "kind": kind,  # htb | pg | oscp | ...
            "date": date,
            "summary": summary,
            "difficulty": card.get("difficulty"),
            "os": card.get("os"),
            "release_date": card.get("release_date"),
            "retire_date": card.get("retire_date"),
            "creator": card.get("creator"),
            "sections": sections,
            "services": services,
            "techniques": techniques,
            "tools": tools,
            "metadata_inferred": {
                "services": True,
                "techniques": True,
                "tools": True,
                "machine_name": kind == "unknown",
            },
        }

        categories = [c for c in [kind, card.get("os"), card.get("difficulty")] if c]
        return ParsedDocument(
            external_id=path.stem,
            title=raw_title,
            url=url,
            path=str(path),
            categories=categories,
            text=full_text,
            metadata=metadata,
            chunks=chunks,
        )

    @staticmethod
    def _split_title(raw_title: str) -> tuple[str | None, str]:
        m = _TITLE_PREFIX_RE.match(raw_title)
        if m:
            kind = m.group(1).lower()
            if kind == "hackthebox":
                kind = "htb"
            name = raw_title[m.end():].strip()
            return (name or None), kind
        return None, "unknown"

    def _parse_card(self, soup: BeautifulSoup) -> dict[str, str | None]:
        card: dict[str, str | None] = {}
        node = soup.select_one(".htb-card")
        if not node:
            return card
        # The card is a flat label/value sequence in document order.
        texts = [t.strip() for t in node.stripped_strings]
        # Build a label->value lookup from known labels.
        label_map = {
            "OS": "os",
            "Release Date": "release_date",
            "Retire Date": "retire_date",
            "Creator": "creator",
        }
        for i, t in enumerate(texts):
            if t in label_map and i + 1 < len(texts):
                card[label_map[t]] = texts[i + 1]
        # Difficulty appears as a bare token (Easy/Medium/Hard/Insane) near the top.
        for t in texts[:6]:
            if t.lower() in {"easy", "medium", "hard", "insane"}:
                card["difficulty"] = t.capitalize()
                break
        return card

    def _body_to_chunks(self, body, base_path: list[str]) -> tuple[list[RawChunk], list[str]]:
        """Convert the post body to chunks split on H2/H3 section boundaries,
        preserving code blocks, and return the section narrative list."""
        chunks: list[RawChunk] = []
        sections: list[str] = []
        section_stack: list[tuple[int, str]] = []
        buf: list[str] = []
        ordinal = 0

        def current_path() -> list[str]:
            return base_path + [t for (_l, t) in section_stack]

        def flush() -> None:
            nonlocal ordinal
            text = "\n".join(buf).strip()
            buf.clear()
            if not text:
                return
            chunks.append(
                RawChunk(text=text, section_path=current_path(), ordinal=ordinal, kind="text")
            )
            ordinal += 1

        for el in body.descendants:
            if getattr(el, "name", None) in ("h2", "h3"):
                flush()
                level = int(el.name[1])
                title = el.get_text(strip=True)
                sections.append(title)
                while section_stack and section_stack[-1][0] >= level:
                    section_stack.pop()
                section_stack.append((level, title))
            elif getattr(el, "name", None) == "pre":
                # Code block: keep verbatim, don't also capture inner text.
                code = el.get_text()
                if code.strip():
                    buf.append("```\n" + code.strip() + "\n```")
            elif getattr(el, "name", None) is None:
                # NavigableString; only add if it's a direct text node and the
                # parent isn't a <pre> or <code> (already captured).
                parent = getattr(el, "parent", None)
                pname = getattr(parent, "name", None)
                if pname not in ("pre", "code", "script", "style"):
                    text = str(el).strip()
                    if text:
                        buf.append(text)
        flush()
        return chunks, sections

    @staticmethod
    def _extract_signals(text: str) -> tuple[list[str], list[str], list[str]]:
        """Lightweight, confidence-tagged extraction of services / techniques /
        tools mentioned in a writeup. These are *inferred* (marked as such).

        The controlled vocabulary lives in :mod:`blackbook.knowledge.vocab` so
        the graph builder extracts the same terms; this is a thin delegate.
        """
        return extract_signals(text)

    def current_version(self) -> str | None:
        idx = self._workdir() / "index_urls.json"
        if idx.is_file():
            n = len(idx.read_text().splitlines())
            return f"{n} posts"
        return None
