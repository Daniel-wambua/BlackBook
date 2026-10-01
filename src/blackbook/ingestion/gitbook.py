"""GitBook space ingestion adapter.

A GitBook space renders its pages client-side, so crawling the published HTML
yields only sidebar and table-of-contents chrome — never the article text.
GitBook instead publishes two machine-readable companions, and this adapter
uses both:

* ``<base>/llms.txt`` is the space's full page index: one
  ``- [Title](<url>.md)`` line per page. Those URLs are the *published* slugs,
  which routinely differ from any backing repository's file paths, so the index
  is the authoritative source of both the page list and each page's canonical
  link.
* Appending ``.md`` to a page URL returns the page body as clean markdown,
  headings and fenced code intact — exactly what the chunker wants.

Pages are cached under the source's raw directory, so a re-ingestion is cheap
and ``fetch(force=False)`` is a no-op when the index is already present.

Three kinds of GitBook chrome are removed before indexing, because they are the
publisher's boilerplate rather than author content and would otherwise be
repeated into every chunk: the leading "documentation index" blockquote GitBook
prepends to every page, the trailing "Agent Instructions" section it appends,
and its ``{% hint %}`` / ``{% embed %}`` / ``{% tabs %}`` template markers (the
tag goes, the text inside it stays). Pages the index still lists after they
were moved or deleted — served as HTTP 200 with a "Page Not Found" body — are
skipped rather than indexed as bogus documents.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

import httpx

from blackbook.config import SourceConfig
from blackbook.ingestion.base import ParsedDocument, SourceAdapter
from blackbook.retrieval.chunking import chunk_markdown
from blackbook.utils.paths import safe_join

log = logging.getLogger(__name__)

# GitBook spaces can be large (600+ pages); the cap is a safety valve, not a
# target. When it bites, it is logged rather than applied silently.
_DEFAULT_MAX_FILES = 2000

# ``- [Page Title](https://host/path/page.md)`` — one line per page in
# llms.txt. Any trailing ": description" is deliberately left unmatched.
_INDEX_LINE_RE = re.compile(
    r"^\s*[-*]\s+\[(?P<title>[^\]]+)\]\((?P<url>https?://[^)\s]+\.md)\)", re.M
)

# GitBook prepends this blockquote to every markdown page it serves.
_INDEX_BOILERPLATE_RE = re.compile(
    r"^>\s*For the complete documentation index,.*?llms\.txt.*?$", re.M
)

# GitBook appends a site-level "Agent Instructions" section (how to query the
# docs as an agent) to every page of a space that enables it.
_AGENT_SECTION_RE = re.compile(r"^#{1,2}[ \t]+Agent Instructions[ \t]*$", re.M)

# GitBook template syntax: ``{% hint style="info" %}`` … ``{% endhint %}``,
# self-closing ``{% embed url="…" %}``, ``{% tabs %}``/``{% tab %}``.
_GITBOOK_TAG_RE = re.compile(r"\{%-?[ \t]*[a-zA-Z][^%]*?-?%\}")

# Raw HTML GitBook emits around figures. Images carry no indexable prose;
# the block wrappers are unwrapped so their text (captions, notes) survives.
_HTML_IMG_RE = re.compile(r"<img\b[^>]*>", re.I)
_HTML_WRAPPER_RE = re.compile(
    r"</?(?:figure|figcaption|caption|div|span|center|p|table|thead|tbody|tr|td|th)\b[^>]*>",
    re.I,
)
_HTML_BR_RE = re.compile(r"<br\s*/?>", re.I)

# Site-relative links (``](/recon/domain-enum)``) are made absolute so the
# citation resolves wherever it is read.
_REL_LINK_RE = re.compile(r"\]\(<?(?P<path>/[^)>\s]*)>?\)")

_BLANK_RUN_RE = re.compile(r"\n{3,}")

# A soft-404: the page is gone but GitBook still answers 200.
_NOT_FOUND_MARKERS = (
    "# Page Not Found",
    "does not exist. This page may have been moved",
)


class GitbookAdapter(SourceAdapter):
    """Ingest a GitBook space via its ``llms.txt`` index and page markdown."""

    def __init__(self, config: SourceConfig, raw_dir: str | None = None):
        super().__init__(config, raw_dir)
        if not config.url:
            raise ValueError(f"gitbook source {config.id!r} requires a url")
        self.base_url = config.url.rstrip("/")
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"gitbook source {config.id!r} has an invalid url")
        self._origin = (parsed.scheme, parsed.netloc)
        self._base_path = parsed.path.rstrip("/")

    # -- paths -------------------------------------------------------------

    def _workdir(self) -> Path:
        assert self.raw_dir, "raw_dir is required"
        return Path(self.raw_dir) / self.source_id

    def _index_path(self) -> Path:
        return safe_join(self._workdir(), "llms.txt")

    def _pages_index_path(self) -> Path:
        return safe_join(self._workdir(), "index_pages.json")

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return safe_join(self._workdir() / "pages", f"{digest}.md")

    # -- fetching ----------------------------------------------------------

    def fetch(self, force: bool = False) -> None:
        workdir = self._workdir()
        (workdir / "pages").mkdir(parents=True, exist_ok=True)
        index_path = self._index_path()
        delay = max(0.0, float(self.config.request_delay))
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            if force or not index_path.is_file():
                try:
                    response = client.get(f"{self.base_url}/llms.txt")
                    response.raise_for_status()
                    index_path.write_bytes(response.content)
                except Exception as exc:
                    log.error(
                        "%s: failed to fetch %s/llms.txt: %s",
                        self.source_id,
                        self.base_url,
                        exc,
                    )
                    return
                if delay:
                    time.sleep(delay)

            pages = self._parse_index(index_path.read_text(encoding="utf-8"))
            if not pages:
                log.error(
                    "%s: llms.txt at %s/llms.txt listed no pages",
                    self.source_id,
                    self.base_url,
                )
                return
            max_files = self.config.max_files or _DEFAULT_MAX_FILES
            if len(pages) > max_files:
                log.warning(
                    "[%s] llms.txt lists %d pages; capping at max_files=%d",
                    self.source_id,
                    len(pages),
                    max_files,
                )
                pages = pages[:max_files]
            self._pages_index_path().write_text(
                json.dumps(pages, ensure_ascii=False), encoding="utf-8"
            )

            for entry in pages:
                url = entry["url"]
                cache_path = self._cache_path(url)
                if cache_path.is_file() and not force:
                    continue
                try:
                    response = client.get(url)
                    response.raise_for_status()
                    if len(response.content) > self.config.max_document_bytes:
                        log.warning(
                            "%s: skipping oversized page %s", self.source_id, url
                        )
                        continue
                    cache_path.write_bytes(response.content)
                except Exception as exc:
                    log.warning("%s: failed to fetch %s: %s", self.source_id, url, exc)
                if delay:
                    time.sleep(delay)

    def _parse_index(self, text: str) -> list[dict[str, str]]:
        """Page entries from llms.txt, in the book's own reading order.

        Only links inside the configured space are kept: llms.txt can carry
        links out to other spaces, and crawling those would silently widen the
        corpus beyond the configured source.
        """
        entries: list[dict[str, str]] = []
        seen: set[str] = set()
        for match in _INDEX_LINE_RE.finditer(text):
            url = match.group("url")
            parsed = urlparse(url)
            if (parsed.scheme, parsed.netloc) != self._origin:
                continue
            if not url.startswith(self.base_url + "/"):
                continue
            if url in seen:
                continue
            seen.add(url)
            entries.append({"title": match.group("title").strip(), "url": url})
        return entries

    # -- parsing -----------------------------------------------------------

    def iter_documents(self) -> Iterator[ParsedDocument]:
        index_path = self._pages_index_path()
        if not index_path.is_file():
            log.error("%s pages not fetched; run fetch() first", self.source_id)
            return
        for entry in json.loads(index_path.read_text(encoding="utf-8")):
            url = entry["url"]
            path = self._cache_path(url)
            if not path.is_file():
                continue
            raw = path.read_text(encoding="utf-8", errors="replace")
            if self._is_not_found(raw):
                log.info("%s: skipping soft-404 %s", self.source_id, url)
                continue
            body = self._clean_markdown(raw)
            if not body.strip():
                continue
            title = self._extract_title(body) or entry.get("title") or url
            categories = list(self.config.categories) + self._categories_from_url(url)
            yield ParsedDocument(
                external_id=url,
                title=title,
                url=self._source_url(url),
                path=str(path),
                categories=categories,
                text=body,
                metadata={
                    "source_type": "gitbook",
                    "format": "markdown",
                    "markdown_url": url,
                },
                chunks=chunk_markdown(body, title_path=categories + [title]),
            )

    @staticmethod
    def _is_not_found(body: str) -> bool:
        head = body.lstrip()
        if head.startswith(_NOT_FOUND_MARKERS[0]):
            return True
        return _NOT_FOUND_MARKERS[1] in head[:600]

    def _clean_markdown(self, body: str) -> str:
        """Strip GitBook's per-page chrome, keeping the author's text."""
        body = _INDEX_BOILERPLATE_RE.sub("", body)
        agent = _AGENT_SECTION_RE.search(body)
        if agent:
            body = body[: agent.start()]
        body = _GITBOOK_TAG_RE.sub("", body)
        body = _HTML_BR_RE.sub("\n", body)
        body = _HTML_IMG_RE.sub("", body)
        body = _HTML_WRAPPER_RE.sub("", body)
        body = _REL_LINK_RE.sub(lambda m: f"]({self.base_url}{m.group('path')})", body)
        return _BLANK_RUN_RE.sub("\n\n", body).strip()

    @staticmethod
    def _extract_title(body: str) -> str | None:
        for line in body.splitlines():
            s = line.strip()
            if s.startswith("# "):
                return s[2:].strip()
            if s:  # first non-empty non-heading line -> stop looking
                break
        return None

    @staticmethod
    def _source_url(md_url: str) -> str:
        """The human-facing page URL: the markdown URL minus ``.md``."""
        return md_url[:-3] if md_url.lower().endswith(".md") else md_url

    def _categories_from_url(self, url: str) -> list[str]:
        """Category breadcrumb from the page's URL path.

        The final segment is the page's own slug — it becomes the document
        title, so it is dropped here, exactly as a repository adapter drops a
        filename. What remains is the space's grouping (``training-and-career``),
        which is the load-bearing part for retrieval.
        """
        path = urlparse(url).path
        if self._base_path and path.startswith(self._base_path + "/"):
            path = path[len(self._base_path) + 1 :]
        elif self._base_path and path == self._base_path:
            path = ""
        if path.lower().endswith(".md"):
            path = path[:-3]
        parts = [p for p in path.strip("/").split("/") if p]
        parts = parts[:-1]
        return [
            self._humanize(p)
            for p in parts
            if p.lower() not in {"readme", "index"}
        ]

    @staticmethod
    def _humanize(slug: str) -> str:
        return slug.replace("-", " ").replace("_", " ").strip().title()
