"""HTML function-index ingestion adapter.

Some reference dumps publish one big HTML index that enumerates every item by
name, each entry linking to a small page of generated body text. The Windows
NT kernel function browser at ``likeagod.revers.engineering/nt/`` is the
motivating case: a single index page lists ~24k decompiled functions, and each
links to a page holding that function's Hex-Rays pseudocode plus the functions
that call it.

Such a site has no feed, sitemap or robots.txt, so the index is the *only*
discovery path. A generic website crawl of the same URL is worse than useless
here: bounded by ``max_files`` it captures an alphabetical slice of the index
and stops, silently. This adapter reads the index deliberately, so the cap is a
truncation guard on a known-selected set rather than a race against link order.

Order of operations, which matters:

1. Fetch the index and cache it (``index.html``).
2. Parse its entries, keep only names matching ``include_prefixes``, *then*
   apply ``max_files``. Filtering before capping is what lets a prefix list
   select the interesting thousands out of an alphabetically-ordered index;
   capping first would truncate the index order instead.
3. Fetch each selected page into ``pages/<sha256(url)>.html`` and stop there
   once cached, so an interrupted run resumes instead of restarting.

Each page becomes a document titled by its function name, whose body is the
pseudocode in a fenced ``c`` block followed by a ``## Referenced by`` list of
callers. Two deliberate omissions:

* No ``# FunctionName`` heading. The chunker copies the document title into
  every chunk's breadcrumb, and the name is in the pseudocode's own signature
  line, so a heading would only duplicate it.
* No section is emitted when the page reports no callers. The site's "No
  references." line is identical across thousands of leaf functions, so it
  would add a chunk that can never rank usefully; an absent section means the
  same thing.

The pseudocode is fenced rather than prose on purpose:
:func:`~blackbook.retrieval.chunking.chunk_markdown` never splits a fenced
block, so a function stays one intact code chunk instead of being cut mid
statement. The cost is that the handful of very large functions (the site's
own pages top out around 8 KB) become a single oversized chunk.
"""

from __future__ import annotations

import hashlib
import html as html_lib
import json
import logging
import re
import time
from pathlib import Path
from typing import Iterator
from urllib.parse import urljoin, urlparse

import httpx

from blackbook.config import SourceConfig
from blackbook.ingestion.base import ParsedDocument, SourceAdapter
from blackbook.retrieval.chunking import chunk_markdown
from blackbook.utils.paths import safe_join

log = logging.getLogger(__name__)

# Safety valve rather than a target: a function index can list tens of
# thousands of entries. When the cap bites it is logged, never silent.
_DEFAULT_MAX_FILES = 5000

# Report progress every N page fetches. A full source is thousands of requests
# at request_delay apiece, so a run that looks hung needs to say otherwise.
_PROGRESS_EVERY = 250

# ``<a href='96b8...html'>NtCreateFile</a>``. Attribute spacing, quote style and
# inner markup all vary between generators, so match loosely and clean after.
_ANCHOR_RE = re.compile(
    r"<a\s+[^>]*href=['\"](?P<href>[^'\"]+)['\"][^>]*>(?P<text>.*?)</a>",
    re.I | re.S,
)

# A link to another item page: a relative ``.html``/``.htm`` target. Absolute
# links are filtered separately so a link out to another site is never followed.
_PAGE_HREF_RE = re.compile(r"[^/?#]+\.html?$", re.I)

_H2_RE = re.compile(r"<h2[^>]*>(?P<title>.*?)</h2>", re.I | re.S)
_PRE_RE = re.compile(r"<pre[^>]*>(?P<body>.*?)</pre>", re.I | re.S)

# Tags in a *short text fragment* — a function name, an anchor's label — where
# there is no source code to damage, so a permissive pattern is safe.
_TAG_RE = re.compile(r"<[^>]+>")

# Tags inside a *pseudocode block*. Deliberately an allowlist of the markup a
# generator emits there (call links, inline formatting) and never a bare
# ``<[^>]+>``: decompiled C is full of literal ``<`` comparisons, and a
# permissive pattern swallows everything from ``if( a < b )`` through the next
# real tag — silently deleting code. Only multi-letter tag names are listed, so
# ``<b && c>`` in a comparison cannot be mistaken for a ``<b>`` element.
_CODE_TAG_RE = re.compile(
    r"</?(?:a|span|font|em|strong|code|tt|var|sub|sup|wbr)\b[^>]*>", re.I
)

# A line break inside the block becomes a newline rather than nothing: deleting
# it would splice two statements together and corrupt their identifiers.
_BR_RE = re.compile(r"<br\s*/?>", re.I)

# The camelCase module prefix at the head of a name: ``MmMapIoSpace`` -> ``Mm``.
# This site's names are prefixed by the component they belong to, which is a
# useful subsystem facet. Names in ALL_CAPS or with no capitalized run (macro
# constants, mostly) yield nothing and simply get no facet.
_PREFIX_RE = re.compile(r"[A-Z][a-z]+")

_WS_RE = re.compile(r"[ \t\r\f\v]+")


def _text(fragment: str) -> str:
    """Visible text of an HTML fragment, entities decoded and spacing folded."""
    return _WS_RE.sub(" ", html_lib.unescape(_TAG_RE.sub("", fragment))).strip()


class FunctionIndexAdapter(SourceAdapter):
    """Ingest a site that indexes items by name, one page per item."""

    def __init__(self, config: SourceConfig, raw_dir: str | None = None):
        super().__init__(config, raw_dir)
        if not config.url:
            raise ValueError(f"function-index source {config.id!r} requires a url")
        self.base_url = config.url.rstrip("/")
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"function-index source {config.id!r} has an invalid url")
        self._origin = (parsed.scheme, parsed.netloc)
        self._prefixes = tuple(p for p in config.include_prefixes if p)

    # -- paths -------------------------------------------------------------

    def _workdir(self) -> Path:
        assert self.raw_dir, "raw_dir is required"
        return Path(self.raw_dir) / self.source_id

    def _index_path(self) -> Path:
        return safe_join(self._workdir(), "index.html")

    def _items_path(self) -> Path:
        return safe_join(self._workdir(), "index_items.json")

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return safe_join(self._workdir() / "pages", f"{digest}.html")

    # -- fetching ----------------------------------------------------------

    def fetch(self, force: bool = False) -> None:
        workdir = self._workdir()
        (workdir / "pages").mkdir(parents=True, exist_ok=True)
        index_path = self._index_path()
        delay = max(0.0, float(self.config.request_delay))
        # Retries cover the transient handshake/read failures a long run
        # against a small server will meet; a page that fails for good is simply
        # left uncached and retried on the next ingest.
        transport = httpx.HTTPTransport(retries=2)
        with httpx.Client(
            timeout=30.0, follow_redirects=True, transport=transport
        ) as client:
            if force or not index_path.is_file():
                if not self._fetch_index(client, index_path):
                    return
                if delay:
                    time.sleep(delay)

            items = self._parse_index(index_path.read_text(encoding="utf-8"))
            if not items:
                log.error(
                    "%s: %s listed no item pages%s",
                    self.source_id,
                    self.base_url,
                    f" matching include_prefixes={list(self._prefixes)}"
                    if self._prefixes
                    else "",
                )
                return
            max_files = self.config.max_files or _DEFAULT_MAX_FILES
            if len(items) > max_files:
                log.warning(
                    "[%s] index lists %d items; capping at max_files=%d",
                    self.source_id,
                    len(items),
                    max_files,
                )
                items = items[:max_files]
            self._items_path().write_text(
                json.dumps(items, ensure_ascii=False), encoding="utf-8"
            )

            pending = [
                entry
                for entry in items
                if force or not self._cache_path(entry["url"]).is_file()
            ]
            if pending:
                log.info(
                    "[%s] fetching %d of %d indexed pages",
                    self.source_id,
                    len(pending),
                    len(items),
                )
            for done, entry in enumerate(pending, start=1):
                self._fetch_page(client, entry["url"])
                if delay:
                    time.sleep(delay)
                if done % _PROGRESS_EVERY == 0:
                    log.info(
                        "[%s] fetched %d/%d pages", self.source_id, done, len(pending)
                    )

    def _fetch_index(self, client: httpx.Client, index_path: Path) -> bool:
        try:
            response = client.get(self.base_url + "/")
            response.raise_for_status()
            index_path.write_bytes(response.content)
            return True
        except Exception as exc:
            log.error(
                "%s: failed to fetch %s/: %s", self.source_id, self.base_url, exc
            )
            return False

    def _fetch_page(self, client: httpx.Client, url: str) -> None:
        try:
            response = client.get(url)
            response.raise_for_status()
        except Exception as exc:
            log.warning("%s: failed to fetch %s: %s", self.source_id, url, exc)
            return
        if len(response.content) > self.config.max_document_bytes:
            log.warning("%s: skipping oversized page %s", self.source_id, url)
            return
        self._cache_path(url).write_bytes(response.content)

    # -- parsing -----------------------------------------------------------

    def _parse_index(self, text: str) -> list[dict[str, str]]:
        """Selected item entries, in the index's own order.

        Only same-origin ``.html`` links are kept: the index may link out to
        other sites, and following those would widen the corpus past the
        configured source.
        """
        entries: list[dict[str, str]] = []
        seen: set[str] = set()
        for match in _ANCHOR_RE.finditer(text):
            url = self._item_url(match.group("href"))
            if url is None or url in seen:
                continue
            name = _text(match.group("text"))
            if not name:
                continue
            if self._prefixes and not name.startswith(self._prefixes):
                continue
            seen.add(url)
            entries.append({"name": name, "url": url})
        return entries

    def _item_url(self, href: str) -> str | None:
        """Resolve an index href to a same-origin item page URL, else ``None``."""
        href = href.strip()
        if not href or href.startswith(("#", "mailto:", "javascript:")):
            return None
        if urlparse(href).scheme or href.startswith("//"):
            return None
        if not _PAGE_HREF_RE.search(href):
            return None
        url = urljoin(self.base_url + "/", href)
        if urlparse(url)[:2] != self._origin:
            return None
        return url

    def iter_documents(self) -> Iterator[ParsedDocument]:
        items_path = self._items_path()
        if not items_path.is_file():
            log.error("%s items not fetched; run fetch() first", self.source_id)
            return
        for entry in json.loads(items_path.read_text(encoding="utf-8")):
            url = entry["url"]
            path = self._cache_path(url)
            if not path.is_file():
                continue
            title, code, references = self._parse_page(
                path.read_text(encoding="utf-8", errors="replace"), entry["name"]
            )
            if not code:
                log.warning("%s: no pseudocode found in %s", self.source_id, url)
                continue
            categories = list(self.config.categories)
            subsystem = self._subsystem(title)
            if subsystem:
                categories.append(subsystem)
            body = self._render(code, references)
            yield ParsedDocument(
                external_id=url,
                title=title,
                url=url,
                path=str(path),
                categories=categories,
                text=body,
                metadata={
                    "source_type": "function_index",
                    "format": "html",
                    "subsystem": subsystem,
                    "reference_count": len(references),
                },
                chunks=chunk_markdown(body, title_path=categories + [title]),
            )

    def _parse_page(
        self, text: str, fallback_name: str
    ) -> tuple[str, str, list[str]]:
        """Split an item page into ``(title, pseudocode, caller names)``."""
        title = fallback_name
        heading = _H2_RE.search(text)
        if heading:
            found = _text(heading.group("title"))
            if found:
                title = found
        code = ""
        block = _PRE_RE.search(text)
        if block:
            # Inline ``<a>`` tags wrap calls to other functions; the name is
            # the content, so dropping the tag keeps the call readable. A
            # line break is a newline for the same reason — dropping it would
            # splice the statements on either side of it together.
            body = _BR_RE.sub("\n", block.group("body"))
            code = html_lib.unescape(_CODE_TAG_RE.sub("", body)).strip("\n")
        return title, code, self._parse_references(text)

    def _parse_references(self, text: str) -> list[str]:
        """Names of the callers listed outside the pseudocode block.

        The ``<pre>`` is removed first because its inline call links are
        same-shaped anchors; what remains is the "Referenced by" list.
        """
        names: list[str] = []
        seen: set[str] = set()
        for match in _ANCHOR_RE.finditer(_PRE_RE.sub("", text)):
            if self._item_url(match.group("href")) is None:
                continue
            name = _text(match.group("text"))
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)
        return names

    def _render(self, code: str, references: list[str]) -> str:
        """Markdown body: the pseudocode, then the callers that were listed.

        The caller list follows its heading with no blank line between them:
        the chunker treats a heading as the start of the following chunk, so a
        blank line there would emit the heading alone as a 16-character chunk
        for every function in the corpus.
        """
        parts = [f"```c\n{code}\n```"]
        if references:
            callers = "\n".join(f"- {name}" for name in references)
            parts.append(f"## Referenced by\n{callers}")
        return "\n\n".join(parts) + "\n"

    @staticmethod
    def _subsystem(name: str) -> str | None:
        """Lowercased module prefix, e.g. ``MmMapIoSpace`` -> ``mm``."""
        match = _PREFIX_RE.match(name)
        return match.group(0).lower() if match else None
