"""Generic same-origin website ingestion adapter.

The adapter is intentionally conservative: it follows links from one configured
entry point, stays on the configured origin, caches pages locally, and applies
an explicit file limit. It is suitable for documentation and disclosure sites,
not arbitrary crawling.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Iterator
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from blackbook.config import SourceConfig
from blackbook.ingestion.base import ParsedDocument, SourceAdapter
from blackbook.retrieval.chunking import RawChunk
from blackbook.utils.paths import safe_join

log = logging.getLogger(__name__)

_DEFAULT_MAX_FILES = 500
_ALLOWED_CONTENT_TYPES = {"text/html", "application/xhtml+xml"}


class WebsiteAdapter(SourceAdapter):
    """Ingest linked HTML pages from one configured website origin."""

    def __init__(self, config: SourceConfig, raw_dir: str | None = None):
        super().__init__(config, raw_dir)
        if not config.url:
            raise ValueError(f"website source {config.id!r} requires a url")
        self.base_url = config.url.rstrip("/")
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"website source {config.id!r} has an invalid url")
        self._origin = (parsed.scheme, parsed.netloc)
        self._path_prefix = config.path_prefix.rstrip("/")

    def _workdir(self) -> Path:
        assert self.raw_dir, "raw_dir is required"
        return Path(self.raw_dir) / self.config.id

    def fetch(self, force: bool = False) -> None:
        workdir = self._workdir()
        pages_dir = workdir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        index_path = workdir / "index_urls.json"
        urls = self._discover_urls()
        index_path.write_text(json.dumps(urls), encoding="utf-8")

        delay = max(0.0, float(self.config.request_delay))
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            for url in urls:
                cache_path = self._cache_path(url)
                if cache_path.is_file() and not force:
                    continue
                try:
                    response = client.get(url)
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").split(";", 1)[0]
                    if content_type not in _ALLOWED_CONTENT_TYPES:
                        continue
                    if len(response.content) > self.config.max_document_bytes:
                        log.warning("%s: skipping oversized page %s", self.config.id, url)
                        continue
                    cache_path.write_bytes(response.content)
                except Exception as exc:
                    log.warning("%s: failed to fetch %s: %s", self.config.id, url, exc)
                if delay:
                    time.sleep(delay)

    def _discover_urls(self) -> list[str]:
        assert self.config.url
        max_files = self.config.max_files or _DEFAULT_MAX_FILES
        urls: list[str] = []
        queue = [self.base_url]
        seen: set[str] = set()
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            while queue and len(urls) < max_files:
                url = self._canonical_url(queue.pop(0))
                if not url or url in seen:
                    continue
                seen.add(url)
                try:
                    response = client.get(url)
                    response.raise_for_status()
                except Exception as exc:
                    log.warning("%s: failed to discover %s: %s", self.config.id, url, exc)
                    continue
                content_type = response.headers.get("content-type", "").split(";", 1)[0]
                if content_type not in _ALLOWED_CONTENT_TYPES:
                    continue
                urls.append(url)
                soup = BeautifulSoup(response.text, "lxml")
                for anchor in soup.find_all("a", href=True):
                    linked = self._canonical_url(urljoin(url, anchor["href"]))
                    if linked and linked not in seen:
                        queue.append(linked)
                if self.config.request_delay:
                    time.sleep(max(0.0, float(self.config.request_delay)))
        return urls

    def _canonical_url(self, url: str) -> str | None:
        url, _fragment = urldefrag(url)
        parsed = urlparse(url)
        if (parsed.scheme, parsed.netloc) != self._origin:
            return None
        if self._path_prefix and not (
            parsed.path == self._path_prefix
            or parsed.path.startswith(self._path_prefix + "/")
        ):
            return None
        if parsed.path.lower().endswith(('.pdf', '.zip', '.png', '.jpg', '.jpeg', '.gif', '.svg')):
            return None
        return url.rstrip("/") or f"{parsed.scheme}://{parsed.netloc}"

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return safe_join(self._workdir() / "pages", f"{digest}.html")

    def iter_documents(self) -> Iterator[ParsedDocument]:
        workdir = self._workdir()
        index_path = workdir / "index_urls.json"
        pages_dir = workdir / "pages"
        if not index_path.is_file() or not pages_dir.is_dir():
            log.error("%s pages not fetched; run fetch() first", self.config.id)
            return
        for raw_url in json.loads(index_path.read_text(encoding="utf-8")):
            url = self._canonical_url(raw_url)
            if not url:
                continue
            path = self._cache_path(url)
            if not path.is_file():
                continue
            document = self._parse_page(url, path)
            if document:
                yield document

    def _parse_page(self, url: str, path: Path) -> ParsedDocument | None:
        soup = BeautifulSoup(path.read_bytes(), "lxml")
        for node in soup(["script", "style", "nav", "footer", "noscript"]):
            node.decompose()
        title = soup.title.get_text(" ", strip=True) if soup.title else urlparse(url).path
        main = soup.find("main") or soup.find("article") or soup.body
        if main is None:
            return None
        chunks: list[RawChunk] = []
        ordinal = 0
        for node in main.find_all(["h1", "h2", "h3", "p", "pre", "li"]):
            text = node.get_text("\n" if node.name == "pre" else " ", strip=True)
            if not text:
                continue
            kind = "code" if node.name == "pre" else "text"
            chunks.append(RawChunk(text=text, section_path=[title], ordinal=ordinal, kind=kind))
            ordinal += 1
        text = "\n\n".join(chunk.text for chunk in chunks)
        if not text:
            return None
        return ParsedDocument(
            external_id=url,
            title=title,
            url=url,
            path=str(path),
            categories=self.config.categories,
            text=text,
            metadata={"source_type": "website", "metadata_inferred": True},
            chunks=chunks,
        )
