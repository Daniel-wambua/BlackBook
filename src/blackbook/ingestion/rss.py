"""Configured RSS/Atom ingestion for sources with crawlable public feeds."""

from __future__ import annotations

import hashlib
import html
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

import httpx
from bs4 import BeautifulSoup

from blackbook.config import SourceConfig
from blackbook.ingestion.base import ParsedDocument, SourceAdapter
from blackbook.retrieval.chunking import RawChunk
from blackbook.utils.paths import safe_join

log = logging.getLogger(__name__)


class RssAdapter(SourceAdapter):
    """Ingest entries from one explicitly configured RSS or Atom feed."""

    def __init__(self, config: SourceConfig, raw_dir: str | None = None):
        super().__init__(config, raw_dir)
        if not config.feed_url:
            raise ValueError(f"RSS source {config.id!r} requires feed_url")
        self.feed_url = config.feed_url

    def _workdir(self) -> Path:
        assert self.raw_dir, "raw_dir is required"
        return Path(self.raw_dir) / self.source_id

    def fetch(self, force: bool = False) -> None:
        workdir = self._workdir()
        workdir.mkdir(parents=True, exist_ok=True)
        feed_path = safe_join(workdir, "feed.xml")
        metadata_path = safe_join(workdir, "feed-meta.json")
        metadata: dict = {}
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                metadata = {}
        headers = {}
        if not force and metadata.get("etag"):
            headers["If-None-Match"] = metadata["etag"]
        if not force and metadata.get("last_modified"):
            headers["If-Modified-Since"] = metadata["last_modified"]
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = client.get(self.feed_url, headers=headers)
        if response.status_code == 304 and feed_path.is_file():
            log.info("[%s] feed is up to date", self.source_id)
            return
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if "xml" not in content_type and not response.text.lstrip().startswith(("<rss", "<feed")):
            raise ValueError(f"{self.source_id}: configured feed did not return RSS/Atom XML")
        if len(response.content) > self.config.max_document_bytes:
            raise ValueError(f"{self.source_id}: feed exceeds max_document_bytes")
        feed_path.write_bytes(response.content)
        metadata_path.write_text(
            json.dumps({
                "etag": response.headers.get("etag"),
                "last_modified": response.headers.get("last-modified"),
                "content_hash": hashlib.sha256(response.content).hexdigest(),
            }),
            encoding="utf-8",
        )

    def iter_documents(self) -> Iterator[ParsedDocument]:
        feed_path = safe_join(self._workdir(), "feed.xml")
        if not feed_path.is_file():
            log.error("%s feed not fetched; run fetch() first", self.source_id)
            return
        root = ET.fromstring(feed_path.read_bytes())
        entries = list(root.findall(".//item"))
        if not entries:
            entries = list(root.findall(".//{http://www.w3.org/2005/Atom}entry"))
        max_files = self.config.max_files or 500
        for entry in entries[:max_files]:
            title = self._text(entry, "title") or "Untitled feed entry"
            link = self._link(entry)
            guid = self._text(entry, "guid") or link or title
            description = self._text(entry, "description")
            if not description:
                description = self._text(entry, "{http://purl.org/rss/1.0/modules/content/}encoded")
            if not description:
                description = self._text(entry, "{http://www.w3.org/2005/Atom}content")
            body = BeautifulSoup(html.unescape(description or ""), "lxml").get_text(" ", strip=True)
            if not body:
                continue
            text = f"{title}\n\n{body}"
            yield ParsedDocument(
                external_id=guid,
                title=title,
                url=link or self.config.url,
                path=str(feed_path),
                categories=self.config.categories,
                text=text,
                metadata={"source_type": "rss", "feed_url": self.feed_url},
                chunks=[RawChunk(text=text, section_path=[title], ordinal=0, kind="text")],
            )

    @staticmethod
    def _text(entry: ET.Element, name: str) -> str:
        node = entry.find(name)
        return (node.text or "").strip() if node is not None and node.text else ""

    @staticmethod
    def _link(entry: ET.Element) -> str | None:
        node = entry.find("link")
        if node is not None and node.text:
            return node.text.strip()
        atom = entry.find("{http://www.w3.org/2005/Atom}link")
        return atom.get("href") if atom is not None else None
