"""HackTricks ingestion adapter.

HackTricks is a GitBook/MkDocs-style markdown tree on GitHub. We fetch the
repository as a tarball over plain HTTP (no shell/git execution), extract the
markdown, and parse each page while preserving its *category hierarchy* — both
the directory path and the in-page heading breadcrumb — into chunks.

The hierarchy is load-bearing for retrieval quality, so it is never flattened
away: every chunk records the full ``section_path``.
"""

from __future__ import annotations

import io
import json
import logging
import tarfile
from pathlib import Path
from typing import Iterator

import httpx

from blackbook.config import SourceConfig
from blackbook.ingestion.base import ParsedDocument, SourceAdapter
from blackbook.retrieval.chunking import chunk_markdown
from blackbook.utils.paths import is_within

log = logging.getLogger(__name__)

_COMMIT_API = "https://api.github.com/repos/HackTricks-wiki/hacktricks/commits/master"
_TARBALL_URL = (
    "https://codeload.github.com/HackTricks-wiki/hacktricks/tar.gz/refs/heads/master"
)

# Directories that are not knowledge content (images, CI, theme, etc.).
_SKIP_DIRS = {
    ".git",
    ".github",
    "node_modules",
    "theme",
    "src/images",
    "images",
    "static",
    ".assets",
}
_SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".pdf"}


class HackTricksAdapter(SourceAdapter):
    """Ingests the HackTricks markdown book."""

    source_id = "hacktricks"

    def __init__(self, config: SourceConfig, raw_dir: str | None = None):
        super().__init__(config, raw_dir)
        self._extract_root: Path | None = None
        self._docs: list[Path] = []

    # -- fetching ----------------------------------------------------------

    def _workdir(self) -> Path:
        assert self.raw_dir, "raw_dir is required"
        return Path(self.raw_dir) / "hacktricks"

    def fetch(self, force: bool = False) -> None:
        workdir = self._workdir()
        marker = workdir / ".commit"
        workdir.mkdir(parents=True, exist_ok=True)

        # Determine the latest commit to skip no-op fetches.
        latest = self._latest_commit()
        if not force and marker.is_file() and latest and marker.read_text().strip() == latest:
            log.info("hacktricks already at latest commit %s", latest[:12])
            self._extract_root = self._find_extract_root(workdir)
            return

        log.info("downloading HackTricks tarball (latest=%s)", (latest or "?")[:12])
        tarball = self._download_tarball()
        self._safe_extract(tarball, workdir)
        self._extract_root = self._find_extract_root(workdir)
        if latest:
            marker.write_text(latest)

    def _latest_commit(self) -> str | None:
        try:
            with httpx.Client(timeout=20.0, follow_redirects=True) as client:
                r = client.get(_COMMIT_API, headers={"Accept": "application/vnd.github+json"})
                r.raise_for_status()
                return r.json().get("sha")
        except Exception as e:  # network is best-effort; fall back to cached
            log.warning("could not query latest hacktricks commit: %s", e)
            return None

    def _download_tarball(self) -> bytes:
        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            with client.stream("GET", _TARBALL_URL) as r:
                r.raise_for_status()
                return r.read()

    def _safe_extract(self, tarball: bytes, dest: Path) -> None:
        """Extract the tarball, refusing any member that escapes ``dest``."""
        dest_resolved = dest.resolve()
        with tarfile.open(fileobj=io.BytesIO(tarball), mode="r:gz") as tf:
            for member in tf.getmembers():
                # Zip-slip / path traversal protection.
                target = (dest_resolved / member.name).resolve()
                if not is_within(target, dest_resolved):
                    log.warning("skipping unsafe tar member: %s", member.name)
                    continue
                if member.isdev() or member.issym() or member.islnk():
                    # Don't trust device files / links from a tarball.
                    continue
                tf.extract(member, dest_resolved, filter="data")

    def _find_extract_root(self, workdir: Path) -> Path:
        """The tarball expands to a single top-level dir like hacktricks-master."""
        for child in sorted(workdir.iterdir()):
            if child.is_dir() and child.name.startswith("hacktricks"):
                return child
        return workdir

    # -- parsing -----------------------------------------------------------

    def iter_documents(self) -> Iterator[ParsedDocument]:
        root = self._extract_root or self._find_extract_root(self._workdir())
        if not root or not root.is_dir():
            log.error("hacktricks extract root not found: %s", root)
            return
        md_files = sorted(self._iter_markdown(root))
        max_files = self.config.max_files
        count = 0
        for path in md_files:
            if max_files is not None and count >= max_files:
                break
            count += 1
            try:
                doc = self._parse_file(root, path)
                if doc is not None:
                    yield doc
            except Exception as e:  # a single bad file must not kill ingestion
                log.warning("failed to parse %s: %s", path, e)

    def _iter_markdown(self, root: Path) -> Iterator[Path]:
        for path in root.rglob("*.md"):
            rel = path.relative_to(root)
            # Skip non-content directories.
            if any(part in _SKIP_DIRS for part in rel.parts):
                continue
            # Skip READMEs that are just index pages unless they have content.
            yield path

    def _category_from_path(self, root: Path, path: Path) -> list[str]:
        """Derive the category breadcrumb from the directory structure."""
        rel = path.relative_to(root)
        parts = [p for p in rel.parts[:-1]]  # drop the filename
        cats = [self._humanize(p) for p in parts]
        return cats

    @staticmethod
    def _humanize(slug: str) -> str:
        return slug.replace("-", " ").replace("_", " ").strip().title()

    def _parse_file(self, root: Path, path: Path) -> ParsedDocument | None:
        raw = path.read_text(encoding="utf-8", errors="replace")
        if not raw.strip():
            return None
        # Strip YAML front matter if present.
        body = self._strip_front_matter(raw)
        rel = path.relative_to(root)
        external_id = str(rel)
        categories = self._category_from_path(root, path)
        title = self._extract_title(body) or self._humanize(path.stem)
        url = self._source_url(rel)
        chunks = chunk_markdown(body, title_path=categories + [title])
        full_text = body
        return ParsedDocument(
            external_id=external_id,
            title=title,
            url=url,
            path=str(path),
            categories=categories,
            text=full_text,
            metadata={"rel_path": external_id, "format": "markdown"},
            chunks=chunks,
        )

    @staticmethod
    def _strip_front_matter(text: str) -> str:
        if text.startswith("---"):
            end = text.find("\n---", 3)
            if end != -1:
                return text[end + 4 :].lstrip("\n")
        return text

    @staticmethod
    def _extract_title(body: str) -> str | None:
        for line in body.splitlines():
            s = line.strip()
            if s.startswith("# "):
                return s[2:].strip()
            if s:  # first non-empty non-heading line -> stop looking
                break
        return None

    def _source_url(self, rel: Path) -> str:
        # Map the repo path to the published HackTricks page URL. The book is
        # served at book.hacktricks.xyz with the directory structure mirrored
        # and .md dropped.
        rel_str = str(rel).replace("\\", "/")
        if rel_str.endswith(".md"):
            rel_str = rel_str[:-3]
        return f"https://book.hacktricks.xyz/{rel_str}"

    # -- version info --------------------------------------------------------

    def current_version(self) -> str | None:
        marker = self._workdir() / ".commit"
        return marker.read_text().strip() if marker.is_file() else None
