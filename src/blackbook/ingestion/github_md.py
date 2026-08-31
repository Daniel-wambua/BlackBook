"""Generic GitHub markdown-book ingestion adapter.

Turns any GitHub repository of markdown pages into a knowledge source:
fetches the repo as a tarball (via :class:`GithubTarballAdapter`), parses
each markdown page while preserving its category hierarchy (directory path
plus in-page heading breadcrumb) into chunks.

Per-source behaviour is configuration, not code:

* ``url`` + ``ref``            which repo/branch to fetch
* ``include_glob``             which files to parse (default ``**/*.md``)
* ``content_root``             ingest only this repo subtree (default: all)
* ``site_url``                 base URL of the published site; page URLs are
  mapped under it. When unset, page URLs point at the GitHub blob page,
  which always exists even for repos with no separate website.

The hierarchy is load-bearing for retrieval quality, so it is never
flattened away: every chunk records the full ``section_path``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from blackbook.config import SourceConfig
from blackbook.ingestion.base import ParsedDocument
from blackbook.ingestion.github_base import SKIP_DIRS, SKIP_SUFFIXES, GithubTarballAdapter
from blackbook.retrieval.chunking import chunk_markdown

log = logging.getLogger(__name__)


class GithubMarkdownAdapter(GithubTarballAdapter):
    """Ingests a GitHub repository of markdown pages."""

    def __init__(self, config: SourceConfig, raw_dir: str | None = None):
        super().__init__(config, raw_dir)
        self._extract_root: Path | None = None
        self._docs: list[Path] = []

    # -- parsing -----------------------------------------------------------

    def iter_documents(self) -> Iterator[ParsedDocument]:
        root = self._extract_root or self._find_extract_root(self._workdir())
        if not root or not root.is_dir():
            log.error("[%s] extract root not found: %s", self.source_id, root)
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
        glob = self.config.include_glob or "**/*.md"
        content_root = (self.config.content_root or "").strip("/")
        for path in root.glob(glob):
            rel = path.relative_to(root)
            # Skip non-content directories.
            if any(part in SKIP_DIRS for part in rel.parts):
                continue
            if path.suffix.lower() in SKIP_SUFFIXES:
                continue
            # Restrict to the configured subtree when one is set.
            if content_root and (
                len(rel.parts) <= len(content_root.split("/"))
                or rel.parts[: len(content_root.split("/"))]
                != tuple(content_root.split("/"))
            ):
                continue
            yield path

    def _rel_to_repo(self, root: Path, path: Path) -> Path:
        """Repo-root-relative path (``content_root`` NOT stripped)."""
        return path.relative_to(root)

    def _category_from_path(self, root: Path, path: Path) -> list[str]:
        """Category breadcrumb from the directory structure, with the
        configured ``content_root`` stripped so categories are semantic
        (``Active Directory``), not repo plumbing (``docs``)."""
        rel = self._rel_to_repo(root, path)
        parts = list(rel.parts[:-1])  # drop the filename
        content_root = (self.config.content_root or "").strip("/")
        if content_root:
            n = len(content_root.split("/"))
            parts = parts[n:]
        return [self._humanize(p) for p in parts]

    @staticmethod
    def _humanize(slug: str) -> str:
        return slug.replace("-", " ").replace("_", " ").strip().title()

    def _parse_file(self, root: Path, path: Path) -> ParsedDocument | None:
        raw = path.read_text(encoding="utf-8", errors="replace")
        if not raw.strip():
            return None
        # Strip YAML front matter if present.
        body = self._strip_front_matter(raw)
        rel = self._rel_to_repo(root, path)
        external_id = str(rel)
        categories = self._category_from_path(root, path)
        title = self._extract_title(body) or self._fallback_title(path, root)
        url = self._source_url(root, path)
        chunks = chunk_markdown(body, title_path=categories + [title])
        return ParsedDocument(
            external_id=external_id,
            title=title,
            url=url,
            path=str(path),
            categories=categories,
            text=body,
            metadata={"rel_path": external_id, "format": "markdown"},
            chunks=chunks,
        )

    @classmethod
    def _fallback_title(cls, path: Path, root: Path) -> str:
        # Jekyll-style repos put each page at <name>/index.md: the useful
        # title is the directory, not "index". Only applies when the file
        # actually sits inside a subdirectory of the repo root.
        stem = path.stem
        if stem.lower() == "index" and path.parent != root:
            return cls._humanize(path.parent.name)
        return cls._humanize(stem)

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

    def _source_url(self, root: Path, path: Path) -> str:
        """Map the repo path to a URL for the published page.

        With ``site_url`` set, the page URL is the site base plus the path
        relative to ``content_root`` (minus ``.md``, and a trailing
        ``index`` collapses to the directory). Without it, the GitHub blob
        URL is used: it always resolves, even for repos without a site.
        """
        rel = self._rel_to_repo(root, path)
        rel_str = str(rel).replace("\\", "/")
        site = (self.config.site_url or "").rstrip("/")
        if not site:
            owner, repo = self._repo_slug()
            return f"https://github.com/{owner}/{repo}/blob/{self._ref()}/{rel_str}"
        content_root = (self.config.content_root or "").strip("/")
        if content_root:
            prefix = content_root + "/"
            if rel_str.startswith(prefix):
                rel_str = rel_str[len(prefix) :]
        if rel_str.lower().endswith(".md"):
            rel_str = rel_str[:-3]
        if rel_str.lower().endswith("/index"):
            rel_str = rel_str[: -len("index")]
        return f"{site}/{rel_str}"
