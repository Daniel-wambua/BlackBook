"""WADComs ingestion adapter.

WADComs (WADComs/WADComs.github.io) is an interactive cheat sheet of
offensive security commands to use against Windows and Active Directory
environments. The site is generated from a tree of markdown files
(``_wadcoms/<Name>.md``), one per command, but the *entire* payload lives in
the YAML front matter (``description``, ``command``, ``services``, ``items``,
``OS``, ``attack_types``, ``references``) with an empty markdown body.

Each file becomes one document: the description and the exact command are
preserved so a query for a tool, service, or abuse pattern lands on the
command. Fetching is shared with the other GitHub sources
(:class:`GithubTarballAdapter`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

from ruamel.yaml import YAML

from blackbook.ingestion.base import ParsedDocument
from blackbook.ingestion.github_base import GithubTarballAdapter
from blackbook.retrieval.chunking import chunk_markdown

log = logging.getLogger(__name__)


def _slug(text: str) -> str:
    return "-".join(str(text).lower().split())


class WadcomsAdapter(GithubTarballAdapter):
    """Ingests the WADComs front-matter cheat-sheet corpus."""

    source_id = "wadcoms"

    def __init__(self, config, raw_dir: str | None = None):
        super().__init__(config, raw_dir)
        self._extract_root: Path | None = None
        self._yaml = YAML(typ="safe", pure=True)

    # -- parsing -----------------------------------------------------------

    def _content_dir(self, root: Path) -> Path:
        return root / (self.config.content_root or "_wadcoms")

    def iter_documents(self) -> Iterator[ParsedDocument]:
        root = self._extract_root or self._find_extract_root(self._workdir())
        if not root or not root.is_dir():
            log.error("[wadcoms] extract root not found: %s", root)
            return
        content = self._content_dir(root)
        if not content.is_dir():
            log.error("[wadcoms] content root not found: %s", content)
            return

        max_files = self.config.max_files
        count = 0
        for path in sorted(content.glob("*.md")):
            if max_files is not None and count >= max_files:
                break
            count += 1
            try:
                doc = self._parse_file(path)
                if doc is not None:
                    yield doc
            except Exception as e:  # a single bad file must not kill ingestion
                log.warning("failed to parse %s: %s", path, e)

    def _parse_file(self, path: Path) -> ParsedDocument | None:
        data = self._front_matter(path.read_text(encoding="utf-8", errors="replace"))
        if not data or not data.get("description") or not data.get("command"):
            return None

        stem = path.stem
        title = stem.replace("-", " ")
        os_list = [str(o) for o in data.get("OS") or []]
        attack_types = [str(a) for a in data.get("attack_types") or []]
        # OS values become lowercase platform tags (hard filters work);
        # attack types become dashed slugs alongside them.
        categories = [o.lower() for o in os_list] + [_slug(a) for a in attack_types]

        body = self._render(title, data)
        return ParsedDocument(
            external_id=f"_wadcoms/{stem}.md",
            title=title,
            url=self._source_url(stem),
            path=str(path),
            categories=categories,
            text=body,
            metadata={
                "rel_path": f"_wadcoms/{stem}.md",
                "format": "markdown",
                "services": [str(s) for s in data.get("services") or []],
                "items": [str(i) for i in data.get("items") or []],
                "os": os_list,
                "attack_types": attack_types,
                "references": [str(r) for r in data.get("references") or []],
            },
            chunks=chunk_markdown(body, title_path=["wadcoms", title]),
        )

    def _front_matter(self, text: str) -> dict[str, Any] | None:
        """The whole file is front matter; anything after the closing ``---``
        (some entries carry stray body text) is ignored."""
        if not text.startswith("---"):
            return None
        parts = text.split("---", 2)
        if len(parts) < 3:
            return None
        try:
            data = self._yaml.load(parts[1])
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def _source_url(self, stem: str) -> str:
        site = (self.config.site_url or "").rstrip("/")
        if site:
            return f"{site}/wadcoms/{stem}/"
        owner, repo = self._repo_slug()
        return (
            f"https://github.com/{owner}/{repo}/blob/{self._ref()}"
            f"/_wadcoms/{stem}.md"
        )

    # -- rendering -----------------------------------------------------------

    @staticmethod
    def _render(title: str, data: dict[str, Any]) -> str:
        """Render the front matter as markdown so chunks carry structure."""
        lines = [f"# {title}", ""]
        if data.get("description"):
            lines += [str(data["description"]).strip(), ""]

        if data.get("command"):
            lines += ["## Command", "", "```", str(data["command"]).rstrip(), "```", ""]

        meta = []
        if data.get("services"):
            meta.append("Services: " + ", ".join(map(str, data["services"])))
        if data.get("items"):
            meta.append("Requires: " + ", ".join(map(str, data["items"])))
        if data.get("OS"):
            meta.append("OS: " + ", ".join(map(str, data["OS"])))
        if data.get("attack_types"):
            meta.append("Attack types: " + ", ".join(map(str, data["attack_types"])))
        if meta:
            lines += [" · ".join(meta), ""]

        references = data.get("references") or []
        if references:
            lines += ["## References", ""]
            lines += [f"- {r}" for r in references]
            lines.append("")

        return "\n".join(lines)
