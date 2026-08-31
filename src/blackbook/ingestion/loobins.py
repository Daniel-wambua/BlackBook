"""LOOBins ingestion adapter.

LOOBins (infosecB/LOOBins, "Living Off the Orchard") documents built-in
macOS binaries and how threat actors can abuse them. The corpus is a tree of
YAML files (``LOOBins/<Name>.yml``), one per binary, each carrying a full
description, abuse use cases with code and ATT&CK tactics, binary paths,
detections, and resources.

Each YAML file becomes one document. It fills the macOS platform gap beside
LOLBAS (Windows) and GTFOBins (Unix). Fetching is shared with the other
GitHub sources (:class:`GithubTarballAdapter`).
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
    """ATT&CK tactic name -> the dashed lowercase slug used as a category."""
    return "-".join(text.lower().split())


class LooBinsAdapter(GithubTarballAdapter):
    """Ingests the LOOBins macOS YAML corpus."""

    source_id = "loobins"

    def __init__(self, config, raw_dir: str | None = None):
        super().__init__(config, raw_dir)
        self._extract_root: Path | None = None
        self._yaml = YAML(typ="safe", pure=True)

    # -- parsing -----------------------------------------------------------

    def _content_dir(self, root: Path) -> Path:
        return root / (self.config.content_root or "LOOBins")

    def iter_documents(self) -> Iterator[ParsedDocument]:
        root = self._extract_root or self._find_extract_root(self._workdir())
        if not root or not root.is_dir():
            log.error("[loobins] extract root not found: %s", root)
            return
        content = self._content_dir(root)
        if not content.is_dir():
            log.error("[loobins] content root not found: %s", content)
            return

        max_files = self.config.max_files
        count = 0
        for path in sorted(content.glob("*.yml")):
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
        data = self._yaml.load(path.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(data, dict) or not data.get("name"):
            return None

        name = str(data["name"])
        use_cases = data.get("example_use_cases") or []
        tactics = self._tactics(use_cases)
        categories = ["macos"] + [_slug(t) for t in tactics]

        body = self._render(name, data)
        return ParsedDocument(
            external_id=f"LOOBins/{name}.yml",
            title=name,
            url=self._source_url(name),
            path=str(path),
            categories=categories,
            text=body,
            metadata={
                "rel_path": f"LOOBins/{name}.yml",
                "format": "yaml",
                "author": data.get("author"),
                "tactics": tactics,
                "tags": self._tags(use_cases),
                "paths": [str(p) for p in data.get("paths") or []],
            },
            chunks=chunk_markdown(body, title_path=["macos", name]),
        )

    def _source_url(self, name: str) -> str:
        site = (self.config.site_url or "").rstrip("/")
        if site:
            return f"{site}/binaries/{name}/"
        owner, repo = self._repo_slug()
        return (
            f"https://github.com/{owner}/{repo}/blob/{self._ref()}"
            f"/LOOBins/{name}.yml"
        )

    @staticmethod
    def _tactics(use_cases: list[Any]) -> list[str]:
        """Deduped ATT&CK tactic names across all use cases, in order."""
        tactics: list[str] = []
        for uc in use_cases:
            for tactic in (uc.get("tactics") or []) if isinstance(uc, dict) else []:
                t = str(tactic)
                if t not in tactics:
                    tactics.append(t)
        return tactics

    @staticmethod
    def _tags(use_cases: list[Any]) -> list[str]:
        tags: list[str] = []
        for uc in use_cases:
            for tag in (uc.get("tags") or []) if isinstance(uc, dict) else []:
                t = str(tag)
                if t not in tags:
                    tags.append(t)
        return tags

    # -- rendering -----------------------------------------------------------

    def _render(self, name: str, data: dict[str, Any]) -> str:
        """Render the YAML entry as markdown so chunks carry structure."""
        lines = [f"# {name}", ""]
        if data.get("short_description"):
            lines += [str(data["short_description"]).strip(), ""]
        if data.get("full_description"):
            lines += [str(data["full_description"]).strip(), ""]

        use_cases = data.get("example_use_cases") or []
        if use_cases:
            lines += ["## Use Cases", ""]
            for uc in use_cases:
                if not isinstance(uc, dict):
                    continue
                if uc.get("name"):
                    lines += [f"### {uc['name']}", ""]
                if uc.get("description"):
                    lines += [str(uc["description"]).strip(), ""]
                if uc.get("code"):
                    lines += ["```", str(uc["code"]).rstrip(), "```", ""]
                meta = []
                if uc.get("tactics"):
                    meta.append("Tactics: " + ", ".join(map(str, uc["tactics"])))
                if uc.get("tags"):
                    meta.append("Tags: " + ", ".join(map(str, uc["tags"])))
                if meta:
                    lines += [" · ".join(meta), ""]

        paths = data.get("paths") or []
        if paths:
            lines += ["## Paths", ""]
            lines += [f"- `{p}`" for p in paths]
            lines.append("")

        detections = data.get("detections") or []
        if detections:
            lines += ["## Detection", ""]
            for d in detections:
                if isinstance(d, dict) and d.get("name"):
                    url = d.get("url")
                    if url and url != "N/A":
                        lines.append(f"- [{d['name']}]({url})")
                    else:
                        lines.append(f"- {d['name']}")
            lines.append("")

        resources = data.get("resources") or []
        if resources:
            lines += ["## Resources", ""]
            for r in resources:
                if isinstance(r, dict) and r.get("name"):
                    if r.get("url"):
                        lines.append(f"- [{r['name']}]({r['url']})")
                    else:
                        lines.append(f"- {r['name']}")
            lines.append("")

        return "\n".join(lines)
