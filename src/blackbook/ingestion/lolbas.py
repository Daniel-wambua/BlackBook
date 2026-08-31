"""LOLBAS ingestion adapter.

The LOLBAS project (LOLBAS-Project/LOLBAS) documents Living-Off-the-Land
binaries and scripts on Windows. Unlike the other GitHub sources it is a
tree of YAML files (``yml/<Category>/<Name>.yml``), one per binary, each
carrying a description, abuse commands with ATT&CK mappings, full paths,
and detection pointers.

Each YAML file becomes one document: the description plus every command
(with its use-case and ATT&CK ID) is preserved so a query for a binary
name or an abuse pattern lands on the exact command. Fetching is shared
with the other GitHub sources (:class:`GithubTarballAdapter`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from ruamel.yaml import YAML

from blackbook.config import SourceConfig
from blackbook.ingestion.base import ParsedDocument
from blackbook.ingestion.github_base import GithubTarballAdapter
from blackbook.retrieval.chunking import chunk_markdown

log = logging.getLogger(__name__)


class LolbasAdapter(GithubTarballAdapter):
    """Ingests the LOLBAS YAML corpus."""

    source_id = "lolbas"

    def __init__(self, config: SourceConfig, raw_dir: str | None = None):
        super().__init__(config, raw_dir)
        self._extract_root: Path | None = None
        self._yaml = YAML(typ="safe", pure=True)

    # -- parsing -----------------------------------------------------------

    def iter_documents(self) -> Iterator[ParsedDocument]:
        root = self._extract_root or self._find_extract_root(self._workdir())
        if not root or not root.is_dir():
            log.error("[lolbas] extract root not found: %s", root)
            return
        max_files = self.config.max_files
        count = 0
        for path in sorted(root.rglob("*.yml")):
            if max_files is not None and count >= max_files:
                break
            rel = path.relative_to(root)
            if any(part.startswith(".") for part in rel.parts):
                continue
            count += 1
            try:
                doc = self._parse_file(root, path)
                if doc is not None:
                    yield doc
            except Exception as e:  # a single bad file must not kill ingestion
                log.warning("failed to parse %s: %s", path, e)

    def _parse_file(self, root: Path, path: Path) -> ParsedDocument | None:
        data = self._yaml.load(path.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(data, dict) or not data.get("Name"):
            return None

        rel = path.relative_to(root)
        external_id = str(rel)
        name = str(data["Name"])
        description = str(data.get("Description") or "").strip()

        # Category from the repo layout: yml/OSBinaries/Certutil.yml -> "os binaries"
        parts = list(rel.parts[:-1])
        if parts and parts[0].lower() == "yml":
            parts = parts[1:]
        categories = [self._humanize(p) for p in parts]

        body = self._render(name, description, data)
        chunks = chunk_markdown(body, title_path=categories + [name])
        return ParsedDocument(
            external_id=external_id,
            title=name,
            url=self._source_url(root, rel),
            path=str(path),
            categories=categories,
            text=body,
            metadata={
                "rel_path": external_id,
                "format": "yaml",
                "mitre_ids": self._mitre_ids(data),
                "full_path": [
                    str(fp.get("Path")) for fp in data.get("Full_Path") or []
                    if isinstance(fp, dict) and fp.get("Path")
                ],
            },
            chunks=chunks,
        )

    @staticmethod
    def _humanize(slug: str) -> str:
        return slug.replace("-", " ").replace("_", " ").strip().title()

    def _source_url(self, root: Path, rel: Path) -> str:
        # yml/OSBinaries/Certutil.yml -> <site>/osbinaries/certutil/
        rel_str = str(rel).replace("\\", "/")
        site = (self.config.site_url or "").rstrip("/")
        parts = rel_str.split("/")
        if parts[0].lower() == "yml":
            parts = parts[1:]
        page = "/".join(p.lower() for p in parts)
        if page.lower().endswith(".yml"):
            page = page[: -len(".yml")]
        if not site:
            owner, repo = self._repo_slug()
            return f"https://github.com/{owner}/{repo}/blob/{self._ref()}/{rel_str}"
        return f"{site}/{page}/"

    @staticmethod
    def _mitre_ids(data: dict) -> list[str]:
        ids: list[str] = []
        for cmd in data.get("Commands") or []:
            if isinstance(cmd, dict) and cmd.get("MitreID"):
                mid = str(cmd["MitreID"])
                if mid not in ids:
                    ids.append(mid)
        return ids

    def _render(self, name: str, description: str, data: dict) -> str:
        """Render the YAML entry as markdown so chunks carry structure."""
        lines = [f"# {name}", ""]
        if description:
            lines += [description, ""]
        commands = data.get("Commands") or []
        if commands:
            lines += ["## Commands", ""]
            for i, cmd in enumerate(commands, start=1):
                if not isinstance(cmd, dict):
                    continue
                lines.append(f"### {i}. {cmd.get('Category', 'Abuse')}")
                if cmd.get("Description"):
                    lines += [str(cmd["Description"]), ""]
                if cmd.get("Command"):
                    lines += ["```", str(cmd["Command"]), "```", ""]
                meta = []
                if cmd.get("Usecase"):
                    meta.append(f"Usecase: {cmd['Usecase']}")
                if cmd.get("Privileges"):
                    meta.append(f"Privileges: {cmd['Privileges']}")
                if cmd.get("MitreID"):
                    meta.append(f"ATT&CK: {cmd['MitreID']}")
                if meta:
                    lines += [" · ".join(meta), ""]
        full_path = data.get("Full_Path") or []
        if full_path:
            lines += ["## Full Path", ""]
            for fp in full_path:
                if isinstance(fp, dict) and fp.get("Path"):
                    lines.append(f"- `{fp['Path']}`")
            lines.append("")
        detection = data.get("Detection") or []
        if detection:
            lines += ["## Detection", ""]
            for d in detection:
                if isinstance(d, dict):
                    label = next(iter(d), "")
                    if label and d[label]:
                        lines.append(f"- {label}: {d[label]}")
                elif d:
                    lines.append(f"- {d}")
            lines.append("")
        return "\n".join(lines)
