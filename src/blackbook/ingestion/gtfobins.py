"""GTFOBins ingestion adapter.

The GTFOBins project (GTFOBins/GTFOBins.github.io) documents Unix binaries
that can be abused for file reads/writes, shells, or privilege escalation.
The site is generated from a tree of extensionless YAML files
(``_gtfobins/<name>``), one per binary, each mapping abuse *functions*
(shell, file-read, sudo, ...) to code snippets and the contexts
(sudo, suid, unprivileged) in which they work.

Each YAML file becomes one document: every function with its code and the
contexts it works in is preserved, so a query for a binary name or an abuse
pattern lands on the exact snippet. Alias-only files (``alias: mawk``) are
folded into their target as "also known as" names instead of becoming
near-empty documents. Fetching is shared with the other GitHub sources
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

# GTFOBins documents Unix binaries; both tags make platform filters hit.
CATEGORIES = ["linux", "unix"]


class GtfoBinsAdapter(GithubTarballAdapter):
    """Ingests the GTFOBins YAML corpus."""

    source_id = "gtfobins"

    def __init__(self, config, raw_dir: str | None = None):
        super().__init__(config, raw_dir)
        self._extract_root: Path | None = None
        self._yaml = YAML(typ="safe", pure=True)

    # -- parsing -----------------------------------------------------------

    def _content_dir(self, root: Path) -> Path:
        rel = self.config.content_root or "_gtfobins"
        return root / rel

    def iter_documents(self) -> Iterator[ParsedDocument]:
        root = self._extract_root or self._find_extract_root(self._workdir())
        if not root or not root.is_dir():
            log.error("[gtfobins] extract root not found: %s", root)
            return
        content = self._content_dir(root)
        if not content.is_dir():
            log.error("[gtfobins] content root not found: %s", content)
            return

        files = [
            p for p in sorted(content.iterdir())
            if p.is_file() and not p.name.startswith(".")
        ]
        # First pass: collect alias-only entries (alias: mawk) so the target
        # document can advertise its alternate names.
        aliases = self._alias_map(files)

        max_files = self.config.max_files
        count = 0
        for path in files:
            if max_files is not None and count >= max_files:
                break
            count += 1
            try:
                doc = self._parse_file(path, aliases.get(path.stem, []))
                if doc is not None:
                    yield doc
            except Exception as e:  # a single bad file must not kill ingestion
                log.warning("failed to parse %s: %s", path, e)

    def _alias_map(self, files: list[Path]) -> dict[str, list[str]]:
        """Map target binary name -> the alias file names pointing at it."""
        aliases: dict[str, list[str]] = {}
        for path in files:
            try:
                data = self._yaml.load(
                    path.read_text(encoding="utf-8", errors="replace")
                )
            except Exception:
                continue
            if isinstance(data, dict) and data.get("alias") and not data.get("functions"):
                target = str(data["alias"])
                aliases.setdefault(target, []).append(path.stem)
        return aliases

    def _parse_file(
        self, path: Path, aliases: list[str]
    ) -> ParsedDocument | None:
        data = self._yaml.load(path.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(data, dict) or not data.get("functions"):
            return None  # alias-only and empty entries are handled elsewhere

        name = path.stem
        functions = data["functions"]
        body = self._render(name, functions, aliases)
        return ParsedDocument(
            external_id=f"_gtfobins/{name}",
            title=name,
            url=self._source_url(name),
            path=str(path),
            categories=list(CATEGORIES),
            text=body,
            metadata={
                "rel_path": f"_gtfobins/{name}",
                "format": "yaml",
                "functions": list(functions.keys()),
                "aliases": aliases,
                "contexts": sorted(self._contexts(functions)),
            },
            chunks=chunk_markdown(body, title_path=CATEGORIES + [name]),
        )

    def _source_url(self, name: str) -> str:
        site = (self.config.site_url or "").rstrip("/")
        if site:
            return f"{site}/{name}/"
        owner, repo = self._repo_slug()
        return (
            f"https://github.com/{owner}/{repo}/blob/{self._ref()}"
            f"/_gtfobins/{name}"
        )

    @staticmethod
    def _contexts(functions: dict[str, Any]) -> set[str]:
        found: set[str] = set()
        for entries in functions.values():
            for entry in entries or []:
                if isinstance(entry, dict):
                    found.update((entry.get("contexts") or {}).keys())
        return found

    # -- rendering -----------------------------------------------------------

    def _render(
        self, name: str, functions: dict[str, Any], aliases: list[str]
    ) -> str:
        """Render the YAML entry as markdown so chunks carry structure."""
        lines = [f"# {name}", ""]
        summary = (
            f"GTFOBins entry for the Unix binary `{name}`: abuse functions "
            "and the contexts (sudo, suid, unprivileged) in which they work."
        )
        if aliases:
            summary += f" Also known as: {', '.join(aliases)}."
        lines += [summary, ""]

        for fname, entries in functions.items():
            lines += [f"## {fname}", ""]
            for entry in entries or []:
                if not isinstance(entry, dict):
                    continue
                if entry.get("code"):
                    lines += ["```", str(entry["code"]).rstrip(), "```", ""]
                if entry.get("comment"):
                    lines += [str(entry["comment"]).strip(), ""]
                variants = self._context_variants(entry)
                if variants:
                    lines += [variants, ""]

        return "\n".join(lines)

    @staticmethod
    def _context_variants(entry: dict[str, Any]) -> str:
        """One line naming the contexts, plus per-context code variants."""
        contexts = entry.get("contexts") or {}
        if not contexts:
            return ""
        lines = [
            "Works with: "
            + ", ".join(
                c if contexts.get(c) not in (None, False) else f"{c} (limited)"
                for c in contexts
            )
            + "."
        ]
        for ctx, spec in contexts.items():
            # A context can carry its own snippet, e.g. the suid version of a
            # shell one-liner needs the -p flag.
            if isinstance(spec, dict) and spec.get("code"):
                lines += [
                    "",
                    f"`{ctx}` variant:",
                    "",
                    "```",
                    str(spec["code"]).rstrip(),
                    "```",
                ]
        return "\n".join(lines)
