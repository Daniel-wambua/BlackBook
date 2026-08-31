"""MITRE ATT&CK ingestion adapter.

Fetches an ATT&CK STIX bundle (``attack-stix-data`` on GitHub) and turns
every ``attack-pattern`` object into a document: the technique's
description, detection guidance, tactics and platforms. This grounds the
curated ATT&CK-ID mapping in ``knowledge/vocab.py`` with the real, current
ATT&CK data set: ``knowledge_technique`` can look a technique up by its
ATT&CK ID and cite the actual indexed text.

Only ``attack-pattern`` objects are ingested (mitigations/courses-of-action
may follow later). Revoked and deprecated techniques are skipped so a
citation never points at a retired page.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

import httpx

from blackbook.config import SourceConfig
from blackbook.ingestion.base import ParsedDocument, SourceAdapter
from blackbook.retrieval.chunking import chunk_markdown

log = logging.getLogger(__name__)


class MitreAttackAdapter(SourceAdapter):
    """Ingests MITRE ATT&CK techniques from a STIX bundle."""

    source_id = "attack"

    def __init__(self, config: SourceConfig, raw_dir: str | None = None):
        super().__init__(config, raw_dir)
        self._bundle_path: Path | None = None

    # -- fetching ----------------------------------------------------------

    def _workdir(self) -> Path:
        assert self.raw_dir, "raw_dir is required"
        return Path(self.raw_dir) / "attack"

    def fetch(self, force: bool = False) -> None:
        if not self.config.url:
            log.error("[attack] no url configured")
            return
        workdir = self._workdir()
        workdir.mkdir(parents=True, exist_ok=True)
        target = workdir / "bundle.json"
        marker = workdir / ".version"

        if not force and target.is_file() and marker.is_file():
            latest = self._latest_version()
            if latest is None or marker.read_text().strip() == latest:
                log.info("[attack] bundle already at latest version %s", latest or "?")
                self._bundle_path = target
                return

        log.info("[attack] downloading ATT&CK STIX bundle")
        with httpx.Client(timeout=600.0, follow_redirects=True) as client:
            with client.stream("GET", self.config.url) as r:
                r.raise_for_status()
                with open(target.with_suffix(".part"), "wb") as fh:
                    for chunk in r.iter_bytes():
                        fh.write(chunk)
        target.with_suffix(".part").replace(target)
        latest = self._latest_version()
        if latest:
            marker.write_text(latest)
        self._bundle_path = target

    def _latest_version(self) -> str | None:
        """Latest commit sha of the bundle file, to skip no-op downloads.

        The STIX bundle lives in the mitre-attack/attack-stix-data repo, so
        the GitHub commits API for the file path gives the version marker.
        """
        url = self.config.url or ""
        marker = "raw.githubusercontent.com/"
        if marker not in url:
            return None
        tail = url.split(marker, 1)[1]
        parts = tail.split("/")
        if len(parts) < 4:
            return None
        owner, repo, ref = parts[0], parts[1], parts[2]
        path = "/".join(parts[3:])
        api = (
            f"https://api.github.com/repos/{owner}/{repo}/commits"
            f"?path={path}&per_page=1"
        )
        try:
            with httpx.Client(timeout=20.0, follow_redirects=True) as client:
                r = client.get(api, headers={"Accept": "application/vnd.github+json"})
                r.raise_for_status()
                commits = r.json()
                if isinstance(commits, list) and commits:
                    return commits[0].get("sha")
        except Exception as e:  # network is best-effort; fall back to cached
            log.warning("[attack] could not query latest bundle commit: %s", e)
        return None

    # -- parsing -----------------------------------------------------------

    def iter_documents(self) -> Iterator[ParsedDocument]:
        path = self._bundle_path or (self._workdir() / "bundle.json")
        if not path.is_file():
            log.error("[attack] bundle not found: %s", path)
            return
        try:
            bundle = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            log.error("[attack] cannot parse STIX bundle: %s", e)
            return

        max_files = self.config.max_files
        count = 0
        for obj in bundle.get("objects") or []:
            if obj.get("type") != "attack-pattern":
                continue
            if obj.get("revoked") or obj.get("x_mitre_deprecated"):
                continue  # never cite a retired technique
            if max_files is not None and count >= max_files:
                break
            count += 1
            try:
                doc = self._parse_technique(obj)
                if doc is not None:
                    yield doc
            except Exception as e:
                log.warning("failed to parse ATT&CK object: %s", e)

    def _parse_technique(self, obj: dict) -> ParsedDocument | None:
        name = (obj.get("name") or "").strip()
        if not name:
            return None
        attack_id, url = self._attack_ref(obj)
        if not attack_id:
            return None  # sub-objects without an ATT&CK ID are not citable

        description = (obj.get("description") or "").strip()
        detection = (obj.get("x_mitre_detection") or "").strip()
        tactics = sorted(
            {
                phase.get("phase_name")
                for phase in obj.get("kill_chain_phases") or []
                if phase.get("phase_name")
            }
        )
        platforms = sorted(
            {str(p) for p in obj.get("x_mitre_platforms") or [] if p}
        )

        # Platforms become lowercase category tags so the platform hard
        # filter works against ATT&CK documents too ("windows", "linux",
        # "office 365", ...); tactics keep their dashed ATT&CK slugs.
        categories = [p.lower() for p in platforms] + [t.lower() for t in tactics]

        body = self._render(
            name, attack_id, description, detection, tactics, platforms
        )
        title = f"{name} ({attack_id})"
        chunks = chunk_markdown(body, title_path=["ATT&CK", title])
        return ParsedDocument(
            external_id=attack_id,
            title=title,
            url=url,
            path=str(self._bundle_path or ""),
            categories=categories,
            text=body,
            metadata={
                "format": "stix",
                "attack_id": attack_id,
                "tactics": tactics,
                "platforms": platforms,
                "is_subtechnique": bool(obj.get("x_mitre_is_subtechnique")),
                "date": (obj.get("modified") or obj.get("created") or "")[:10],
            },
            chunks=chunks,
        )

    @staticmethod
    def _attack_ref(obj: dict) -> tuple[str | None, str | None]:
        for ref in obj.get("external_references") or []:
            if ref.get("source_name", "").startswith("mitre-attack"):
                return ref.get("external_id"), ref.get("url")
        return None, None

    @staticmethod
    def _render(
        name: str,
        attack_id: str,
        description: str,
        detection: str,
        tactics: list[str],
        platforms: list[str],
    ) -> str:
        lines = [f"# {name} ({attack_id})", ""]
        if description:
            lines += ["## Description", "", description, ""]
        if tactics:
            lines += ["## Tactics", "", ", ".join(tactics), ""]
        if platforms:
            lines += ["## Platforms", "", ", ".join(platforms), ""]
        if detection:
            lines += ["## Detection", "", detection, ""]
        return "\n".join(lines)

    # -- version info --------------------------------------------------------

    def current_version(self) -> str | None:
        marker = self._workdir() / ".version"
        return marker.read_text().strip() if marker.is_file() else None
