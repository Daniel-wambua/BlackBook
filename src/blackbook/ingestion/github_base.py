"""Shared GitHub-tarball fetching for git sources.

Covers the mechanics common to every "fetch a GitHub repo as a tarball"
source: latest-commit check (to skip no-op fetches), codeload download,
zip-slip-safe extraction, and the version marker. Parsing is left to
subclasses (:class:`GithubMarkdownAdapter` in ``github_md.py`` iterates
markdown; ``lolbas.py`` iterates YAML).

The repo and branch come from the source config (``url`` + ``ref``), so new
sources of the same shape are configuration, not code.
"""

from __future__ import annotations

import io
import logging
import tarfile
from pathlib import Path

import httpx

from blackbook.ingestion.base import SourceAdapter
from blackbook.utils.paths import is_within

log = logging.getLogger(__name__)

# Directories that are not knowledge content (images, CI, theme, etc.).
SKIP_DIRS = {
    ".git",
    ".github",
    ".gitbook",
    "node_modules",
    "theme",
    "src/images",
    "images",
    "static",
    ".assets",
}
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".pdf"}


class GithubTarballAdapter(SourceAdapter):
    """Fetches a GitHub repository as a tarball over plain HTTP.

    No shell or git execution: the tarball comes from codeload.github.com
    and is extracted with per-member path-traversal checks.
    """

    def _repo_slug(self) -> tuple[str, str]:
        """(owner, repo) parsed from the configured ``url``."""
        url = (self.config.url or "").rstrip("/")
        if url.endswith(".git"):
            url = url[:-4]
        parts = [p for p in url.split("/") if p]
        if len(parts) < 2:
            raise ValueError(
                f"source {self.config.id!r}: cannot parse owner/repo from url "
                f"{self.config.url!r} (expected https://github.com/<owner>/<repo>[.git])"
            )
        return parts[-2], parts[-1]

    def _ref(self) -> str:
        return self.config.ref or "master"

    # -- fetching ----------------------------------------------------------

    def _workdir(self) -> Path:
        assert self.raw_dir, "raw_dir is required"
        return Path(self.raw_dir) / self.source_id

    def fetch(self, force: bool = False) -> None:
        workdir = self._workdir()
        marker = workdir / ".commit"
        workdir.mkdir(parents=True, exist_ok=True)

        # Determine the latest commit to skip no-op fetches.
        latest = self._latest_commit()
        if (
            not force
            and marker.is_file()
            and latest
            and marker.read_text().strip() == latest
        ):
            log.info("[%s] already at latest commit %s", self.source_id, latest[:12])
            self._extract_root = self._find_extract_root(workdir)
            return

        log.info(
            "[%s] downloading tarball (latest=%s)", self.source_id, (latest or "?")[:12]
        )
        tarball = self._download_tarball()
        self._safe_extract(tarball, workdir)
        self._extract_root = self._find_extract_root(workdir)
        if latest:
            marker.write_text(latest)

    def _latest_commit(self) -> str | None:
        owner, repo = self._repo_slug()
        url = f"https://api.github.com/repos/{owner}/{repo}/commits/{self._ref()}"
        try:
            with httpx.Client(timeout=20.0, follow_redirects=True) as client:
                r = client.get(url, headers={"Accept": "application/vnd.github+json"})
                r.raise_for_status()
                return r.json().get("sha")
        except Exception as e:  # network is best-effort; fall back to cached
            log.warning(
                "[%s] could not query latest commit: %s", self.source_id, e
            )
            return None

    def _download_tarball(self) -> bytes:
        owner, repo = self._repo_slug()
        url = (
            f"https://codeload.github.com/{owner}/{repo}"
            f"/tar.gz/refs/heads/{self._ref()}"
        )
        with httpx.Client(timeout=300.0, follow_redirects=True) as client:
            with client.stream("GET", url) as r:
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
        """The tarball expands to a single top-level dir like ``repo-<ref>``.

        Codeload tarballs always do; fall back to the workdir itself for
        hand-seeded test fixtures.
        """
        dirs = [c for c in sorted(workdir.iterdir()) if c.is_dir() and not c.name.startswith(".")]
        if len(dirs) == 1:
            return dirs[0]
        return workdir

    # -- version info --------------------------------------------------------

    def current_version(self) -> str | None:
        marker = self._workdir() / ".commit"
        return marker.read_text().strip() if marker.is_file() else None
