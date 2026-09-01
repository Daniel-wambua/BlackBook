"""BlackBook configuration.

Configuration is loaded from three layers, in increasing priority:

1. Defaults defined in :class:`Settings`.
2. A YAML config file (``~/.blackbook/config.yaml`` by default).
3. Environment variables prefixed with ``BLACKBOOK_``.

Paths support ``~`` expansion. The settings object is the single source of
truth for the source registry (which sources exist, whether they are enabled,
and how they are fetched) and for retrieval behaviour.
"""

from __future__ import annotations

import ipaddress
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ruamel.yaml import YAML


def expand_path(p: str | Path) -> Path:
    """Expand ``~`` and environment variables in a path, without requiring it
    to exist. Returned path is absolute (resolved as far as possible)."""
    s = os.path.expandvars(str(p))
    s = os.path.expanduser(s)
    return Path(s)


class SourceConfig(BaseModel):
    """Configuration for a single knowledge source."""

    id: str
    name: str
    enabled: bool = True
    # "git" (clone + parse), "website" (crawl/scrape), "filesystem" (local files)
    type: Literal["git", "website", "filesystem"] = "filesystem"
    # Source authority affects ranking weight and how results are presented.
    authority: Literal["official", "trusted", "user", "unknown"] = "unknown"
    # Optional category tags propagated to every chunk from this source.
    categories: list[str] = Field(default_factory=list)

    # git / website sources
    url: str | None = None
    ref: str | None = None  # git branch/tag/commit
    # Base URL of the published site for GitHub-markdown sources. When set,
    # page URLs map under it; when unset, URLs point at the GitHub blob page.
    site_url: str | None = None
    # Ingest only this repo subtree (repo-relative, e.g. "docs"); "" = all.
    content_root: str = ""

    # filesystem sources
    directory: str | None = None
    # "" = adapter default ("**/*.pdf" for filesystem, "**/*.md" for
    # GitHub-markdown sources)
    include_glob: str = ""
    # Comma-separated glob patterns (fnmatch semantics, "/"-separated
    # repo-relative paths) for files to skip even when include_glob matches.
    # Use to drop repo plumbing or link-index files from a GitHub source.
    exclude_glob: str = ""

    # Fetch limits (security / cost control)
    max_files: int | None = None
    max_document_bytes: int = 200 * 1024 * 1024  # 200 MB per document cap
    # Politeness delay between HTTP requests when crawling a website source
    # (seconds). Applies per request, not per document.
    request_delay: float = 0.5

    @field_validator("id", "name", mode="before")
    @classmethod
    def _coerce_str(cls, v):
        # Safety net: YAML 1.1 can parse an id like ``0xdf`` as an integer
        # (hex). Coerce to string rather than crash. Config files should quote
        # such ids (``id: "0xdf"``) to preserve the intended spelling.
        return str(v) if v is not None else v


class EmbeddingsConfig(BaseModel):
    enabled: bool = False
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32


class RetrievalConfig(BaseModel):
    default_limit: int = 8
    max_limit: int = 50
    max_context_chunks: int = 12
    # Source-diversity: at most this many chunks may come from one document.
    per_document_cap: int = 2
    snippet_chars: int = 400


class DatabaseConfig(BaseModel):
    # When empty, the db path is derived from ``Settings.home`` as
    # ``<home>/data.db``. Set this explicitly to override.
    path: str = ""


class LoggingConfig(BaseModel):
    level: str = "INFO"


class ServerConfig(BaseModel):
    """Network settings for the HTTP transport.

    Only consumed when the server is started with an HTTP transport
    (``blackbook serve --http``). The stdio transport ignores these.
    """

    host: str = "127.0.0.1"
    port: int = 8890
    # Path the JSON-RPC (streamable-http) endpoint is mounted at. Clients
    # connect to ``http://<host>:<port><path>``.
    path: str = "/mcp"
    # Optional bearer token guarding the HTTP transport. When set, requests
    # must carry ``Authorization: Bearer <token>`` or they get 401. Strongly
    # recommended whenever binding a non-loopback address.
    auth_token: str = ""
    # Refuse to start on a non-loopback address when no auth_token is set.
    # Flip to false only if you understand the exposure.
    require_auth_off_loopback: bool = True


def _default_sources() -> list[SourceConfig]:
    return [
        SourceConfig(
            id="hacktricks",
            name="HackTricks",
            enabled=True,
            type="git",
            authority="trusted",
            url="https://github.com/HackTricks-wiki/hacktricks.git",
            ref="master",
        ),
        SourceConfig(
            id="0xdf",
            name="0xdf",
            enabled=True,
            type="website",
            authority="trusted",
            url="https://0xdf.gitlab.io/",
        ),
        SourceConfig(
            id="local_pdfs",
            name="Local PDFs",
            enabled=True,
            type="filesystem",
            authority="user",
            directory="~/knowledge/pdfs",
        ),
        # GitHub markdown sources (GithubMarkdownAdapter): repo + branch +
        # optional content subtree and published-site mapping, all config.
        SourceConfig(
            id="payloads",
            name="PayloadsAllTheThings",
            enabled=True,
            type="git",
            authority="trusted",
            url="https://github.com/swisskyrepo/PayloadsAllTheThings.git",
            ref="master",
        ),
        SourceConfig(
            id="hacker_recipes",
            name="The Hacker Recipes",
            enabled=True,
            type="git",
            authority="trusted",
            url="https://github.com/The-Hacker-Recipes/The-Hacker-Recipes.git",
            ref="main",
            content_root="docs",
            site_url="https://thehacker.recipes",
        ),
        SourceConfig(
            id="gtfobins",
            name="GTFOBins",
            enabled=True,
            type="git",
            authority="trusted",
            url="https://github.com/GTFOBins/GTFOBins.github.io.git",
            ref="master",
            content_root="_gtfobins",
            site_url="https://gtfobins.github.io",
        ),
        # LOLBAS is YAML, not markdown: its own adapter over the same
        # tarball-fetching base.
        SourceConfig(
            id="lolbas",
            name="LOLBAS",
            enabled=True,
            type="git",
            authority="trusted",
            url="https://github.com/LOLBAS-Project/LOLBAS.git",
            ref="master",
            site_url="https://lolbas-project.github.io",
        ),
        # MITRE ATT&CK STIX data (attack-pattern objects -> documents).
        SourceConfig(
            id="attack",
            name="MITRE ATT&CK",
            enabled=True,
            type="git",
            authority="official",
            url=(
                "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/"
                "master/enterprise-attack/enterprise-attack.json"
            ),
            ref="master",
        ),
        # LOOBins: macOS binaries, YAML corpus (fills the macOS gap beside
        # LOLBAS/Windows and GTFOBins/Unix).
        SourceConfig(
            id="loobins",
            name="LOOBins",
            enabled=True,
            type="git",
            authority="trusted",
            url="https://github.com/infosecB/LOOBins.git",
            ref="main",
            content_root="LOOBins",
            site_url="https://loobins.io",
        ),
        # WADComs: offensive Windows/AD command cheat sheets; the whole
        # payload lives in markdown front matter.
        SourceConfig(
            id="wadcoms",
            name="WADComs",
            enabled=True,
            type="git",
            authority="trusted",
            url="https://github.com/WADComs/WADComs.github.io.git",
            ref="master",
            content_root="_wadcoms",
            site_url="https://wadcoms.github.io",
        ),
        # Internal All The Things: swisskyrepo's AD / internal-network
        # pentest cheat sheets (MkDocs, content under docs/).
        SourceConfig(
            id="internal_all_the_things",
            name="Internal All The Things",
            enabled=True,
            type="git",
            authority="trusted",
            url="https://github.com/swisskyrepo/InternalAllTheThings.git",
            ref="main",
            content_root="docs",
            site_url="https://swisskyrepo.github.io/InternalAllTheThings",
        ),
        # Moamen Basel's HTB writeups collection (Jekyll/Just the Docs).
        # Full in-repo machine writeups, challenges and cheatsheets; the
        # 0xdf/IppSec link indexes duplicate other sources, and templates /
        # CONTRIBUTING / the GitHub landing README are repo plumbing, so all
        # are excluded. Every real page carries a Jekyll permalink, which the
        # adapter uses as its canonical citation URL.
        SourceConfig(
            id="htb_writeups",
            name="HTB Writeups (Moamen Basel)",
            enabled=True,
            type="git",
            authority="trusted",
            url="https://github.com/momenbasel/htb-writeups.git",
            ref="main",
            site_url="https://www.moamenbasel.com/htb-writeups",
            exclude_glob=(
                "templates/**, 0xdf-htb-machines.md, ippsec-video-index.md, "
                "CONTRIBUTING.md, README.md"
            ),
        ),
    ]


class Settings(BaseSettings):
    """Top-level application settings."""

    model_config = SettingsConfigDict(
        env_prefix="BLACKBOOK_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    home: Path = Field(default_factory=lambda: expand_path("~/.blackbook"))
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    sources: list[SourceConfig] = Field(default_factory=_default_sources)

    # ---- derived helpers -------------------------------------------------

    @property
    def db_path(self) -> Path:
        if self.database.path:
            return expand_path(self.database.path)
        return self.home / "data.db"

    @property
    def raw_dir(self) -> Path:
        """Directory where raw source checkouts/downloads live."""
        return self.home / "raw"

    def get_source(self, source_id: str) -> SourceConfig | None:
        for s in self.sources:
            if s.id == source_id:
                return s
        return None

    def enabled_sources(self) -> list[SourceConfig]:
        return [s for s in self.sources if s.enabled]

    def source_ids(self, requested: list[str] | None) -> list[str] | None:
        """Resolve a requested source filter to concrete source IDs.

        Returns ``None`` for "every enabled source" (no filter), or a list of
        matching IDs. Unknown IDs are dropped so a typo can never *widen* the
        query beyond intent: if every requested ID is unknown the result is an
        empty list, which callers must treat as "search nothing" — never as
        "search everything".
        """
        enabled = {s.id for s in self.enabled_sources()}
        if not requested or requested == ["all"]:
            return None
        return [sid for sid in requested if sid in enabled]


def is_loopback_host(host: str) -> bool:
    """True when ``host`` binds a loopback-only interface.

    Accepts the obvious spellings (``127.0.0.1``, ``localhost``, ``::1``) and
    anything inside the loopback ranges; unparseable hostnames (e.g. a DNS
    name that resolves elsewhere) are conservatively treated as non-loopback.
    """
    h = (host or "").strip()
    if h.lower() in ("localhost", "::1", "ip6-localhost"):
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def load_config(config_path: str | Path | None = None) -> Settings:
    """Load settings, merging a YAML config file if present.

    Environment variables (``BLACKBOOK_*``) always win over the YAML file.
    """
    # Resolve which config file to read.
    env_cfg = os.environ.get("BLACKBOOK_CONFIG")
    if config_path is not None:
        cfg_path = expand_path(config_path)
    elif env_cfg:
        cfg_path = expand_path(env_cfg)
    else:
        cfg_path = expand_path("~/.blackbook/config.yaml")

    data: dict = {}
    if cfg_path.is_file():
        yaml = YAML(typ="safe", pure=True)
        loaded = yaml.load(cfg_path.read_text())
        if isinstance(loaded, dict):
            data = loaded

    # pydantic-settings reads env vars on top of the init kwargs we pass.
    settings = Settings(**data)
    # Ensure ~ expansion for home and any filesystem source directories.
    settings.home = expand_path(settings.home)
    for src in settings.sources:
        if src.directory:
            src.directory = str(expand_path(src.directory))
    return settings


def ensure_dirs(settings: Settings) -> None:
    """Create the directories BlackBook needs at runtime."""
    settings.home.mkdir(parents=True, exist_ok=True)
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
