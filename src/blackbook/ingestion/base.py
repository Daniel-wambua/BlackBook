"""Ingestion adapter interface.

Each knowledge source (HackTricks, 0xdf, local PDFs, future sources) provides
an adapter implementing :class:`SourceAdapter`. The adapter is responsible for
*fetching* and *parsing* raw source material into a normalized stream of
:class:`ParsedDocument` objects. Persistence, dedup, and indexing are handled
centrally by :mod:`blackbook.ingestion.pipeline`.

Adapters never talk to the database directly and never execute shell commands.
Fetching is limited to the configured source URL/directory.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator

from blackbook.config import SourceConfig
from blackbook.retrieval.chunking import RawChunk


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class ParsedDocument:
    """A source-agnostic parsed document ready for chunking + indexing."""

    external_id: str  # stable ID within the source (path, slug, url)
    title: str
    url: str | None = None
    path: str | None = None
    categories: list[str] = field(default_factory=list)
    # Full raw text for hashing/change-detection; chunks are the indexed form.
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    # If the adapter can pre-chunk (preserving structure), provide chunks.
    # Otherwise the pipeline will chunk ``text`` generically.
    chunks: list[RawChunk] = field(default_factory=list)


class IngestStats:
    """Counters reported by an ingestion run."""

    def __init__(self) -> None:
        self.discovered = 0
        self.parsed = 0
        self.skipped_unchanged = 0
        self.errors = 0
        self.chunks_written = 0
        self.error_messages: list[str] = []

    def as_dict(self) -> dict[str, Any]:
        return {
            "discovered": self.discovered,
            "parsed": self.parsed,
            "skipped_unchanged": self.skipped_unchanged,
            "errors": self.errors,
            "chunks_written": self.chunks_written,
            "error_messages": list(self.error_messages),
        }


class SourceAdapter(ABC):
    """Abstract base class for a knowledge-source adapter."""

    #: The ``SourceConfig.id`` this adapter handles.
    source_id: str = ""

    def __init__(self, config: SourceConfig, raw_dir: str | None = None):
        self.config = config
        self.raw_dir = raw_dir

    # -- fetching ----------------------------------------------------------

    @abstractmethod
    def fetch(self, force: bool = False) -> None:
        """Acquire/refresh the raw source material (clone, download, scan).

        Should be a no-op when material is already present and ``force`` is
        false. Must not execute shell commands; use git via subprocess-free
        means (e.g. dulwich) or plain HTTP.
        """

    @abstractmethod
    def iter_documents(self) -> Iterator[ParsedDocument]:
        """Yield parsed documents from the fetched material."""

    # -- helpers -----------------------------------------------------------

    def stats(self) -> IngestStats:
        return IngestStats()
