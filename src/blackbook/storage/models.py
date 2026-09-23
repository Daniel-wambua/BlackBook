"""Core data models for BlackBook's storage layer.

These are plain dataclasses that map 1:1 to SQLite rows. They carry no
behaviour beyond construction; persistence lives in
:mod:`blackbook.storage.database`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Source:
    """A configured knowledge source (HackTricks, 0xdf, local PDFs, ...)."""

    source_id: str
    name: str
    authority: str = "unknown"  # official | trusted | user | unknown
    enabled: bool = True
    source_type: str = "filesystem"  # git | website | filesystem
    url: str | None = None
    last_fetched: str | None = None
    version: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """A single logical document within a source (a HackTricks page, an 0xdf
    writeup, a PDF)."""

    source_id: str
    external_id: str  # stable ID within the source (path, slug, URL)
    title: str
    doc_id: int | None = None  # set on insert
    url: str | None = None
    path: str | None = None  # local path for filesystem sources
    content_hash: str = ""  # sha256 of full document text (dedup / change detection)
    metadata: dict[str, Any] = field(default_factory=dict)
    categories: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    """A retrievable unit of a document, aligned to semantic boundaries."""

    doc_id: int
    ordinal: int  # position within the document
    text: str
    chunk_id: int | None = None  # set on insert (rowid)
    section_path: list[str] = field(default_factory=list)
    page: int | None = None
    token_estimate: int = 0
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Entity:
    """A node in the knowledge graph (Technique, Tool, Protocol, ...)."""

    name: str
    entity_type: str  # technique | tool | protocol | service | os | platform | ...
    entity_id: int | None = None
    description: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Relationship:
    """An edge in the knowledge graph.

    ``support`` is how many documents back the claim, and it means different
    things for the two kinds of edge. A *documentary* edge (a writeup is
    published by its source, a writeup runs on an OS) is a fact about a single
    document, so its support is 1 and ``evidence_doc_id`` names that document. A
    *statistical* edge (two terms co-occur) is one claim with many possible
    witnesses: it is stored once per pair, ``support`` counts the distinct
    documents that mentioned both, and ``evidence_doc_id`` names the lowest of
    them as a representative citation. See
    ``blackbook.knowledge.graph._COOCCURRENCE_PREDICATES``.
    """

    subject_id: int
    predicate: str  # related_to | requires | associated_with | ...
    object_id: int
    rel_id: int | None = None
    evidence_doc_id: int | None = None
    confidence: float = 1.0
    inferred: bool = False
    support: int = 1


@dataclass
class CaseObservation:
    """A single observation recorded on an investigation case."""

    case_id: int
    kind: str  # observation | finding | hypothesis | technique | note
    text: str
    obs_id: int | None = None
    status: str = "open"  # open | tested | confirmed | refuted | resolved
    created_at: str | None = None


@dataclass
class Case:
    """An investigation's knowledge state."""

    name: str
    case_id: int | None = None
    target: str = ""
    platform: str = ""
    created_at: str | None = None
    updated_at: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryLogEntry:
    """One question put to BlackBook, and what came back.

    Diagnostic, not corpus data: it records the phrasing, which sources were
    searched, how many results came back and how strong the best one was. The
    most useful field is ``result_count``: a zero means the corpus could not
    answer that phrasing, which is the one signal about retrieval quality that
    the index itself cannot report.
    """

    tool: str
    query: str
    mode: str = ""
    sources: list[str] = field(default_factory=list)
    result_count: int = 0
    top_score: float | None = None
    latency_ms: float = 0.0
    backend: str = ""
    degraded: bool = False
    query_id: int | None = None  # set on insert
    created_at: str | None = None
