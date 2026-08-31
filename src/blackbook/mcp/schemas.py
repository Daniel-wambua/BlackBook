"""Pydantic schemas for MCP tool inputs/outputs.

These give the MCP server strict, validated inputs and well-typed outputs.
They also document the response model that separates observed / documented /
inferred material and carries provenance on every claim.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

SearchMode = Literal["keyword", "semantic", "hybrid", "case_similarity", "technique"]
Detail = Literal["brief", "standard", "deep"]


class SearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    sources: list[str] | None = None
    categories: list[str] | None = None
    platform: str | None = None
    techniques: list[str] | None = None
    mode: SearchMode = "hybrid"
    limit: int = Field(default=8, ge=1, le=50)
    detail: Detail = "standard"


class SourceRef(BaseModel):
    """A verifiable reference to an indexed chunk."""

    chunk_id: int
    doc_id: int
    title: str
    source: str
    source_name: str
    authority: str
    url: str | None = None
    path: str | None = None
    page: int | None = None
    section_path: list[str] = Field(default_factory=list)


class SearchResultItem(BaseModel):
    title: str
    source: str
    source_name: str
    authority: str
    relevance: float
    snippet: str
    ref: SourceRef


class SearchOutput(BaseModel):
    query: str
    mode: str
    sources_searched: list[str]
    count: int
    results: list[SearchResultItem]
    note: str = ""


class GetSourceInput(BaseModel):
    chunk_id: int | None = None
    doc_id: int | None = None
    source: str | None = None
    document: str | None = None  # external_id within the source
    title_contains: str | None = None
    section: str | None = None
    max_excerpts: int = Field(default=5, ge=1, le=20)


class SourceExcerptItem(BaseModel):
    ref: SourceRef
    ordinal: int
    text: str


class SourceOutput(BaseModel):
    count: int
    excerpts: list[SourceExcerptItem]
    note: str = ""


# -- Phase 4: knowledge graph tools -----------------------------------------


class EvidenceRef(BaseModel):
    """A document-level citation backing a graph relationship.

    Graph edges are derived from a specific document, so every non-structural
    edge carries the document it came from. Fields are optional because a
    structural edge's evidence may have been pruned (``ON DELETE SET NULL``),
    in which case the edge is still reported but without a dangling citation.
    """

    doc_id: int | None = None
    title: str | None = None
    source: str | None = None
    source_name: str | None = None
    authority: str | None = None
    url: str | None = None
    external_id: str | None = None


class GraphRef(BaseModel):
    """A neighbour of a graph entity, with the edge's confidence/provenance."""

    name: str
    entity_type: str
    predicate: str
    confidence: float
    inferred: bool
    evidence: EvidenceRef | None = None


class TechniqueInput(BaseModel):
    technique: str = Field(min_length=1, max_length=200)
    sources: list[str] | None = None
    limit: int = Field(default=6, ge=1, le=20)


class TechniqueOutput(BaseModel):
    technique: str            # canonical term, or the caller's input echoed back
    resolved: bool            # True when it mapped to a controlled vocabulary term
    in_graph: bool            # True when a graph entity exists for it
    attack_id: str | None = None  # curated MITRE ATT&CK ID, None when unmapped
    # Filled from the indexed MITRE ATT&CK source when the technique's
    # document exists there; empty otherwise.
    tactics: list[str] = Field(default_factory=list)
    platforms: list[str] = Field(default_factory=list)
    mitre_url: str | None = None
    documented_by: list[GraphRef] = Field(default_factory=list)   # sources
    related_tools: list[GraphRef] = Field(default_factory=list)
    related_services: list[GraphRef] = Field(default_factory=list)
    demonstrated_in: list[GraphRef] = Field(default_factory=list)  # writeups
    references: list[SearchResultItem] = Field(default_factory=list)  # real excerpts
    note: str = ""


class CaseSearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    sources: list[str] | None = None
    platform: str | None = None
    techniques: list[str] | None = None
    limit: int = Field(default=6, ge=1, le=20)


class CaseItem(BaseModel):
    title: str
    source: str
    source_name: str
    authority: str
    relevance: float
    snippet: str
    ref: SourceRef
    techniques: list[str] = Field(default_factory=list)  # from graph, if built


class CaseSearchOutput(BaseModel):
    query: str
    count: int
    results: list[CaseItem]
    note: str = ""


# -- Phase 5: research packet ------------------------------------------------


class ResearchInput(BaseModel):
    observation: str = Field(min_length=1, max_length=4000)
    sources: list[str] | None = None
    platform: str | None = None
    techniques: list[str] | None = None
    limit: int = Field(default=6, ge=1, le=20)
    include_cases: bool = True


class ResearchSignals(BaseModel):
    """Controlled-vocabulary terms detected in the observation text."""

    services: list[str] = Field(default_factory=list)
    techniques: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)


class TechniqueBrief(BaseModel):
    """A compact, graph-backed note for one detected technique."""

    technique: str            # canonical term
    resolved: bool            # mapped to the controlled vocabulary
    in_graph: bool            # a graph entity exists for it
    attack_id: str | None = None  # curated MITRE ATT&CK ID, None when unmapped
    documented_by: list[GraphRef] = Field(default_factory=list)


class ResearchOutput(BaseModel):
    observation: str
    signals: ResearchSignals
    techniques: list[TechniqueBrief] = Field(default_factory=list)
    references: list[SearchResultItem] = Field(default_factory=list)
    related_cases: list[CaseItem] = Field(default_factory=list)
    note: str = ""


# -- Phase 5: investigation context (local case layer) ----------------------

CaseAction = Literal["create", "add", "update_observation", "get", "list", "export"]
ObservationKind = Literal["observation", "finding", "hypothesis", "technique", "note"]
ObservationStatus = Literal["open", "tested", "confirmed", "refuted", "resolved"]

# Bound on the serialized size of a case's free-form ``meta`` dict. It is
# user-supplied JSON written into SQLite over an (optionally network-exposed)
# write path, so it gets a hard ceiling rather than a free pass.
_META_MAX_BYTES = 16 * 1024


def _check_meta(v: dict | None) -> dict | None:
    import json

    if v is None:
        return None
    if len(json.dumps(v, default=str)) > _META_MAX_BYTES:
        raise ValueError(
            f"meta is too large (limit {_META_MAX_BYTES} bytes serialized)"
        )
    return v


class ContextInput(BaseModel):
    action: CaseAction
    case: str | None = Field(default=None, max_length=200)
    target: str = Field(default="", max_length=500)
    platform: str = Field(default="", max_length=100)
    kind: ObservationKind = "observation"
    text: str | None = Field(default=None, max_length=4000)
    obs_id: int | None = None
    status: ObservationStatus | None = None
    meta: dict | None = None

    @field_validator("meta")
    @classmethod
    def _bound_meta(cls, v: dict | None) -> dict | None:
        return _check_meta(v)


class ObservationItem(BaseModel):
    obs_id: int
    kind: str
    text: str
    status: str
    created_at: str | None = None


class CaseState(BaseModel):
    case_id: int
    name: str
    target: str = ""
    platform: str = ""
    created_at: str | None = None
    updated_at: str | None = None
    observations: list[ObservationItem] = Field(default_factory=list)


class CaseSummary(BaseModel):
    case_id: int
    name: str
    target: str = ""
    platform: str = ""
    observation_count: int = 0
    updated_at: str | None = None


class ContextOutput(BaseModel):
    action: str
    ok: bool
    case: CaseState | None = None
    cases: list[CaseSummary] = Field(default_factory=list)
    # Populated by action='export': the case rendered as portable Markdown.
    # The server returns it in-band; use the CLI (`blackbook case export`) to
    # write it to a file.
    markdown: str = ""
    note: str = ""
