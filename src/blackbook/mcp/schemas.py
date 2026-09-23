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
    # Which retrieval backend(s) actually produced these results: "lexical"
    # (FTS5/BM25), "semantic" (dense vectors), "lexical+semantic" (merged), or
    # "none" when nothing ran. Reported so a caller can tell a genuine hybrid
    # response from a silently lexical-only one.
    backend: str = "lexical"
    # True when semantic retrieval was requested but did not contribute (it is
    # disabled, its dependency is missing, or the vector index is empty). The
    # results are then lexical-only, and ``note`` says why.
    degraded: bool = False
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
    # How many documents back this edge. Always 1 for a documentary edge; for the
    # collapsed co-occurrence predicates it is the number of documents that
    # mentioned both terms, which is the only signal separating a claim made
    # everywhere from one made once.
    support: int = 1


EntityType = Literal["technique", "tool", "service", "os", "writeup", "source"]
GraphDirection = Literal["out", "in", "both"]


class GraphTraversalInput(BaseModel):
    """Walk the knowledge graph out from one entity.

    ``depth`` is capped at 3 because the graph is dense: a single technique can
    carry hundreds of co-occurrence edges, so a second hop is already a large
    neighbourhood and a deeper walk over the real corpus would be unreadable
    rather than more informative. ``limit`` bounds how many neighbours each node
    expands, and ``max_nodes`` bounds the whole result; both are reported as
    truncation rather than applied silently.
    """

    entity: str = Field(min_length=1, max_length=200)
    entity_type: EntityType | None = None
    depth: int = Field(default=1, ge=1, le=3)
    direction: GraphDirection = "both"
    predicates: list[str] | None = None
    limit: int = Field(default=25, ge=1, le=100)
    max_nodes: int = Field(default=60, ge=1, le=300)


class GraphNode(BaseModel):
    """One entity reached by the walk, with how it was reached."""

    entity_id: int
    name: str
    entity_type: str
    description: str = ""
    hop: int      # 0 for the start entity
    via: str | None = None  # predicate that first reached it, None for the start


class GraphEdge(BaseModel):
    """One edge walked, oriented as stored (subject -> predicate -> object)."""

    subject: str
    predicate: str
    object: str
    confidence: float
    inferred: bool
    support: int = 1
    evidence: EvidenceRef | None = None


class GraphTraversalOutput(BaseModel):
    entity: str                    # the caller's input, echoed
    resolved: str | None = None    # the stored entity name, None when not found
    entity_type: str | None = None
    found: bool
    depth: int
    direction: str
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    # True when the walk stopped short of the full neighbourhood, for either of
    # the two reasons ``note`` distinguishes: per-node ``limit`` or ``max_nodes``.
    truncated: bool = False
    candidates: list[str] = Field(default_factory=list)  # near misses when unresolved
    note: str = ""


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


# -- bug bounty workflow tools ----------------------------------------------


class HuntPlanInput(BaseModel):
    observation: str = Field(min_length=1, max_length=4000)
    target: str = Field(default="", max_length=500)
    platform: str | None = Field(default=None, max_length=100)
    sources: list[str] | None = None
    techniques: list[str] | None = None
    limit: int = Field(default=6, ge=1, le=20)


class HuntPlanItem(BaseModel):
    title: str
    category: str
    rationale: str
    validation_focus: list[str] = Field(default_factory=list)
    references: list[SearchResultItem] = Field(default_factory=list)


class HuntPlanOutput(BaseModel):
    observation: str
    target: str = ""
    signals: ResearchSignals
    plans: list[HuntPlanItem] = Field(default_factory=list)
    note: str = ""


class FindingReviewInput(BaseModel):
    finding: str = Field(min_length=1, max_length=4000)
    case: str | None = Field(default=None, max_length=200)
    sources: list[str] | None = None
    platform: str | None = Field(default=None, max_length=100)
    limit: int = Field(default=6, ge=1, le=20)


class FindingReviewOutput(BaseModel):
    finding: str
    case: str | None = None
    signals: ResearchSignals
    evidence_status: str
    observed_evidence: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    references: list[SearchResultItem] = Field(default_factory=list)
    severity_guidance: list[SearchResultItem] = Field(default_factory=list)
    note: str = ""


class ReportDraftInput(BaseModel):
    case: str = Field(min_length=1, max_length=200)
    sources: list[str] | None = None
    limit: int = Field(default=8, ge=1, le=20)


class ReportDraftOutput(BaseModel):
    case: str
    title: str
    summary: str
    observed_evidence: list[str] = Field(default_factory=list)
    reproduction_steps: list[str] = Field(default_factory=list)
    impact: str
    remediation: str
    severity_basis: list[SearchResultItem] = Field(default_factory=list)
    references: list[SearchResultItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class KnowledgeSourceInput(BaseModel):
    source: str | None = Field(default=None, max_length=100)


class KnowledgeSourceStatus(BaseModel):
    id: str
    name: str
    enabled: bool
    authority: str
    source_type: str
    url: str | None = None
    indexed_documents: int = 0
    indexed_chunks: int = 0
    # Freshness: when the source was last pulled successfully, and at which
    # revision. Both are None for a source that has never been fetched by an
    # ingest run (a corpus seeded another way, or a source not yet run). The
    # timestamp is UTC; ``version`` is a commit for repository-backed sources
    # and None for the ones with no revision to speak of.
    last_fetched: str | None = None
    version: str | None = None


class SemanticStatus(BaseModel):
    """Whether the optional dense-vector backend can actually contribute.

    Semantic retrieval has two independent ways to be inert: it is off by
    default (``embeddings.enabled``), and even when on it does nothing until
    ``blackbook embed`` has written vectors. Both used to be invisible - a
    ``mode="hybrid"`` search would quietly return lexical results. This is
    reported by ``knowledge_sources`` so the state is inspectable.

    Computed from config plus a single ``COUNT(*)``: it never loads the
    embedding model, so asking costs nothing.
    """

    enabled: bool
    model: str
    vectors: int = 0
    # True only when the backend would actually contribute to a search.
    ready: bool = False
    note: str = ""


class KnowledgeSourcesOutput(BaseModel):
    count: int
    sources: list[KnowledgeSourceStatus] = Field(default_factory=list)
    # None when the caller did not ask for global status (a single-source
    # lookup); otherwise the semantic backend's readiness.
    semantic: SemanticStatus | None = None
    note: str = ""


class KnowledgeCompareInput(BaseModel):
    topic: str = Field(min_length=1, max_length=2000)
    sources: list[str] = Field(min_length=2, max_length=10)
    platform: str | None = Field(default=None, max_length=100)
    limit: int = Field(default=4, ge=1, le=10)


class KnowledgeCompareView(BaseModel):
    source: str
    source_name: str
    authority: str
    results: list[SearchResultItem] = Field(default_factory=list)


class KnowledgeCompareOutput(BaseModel):
    topic: str
    sources_compared: list[str]
    views: list[KnowledgeCompareView] = Field(default_factory=list)
    shared_terms: list[str] = Field(default_factory=list)
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
