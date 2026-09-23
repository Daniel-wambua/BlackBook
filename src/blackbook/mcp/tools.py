"""MCP tool implementations.

The registered tool surface:

* ``knowledge_search`` — source-grounded search across the indexed corpus
* ``knowledge_source`` — resolve a reference to the exact source excerpt
* ``knowledge_technique`` — a graph-enhanced, cited dossier for a technique
* ``knowledge_graph`` — traverse the graph outward from one entity
* ``knowledge_case_search`` — find hands-on writeups similar to a situation
* ``knowledge_research`` — turn a free-text observation into a source-grounded
  research packet (detected signals, technique briefs, cited references, cases)
* ``knowledge_context`` — manage local investigation state (cases + observations)
* ``knowledge_hunt_plan`` — build a cited, non-executing validation plan
* ``knowledge_finding_review`` — review evidence gaps and severity guidance
* ``knowledge_report_draft`` — draft from local case observations
* ``knowledge_sources`` — inspect configured sources and index counts
* ``knowledge_compare`` — compare independent retrieval views by source

All search/technique/research tools are read-only over the index and always
return structured, provenance-tagged output; every citation resolves to a real
indexed chunk. ``knowledge_context`` reads and writes only the *local*,
user-authored case layer — it never touches, executes against, or fetches from
any external system.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, deque

from blackbook.config import Settings
from blackbook.knowledge import query_log
from blackbook.knowledge.attack_index import AttackIndex
from blackbook.knowledge.case_export import build_case_state, render_case_markdown
from blackbook.knowledge.graph import E_TECHNIQUE, E_WRITEUP, P_DEMONSTRATED_IN
from blackbook.knowledge.sources import find_document, get_chunk_excerpt, list_document_chunks
from blackbook.knowledge.vocab import attack_id, extract_signals, resolve_technique
from blackbook.mcp.schemas import (
    CaseItem,
    CaseSearchInput,
    CaseSearchOutput,
    CaseState,
    CaseSummary,
    ContextInput,
    ContextOutput,
    EvidenceRef,
    FindingReviewInput,
    FindingReviewOutput,
    GetSourceInput,
    GraphEdge,
    GraphNode,
    GraphRef,
    GraphTraversalInput,
    GraphTraversalOutput,
    HuntPlanInput,
    HuntPlanItem,
    HuntPlanOutput,
    KnowledgeCompareInput,
    KnowledgeCompareOutput,
    KnowledgeCompareView,
    KnowledgeSourceInput,
    KnowledgeSourceStatus,
    KnowledgeSourcesOutput,
    SemanticStatus,
    ReportDraftInput,
    ReportDraftOutput,
    ResearchInput,
    ResearchOutput,
    ResearchSignals,
    SearchInput,
    SearchOutput,
    SearchResultItem,
    SourceExcerptItem,
    SourceOutput,
    SourceRef,
    TechniqueBrief,
    TechniqueInput,
    TechniqueOutput,
)
from blackbook.retrieval import HybridRetriever, SearchDiagnostics, SearchResult
from blackbook.storage.database import Database
from blackbook.storage.models import Case, CaseObservation

log = logging.getLogger(__name__)


def _load_doc_metadata(doc: dict) -> dict:
    """Parse a document row's stored metadata blob, tolerating garbage."""
    raw = doc.get("metadata")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            return {}
    return {}


class KnowledgeTools:
    """Holds shared state (db, settings, retriever) for the MCP tools."""

    def __init__(self, db: Database, settings: Settings):
        self.db = db
        self.settings = settings
        self.retriever = HybridRetriever(db, settings)
        # Built on first use rather than here: constructing it costs a query and
        # a full pass over the ATT&CK source, and a server that never answers a
        # technique question should not pay that at start-up. Held for the life
        # of the instance, which under streamable-http is the life of the
        # process, so the ATT&CK lookup is not rebuilt per request.
        self._attack_index: AttackIndex | None = None

    def _attack_ids(self, term: str) -> str | None:
        """ATT&CK ID for a term, curated map first, indexed corpus second.

        The curated map in :mod:`blackbook.knowledge.vocab` encodes our own
        judgement about which ATT&CK technique one of *our* vocabulary terms
        corresponds to. It deliberately holds only the terms someone has
        actually decided on, so it returns ``None`` for a caller that names a
        real ATT&CK technique by name ("Scheduled Task/Job") or by ID
        ("T1566.001") and simply never wrote it down here.

        Falling back to :class:`AttackIndex` closes that gap from the other
        direction: it resolves only names and IDs that exist in the indexed
        ATT&CK source, so it cannot invent a mapping, and it returns ``None``
        rather than a guess when ATT&CK gives one name to several techniques.
        """
        curated = attack_id(term)
        if curated is not None:
            return curated
        if self._attack_index is None:
            self._attack_index = AttackIndex(self.db)
        return self._attack_index.resolve(term)

    # -- knowledge_search ---------------------------------------------------

    def knowledge_search(self, inp: SearchInput) -> SearchOutput:
        requested, source_ids = self._resolve_sources(inp.sources)
        # Techniques resolve through the controlled vocabulary: resolved terms
        # still join the query (recall) *and* bias reranking toward chunks whose
        # title/section names them (precision); unresolved ones remain plain
        # query terms so a non-vocab technique is still searched.
        query, resolved, unresolved = self._merge_techniques(inp.query, inp.techniques)

        diag = SearchDiagnostics()
        with query_log.timed() as timing:
            results: list[SearchResult] = self.retriever.search(
                query,
                mode=inp.mode,
                source_ids=source_ids,
                platform=inp.platform,
                categories=inp.categories,
                techniques=resolved,
                limit=inp.limit,
                diagnostics=diag,
            )
        items = [self._to_item(r, inp.detail) for r in results]
        notes: list[str] = []
        if source_ids == []:
            notes.append(
                f"No enabled source matches {requested}; nothing was searched. "
                "Check the source ID with `blackbook sources`."
            )
        elif not items:
            notes.append("No matching material found in the selected sources.")
        # A mode="semantic" request degrades to lexical rather than returning an
        # empty list; say so instead of letting the caller assume vectors were
        # used. This is the silent-empty path the facade now closes.
        if diag.degraded:
            notes.append(self._degradation_note(diag))
        if unresolved:
            notes.append(
                "Techniques not in the controlled vocabulary were searched as "
                f"plain terms: {', '.join(unresolved)}."
            )
        # Logged before returning, and after the result is fully assembled, so
        # the entry describes the answer the caller actually received rather
        # than the intermediate state the retriever handed back.
        query_log.record(
            self.db,
            self.settings,
            tool="knowledge_search",
            query=inp.query,
            mode=inp.mode,
            sources=source_ids if source_ids is not None else ["all"],
            result_count=len(items),
            top_score=items[0].relevance if items else None,
            latency_ms=timing["latency_ms"],
            backend=diag.backend,
            degraded=diag.degraded,
        )
        return SearchOutput(
            query=inp.query,
            mode=inp.mode,
            sources_searched=source_ids if source_ids is not None else ["all"],
            count=len(items),
            results=items,
            backend=diag.backend,
            degraded=diag.degraded,
            note=" ".join(n for n in notes if n).strip(),
        )

    @staticmethod
    def _degradation_note(diag: SearchDiagnostics) -> str:
        """Explain why a semantic request came back lexical-only."""
        if diag.semantic_error:
            return (
                f"Semantic retrieval could not start ({diag.semantic_error}); "
                "these results are lexical (BM25) only."
            )
        if not diag.semantic_enabled:
            return (
                "Semantic retrieval is disabled (embeddings.enabled is false); "
                "these results are lexical (BM25) only. Enable embeddings and "
                "run `blackbook embed` to index vectors."
            )
        if diag.semantic_vectors <= 0:
            return (
                "Semantic retrieval is enabled but the vector index is empty "
                "(0 embeddings stored); these results are lexical (BM25) only. "
                "Run `blackbook embed` to build it."
            )
        return (
            "Semantic retrieval returned no candidates for this query; these "
            "results are lexical (BM25) only."
        )

    @staticmethod
    def _merge_techniques(
        query: str, techniques: list[str] | None
    ) -> tuple[str, list[str], list[str]]:
        """Fold ``techniques`` into ``query``, splitting resolved/unresolved.

        Returns ``(query, resolved, unresolved)``: every technique term joins
        the query text for recall, while the vocabulary-resolved ones are
        returned separately so the caller can pass them on as a reranking bias
        (they favour chunks whose title/section names the technique).
        """
        resolved: list[str] = []
        unresolved: list[str] = []
        for t in techniques or []:
            canonical = resolve_technique(t)
            if canonical:
                if canonical not in resolved:
                    resolved.append(canonical)
            else:
                unresolved.append(t)
        extra = " ".join(resolved + unresolved)
        merged = f"{query} {extra}".strip() if extra else query
        return merged, resolved, unresolved

    def _resolve_sources(
        self, requested: list[str] | None
    ) -> tuple[list[str], list[str] | None]:
        """Resolve a source filter, keeping the caller's intent visible.

        Returns ``(requested, resolved)`` where ``resolved`` is ``None`` for
        "every enabled source" and possibly ``[]`` when nothing matched —
        which callers surface as an explicit note rather than silently
        searching everything.
        """
        return requested or [], self.settings.source_ids(requested)

    def _to_item(self, r: SearchResult, detail: str) -> SearchResultItem:
        snippet = r.snippet if detail != "deep" else r.text[:1200]
        return SearchResultItem(
            title=r.title,
            source=r.source_id,
            source_name=r.source_name,
            authority=r.authority,
            relevance=round(r.score, 4),
            snippet=snippet,
            ref=SourceRef(
                chunk_id=r.chunk_id,
                doc_id=r.doc_id,
                title=r.title,
                source=r.source_id,
                source_name=r.source_name,
                authority=r.authority,
                url=r.url,
                path=r.path,
                page=r.page,
                section_path=r.section_path,
            ),
        )

    # -- knowledge_source ---------------------------------------------------

    def knowledge_source(self, inp: GetSourceInput) -> SourceOutput:
        excerpts: list[SourceExcerptItem] = []

        if inp.chunk_id is not None:
            ex = get_chunk_excerpt(self.db, inp.chunk_id)
            if ex:
                excerpts.append(self._excerpt_item(ex))
            return SourceOutput(
                count=len(excerpts),
                excerpts=excerpts,
                note="" if excerpts else "chunk_id not found in the index.",
            )

        # Resolve a document, then return its chunks (optionally section-filtered).
        doc = None
        if inp.doc_id is not None:
            doc = find_document(self.db, doc_id=inp.doc_id)
        elif inp.source and inp.document:
            doc = find_document(self.db, source_id=inp.source, external_id=inp.document)
        elif inp.title_contains:
            doc = find_document(self.db, title_like=inp.title_contains)

        if not doc:
            if inp.source and not inp.document:
                # The caller named a source but no document — a source alone
                # can't identify an excerpt. Point at real documents from that
                # source so the next call has something concrete to use.
                sample = [
                    d["title"]
                    for d in list(self.db.iter_documents([inp.source]))[:5]
                ]
                hint = (
                    f"'{inp.source}' is a source, not a document — provide its "
                    "`document` (external_id) or a `title_contains`. Indexed "
                    f"documents from this source include: "
                    + (", ".join(sample) if sample else "(none indexed)")
                )
                return SourceOutput(count=0, excerpts=[], note=hint)
            return SourceOutput(
                count=0,
                excerpts=[],
                note="Document not found. Provide chunk_id, or source+document, or title_contains.",
            )

        chunks = list_document_chunks(self.db, int(doc["doc_id"]))
        if inp.section:
            needle = inp.section.lower()
            chunks = [c for c in chunks if any(needle in s.lower() for s in c.section_path)]
        for ex in chunks[: inp.max_excerpts]:
            excerpts.append(self._excerpt_item(ex))
        note = ""
        if not excerpts:
            note = "Document found but no matching sections."
        return SourceOutput(count=len(excerpts), excerpts=excerpts, note=note)

    @staticmethod
    def _excerpt_item(ex) -> SourceExcerptItem:
        return SourceExcerptItem(
            ref=SourceRef(
                chunk_id=ex.chunk_id,
                doc_id=ex.doc_id,
                title=ex.title,
                source=ex.source_id,
                source_name=ex.source_name,
                authority=ex.authority,
                url=ex.url,
                path=ex.path,
                page=ex.page,
                section_path=ex.section_path,
            ),
            ordinal=ex.ordinal,
            text=ex.text,
        )

    # -- knowledge_technique (Phase 4) -------------------------------------

    def knowledge_technique(self, inp: TechniqueInput) -> TechniqueOutput:
        """Assemble a structured dossier for a technique.

        The graph *enhances* this dossier (which sources document it, which
        tools/services/writeups it associates with) but never gates it: even
        with an empty graph the tool still returns real, cited excerpts from a
        technique-biased search. Nothing here is fabricated — graph neighbours
        come from evidence-linked edges and references come from the index.
        """
        source_ids = self.settings.source_ids(inp.sources)
        canonical = resolve_technique(inp.technique)
        term = canonical or inp.technique
        if source_ids == []:
            return TechniqueOutput(
                technique=term,
                resolved=canonical is not None,
                in_graph=False,
                note=(
                    f"No enabled source matches {inp.sources}; nothing was searched. "
                    "Check the source ID with `blackbook sources`."
                ),
            )

        documented_by: list[GraphRef] = []
        related_tools: list[GraphRef] = []
        related_services: list[GraphRef] = []
        demonstrated_in: list[GraphRef] = []

        ent = self.db.get_entity(term, "technique") if canonical else None
        in_graph = ent is not None
        if ent is not None:
            rels = self.db.entity_relationships(int(ent["entity_id"]))
            bucket = {
                "documented_by": documented_by,
                "uses": related_tools,
                "targets": related_services,
                "demonstrated_in": demonstrated_in,
            }
            for rel in rels:
                target = bucket.get(rel["predicate"])
                if target is not None and rel["direction"] == "out":
                    target.append(self._graph_ref(rel))

        # Real, cited excerpts — always available, graph or not.
        results = self.retriever.search(
            term,
            mode="technique",
            source_ids=source_ids,
            platform=None,
            categories=None,
            limit=inp.limit,
        )
        references = [self._to_item(r, "standard") for r in results]

        # MITRE ATT&CK enrichment: when the technique has an ATT&CK ID and
        # the ATT&CK source is indexed, ground the dossier in the real
        # technique record (tactics, platforms, official URL + a citation).
        # The lookup falls back to the indexed ATT&CK corpus, so naming a
        # technique ATT&CK knows ("Scheduled Task/Job", "T1566.001") is enough
        # to get the same enrichment as a term from our own vocabulary.
        aid = self._attack_ids(term)
        tactics: list[str] = []
        platforms: list[str] = []
        mitre_url = None
        mitre_ref = None
        if aid:
            doc = self.db.get_document_by_external("attack", aid)
            if doc:
                meta = _load_doc_metadata(doc)
                tactics = list(meta.get("tactics") or [])
                platforms = list(meta.get("platforms") or [])
                mitre_url = doc.get("url")
                chunks = self.db.document_chunks(int(doc["doc_id"]))
                if chunks:
                    first = chunks[0]
                    section = first.get("section_path") or "[]"
                    if isinstance(section, str):
                        try:
                            section = json.loads(section)
                        except ValueError:
                            section = []
                    mitre_ref = SearchResultItem(
                        title=doc.get("title") or term,
                        source="attack",
                        source_name="MITRE ATT&CK",
                        authority="official",
                        relevance=1.0,
                        snippet=first["text"][:400],
                        ref=SourceRef(
                            chunk_id=first["chunk_id"],
                            doc_id=int(doc["doc_id"]),
                            title=doc.get("title") or term,
                            source="attack",
                            source_name="MITRE ATT&CK",
                            authority="official",
                            url=mitre_url,
                            path=doc.get("path"),
                            page=None,
                            section_path=section,
                        ),
                    )
        if mitre_ref is not None and not any(
            r.ref.source == "attack" for r in references
        ):
            references.insert(0, mitre_ref)

        note = ""
        if not in_graph and not references:
            note = (
                "No graph entity or indexed references for this technique. "
                "Run 'blackbook graph build' after ingesting, and confirm the "
                "term is in the controlled vocabulary."
            )
        elif not in_graph:
            note = "Not in the knowledge graph yet; showing indexed references only."
        return TechniqueOutput(
            technique=term,
            resolved=canonical is not None,
            in_graph=in_graph,
            attack_id=aid,
            tactics=tactics,
            platforms=platforms,
            mitre_url=mitre_url,
            documented_by=documented_by,
            related_tools=related_tools,
            related_services=related_services,
            demonstrated_in=demonstrated_in,
            references=references,
            note=note,
        )

    @staticmethod
    def _evidence_ref(rel: dict) -> EvidenceRef | None:
        """Turn a joined relationship row's evidence columns into a citation.

        Shared by every graph-facing output so there is one place that decides
        what a citation looks like. Returns ``None`` rather than an empty ref
        when the edge carries no document, which is the honest answer for a
        structural edge whose evidence was pruned.
        """
        if rel.get("evidence_doc_id") is None:
            return None
        return EvidenceRef(
            doc_id=rel.get("evidence_doc_id"),
            title=rel.get("evidence_title"),
            source=rel.get("evidence_source_id"),
            source_name=rel.get("evidence_source_name"),
            authority=rel.get("evidence_authority"),
            url=rel.get("evidence_url"),
            external_id=rel.get("evidence_external_id"),
        )

    @classmethod
    def _graph_ref(cls, rel: dict) -> GraphRef:
        return GraphRef(
            name=rel["other_name"],
            entity_type=rel["other_type"],
            predicate=rel["predicate"],
            confidence=round(float(rel["confidence"]), 4),
            inferred=bool(rel["inferred"]),
            evidence=cls._evidence_ref(rel),
            support=int(rel.get("support") or 1),
        )

    # -- knowledge_graph (graph traversal) ---------------------------------

    def knowledge_graph(self, inp: GraphTraversalInput) -> GraphTraversalOutput:
        """Walk the graph outward from one entity, up to ``inp.depth`` hops.

        ``knowledge_technique`` answers a fixed question about one technique
        (which sources document it, which tools it uses). This answers the
        open-ended one: what is around this entity at all, through any predicate,
        in either direction. It is the same evidence-linked edges, walked.

        Nothing is invented to fill the result. Resolution is exact-match only:
        a near miss returns the candidate names rather than guessing, because
        traversing the wrong entity would silently produce a plausible and
        entirely wrong neighbourhood. Both bounds (``limit`` per node,
        ``max_nodes`` overall) are reported through ``truncated`` and ``note``
        instead of being applied quietly.
        """

        def _empty(**kw) -> GraphTraversalOutput:
            return GraphTraversalOutput(
                entity=inp.entity, depth=inp.depth, direction=inp.direction, **kw
            )

        ent = self.db.get_entity(inp.entity, inp.entity_type)
        if ent is None:
            candidates = [
                f"{r['name']} ({r['entity_type']})"
                for r in self.db.find_entities(inp.entity, inp.entity_type)
            ]
            if candidates:
                note = (
                    f"No entity named {inp.entity!r}. Close matches: "
                    + ", ".join(candidates[:8])
                    + ". Traversal is exact-match only; pass one of these names."
                )
            elif self.db.entity_count() == 0:
                note = (
                    "No entity by that name, and the graph is empty. Run "
                    "'blackbook graph build' after ingesting to populate it."
                )
            else:
                note = (
                    f"No entity named {inp.entity!r} in the graph"
                    + (f" of type {inp.entity_type!r}." if inp.entity_type else ".")
                )
            return _empty(found=False, candidates=candidates[:8], note=note)

        start_id = int(ent["entity_id"])
        nodes: dict[int, GraphNode] = {
            start_id: GraphNode(
                entity_id=start_id,
                name=ent["name"],
                entity_type=ent["entity_type"],
                description=ent.get("description") or "",
                hop=0,
                via=None,
            )
        }
        edges: list[GraphEdge] = []
        seen_edges: set[tuple[int, str, int]] = set()
        queue: deque[tuple[int, int]] = deque([(start_id, 0)])

        wanted = {p.strip() for p in (inp.predicates or []) if p.strip()}
        dropped = 0       # neighbours cut by the per-node ``limit``
        node_capped = 0   # neighbours not reached because ``max_nodes`` was hit

        while queue:
            node_id, hop = queue.popleft()
            if hop >= inp.depth:
                continue
            rels = self.db.entity_relationships(node_id)
            if inp.direction != "both":
                rels = [r for r in rels if r["direction"] == inp.direction]
            if wanted:
                rels = [r for r in rels if r["predicate"] in wanted]
            # Already sorted strongest-first by the database, so a cut keeps the
            # most confident and best-supported neighbours.
            if len(rels) > inp.limit:
                dropped += len(rels) - inp.limit
                rels = rels[: inp.limit]
            for rel in rels:
                other_id = int(rel["other_id"])
                if other_id not in nodes:
                    if len(nodes) >= inp.max_nodes:
                        node_capped += 1
                        continue  # skip the edge too: it would dangle otherwise
                    nodes[other_id] = GraphNode(
                        entity_id=other_id,
                        name=rel["other_name"],
                        entity_type=rel["other_type"],
                        description=rel.get("other_description") or "",
                        hop=hop + 1,
                        via=rel["predicate"],
                    )
                    queue.append((other_id, hop + 1))
                if rel["direction"] == "out":
                    subject, obj, key = nodes[node_id].name, rel["other_name"], (
                        node_id, rel["predicate"], other_id,
                    )
                else:
                    subject, obj, key = rel["other_name"], nodes[node_id].name, (
                        other_id, rel["predicate"], node_id,
                    )
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                edges.append(
                    GraphEdge(
                        subject=subject,
                        predicate=rel["predicate"],
                        object=obj,
                        confidence=round(float(rel["confidence"]), 4),
                        inferred=bool(rel["inferred"]),
                        support=int(rel.get("support") or 1),
                        evidence=self._evidence_ref(rel),
                    )
                )

        truncated = bool(dropped or node_capped)
        if not edges:
            note = (
                f"{ent['name']} has no relationships within depth {inp.depth}"
                + (f" matching {sorted(wanted)}." if wanted else ".")
            )
        elif truncated:
            parts = []
            if dropped:
                parts.append(
                    f"{dropped} lower-confidence neighbours were left out "
                    f"(per-node limit {inp.limit})"
                )
            if node_capped:
                parts.append(
                    f"{node_capped} more were not reached (max_nodes {inp.max_nodes})"
                )
            note = (
                "Partial neighbourhood: " + "; ".join(parts)
                + ". This is a bounded view, not the whole neighbourhood."
            )
        else:
            note = ""

        return GraphTraversalOutput(
            entity=inp.entity,
            resolved=ent["name"],
            entity_type=ent["entity_type"],
            found=True,
            depth=inp.depth,
            direction=inp.direction,
            nodes=list(nodes.values()),
            edges=edges,
            truncated=truncated,
            note=note,
        )

    # -- knowledge_case_search (Phase 4) -----------------------------------

    def knowledge_case_search(self, inp: CaseSearchInput) -> CaseSearchOutput:
        """Find hands-on writeups/case studies similar to a situation.

        Uses ``case_similarity`` mode so writeup-category material is favoured,
        then annotates each hit with the techniques the graph records for that
        document (when the graph is built). Results are always real indexed
        chunks with full provenance.
        """
        source_ids = self.settings.source_ids(inp.sources)
        query, resolved, _ = self._merge_techniques(inp.query, inp.techniques)
        if source_ids == []:
            return CaseSearchOutput(
                query=inp.query,
                count=0,
                results=[],
                note=(
                    f"No enabled source matches {inp.sources}; nothing was searched. "
                    "Check the source ID with `blackbook sources`."
                ),
            )

        results = self.retriever.search(
            query,
            mode="case_similarity",
            source_ids=source_ids,
            platform=inp.platform,
            categories=None,
            techniques=resolved,
            limit=inp.limit,
        )
        items = [self._case_item(r) for r in results]
        note = "" if items else "No matching case material found in the selected sources."
        return CaseSearchOutput(
            query=inp.query, count=len(items), results=items, note=note
        )

    def _case_item(self, r: SearchResult) -> CaseItem:
        techniques = self._doc_techniques(r.title, r.doc_id)
        base = self._to_item(r, "standard")
        return CaseItem(
            title=base.title,
            source=base.source,
            source_name=base.source_name,
            authority=base.authority,
            relevance=base.relevance,
            snippet=base.snippet,
            ref=base.ref,
            techniques=techniques,
        )

    def _doc_techniques(self, title: str, doc_id: int) -> list[str]:
        """Techniques the graph links to this writeup, via ``demonstrated_in``.

        Writeup entities are keyed by document title; a technique edge points
        *into* the writeup, so we collect incoming ``demonstrated_in`` neighbours
        of type ``technique``. Returns [] when the graph is not built (no writeup
        entity) — the tool then degrades to a plain, still-cited case search
        rather than failing. Guarded by ``doc_id`` so a title collision that
        merged two writeups doesn't attribute another document's techniques.
        """
        writeup = self.db.get_entity(title, E_WRITEUP)
        if not writeup:
            return []
        meta = writeup.get("meta")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (ValueError, TypeError):
                meta = {}
        if isinstance(meta, dict) and meta.get("doc_id") not in (None, doc_id):
            return []
        names = {
            rel["other_name"]
            for rel in self.db.entity_relationships(
                int(writeup["entity_id"]), predicate=P_DEMONSTRATED_IN
            )
            if rel["direction"] == "in" and rel["other_type"] == E_TECHNIQUE
        }
        return sorted(names)

    # -- knowledge_research (Phase 5) --------------------------------------

    def knowledge_research(self, inp: ResearchInput) -> ResearchOutput:
        """Turn a free-text observation into a source-grounded research packet.

        The packet is assembled from three real sources of truth and nothing
        else:

        * **signals** — controlled-vocabulary services/techniques/tools found in
          the observation by pure substring matching (``vocab.extract_signals``),
          so no term can be invented;
        * **technique briefs** — for each detected/requested technique, whether it
          resolved to the vocabulary and, if a graph entity exists, the sources
          that document it (evidence-linked graph edges only);
        * **references / related_cases** — real indexed chunks from a
          technique-biased (and, optionally, case-biased) search.

        Everything returned is either vocabulary-derived or a resolvable citation;
        there is no free-text synthesis and no fabricated entity or source.
        """
        source_ids = self.settings.source_ids(inp.sources)
        services, techniques, tools = extract_signals(inp.observation)
        if source_ids == []:
            return ResearchOutput(
                observation=inp.observation,
                signals=ResearchSignals(
                    services=services, techniques=techniques, tools=tools
                ),
                note=(
                    f"No enabled source matches {inp.sources}; nothing was searched. "
                    "Check the source ID with `blackbook sources`."
                ),
            )

        # Union of techniques detected in the text and any explicitly supplied,
        # each mapped through the controlled vocabulary; unresolved extras drop.
        canonical_techs = list(techniques)
        for t in inp.techniques or []:
            resolved = resolve_technique(t)
            if resolved and resolved not in canonical_techs:
                canonical_techs.append(resolved)

        briefs: list[TechniqueBrief] = []
        for term in canonical_techs:
            documented_by: list[GraphRef] = []
            ent = self.db.get_entity(term, "technique")
            in_graph = ent is not None
            if ent is not None:
                for rel in self.db.entity_relationships(int(ent["entity_id"])):
                    if rel["predicate"] == "documented_by" and rel["direction"] == "out":
                        documented_by.append(self._graph_ref(rel))
            briefs.append(
                TechniqueBrief(
                    technique=term,
                    resolved=resolve_technique(term) is not None,
                    in_graph=in_graph,
                    attack_id=self._attack_ids(term),
                    documented_by=documented_by,
                )
            )

        # Bias the retrieval query with the resolved signals so the packet is
        # centred on what the observation is actually about.
        query = inp.observation
        extra = canonical_techs + services + tools
        if extra:
            query = query + " " + " ".join(extra)

        results = self.retriever.search(
            query,
            mode="technique",
            source_ids=source_ids,
            platform=inp.platform,
            categories=None,
            techniques=canonical_techs,
            limit=inp.limit,
        )
        references = [self._to_item(r, "standard") for r in results]

        related_cases: list[CaseItem] = []
        if inp.include_cases:
            case_hits = self.retriever.search(
                query,
                mode="case_similarity",
                source_ids=source_ids,
                platform=inp.platform,
                categories=None,
                techniques=canonical_techs,
                limit=inp.limit,
            )
            related_cases = [self._case_item(r) for r in case_hits]

        note = ""
        if not references and not briefs:
            note = (
                "No known signals and no indexed references for this observation. "
                "Confirm the corpus is ingested and the terms are in scope."
            )
        return ResearchOutput(
            observation=inp.observation,
            signals=ResearchSignals(
                services=services, techniques=techniques, tools=tools
            ),
            techniques=briefs,
            references=references,
            related_cases=related_cases,
            note=note,
        )

    # -- bug bounty workflow tools ----------------------------------------

    def knowledge_hunt_plan(self, inp: HuntPlanInput) -> HuntPlanOutput:
        """Build a retrieval-backed, non-executing validation plan."""
        source_ids = self.settings.source_ids(inp.sources)
        services, techniques, tools = extract_signals(inp.observation)
        if source_ids == []:
            return HuntPlanOutput(
                observation=inp.observation,
                target=inp.target,
                signals=ResearchSignals(services=services, techniques=techniques, tools=tools),
                note=f"No enabled source matches {inp.sources}; nothing was searched.",
            )

        candidates: list[tuple[str, str]] = []
        for term in techniques:
            candidates.append((term, "technique"))
        for term in services:
            candidates.append((term, "service"))
        for term in tools:
            candidates.append((term, "tool"))
        for term in inp.techniques or []:
            canonical = resolve_technique(term) or term.strip()
            if canonical and not any(c[0].lower() == canonical.lower() for c in candidates):
                candidates.append((canonical, "technique"))
        if not candidates:
            candidates.append((inp.observation[:80].strip(), "observation"))

        plans: list[HuntPlanItem] = []
        for term, category in candidates[: inp.limit]:
            query = f"{term} {inp.observation}".strip()
            results = self.retriever.search(
                query,
                mode="technique" if category == "technique" else "hybrid",
                source_ids=source_ids,
                platform=inp.platform,
                categories=None,
                techniques=[term] if category == "technique" and resolve_technique(term) else [],
                limit=min(inp.limit, 5),
            )
            if category == "technique":
                focus = [
                    "Confirm the behavior with a controlled request and response pair.",
                    "Test the authorization or trust boundary using only in-scope accounts and objects.",
                    "Capture reproducible evidence before assigning impact.",
                ]
            elif category == "service":
                focus = [
                    "Identify the exposed feature and its authentication boundary.",
                    "Check documented misconfigurations and affected versions.",
                    "Record a harmless, reproducible response as evidence.",
                ]
            elif category == "tool":
                focus = [
                    "Confirm the tool or integration is actually reachable in scope.",
                    "Check input, authorization, and output handling.",
                    "Avoid execution or state-changing validation through Blackbook.",
                ]
            else:
                focus = [
                    "Break the observation into a concrete, testable hypothesis.",
                    "Find an observable security impact, not only a theoretical weakness.",
                    "Preserve request, response, account, and object identifiers as evidence.",
                ]
            plans.append(
                HuntPlanItem(
                    title=f"Validate {term}",
                    category=category,
                    rationale=f"The observation contains or requests the {category} signal '{term}'.",
                    validation_focus=focus,
                    references=[self._to_item(r, "standard") for r in results],
                )
            )
        return HuntPlanOutput(
            observation=inp.observation,
            target=inp.target,
            signals=ResearchSignals(services=services, techniques=techniques, tools=tools),
            plans=plans,
            note="Plan items are research guidance only; Blackbook performs no target testing.",
        )

    def knowledge_finding_review(self, inp: FindingReviewInput) -> FindingReviewOutput:
        """Review a suspected finding without upgrading claims beyond evidence."""
        source_ids = self.settings.source_ids(inp.sources)
        services, techniques, tools = extract_signals(inp.finding)
        observed: list[str] = []
        missing = [
            "A reproducible request/response pair proving the behavior.",
            "A demonstrated security impact affecting an in-scope asset or account.",
            "Evidence that the behavior is not expected functionality or a duplicate.",
        ]
        if inp.case:
            state = self._case_state(inp.case)
            if state:
                observed = [
                    f"[{o.status}] {o.kind}: {o.text}"
                    for o in state.observations
                    if o.status in {"tested", "confirmed", "resolved"}
                ]
                if any(o.status == "confirmed" for o in state.observations):
                    missing.pop(0)
        if source_ids == []:
            return FindingReviewOutput(
                finding=inp.finding,
                case=inp.case,
                signals=ResearchSignals(services=services, techniques=techniques, tools=tools),
                evidence_status="no_matching_sources",
                observed_evidence=observed,
                missing_evidence=missing,
                note=f"No enabled source matches {inp.sources}; nothing was searched.",
            )
        query = f"{inp.finding} impact severity triage"
        references = self.retriever.search(
            query,
            mode="technique",
            source_ids=source_ids,
            platform=inp.platform,
            categories=None,
            techniques=techniques,
            limit=inp.limit,
        )
        severity = self.retriever.search(
            f"{inp.finding} severity impact vulnerability rating",
            mode="hybrid",
            source_ids=source_ids,
            platform=inp.platform,
            categories=None,
            techniques=techniques,
            limit=inp.limit,
        )
        status = "case_has_confirmed_evidence" if any(
            line.startswith("[confirmed]") for line in observed
        ) else "documentation_only"
        return FindingReviewOutput(
            finding=inp.finding,
            case=inp.case,
            signals=ResearchSignals(services=services, techniques=techniques, tools=tools),
            evidence_status=status,
            observed_evidence=observed,
            missing_evidence=missing,
            references=[self._to_item(r, "standard") for r in references],
            severity_guidance=[self._to_item(r, "standard") for r in severity],
            note="Documentation supports review criteria; it does not prove this finding.",
        )

    def knowledge_report_draft(self, inp: ReportDraftInput) -> ReportDraftOutput:
        """Draft a report from local case observations and cited guidance."""
        state = self._case_state(inp.case)
        if state is None:
            return ReportDraftOutput(
                case=inp.case,
                title=f"Unresolved security finding: {inp.case}",
                summary="No local case was found; no report claims were generated.",
                impact="Impact is unverified.",
                remediation="No remediation was generated without a local case and evidence.",
                warnings=[f"Case '{inp.case}' was not found."],
            )
        findings = [o for o in state.observations if o.kind == "finding"]
        confirmed = [o for o in state.observations if o.status == "confirmed"]
        evidence = [f"[{o.status}] {o.kind}: {o.text}" for o in state.observations]
        primary = findings[0].text if findings else (confirmed[0].text if confirmed else "Security observation")
        references = self.retriever.search(
            f"{primary} remediation impact",
            mode="hybrid",
            source_ids=self.settings.source_ids(inp.sources),
            platform=state.platform or None,
            categories=None,
            techniques=extract_signals(primary)[1],
            limit=inp.limit,
        )
        refs = [self._to_item(r, "standard") for r in references]
        warnings = [
            "This is a draft; verify every claim against the target evidence before submission.",
            "Blackbook does not infer exploitability, ownership, or severity from documentation alone.",
        ]
        if not confirmed:
            warnings.append("No observation is marked confirmed in the case.")
        return ReportDraftOutput(
            case=inp.case,
            title=f"{primary[:120]}",
            summary=(
                f"Case '{inp.case}' records a suspected security issue for "
                f"{state.target or 'the documented target'}."
            ),
            observed_evidence=evidence,
            reproduction_steps=[
                "Reproduce the behavior using the exact in-scope request and response recorded in the case.",
                "Repeat with the documented control account, object, or baseline.",
                "Capture the smallest request/response pair that demonstrates impact.",
            ],
            impact=(
                "Impact is supported by confirmed case observations only; quantify the affected "
                "asset, account, data, or action before submission."
            ),
            remediation="Apply the control described by the cited guidance after confirming the affected component.",
            severity_basis=refs[: inp.limit],
            references=refs,
            warnings=warnings,
        )

    @staticmethod
    def _semantic_status(db, settings) -> SemanticStatus:
        """Report whether dense retrieval can contribute, without loading a model.

        Semantic retrieval is inert in two independent ways and both used to be
        silent: it is off by default, and even when enabled it contributes
        nothing until ``blackbook embed`` has written vectors. One ``COUNT(*)``
        distinguishes them for the caller.
        """
        enabled = bool(settings.embeddings.enabled)
        model = settings.embeddings.model
        vectors = db.embedding_count(model)
        ready = enabled and vectors > 0
        if not enabled:
            note = (
                "Disabled (embeddings.enabled is false); searches cannot use the "
                "semantic backend. Enable it and run `blackbook embed` to index vectors."
            )
        elif vectors <= 0:
            note = (
                f"Enabled but no vectors are indexed for model {model}; run "
                "`blackbook embed` to build the index. Until then searches fall "
                "back to lexical (BM25)."
            )
        else:
            note = f"{vectors} vectors indexed for {model}."
        return SemanticStatus(
            enabled=enabled, model=model, vectors=vectors, ready=ready, note=note
        )

    def knowledge_sources(self, inp: KnowledgeSourceInput) -> KnowledgeSourcesOutput:
        """Return configured sources with real indexed document/chunk counts."""
        configured = {source.id: source for source in self.settings.sources}
        if inp.source is not None and inp.source not in configured:
            return KnowledgeSourcesOutput(count=0, note=f"Unknown configured source: {inp.source}")
        counts = self.db.source_index_counts()
        rows = self.db.list_sources()
        indexed = {row["source_id"]: row for row in rows}
        selected = [configured[inp.source]] if inp.source else list(self.settings.sources)
        statuses = [
            KnowledgeSourceStatus(
                id=source.id,
                name=source.name,
                enabled=source.enabled,
                authority=source.authority,
                source_type=source.type,
                url=source.url,
                indexed_documents=counts.get(source.id, {}).get("documents", 0),
                indexed_chunks=counts.get(source.id, {}).get("chunks", 0),
                # Freshness is a property of the stored row, not the config, so
                # it reads from the indexed row and stays None until an ingest
                # has actually pulled the source.
                last_fetched=(indexed.get(source.id) or {}).get("last_fetched"),
                version=(indexed.get(source.id) or {}).get("version"),
            )
            for source in selected
        ]
        return KnowledgeSourcesOutput(
            count=len(statuses),
            sources=statuses,
            # A single-source lookup is a per-source question, not a global one;
            # the corpus-wide backend status would be noise there.
            semantic=None if inp.source else self._semantic_status(self.db, self.settings),
            note=("Configured source is not indexed yet." if statuses and not indexed.get(statuses[0].id) and inp.source else ""),
        )

    def knowledge_compare(self, inp: KnowledgeCompareInput) -> KnowledgeCompareOutput:
        """Retrieve comparable evidence independently from each selected source."""
        source_ids = self.settings.source_ids(inp.sources)
        if source_ids == []:
            return KnowledgeCompareOutput(
                topic=inp.topic,
                sources_compared=[],
                note="No requested sources are enabled; nothing was searched.",
            )
        views: list[KnowledgeCompareView] = []
        token_sources: dict[str, set[str]] = {}
        for source_id in source_ids or []:
            config = self.settings.get_source(source_id)
            if config is None:
                continue
            results = self.retriever.search(
                inp.topic,
                mode="hybrid",
                source_ids=[source_id],
                platform=inp.platform,
                categories=None,
                techniques=[],
                limit=inp.limit,
            )
            items = [self._to_item(r, "standard") for r in results]
            views.append(KnowledgeCompareView(
                source=source_id,
                source_name=config.name,
                authority=config.authority,
                results=items,
            ))
            for item in items:
                words = set(re.findall(r"[a-z][a-z0-9-]{3,}", item.snippet.lower()))
                for word in words:
                    token_sources.setdefault(word, set()).add(source_id)
        stop = {"that", "this", "with", "from", "into", "when", "which", "their", "have", "will", "about", "your"}
        shared = sorted(word for word, sources in token_sources.items() if len(sources) >= 2 and word not in stop)
        return KnowledgeCompareOutput(
            topic=inp.topic,
            sources_compared=[view.source for view in views],
            views=views,
            shared_terms=shared[:30],
            note="Views are retrieved independently; shared_terms are lexical overlap, not a claim of factual agreement.",
        )

    # -- knowledge_context (Phase 5) ---------------------------------------

    def knowledge_context(self, inp: ContextInput) -> ContextOutput:
        """Manage local investigation state: cases and their observations.

        This is the one tool that *writes*, but only to the local, user-authored
        case layer inside the same SQLite file — it never touches, executes
        against, or fetches from any external system. Actions:

        * ``create`` — upsert a case by name (target/platform/meta optional);
        * ``add`` — append an observation/finding/hypothesis/etc. to a case;
        * ``update_observation`` — set an existing observation's status;
        * ``get`` — return a case's full current state;
        * ``list`` — summarise all cases;
        * ``export`` — render a case as portable Markdown (returned in-band;
          the server never writes files — ``blackbook case export`` does).

        There is deliberately no delete action — the tool cannot destroy state.
        """
        action = inp.action

        if action == "list":
            cases = [
                CaseSummary(
                    case_id=int(c["case_id"]),
                    name=c["name"],
                    target=c.get("target") or "",
                    platform=c.get("platform") or "",
                    observation_count=int(c.get("observation_count") or 0),
                    updated_at=c.get("updated_at"),
                )
                for c in self.db.list_cases()
            ]
            return ContextOutput(
                action=action,
                ok=True,
                cases=cases,
                note="" if cases else "No cases yet.",
            )

        if action == "export":
            if not inp.case:
                return ContextOutput(
                    action=action, ok=False, note="'case' is required for export."
                )
            state = self._case_state(inp.case)
            if state is None:
                return ContextOutput(
                    action=action, ok=False, note=f"Case '{inp.case}' not found."
                )
            markdown = render_case_markdown(state)
            return ContextOutput(
                action=action,
                ok=True,
                case=state,
                markdown=markdown,
                note=(
                    "Markdown returned in-band; run 'blackbook case export "
                    f"{inp.case}' to write it to a file."
                ),
            )

        if action == "get":
            if not inp.case:
                return ContextOutput(
                    action=action, ok=False, note="'case' is required for get."
                )
            state = self._case_state(inp.case)
            if state is None:
                return ContextOutput(
                    action=action, ok=False, note=f"Case '{inp.case}' not found."
                )
            return ContextOutput(action=action, ok=True, case=state)

        if action == "create":
            if not inp.case:
                return ContextOutput(
                    action=action, ok=False, note="'case' is required for create."
                )
            with self.db.session():
                self.db.upsert_case(
                    Case(
                        name=inp.case,
                        target=inp.target,
                        platform=inp.platform,
                        meta=inp.meta or {},
                    )
                )
            return ContextOutput(
                action=action, ok=True, case=self._case_state(inp.case)
            )

        if action == "add":
            if not inp.case or not inp.text:
                return ContextOutput(
                    action=action,
                    ok=False,
                    note="'case' and 'text' are required for add.",
                )
            existing = self.db.get_case(inp.case)
            if existing is None:
                return ContextOutput(
                    action=action,
                    ok=False,
                    note=f"Case '{inp.case}' not found; create it first.",
                )
            with self.db.session():
                self.db.add_observation(
                    CaseObservation(
                        case_id=int(existing["case_id"]),
                        kind=inp.kind,
                        text=inp.text,
                    )
                )
            return ContextOutput(
                action=action, ok=True, case=self._case_state(inp.case)
            )

        if action == "update_observation":
            if not inp.case or inp.obs_id is None or inp.status is None:
                return ContextOutput(
                    action=action,
                    ok=False,
                    note="'case', 'obs_id', and 'status' are required for update_observation.",
                )
            case = self.db.get_case(inp.case)
            if case is None:
                return ContextOutput(
                    action=action, ok=False, note=f"Case '{inp.case}' not found."
                )
            obs = self.db.get_observation(inp.obs_id)
            if obs is None or int(obs["case_id"]) != int(case["case_id"]):
                return ContextOutput(
                    action=action,
                    ok=False,
                    note=f"Observation {inp.obs_id} not found in case '{inp.case}'.",
                )
            with self.db.session():
                self.db.set_observation_status(inp.obs_id, inp.status)
            return ContextOutput(
                action=action, ok=True, case=self._case_state(inp.case)
            )

        # Unreachable: pydantic constrains `action` to the Literal set.
        return ContextOutput(action=action, ok=False, note="Unknown action.")

    def _case_state(self, name: str) -> CaseState | None:
        """Compose a case's row and its observations into a CaseState."""
        return build_case_state(self.db, name)
