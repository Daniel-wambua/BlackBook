"""MCP tool implementations.

The registered tool surface:

* ``knowledge_search`` — source-grounded search across the indexed corpus
* ``knowledge_source`` — resolve a reference to the exact source excerpt
* ``knowledge_technique`` — a graph-enhanced, cited dossier for a technique
* ``knowledge_case_search`` — find hands-on writeups similar to a situation
* ``knowledge_research`` — turn a free-text observation into a source-grounded
  research packet (detected signals, technique briefs, cited references, cases)
* ``knowledge_context`` — manage local investigation state (cases + observations)

All search/technique/research tools are read-only over the index and always
return structured, provenance-tagged output; every citation resolves to a real
indexed chunk. ``knowledge_context`` reads and writes only the *local*,
user-authored case layer — it never touches, executes against, or fetches from
any external system.
"""

from __future__ import annotations

import json
import logging

from blackbook.config import Settings
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
    GetSourceInput,
    GraphRef,
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
from blackbook.retrieval import HybridRetriever, SearchResult
from blackbook.storage.database import Database
from blackbook.storage.models import Case, CaseObservation

log = logging.getLogger(__name__)


class KnowledgeTools:
    """Holds shared state (db, settings, retriever) for the MCP tools."""

    def __init__(self, db: Database, settings: Settings):
        self.db = db
        self.settings = settings
        self.retriever = HybridRetriever(db, settings)

    # -- knowledge_search ---------------------------------------------------

    def knowledge_search(self, inp: SearchInput) -> SearchOutput:
        requested, source_ids = self._resolve_sources(inp.sources)
        # Techniques resolve through the controlled vocabulary: resolved terms
        # still join the query (recall) *and* bias reranking toward chunks whose
        # title/section names them (precision); unresolved ones remain plain
        # query terms so a non-vocab technique is still searched.
        query, resolved, unresolved = self._merge_techniques(inp.query, inp.techniques)

        results: list[SearchResult] = self.retriever.search(
            query,
            mode=inp.mode,
            source_ids=source_ids,
            platform=inp.platform,
            categories=inp.categories,
            techniques=resolved,
            limit=inp.limit,
        )
        items = [self._to_item(r, inp.detail) for r in results]
        note = ""
        if source_ids == []:
            note = (
                f"No enabled source matches {requested}; nothing was searched. "
                "Check the source ID with `blackbook sources`."
            )
        elif not items:
            note = "No matching material found in the selected sources."
        if unresolved:
            extra = (
                "Techniques not in the controlled vocabulary were searched as "
                f"plain terms: {', '.join(unresolved)}."
            )
            note = f"{note} {extra}".strip()
        return SearchOutput(
            query=inp.query,
            mode=inp.mode,
            sources_searched=source_ids if source_ids is not None else ["all"],
            count=len(items),
            results=items,
            note=note,
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
            attack_id=attack_id(term),
            documented_by=documented_by,
            related_tools=related_tools,
            related_services=related_services,
            demonstrated_in=demonstrated_in,
            references=references,
            note=note,
        )

    @staticmethod
    def _graph_ref(rel: dict) -> GraphRef:
        evidence = None
        if rel.get("evidence_doc_id") is not None:
            evidence = EvidenceRef(
                doc_id=rel.get("evidence_doc_id"),
                title=rel.get("evidence_title"),
                source=rel.get("evidence_source_id"),
                source_name=rel.get("evidence_source_name"),
                authority=rel.get("evidence_authority"),
                url=rel.get("evidence_url"),
                external_id=rel.get("evidence_external_id"),
            )
        return GraphRef(
            name=rel["other_name"],
            entity_type=rel["other_type"],
            predicate=rel["predicate"],
            confidence=round(float(rel["confidence"]), 4),
            inferred=bool(rel["inferred"]),
            evidence=evidence,
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
                    attack_id=attack_id(term),
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
