from blackbook.config import Settings, DatabaseConfig
from blackbook.knowledge.graph import GraphBuilder
from blackbook.mcp.schemas import (
    CaseSearchInput,
    ContextInput,
    GetSourceInput,
    GraphTraversalInput,
    ResearchInput,
    SearchInput,
    TechniqueInput,
)
from blackbook.mcp.tools import KnowledgeTools


def make_tools(tmp_path, seeded_db):
    settings = Settings(home=tmp_path, database=DatabaseConfig(path=str(tmp_path / "d.db")))
    return KnowledgeTools(seeded_db, settings)


def test_knowledge_search_returns_provenance(tmp_path, seeded_db):
    tools = make_tools(tmp_path, seeded_db)
    out = tools.knowledge_search(SearchInput(query="kerberoasting"))
    assert out.count >= 1
    item = out.results[0]
    # provenance present and traceable
    assert item.ref.chunk_id > 0
    assert item.ref.source in ("hacktricks", "0xdf")
    assert item.ref.title
    assert item.authority in ("trusted", "user", "official", "unknown")


def test_knowledge_search_source_filter(tmp_path, seeded_db):
    tools = make_tools(tmp_path, seeded_db)
    out = tools.knowledge_search(SearchInput(query="kerberoast", sources=["0xdf"]))
    assert out.count >= 1
    assert all(r.source == "0xdf" for r in out.results)


def test_knowledge_search_empty_result_note(tmp_path, seeded_db):
    tools = make_tools(tmp_path, seeded_db)
    out = tools.knowledge_search(SearchInput(query="zzz_no_such_term_zzz"))
    assert out.count == 0
    assert out.note


def test_knowledge_source_by_chunk_id(tmp_path, seeded_db):
    tools = make_tools(tmp_path, seeded_db)
    search = tools.knowledge_search(SearchInput(query="kerberoasting"))
    chunk_id = search.results[0].ref.chunk_id
    out = tools.knowledge_source(GetSourceInput(chunk_id=chunk_id))
    assert out.count == 1
    ex = out.excerpts[0]
    assert ex.ref.chunk_id == chunk_id
    assert ex.text  # the exact excerpt text is returned


def test_knowledge_source_document_section_filter(tmp_path, seeded_db):
    tools = make_tools(tmp_path, seeded_db)
    out = tools.knowledge_source(
        GetSourceInput(source="hacktricks", document="ad/kerberoasting.md", section="Tools")
    )
    assert out.count >= 1
    assert any("Tools" in e.ref.section_path for e in out.excerpts)


def test_knowledge_source_not_found(tmp_path, seeded_db):
    tools = make_tools(tmp_path, seeded_db)
    out = tools.knowledge_source(GetSourceInput(chunk_id=99999))
    assert out.count == 0
    assert "not found" in out.note.lower()


def test_provenance_chain_roundtrip(tmp_path, seeded_db):
    """Every citation returned must resolve back to real indexed content."""
    tools = make_tools(tmp_path, seeded_db)
    search = tools.knowledge_search(SearchInput(query="service tickets", limit=5))
    assert search.count >= 1
    for item in search.results:
        src = tools.knowledge_source(GetSourceInput(chunk_id=item.ref.chunk_id))
        assert src.count == 1
        # The cited snippet must be a substring of the resolvable source text
        assert src.excerpts[0].text
        assert item.ref.title == src.excerpts[0].ref.title


# -- knowledge_technique (Phase 4) ------------------------------------------


def test_technique_works_without_graph(tmp_path, seeded_db):
    """With no graph built, the dossier still returns real, cited references."""
    tools = make_tools(tmp_path, seeded_db)
    out = tools.knowledge_technique(TechniqueInput(technique="kerberoasting"))
    assert out.resolved is True          # in the controlled vocabulary
    assert out.in_graph is False         # graph not built yet
    assert out.references                # but real excerpts are still returned
    assert all(r.ref.chunk_id > 0 for r in out.references)
    assert out.note                      # explains the graph is absent


def test_technique_enriched_by_graph(tmp_path, seeded_db):
    """After a graph build the dossier carries evidence-linked neighbours."""
    GraphBuilder(seeded_db).rebuild()
    tools = make_tools(tmp_path, seeded_db)
    out = tools.knowledge_technique(TechniqueInput(technique="kerberoasting"))
    assert out.in_graph is True
    # Documented by HackTricks, with a real backing document.
    assert any(g.name == "HackTricks" and g.entity_type == "source"
               for g in out.documented_by)
    for g in out.documented_by:
        assert g.evidence is not None and g.evidence.doc_id is not None
    # Co-occurring tool/service neighbours are present and evidence-linked.
    assert any(g.name == "impacket" for g in out.related_tools)
    assert {g.name for g in out.related_services} >= {"kerberos"}
    # References are always real chunks regardless of the graph.
    assert out.references and out.references[0].ref.chunk_id > 0


def test_technique_no_fabricated_neighbours(tmp_path, seeded_db):
    """Graph neighbours are only ever real, evidence-linked edges."""
    GraphBuilder(seeded_db).rebuild()
    tools = make_tools(tmp_path, seeded_db)
    out = tools.knowledge_technique(TechniqueInput(technique="kerberoasting"))
    for g in out.documented_by + out.related_tools + out.related_services + out.demonstrated_in:
        assert g.evidence is not None
        assert g.evidence.doc_id is not None   # a citable document
        assert 0.0 < g.confidence <= 1.0


def test_technique_unresolved_term(tmp_path, seeded_db):
    """An unknown term is echoed back, not resolved, and never invents graph data."""
    tools = make_tools(tmp_path, seeded_db)
    out = tools.knowledge_technique(TechniqueInput(technique="totally unknown thing"))
    assert out.resolved is False
    assert out.in_graph is False
    assert out.documented_by == []
    assert out.technique == "totally unknown thing"


# -- knowledge_graph (traversal) --------------------------------------------
#
# The seeded graph is fully determined (see test_graph.py): kerberoasting has
# four outgoing edges (documented_by HackTricks at 0.9, uses impacket and
# targets active directory / kerberos, all at 0.5) and no incoming ones, while
# 0xdf has two incoming edges and no outgoing ones. That asymmetry is what
# makes the direction filter testable without contriving a fixture.


def _traverse(tmp_path, db, **kw):
    return make_tools(tmp_path, db).knowledge_graph(GraphTraversalInput(**kw))


def test_graph_traversal_returns_the_neighbourhood_with_evidence(tmp_path, seeded_db):
    GraphBuilder(seeded_db).rebuild()
    out = _traverse(tmp_path, seeded_db, entity="kerberoasting")
    assert out.found is True
    assert out.resolved == "kerberoasting" and out.entity_type == "technique"
    # Four outgoing edges, five entities counting the start node.
    assert len(out.edges) == 4
    assert len(out.nodes) == 5
    assert {n.name for n in out.nodes} == {
        "kerberoasting", "HackTricks", "impacket", "active directory", "kerberos",
    }
    # Every edge is citable, and every neighbour other than the start has a hop.
    for e in out.edges:
        assert e.evidence is not None and e.evidence.doc_id is not None
        assert e.subject == "kerberoasting"  # all four leave the technique
    assert {n.hop for n in out.nodes if n.name != "kerberoasting"} == {1}
    assert out.truncated is False
    assert out.note == ""


def test_graph_traversal_direction_filter(tmp_path, seeded_db):
    """Edges are stored subject -> object, so direction is a real asymmetry.

    Nothing points at 0xdf from the corpus side, so its outgoing walk is empty
    and its incoming walk is not. A filter that quietly ignored direction would
    return the same thing twice here.
    """
    GraphBuilder(seeded_db).rebuild()
    out_only = _traverse(tmp_path, seeded_db, entity="0xdf", direction="out")
    assert out_only.found is True
    assert out_only.edges == []
    assert out_only.note  # says so rather than returning a bare empty result

    in_only = _traverse(tmp_path, seeded_db, entity="0xdf", direction="in")
    assert len(in_only.edges) == 2
    assert {e.subject for e in in_only.edges} == {"as-rep roasting", "HTB: Forest"}
    assert all(e.object == "0xdf" for e in in_only.edges)


def test_graph_traversal_depth_reaches_further(tmp_path, seeded_db):
    """A second hop surfaces an edge whose endpoints were both already visible.

    From 0xdf the one-hop view already contains as-rep roasting and HTB: Forest,
    but not the demonstrated_in edge between them: that edge leaves as-rep
    roasting, and the walk had not expanded it yet. depth=2 expands it and the
    edge appears. The node set is unchanged, which is the point — depth buys
    edges, not only more entities.
    """
    GraphBuilder(seeded_db).rebuild()
    edge_set = ("as-rep roasting", "demonstrated_in", "HTB: Forest")
    one = _traverse(tmp_path, seeded_db, entity="0xdf", depth=1)
    two = _traverse(tmp_path, seeded_db, entity="0xdf", depth=2)
    assert len(one.nodes) == 3 and len(one.edges) == 2
    assert edge_set not in {(e.subject, e.predicate, e.object) for e in one.edges}
    assert len(two.nodes) == 3
    assert {(e.subject, e.predicate, e.object) for e in two.edges} == {
        ("as-rep roasting", "documented_by", "0xdf"),
        ("HTB: Forest", "documented_by", "0xdf"),
        edge_set,
    }


def test_graph_traversal_depth_does_not_invent_a_second_hop(tmp_path, seeded_db):
    """Every hop-1 neighbour of kerberoasting is a leaf here, so depth 2 adds none.

    The walk reports the corpus as it is. A deeper setting on a shallow
    neighbourhood returns the same thing rather than padding the result to look
    like it explored more.
    """
    GraphBuilder(seeded_db).rebuild()
    one = _traverse(tmp_path, seeded_db, entity="kerberoasting", depth=1)
    two = _traverse(tmp_path, seeded_db, entity="kerberoasting", depth=2)
    assert len(one.nodes) == 5 and len(one.edges) == 4
    assert [n.name for n in two.nodes] == [n.name for n in one.nodes]
    assert len(two.edges) == 4
    # Everything reached came through a predicate that is on an edge.
    predicates = {e.predicate for e in two.edges}
    assert {n.via for n in two.nodes if n.hop} <= predicates


def test_graph_traversal_reports_max_nodes_truncation_without_dangling_edges(
    tmp_path, seeded_db
):
    """A bounded walk must not reference an entity it refused to include.

    Cutting the node budget mid-edge would leave an edge naming an entity that is
    absent from ``nodes``, which a caller could not resolve. The edge is dropped
    with the node, and the shortfall is reported.
    """
    GraphBuilder(seeded_db).rebuild()
    out = _traverse(tmp_path, seeded_db, entity="kerberoasting", max_nodes=3)
    assert out.truncated is True
    assert "max_nodes" in out.note
    assert len(out.nodes) <= 3
    known = {n.name for n in out.nodes}
    for e in out.edges:
        assert e.subject in known and e.object in known


def test_graph_traversal_per_node_limit_keeps_the_strongest(tmp_path, seeded_db):
    """The cut is taken from the weakest end, and it is announced.

    ``entity_relationships`` orders strongest-first, so bounding each node keeps
    the 0.9 documentary edge over the 0.5 co-occurrence ones rather than
    whichever happened to be stored first.
    """
    GraphBuilder(seeded_db).rebuild()
    out = _traverse(tmp_path, seeded_db, entity="kerberoasting", limit=2)
    assert out.truncated is True
    assert len(out.edges) == 2
    assert out.edges[0].predicate == "documented_by"
    assert out.edges[0].confidence == 0.9
    assert "limit 2" in out.note


def test_graph_traversal_predicate_filter(tmp_path, seeded_db):
    GraphBuilder(seeded_db).rebuild()
    out = _traverse(tmp_path, seeded_db, entity="kerberoasting", predicates=["targets"])
    assert len(out.edges) == 2
    assert {e.predicate for e in out.edges} == {"targets"}
    assert {e.object for e in out.edges} == {"active directory", "kerberos"}
    # The unfiltered neighbours are absent, not merely unlabelled.
    assert "impacket" not in {n.name for n in out.nodes}
    assert out.truncated is False


def test_graph_traversal_does_not_guess_a_near_miss(tmp_path, seeded_db):
    """A substring match is offered as a candidate, never silently traversed.

    "kerberoast" is a prefix of the real entity. Resolving it automatically would
    return a plausible neighbourhood for an entity the caller never named, so the
    walk declines and names what it found instead.
    """
    GraphBuilder(seeded_db).rebuild()
    out = _traverse(tmp_path, seeded_db, entity="kerberoast")
    assert out.found is False
    assert out.nodes == [] and out.edges == []
    assert "kerberoasting (technique)" in out.candidates
    assert "kerberoasting (technique)" in out.note


def test_graph_traversal_type_scope_is_enforced(tmp_path, seeded_db):
    """Asking for the wrong type is a miss, not a fallback to the right one."""
    GraphBuilder(seeded_db).rebuild()
    out = _traverse(tmp_path, seeded_db, entity="kerberoasting", entity_type="tool")
    assert out.found is False
    assert out.candidates == []
    assert "tool" in out.note


def test_graph_traversal_without_a_graph_says_how_to_build_one(tmp_path, db):
    """An empty graph and an absent entity are different answers."""
    out = _traverse(tmp_path, db, entity="kerberoasting")
    assert out.found is False
    assert "graph build" in out.note


def test_graph_traversal_is_deterministic(tmp_path, seeded_db):
    GraphBuilder(seeded_db).rebuild()
    first = _traverse(tmp_path, seeded_db, entity="kerberoasting", depth=2)
    second = _traverse(tmp_path, seeded_db, entity="kerberoasting", depth=2)
    assert first.model_dump() == second.model_dump()


# -- knowledge_case_search (Phase 4) ----------------------------------------


def test_case_search_favours_writeups(tmp_path, seeded_db):
    """case_similarity mode lifts the hands-on writeup above a plain search.

    It nudges rather than gates, so a strongly-matching reference page can still
    lead; what must hold is that the writeup scores strictly higher under
    case_similarity than under a plain hybrid search of the same query.
    """
    tools = make_tools(tmp_path, seeded_db)
    query = "kerberoast a service account"
    plain = tools.retriever.search(query, mode="hybrid", limit=10)
    cased = tools.retriever.search(query, mode="case_similarity", limit=10)

    def forest_score(results):
        return next(r.score for r in results if r.title == "HTB: Forest")

    assert forest_score(cased) > forest_score(plain)

    out = tools.knowledge_case_search(CaseSearchInput(query=query))
    assert out.count >= 1
    forest = next(r for r in out.results if r.title == "HTB: Forest")
    assert forest.ref.chunk_id > 0            # real provenance
    assert forest.authority in ("trusted", "user", "official", "unknown")


def test_case_search_annotates_techniques_from_graph(tmp_path, seeded_db):
    """When the graph is built, writeup hits carry their demonstrated techniques."""
    GraphBuilder(seeded_db).rebuild()
    tools = make_tools(tmp_path, seeded_db)
    out = tools.knowledge_case_search(CaseSearchInput(query="kerberoast"))
    forest = next(r for r in out.results if r.title == "HTB: Forest")
    # HTB: Forest demonstrates AS-REP roasting per the graph.
    assert "as-rep roasting" in forest.techniques


def test_case_search_without_graph_has_no_techniques(tmp_path, seeded_db):
    """Without a graph, case results are still returned, just unannotated."""
    tools = make_tools(tmp_path, seeded_db)
    out = tools.knowledge_case_search(CaseSearchInput(query="kerberoast"))
    assert out.count >= 1
    assert all(r.techniques == [] for r in out.results)


def test_case_search_source_filter(tmp_path, seeded_db):
    tools = make_tools(tmp_path, seeded_db)
    out = tools.knowledge_case_search(
        CaseSearchInput(query="kerberoast", sources=["0xdf"])
    )
    assert out.count >= 1
    assert all(r.source == "0xdf" for r in out.results)


def test_case_search_empty_note(tmp_path, seeded_db):
    tools = make_tools(tmp_path, seeded_db)
    out = tools.knowledge_case_search(CaseSearchInput(query="zzz_no_such_term_zzz"))
    assert out.count == 0
    assert out.note


# -- knowledge_research (Phase 5) -------------------------------------------

OBSERVATION = (
    "Found a kerberos SPN on the active directory domain; "
    "considering kerberoasting the service account with impacket."
)


def test_research_extracts_signals_from_vocab(tmp_path, seeded_db):
    """Signals are controlled-vocabulary terms found in the text — nothing else."""
    tools = make_tools(tmp_path, seeded_db)
    out = tools.knowledge_research(ResearchInput(observation=OBSERVATION))
    assert "kerberoasting" in out.signals.techniques
    assert "impacket" in out.signals.tools
    assert {"active directory", "kerberos"} <= set(out.signals.services)


def test_research_references_are_real_chunks(tmp_path, seeded_db):
    """Every reference resolves back to real indexed content."""
    tools = make_tools(tmp_path, seeded_db)
    out = tools.knowledge_research(ResearchInput(observation=OBSERVATION))
    assert out.references
    for item in out.references:
        assert item.ref.chunk_id > 0
        src = tools.knowledge_source(GetSourceInput(chunk_id=item.ref.chunk_id))
        assert src.count == 1
        assert src.excerpts[0].text


def test_research_technique_brief_without_graph(tmp_path, seeded_db):
    """A detected technique resolves to the vocabulary even before a graph build."""
    tools = make_tools(tmp_path, seeded_db)
    out = tools.knowledge_research(ResearchInput(observation=OBSERVATION))
    brief = next(b for b in out.techniques if b.technique == "kerberoasting")
    assert brief.resolved is True
    assert brief.in_graph is False
    assert brief.documented_by == []


def test_research_technique_brief_enriched_by_graph(tmp_path, seeded_db):
    """After a graph build, the brief carries evidence-linked source edges."""
    GraphBuilder(seeded_db).rebuild()
    tools = make_tools(tmp_path, seeded_db)
    out = tools.knowledge_research(ResearchInput(observation=OBSERVATION))
    brief = next(b for b in out.techniques if b.technique == "kerberoasting")
    assert brief.in_graph is True
    assert any(g.name == "HackTricks" for g in brief.documented_by)
    for g in brief.documented_by:
        assert g.evidence is not None and g.evidence.doc_id is not None


def test_research_explicit_techniques_merged_and_resolved(tmp_path, seeded_db):
    """Explicit technique aliases are canonicalised and merged with detected ones."""
    tools = make_tools(tmp_path, seeded_db)
    out = tools.knowledge_research(
        ResearchInput(observation="unrelated preamble", techniques=["kerberoast"])
    )
    # 'kerberoast' resolves to the canonical 'kerberoasting'.
    assert any(b.technique == "kerberoasting" for b in out.techniques)


def test_research_includes_related_cases(tmp_path, seeded_db):
    tools = make_tools(tmp_path, seeded_db)
    out = tools.knowledge_research(ResearchInput(observation=OBSERVATION))
    assert any(c.title == "HTB: Forest" for c in out.related_cases)
    assert all(c.ref.chunk_id > 0 for c in out.related_cases)


def test_research_can_skip_cases(tmp_path, seeded_db):
    tools = make_tools(tmp_path, seeded_db)
    out = tools.knowledge_research(
        ResearchInput(observation=OBSERVATION, include_cases=False)
    )
    assert out.related_cases == []


def test_research_unknown_observation_notes(tmp_path, seeded_db):
    tools = make_tools(tmp_path, seeded_db)
    out = tools.knowledge_research(
        ResearchInput(observation="zzz_no_such_term_zzz nothing here")
    )
    assert out.signals.techniques == []
    assert out.references == []
    assert out.note


# -- knowledge_context (Phase 5) --------------------------------------------


def test_context_create_and_get(tmp_path, seeded_db):
    tools = make_tools(tmp_path, seeded_db)
    created = tools.knowledge_context(
        ContextInput(action="create", case="op-forest",
                     target="10.10.10.161", platform="windows")
    )
    assert created.ok is True
    assert created.case is not None
    assert created.case.name == "op-forest"
    assert created.case.target == "10.10.10.161"
    assert created.case.observations == []

    got = tools.knowledge_context(ContextInput(action="get", case="op-forest"))
    assert got.ok is True
    assert got.case is not None
    assert got.case.case_id == created.case.case_id


def test_context_get_missing_case(tmp_path, seeded_db):
    tools = make_tools(tmp_path, seeded_db)
    out = tools.knowledge_context(ContextInput(action="get", case="nope"))
    assert out.ok is False
    assert out.case is None
    assert out.note


def test_context_add_observation(tmp_path, seeded_db):
    tools = make_tools(tmp_path, seeded_db)
    tools.knowledge_context(ContextInput(action="create", case="op-forest"))
    out = tools.knowledge_context(
        ContextInput(action="add", case="op-forest", kind="finding",
                     text="svc-alfresco is AS-REP roastable")
    )
    assert out.ok is True
    assert out.case is not None
    assert len(out.case.observations) == 1
    obs = out.case.observations[0]
    assert obs.kind == "finding"
    assert obs.status == "open"
    assert obs.text == "svc-alfresco is AS-REP roastable"


def test_context_add_requires_existing_case(tmp_path, seeded_db):
    tools = make_tools(tmp_path, seeded_db)
    out = tools.knowledge_context(
        ContextInput(action="add", case="ghost", text="orphan note")
    )
    assert out.ok is False
    assert "not found" in out.note.lower()


def test_context_update_observation_status(tmp_path, seeded_db):
    tools = make_tools(tmp_path, seeded_db)
    tools.knowledge_context(ContextInput(action="create", case="op-forest"))
    added = tools.knowledge_context(
        ContextInput(action="add", case="op-forest", kind="hypothesis",
                     text="kerberoasting will yield a crackable ticket")
    )
    obs_id = added.case.observations[0].obs_id
    out = tools.knowledge_context(
        ContextInput(action="update_observation", case="op-forest",
                     obs_id=obs_id, status="confirmed")
    )
    assert out.ok is True
    assert out.case.observations[0].status == "confirmed"


def test_context_update_rejects_foreign_observation(tmp_path, seeded_db):
    """An obs_id that belongs to another case is refused, not misattributed."""
    tools = make_tools(tmp_path, seeded_db)
    tools.knowledge_context(ContextInput(action="create", case="case-a"))
    tools.knowledge_context(ContextInput(action="create", case="case-b"))
    added = tools.knowledge_context(
        ContextInput(action="add", case="case-a", text="belongs to A")
    )
    foreign_id = added.case.observations[0].obs_id
    out = tools.knowledge_context(
        ContextInput(action="update_observation", case="case-b",
                     obs_id=foreign_id, status="resolved")
    )
    assert out.ok is False
    assert "not found" in out.note.lower()


def test_context_list_cases(tmp_path, seeded_db):
    tools = make_tools(tmp_path, seeded_db)
    tools.knowledge_context(ContextInput(action="create", case="case-a"))
    tools.knowledge_context(ContextInput(action="create", case="case-b"))
    tools.knowledge_context(
        ContextInput(action="add", case="case-a", text="one observation")
    )
    out = tools.knowledge_context(ContextInput(action="list"))
    assert out.ok is True
    names = {c.name for c in out.cases}
    assert {"case-a", "case-b"} <= names
    a = next(c for c in out.cases if c.name == "case-a")
    assert a.observation_count == 1


def test_context_persists_to_db(tmp_path, seeded_db):
    """Writes are committed: a fresh tools instance sees the same state."""
    tools = make_tools(tmp_path, seeded_db)
    tools.knowledge_context(
        ContextInput(action="create", case="op-forest", platform="windows")
    )
    tools.knowledge_context(
        ContextInput(action="add", case="op-forest", text="persisted note")
    )
    other = make_tools(tmp_path, seeded_db)
    out = other.knowledge_context(ContextInput(action="get", case="op-forest"))
    assert out.ok is True
    assert out.case.platform == "windows"
    assert [o.text for o in out.case.observations] == ["persisted note"]

