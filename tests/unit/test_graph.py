"""Tests for the evidence-linked knowledge-graph builder.

Every assertion here pins the graph produced from the ``seeded_db`` fixture: two
documents (a HackTricks Kerberoasting page and an 0xdf HTB: Forest writeup) whose
exact text/section headings are known, so the extracted entities and edges are
fully determined. The suite also guards the two invariants that matter most:
the graph fabricates nothing, and retrieval-critical properties (idempotence,
empty-corpus safety, source scoping, evidence links) hold.

The last section covers :func:`writeup_coverage`, the diagnostic for a gap the
graph's own totals cannot show: ``GraphStats.writeups`` is one number, so a
corpus whose writeups all come from a single source while its largest source
contributes none looks fully populated. The seeded fixture is that situation in
miniature, and the tests pin three properties of the answer: it agrees with the
builder when the graph is a full rebuild, it is correct with no graph at all,
and it does not accept a source's name as evidence.

Before that, a section pins the co-occurrence collapse: ``uses``/``targets``
edges are stored once per pair with a ``support`` count rather than once per
witnessing document. Those tests seed a second corpus (two documents claiming
the same technique/tool pair) inside the test, because the shared ``seeded_db``
fixture happens not to make the same claim twice, and the collapse is exactly
about what happens when it does.
"""

from __future__ import annotations

import json

import pytest

from blackbook.knowledge import graph
from blackbook.knowledge.graph import (
    E_SOURCE,
    E_TECHNIQUE,
    E_WRITEUP,
    P_DEMONSTRATED_IN,
    P_DOCUMENTED_BY,
    P_USES,
    GraphBuilder,
    is_writeup_document,
    writeup_coverage,
)
from blackbook.knowledge.vocab import SERVICE_TERMS, TECHNIQUE_TERMS, TOOL_TERMS
from blackbook.storage import Chunk, Document, Source
from blackbook.storage.database import sha256_text


def _edge(db, name, entity_type, predicate):
    ent = db.get_entity(name, entity_type)
    assert ent is not None, f"missing entity {name!r} ({entity_type})"
    return db.entity_relationships(int(ent["entity_id"]), predicate=predicate)


def test_rebuild_stats(seeded_db):
    stats = GraphBuilder(seeded_db).rebuild()
    assert stats.documents == 2
    assert stats.writeups == 1
    assert stats.entities == 8
    assert stats.relationships == 7
    assert stats.by_entity_type == {
        "source": 2, "technique": 2, "service": 2, "tool": 1, "writeup": 1,
    }
    assert stats.by_predicate == {
        "documented_by": 3, "uses": 1, "targets": 2, "demonstrated_in": 1,
    }


def test_no_fabricated_entities(seeded_db):
    GraphBuilder(seeded_db).rebuild()
    allowed_terms = set(SERVICE_TERMS) | set(TECHNIQUE_TERMS) | set(TOOL_TERMS)
    real_sources = {"HackTricks", "0xdf"}
    real_titles = {"Kerberoasting", "HTB: Forest"}
    for e in seeded_db.list_entities():
        name, etype = e["name"], e["entity_type"]
        if etype == E_SOURCE:
            assert name in real_sources
        elif etype == E_WRITEUP:
            assert name in real_titles
        else:
            # technique / tool / service must be a literal vocabulary term.
            assert name in allowed_terms, f"fabricated {etype} entity {name!r}"


def test_technique_documented_by_source_is_heading_confidence(seeded_db):
    GraphBuilder(seeded_db).rebuild()
    edges = _edge(seeded_db, "kerberoasting", E_TECHNIQUE, P_DOCUMENTED_BY)
    assert len(edges) == 1
    e = edges[0]
    assert e["other_name"] == "HackTricks"
    assert e["other_type"] == E_SOURCE
    assert e["confidence"] == pytest.approx(0.9)   # named in a heading
    assert e["inferred"] == 1                        # keyword-derived
    # Evidence points at the real HackTricks document.
    assert e["evidence_title"] == "Kerberoasting"
    assert e["evidence_source_id"] == "hacktricks"
    assert e["evidence_doc_id"] is not None


def test_writeup_structural_edge_is_not_inferred(seeded_db):
    GraphBuilder(seeded_db).rebuild()
    edges = _edge(seeded_db, "HTB: Forest", E_WRITEUP, P_DOCUMENTED_BY)
    assert len(edges) == 1
    e = edges[0]
    assert e["other_name"] == "0xdf"
    assert e["other_type"] == E_SOURCE
    assert e["confidence"] == pytest.approx(1.0)
    assert e["inferred"] == 0    # structural, not a keyword guess


def test_asrep_demonstrated_in_writeup(seeded_db):
    GraphBuilder(seeded_db).rebuild()
    edges = _edge(seeded_db, "as-rep roasting", E_TECHNIQUE, P_DEMONSTRATED_IN)
    assert len(edges) == 1
    e = edges[0]
    assert e["other_name"] == "HTB: Forest"
    assert e["other_type"] == E_WRITEUP
    assert e["confidence"] == pytest.approx(0.6)   # body-only mention
    assert e["inferred"] == 1
    assert e["evidence_external_id"] == "htb-forest"


def test_cooccurrence_uses_edge(seeded_db):
    GraphBuilder(seeded_db).rebuild()
    edges = _edge(seeded_db, "kerberoasting", E_TECHNIQUE, P_USES)
    assert len(edges) == 1
    e = edges[0]
    assert e["other_name"] == "impacket"
    assert e["confidence"] == pytest.approx(0.5)
    assert e["inferred"] == 1


def test_every_inferred_edge_has_evidence(seeded_db):
    GraphBuilder(seeded_db).rebuild()
    for ent in seeded_db.list_entities():
        for rel in seeded_db.entity_relationships(int(ent["entity_id"])):
            if rel["inferred"] == 1:
                assert rel["evidence_doc_id"] is not None, (
                    f"inferred edge {rel['predicate']} without evidence"
                )


def test_rebuild_is_idempotent(seeded_db):
    b = GraphBuilder(seeded_db)
    first = b.rebuild().as_dict()
    second = b.rebuild().as_dict()
    assert first == second
    # A second builder over the already-built corpus reproduces it too.
    assert GraphBuilder(seeded_db).rebuild().as_dict() == first


def test_empty_corpus_builds_empty_graph(db):
    stats = GraphBuilder(db).rebuild()
    assert stats.documents == 0
    assert stats.entities == 0
    assert stats.relationships == 0


def test_source_scoped_rebuild(seeded_db):
    stats = GraphBuilder(seeded_db).rebuild(source_ids=["hacktricks"])
    assert stats.documents == 1
    assert stats.writeups == 0
    # The 0xdf-only technique never appears when 0xdf is out of scope.
    assert seeded_db.get_entity("as-rep roasting", E_TECHNIQUE) is None
    assert seeded_db.get_entity("kerberoasting", E_TECHNIQUE) is not None


# -- writeup coverage ------------------------------------------------------


def test_coverage_agrees_with_the_builder_after_a_full_rebuild(seeded_db):
    """The two counts are computed differently and must land on the same number.

    The builder decides writeup-hood while walking documents; the diagnostic
    re-derives it from the stored rows. If those disagreed the diagnostic would
    be describing something other than the graph it reports on, so the
    agreement is asserted rather than assumed.
    """
    stats = GraphBuilder(seeded_db).rebuild()
    coverage = writeup_coverage(seeded_db)
    assert coverage.eligible == stats.writeups == 1
    assert coverage.graph_writeups == stats.writeups
    assert coverage.graph_built is True


def test_coverage_names_the_sources_that_contribute_none(seeded_db):
    coverage = writeup_coverage(seeded_db)
    assert coverage.documents == 2
    assert coverage.docs_by_source == {"hacktricks": 1, "0xdf": 1}
    # hacktricks carries a zero rather than being absent, so a caller cannot
    # mistake "contributes none" for "not present in this corpus".
    assert coverage.by_source == {"hacktricks": 0, "0xdf": 1}
    assert coverage.sources_with_writeups == 1
    assert coverage.gaps == [
        {
            "source_id": "hacktricks",
            "documents": 1,
            "reason": (
                "none of the 1 documents carries a writeup category marker, "
                "a machine_name, or a kind"
            ),
        }
    ]


def test_coverage_is_correct_with_no_graph_at_all(seeded_db):
    """The diagnostic must not depend on the thing it reports on.

    Coverage is read from the indexed documents, so it answers before a graph
    exists. ``graph_built`` is what separates "not built yet" from "built and
    disagrees" — the graph is optional, so a missing one is not a fault.
    """
    coverage = writeup_coverage(seeded_db)
    assert coverage.graph_built is False
    assert coverage.graph_writeups == 0
    assert seeded_db.counts()["entities"] == 0
    assert coverage.eligible == 1
    assert coverage.gaps[0]["source_id"] == "hacktricks"


def test_coverage_reports_a_scoped_rebuild_as_a_disagreement(seeded_db):
    """A scoped rebuild legitimately leaves the graph short of the corpus.

    This is the case the doctor check escalates on, so it is pinned here: the
    graph is built (hacktricks yields entities) but holds no writeups while one
    document in the corpus qualifies.
    """
    GraphBuilder(seeded_db).rebuild(source_ids=["hacktricks"])
    coverage = writeup_coverage(seeded_db)
    assert coverage.graph_built is True
    assert coverage.graph_writeups == 0
    assert coverage.eligible == 1


def test_coverage_does_not_trust_a_source_name_that_says_writeup(db):
    """A name is not evidence, which is exactly how the real gap arises.

    ``htb_writeups`` is a source whose name promises writeups. Its documents are
    ordinary reference rows carrying no category marker and no HTB card, so the
    rule declines every one of them and the source reports zero. Nothing about
    the source id reaches the decision except the literal ``0xdf`` test, so this
    test also pins that no other name gets special treatment.
    """
    with db.session():
        db.upsert_source(Source(source_id="htb_writeups", name="HTB Writeups", authority="trusted"))
        db.upsert_document(
            Document(
                source_id="htb_writeups",
                external_id="forest",
                title="Forest",
                content_hash=sha256_text("forest"),
                categories=["bug-bounty"],
            )
        )
    coverage = writeup_coverage(db)
    assert coverage.by_source == {"htb_writeups": 0}
    assert coverage.eligible == 0
    assert [g["source_id"] for g in coverage.gaps] == ["htb_writeups"]
    assert is_writeup_document("htb_writeups", ["bug-bounty"], {}) is False


def test_coverage_on_an_empty_corpus_is_empty_not_misleading(db):
    coverage = writeup_coverage(db)
    assert coverage.documents == 0
    assert coverage.eligible == 0
    assert coverage.docs_by_source == {}
    assert coverage.by_source == {}
    assert coverage.gaps == []
    assert coverage.sources_with_writeups == 0
    assert coverage.graph_built is False


# -- co-occurrence collapse ------------------------------------------------


def _seed_shared_technique(db, *, extra_tool_in_first: str = "") -> tuple[int, int]:
    """Two documents that both mention kerberoasting and Impacket.

    ``extra_tool_in_first`` adds a tool the first document mentions and the
    second does not, which is how a test gets two ``uses`` edges with different
    support to check the tie-break ordering against. Returns both doc ids.
    """
    with db.session():
        db.upsert_source(Source(source_id="hacktricks", name="HackTricks", authority="trusted"))
        db.upsert_source(Source(source_id="0xdf", name="0xdf", authority="trusted"))
        d1 = db.upsert_document(
            Document(
                source_id="hacktricks",
                external_id="ad/kerberoasting.md",
                title="Kerberoasting",
                content_hash=sha256_text("kerb"),
                categories=["active-directory"],
            )
        )
        d2 = db.upsert_document(
            Document(
                source_id="0xdf",
                external_id="htb-forest",
                title="HTB: Forest",
                content_hash=sha256_text("forest"),
                categories=["htb"],
            )
        )
        db.replace_chunks(
            d1,
            [
                Chunk(
                    doc_id=d1, ordinal=0,
                    text="Impacket GetUserSPNs.py does kerberoasting. " + extra_tool_in_first,
                    section_path=["Kerberoasting"],
                    token_estimate=10, content_hash=sha256_text("k1"),
                )
            ],
        )
        db.replace_chunks(
            d2,
            [
                Chunk(
                    doc_id=d2, ordinal=0,
                    text="Impacket GetUserSPNs.py does kerberoasting.",
                    section_path=["HTB: Forest"],
                    token_estimate=10, content_hash=sha256_text("k2"),
                )
            ],
        )
    return d1, d2


def test_cooccurrence_edges_collapse_to_one_row_that_counts_its_witnesses(db):
    """Two documents claiming the same pair are one claim, not two edges.

    The pair is stored once, ``support`` says how many documents made it, and the
    citation is the lower doc id so the stored edge does not depend on the order
    the documents were indexed in.
    """
    d1, d2 = _seed_shared_technique(db)
    GraphBuilder(db).rebuild()
    edges = _edge(db, "kerberoasting", E_TECHNIQUE, P_USES)
    assert len(edges) == 1
    e = edges[0]
    assert e["other_name"] == "impacket"
    assert e["support"] == 2
    assert e["evidence_doc_id"] == min(d1, d2)
    # One row in the table, not one per witness.
    rows = db.conn.execute(
        "SELECT COUNT(*) FROM relationships WHERE predicate = ?", (P_USES,)
    ).fetchone()[0]
    assert rows == 1


def test_collapse_is_order_independent(db):
    """The surviving citation is the lowest doc id whichever way the docs land.

    A scoped rebuild can visit a subset of the corpus, so the first witness the
    builder meets is not fixed. Picking the minimum doc id makes the stored edge
    a function of the corpus rather than of the iteration order.
    """
    d1, d2 = _seed_shared_technique(db)
    GraphBuilder(db).rebuild(source_ids=["0xdf"])
    assert _edge(db, "kerberoasting", E_TECHNIQUE, P_USES)[0]["evidence_doc_id"] == d2
    GraphBuilder(db).rebuild()
    assert _edge(db, "kerberoasting", E_TECHNIQUE, P_USES)[0]["evidence_doc_id"] == d1


def test_documentary_edges_keep_one_row_per_document(db):
    """The collapse is scoped to co-occurrence; citations are not lost to it.

    A technique being named by two sources is two facts with two citations, so
    each keeps its own row and its support stays 1.
    """
    d1, d2 = _seed_shared_technique(db)
    GraphBuilder(db).rebuild()
    edges = _edge(db, "kerberoasting", E_TECHNIQUE, P_DOCUMENTED_BY)
    assert len(edges) == 2
    assert {e["evidence_doc_id"] for e in edges} == {d1, d2}
    assert {e["other_name"] for e in edges} == {"HackTricks", "0xdf"}
    assert all(e["support"] == 1 for e in edges)


def test_collapsed_edge_still_cites_a_real_document(db):
    """A collapsed edge must remain citable, since evidence is the invariant.

    The key that folds the witnesses together has no room for a document, so the
    representative id has to survive on the edge itself. An edge that carried
    support but no citation would be a claim the graph cannot point at.
    """
    _seed_shared_technique(db)
    GraphBuilder(db).rebuild()
    e = _edge(db, "kerberoasting", E_TECHNIQUE, P_USES)[0]
    assert e["evidence_doc_id"] is not None
    assert e["evidence_title"] == "Kerberoasting"
    assert e["evidence_source_id"] == "hacktricks"


def test_better_supported_edges_are_ordered_first(db):
    """Support breaks ties that confidence cannot, because every co-occurrence
    edge carries the same tier. This is what stops a claim witnessed by two
    documents from lining up next to one witnessed by one, indistinguishable."""
    _seed_shared_technique(db, extra_tool_in_first="Nmap scans the host.")
    GraphBuilder(db).rebuild()
    edges = _edge(db, "kerberoasting", E_TECHNIQUE, P_USES)
    assert [(e["other_name"], e["support"]) for e in edges] == [
        ("impacket", 2),
        ("nmap", 1),
    ]
    assert len({e["confidence"] for e in edges}) == 1  # confidence alone ties


# ---------------------------------------------------------------------------
# Incremental rebuild
# ---------------------------------------------------------------------------
#
# Term extraction is the expensive half of a build: a regex pass per vocabulary
# term over every document's headings and its full body text, which on a real
# corpus is about nine tenths of the wall clock. So it is cached per document
# and replayed while the document has not changed.
#
# Two properties make that safe to rely on, and they are what these tests pin.
# First, an incremental rebuild has to produce exactly the graph a full one
# would: the moment it can diverge, every later build is suspect. Second, the
# skip has to be real. A "no change" path that quietly re-extracts everything
# would still pass every correctness test while buying nothing.


def _snapshot(db):
    """Everything the graph asserts, as comparable tuples.

    Read straight from the tables rather than through a read API: the question is
    what is stored, so a projection would only hide the difference this is meant
    to catch. ``evidence_doc_id`` is included because a citation that drifted
    between two builds is exactly the kind of silent damage to look for.
    """
    entities = sorted((e["name"], e["entity_type"]) for e in db.list_entities())
    rows = db.conn.execute(
        "SELECT s.name, r.predicate, o.name, r.confidence, r.inferred, r.support,"
        "       r.evidence_doc_id"
        " FROM relationships r"
        " JOIN entities s ON s.entity_id = r.subject_id"
        " JOIN entities o ON o.entity_id = r.object_id"
    ).fetchall()
    return entities, sorted(tuple(r) for r in rows)


def _count_extractions(monkeypatch, calls):
    """Record which documents the builder actually re-extracts."""
    real = GraphBuilder.extract_document_terms

    def counted(self, doc):
        calls.append(int(doc["doc_id"]))
        return real(self, doc)

    monkeypatch.setattr(GraphBuilder, "extract_document_terms", counted)
    return calls


def _refuse_extraction(monkeypatch):
    """Make re-extraction an error, so a test can prove the cache was used."""

    def boom(self, doc):
        raise AssertionError(f"re-extracted document {doc['doc_id']}")

    monkeypatch.setattr(GraphBuilder, "extract_document_terms", boom)


def _rewrite(db, doc_id, text, **fields):
    """Change a document's body the way an ingesting pipeline would.

    The content hash moves with the text, which is what makes the second half of
    the fingerprint agree that this document is new.
    """
    row = db.get_document(doc_id)
    with db.session():
        db.upsert_document(Document(
            source_id=row["source_id"],
            external_id=row["external_id"],
            title=fields.get("title", row["title"]),
            url=row["url"],
            content_hash=sha256_text(text),
            metadata=json.loads(row["metadata"]),
            categories=json.loads(row["categories"]),
        ))
        db.replace_chunks(doc_id, [Chunk(
            doc_id=doc_id, ordinal=0, text=text, section_path=[],
            token_estimate=len(text.split()), content_hash=sha256_text(text),
        )])


def test_incremental_rebuild_matches_a_full_one(seeded_db):
    """The invariant everything else rests on. One document changes, the
    incremental rebuild runs, and the graph it leaves must be byte-identical to
    the one a from-scratch build produces."""
    builder = GraphBuilder(seeded_db)
    builder.rebuild()
    _rewrite(seeded_db, 1, "SQL injection in the login form, dumped with sqlmap.")

    incremental = builder.rebuild()
    assert incremental.skipped is False
    assert incremental.reused == 1  # document 2 was untouched
    after_incremental = _snapshot(seeded_db)

    builder.rebuild(incremental=False)
    assert _snapshot(seeded_db) == after_incremental


def test_an_unchanged_corpus_skips_the_rebuild(seeded_db, monkeypatch):
    builder = GraphBuilder(seeded_db)
    builder.rebuild()
    before = _snapshot(seeded_db)

    _refuse_extraction(monkeypatch)
    stats = builder.rebuild()

    assert stats.skipped is True
    assert stats.reused == stats.documents == 2
    assert _snapshot(seeded_db) == before


def test_only_the_changed_document_is_re_extracted(seeded_db, monkeypatch):
    builder = GraphBuilder(seeded_db)
    builder.rebuild()
    _rewrite(seeded_db, 1, "SQL injection in the login form, dumped with sqlmap.")

    calls = _count_extractions(monkeypatch, [])
    stats = builder.rebuild()

    assert calls == [1]
    assert stats.reused == 1
    assert stats.documents == 2


def test_a_changed_document_reaches_the_graph(seeded_db):
    builder = GraphBuilder(seeded_db)
    builder.rebuild()
    assert seeded_db.get_entity("sql injection", E_TECHNIQUE) is None
    assert seeded_db.get_entity("sqlmap", "tool") is None

    _rewrite(seeded_db, 1, "SQL injection in the login form, dumped with sqlmap.")
    builder.rebuild()

    assert seeded_db.get_entity("sql injection", E_TECHNIQUE) is not None
    assert seeded_db.get_entity("sqlmap", "tool") is not None


def test_a_title_change_alone_invalidates_the_cache(seeded_db, monkeypatch):
    """A citation refresh updates a title in place and keeps the content hash, so
    the fingerprint has to cover the title itself. It matters: a writeup entity
    is named after its document's title, so a title the cache did not notice
    would leave the graph naming a document by a name it no longer has."""
    builder = GraphBuilder(seeded_db)
    builder.rebuild()

    with seeded_db.session():
        seeded_db.conn.execute(
            "UPDATE documents SET title = ? WHERE doc_id = 2", ("HTB: Forest (v2)",)
        )
    calls = _count_extractions(monkeypatch, [])
    builder.rebuild()

    assert calls == [2]
    assert seeded_db.get_entity("HTB: Forest (v2)", E_WRITEUP) is not None
    assert seeded_db.get_entity("HTB: Forest", E_WRITEUP) is None


def test_a_deleted_document_leaves_the_graph(seeded_db):
    builder = GraphBuilder(seeded_db)
    builder.rebuild()
    assert seeded_db.get_entity("HTB: Forest", E_WRITEUP) is not None

    with seeded_db.session():
        seeded_db.conn.execute("DELETE FROM documents WHERE doc_id = 2")
    stats = builder.rebuild()

    assert stats.documents == 1
    assert stats.skipped is False
    assert seeded_db.get_entity("HTB: Forest", E_WRITEUP) is None
    rebuilt = _snapshot(seeded_db)
    builder.rebuild(incremental=False)
    assert _snapshot(seeded_db) == rebuilt


def test_a_scope_change_is_not_mistaken_for_an_unchanged_corpus(seeded_db):
    """A build is current only for the scope it was made under. Rebuilding one
    source after building all of them has to narrow the graph, and rebuilding
    everything after building one has to widen it back."""
    GraphBuilder(seeded_db).rebuild()

    narrow = GraphBuilder(seeded_db).rebuild(source_ids=["hacktricks"])
    assert narrow.skipped is False
    assert narrow.documents == 1
    assert seeded_db.get_entity("HTB: Forest", E_WRITEUP) is None

    wide = GraphBuilder(seeded_db).rebuild()
    assert wide.skipped is False
    assert wide.documents == 2
    assert seeded_db.get_entity("HTB: Forest", E_WRITEUP) is not None


def test_a_widened_scope_over_the_same_documents_still_rebuilds(seeded_db):
    """The scope is checked on its own, not just through the documents it
    selects. Two scopes can name the same set of documents (here, by adding a
    source that contributes none), so corpus fingerprint and document count
    alone would call the build current and leave the record claiming a corpus it
    was not built from. The recorded scope is what a caller reads to find out
    what the graph covers, so it follows the call.
    """
    with seeded_db.session():
        seeded_db.upsert_source(Source(source_id="empty", name="Empty"))
    GraphBuilder(seeded_db).rebuild(source_ids=["hacktricks"])

    stats = GraphBuilder(seeded_db).rebuild(source_ids=["hacktricks", "empty"])

    assert stats.skipped is False
    assert stats.reused == 1  # the terms still came from the cache
    assert seeded_db.get_graph_build()["sources"] == ["empty", "hacktricks"]


def test_an_emptied_relationship_table_is_rebuilt_from_the_cache(
    seeded_db, monkeypatch
):
    """The build record and the term cache both outlive a cleared graph, so the
    record alone cannot say whether a graph is still there. When the rows are
    gone the graph has to be assembled again, and the cached terms are what make
    that cheap."""
    builder = GraphBuilder(seeded_db)
    builder.rebuild()
    before = _snapshot(seeded_db)

    with seeded_db.session():
        seeded_db.conn.execute("DELETE FROM relationships")

    _refuse_extraction(monkeypatch)
    stats = builder.rebuild()

    assert stats.skipped is False  # the graph was gone, so it was rebuilt
    assert stats.reused == 2  # ...without reading any document text again
    assert _snapshot(seeded_db) == before


def test_an_unreadable_cache_blob_is_re_extracted_not_fatal(seeded_db, monkeypatch):
    builder = GraphBuilder(seeded_db)
    builder.rebuild()
    before = _snapshot(seeded_db)

    with seeded_db.session():
        seeded_db.conn.execute(
            "UPDATE document_graph SET terms = 'not json' WHERE doc_id = 1"
        )
        seeded_db.conn.execute("DELETE FROM relationships")

    calls = _count_extractions(monkeypatch, [])
    stats = builder.rebuild()

    assert calls == [1]  # only the row that could not be read
    assert stats.reused == 1
    assert _snapshot(seeded_db) == before


def test_a_terms_version_bump_invalidates_every_cache_row(seeded_db, monkeypatch):
    """The vocabulary itself changes between versions. Without the version in the
    fingerprint, a rebuild would keep replaying terms the current code would no
    longer produce."""
    GraphBuilder(seeded_db).rebuild()
    monkeypatch.setattr(
        graph, "_TERMS_CACHE_VERSION", graph._TERMS_CACHE_VERSION + 1
    )

    calls = _count_extractions(monkeypatch, [])
    stats = GraphBuilder(seeded_db).rebuild()

    assert calls == [1, 2]
    assert stats.reused == 0
