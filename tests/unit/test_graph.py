"""Tests for the evidence-linked knowledge-graph builder.

Every assertion here pins the graph produced from the ``seeded_db`` fixture: two
documents (a HackTricks Kerberoasting page and an 0xdf HTB: Forest writeup) whose
exact text/section headings are known, so the extracted entities and edges are
fully determined. The suite also guards the two invariants that matter most:
the graph fabricates nothing, and retrieval-critical properties (idempotence,
empty-corpus safety, source scoping, evidence links) hold.
"""

from __future__ import annotations

import pytest

from blackbook.knowledge.graph import (
    E_SOURCE,
    E_TECHNIQUE,
    E_WRITEUP,
    P_DEMONSTRATED_IN,
    P_DOCUMENTED_BY,
    P_USES,
    GraphBuilder,
)
from blackbook.knowledge.vocab import SERVICE_TERMS, TECHNIQUE_TERMS, TOOL_TERMS


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
