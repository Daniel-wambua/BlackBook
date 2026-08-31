"""Tests for the #9 UX bundle: case export, ATT&CK IDs, source hints, recency."""

from __future__ import annotations

from blackbook.config import Settings
from blackbook.knowledge.case_export import (
    build_case_state,
    export_filename,
    render_case_markdown,
)
from blackbook.knowledge.vocab import attack_id
from blackbook.mcp.schemas import ContextInput, GetSourceInput, SearchInput
from blackbook.mcp.tools import KnowledgeTools
from blackbook.retrieval.lexical import LexicalHit
from blackbook.retrieval.reranker import rerank
from blackbook.storage.models import Case, CaseObservation


def _tools(seeded_db) -> KnowledgeTools:
    return KnowledgeTools(seeded_db, Settings())


# -- case export --------------------------------------------------------------


def _seeded_case(db) -> None:
    with db.session():
        db.upsert_case(Case(name="acme-htb", target="10.10.10.10", platform="Windows"))
        case = db.get_case("acme-htb")
        db.add_observation(
            CaseObservation(case_id=int(case["case_id"]), kind="finding", text="AS-REP roastable account")
        )
        db.add_observation(
            CaseObservation(case_id=int(case["case_id"]), kind="note", text="SMB signing disabled")
        )


def test_render_case_markdown(seeded_db):
    _seeded_case(seeded_db)
    state = build_case_state(seeded_db, "acme-htb")
    md = render_case_markdown(state)
    assert md.startswith("# Case: acme-htb")
    assert "**Target:** 10.10.10.10" in md
    assert "**Platform:** Windows" in md
    assert "**Observations:** 2" in md
    assert "AS-REP roastable account" in md
    assert "## Timeline" in md
    assert "[open] finding" in md


def test_render_empty_case(seeded_db):
    with seeded_db.session():
        seeded_db.upsert_case(Case(name="empty-case"))
    state = build_case_state(seeded_db, "empty-case")
    assert "_No observations recorded._" in render_case_markdown(state)


def test_build_case_state_missing(seeded_db):
    assert build_case_state(seeded_db, "nope") is None


def test_export_filename_slugifies():
    assert export_filename("ACME HTB: Forest!!").startswith("acme-htb-forest-")
    assert export_filename("   ").startswith("case-")
    assert export_filename("x").endswith(".md")


def test_context_export_action(seeded_db):
    _seeded_case(seeded_db)
    tools = _tools(seeded_db)
    out = tools.knowledge_context(ContextInput(action="export", case="acme-htb"))
    assert out.ok and out.markdown.startswith("# Case: acme-htb")
    assert "blackbook case export" in out.note


def test_context_export_missing_case(seeded_db):
    tools = _tools(seeded_db)
    out = tools.knowledge_context(ContextInput(action="export", case="ghost"))
    assert not out.ok and "not found" in out.note


def test_context_export_requires_case(seeded_db):
    out = _tools(seeded_db).knowledge_context(ContextInput(action="export"))
    assert not out.ok and "'case' is required" in out.note


# -- MITRE ATT&CK IDs ----------------------------------------------------------


def test_attack_id_direct_and_alias():
    assert attack_id("kerberoasting") == "T1558.003"
    assert attack_id("kerberoast") == "T1558.003"  # alias resolves
    assert attack_id("pth") == "T1550.002"
    assert attack_id("DCSync") == "T1003.006"  # case-insensitive


def test_attack_id_unmapped_is_none():
    # "smb" is a service, not a technique; unknown terms never get a guess.
    assert attack_id("smb") is None
    assert attack_id("definitely-not-a-technique") is None
    assert attack_id("") is None


def test_technique_output_carries_attack_id(seeded_db):
    from blackbook.mcp.schemas import TechniqueInput

    tools = _tools(seeded_db)
    res = tools.knowledge_technique(TechniqueInput(technique="kerberoast"))
    assert res.attack_id == "T1558.003"
    res2 = tools.knowledge_technique(TechniqueInput(technique="not-a-real-technique"))
    assert res2.attack_id is None


# -- knowledge_source hint ------------------------------------------------------


def test_source_without_document_lists_documents(seeded_db):
    tools = _tools(seeded_db)
    out = tools.knowledge_source(GetSourceInput(source="0xdf"))
    assert out.count == 0
    assert "is a source, not a document" in out.note
    assert "HTB: Forest" in out.note  # real indexed document as a hint


def test_source_without_document_unknown_source(seeded_db):
    tools = _tools(seeded_db)
    out = tools.knowledge_source(GetSourceInput(source="typo-source"))
    assert "none indexed" in out.note


# -- recency boost --------------------------------------------------------------


def _dated_hit(doc_id: int, title: str, date: str | None, score: float = 0.5) -> LexicalHit:
    return LexicalHit(
        chunk_id=doc_id,
        doc_id=doc_id,
        text=f"text {doc_id} {title}",
        title=title,
        source_id="s",
        source_name="s",
        authority="trusted",
        bm25=0.0,
        score=score,
        metadata={"categories": ["htb"], "date": date},
    )


def test_recency_boost_promotes_newer_writeup():
    from datetime import date as _d

    this_year = str(_d.today().year) + "-01-01"
    old = _dated_hit(1, "Old box", "2015-06-01")
    new = _dated_hit(2, "New box", this_year)
    out = rerank(
        [old, new], query="box", limit=2, mode="case_similarity",
    )
    assert out[0].chunk_id == 2
    assert out[0].score > out[1].score


def test_recency_boost_requires_case_mode():
    # In a non-case mode the dates must not influence ranking at all.
    old = _dated_hit(1, "Old box", "2015-06-01", score=0.6)
    new = _dated_hit(2, "New box", "3000-01-01", score=0.5)
    out = rerank([old, new], query="box", limit=2, mode="keyword")
    assert out[0].chunk_id == 1


def test_recency_boost_no_effect_undated_or_beyond_horizon():
    # Within case mode: an undated hit and one older than the horizon score
    # identically — recency only differentiates *within* the horizon.
    undated = _dated_hit(1, "A", None)
    ancient = _dated_hit(2, "B", "2001-01-01")
    a = rerank([undated], query="q", limit=1, mode="case_similarity")[0].score
    b = rerank([ancient], query="q", limit=1, mode="case_similarity")[0].score
    assert a == b


def test_recency_boost_ignores_garbage_dates():
    hit = _dated_hit(1, "Weird", "not-a-date")
    no_date = _dated_hit(2, "Weird", None)
    a = rerank([hit], query="q", limit=1, mode="case_similarity")[0].score
    b = rerank([no_date], query="q", limit=1, mode="case_similarity")[0].score
    assert a == b  # unparseable date behaves like no date
