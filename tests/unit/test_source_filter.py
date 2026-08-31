"""Regression tests for source-filter resolution.

A typo'd source filter must never *widen* a query to the whole corpus: an
empty resolved source list means "search nothing" and is surfaced as an
explicit note, not silently treated as "no filter".
"""

from __future__ import annotations

import pytest

from blackbook.config import Settings
from blackbook.mcp.schemas import (
    CaseSearchInput,
    ResearchInput,
    SearchInput,
    TechniqueInput,
)
from blackbook.mcp.tools import KnowledgeTools
from blackbook.retrieval.hybrid import HybridRetriever
from blackbook.storage.database import Database


@pytest.fixture()
def db(tmp_path):
    with Database(tmp_path / "t.db") as d:
        _seed(d)
        yield d


def _seed(db: Database) -> None:
    from blackbook.storage.models import Chunk, Document, Source

    with db.session():
        db.upsert_source(Source(source_id="hacktricks", name="HackTricks",
                                authority="trusted"))
        db.upsert_source(Source(source_id="0xdf", name="0xdf",
                                authority="trusted"))
        doc = Document(source_id="hacktricks", external_id="a",
                       title="Kerberoasting", content_hash="h1")
        doc_id = db.upsert_document(doc)
        db.replace_chunks(doc_id, [
            Chunk(doc_id=doc_id, ordinal=0, text="kerberoasting attacks",
                  section_path=["Kerberoasting"], content_hash="c1"),
        ])


def test_source_ids_none_means_all():
    s = Settings()
    assert s.source_ids(None) is None
    assert s.source_ids([]) is None
    assert s.source_ids(["all"]) is None


def test_source_ids_typo_yields_empty_not_widening():
    s = Settings()
    assert s.source_ids(["hacktrick"]) == []
    assert s.source_ids(["hacktricks", "typo"]) == ["hacktricks"]


def test_fts_empty_source_list_matches_nothing(db):
    assert db.fts_search('"kerberoasting"', source_ids=[]) == []
    assert db.fts_search('"kerberoasting"', source_ids=None)


def test_load_embeddings_empty_source_list(db):
    assert db.load_embeddings("m", source_ids=[]) == ([], [])


def test_iter_documents_empty_source_list(db):
    assert list(db.iter_documents([])) == []
    assert list(db.iter_documents(None))


def test_search_typoed_source_returns_note_not_everything(db, tmp_path):
    tools = KnowledgeTools(db, Settings())
    out = tools.knowledge_search(
        SearchInput(query="kerberoasting", sources=["hacktrick"])
    )
    assert out.count == 0
    assert out.results == []
    assert "hacktrick" in out.note
    assert "nothing was searched" in out.note


def test_search_valid_source_still_filters(db):
    tools = KnowledgeTools(db, Settings())
    out = tools.knowledge_search(
        SearchInput(query="kerberoasting", sources=["hacktricks"])
    )
    assert out.count == 1
    assert out.sources_searched == ["hacktricks"]


def test_technique_typoed_source_note(db):
    tools = KnowledgeTools(db, Settings())
    out = tools.knowledge_technique(
        TechniqueInput(technique="kerberoasting", sources=["nope"])
    )
    assert not out.in_graph
    assert "nothing was searched" in out.note


def test_case_search_typoed_source_note(db):
    tools = KnowledgeTools(db, Settings())
    out = tools.knowledge_case_search(
        CaseSearchInput(query="kerberoasting", sources=["nope"])
    )
    assert out.count == 0
    assert "nothing was searched" in out.note


def test_research_typoed_source_note(db):
    tools = KnowledgeTools(db, Settings())
    out = tools.knowledge_research(
        ResearchInput(observation="smb and kerberoasting", sources=["nope"])
    )
    assert out.references == []
    assert "nothing was searched" in out.note


def test_hybrid_retriever_empty_sources(db, tmp_path):
    r = HybridRetriever(db, Settings())
    assert r.search("kerberoasting", source_ids=[]) == []
