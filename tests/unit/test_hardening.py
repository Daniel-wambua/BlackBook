"""Adversarial / hardening tests for the MCP tool surface.

These exercise the guarantees the security spec demands at the *tool* boundary:

* free-text queries can't inject FTS5 syntax or crash the retriever, whatever
  punctuation, unicode, control characters, or SQL-looking text they contain;
* every result set is bounded — schema-capped at the input, and clamped again
  inside the retriever so an oversized request can't exhaust memory;
* malformed source references resolve to a clean "not found", never an error or
  a fabricated citation;
* ``knowledge_context`` writes only to the local case layer: it has no delete
  action, never mutates on a rejected call (no partial writes, no cross-case
  edits), and its observations never leak into the searchable index.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from blackbook.config import Settings, DatabaseConfig
from blackbook.eval import build_eval_corpus
from blackbook.mcp.schemas import ContextInput, GetSourceInput, SearchInput
from blackbook.mcp.tools import KnowledgeTools
from blackbook.retrieval import HybridRetriever
from blackbook.storage import Database


def make_tools(tmp_path, db):
    settings = Settings(home=tmp_path, database=DatabaseConfig(path=str(tmp_path / "d.db")))
    return KnowledgeTools(db, settings)


# --------------------------------------------------------------------------- #
# Query robustness: injection, empties, unicode, control chars                #
# --------------------------------------------------------------------------- #
FTS_INJECTIONS = [
    'kerberoasting" OR chunks_fts MATCH "x',
    "kerberoasting NEAR/2 spn",
    "title: kerberoasting AND text: *",
    'kerberoasting" ) ) ) --',
    "*",
    "^kerberoasting$",
    "'; DROP TABLE chunks; --",
    "kerberoast* AND (spn OR NOT tgs)",
]


def test_fts_injection_queries_never_raise(tmp_path, seeded_db):
    tools = make_tools(tmp_path, seeded_db)
    for q in FTS_INJECTIONS:
        out = tools.knowledge_search(SearchInput(query=q, limit=8))
        assert out.count >= 0
        assert len(out.results) <= 8


def test_contentless_queries_return_empty(tmp_path, seeded_db):
    tools = make_tools(tmp_path, seeded_db)
    for q in [" ", "!!!", "??? ...", "\t\n ", "-", "()"]:
        out = tools.knowledge_search(SearchInput(query=q))
        assert out.count == 0
        assert out.note  # a helpful note, not a crash


def test_unicode_and_control_chars(tmp_path, seeded_db):
    tools = make_tools(tmp_path, seeded_db)
    for q in ["🔐 kerberoasting", "домен kerberos", "kerber\x00oasting", "a\x01\x02b"]:
        out = tools.knowledge_search(SearchInput(query=q))
        assert out.count >= 0  # resolves cleanly regardless of encoding


def test_query_length_bounds(tmp_path, seeded_db):
    tools = make_tools(tmp_path, seeded_db)
    ok = tools.knowledge_search(SearchInput(query="kerberoasting " * 100))  # < 2000
    assert ok.count >= 0
    with pytest.raises(ValidationError):
        SearchInput(query="x" * 2001)
    with pytest.raises(ValidationError):
        SearchInput(query="")


# --------------------------------------------------------------------------- #
# Bounded output                                                              #
# --------------------------------------------------------------------------- #
def test_limit_is_schema_bounded():
    with pytest.raises(ValidationError):
        SearchInput(query="x", limit=51)
    with pytest.raises(ValidationError):
        SearchInput(query="x", limit=0)


def test_retriever_clamps_oversized_limit(tmp_path):
    db = Database(tmp_path / "e.db")
    try:
        build_eval_corpus(db)  # 68 chunks — more than max_limit
        r = HybridRetriever(db, Settings())
        results = r.search("kerberoasting service principal name crack", limit=10_000)
        assert len(results) <= 50  # RetrievalConfig.max_limit
    finally:
        db.close()


def test_max_excerpts_bounded():
    with pytest.raises(ValidationError):
        GetSourceInput(chunk_id=1, max_excerpts=21)
    with pytest.raises(ValidationError):
        GetSourceInput(chunk_id=1, max_excerpts=0)


# --------------------------------------------------------------------------- #
# Malformed source references                                                 #
# --------------------------------------------------------------------------- #
def test_knowledge_source_missing_or_invalid_refs(tmp_path, seeded_db):
    tools = make_tools(tmp_path, seeded_db)
    # Non-existent chunk id -> clean "not found", never a fabricated excerpt.
    miss = tools.knowledge_source(GetSourceInput(chunk_id=999999))
    assert miss.count == 0 and miss.note
    # Negative / non-existent doc id.
    assert tools.knowledge_source(GetSourceInput(chunk_id=-1)).count == 0
    assert tools.knowledge_source(GetSourceInput(doc_id=123456)).count == 0
    # No identifiers at all -> guidance note, no excerpts.
    empty = tools.knowledge_source(GetSourceInput())
    assert empty.count == 0 and empty.note


# --------------------------------------------------------------------------- #
# knowledge_context: local-only, no delete, no partial / cross mutation       #
# --------------------------------------------------------------------------- #
def test_context_has_no_delete_action():
    # The Literal action set must not admit a destructive verb.
    with pytest.raises(ValidationError):
        ContextInput(action="delete", case="c")


def test_context_add_to_missing_case_writes_nothing(tmp_path, seeded_db):
    tools = make_tools(tmp_path, seeded_db)
    out = tools.knowledge_context(ContextInput(action="add", case="ghost", text="x"))
    assert out.ok is False
    listing = tools.knowledge_context(ContextInput(action="list"))
    assert all(c.name != "ghost" for c in listing.cases)


def test_context_invalid_add_no_partial_write(tmp_path, seeded_db):
    tools = make_tools(tmp_path, seeded_db)
    tools.knowledge_context(ContextInput(action="create", case="c1", target="10.0.0.1"))
    bad = tools.knowledge_context(ContextInput(action="add", case="c1"))  # no text
    assert bad.ok is False
    state = tools.knowledge_context(ContextInput(action="get", case="c1"))
    assert state.ok and state.case is not None
    assert len(state.case.observations) == 0  # nothing was written


def test_context_roundtrip_persists(tmp_path, seeded_db):
    tools = make_tools(tmp_path, seeded_db)
    tools.knowledge_context(ContextInput(action="create", case="c2"))
    add = tools.knowledge_context(
        ContextInput(action="add", case="c2", text="found an SPN", kind="finding")
    )
    assert add.ok and add.case and len(add.case.observations) == 1
    obs_id = add.case.observations[0].obs_id
    upd = tools.knowledge_context(
        ContextInput(action="update_observation", case="c2", obs_id=obs_id, status="confirmed")
    )
    assert upd.ok
    got = tools.knowledge_context(ContextInput(action="get", case="c2"))
    assert got.case.observations[0].status == "confirmed"


def test_context_update_wrong_case_no_cross_mutation(tmp_path, seeded_db):
    tools = make_tools(tmp_path, seeded_db)
    tools.knowledge_context(ContextInput(action="create", case="ca"))
    tools.knowledge_context(ContextInput(action="create", case="cb"))
    add = tools.knowledge_context(
        ContextInput(action="add", case="ca", text="obs", kind="note")
    )
    obs_id = add.case.observations[0].obs_id
    # Update ca's observation while naming cb -> rejected, status untouched.
    out = tools.knowledge_context(
        ContextInput(action="update_observation", case="cb", obs_id=obs_id, status="confirmed")
    )
    assert out.ok is False
    got = tools.knowledge_context(ContextInput(action="get", case="ca"))
    assert got.case.observations[0].status == "open"


def test_context_writes_do_not_touch_index(tmp_path, seeded_db):
    """Case observations are local state — they must never become retrievable."""
    tools = make_tools(tmp_path, seeded_db)
    before = tools.knowledge_search(SearchInput(query="kerberoasting")).count
    tools.knowledge_context(ContextInput(action="create", case="c3"))
    tools.knowledge_context(
        ContextInput(action="add", case="c3", text="kerberoasting observation note")
    )
    after = tools.knowledge_search(SearchInput(query="kerberoasting")).count
    assert after == before  # the note did not enter the knowledge index
