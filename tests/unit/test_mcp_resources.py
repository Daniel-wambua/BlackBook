"""The resource and prompt surface: what is registered, and what it returns.

The tool set is pinned in test_server.py and stays pinned there. These tests
cover the second half of the MCP surface, which is easy to register and easy to
register wrongly: a resource whose body raises, a template whose parameter name
does not match the URI, a prompt whose declared argument is not the one the
function takes.
"""

from __future__ import annotations

import asyncio
import json

from blackbook.config import Settings, SourceConfig
from blackbook.mcp import prompts, resources
from blackbook.server import build_server
from blackbook.storage import Source
from blackbook.storage.database import Database


def _server(db=None, settings=None):
    return build_server(settings or Settings(), db or Database(":memory:"))


async def _reads(server, uri):
    return await server.read_resource(uri)


def _text(contents) -> str:
    """The text of a resource read, whichever way the SDK hands it back."""
    first = contents[0]
    return first.content if hasattr(first, "content") else str(first)


# -- registration ---------------------------------------------------------


def test_server_registers_the_expected_resources_and_prompts():
    db = Database(":memory:")
    try:
        server = _server(db)

        async def survey():
            listed = await server.list_resources()
            templates = await server.list_resource_templates()
            prompt_list = await server.list_prompts()
            return listed, templates, prompt_list

        listed, templates, prompt_list = asyncio.run(survey())

        assert {str(r.uri) for r in listed} == {
            "blackbook://sources",
            "blackbook://corpus",
            "blackbook://vocabulary",
            "blackbook://cases",
        }
        assert {t.uriTemplate for t in templates} == {"blackbook://case/{name}"}
        assert {p.name for p in prompt_list} == {
            "triage_observation",
            "explain_technique",
            "review_finding",
            "draft_report",
        }
    finally:
        db.close()


def test_resources_do_not_disturb_the_tool_registration():
    """Resources and prompts must not cost a tool, or change one's schema."""
    db = Database(":memory:")
    try:
        server = _server(db)

        async def list_tools():
            return await server.list_tools()

        tools = asyncio.run(list_tools())
        assert len(tools) == 12
        assert all(t.outputSchema for t in tools)
    finally:
        db.close()


def test_every_prompt_declares_the_arguments_its_function_takes():
    """A prompt that lists an argument it does not accept fails at call time."""
    db = Database(":memory:")
    try:
        server = _server(db)

        async def survey():
            return await server.list_prompts()

        for listed in asyncio.run(survey()):
            kwargs = {a.name: f"value-for-{a.name}" for a in (listed.arguments or [])}
            # Raises if the declared name and the function's parameter disagree.
            asyncio.run(server.get_prompt(listed.name, kwargs))
    finally:
        db.close()


# -- resource bodies ------------------------------------------------------


def test_corpus_resource_reports_the_counts_and_the_query_log():
    db = Database(":memory:")
    try:
        db.upsert_source(Source(source_id="s", name="S"))
        payload = json.loads(resources.corpus_json(db))
        assert payload["sources"] == 1
        assert "chunks" in payload
        # The log is empty here, which is a real reading, not a failure.
        assert payload["query_log"]["total"] == 0
    finally:
        db.close()


def test_sources_resource_matches_the_tool_payload():
    db = Database(":memory:")
    try:
        db.upsert_source(Source(source_id="hacktricks", name="HackTricks"))
        settings = Settings(
            sources=[SourceConfig(id="hacktricks", name="HackTricks", type="git")]
        )
        payload = json.loads(resources.sources_json(db, settings))
        assert payload["sources"][0]["id"] == "hacktricks"
    finally:
        db.close()


def test_vocabulary_resource_lists_the_terms_and_the_attack_ids():
    payload = json.loads(resources.vocabulary_json())
    assert {t["term"] for t in payload["techniques"]}
    assert payload["services"] and payload["tools"]
    # At least one technique maps to an ATT&CK id, and the ones that do carry it.
    assert any(t["attack_id"] for t in payload["techniques"])
    # The aliases are the spellings the filters resolve, so they belong here too.
    assert payload["aliases"]


def test_cases_resource_is_empty_before_any_case_exists():
    db = Database(":memory:")
    try:
        assert json.loads(resources.cases_json(db)) == {"count": 0, "cases": []}
    finally:
        db.close()


def test_case_resource_renders_markdown_for_a_real_case():
    db = Database(":memory:")
    try:
        from blackbook.mcp.schemas import ContextInput
        from blackbook.mcp.tools import KnowledgeTools

        KnowledgeTools(db, Settings()).knowledge_context(
            ContextInput(action="create", case="acme", target="10.0.0.5")
        )
        text = resources.case_markdown(db, "acme")
        assert text.startswith("# Case: acme")
        assert "10.0.0.5" in text
    finally:
        db.close()


def test_case_resource_explains_a_missing_case_instead_of_raising():
    db = Database(":memory:")
    try:
        text = resources.case_markdown(db, "no-such-case")
        assert "not found" in text.lower()
        assert "blackbook://cases" in text
    finally:
        db.close()


# -- through the protocol -------------------------------------------------


def test_case_template_reads_end_to_end_over_the_protocol():
    """The URI parameter name must match the function's, or the read 404s."""
    db = Database(":memory:")
    try:
        from blackbook.mcp.schemas import ContextInput
        from blackbook.mcp.tools import KnowledgeTools

        KnowledgeTools(db, Settings()).knowledge_context(
            ContextInput(action="create", case="acme")
        )
        contents = asyncio.run(_reads(_server(db), "blackbook://case/acme"))
        assert "# Case: acme" in _text(contents)
    finally:
        db.close()


def test_prompt_returns_one_message_that_names_the_tools():
    db = Database(":memory:")
    try:
        result = asyncio.run(
            _server(db).get_prompt("triage_observation", {"observation": "445 open"})
        )
        text = result.messages[0].content.text
        assert "knowledge_research" in text
        assert "445 open" in text
    finally:
        db.close()


def test_prompts_stay_short_enough_to_be_framings_not_instructions():
    """A prompt longer than the tool contract is one an agent may follow instead."""
    assert len(prompts.triage_observation("obs", "10.0.0.5")) < 2000
    assert len(prompts.explain_technique("kerberoasting", "windows")) < 2000
    assert len(prompts.review_finding("finding")) < 2000
    assert len(prompts.draft_report("acme")) < 2000
