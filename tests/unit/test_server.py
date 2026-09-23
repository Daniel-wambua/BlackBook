"""Regression tests for the FastMCP protocol wiring."""

from __future__ import annotations

import asyncio

import httpx

from blackbook.config import Settings
from blackbook.server import build_server
from blackbook.storage import Source
from blackbook.storage.database import Database


def test_server_registers_all_tools_with_structured_schemas():
    db = Database(":memory:")
    try:
        server = build_server(Settings(), db)

        async def list_registered_tools():
            return await server.list_tools()

        tools = asyncio.run(list_registered_tools())
        by_name = {tool.name: tool for tool in tools}
        expected = {
            "knowledge_search",
            "knowledge_source",
            "knowledge_technique",
            "knowledge_graph",
            "knowledge_case_search",
            "knowledge_research",
            "knowledge_context",
            "knowledge_hunt_plan",
            "knowledge_finding_review",
            "knowledge_report_draft",
            "knowledge_sources",
            "knowledge_compare",
        }
        assert set(by_name) == expected
        assert all(tool.inputSchema.get("type") == "object" for tool in by_name.values())
        assert all(tool.outputSchema for tool in by_name.values())
    finally:
        db.close()


def _get(app, path: str) -> httpx.Response:
    async def go() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://blackbook.test"
        ) as client:
            return await client.get(path)

    return asyncio.run(go())


def test_health_and_landing_page_report_the_corpus_over_http():
    """The two browser-facing readouts, against a real ASGI app.

    /health and / are the endpoints a monitor or a browser hits on a timer, and
    they are the two that read the counts through Database.counts_cached(). So
    this covers the caching path as well as the payloads: a write committed in
    this process has to show up on the very next request, not after the TTL.
    """
    db = Database(":memory:")
    try:
        app = build_server(Settings(), db).streamable_http_app()

        health = _get(app, "/health")
        assert health.status_code == 200
        payload = health.json()
        assert payload["status"] == "ok"
        assert payload["service"] == "blackbook"
        assert payload["corpus"]["documents"] == 0
        # The whole surface, so an operator can see it without reading source.
        assert "knowledge_search" in payload["tools"]
        assert "blackbook://corpus" in payload["resources"]
        assert "blackbook://case/{name}" in payload["resource_templates"]
        assert "triage_observation" in payload["prompts"]

        # A committed write is visible immediately, despite the cache.
        with db.session():
            db.upsert_source(Source(source_id="s1", name="S1"))
        assert _get(app, "/health").json()["corpus"]["sources"] == 1

        page = _get(app, "/")
        assert page.status_code == 200
        assert "BlackBook" in page.text
        assert "blackbook://sources" in page.text
        assert "draft_report" in page.text
    finally:
        db.close()
