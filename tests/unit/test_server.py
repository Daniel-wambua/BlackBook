"""Regression tests for the FastMCP protocol wiring."""

from __future__ import annotations

import asyncio

from blackbook.config import Settings
from blackbook.server import build_server
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
            "knowledge_case_search",
            "knowledge_research",
            "knowledge_context",
        }
        assert set(by_name) == expected
        assert all(tool.inputSchema.get("type") == "object" for tool in by_name.values())
        assert all(tool.outputSchema for tool in by_name.values())
    finally:
        db.close()
