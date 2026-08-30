"""BlackBook MCP server entrypoint.

Wires the FastMCP server over stdio and registers the knowledge tools. The
server is read-only with respect to external systems: it only reads from the
local SQLite knowledge base and the configured knowledge directories.

Run directly (``python -m blackbook.server``) or via the CLI
(``blackbook serve``).
"""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from blackbook.config import ensure_dirs, load_config
from blackbook.mcp.schemas import (
    CaseSearchInput,
    CaseSearchOutput,
    ContextInput,
    ContextOutput,
    GetSourceInput,
    ResearchInput,
    ResearchOutput,
    SearchInput,
    SearchOutput,
    SourceOutput,
    TechniqueInput,
    TechniqueOutput,
)
from blackbook.mcp.tools import KnowledgeTools
from blackbook.storage.database import Database

log = logging.getLogger(__name__)


def build_server(settings=None, db: Database | None = None) -> FastMCP:
    """Construct the FastMCP server with all tools registered.

    Accepts optional pre-built ``settings``/``db`` so tests can inject an
    isolated database instead of touching the real one.
    """
    settings = settings or load_config()
    ensure_dirs(settings)
    db = db or Database(settings.db_path)
    tools = KnowledgeTools(db, settings)

    mcp = FastMCP(
        name="blackbook",
        instructions=(
            "BlackBook is a source-grounded cybersecurity knowledge server. "
            "Use knowledge_search to find documented techniques and similar cases, "
            "knowledge_technique for a structured, graph-enriched dossier on a "
            "technique, knowledge_case_search to find similar hands-on writeups, "
            "knowledge_research to turn a free-text observation into a "
            "source-grounded research packet (detected signals, technique briefs, "
            "cited references and related cases), and knowledge_source to retrieve "
            "exact supporting excerpts. Use knowledge_context to keep local "
            "investigation state — create a case and record observations, "
            "findings, and hypotheses as you work. Every knowledge result carries "
            "verifiable provenance. BlackBook never executes commands or touches "
            "remote systems."
        ),
        host=settings.server.host,
        port=settings.server.port,
        streamable_http_path=settings.server.path,
    )

    # HTTP convenience routes. These are only reachable when the server runs
    # under an HTTP transport (``--http``); they are inert under stdio. The MCP
    # endpoint itself (``settings.server.path``, default ``/mcp``) is owned by
    # the streamable-http transport and is a machine interface: a bare browser
    # ``GET /mcp`` (no ``Accept: text/event-stream``) correctly gets 406, which
    # looks alarming in the logs. So we give humans real places to land — a root
    # page and ``/health`` — and silence the browser's favicon probe.
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, JSONResponse, Response

    def _health_payload() -> dict:
        from blackbook import __version__

        payload: dict = {
            "status": "ok",
            "service": "blackbook",
            "version": __version__,
            "transport": "streamable-http",
            "mcp_endpoint": settings.server.path,
        }
        try:
            payload["corpus"] = db.counts()
        except Exception:  # pragma: no cover - health must never crash
            payload["corpus"] = None
        return payload

    @mcp.custom_route("/health", methods=["GET"])
    async def health(_request: Request) -> JSONResponse:
        payload = _health_payload()
        payload["tools"] = [t.name for t in await mcp.list_tools()]
        return JSONResponse(payload)

    @mcp.custom_route("/", methods=["GET"])
    async def index(_request: Request) -> HTMLResponse:
        # Friendly landing page so opening the base URL in a browser shows what
        # this is and where to go — rather than a blank 404. Never raises.
        from blackbook import __version__

        mcp_path = settings.server.path
        try:
            tools = [t.name for t in await mcp.list_tools()]
        except Exception:  # pragma: no cover - landing page must never crash
            tools = []
        try:
            c = db.counts()
            corpus = (
                f"{c['sources']} sources · {c['documents']} docs · "
                f"{c['chunks']} chunks · {c['entities']} entities · "
                f"{c['relationships']} relationships"
            )
        except Exception:  # pragma: no cover
            corpus = "unavailable"
        tool_items = "".join(f"<li><code>{t}</code></li>" for t in tools)
        html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BlackBook MCP</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.6 system-ui, sans-serif; max-width: 46rem; margin: 3rem auto;
         padding: 0 1.25rem; }}
  h1 {{ margin: 0 0 .25rem; letter-spacing: .02em; }}
  .sub {{ opacity: .7; margin: 0 0 1.5rem; }}
  code {{ background: color-mix(in srgb, currentColor 12%, transparent);
          padding: .1em .4em; border-radius: .3em; }}
  .grid {{ display: grid; grid-template-columns: 8rem 1fr; gap: .4rem 1rem; margin: 1rem 0; }}
  .k {{ opacity: .6; }}
  .note {{ opacity: .75; border-left: 3px solid color-mix(in srgb, currentColor 30%, transparent);
           padding: .5rem .9rem; margin: 1.5rem 0; }}
  ul {{ columns: 2; margin: .5rem 0; }}
  a {{ color: inherit; }}
</style></head><body>
<h1>BlackBook MCP</h1>
<p class="sub">Source-grounded cybersecurity knowledge &amp; research · v{__version__}</p>
<div class="grid">
  <span class="k">MCP endpoint</span><span><code>{mcp_path}</code> · streamable-http (for MCP clients)</span>
  <span class="k">Health</span><span><a href="/health"><code>/health</code></a> · JSON status</span>
  <span class="k">Corpus</span><span>{corpus}</span>
</div>
<p><strong>Tools</strong></p>
<ul>{tool_items}</ul>
<div class="note">
  <code>{mcp_path}</code> is a machine endpoint — connect an MCP client to it, don't open it
  in a browser (a plain browser request returns <code>406 Not Acceptable</code> by design).
  To check the server from a browser, use <a href="/health">/health</a>.
</div>
</body></html>"""
        return HTMLResponse(html)

    @mcp.custom_route("/favicon.ico", methods=["GET"])
    async def favicon(_request: Request) -> Response:
        # Browsers auto-request this; answer 204 so it does not show as a 404.
        return Response(status_code=204)


    @mcp.tool(
        name="knowledge_search",
        description=(
            "Search the indexed cybersecurity knowledge corpus (HackTricks, 0xdf "
            "writeups, local PDFs). Returns concise, ranked results with exact "
            "source references (chunk_id/doc_id/url/page/section) that can be "
            "resolved with knowledge_source. Filter by sources, platform, "
            "categories, or techniques. Read-only."
        ),
    )
    def knowledge_search(
        query: str,
        sources: list[str] | None = None,
        categories: list[str] | None = None,
        platform: str | None = None,
        techniques: list[str] | None = None,
        mode: str = "hybrid",
        limit: int = 8,
        detail: str = "standard",
    ) -> SearchOutput:
        inp = SearchInput(
            query=query,
            sources=sources,
            categories=categories,
            platform=platform,
            techniques=techniques,
            mode=mode,  # type: ignore[arg-type]
            limit=limit,
            detail=detail,  # type: ignore[arg-type]
        )
        return tools.knowledge_search(inp)

    @mcp.tool(
        name="knowledge_source",
        description=(
            "Retrieve the exact source excerpt for a reference returned by "
            "knowledge_search. Provide chunk_id for a precise chunk, or "
            "source+document (or title_contains) to read a document's sections. "
            "Returns the smallest useful excerpt with full provenance. Read-only."
        ),
    )
    def knowledge_source(
        chunk_id: int | None = None,
        doc_id: int | None = None,
        source: str | None = None,
        document: str | None = None,
        title_contains: str | None = None,
        section: str | None = None,
        max_excerpts: int = 5,
    ) -> SourceOutput:
        inp = GetSourceInput(
            chunk_id=chunk_id,
            doc_id=doc_id,
            source=source,
            document=document,
            title_contains=title_contains,
            section=section,
            max_excerpts=max_excerpts,
        )
        return tools.knowledge_source(inp)

    @mcp.tool(
        name="knowledge_technique",
        description=(
            "Assemble a source-grounded dossier for a technique. Returns which "
            "sources document it, which tools/services/writeups the knowledge "
            "graph associates with it (each edge carrying confidence and the "
            "document it was derived from), plus real, cited excerpts from the "
            "index. Works even before the graph is built — it always returns "
            "indexed references. Read-only; never fabricates a citation."
        ),
    )
    def knowledge_technique(
        technique: str,
        sources: list[str] | None = None,
        limit: int = 6,
    ) -> TechniqueOutput:
        inp = TechniqueInput(technique=technique, sources=sources, limit=limit)
        return tools.knowledge_technique(inp)

    @mcp.tool(
        name="knowledge_case_search",
        description=(
            "Find hands-on writeups and case studies similar to a situation "
            "(favouring practical, walkthrough-style material). Each hit carries "
            "full provenance and, when the graph is built, the techniques it "
            "demonstrates. Filter by sources or platform, and add known "
            "techniques to sharpen the query. Read-only."
        ),
    )
    def knowledge_case_search(
        query: str,
        sources: list[str] | None = None,
        platform: str | None = None,
        techniques: list[str] | None = None,
        limit: int = 6,
    ) -> CaseSearchOutput:
        inp = CaseSearchInput(
            query=query,
            sources=sources,
            platform=platform,
            techniques=techniques,
            limit=limit,
        )
        return tools.knowledge_case_search(inp)

    @mcp.tool(
        name="knowledge_research",
        description=(
            "Turn a free-text observation (e.g. a service banner, a foothold, a "
            "suspicious finding) into a source-grounded research packet: the "
            "controlled-vocabulary services/techniques/tools detected in the text, "
            "a short graph-backed brief per technique (which sources document it), "
            "real cited references from the index, and related hands-on cases. "
            "Every item is either vocabulary-derived or a resolvable citation — "
            "nothing is synthesised or fabricated. Read-only."
        ),
    )
    def knowledge_research(
        observation: str,
        sources: list[str] | None = None,
        platform: str | None = None,
        techniques: list[str] | None = None,
        limit: int = 6,
        include_cases: bool = True,
    ) -> ResearchOutput:
        inp = ResearchInput(
            observation=observation,
            sources=sources,
            platform=platform,
            techniques=techniques,
            limit=limit,
            include_cases=include_cases,
        )
        return tools.knowledge_research(inp)

    @mcp.tool(
        name="knowledge_context",
        description=(
            "Manage local investigation state. Actions: 'create' a case by name "
            "(optional target/platform/meta); 'add' an observation/finding/"
            "hypothesis/technique/note to a case; 'update_observation' to set an "
            "observation's status (open/tested/confirmed/refuted/resolved); 'get' a "
            "case's full state; 'list' all cases. Reads and writes only the local, "
            "user-authored case layer in the knowledge base — it never executes "
            "anything or touches remote systems. There is no delete action."
        ),
    )
    def knowledge_context(
        action: str,
        case: str | None = None,
        target: str = "",
        platform: str = "",
        kind: str = "observation",
        text: str | None = None,
        obs_id: int | None = None,
        status: str | None = None,
        meta: dict | None = None,
    ) -> ContextOutput:
        inp = ContextInput(
            action=action,  # type: ignore[arg-type]
            case=case,
            target=target,
            platform=platform,
            kind=kind,  # type: ignore[arg-type]
            text=text,
            obs_id=obs_id,
            status=status,  # type: ignore[arg-type]
            meta=meta,
        )
        return tools.knowledge_context(inp)

    return mcp


def main() -> None:
    # Build settings + db first so the banner can report live corpus/graph
    # stats, then hand them to the server so nothing is opened twice. All
    # chrome (banner + logs) goes to stderr — stdout is the JSON-RPC stream.
    #
    # Transport defaults to stdio (how Claude Code / Cursor spawn the server).
    # Set BLACKBOOK_TRANSPORT=streamable-http (or sse) to run as a network
    # server reachable over a URL + port; the CLI ``serve --http`` does this.
    import os

    from blackbook import ui

    settings = load_config()
    ensure_dirs(settings)
    ui.configure_logging()

    transport = os.environ.get("BLACKBOOK_TRANSPORT", "stdio").strip() or "stdio"
    db = Database(settings.db_path)
    server = build_server(settings, db)

    if transport in ("streamable-http", "sse"):
        endpoint = f"http://{settings.server.host}:{settings.server.port}"
        ui.print_banner(db=db, transport=f"{transport} · {endpoint}{settings.server.path}")
        server.run(transport=transport)
    else:
        ui.print_banner(db=db, transport="stdio")
        server.run(transport="stdio")


if __name__ == "__main__":
    main()
