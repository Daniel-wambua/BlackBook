"""BlackBook command-line interface.

Administration and diagnostics for the knowledge base:

    blackbook ingest [--source ID] [--force]
    blackbook search "kerberoasting"
    blackbook stats
    blackbook sources
    blackbook queries        # what was asked, and what came back empty
    blackbook doctor
    blackbook eval           # run the offline retrieval benchmark
    blackbook rebuild-index
    blackbook serve          # run the MCP server over stdio
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table

from blackbook import ui
from blackbook.config import ensure_dirs, load_config
from blackbook.ingestion import adapter_for
from blackbook.ingestion.pipeline import IngestionPipeline
from blackbook.knowledge import query_log
from blackbook.retrieval import HybridRetriever
from blackbook.storage.database import Database

app = typer.Typer(add_completion=False, help="BlackBook MCP — cybersecurity knowledge server")
console = Console()
err_console = Console(stderr=True)


def _setup(verbose: bool):
    ui.configure_logging(verbose)
    settings = load_config()
    ensure_dirs(settings)
    return settings


# Doctor check severities
_OK = "ok"
_WARN = "warn"
_FAIL = "fail"


def _db(settings) -> Database:
    return Database(settings.db_path)


def _ingest_targets(settings, source, force, verbose, pdf_dir=None, rebuild_graph=True) -> None:
    db = _db(settings)
    # Build the embedder once (None when embeddings are disabled or the semantic
    # extra is missing); the pipeline embeds new chunks as it ingests each source.
    from blackbook.embeddings import try_build_embedder

    embedder = try_build_embedder(settings)
    if settings.embeddings.enabled and embedder is None:
        err_console.print(
            "[yellow]Embeddings enabled but the semantic extra is unavailable; "
            "ingesting lexical-only. Install with: pip install "
            '"blackbook-mcp[semantic]"[/yellow]'
        )
    pipeline = IngestionPipeline(db, embedder=embedder)

    targets = settings.enabled_sources()
    if source:
        targets = [s for s in targets if s.id == source]
        if not targets:
            err_console.print(f"[red]Unknown or disabled source:[/red] {source}")
            db.close()
            raise typer.Exit(code=2)

    # --pdf-dir overrides the local_pdfs directory and forces it on/targeted.
    if pdf_dir:
        from pathlib import Path as _P

        pd = _P(pdf_dir).expanduser()
        if not pd.is_dir():
            err_console.print(f"[red]PDF directory not found:[/red] {pd}")
            db.close()
            raise typer.Exit(code=2)
        for t in targets:
            if t.type == "filesystem":
                t.directory = str(pd)
                t.enabled = True
        if not any(t.type == "filesystem" for t in targets):
            from blackbook.config import SourceConfig

            targets.append(
                SourceConfig(
                    id="local_pdfs", name="Local PDFs", type="filesystem",
                    authority="user", directory=str(pd),
                )
            )

    total_chunks_written = 0
    for src_cfg in targets:
        # Register/refresh the source row.
        from blackbook.storage.models import Source

        with db.session():
            db.upsert_source(
                Source(
                    source_id=src_cfg.id,
                    name=src_cfg.name,
                    authority=src_cfg.authority,
                    enabled=src_cfg.enabled,
                    source_type=src_cfg.type,
                    url=src_cfg.url,
                )
            )
        console.print(f"\n[bold cyan]Ingesting[/bold cyan] {src_cfg.name} ({src_cfg.id})")
        try:
            adapter = adapter_for(src_cfg, raw_dir=str(settings.raw_dir))
            progress = None
            progress_task = None
            if src_cfg.type == "filesystem" and hasattr(adapter, "candidate_files"):
                total_files = len(adapter.candidate_files(warn_oversized=False))
                if total_files:
                    progress = Progress(
                        TextColumn("  {task.description}"),
                        BarColumn(),
                        TextColumn("{task.completed}/{task.total}"),
                        TextColumn("{task.percentage:>5.1f}%"),
                        TimeRemainingColumn(),
                        console=console,
                    )
                    progress.start()
                    progress_task = progress.add_task("", total=total_files)

                    def on_pdf_progress(done, total, current):
                        progress.update(
                            progress_task,
                            completed=done,
                            description=current,
                        )

                    adapter.progress_callback = on_pdf_progress
            try:
                result = pipeline.run(adapter, force=force)
            finally:
                if progress is not None:
                    progress.stop()
            st = result.stats
            total_chunks_written += st.chunks_written
            embed_note = (
                f" embedded={result.embedded}" if embedder is not None else ""
            )
            console.print(
                f"  discovered={st.discovered} parsed={st.parsed} "
                f"unchanged={st.skipped_unchanged} chunks={st.chunks_written}"
                f"{embed_note} errors={st.errors}"
            )
            if st.errors == 0 and st.discovered > 0 and st.parsed == 0:
                console.print("  status=up-to-date (no source documents changed)")
            elif st.errors == 0 and st.parsed > 0:
                console.print(f"  status=updated ({st.parsed} document(s) changed)")
            if st.error_messages and verbose:
                for m in st.error_messages[:10]:
                    err_console.print(f"    [yellow]{m}[/yellow]")
        except Exception as e:
            ui.fail(f"{src_cfg.id}: {e}", err_console)
            if verbose:
                raise

    # Compact the FTS index once, after all sources are in. FTS5's 'optimize'
    # merges the b-tree segments that incremental inserts leave behind, which
    # keeps BM25 query latency flat as the corpus grows. Runs only when new
    # chunks actually landed (nothing to merge otherwise) and is independent of
    # the graph rebuild below, so it happens regardless of graph outcome.
    if total_chunks_written > 0:
        with db.session():
            db.optimize_fts()

    # Keep the knowledge graph in step with the index. The rebuild runs only
    # when new/changed chunks landed, and never aborts the ingest: the graph
    # merely enhances retrieval, so a graph failure is a warning, not a failed
    # ingest. It reuses the cached term extraction for the documents this ingest
    # did not touch, which on an incremental ingest is nearly all of them.
    if rebuild_graph and total_chunks_written > 0:
        from blackbook.knowledge.graph import GraphBuilder

        console.print("\n[bold cyan]Rebuilding knowledge graph[/bold cyan]")
        try:
            stats = GraphBuilder(db).rebuild()
            console.print(
                f"  entities={stats.entities} relationships={stats.relationships} "
                f"writeups={stats.writeups} "
                f"reused={stats.reused}/{stats.documents}"
            )
        except Exception as e:  # pragma: no cover - defensive
            ui.warn(f"Graph rebuild skipped: {e}", err_console)
            if verbose:
                raise
    if total_chunks_written > 0:
        ui.success(f"Ingest complete — {total_chunks_written} chunks indexed.", console)
    db.close()


@app.command()
def ingest(
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Source ID to ingest (default: all enabled)"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-fetch and re-ingest even if unchanged"),
    pdf_dir: Optional[str] = typer.Option(None, "--pdf-dir", help="Ingest PDFs from this directory (overrides local_pdfs.directory)"),
    no_graph: bool = typer.Option(False, "--no-graph", help="Skip the knowledge-graph rebuild after ingest"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Incrementally ingest one or all enabled sources; use --force for a full refresh."""
    settings = _setup(verbose)
    _ingest_targets(settings, source, force, verbose, pdf_dir=pdf_dir, rebuild_graph=not no_graph)


@app.command()
def update(
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Source ID to update (default: all enabled)"),
    no_graph: bool = typer.Option(False, "--no-graph", help="Skip the knowledge-graph rebuild after update"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Check all enabled sources and ingest only new or changed material."""
    settings = _setup(verbose)
    # update == a non-forced ingest: fetch() refreshes, pipeline skips unchanged.
    _ingest_targets(settings, source, force=False, verbose=verbose, rebuild_graph=not no_graph)


@app.command(name="source-show")
def source_show(
    source: str = typer.Argument(..., help="Source ID to inspect"),
    limit: int = typer.Option(20, "--limit", "-n"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Show details and indexed documents for one source."""
    settings = _setup(verbose)
    db = _db(settings)
    src = db.get_source(source)
    if not src:
        err_console.print(f"[red]Source not indexed:[/red] {source}")
        db.close()
        raise typer.Exit(code=2)
    console.print(f"[bold cyan]{src['name']}[/bold cyan] ({src['source_id']})")
    console.print(f"  authority={src['authority']} type={src['source_type']} enabled={bool(src['enabled'])}")
    if src.get("url"):
        console.print(f"  url={src['url']}")
    docs = db.conn.execute(
        "SELECT doc_id, title, url, path FROM documents WHERE source_id = ? ORDER BY title LIMIT ?",
        (source, limit),
    ).fetchall()
    if docs:
        table = Table(title=f"Documents ({len(docs)} shown)")
        table.add_column("doc_id", style="dim", width=7)
        table.add_column("Title", style="bold")
        table.add_column("Location", overflow="fold")
        for d in docs:
            table.add_row(str(d["doc_id"]), d["title"], d["url"] or d["path"] or "")
        console.print(table)
    else:
        console.print("  [dim]no documents indexed[/dim]")
    db.close()


@app.command()
def search(
    query: str,
    source: Optional[str] = typer.Option(None, "--source", "-s"),
    platform: Optional[str] = typer.Option(None, "--platform", "-p"),
    mode: str = typer.Option("hybrid", "--mode", "-m"),
    limit: int = typer.Option(8, "--limit", "-n"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Search the knowledge base."""
    settings = _setup(verbose)
    db = _db(settings)
    retriever = HybridRetriever(db, settings)
    source_ids = settings.source_ids([source] if source else None)
    if source_ids == []:
        err_console.print(
            f"[red]No enabled source matches:[/red] {source} — run `blackbook sources` "
            "to list valid IDs. Refusing to search everything by accident."
        )
        db.close()
        raise typer.Exit(code=2)
    with query_log.timed() as timing:
        results = retriever.search(
            query, mode=mode, source_ids=source_ids, platform=platform, limit=limit
        )
    # Logged before the early return below, so a CLI search that finds nothing
    # is recorded too: a miss is the entry worth having.
    query_log.record(
        db,
        settings,
        tool="cli:search",
        query=query,
        mode=mode,
        sources=source_ids if source_ids is not None else ["all"],
        result_count=len(results),
        top_score=results[0].score if results else None,
        latency_ms=timing["latency_ms"],
    )

    if not results:
        console.print("[yellow]No results.[/yellow]")
        db.close()
        return
    table = Table(title=f"Results for: {query}", show_lines=False)
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", width=7)
    table.add_column("Source", style="cyan")
    table.add_column("Title", style="bold")
    table.add_column("Section", overflow="fold")
    for i, r in enumerate(results, 1):
        table.add_row(
            str(i),
            f"{r.score:.3f}",
            r.source_id,
            r.title,
            " > ".join(r.section_path[-3:]),
        )
    console.print(table)
    console.print("\n[dim]Top snippet:[/dim]", results[0].snippet[:300])
    db.close()


@app.command()
def stats(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
):
    """Show knowledge-base statistics."""
    settings = _setup(verbose)
    db = _db(settings)
    counts = db.counts()
    rows = db.conn.execute(
        """
        SELECT d.source_id, COUNT(DISTINCT d.doc_id) AS docs, COUNT(c.chunk_id) AS chunks
        FROM documents d LEFT JOIN chunks c ON c.doc_id = d.doc_id
        GROUP BY d.source_id ORDER BY d.source_id
        """
    ).fetchall()
    per_source = {r["source_id"]: {"docs": r["docs"], "chunks": r["chunks"]} for r in rows}
    if as_json:
        console.print_json(json.dumps({"counts": counts, "per_source": per_source}))
        db.close()
        return

    table = Table(title="BlackBook Knowledge Base")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right")
    for k, v in counts.items():
        table.add_row(k.capitalize(), str(v))
    console.print(table)

    # Per-source document counts.
    if rows:
        st = Table(title="Per-source")
        st.add_column("Source", style="cyan")
        st.add_column("Docs", justify="right")
        st.add_column("Chunks", justify="right")
        for r in rows:
            st.add_row(r["source_id"], str(r["docs"]), str(r["chunks"]))
        console.print(st)
    db.close()


@app.command()
def sources(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
):
    """List configured and indexed sources."""
    settings = _setup(verbose)
    db = _db(settings)
    configured = {s.id: s for s in settings.sources}
    indexed = {s["source_id"]: s for s in db.list_sources()}

    if as_json:
        payload = [
            {
                "id": sid,
                "name": cfg.name,
                "type": cfg.type,
                "authority": cfg.authority,
                "enabled": cfg.enabled,
                "indexed": sid in indexed,
                "last_fetched": (indexed.get(sid) or {}).get("last_fetched"),
                "version": (indexed.get(sid) or {}).get("version"),
            }
            for sid, cfg in configured.items()
        ]
        console.print_json(json.dumps(payload))
        db.close()
        return

    table = Table(title="Sources")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Type")
    table.add_column("Authority")
    table.add_column("Enabled")
    table.add_column("Indexed")
    table.add_column("Fetched")
    table.add_column("Rev")
    for sid, cfg in configured.items():
        row = indexed.get(sid) or {}
        version = row.get("version") or ""
        table.add_row(
            sid,
            cfg.name,
            cfg.type,
            cfg.authority,
            "yes" if cfg.enabled else "no",
            "yes" if sid in indexed else "no",
            # A source that has never been fetched shows a dash rather than a
            # blank, so "no record" is visibly distinct from an empty value.
            row.get("last_fetched") or "-",
            version[:8] if version else "-",
        )
    console.print(table)
    console.print(
        "[dim]Fetched is when the source was last pulled successfully (UTC); "
        "Rev is the commit it was pulled at, for sources that have one.[/dim]"
    )
    db.close()


@app.command()
def queries(
    limit: int = typer.Option(25, "--limit", "-n", help="How many recent entries to show"),
    empty: bool = typer.Option(
        False, "--empty", "-e", help="Only show queries that returned nothing"
    ),
    clear: bool = typer.Option(False, "--clear", help="Delete the whole query log"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
):
    """Show the local query log: what was asked, and what came back.

    The useful signal is the empty rows. A query that returns nothing is the
    only evidence the index gives about what it cannot answer, and it is not
    visible from the corpus side. Logging is on by default and can be turned
    off with ``query_log.enabled: false`` in the config.
    """
    settings = _setup(verbose)
    db = _db(settings)
    if clear:
        removed = db.clear_query_log()
        db.close()
        console.print(f"[green]Cleared {removed} query log entries.[/green]")
        return

    stats = db.query_log_stats()
    entries = db.list_queries(limit=limit, empty_only=empty)
    if as_json:
        console.print_json(json.dumps({"stats": stats, "entries": entries}))
        db.close()
        return

    if stats["total"] == 0:
        console.print(
            "[yellow]Query log is empty.[/yellow] Searches are recorded from "
            "the MCP tools and `blackbook search`."
        )
        if not settings.query_log.enabled:
            console.print(
                "[dim]Logging is currently disabled "
                "(query_log.enabled is false).[/dim]"
            )
        db.close()
        return

    console.print(
        f"[bold]{stats['total']}[/bold] logged, "
        f"[bold]{stats['empty']}[/bold] returned nothing "
        f"({stats['empty_rate'] * 100:.1f}%), "
        f"avg [bold]{stats['avg_latency_ms']:.1f} ms[/bold]; "
        f"{stats['first_at']} to {stats['last_at']} UTC"
    )
    if not entries:
        console.print("[dim]No entries to show.[/dim]")
        db.close()
        return

    table = Table(title=f"Recent queries{' (empty only)' if empty else ''}")
    table.add_column("When (UTC)", style="dim")
    table.add_column("Query", overflow="fold")
    table.add_column("Mode")
    table.add_column("Hits", justify="right")
    table.add_column("Top", justify="right")
    table.add_column("ms", justify="right")
    for row in entries:
        hits = row["result_count"]
        table.add_row(
            (row["created_at"] or "")[:19],
            row["query"],
            row["mode"],
            f"[red]{hits}[/red]" if hits == 0 else str(hits),
            "-" if row["top_score"] is None else f"{row['top_score']:.3f}",
            f"{row['latency_ms']:.1f}",
        )
    console.print(table)
    db.close()


@app.command(name="rebuild-index")
def rebuild_index(verbose: bool = typer.Option(False, "--verbose", "-v")):
    """Rebuild the full-text search index."""
    settings = _setup(verbose)
    db = _db(settings)
    console.print("Rebuilding FTS index…")
    with db.session():
        db.rebuild_fts()
        db.optimize_fts()
    console.print("[green]Done.[/green]")
    db.close()


graph_app = typer.Typer(
    add_completion=False,
    help="Knowledge-graph maintenance (entities/relationships derived from the index).",
)
app.add_typer(graph_app, name="graph")


def _print_graph_stats(stats) -> None:
    table = Table(title="Knowledge graph")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right")
    table.add_row("Documents scanned", str(stats.documents))
    table.add_row("Writeups", str(stats.writeups))
    table.add_row("Entities", str(stats.entities))
    table.add_row("Relationships", str(stats.relationships))
    # Term extraction is the expensive half of a build and is cached per
    # document, so this row is what tells a reader why a rebuild of a
    # mostly-unchanged corpus finished in a second. A skipped build reports it
    # equal to the document count, which is the other thing worth seeing.
    table.add_row("Reused cached extraction", str(stats.reused))
    console.print(table)
    if stats.skipped:
        console.print(
            "[green]Graph already current:[/green] every document matched the "
            "cached extraction, nothing was rebuilt."
        )
    if stats.by_entity_type:
        et = Table(title="Entities by type")
        et.add_column("Type", style="cyan")
        et.add_column("Count", justify="right")
        for k in sorted(stats.by_entity_type):
            et.add_row(k, str(stats.by_entity_type[k]))
        console.print(et)
    if stats.by_predicate:
        pt = Table(title="Relationships by predicate")
        pt.add_column("Predicate", style="cyan")
        pt.add_column("Count", justify="right")
        for k in sorted(stats.by_predicate):
            pt.add_row(k, str(stats.by_predicate[k]))
        console.print(pt)
        # ``uses`` and ``targets`` count distinct pairs, not mentions: those two
        # collapse every witnessing document into one row carrying a support
        # count, so they read far smaller than the other predicates over the same
        # corpus. Without saying so, the drop looks like lost data.
        console.print(
            "[dim]uses/targets are stored once per pair with a support count, "
            "not once per witnessing document.[/dim]"
        )


def _print_writeup_coverage(coverage) -> None:
    """Render writeup coverage, contributing sources first.

    The ordering is the point. A table sorted by name buries the finding: the
    zero rows are what the reader needs to see, and they sort next to their
    document counts so the size of each gap is legible.
    """
    table = Table(title="Writeup coverage")
    table.add_column("Source", style="cyan")
    table.add_column("Documents", justify="right")
    table.add_column("Writeups", justify="right")
    for source_id in sorted(
        coverage.docs_by_source,
        key=lambda s: (-coverage.by_source.get(s, 0), -coverage.docs_by_source[s], s),
    ):
        n = coverage.by_source.get(source_id, 0)
        table.add_row(
            source_id,
            str(coverage.docs_by_source[source_id]),
            str(n) if n else "[yellow]0[/yellow]",
        )
    console.print(table)
    console.print(
        f"{coverage.eligible}/{coverage.documents} documents count as writeups, "
        f"from {coverage.sources_with_writeups} of "
        f"{len(coverage.docs_by_source)} sources."
    )
    if coverage.graph_built:
        console.print(f"Graph holds {coverage.graph_writeups} writeup entities.")


@graph_app.command("build")
def graph_build(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    full: bool = typer.Option(
        False, "--full",
        help="Re-extract every document, ignoring the per-document term cache",
    ),
):
    """Rebuild the knowledge graph from the indexed corpus.

    An idempotent rebuild over every indexed document. The graph only *enhances*
    retrieval, and search works without it, so this is safe to run (or skip) at
    any time. Nothing is fetched or executed; it is a pure transform of
    already-indexed rows.

    Term extraction (a regex pass per vocabulary term over every document's
    text) is cached per document and replayed while the document is unchanged,
    so a rebuild after a small ingest re-extracts only what moved and finishes
    in a fraction of the time. The graph it produces is the same one a
    from-scratch build would produce. Use --full to distrust that cache and
    re-extract everything.
    """
    settings = _setup(verbose)
    db = _db(settings)
    from blackbook.knowledge.graph import GraphBuilder

    console.print("Building knowledge graph…")
    stats = GraphBuilder(db).rebuild(incremental=not full)
    _print_graph_stats(stats)
    console.print("[green]Done.[/green]")
    db.close()


@graph_app.command("show")
def graph_show(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
):
    """Show current knowledge-graph statistics without rebuilding."""
    settings = _setup(verbose)
    db = _db(settings)
    from blackbook.knowledge.graph import GraphStats, writeup_coverage

    counts = db.counts()
    stats = GraphStats(
        entities=counts.get("entities", 0),
        relationships=counts.get("relationships", 0),
        documents=counts.get("documents", 0),
    )
    for e in db.list_entities():
        stats.by_entity_type[e["entity_type"]] = (
            stats.by_entity_type.get(e["entity_type"], 0) + 1
        )
    coverage = writeup_coverage(db)
    if as_json:
        payload = {
            "entities": stats.entities,
            "relationships": stats.relationships,
            "documents": stats.documents,
            "by_entity_type": stats.by_entity_type,
            "by_predicate": stats.by_predicate,
            "writeup_coverage": coverage.as_dict(),
        }
        console.print_json(json.dumps(payload))
        db.close()
        return
    if stats.entities == 0:
        console.print(
            "[yellow]Graph is empty.[/yellow] Run `blackbook graph build` "
            "(retrieval still works without it)."
        )
    else:
        _print_graph_stats(stats)
    if coverage.documents:
        _print_writeup_coverage(coverage)
    db.close()


@graph_app.command("neighbors")
def graph_neighbors(
    entity: str = typer.Argument(..., help="Entity name exactly as stored in the graph"),
    entity_type: Optional[str] = typer.Option(None, "--type", help="Restrict the lookup to one entity type"),
    depth: int = typer.Option(1, "--depth", "-d", min=1, max=3, help="Hops to walk (1-3)"),
    direction: str = typer.Option("both", "--direction", help="Follow out, in, or both directions"),
    predicate: Optional[list[str]] = typer.Option(None, "--predicate", "-p", help="Only follow these predicates (repeatable)"),
    limit: int = typer.Option(25, "--limit", min=1, max=100, help="Max neighbours expanded per node"),
    max_nodes: int = typer.Option(60, "--max-nodes", min=1, max=300, help="Max entities in the result"),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Walk the knowledge graph outward from one entity.

    The same traversal the ``knowledge_graph`` MCP tool performs, so the terminal
    and the server cannot drift apart. Traversal is exact-match on the stored
    name; a near miss prints the candidates instead of guessing.
    """
    from blackbook.mcp.schemas import GraphTraversalInput
    from blackbook.mcp.tools import KnowledgeTools

    settings = _setup(verbose)
    db = _db(settings)
    out = KnowledgeTools(db, settings).knowledge_graph(
        GraphTraversalInput(
            entity=entity,
            entity_type=entity_type,  # type: ignore[arg-type]
            depth=depth,
            direction=direction,  # type: ignore[arg-type]
            predicates=predicate,
            limit=limit,
            max_nodes=max_nodes,
        )
    )
    if as_json:
        console.print_json(out.model_dump_json())
        db.close()
        if not out.found:
            raise typer.Exit(code=1)
        return

    if not out.found:
        console.print(f"[yellow]{out.note}[/yellow]")
        db.close()
        raise typer.Exit(code=1)

    console.print(
        f"[bold]{out.resolved}[/bold] [dim]({out.entity_type})[/dim]"
        f"  {len(out.nodes) - 1} entities, {len(out.edges)} edges"
        f" [dim]depth {out.depth}, {out.direction}[/dim]"
    )
    if out.edges:
        table = Table(show_lines=False)
        table.add_column("subject", overflow="fold")
        table.add_column("predicate", style="cyan", no_wrap=True)
        table.add_column("object", overflow="fold")
        table.add_column("conf", justify="right", no_wrap=True)
        table.add_column("sup", justify="right", no_wrap=True)
        table.add_column("evidence", overflow="fold", style="dim")
        for e in out.edges:
            table.add_row(
                e.subject,
                e.predicate,
                e.object,
                f"{e.confidence:.2f}",
                str(e.support),
                (e.evidence.title or "") if e.evidence else "",
            )
        console.print(table)
    if out.note:
        console.print(f"[yellow]{out.note}[/yellow]")
    db.close()


@app.command()
def embed(
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Only embed chunks from this source"),
    reembed: bool = typer.Option(False, "--reembed", help="Delete existing vectors for the model first"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """(Re)embed indexed chunks for semantic search without re-ingesting.

    Requires the semantic extra (``pip install "blackbook-mcp[semantic]"``) and
    ``embeddings.enabled`` in config. Embeds only chunks missing a current-model
    vector unless ``--reembed`` is given.
    """
    settings = _setup(verbose)
    if not settings.embeddings.enabled:
        err_console.print(
            "[red]Embeddings are disabled.[/red] Set embeddings.enabled=true in your config."
        )
        raise typer.Exit(code=2)

    from blackbook.embeddings import Embedder, EmbeddingsUnavailable, embed_missing_chunks

    try:
        embedder = Embedder(
            settings.embeddings.model,
            device=settings.embeddings.device,
            batch_size=settings.embeddings.batch_size,
        )
    except EmbeddingsUnavailable as e:
        err_console.print(f"[red]Semantic extra unavailable:[/red] {e}")
        raise typer.Exit(code=2)

    db = _db(settings)
    # None => every indexed chunk; an explicit --source scopes to that source
    # (even if it is currently disabled in config, so re-embeds still reach it).
    if source and db.get_source(source) is None:
        err_console.print(f"[red]Source not indexed:[/red] {source}")
        db.close()
        raise typer.Exit(code=2)
    source_ids = [source] if source else None

    if reembed:
        with db.session():
            removed = db.delete_embeddings(embedder.model_name, source_ids=source_ids)
        scope = f" in {source}" if source else ""
        console.print(
            f"Removed {removed} existing vectors for {embedder.model_name}{scope}."
        )

    console.print(
        f"Embedding with [cyan]{embedder.model_name}[/cyan] (dim={embedder.dim})…"
    )
    with console.status("Encoding chunks…") as status:
        def progress(done, total):
            status.update(f"Encoding chunks… {done}/{total}")

        n = embed_missing_chunks(db, embedder, source_ids=source_ids, on_progress=progress)

    total = db.embedding_count(embedder.model_name)
    ui.success(f"Embedded {n} chunks. Total vectors: {total}", console)
    db.close()


@app.command()
def doctor(verbose: bool = typer.Option(False, "--verbose", "-v")):
    """Run diagnostics on the BlackBook installation."""
    settings = _setup(verbose)
    checks: list[tuple[str, str, str]] = []  # (name, severity, detail)

    # Database
    try:
        db = _db(settings)
        counts = db.counts()
        checks.append(("database", _OK, f"ok ({settings.db_path})"))
    except Exception as e:
        checks.append(("database", _FAIL, str(e)))
        db = None

    # FTS5
    if db is not None:
        try:
            db.conn.execute("SELECT rowid FROM chunks_fts LIMIT 1")
            checks.append(("fts5 index", _OK, "queryable"))
        except Exception as e:
            checks.append(("fts5 index", _FAIL, str(e)))

    # Sources configured + directory availability
    for s in settings.sources:
        if s.type == "filesystem" and s.directory:
            from pathlib import Path

            if not s.enabled:
                checks.append((f"source dir: {s.id}", _OK, "disabled"))
            else:
                ok = Path(s.directory).is_dir()
                sev = _OK if ok else _WARN  # empty optional source is a warning
                checks.append((f"source dir: {s.id}", sev, s.directory))
        else:
            checks.append((f"source: {s.id}", _OK, f"enabled={s.enabled}"))

    # Embeddings
    if settings.embeddings.enabled:
        try:
            import sentence_transformers  # noqa: F401

            checks.append(("embeddings", _OK, settings.embeddings.model))
            # Coverage: how many chunks have a vector for the configured model.
            if db is not None:
                total_chunks = counts.get("chunks", 0)
                embedded = db.embedding_count(settings.embeddings.model)
                if total_chunks == 0:
                    checks.append(("embedding coverage", _OK, "no chunks yet"))
                elif embedded == 0:
                    checks.append(
                        ("embedding coverage", _WARN, "0 embedded — run `blackbook embed`")
                    )
                elif embedded < total_chunks:
                    checks.append(
                        (
                            "embedding coverage",
                            _WARN,
                            f"{embedded}/{total_chunks} embedded — run `blackbook embed`",
                        )
                    )
                else:
                    checks.append(
                        ("embedding coverage", _OK, f"{embedded}/{total_chunks} embedded")
                    )
        except Exception as e:
            checks.append(("embeddings", _FAIL, f"enabled but unavailable: {e}"))
    else:
        checks.append(("embeddings", _OK, "disabled (lexical-only)"))

    # Search diagnostics: which backends a search can actually use, as opposed
    # to what is configured. The embeddings rows above report the raw facts;
    # this row states their consequence, which is the part a reader cannot
    # derive from them, because a semantic request that cannot be served
    # degrades to lexical and answers anyway.
    #
    # It reuses the MCP layer's readiness reporter (config plus one COUNT(*),
    # never a model load) so `doctor` and the `blackbook://sources` view cannot
    # disagree about the same install. WARN, not FAIL, when semantic is
    # switched on but inert: retrieval still works, from lexical, and every
    # result says so. A missing [semantic] extra is the one case deliberately
    # not handled here, because the `embeddings` check above already FAILs on
    # it; re-testing it would duplicate the detection rather than add to it.
    if db is not None:
        from blackbook.mcp.tools import KnowledgeTools

        semantic = KnowledgeTools._semantic_status(db, settings)
        if not semantic.enabled:
            checks.append(
                ("search diagnostics", _OK, "lexical only (semantic disabled)")
            )
        elif semantic.vectors:
            checks.append(
                (
                    "search diagnostics",
                    _OK,
                    f"lexical + semantic ({semantic.vectors} vectors)",
                )
            )
        else:
            checks.append(
                (
                    "search diagnostics",
                    _WARN,
                    f"semantic enabled but no vectors for {semantic.model}; "
                    "searches degrade to lexical, run `blackbook embed`",
                )
            )

    # Index staleness
    if db is not None:
        docs = counts.get("documents", 0)
        chunks = counts.get("chunks", 0)
        if docs == 0:
            checks.append(("index populated", _WARN, "empty — run `blackbook ingest`"))
        elif chunks == 0:
            checks.append(("index populated", _FAIL, f"{docs} docs but 0 chunks"))
        else:
            checks.append(("index populated", _OK, f"{docs} docs / {chunks} chunks"))

    # Writeup coverage: which sources feed the case-study layer.
    #
    # Reported OK, not WARN, when sources contribute no writeups. A reference
    # source such as hacktricks contributes none by design, so warning on every
    # gap would cry wolf on a healthy install and dilute the severities that do
    # mean something. The detail names the largest gap instead, which is the
    # judgement a human wants to make. The one case that does escalate is a
    # built graph disagreeing with the corpus, because that is both actionable
    # and invisible from the graph's own totals.
    if db is not None and counts.get("documents", 0):
        from blackbook.knowledge.graph import writeup_coverage

        coverage = writeup_coverage(db)
        detail = (
            f"{coverage.eligible}/{coverage.documents} documents are writeups, "
            f"from {coverage.sources_with_writeups} of "
            f"{len(coverage.docs_by_source)} sources"
        )
        if coverage.gaps:
            biggest = max(coverage.gaps, key=lambda g: g["documents"])
            detail += (
                f"; largest gap {biggest['source_id']} "
                f"({biggest['documents']} docs)"
            )
        sev = _OK
        if coverage.graph_built and coverage.graph_writeups != coverage.eligible:
            sev = _WARN
            detail += (
                f"; graph holds {coverage.graph_writeups} writeup entities, "
                "run `blackbook graph build`"
            )
        checks.append(("writeup coverage", sev, detail))

    table = Table(title="BlackBook Doctor")
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_column("Detail", overflow="fold")
    any_fail = False
    sev_label = {
        _OK: "[green]OK[/green]",
        _WARN: "[yellow]WARN[/yellow]",
        _FAIL: "[red]FAIL[/red]",
    }
    for name, sev, detail in checks:
        if sev == _FAIL:
            any_fail = True
        table.add_row(name, sev_label[sev], detail)
    console.print(table)
    if db is not None:
        db.close()
    if any_fail:
        raise typer.Exit(code=1)


@app.command(name="eval")
def eval_cmd(
    k: int = typer.Option(5, "--k", "-k", help="Rank cutoff for recall@k / hit-rate"),
    min_mrr: float = typer.Option(0.8, "--min-mrr", help="Fail if MRR falls below this floor"),
    min_recall: float = typer.Option(0.8, "--min-recall", help="Fail if mean recall@k falls below this floor"),
    as_json: bool = typer.Option(False, "--json", help="Emit the full report as JSON instead of tables"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run the offline retrieval benchmark and report quality metrics.

    Builds a small, clearly-synthetic benchmark corpus in an in-memory database
    (nothing touches your real index), indexes it through the production chunker
    and storage layer, then runs the labelled gold query set through the exact
    retriever the MCP tools use. It measures three things:

    * citation integrity — every returned chunk must resolve to real indexed
      text (the "no hallucinated citations" invariant). A hard gate: any
      unresolved citation exits non-zero.
    * ranking quality — recall@k and MRR against the gold labels, gated by the
      ``--min-mrr`` / ``--min-recall`` regression floors.
    * latency — per-query p50 / p95, reported for regression tracking.

    The run is hermetic and lexical-only (embeddings disabled) so results are
    deterministic and reproducible across machines.
    """
    ui.configure_logging(verbose, level=logging.DEBUG if verbose else logging.WARNING)

    # A clean, embeddings-off Settings so the benchmark is deterministic and
    # offline regardless of the user's config; an isolated in-memory DB so it
    # never reads or writes the real knowledge base.
    from blackbook.config import Settings
    from blackbook.eval import build_eval_corpus, run_evaluation

    settings = Settings()
    db = Database(":memory:")
    try:
        corpus = build_eval_corpus(db)
        report = run_evaluation(db, settings, k=k)
    finally:
        db.close()

    if as_json:
        import json

        payload = {"corpus": corpus, **report.as_dict()}
        # Raw stdout so the output is machine-parseable (no Rich colouring,
        # wrapping, or trailing human text).
        print(json.dumps(payload, indent=2))
    else:
        summary = Table(title="BlackBook Retrieval Benchmark")
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", justify="right")
        summary.add_row("Corpus", f"{corpus['documents']} docs / {corpus['chunks']} chunks")
        summary.add_row("Queries", str(len(report.query_results)))
        summary.add_row("MRR", f"{report.mrr:.3f}")
        summary.add_row(f"Hit-rate@{k}", f"{report.hit_rate:.3f}")
        summary.add_row(f"Mean recall@{k}", f"{report.mean_recall_at_k:.3f}")
        summary.add_row(
            "Citation integrity",
            f"{report.citation_integrity:.4f} "
            f"({report.citations_resolved}/{report.citations_total})",
        )
        summary.add_row(
            "Latency p50 / p95",
            f"{report.latency_p50_ms:.2f} / {report.latency_p95_ms:.2f} ms",
        )
        console.print(summary)

        detail = Table(title="Per-query")
        detail.add_column("Query", style="cyan")
        detail.add_column("Mode")
        detail.add_column("RR", justify="right")
        detail.add_column(f"R@{k}", justify="right")
        detail.add_column("Top result", overflow="fold")
        for qr in report.query_results:
            top = qr.ranked[0] if qr.ranked else "—"
            detail.add_row(
                qr.qid, qr.mode, f"{qr.reciprocal_rank:.2f}",
                f"{qr.recall_at_k:.2f}", top,
                style=None if qr.hit else "yellow",
            )
        console.print(detail)

    # Gates. Citation integrity is the hard invariant; the ranking floors guard
    # against regressions in the shipped retriever.
    failures: list[str] = []
    if report.citation_integrity < 1.0:
        unresolved = report.citations_total - report.citations_resolved
        failures.append(
            f"citation integrity {report.citation_integrity:.4f} ({unresolved} unresolved)"
        )
    if report.mrr < min_mrr:
        failures.append(f"MRR {report.mrr:.3f} < {min_mrr:.3f}")
    if report.mean_recall_at_k < min_recall:
        failures.append(f"mean recall@{k} {report.mean_recall_at_k:.3f} < {min_recall:.3f}")

    if failures:
        ui.fail("Benchmark FAILED: " + "; ".join(failures), err_console)
        raise typer.Exit(code=1)
    if not as_json:
        ui.success("Benchmark passed.", console)


@app.command()
def serve(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    banner: bool = typer.Option(True, "--banner/--no-banner", help="Show the startup banner."),
    http: bool = typer.Option(
        False,
        "--http/--stdio",
        help="Serve over HTTP (streamable-http) at a URL+port instead of stdio.",
    ),
    host: Optional[str] = typer.Option(
        None, "--host", help="HTTP bind address (default from config: 127.0.0.1)."
    ),
    port: Optional[int] = typer.Option(
        None, "--port", help="HTTP port (default from config: 8890)."
    ),
):
    """Run the MCP server.

    Default is stdio (for Claude Code / Cursor / VS Code, which spawn the
    process and talk over stdin/stdout). Pass ``--http`` to run a long-lived
    network server reachable over a URL + port, with a ``GET /health`` check —
    connect a client to ``http://<host>:<port>/mcp``.
    """
    settings = _setup(verbose)
    from blackbook.server import build_server

    # CLI flags override config for this run.
    if host is not None:
        settings.server.host = host
    if port is not None:
        settings.server.port = port

    # Open the db up front so the banner can report live corpus/graph stats
    # and the server reuses the same handle. All chrome goes to stderr — the
    # JSON-RPC protocol owns stdout.
    db = _db(settings)
    server = build_server(settings, db)

    if http:
        from blackbook.server import run_http

        base = f"http://{settings.server.host}:{settings.server.port}"
        if banner:
            ui.print_banner(
                db=db, transport=f"streamable-http · {base}{settings.server.path}"
            )
        ui.info(f"MCP endpoint:  {base}{settings.server.path}", err_console)
        ui.info(f"Health check:  {base}/health", err_console)
        if (settings.server.auth_token or "").strip():
            ui.info("Auth:          bearer token required", err_console)
        run_http(server, settings)
    else:
        if banner:
            ui.print_banner(db=db, transport="stdio")
        server.run(transport="stdio")


case_app = typer.Typer(
    add_completion=False,
    help="Investigation case management (the local case layer).",
)
app.add_typer(case_app, name="case")


@case_app.command("export")
def case_export(
    name: str = typer.Argument(..., help="Case name"),
    out: Optional[Path] = typer.Option(
        None, "--out", "-o", help="Output file (default: ~/.blackbook/exports/<case>-<date>.md)"
    ),
    stdout: bool = typer.Option(False, "--stdout", help="Print to stdout instead of a file"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Export a case as a standalone Markdown report."""
    from blackbook.knowledge.case_export import (
        build_case_state,
        export_filename,
        render_case_markdown,
    )

    settings = _setup(verbose)
    db = _db(settings)
    state = build_case_state(db, name)
    if state is None:
        console.print(f"[red]Case '{name}' not found.[/red]")
        raise typer.Exit(code=1)
    markdown = render_case_markdown(state)
    db.close()

    if stdout:
        console.print(markdown, markup=False, highlight=False)
        return
    if out is None:
        out = settings.home / "exports" / export_filename(name)
    out = out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")
    console.print(f"[green]Exported {len(state.observations)} observations to {out}[/green]")


@app.command()
def backup(
    out: Optional[Path] = typer.Option(
        None, "--out", "-o", help="Backup file (default: ~/.blackbook/backups/blackbook-<date>.db)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Create a consistent, compact backup of the knowledge base.

    Uses SQLite's ``VACUUM INTO``, which snapshots the database while it is
    in use (readers keep working) and defragments the copy. The backup is a
    fully standalone database — restore by pointing ``BLACKBOOK_DATABASE__PATH``
    (or the config's ``database.path``) at it.
    """
    from datetime import date

    settings = _setup(verbose)
    db = _db(settings)
    if out is None:
        out = settings.home / "backups" / f"blackbook-{date.today().isoformat()}.db"
    out = out.expanduser()
    if out.exists():
        console.print(f"[red]{out} already exists; refusing to overwrite.[/red]")
        raise typer.Exit(code=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        db.conn.execute("VACUUM INTO ?", (str(out),))
    except Exception as e:
        console.print(f"[red]Backup failed:[/red] {e}")
        raise typer.Exit(code=1) from e
    size = out.stat().st_size
    console.print(
        f"[green]Backed up {settings.db_path} -> {out}[/green] "
        f"({size / (1024 * 1024):.1f} MiB)"
    )
    db.close()


@app.command()
def version():
    """Print the BlackBook version."""
    from blackbook import __version__

    console.print(f"blackbook {__version__}")


if __name__ == "__main__":
    app()
