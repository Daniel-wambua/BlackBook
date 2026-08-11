# Architecture

BlackBook MCP is a **hybrid knowledge system**, not a query→embedding→dump pipeline.

```
                 USER / CLAUDE
                       |
                       v
               Research Request
                       |
          +------------+-------------+
          |            |             |
          v            v             v
      Metadata     Full Text      Semantic      (semantic is optional, Phase 3)
       Search       Search         Search
          |            |             |
          +------------+-------------+
                       |
                       v
                   Reranking          (authority, platform/category match, diversity,
                       |                intent-mode bias for technique/case_similarity)
                       v
               Source Validation      (every claim carries provenance)
                       |
                       v
              Research Response
                       |
                       v
               Exact Citations        (chunk_id -> exact excerpt)
```

The **knowledge graph** sits *beside* this path, not inside it: retrieval never
expands a query through graph edges. The `knowledge_technique` and
`knowledge_case_search` tools consult the graph to enrich already-retrieved results
with evidence-linked neighbours and technique annotations. Retrieval behaves
identically whether or not the graph is built.

## Layers

| Layer | Module | Responsibility |
|-------|--------|----------------|
| MCP | `blackbook/mcp/` | Tool surface, input validation, structured output |
| Server | `blackbook/server.py` | FastMCP wiring over stdio |
| Retrieval | `blackbook/retrieval/` | Lexical (FTS5), semantic (optional), hybrid facade, reranking, chunking |
| Ingestion | `blackbook/ingestion/` | `SourceAdapter` per source; fetch + parse to normalized documents |
| Knowledge | `blackbook/knowledge/` | Source resolution (citation -> excerpt); knowledge graph (`graph.py`); controlled vocabulary (`vocab.py`); research packets |
| Storage | `blackbook/storage/` | SQLite (FTS5 + JSON1), schema, migrations, persistence |
| Config | `blackbook/config.py` | Layered settings (defaults < YAML < env) |
| CLI | `blackbook/cli/` | Administration & diagnostics |

## Key design decisions

1. **Single SQLite file** holds documents, chunks, the FTS5 index, the knowledge
   graph, and case state. FTS5 provides BM25 lexical search; JSON1 carries flexible
   metadata. One embeddable, transactional store instead of several services.

2. **Lexical-first retrieval.** FTS5 BM25 is the always-available path. Semantic
   search is an *optional* backend merged into the same `HybridRetriever`, so it can
   be disabled without changing any caller. We did not add vectors "because RAG."

3. **SourceAdapter interface.** Every source (HackTricks, 0xdf, local PDFs, future
   ones) implements `fetch()` + `iter_documents()`. Adding a source never requires
   redesigning retrieval — only a new adapter.

4. **Structure-preserving chunking.** Documents are split on heading/code/paragraph
   boundaries and every chunk keeps its `section_path` breadcrumb. The hierarchy is
   load-bearing for citation quality and is never flattened away.

5. **Provenance is mandatory.** Every returned reference maps to a real indexed
   chunk and can be resolved to its exact text via `knowledge_source`. Nothing is
   fabricated.

## Data flow (ingestion)

```
SourceAdapter.fetch()         (git tarball over HTTPS / website / filesystem scan)
        |
        v
iter_documents()  ->  ParsedDocument{title, url, categories, metadata, chunks}
        |
        v
IngestionPipeline  ->  content-hash dedup  ->  upsert document  ->  replace chunks
        |
        v
SQLite  (documents + chunks + FTS5 sync via triggers)
```

## Knowledge graph (Phase 4)

A lightweight graph derived **entirely from already-indexed rows** — `GraphBuilder`
reads documents/chunks and their metadata; it never fetches or executes anything. It
is rebuilt idempotently (a full clear-and-rebuild) after any ingest that writes chunks,
or on demand via `blackbook graph build`.

**Entities** (`entities`, unique on `(name, entity_type)`):

| Type | Meaning | Keyed by |
|------|---------|----------|
| `technique` | An attack/technique term from the controlled vocabulary | canonical technique name |
| `tool` | A named tool (impacket, nmap, …) | tool name |
| `service` | A network service / protocol (kerberos, smb, …) | service name |
| `os` | An operating system / platform | platform name |
| `writeup` | A hands-on case/writeup document | document **title** (with `doc_id` in `meta`) |
| `source` | A configured corpus source | source id/name |

**Relationships** (`relationships`: `subject_id --predicate--> object_id`):

| Predicate | Subject → Object | Typical origin |
|-----------|------------------|----------------|
| `documented_by` | technique / writeup → source | item appears in a document owned by that source |
| `demonstrated_in` | technique → writeup | writeup document exercises the technique |
| `uses` | technique → tool | tool co-occurs with the technique (inferred) |
| `targets` | technique → service | service co-occurs with the technique (inferred) |
| `used_in` | tool → writeup | tool appears in a writeup |
| `present_in` | service → writeup | service appears in a writeup |
| `runs_on` | writeup → os | writeup targets a machine on that platform |

Every non-structural edge records the document it was extracted from
(`evidence_doc_id`, resolvable to real indexed text), a `confidence` in `[0,1]`, and an
`inferred` flag for heuristic (co-occurrence) edges versus directly-observed ones. No
edge is fabricated: an edge with no backing document is never written.

The graph **enhances the tool layer, not retrieval** (see the diagram note above and
[retrieval.md](retrieval.md)). It is consumed by `knowledge_technique` (technique →
sources/tools/services/writeups dossier) and `knowledge_case_search` (annotating each
writeup hit with the techniques it demonstrates).

## Read-only boundary

BlackBook reads its local database and configured knowledge directories only. It has
no tool that executes commands, fetches arbitrary user-supplied URLs, or touches
remote systems. Execution is the job of a separate MCP (e.g. HexStrike).

The one tool that *writes*, `knowledge_context`, stays inside this boundary: it
persists **local investigation state** — cases and their observations — to the same
SQLite file, a separate user-authored layer that is never derived from ingestion and
never leaves the machine. It executes nothing, fetches nothing, and touches no remote
system; it also has no delete action, so it cannot destroy prior state. Everything
else in the tool surface is strictly read-only over the index.
