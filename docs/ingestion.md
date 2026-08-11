# Ingestion

Ingestion turns raw source material into normalized, chunked, indexed documents.

## The adapter interface

```python
class SourceAdapter(ABC):
    source_id: str
    def fetch(self, force: bool = False) -> None: ...       # acquire raw material
    def iter_documents(self) -> Iterator[ParsedDocument]: ...  # parse + chunk
```

Adapters never touch the database and never execute shell commands. Persistence,
dedup, and indexing are centralized in `IngestionPipeline`.

Registered adapters:

| Source ID | Adapter | Type | Notes |
|-----------|---------|------|-------|
| `hacktricks` | `HackTricksAdapter` | git | Tarball over HTTPS (no shell/git exec) |
| `0xdf` | `ZeroDFAdapter` | website | Jekyll blog; rich structured metadata |
| `local_pdfs` | `PDFAdapter` | filesystem | Font-aware heading/code detection, per-page citations |

Add a future source by subclassing `SourceAdapter` and registering it in
`blackbook/ingestion/__init__.py:ADAPTER_REGISTRY`.

## HackTricks

* Fetched as a **git tarball over HTTPS** (`codeload.github.com`), not via a shell.
* Change detection queries the latest commit SHA and skips re-download when current.
* Tarball extraction is hardened against zip-slip/path traversal and skips device
  files and links.
* Each page's **category hierarchy** (directory path) and **in-page heading
  breadcrumb** are preserved into every chunk's `section_path`.
* URLs are mapped to the published `book.hacktricks.xyz` pages.

## 0xdf

* The index page is scanned for dated post URLs; each post is cached under
  `raw/0xdf/pages/` and only re-fetched when new (or `--force`).
* Structured metadata is extracted with confidence tagging:
  * `og:title` → machine name + kind (`HTB`, `PG`, ...)
  * `article:published_time` → date
  * `og:description` → the author's attack-chain summary
  * `.htb-card` → difficulty, OS, release/retire dates, creator
  * H2/H3 headings → the attack-chain narrative (`Recon`, `Shell as X`, ...)
* Services / techniques / tools are extracted by a lightweight heuristic and are
  explicitly marked `metadata_inferred: true`. Fields that can't be confidently
  extracted are left `None` (nullable metadata) rather than guessed.

## Local PDFs

* Only files inside the configured `directory` are read; path traversal out of it is
  rejected, and oversized files are skipped.
* Text is extracted per page with `pypdf`. A `visitor_text` hook captures each text
  run's **font name and size**, which drives structural detection:
  * **Headings** — a run whose font is ≥1.15× the page's median body size, or that
    matches a numbered-section pattern (`1.`, `2.3`, …), becomes a section breadcrumb.
  * **Code blocks** — runs in a monospaced font (Courier/Consolas/Menlo/…) are emitted
    as intact `code` chunks rather than being reflowed as prose.
* Chunks are built per page with `chunk_structured_pages`, carrying the running heading
  breadcrumb and the exact **page number** for citation (`Windows Privilege Escalation
  > 1. Services`, page 1).
* Document metadata (title/author/subject) is read from the PDF. Generator boilerplate
  (`untitled`, `anonymous`, …) is discarded; when the title is missing it falls back to
  the filename and is flagged `title_inferred`. Everything detected structurally is
  marked `inferred` in the document metadata.
* PDFs default to `authority: user` — they are **not** assumed authoritative, and
  that authority is surfaced in results.

Ingest an ad-hoc directory without editing config:

```bash
blackbook ingest --pdf-dir /path/to/pdfs --source local_pdfs
```

## Embedding (optional, Phase 3)

When `embeddings.enabled` is true **and** the `[semantic]` extra is installed, the
pipeline embeds each source's new chunks inline right after they are written — so a
plain `blackbook ingest` leaves the semantic index current with no extra step. The
embedder is built once per run and passed to `IngestionPipeline(db, embedder=...)`;
when embeddings are disabled or the extra is missing, `embedder` is `None`, ingestion
is lexical-only, and (if enabled-but-missing) a one-line warning names the extra to
install. Only chunks lacking a current-model vector are embedded, so re-ingesting an
unchanged source embeds nothing. See [retrieval.md](retrieval.md) for how the vectors
are stored and searched, and `blackbook embed` for (re)embedding without re-ingesting.

## Knowledge graph (Phase 4)

After a run that writes new or changed chunks, ingestion **automatically rebuilds
the knowledge graph** from the index (a full, idempotent transform of already-indexed
rows — nothing is fetched or executed). The rebuild is best-effort: a graph failure
logs a warning and never fails the ingest, because the graph only *enhances*
retrieval. It is skipped when no chunks changed, and can be disabled with
`ingest --no-graph` / `update --no-graph`. Build or inspect it directly with
`blackbook graph build` / `blackbook graph show`. See
[architecture.md](architecture.md) for the entity/relationship model and
[retrieval.md](retrieval.md) for where the graph is (and is not) consulted.

## Change detection & dedup

* Each document's full text is hashed (SHA-256). Re-ingesting an unchanged document
  is a no-op (`skipped_unchanged`).
* Within a document, repeated chunks are collapsed on a **normalized** hash
  (lowercased, alphanumeric tokens only), so re-formatted copies that differ only in
  whitespace/casing/punctuation still dedupe.
* Across documents, the reranker drops a hit that is a near-duplicate (word-shingle
  Jaccard ≥ 0.9) of one already selected — common when a PDF restates HackTricks
  material — so the result set stays diverse. See `retrieval/dedup.py`.

## Running

```bash
blackbook ingest                       # all enabled sources (+ graph rebuild)
blackbook ingest --source hacktricks   # one source
blackbook ingest --source 0xdf --force # re-fetch even if unchanged
blackbook ingest --no-graph            # skip the post-ingest graph rebuild
```
