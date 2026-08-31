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
| `payloads`, `hacker_recipes` | `GithubMarkdownAdapter` | git | Generic GitHub-markdown adapter, config-driven |
| `gtfobins` | `GtfoBinsAdapter` | git | Extensionless YAML entries under `_gtfobins/`, one doc per binary |
| `lolbas` | `LolbasAdapter` | git | `yml/<Category>/<Name>.yml` entries rendered to markdown |
| `loobins` | `LooBinsAdapter` | git | macOS binaries; `LOOBins/<Name>.yml` with use cases + ATT&CK tactics |
| `wadcoms` | `WadcomsAdapter` | git | Windows/AD cheat sheets; payload lives in markdown front matter |
| `attack` | `MitreAttackAdapter` | git | MITRE ATT&CK enterprise STIX bundle; techniques by ATT&CK ID |

Add a future source by subclassing `SourceAdapter` (or, for a GitHub repo,
`GithubTarballAdapter`) and registering it in
`blackbook/ingestion/__init__.py:ADAPTER_REGISTRY`. Unregistered `git`-type
sources fall back to the generic GitHub-markdown adapter, and `filesystem`
sources to the PDF adapter.

## GitHub sources (generic)

All GitHub-backed sources share `GithubTarballAdapter`'s fetch mechanics:

* Downloaded as a **tarball over HTTPS** (`codeload.github.com`), never via
  shell or git execution.
* Change detection queries the latest commit SHA and skips re-download when
  current.
* Tarball extraction is hardened against zip-slip/path traversal and skips
  device files and links.

Parsing is per-source. `GithubMarkdownAdapter` walks markdown files and is
configuration-driven, so most new markdown sources need no code:

* `ref` — branch to track.
* `include_glob` — which files to index (default `**/*.md`).
* `content_root` — restrict indexing to a repo subtree (and strip it from
  category breadcrumbs).
* `site_url` — map citations to the published site (`/index` collapses to the
  directory); without it, citations point at the GitHub blob URL.

GTFOBins and LOLBAS render their YAML corpora to structured markdown instead:
one document per binary, every abuse function/command preserved with the
contexts (sudo, suid, unprivileged) it works in. GTFOBins alias-only entries
(`alias: mawk`) are folded into their target as alternate names rather than
becoming near-empty documents. LOOBins does the same for macOS binaries
(use cases with code, ATT&CK tactics as slugified categories, paths,
detections), and WADComs parses markdown files whose entire payload is the
YAML front matter (description, command, services, items, OS, references)
with an empty body.

## MITRE ATT&CK

* The enterprise-attack STIX bundle is downloaded from the official
  `mitre-attack/attack-stix-data` repository (~54 MB) and cached under
  `raw/attack/bundle.json`; the file's latest commit SHA is the no-op marker.
* Each `attack-pattern` object becomes one document whose `external_id` is the
  ATT&CK technique ID (e.g. `T1558.003`). Revoked and deprecated objects are
  skipped.
* Platforms land as lowercase category tags and tactics as dashed slugs, so
  `platform`/`categories` hard filters work against ATT&CK material.
* With the source ingested, `knowledge_technique` resolves its term to an
  ATT&CK ID and enriches the dossier with the official tactics, platforms, and
  attack.mitre.org link, citing the ATT&CK record first.

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
