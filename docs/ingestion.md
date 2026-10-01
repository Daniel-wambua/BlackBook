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
| `internal_all_the_things` | `GithubMarkdownAdapter` | git | AD/internal-network cheat sheets (MkDocs, `docs/` subtree); config-only source via the git-type fallback |
| `htb_writeups` | `GithubMarkdownAdapter` | git | Moamen Basel's HTB writeups + cheatsheets (Jekyll); config-only source via the git-type fallback |
| `website` sources | `WebsiteAdapter` | website | Bounded same-origin HTML crawling for configured documentation and disclosure pages; optional `path_prefix` prevents unrelated-site crawling |
| `rss` sources | `RssAdapter` | rss | Conditional RSS/Atom feed ingestion with cached validators |
| `webhacklist` | `GithubMarkdownAdapter` | git | Full WebHackList Markdown archive with exact cross-document deduplication |
| `martian1337_notes`, `pentest_book` | `GitbookAdapter` | gitbook | Published GitBook spaces, read via `llms.txt` + per-page markdown |
| `nt_functions` | `FunctionIndexAdapter` | function_index | One HTML index of ~24k decompiled NT kernel functions, one page per function; `include_prefixes` selects ~7.6k |

Add a future source by subclassing `SourceAdapter` (or, for a GitHub repo,
`GithubTarballAdapter`) and registering it in
`blackbook/ingestion/__init__.py:ADAPTER_REGISTRY`. Unregistered `git`-type
sources fall back to the generic GitHub-markdown adapter, `website` sources to
the bounded website adapter, `rss` sources to the feed adapter, `gitbook`
sources to the GitBook adapter, `function_index` sources to the function-index
adapter, and `filesystem` sources to the PDF adapter.

Generic website sources follow links from the configured URL, stay on the same
origin, optionally stay below `path_prefix`, skip non-HTML assets, cache pages
under the configured raw directory, and respect `max_files`,
`max_document_bytes`, and `request_delay`. GitHub sources may use
comma-separated `include_glob` patterns such as `**/*.md,**/*.txt`.
Sources that set `deduplicate_chunks: true` skip exact normalized chunk copies
already present anywhere in the corpus while retaining the first citation.

WebHackList uses this mode and indexes all Markdown files in the repository,
including yearly technique lists, archived reference Markdown, and evaluation
notes. PDFs and other binary assets are intentionally skipped by the GitHub
extractor because the Markdown is the searchable source material.

Sources with an official RSS/Atom feed can use `type: rss` and `feed_url`.
Feed validators are cached and conditional requests are used when the server
supports them. The repository
sources `hacker101`, `hackerone_reports_index`, `hackerone_disclosed_reports`,
`hackerone_reports_metadata`, and `hackerone_bug_bounty_reports` provide the
available public educational, report-index, metadata, and disclosed-report
material instead.

## GitHub sources (generic)

All GitHub-backed sources share `GithubTarballAdapter`'s fetch mechanics:

* Downloaded as a **tarball over HTTPS** (`codeload.github.com`), never via
  shell or git execution.
* Large tarballs stream to a temporary file instead of being held in memory.
* Change detection queries the latest commit SHA and skips re-download when
  current.
* Tarball extraction is hardened against zip-slip/path traversal and skips
  device files, links, images, PDFs, and configured non-content directories.

Parsing is per-source. `GithubMarkdownAdapter` walks markdown files and is
configuration-driven, so most new markdown sources need no code:

* `ref` — branch to track.
* `include_glob` — which files to index (default `**/*.md`).
* `exclude_glob` — comma-separated fnmatch patterns (repo-relative,
  `/`-separated) for files to skip even when `include_glob` matches: repo
  plumbing, templates, link-index pages that duplicate other sources.
* `content_root` — restrict indexing to a repo subtree (and strip it from
  category breadcrumbs).
* `site_url` — map citations to the published site (`/index` collapses to the
  directory); without it, citations point at the GitHub blob URL. When a page
  carries a Jekyll `permalink` in its front matter, that permalink wins over
  the path-derived URL — it is the author's canonical link and can differ
  from the file path in case or shape.
* `categories` — source-level tags copied into every document, useful for
  separating educational material, report indexes, and archive content.
* `deduplicate_chunks` — when true, exact normalized chunk copies already in
  the corpus are skipped while preserving the first source citation.

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

## GitBook spaces

GitBook renders page bodies client-side, so crawling the published HTML yields
only sidebar and table-of-contents chrome. `type: gitbook` reads the two
machine-readable companions GitBook publishes instead:

* `<base_url>/llms.txt` — the space's full page index, one
  `- [Title](<url>.md)` line per page. Those URLs are the **published slugs**
  and routinely differ from any backing repository's file paths, which is why
  the index (not the repo) is the authority for both the page list and each
  page's citation link.
* `<base_url>/<page>.md` — the page body as clean markdown, headings and fenced
  code intact.

Only same-origin links below `base_url` are kept, so a space that links out to
other books does not silently widen the corpus. Pages and the parsed index are
cached under the source's raw directory; `fetch(force=False)` is a no-op once
the index is present. `max_files` caps the page count (default 2000), and the
cap is logged rather than applied silently.

Before indexing, three kinds of publisher chrome are stripped — the leading
"documentation index" blockquote, the trailing `# Agent Instructions` section,
and the `{% hint %}` / `{% tabs %}` / `{% embed %}` template markers (the tag
goes, the text inside it stays). Site-relative links are made absolute so
citations resolve. Pages the index still lists after they were moved or deleted
are served as HTTP 200 with a "Page Not Found" body; those are skipped rather
than indexed as bogus documents.

Because the published slugs are the citation URLs, point a GitBook source at the
live book (`https://www.pentest-book.com`) rather than the source repository —
mapping repo paths onto the site produces dead links.

## Function indexes

Some reference dumps publish one big HTML index that enumerates every item by
name, each entry linking to a small page of generated body text. The Windows NT
kernel function browser (`likeagod.revers.engineering/nt/`) is the motivating
case: a single 2 MB index lists 24,260 decompiled functions, and each links to a
page holding that function's Hex-Rays pseudocode plus the functions that call it.
`type: function_index` reads such a site:

1. Fetch and cache the index (`index.html`).
2. Parse its entries, keep only names matching `include_prefixes`, *then* apply
   `max_files`.
3. Fetch each selected page into `pages/<sha256(url)>.html`, skipping pages
   already cached so an interrupted run resumes instead of restarting.

**Filtering before capping is the point.** Such a site has no sitemap, feed or
`robots.txt`, so the index is the only discovery path — and a generic `website`
crawl bounded by `max_files` captures an *alphabetical slice* of the index and
stops, silently. With `include_prefixes` the cap bounds a known-selected set
rather than truncating index order. `include_prefixes` is a plain `startswith`
test, not an exact-name list, so `Mm` selects `MmMapIoSpace` and
`MmMapIoSpaceEx` alike; an empty list keeps every entry.

Each page becomes a document titled by its function name, whose body is the
pseudocode in a fenced `c` block followed by a `## Referenced by` list of
callers. Two deliberate omissions: no `# FunctionName` heading (the chunker
copies the document title into every chunk's breadcrumb, and the name is already
in the signature line), and no section at all when a page reports no callers
(the site's "No references." line is identical across thousands of leaf
functions, so it could never rank usefully). A lowercased subsystem facet is
derived from the name's module prefix (`MmMapIoSpace` → `mm`).

The pseudocode is fenced on purpose: the chunker never splits a fenced block, so
a function stays one intact code chunk instead of being cut mid-statement. The
cost is that the largest functions become a single oversized chunk — the site's
own pages top out around 8 KB, and a function with a very large caller list
(`ZwClose` lists 728) spreads that list across several word-count-sized text
chunks.

Two parsing details worth keeping in mind if you adapt this adapter. Tags are
stripped from the pseudocode using an **allowlist of tag names**, never a bare
`<[^>]+>`: decompiled C is full of literal `<` comparisons, and a permissive
pattern swallows everything from `if( a < b )` through the next real tag,
silently deleting code. And a `<br>` inside the block becomes a newline rather
than being dropped, so two statements on either side of one are not spliced into
a single corrupted identifier.

## Local PDFs

* Only files inside the configured `directory` are read; path traversal out of it is
  rejected, and oversized files are skipped.
* PDF extraction uses PyMuPDF by default for fast, page-by-page processing and
  font-aware spans. A page extraction failure yields an empty page instead of
  aborting the whole document.
* Text is extracted per page with PyMuPDF. Structured text spans expose each
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
the knowledge graph** from the index (an idempotent transform of already-indexed
rows; nothing is fetched or executed). The rebuild is best-effort: a graph failure
logs a warning and never fails the ingest, because the graph only *enhances*
retrieval. It is skipped when no chunks changed, and can be disabled with
`ingest --no-graph` / `update --no-graph`. Build or inspect it directly with
`blackbook graph build` / `blackbook graph show`; add `--full` to `graph build` to
ignore the per-document term cache and re-extract everything, which is only worth it
after changing the vocabulary itself. See
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
