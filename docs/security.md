# Security model

BlackBook is security-sensitive infrastructure. The design treats it as such.

## Read-only boundary

* **No command execution.** Nothing in BlackBook shells out. HackTricks is fetched
  as a tarball over HTTPS specifically to avoid invoking `git`/a shell.
* **No arbitrary URL fetching.** The only network reads are the configured source
  URLs (HackTricks tarball/commit API, the 0xdf site). Tool parameters cannot be
  used to make BlackBook fetch an arbitrary URL.
* **No remote-system modification.** BlackBook reads its local database and
  configured knowledge directories only.
* **Execution is out of scope.** Running Nmap/ffuf/Metasploit/Impacket, exploiting
  targets, or acting as a general shell is explicitly the job of a separate
  execution MCP (e.g. HexStrike), not BlackBook.

## Filesystem confinement

* MCP tools never expose arbitrary filesystem reads. The only files read are inside
  explicitly-configured knowledge directories.
* `blackbook/utils/paths.py` enforces this: `normalize_rel_path` rejects `..`
  traversal and absolute paths; `safe_join` verifies a resolved path stays inside
  its configured base. The PDF adapter applies these before reading any file.

## Input validation

* Tool inputs are validated with Pydantic schemas (bounded length, typed enums,
  integer ranges).
* FTS5 queries are normalized/quoted so user input cannot inject FTS5 syntax.
* Result sizes and per-document caps bound how much is returned in one call.

## Safe source handling

* Tarball extraction is hardened against zip-slip: every member is resolved and
  verified to stay inside the destination; device files and symlinks are skipped.
* Per-document size caps (`max_document_bytes`) and optional `max_files` bound
  ingestion cost.
* PDF pages that fail to parse yield empty text rather than crashing ingestion.

## Provenance / no fabricated citations

Every reference returned maps to an actually-indexed chunk and can be resolved to
its exact text via `knowledge_source`. BlackBook never fabricates URLs, page
numbers, titles, sections, or quotes — if it cannot be traced to an indexed chunk,
it is not returned.

## Logging & secrets

* Structured logs record queries, filters, candidate counts, and latency for
  debugging (`--verbose`), but BlackBook stores no credentials and never logs
  secrets (there are none in scope). Source material is public knowledge; local
  PDFs stay local.

## Dependencies

Dependencies are pinned to compatible ranges in `pyproject.toml`. The semantic
extra is optional so the default install surface stays small.
