# Retrieval

Retrieval is hybrid by design but **lexical-first**.

## Pipeline

```
Query
  |
  +--> Query normalization       (tokens extracted, FTS5-safe)
  +--> Metadata filtering         (source, platform, category)
  +--> FTS5 retrieval             (BM25; always available)
  +--> Semantic retrieval         (optional; Phase 3)
  +--> Deduplication
  +--> Reranking                  (+ intent-mode bias; Phase 4)
  +--> Source diversity           (per-document cap)
  +--> Final results
```

> The knowledge graph is **not** part of the retrieval pipeline — retrieval never
> expands a query through graph edges. The graph is consumed one layer up, by the
> `knowledge_technique` and `knowledge_case_search` tools, to enrich results with
> evidence-linked neighbours/annotations. Retrieval works identically with an empty
> graph.

## Lexical (FTS5)

`LexicalRetriever` runs SQLite FTS5 BM25 over chunk text (plus title and deepest
section heading). Query text is normalized and quoted so arbitrary input cannot
inject FTS5 syntax. Results carry a BM25 score converted to a 0..1 relevance.

## Reranking

`reranker.rerank` combines signals rather than trusting any single score:

```
final = base_score
        * authority_factor            (official > trusted > user > unknown)
        * (1 + 0.25 * keyword_overlap  # query terms present in title/section
             + 0.15 (platform match)
             + 0.15 (category match)
             + 0.30 (intent-mode match))  # technique / case_similarity only
```

The **intent-mode bonus** is added only in `technique` or `case_similarity` mode,
to hits whose *shape* matches the intent — a technique/reference page for
`technique`, a writeup-category document for `case_similarity` (classified via the
shared `vocab` module, not the graph). It is a nudge on top of the base relevance,
never a filter: a strongly-matching off-shape hit can still outrank an on-shape one,
and every mode returns the same candidate set. This keeps the modes honest — they
reorder, they never hide material.

Then a **per-document cap** (default 2) enforces source diversity so one page cannot
dominate the result set with near-identical chunks. In addition, a hit is dropped when
it is a **cross-document near-duplicate** (word-shingle Jaccard ≥ 0.9) of a hit already
selected — so a PDF that restates HackTricks material doesn't return the same content
twice under different citations.

## Search modes

| Mode | Behaviour |
|------|-----------|
| `keyword` | FTS5 only |
| `semantic` | embeddings only (requires `[semantic]` extra + `embeddings.enabled`) |
| `hybrid` | FTS5 + semantic, merged & reranked (default) |
| `technique` | hybrid, reranked toward technique/reference pages (intent bonus) |
| `case_similarity` | hybrid, reranked toward hands-on writeups/cases (intent bonus) |

## Semantic (optional)

Embeddings are **local** (`sentence-transformers`, default `all-MiniLM-L6-v2`,
384-dim) and **optional** — no text ever leaves the machine, and no network
embedding API is used. When `embeddings.enabled` is false the system is purely
lexical and nothing about the caller changes.

**Storage.** Each chunk vector is L2-normalized at embed time and stored as a
little-endian `float32` blob in `chunk_embeddings`, keyed by `chunk_id` with the
model name and dimension recorded alongside it. The row has an
`ON DELETE CASCADE` foreign key to `chunks`, so re-chunking a document
automatically drops its stale vectors. The storage layer stays numpy-free (blobs
in, blobs out); only `embeddings.py` and `retrieval/semantic.py` import numpy and
the model.

**Index.** `SemanticRetriever` is a **brute-force flat cosine index**: it decodes
the stored blobs into one matrix and scores a query with a single matrix multiply
(`matrix @ q`). Because every vector is unit-normalized, cosine similarity *is* the
dot product. For a single-file knowledge base of this size a flat scan is a few
tens of milliseconds and needs no ANN dependency; the top-`k` are taken with
`argpartition` rather than a full sort. The decoded matrix is cached per
source-filter signature and invalidated when the embedding count changes (i.e.
after a re-embed), so a long-lived server never serves stale vectors. Semantic
hits are returned as the **same** `LexicalHit` type lexical search uses
(`bm25=0.0`, `score=max(0, cosine)`, `metadata.retrieval="semantic"`), so the
merge/rerank pipeline treats both backends uniformly. A vector whose chunk no
longer exists is skipped rather than fabricated into a hit.

**Generating vectors.** Ingestion embeds new chunks inline when an embedder is
available (see [ingestion.md](ingestion.md)). To (re)embed without re-ingesting:

```
blackbook embed [--source ID] [--reembed]
```

`--reembed` deletes existing vectors for the current model first (scoped to
`--source` when given). `blackbook doctor` reports embedding coverage
(`N/M embedded`) and warns when chunks are missing vectors.

**Graceful degradation.** The semantic backend is constructed lazily inside a
`try/except`. If the `[semantic]` extra is not installed, or the model cannot
load, hybrid search silently falls back to lexical-only — the caller sees fewer
results, never an error.

## Source filtering

`Settings.source_ids()` resolves a requested filter to concrete enabled sources.
`None` or `["all"]` searches every enabled source; a specific list searches only
those. Unknown IDs are dropped so a typo never widens a query beyond intent.

## Token efficiency

Results are concise by default (snippets, not documents). The intended flow is:

```
search  ->  candidate refs  ->  knowledge_source(ref)  ->  exact excerpt
```

A `detail` parameter (`brief` | `standard` | `deep`) controls excerpt length.
