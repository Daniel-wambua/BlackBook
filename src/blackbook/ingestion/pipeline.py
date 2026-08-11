"""Ingestion pipeline.

Drives adapters and persists their output. Responsibilities:

* content-hash change detection (skip unchanged documents)
* chunking (uses adapter-provided chunks when present, else generic)
* persistence of documents + chunks
* near-duplicate chunk detection via content hash
* reporting

The pipeline never fetches or parses itself — that is the adapter's job.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from blackbook.ingestion.base import IngestStats, ParsedDocument, SourceAdapter, content_hash
from blackbook.retrieval.chunking import RawChunk, chunk_markdown, estimate_tokens
from blackbook.retrieval.dedup import normalized_hash
from blackbook.storage.database import Database
from blackbook.storage.models import Chunk, Document

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    source_id: str
    stats: IngestStats = field(default_factory=IngestStats)
    embedded: int = 0


class IngestionPipeline:
    """Runs a SourceAdapter and writes results to the Database.

    If an ``embedder`` is supplied (semantic search enabled and the extra is
    installed), chunks that lack a current-model vector are embedded at the end
    of each run — scoped to the ingested source, batched for throughput. When no
    embedder is given, ingestion is lexical-only and never touches the model.
    """

    def __init__(self, db: Database, embedder=None):
        self.db = db
        self.embedder = embedder

    def run(self, adapter: SourceAdapter, force: bool = False) -> PipelineResult:
        result = PipelineResult(source_id=adapter.source_id)
        stats = result.stats

        log.info("[%s] fetching source material", adapter.source_id)
        adapter.fetch(force=force)

        for parsed in adapter.iter_documents():
            stats.discovered += 1
            try:
                wrote = self._ingest_document(adapter, parsed)
                if wrote == "unchanged":
                    stats.skipped_unchanged += 1
                else:
                    stats.parsed += 1
                    stats.chunks_written += wrote if isinstance(wrote, int) else 0
            except Exception as e:
                stats.errors += 1
                msg = f"{parsed.external_id}: {e}"
                stats.error_messages.append(msg)
                log.warning("[%s] ingest error on %s", adapter.source_id, msg)

        # Semantic layer: embed any chunks in this source that lack a vector for
        # the current model. Done once per run so encoding batches are full.
        if self.embedder is not None:
            from blackbook.embeddings import embed_missing_chunks

            result.embedded = embed_missing_chunks(
                self.db, self.embedder, source_ids=[adapter.source_id]
            )
            if result.embedded:
                log.info(
                    "[%s] embedded %d new chunks", adapter.source_id, result.embedded
                )
        return result

    # -- internals ---------------------------------------------------------

    def _ingest_document(
        self, adapter: SourceAdapter, parsed: ParsedDocument
    ) -> int | str:
        """Persist one parsed document.

        Returns the number of chunks written, or the string ``"unchanged"``
        if the document's content hash matches what's already stored.
        """
        doc_hash = content_hash(parsed.text)
        existing = self.db.get_document_by_external(adapter.source_id, parsed.external_id)
        if existing and existing.get("content_hash") == doc_hash:
            return "unchanged"

        # Prepare chunks: adapter-provided, else generic markdown chunking.
        raw_chunks: list[RawChunk] = parsed.chunks or chunk_markdown(
            parsed.text, title_path=parsed.categories + [parsed.title]
        )

        with self.db.session():
            doc = Document(
                source_id=adapter.source_id,
                external_id=parsed.external_id,
                title=parsed.title,
                url=parsed.url,
                path=parsed.path,
                content_hash=doc_hash,
                metadata=parsed.metadata,
                categories=parsed.categories,
            )
            doc_id = self.db.upsert_document(doc)

            chunk_rows = []
            # Dedup within a document on *normalized* hash so re-formatted
            # copies (whitespace/casing/punctuation differences) also collapse.
            seen_hashes: set[str] = set()
            for rc in raw_chunks:
                chash = normalized_hash(rc.text)
                if chash in seen_hashes:
                    continue
                seen_hashes.add(chash)
                chunk_rows.append(
                    Chunk(
                        doc_id=doc_id,
                        ordinal=rc.ordinal,
                        text=rc.text,
                        section_path=rc.section_path,
                        page=rc.page,
                        token_estimate=estimate_tokens(rc.text),
                        content_hash=chash,
                        metadata={"kind": rc.kind},
                    )
                )
            self.db.replace_chunks(doc_id, chunk_rows)
        return len(chunk_rows)
