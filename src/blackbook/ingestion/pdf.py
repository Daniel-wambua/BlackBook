"""Local PDF ingestion adapter.

User-supplied PDFs are a first-class source but are *not* assumed
authoritative — the source config carries ``authority`` and it is surfaced in
citations. Extraction preserves the document title and page boundaries, and
detects headings (by font size / numbering) and code blocks (by monospaced
font) where the signal is strong enough. Everything detected structurally is
marked ``inferred`` in the metadata.

PDFs are chunked per-page on paragraph boundaries with heading breadcrumbs;
the exact page number and section are recorded on every chunk for citation.

Only files inside the configured ``directory`` are read, and path traversal
out of that directory is rejected.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from pypdf import PdfReader

from blackbook.config import SourceConfig
from blackbook.ingestion.base import ParsedDocument, SourceAdapter
from blackbook.ingestion.pdf_structure import PageContent, analyze_page
from blackbook.retrieval.chunking import chunk_structured_pages
from blackbook.utils.paths import is_within

log = logging.getLogger(__name__)


class PDFAdapter(SourceAdapter):
    """Ingests local PDF files from a configured directory."""

    source_id = "local_pdfs"

    def __init__(self, config: SourceConfig, raw_dir: str | None = None):
        super().__init__(config, raw_dir)
        self.directory = Path(config.directory) if config.directory else None

    # -- fetching ----------------------------------------------------------

    def fetch(self, force: bool = False) -> None:
        # PDFs are already local; nothing to fetch. Validate the directory.
        if not self.directory or not self.directory.is_dir():
            log.warning("local_pdfs directory not available: %s", self.directory)

    # -- parsing -----------------------------------------------------------

    def iter_documents(self) -> Iterator[ParsedDocument]:
        if not self.directory or not self.directory.is_dir():
            return
        base = self.directory.resolve()
        glob = self.config.include_glob or "**/*.pdf"
        max_files = self.config.max_files
        count = 0
        for path in sorted(base.glob(glob)):
            if max_files is not None and count >= max_files:
                break
            if not path.is_file():
                continue
            # Enforce the directory boundary.
            if not is_within(path, base):
                log.warning("skipping out-of-directory pdf: %s", path)
                continue
            if path.suffix.lower() != ".pdf":
                continue
            if path.stat().st_size > self.config.max_document_bytes:
                log.warning("skipping oversized pdf: %s", path)
                continue
            count += 1
            try:
                doc = self._parse_pdf(base, path)
                if doc is not None:
                    yield doc
            except Exception as e:
                log.warning("failed to parse pdf %s: %s", path, e)

    def _parse_pdf(self, base: Path, path: Path) -> ParsedDocument | None:
        reader = PdfReader(str(path))
        meta = self._extract_metadata(reader, path)

        # Structural analysis per page (font-aware heading/code detection).
        pages: list[PageContent] = []
        for i, page in enumerate(reader.pages, start=1):
            pages.append(analyze_page(page, i))

        if not any(p.text.strip() for p in pages):
            return None

        title = meta["title"]
        all_headings = [h for p in pages for h in p.headings]
        n_code = sum(len(p.code_blocks) for p in pages)
        rel = path.relative_to(base)

        chunks = chunk_structured_pages(pages, title_path=[title])
        full_text = "\n\n".join(p.text for p in pages)
        return ParsedDocument(
            external_id=str(rel),
            title=title,
            url=None,
            path=str(path),
            categories=["local_pdf"],
            text=full_text,
            metadata={
                "rel_path": str(rel),
                "format": "pdf",
                "page_count": len(pages),
                "author": meta["author"],
                "subject": meta["subject"],
                "headings": all_headings[:100],
                "code_block_count": n_code,
                "metadata_inferred": {
                    "title": meta["title_inferred"],
                    "headings": True,
                    "code_blocks": True,
                },
            },
            chunks=chunks,
        )

    # Values some PDF generators stamp by default; treated as absent.
    _BOILERPLATE = {"untitled", "anonymous", "unknown", "none", ""}

    @classmethod
    def _clean(cls, val) -> str | None:
        if val is None:
            return None
        s = str(val).strip()
        return None if s.lower() in cls._BOILERPLATE else s

    @classmethod
    def _extract_metadata(cls, reader: PdfReader, path: Path) -> dict:
        meta = reader.metadata
        title = author = subject = None
        if meta:
            title = cls._clean(getattr(meta, "title", None))
            author = cls._clean(getattr(meta, "author", None))
            subject = cls._clean(getattr(meta, "subject", None))
        title_inferred = title is None
        if title is None:
            title = path.stem
        return {
            "title": title,
            "author": author,
            "subject": subject,
            "title_inferred": title_inferred,
        }
