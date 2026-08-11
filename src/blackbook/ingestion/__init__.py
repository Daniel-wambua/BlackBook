"""blackbook.ingestion subpackage."""

from blackbook.ingestion.base import IngestStats, ParsedDocument, SourceAdapter
from blackbook.ingestion.pipeline import IngestionPipeline, PipelineResult
from blackbook.ingestion.hacktricks import HackTricksAdapter
from blackbook.ingestion.zerodf import ZeroDFAdapter
from blackbook.ingestion.pdf import PDFAdapter

ADAPTER_REGISTRY = {
    "hacktricks": HackTricksAdapter,
    "0xdf": ZeroDFAdapter,
    "local_pdfs": PDFAdapter,
}


def adapter_for(source_config, raw_dir: str | None = None) -> SourceAdapter:
    """Instantiate the adapter for a source config.

    Falls back to matching by ``type`` when no id-specific adapter exists, so
    future sources can reuse adapters (e.g. a new "filesystem" source uses the
    PDF adapter shape).
    """
    cls = ADAPTER_REGISTRY.get(source_config.id)
    if cls is None:
        by_type = {"filesystem": PDFAdapter}
        cls = by_type.get(source_config.type)
    if cls is None:
        raise ValueError(f"no adapter registered for source {source_config.id!r}")
    return cls(source_config, raw_dir=raw_dir)


__all__ = [
    "SourceAdapter",
    "ParsedDocument",
    "IngestStats",
    "IngestionPipeline",
    "PipelineResult",
    "HackTricksAdapter",
    "ZeroDFAdapter",
    "PDFAdapter",
    "adapter_for",
]
