"""End-to-end pipeline test using offline fixtures (no network).

Drives the IngestionPipeline with a fixture-backed HackTricks adapter and a
fixture-backed 0xdf adapter, then runs a real hybrid search and resolves a
citation back to its exact source text.
"""

from pathlib import Path

from blackbook.config import Settings, DatabaseConfig, SourceConfig
from blackbook.ingestion.hacktricks import HackTricksAdapter
from blackbook.ingestion.pipeline import IngestionPipeline
from blackbook.ingestion.zerodf import ZeroDFAdapter
from blackbook.knowledge.sources import get_chunk_excerpt
from blackbook.retrieval import HybridRetriever
from blackbook.storage import Database, Source

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _settings(tmp_path):
    return Settings(home=tmp_path, database=DatabaseConfig(path=str(tmp_path / "d.db")))


def test_full_pipeline_hacktricks(tmp_path):
    settings = _settings(tmp_path)
    db = Database(settings.db_path)
    with db.session():
        db.upsert_source(Source(source_id="hacktricks", name="HackTricks", authority="trusted"))

    cfg = SourceConfig(id="hacktricks", name="HackTricks", type="git", authority="trusted")
    adapter = HackTricksAdapter(cfg, raw_dir=str(tmp_path))
    adapter._extract_root = FIXTURES / "hacktricks"
    adapter.fetch = lambda force=False: None  # offline: skip network fetch

    pipeline = IngestionPipeline(db)
    result = pipeline.run(adapter)
    assert result.stats.parsed == 1
    assert result.stats.chunks_written > 0

    # Search it
    retriever = HybridRetriever(db, settings)
    results = retriever.search("kerberoasting SPN", source_ids=["hacktricks"], limit=5)
    assert results, "expected a hit in ingested HackTricks fixture"
    top = results[0]
    assert top.source_id == "hacktricks"

    # Resolve the citation to exact text
    ex = get_chunk_excerpt(db, top.chunk_id)
    assert ex is not None
    assert ex.text
    assert ex.chunk_id == top.chunk_id
    db.close()


def test_full_pipeline_zerodf(tmp_path):
    settings = _settings(tmp_path)
    db = Database(settings.db_path)
    with db.session():
        db.upsert_source(Source(source_id="0xdf", name="0xdf", authority="trusted"))

    cfg = SourceConfig(id="0xdf", name="0xdf", type="website", authority="trusted")
    adapter = ZeroDFAdapter(cfg, raw_dir=str(tmp_path / "raw"))

    # Stage the fixture page where the adapter expects cached pages.
    pages_dir = Path(adapter._workdir()) / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    (pages_dir / "htb-forest.html").write_text(
        (FIXTURES / "zerodf" / "htb-forest.html").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    adapter.fetch = lambda force=False: None  # offline: skip network fetch

    pipeline = IngestionPipeline(db)
    result = pipeline.run(adapter)
    assert result.stats.parsed == 1

    retriever = HybridRetriever(db, settings)
    results = retriever.search("kerberoast", source_ids=["0xdf"], limit=5)
    assert results
    assert results[0].source_id == "0xdf"
    db.close()


def test_reingest_is_incremental(tmp_path):
    settings = _settings(tmp_path)
    db = Database(settings.db_path)
    with db.session():
        db.upsert_source(Source(source_id="hacktricks", name="HackTricks", authority="trusted"))
    cfg = SourceConfig(id="hacktricks", name="HackTricks", type="git", authority="trusted")
    adapter = HackTricksAdapter(cfg, raw_dir=str(tmp_path))
    adapter._extract_root = FIXTURES / "hacktricks"
    adapter.fetch = lambda force=False: None  # offline: skip network fetch
    pipeline = IngestionPipeline(db)

    first = pipeline.run(adapter)
    second = pipeline.run(adapter)  # same content -> all unchanged
    assert first.stats.parsed == 1
    assert second.stats.parsed == 0
    assert second.stats.skipped_unchanged == 1
    db.close()
