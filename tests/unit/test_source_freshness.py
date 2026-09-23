"""Per-source freshness: when a source was last pulled, and at which revision."""

from __future__ import annotations

import json

from blackbook.config import Settings, SourceConfig
from blackbook.ingestion.base import ParsedDocument, SourceAdapter
from blackbook.ingestion.pipeline import IngestionPipeline
from blackbook.mcp.schemas import KnowledgeSourceInput
from blackbook.mcp.tools import KnowledgeTools
from blackbook.retrieval.chunking import RawChunk
from blackbook.storage import Source


class _Adapter(SourceAdapter):
    """A pipeline adapter with a settable revision marker."""

    def __init__(self, config, documents, revision=None, raw_dir=None):
        super().__init__(config, raw_dir)
        self.documents = documents
        self.revision = revision
        self.fetched = 0

    def fetch(self, force=False):
        self.fetched += 1

    def iter_documents(self):
        yield from self.documents

    def version(self):
        return self.revision


def _document(external_id, text):
    return ParsedDocument(
        external_id=external_id,
        title=external_id,
        text=text,
        chunks=[RawChunk(text=text, section_path=[external_id], ordinal=0)],
    )


def _config(source_id: str) -> SourceConfig:
    return SourceConfig(id=source_id, name=source_id, type="git", url="https://example.test/r.git")


# -- storage ---------------------------------------------------------------


def test_mark_source_fetched_records_a_timestamp_and_revision(db):
    db.upsert_source(Source(source_id="s", name="S"))
    assert db.mark_source_fetched("s", "abc123") == 1
    row = db.get_source("s")
    assert row["version"] == "abc123"
    assert row["last_fetched"]


def test_mark_source_fetched_without_a_revision_keeps_the_known_one(db):
    """A source with no marker must not erase a commit recorded earlier."""
    db.upsert_source(Source(source_id="s", name="S"))
    db.mark_source_fetched("s", "abc123")
    db.mark_source_fetched("s")
    assert db.get_source("s")["version"] == "abc123"


def test_mark_source_fetched_on_an_unregistered_source_is_a_no_op(db):
    assert db.mark_source_fetched("never-registered", "abc123") == 0
    assert db.get_source("never-registered") is None


def test_mark_source_fetched_leaves_no_open_transaction(db):
    """A held write lock would block a concurrent CLI ingest."""
    db.upsert_source(Source(source_id="s", name="S"))
    db.mark_source_fetched("s", "abc123")
    assert db.conn.in_transaction is False


# -- pipeline --------------------------------------------------------------


def test_pipeline_records_the_source_it_fetched(db):
    db.upsert_source(Source(source_id="s", name="S"))
    pipeline = IngestionPipeline(db)
    adapter = _Adapter(_config("s"), [_document("a", "kerberoasting guidance")], "deadbeef")
    pipeline.run(adapter)

    row = db.get_source("s")
    assert row["last_fetched"]
    assert row["version"] == "deadbeef"


def test_pipeline_records_freshness_for_a_source_without_a_revision(db):
    db.upsert_source(Source(source_id="s", name="S"))
    pipeline = IngestionPipeline(db)
    pipeline.run(_Adapter(_config("s"), [_document("a", "text")]))
    row = db.get_source("s")
    assert row["last_fetched"]
    assert row["version"] is None


def test_a_failed_fetch_leaves_the_previous_stamp_in_place(db):
    """Freshness means last *successful* pull, not last attempt."""
    db.upsert_source(Source(source_id="s", name="S"))
    db.mark_source_fetched("s", "old-commit")

    class _Broken(_Adapter):
        def fetch(self, force=False):
            raise RuntimeError("tarball download failed")

    pipeline = IngestionPipeline(db)
    try:
        pipeline.run(_Broken(_config("s"), [], "new-commit"))
    except RuntimeError:
        pass

    row = db.get_source("s")
    assert row["version"] == "old-commit"


# -- knowledge_sources tool ------------------------------------------------


def test_knowledge_sources_reports_freshness(tmp_path, db):
    db.upsert_source(Source(source_id="hacktricks", name="HackTricks"))
    db.mark_source_fetched("hacktricks", "abc123def456")
    settings = Settings(sources=[SourceConfig(id="hacktricks", name="HackTricks", type="git")])
    out = KnowledgeTools(db, settings).knowledge_sources(KnowledgeSourceInput())

    status = out.sources[0]
    assert status.last_fetched
    assert status.version == "abc123def456"


def test_knowledge_sources_freshness_is_none_before_any_fetch(tmp_path, db):
    db.upsert_source(Source(source_id="hacktricks", name="HackTricks"))
    settings = Settings(sources=[SourceConfig(id="hacktricks", name="HackTricks", type="git")])
    status = KnowledgeTools(db, settings).knowledge_sources(KnowledgeSourceInput()).sources[0]
    assert status.last_fetched is None
    assert status.version is None


# -- CLI -------------------------------------------------------------------


def test_cli_sources_shows_freshness(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from blackbook.cli.main import app
    from blackbook.storage.database import Database

    monkeypatch.setenv("BLACKBOOK_HOME", str(tmp_path))
    config = tmp_path / "config.yaml"
    config.write_text(
        "home: {home}\nsources:\n  - id: hacktricks\n    name: HackTricks\n"
        "    type: git\n    url: https://example.test/r.git\n".format(home=tmp_path)
    )
    monkeypatch.setenv("BLACKBOOK_CONFIG", str(config))
    # The table has enough columns that a default 80-column capture truncates
    # the revision cell away.
    monkeypatch.setenv("COLUMNS", "200")

    db = Database(tmp_path / "data.db")
    db.upsert_source(Source(source_id="hacktricks", name="HackTricks"))
    db.mark_source_fetched("hacktricks", "abc123def456789")
    db.close()

    runner = CliRunner()
    listed = runner.invoke(app, ["sources"])
    assert listed.exit_code == 0
    assert "abc123de" in listed.stdout  # abbreviated revision
    assert "Rev" in listed.stdout

    as_json = json.loads(runner.invoke(app, ["sources", "--json"]).stdout)
    assert as_json[0]["version"] == "abc123def456789"
    assert as_json[0]["last_fetched"]
