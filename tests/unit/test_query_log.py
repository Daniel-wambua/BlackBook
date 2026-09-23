"""Tests for the local query log (table, recording, and the CLI view)."""

from __future__ import annotations

from blackbook.config import DatabaseConfig, QueryLogConfig, Settings
from blackbook.mcp.schemas import SearchInput
from blackbook.mcp.tools import KnowledgeTools
from blackbook.storage import QueryLogEntry
from blackbook.storage.database import Database


def make_settings(tmp_path, **kw) -> Settings:
    return Settings(
        home=tmp_path,
        database=DatabaseConfig(path=str(tmp_path / "d.db")),
        **kw,
    )


# -- storage layer ---------------------------------------------------------


def test_log_query_round_trips_every_field(db):
    db.log_query(
        QueryLogEntry(
            tool="knowledge_search",
            query="kerberoasting",
            mode="hybrid",
            sources=["hacktricks", "0xdf"],
            result_count=3,
            top_score=0.9123,
            latency_ms=4.5,
            backend="lexical+semantic",
            degraded=True,
        )
    )
    rows = db.list_queries()
    assert len(rows) == 1
    row = rows[0]
    assert row["tool"] == "knowledge_search"
    assert row["query"] == "kerberoasting"
    assert row["mode"] == "hybrid"
    # Sources survive as JSON, not as a Python repr that would need eval().
    assert row["sources"] == '["hacktricks", "0xdf"]'
    assert row["result_count"] == 3
    assert row["top_score"] == 0.9123
    assert row["latency_ms"] == 4.5
    assert row["backend"] == "lexical+semantic"
    assert row["degraded"] == 1
    assert row["created_at"]


def test_list_queries_is_newest_first_and_honours_the_limit(db):
    for i in range(5):
        db.log_query(QueryLogEntry(tool="t", query=f"q{i}"))
    rows = db.list_queries(limit=3)
    assert [r["query"] for r in rows] == ["q4", "q3", "q2"]


def test_list_queries_empty_only_filters_to_misses(db):
    db.log_query(QueryLogEntry(tool="t", query="hit", result_count=4))
    db.log_query(QueryLogEntry(tool="t", query="miss", result_count=0))
    assert [r["query"] for r in db.list_queries(empty_only=True)] == ["miss"]
    assert len(db.list_queries(empty_only=False)) == 2


def test_prune_keeps_exactly_the_newest_entries(db):
    for i in range(12):
        db.log_query(QueryLogEntry(tool="t", query=f"q{i}"), max_entries=5)
    rows = db.list_queries(limit=50)
    # Exactly five, not one fewer: the cut row is kept, not deleted with the rest.
    assert len(rows) == 5
    assert [r["query"] for r in rows] == ["q11", "q10", "q9", "q8", "q7"]


def test_prune_leaves_a_log_under_the_bound_untouched(db):
    for i in range(3):
        db.log_query(QueryLogEntry(tool="t", query=f"q{i}"), max_entries=5)
    assert len(db.list_queries(limit=50)) == 3


def test_query_log_stats_separate_misses_from_hits(db):
    db.log_query(QueryLogEntry(tool="t", query="a", result_count=2, latency_ms=4.0))
    db.log_query(QueryLogEntry(tool="t", query="b", result_count=0, latency_ms=6.0))
    db.log_query(QueryLogEntry(tool="t", query="c", result_count=0, latency_ms=8.0))
    stats = db.query_log_stats()
    assert stats["total"] == 3
    assert stats["empty"] == 2
    assert stats["empty_rate"] == 2 / 3
    assert stats["avg_latency_ms"] == 6.0
    assert stats["first_at"] <= stats["last_at"]


def test_query_log_stats_on_an_empty_log_is_all_zeros(db):
    stats = db.query_log_stats()
    assert stats["total"] == 0
    # 0.0 rather than a division by zero, so the CLI can print it unguarded.
    assert stats["empty_rate"] == 0.0
    assert stats["first_at"] is None


def test_clear_query_log_removes_everything(db):
    for i in range(4):
        db.log_query(QueryLogEntry(tool="t", query=f"q{i}"))
    assert db.clear_query_log() == 4
    assert db.query_log_stats()["total"] == 0


def test_log_query_leaves_no_open_transaction(db):
    """A held write lock would block a concurrent CLI ingest.

    The insert must commit, not sit in an implicit transaction for the life of
    the connection. ``in_transaction`` is the direct check for that.
    """
    db.log_query(QueryLogEntry(tool="t", query="q"))
    assert db.conn.in_transaction is False


# -- recording from the search tool ---------------------------------------


def test_knowledge_search_records_the_query_it_answered(tmp_path, seeded_db):
    tools = KnowledgeTools(seeded_db, make_settings(tmp_path))
    out = tools.knowledge_search(SearchInput(query="kerberoasting"))

    rows = seeded_db.list_queries()
    assert len(rows) == 1
    row = rows[0]
    assert row["tool"] == "knowledge_search"
    assert row["query"] == "kerberoasting"
    assert row["mode"] == "hybrid"
    # The logged count and score describe the answer the caller got, not an
    # intermediate stage of it.
    assert row["result_count"] == out.count
    assert row["top_score"] == out.results[0].relevance
    assert row["latency_ms"] >= 0
    assert row["degraded"] == (1 if out.degraded else 0)


def test_a_search_that_finds_nothing_is_logged_as_a_miss(tmp_path, seeded_db):
    tools = KnowledgeTools(seeded_db, make_settings(tmp_path))
    out = tools.knowledge_search(SearchInput(query="zzzqqq nonexistent phrase"))
    assert out.count == 0

    rows = seeded_db.list_queries(empty_only=True)
    assert len(rows) == 1
    # A miss has no best score; NULL says that, where 0.0 would claim a scored
    # result that ranked last.
    assert rows[0]["top_score"] is None


def test_logging_can_be_disabled_entirely(tmp_path, seeded_db):
    settings = make_settings(tmp_path, query_log=QueryLogConfig(enabled=False))
    tools = KnowledgeTools(seeded_db, settings)
    tools.knowledge_search(SearchInput(query="kerberoasting"))
    assert seeded_db.query_log_stats()["total"] == 0


def test_a_failing_log_write_does_not_break_the_search(tmp_path, seeded_db, monkeypatch):
    """The log is diagnostic; the answer is the product."""

    def boom(*a, **kw):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(seeded_db, "log_query", boom)
    tools = KnowledgeTools(seeded_db, make_settings(tmp_path))
    out = tools.knowledge_search(SearchInput(query="kerberoasting"))
    assert out.count >= 1


def test_log_write_uses_the_configured_entry_bound(tmp_path, seeded_db):
    settings = make_settings(tmp_path, query_log=QueryLogConfig(max_entries=3))
    tools = KnowledgeTools(seeded_db, settings)
    for i in range(6):
        tools.knowledge_search(SearchInput(query=f"kerberoasting {i}"))
    assert len(seeded_db.list_queries(limit=50)) == 3


# -- CLI ------------------------------------------------------------------


def test_cli_queries_reports_an_empty_log(tmp_path, monkeypatch, capsys):
    from typer.testing import CliRunner

    from blackbook.cli.main import app

    monkeypatch.setenv("BLACKBOOK_HOME", str(tmp_path))
    result = CliRunner().invoke(app, ["queries"])
    assert result.exit_code == 0
    assert "empty" in result.stdout.lower()


def test_cli_queries_lists_a_logged_entry_and_can_clear_it(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from blackbook.cli.main import app

    monkeypatch.setenv("BLACKBOOK_HOME", str(tmp_path))
    db = Database(tmp_path / "data.db")
    db.log_query(QueryLogEntry(tool="cli:search", query="kerberoasting", result_count=3))
    db.close()

    runner = CliRunner()
    listed = runner.invoke(app, ["queries"])
    assert listed.exit_code == 0
    assert "kerberoasting" in listed.stdout

    as_json = runner.invoke(app, ["queries", "--json"])
    assert as_json.exit_code == 0
    assert '"total": 1' in as_json.stdout

    cleared = runner.invoke(app, ["queries", "--clear"])
    assert cleared.exit_code == 0
    assert "Cleared 1" in cleared.stdout
    assert "empty" in runner.invoke(app, ["queries"]).stdout.lower()
