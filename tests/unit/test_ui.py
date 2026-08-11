"""Tests for the presentation layer (banner + level-styled logging).

The load-bearing guarantee here is **stdio safety**: the MCP server speaks
JSON-RPC over stdout, so none of this chrome may ever reach stdout. Everything
else is cosmetic, but the banner and log helpers must also never raise and must
degrade cleanly when corpus stats are unavailable.
"""

from __future__ import annotations

import io
import logging

import pytest
from rich.console import Console

from blackbook import ui


def _plain_console() -> tuple[Console, io.StringIO]:
    """A Rich console that writes plain (no-ANSI) text to a buffer."""
    buf = io.StringIO()
    con = Console(file=buf, force_terminal=False, no_color=True, width=200)
    return con, buf


class _BrokenDB:
    def counts(self):  # noqa: D401 - deliberately explodes
        raise RuntimeError("db is unavailable")


# --------------------------------------------------------------------------- #
# Banner                                                                       #
# --------------------------------------------------------------------------- #
def test_banner_rows_are_equal_width():
    widths = {len(row) for row in ui._BANNER}
    assert len(ui._BANNER) == 6
    assert len(widths) == 1, f"banner rows differ in width: {widths}"


def test_render_banner_nonempty_and_gridlike():
    text = ui.render_banner()
    plain = text.plain
    assert "█" in plain
    # Six glyph rows, each the same width -> a clean rectangle.
    lines = [ln for ln in plain.split("\n") if ln]
    assert len(lines) == 6
    assert len({len(ln) for ln in lines}) == 1


def test_print_banner_includes_tagline_and_stats():
    con, buf = _plain_console()
    ui.print_banner(console=con, db=None, version="9.9.9", transport="stdio")
    out = buf.getvalue()
    assert "Source-grounded cybersecurity knowledge & research MCP" in out
    assert "9.9.9" in out
    assert "stdio" in out
    assert "read-only" in out


def test_print_banner_renders_corpus_stats(db):
    con, buf = _plain_console()
    ui.print_banner(console=con, db=db, version="1.0")
    out = buf.getvalue()
    assert "corpus" in out and "graph" in out
    assert "sources" in out and "cases" in out


# --------------------------------------------------------------------------- #
# stdio safety: chrome must never reach stdout                                 #
# --------------------------------------------------------------------------- #
def test_print_banner_never_touches_stdout(capsys, db):
    """The default console targets stderr; stdout stays byte-for-byte empty."""
    ui.print_banner(db=db, version="1.0", transport="stdio")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "█" in captured.err  # the banner went to stderr


def test_status_helpers_default_to_stderr(capsys):
    ui.success("done")
    ui.fail("boom")
    ui.warn("careful")
    ui.info("fyi")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "done" in captured.err and "boom" in captured.err


# --------------------------------------------------------------------------- #
# Resilience: never raises, degrades on unavailable stats                      #
# --------------------------------------------------------------------------- #
def test_print_banner_survives_broken_db():
    con, buf = _plain_console()
    # Must not raise, and must still print the wordmark + tagline.
    ui.print_banner(console=con, db=_BrokenDB(), version="1.0")
    out = buf.getvalue()
    assert "Source-grounded" in out
    # No stats block, because counts() failed and was swallowed.
    assert "corpus" not in out


def test_counts_helper_returns_none_on_error():
    assert ui._counts(_BrokenDB()) is None


# --------------------------------------------------------------------------- #
# Level-styled logging                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "level,symbol",
    [
        (logging.DEBUG, "·"),
        (logging.INFO, "*"),
        (logging.WARNING, "!"),
        (logging.ERROR, "-"),
        (logging.CRITICAL, "✗"),
    ],
)
def test_handler_prefixes_level_symbol(level, symbol):
    con, buf = _plain_console()
    handler = ui.RichLevelHandler(console=con)
    handler.setLevel(logging.DEBUG)
    record = logging.LogRecord("t", level, __file__, 1, "hello world", None, None)
    handler.emit(record)
    out = buf.getvalue()
    assert f"[{symbol}]" in out
    assert "hello world" in out


def test_handler_renders_exception():
    con, buf = _plain_console()
    handler = ui.RichLevelHandler(console=con)
    try:
        raise ValueError("kaboom")
    except ValueError:
        import sys

        record = logging.LogRecord(
            "t", logging.ERROR, __file__, 1, "failed", None, sys.exc_info()
        )
    handler.emit(record)
    out = buf.getvalue()
    assert "failed" in out
    assert "ValueError" in out and "kaboom" in out


def test_handler_never_raises_on_bad_record():
    con, buf = _plain_console()
    handler = ui.RichLevelHandler(console=con)
    # %-style arg that can't format -> getMessage would raise; handler must not.
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "bad %d", ("x",), None)
    handler.emit(record)  # should be swallowed via handleError, not propagate


# --------------------------------------------------------------------------- #
# configure_logging idempotency + isolation                                    #
# --------------------------------------------------------------------------- #
def test_configure_logging_is_idempotent_and_leaves_foreign_handlers():
    root = logging.getLogger()
    original = list(root.handlers)
    foreign = logging.NullHandler()
    root.addHandler(foreign)
    try:
        ui.configure_logging()
        ui.configure_logging(verbose=True)  # second call must not stack
        ours = [h for h in root.handlers if getattr(h, "_blackbook", False)]
        assert len(ours) == 1
        assert foreign in root.handlers  # foreign handler untouched
    finally:
        root.handlers = original


def test_configure_logging_level_override():
    root = logging.getLogger()
    original = list(root.handlers)
    try:
        ui.configure_logging(verbose=False, level=logging.WARNING)
        ours = [h for h in root.handlers if getattr(h, "_blackbook", False)]
        assert ours and ours[0].level == logging.WARNING
    finally:
        root.handlers = original
        root.setLevel(logging.WARNING)
