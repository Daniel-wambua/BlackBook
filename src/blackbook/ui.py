"""Presentation layer for BlackBook: the startup banner and level-styled logs.

Pure cosmetics built on Rich (already a core dependency). Two rules shape it:

* The MCP server speaks JSON-RPC over **stdout**, so every byte of chrome here
  is written to **stderr** — it decorates the operator's terminal without ever
  corrupting the protocol stream a client reads from stdout.
* Nothing here changes behaviour. It only styles output already being produced
  (log records) or adds a one-shot banner. No new dependencies, never raises.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from rich.console import Console
from rich.text import Text

# The wordmark, assembled from block glyphs (ANSI-Shadow style). Six rows of
# equal width (74 cols); do not reflow — the alignment is load-bearing.
_BANNER: tuple[str, ...] = (
    "██████╗ ██╗      █████╗  ██████╗██╗  ██╗██████╗  ██████╗  ██████╗ ██╗  ██╗",
    "██╔══██╗██║     ██╔══██╗██╔════╝██║ ██╔╝██╔══██╗██╔═══██╗██╔═══██╗██║ ██╔╝",
    "██████╔╝██║     ███████║██║     █████╔╝ ██████╔╝██║   ██║██║   ██║█████╔╝ ",
    "██╔══██╗██║     ██╔══██║██║     ██╔═██╗ ██╔══██╗██║   ██║██║   ██║██╔═██╗ ",
    "██████╔╝███████╗██║  ██║╚██████╗██║  ██╗██████╔╝╚██████╔╝╚██████╔╝██║  ██╗",
    "╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═════╝  ╚═════╝  ╚═════╝ ╚═╝  ╚═╝",
)
# A cyan->indigo gradient, one colour per row. Intel-blue on purpose: it reads
# distinct from an execution MCP's red banner when the two run side by side.
_ROW_COLORS = ("#22d3ee", "#22b8f0", "#3a9bf0", "#5a82ef", "#7a6bec", "#8f5be8")
_BLOCK = "█"
_SHADOW_STYLE = "grey30"  # box/shadow glyphs get a dim drop-shadow feel


def render_banner() -> Text:
    """The coloured wordmark as a Rich Text: block glyphs gradient-lit, the
    shadow glyphs dimmed for a 3D effect."""
    text = Text()
    for row, color in zip(_BANNER, _ROW_COLORS):
        for ch in row:
            if ch == " ":
                text.append(" ")
            elif ch == _BLOCK:
                text.append(ch, style=color)
            else:
                text.append(ch, style=_SHADOW_STYLE)
        text.append("\n")
    return text


def _counts(db: Any) -> Optional[dict]:
    try:
        return db.counts()
    except Exception:
        return None


def print_banner(
    console: Console | None = None,
    *,
    db: Any = None,
    version: str | None = None,
    transport: str | None = None,
) -> None:
    """Print the banner to stderr (stdio-safe). Best-effort; never raises."""
    con = console or Console(stderr=True)
    if version is None:
        try:
            from blackbook import __version__ as version
        except Exception:
            version = "?"
    try:
        con.print(render_banner())
        con.print(
            "  Source-grounded cybersecurity knowledge & research MCP",
            style="bold white",
        )
        meta = Text("  v", style="dim")
        meta.append(str(version), style="cyan")
        if transport:
            meta.append(f"  ·  {transport}", style="dim")
        meta.append("  ·  read-only · no execution · every claim cited", style="dim")
        con.print(meta)

        c = _counts(db) if db is not None else None
        if c:
            con.print(
                f"  corpus  {c['sources']} sources · {c['documents']} docs · "
                f"{c['chunks']} chunks · {c['embeddings']} embeddings",
                style="grey62",
            )
            con.print(
                f"  graph   {c['entities']} entities · "
                f"{c['relationships']} relationships · {c['cases']} cases",
                style="grey62",
            )
        con.print()
    except Exception:
        # Chrome must never take the server or a command down.
        pass


# --------------------------------------------------------------------------- #
# Level-styled logging: the [*]/[!]/[-] "HexStrike-style" log look             #
# --------------------------------------------------------------------------- #
# level -> (symbol, style). There is no "success" log level, so [+] lives only
# in the success() helper below; log.error() renders as [-], etc.
_LEVEL: dict[int, tuple[str, str]] = {
    logging.DEBUG: ("·", "grey50"),
    logging.INFO: ("*", "cyan"),
    logging.WARNING: ("!", "yellow"),
    logging.ERROR: ("-", "bold red"),
    logging.CRITICAL: ("✗", "bold white on red"),
}


class RichLevelHandler(logging.Handler):
    """Render log records to a Rich console with a coloured ``[symbol]`` prefix.

    Writes to stderr by default, so it is safe under the stdio MCP transport.
    Warnings and errors colour the whole line; info/debug stay neutral so the
    terminal isn't a wall of colour.
    """

    def __init__(self, console: Console | None = None, *, show_name: bool = False):
        super().__init__()
        self.console = console or Console(stderr=True)
        self.show_name = show_name

    def emit(self, record: logging.LogRecord) -> None:
        try:
            sym, style = _LEVEL.get(record.levelno, ("*", "cyan"))
            line = Text()
            line.append("[", style="grey42")
            line.append(sym, style=style)
            line.append("] ", style="grey42")
            if self.show_name:
                line.append(f"{record.name}: ", style="grey42")
            body_style = style if record.levelno >= logging.WARNING else None
            line.append(record.getMessage(), style=body_style)
            self.console.print(line, soft_wrap=True)
            if record.exc_info:
                self.console.print(
                    logging.Formatter().formatException(record.exc_info),
                    style="grey42",
                )
        except Exception:
            self.handleError(record)


def configure_logging(
    verbose: bool = False,
    *,
    level: int | None = None,
    console: Console | None = None,
    show_name: bool | None = None,
) -> None:
    """Install the Rich level handler on the root logger (idempotent, stderr).

    Replaces any handler this function installed previously, so calling it from
    both the CLI and the server never double-prints. It only removes its own
    handlers — it leaves foreign handlers (e.g. pytest's caplog) untouched.

    ``level`` overrides the verbose-derived default (DEBUG when verbose, else
    INFO) — e.g. a benchmark can pass ``logging.WARNING`` to stay quiet.
    """
    if level is None:
        level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    root.handlers = [h for h in root.handlers if not getattr(h, "_blackbook", False)]
    handler = RichLevelHandler(
        console=console,
        show_name=verbose if show_name is None else show_name,
    )
    handler._blackbook = True  # type: ignore[attr-defined]
    handler.setLevel(level)
    root.addHandler(handler)
    root.setLevel(level)


# --------------------------------------------------------------------------- #
# Direct status helpers for command output (success/fail/warn/info)            #
# --------------------------------------------------------------------------- #
def _emit(console: Console | None, sym: str, style: str, msg: str) -> None:
    con = console or Console(stderr=True)
    line = Text()
    line.append("[", style="grey42")
    line.append(sym, style=style)
    line.append("] ", style="grey42")
    line.append(msg)
    con.print(line)


def success(msg: str, console: Console | None = None) -> None:
    _emit(console, "+", "bold green", msg)


def fail(msg: str, console: Console | None = None) -> None:
    _emit(console, "-", "bold red", msg)


def warn(msg: str, console: Console | None = None) -> None:
    _emit(console, "!", "yellow", msg)


def info(msg: str, console: Console | None = None) -> None:
    _emit(console, "*", "cyan", msg)
