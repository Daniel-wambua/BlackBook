"""The local query log: what was asked, and what came back.

This is the one piece of BlackBook's state that is about *use* rather than
about the corpus. It exists to answer the questions the index cannot answer
about itself: which phrasings return nothing, which sources are searched and
never produce a hit, and how often a semantic request silently fell back to
lexical. A retrieval system with no record of its misses can only be tuned by
guesswork.

Two properties shape the implementation:

* **Logging never breaks a query.** A log entry is diagnostic; the search it
  describes is the product. Every failure here is swallowed after being
  reported at debug level, so a locked database or a full disk degrades the
  log, never the answer.
* **The log is bounded.** Entries are pruned to ``query_log.max_entries`` on
  insert, so it stays a recent-history record instead of an ever-growing one.

Callers pass measurements they already have; this module does the timing
arithmetic and none of the retrieval.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator

from blackbook.config import Settings
from blackbook.storage.database import Database
from blackbook.storage.models import QueryLogEntry

logger = logging.getLogger(__name__)


def record(
    db: Database,
    settings: Settings,
    *,
    tool: str,
    query: str,
    mode: str = "",
    sources: list[str] | None = None,
    result_count: int = 0,
    top_score: float | None = None,
    latency_ms: float = 0.0,
    backend: str = "",
    degraded: bool = False,
) -> None:
    """Append one entry to the query log. Never raises.

    Returns ``None`` on every path, including failure: the caller is mid-query
    and has an answer to return, so there is nothing useful for it to do with
    an exception from a diagnostic write.
    """
    if not settings.query_log.enabled:
        return
    try:
        db.log_query(
            QueryLogEntry(
                tool=tool,
                query=query,
                mode=mode,
                sources=list(sources or []),
                result_count=result_count,
                top_score=top_score,
                latency_ms=latency_ms,
                backend=backend,
                degraded=degraded,
            ),
            max_entries=settings.query_log.max_entries,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic path, see docstring
        logger.debug("query log write failed (query unaffected): %s", exc)


@contextmanager
def timed() -> Iterator[dict]:
    """Measure a block in milliseconds, for the ``latency_ms`` field."""
    start = time.perf_counter()
    out: dict = {"latency_ms": 0.0}
    try:
        yield out
    finally:
        out["latency_ms"] = round((time.perf_counter() - start) * 1000, 3)
