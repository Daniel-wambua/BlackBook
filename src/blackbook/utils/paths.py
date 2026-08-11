"""Path safety helpers.

BlackBook never exposes arbitrary filesystem access through MCP. The only
filesystem reads are from explicitly-configured knowledge directories, and
these helpers enforce that boundary.
"""

from __future__ import annotations

from pathlib import Path


def normalize_rel_path(rel: str | Path) -> Path:
    """Normalize a relative path, rejecting anything that escapes upward.

    Raises ``ValueError`` on absolute paths or ``..`` traversal.
    """
    p = Path(str(rel))
    if p.is_absolute():
        raise ValueError(f"absolute path not allowed: {rel!r}")
    parts = [part for part in p.parts if part not in ("", ".")]
    if any(part == ".." for part in parts):
        raise ValueError(f"path traversal not allowed: {rel!r}")
    return Path(*parts) if parts else Path(".")


def is_within(child: Path, parent: Path) -> bool:
    """True if ``child`` resolves to a location inside ``parent``."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def safe_join(base: Path, rel: str | Path) -> Path:
    """Join ``rel`` onto ``base`` and verify the result stays inside ``base``.

    Returns the resolved absolute path. Raises ``ValueError`` if the result
    would escape ``base``.
    """
    base_resolved = base.resolve()
    candidate = (base_resolved / normalize_rel_path(rel)).resolve()
    if not is_within(candidate, base_resolved):
        raise ValueError(f"path escapes base directory: {rel!r}")
    return candidate
