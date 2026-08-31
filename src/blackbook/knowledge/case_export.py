"""Render an investigation case as a portable Markdown report.

Used by the ``knowledge_context`` tool's ``export`` action (which returns the
markdown in-band — the MCP server never writes files) and by the CLI's
``blackbook case export`` command (which writes it to disk).
"""

from __future__ import annotations

import re
from datetime import date

from blackbook.mcp.schemas import CaseState, ObservationItem
from blackbook.storage.database import Database


def build_case_state(db: Database, name: str) -> CaseState | None:
    """Compose a case's row and its observations into a :class:`CaseState`."""
    case = db.get_case(name)
    if case is None:
        return None
    observations = [
        ObservationItem(
            obs_id=int(o["obs_id"]),
            kind=o["kind"],
            text=o["text"],
            status=o["status"],
            created_at=o.get("created_at"),
        )
        for o in db.list_observations(int(case["case_id"]))
    ]
    return CaseState(
        case_id=int(case["case_id"]),
        name=case["name"],
        target=case.get("target") or "",
        platform=case.get("platform") or "",
        created_at=case.get("created_at"),
        updated_at=case.get("updated_at"),
        observations=observations,
    )


_STATUS_ICON = {
    "confirmed": "✅",
    "refuted": "❌",
    "resolved": "✔",
    "tested": "🧪",
    "open": "•",
}


def render_case_markdown(state: CaseState) -> str:
    """Render a case as a standalone Markdown document.

    The output is self-describing (metadata header, observations grouped in
    insertion order with status and timestamps) so it can be pasted into a
    report or committed to a repo without losing provenance of who recorded
    what, when.
    """
    lines: list[str] = [f"# Case: {state.name}", ""]

    meta_rows = [
        ("Target", state.target),
        ("Platform", state.platform),
        ("Created", state.created_at),
        ("Updated", state.updated_at),
        ("Observations", str(len(state.observations))),
    ]
    for label, value in meta_rows:
        if value:
            lines.append(f"- **{label}:** {value}")
    lines.append("")

    if not state.observations:
        lines.append("_No observations recorded._")
        return "\n".join(lines) + "\n"

    lines.append("## Timeline")
    lines.append("")
    for o in state.observations:
        icon = _STATUS_ICON.get(o.status, "•")
        when = (o.created_at or "").split(".")[0].replace("T", " ")
        lines.append(f"### {icon} [{o.status}] {o.kind} — {when}".rstrip(" —"))
        lines.append("")
        lines.append(o.text.strip())
        lines.append("")
    return "\n".join(lines) + "\n"


def export_filename(name: str) -> str:
    """A filesystem-safe filename stem for a case name."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip().lower()).strip("-")
    return f"{slug or 'case'}-{date.today().isoformat()}.md"
