"""MCP resources: read-only views a client can pull into context.

Tools are for asking questions. Resources are for state an agent should be able
to see without spending a tool call on it: which sources exist and how current
they are, what the corpus actually holds, what the controlled vocabulary
accepts, and which local investigation cases exist.

Two properties are deliberate:

* **Every view is computed on read.** Nothing is cached here, so a resource
  cannot describe a database other than the one it was just read from. The
  counts are the same counts the tools would report.
* **Nothing here writes or reaches the network.** These are projections of
  state that already exists locally. The one that renders a case to Markdown
  returns the text; it never creates a file.
"""

from __future__ import annotations

import json

from blackbook.storage.database import Database


def sources_json(db: Database, settings) -> str:
    """Configured sources with index counts and freshness, as JSON.

    Built from the same tool the MCP client would otherwise have to call, so
    the resource and ``knowledge_sources`` cannot disagree about the corpus.
    """
    from blackbook.mcp.schemas import KnowledgeSourceInput
    from blackbook.mcp.tools import KnowledgeTools

    out = KnowledgeTools(db, settings).knowledge_sources(KnowledgeSourceInput())
    return out.model_dump_json(indent=2)


def corpus_json(db: Database) -> str:
    """Corpus counts: what is indexed, and how much of it.

    Includes the query-log totals when the table exists, because "which
    phrasings returned nothing" is corpus intelligence an agent can act on
    (search differently) rather than merely operator telemetry.
    """
    payload: dict = dict(db.counts())
    try:
        payload["query_log"] = db.query_log_stats()
    except Exception:  # pragma: no cover - a pre-v4 database has no such table
        payload["query_log"] = None
    return json.dumps(payload, indent=2)


def vocabulary_json() -> str:
    """The controlled vocabulary: services, techniques, tools, and aliases.

    Worth exposing because these are the exact spellings the ``techniques``
    filter resolves against. An agent that reads this can ask for
    ``kerberoasting`` rather than guess at a phrasing the index will not match.
    The ATT&CK id is included where the technique maps to one, so the same
    identifier the tools report is visible up front.
    """
    from blackbook.knowledge import vocab

    return json.dumps(
        {
            "services": list(vocab.SERVICE_TERMS),
            "techniques": [
                {"term": t, "attack_id": vocab.attack_id(t)}
                for t in vocab.TECHNIQUE_TERMS
            ],
            "tools": list(vocab.TOOL_TERMS),
            "aliases": dict(sorted(vocab._TECHNIQUE_ALIASES.items())),
            "writeup_categories": sorted(vocab.WRITEUP_CATEGORY_MARKERS),
        },
        indent=2,
    )


def cases_json(db: Database) -> str:
    """Local investigation cases: names, targets, platforms, observation counts."""
    rows = [
        {
            "name": c["name"],
            "target": c.get("target") or "",
            "platform": c.get("platform") or "",
            "observations": int(c.get("observation_count") or 0),
            "updated_at": c.get("updated_at"),
        }
        for c in db.list_cases()
    ]
    return json.dumps({"count": len(rows), "cases": rows}, indent=2)


def case_markdown(db: Database, name: str) -> str:
    """One case rendered as portable Markdown.

    A missing case returns a short Markdown note rather than raising: a resource
    read has no useful place to surface an error, and the note tells the agent
    both that the case is absent and how to see which ones exist.
    """
    from blackbook.knowledge.case_export import build_case_state, render_case_markdown

    state = build_case_state(db, name)
    if state is None:
        return (
            f"# Case not found: {name}\n\n"
            "No local case by that name. Read `blackbook://cases` for the ones "
            "that exist, or create it with knowledge_context (action=create).\n"
        )
    return render_case_markdown(state)
