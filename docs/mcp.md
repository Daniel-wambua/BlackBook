# MCP integration

BlackBook runs as a local MCP server over **stdio** — the transport every major
local MCP client (Claude Code, Cursor, VS Code) supports.

## Running the server

```bash
blackbook serve
# equivalently:
python -m blackbook.server
```

## Claude Code

```bash
claude mcp add blackbook -- blackbook serve
```

or in your MCP config:

```json
{
  "mcpServers": {
    "blackbook": { "command": "blackbook", "args": ["serve"] }
  }
}
```

## Cursor

`Settings → MCP → Add server`:

```json
{
  "mcpServers": {
    "blackbook": { "command": "blackbook", "args": ["serve"] }
  }
}
```

## VS Code (with an MCP-capable extension)

`.vscode/mcp.json` or your client config:

```json
{
  "servers": {
    "blackbook": { "command": "blackbook", "args": ["serve"] }
  }
}
```

If you use a virtual environment, point `command` at the interpreter inside it,
e.g. `"/home/you/venv/bin/blackbook"`.

## Tools

### `knowledge_search`

Source-grounded search across the indexed corpus.

**Input**

```json
{
  "query": "kerberoasting SPN",
  "sources": ["hacktricks", "0xdf"],
  "categories": ["active-directory"],
  "platform": "windows",
  "techniques": ["kerberoasting"],
  "mode": "hybrid",
  "limit": 8,
  "detail": "standard"
}
```

**Output** (each result carries a resolvable `ref`)

```json
{
  "query": "kerberoasting SPN",
  "mode": "hybrid",
  "sources_searched": ["0xdf", "hacktricks"],
  "count": 2,
  "results": [
    {
      "title": "Kerberoasting",
      "source": "hacktricks",
      "source_name": "HackTricks",
      "authority": "trusted",
      "relevance": 1.12,
      "snippet": "…request service tickets for SPN accounts…",
      "ref": {
        "chunk_id": 1234, "doc_id": 87, "title": "Kerberoasting",
        "source": "hacktricks", "url": "https://book.hacktricks.xyz/...",
        "page": null, "section_path": ["Active Directory", "Kerberos", "Kerberoasting"]
      }
    }
  ],
  "note": ""
}
```

### `knowledge_source`

Resolve a reference to the exact supporting excerpt.

**Input** — by chunk:

```json
{ "chunk_id": 1234 }
```

or by document + optional section filter:

```json
{ "source": "hacktricks", "document": "ad/kerberoasting.md", "section": "Tools", "max_excerpts": 5 }
```

**Output** — the exact indexed text with provenance (never paraphrased).

### `knowledge_technique`

A structured, source-grounded dossier for a technique. The knowledge graph
*enhances* the dossier — which sources document the technique, and the
tools/services/writeups it associates with — but never gates it: with an empty
graph the tool still returns real, cited excerpts from a technique-biased search.

**Input**

```json
{ "technique": "kerberoasting", "sources": ["hacktricks", "0xdf"], "limit": 6 }
```

**Output** — `resolved` (mapped to the controlled vocabulary) and `in_graph`
(a graph entity exists) are reported separately, so a caller can tell "unknown
term" from "known but graph not built yet". Every graph neighbour carries the
edge's `confidence`, an `inferred` flag, and the `evidence` document it was derived
from; `references` are always resolvable chunks.

```json
{
  "technique": "kerberoasting",
  "resolved": true,
  "in_graph": true,
  "documented_by": [
    { "name": "HackTricks", "entity_type": "source", "predicate": "documented_by",
      "confidence": 0.9, "inferred": false,
      "evidence": { "doc_id": 87, "title": "Kerberoasting", "source": "hacktricks" } }
  ],
  "related_tools": [ { "name": "impacket", "entity_type": "tool", "predicate": "uses", "confidence": 0.6, "inferred": true, "evidence": { "doc_id": 87 } } ],
  "related_services": [ { "name": "kerberos", "entity_type": "service", "predicate": "targets", "confidence": 0.6, "inferred": true, "evidence": { "doc_id": 87 } } ],
  "demonstrated_in": [],
  "references": [ { "title": "Kerberoasting", "source": "hacktricks", "ref": { "chunk_id": 1234, "doc_id": 87 } } ],
  "note": ""
}
```

### `knowledge_case_search`

Find hands-on writeups / case studies similar to a situation (uses
`case_similarity` ranking, favouring practical walkthrough material). When the
graph is built, each hit is annotated with the techniques that document
demonstrates; without a graph the results are the same, just unannotated.

**Input**

```json
{ "query": "crack a service account on an AD box", "platform": "windows",
  "techniques": ["kerberoasting"], "sources": ["0xdf"], "limit": 6 }
```

**Output**

```json
{
  "query": "crack a service account on an AD box",
  "count": 1,
  "results": [
    {
      "title": "HTB: Forest", "source": "0xdf", "source_name": "0xdf",
      "authority": "trusted", "relevance": 0.84,
      "snippet": "…GetUserSPNs.py to request tickets…",
      "ref": { "chunk_id": 5678, "doc_id": 142 },
      "techniques": ["as-rep roasting", "kerberoasting"]
    }
  ],
  "note": ""
}
```

### `knowledge_research`

Turn a free-text observation (a service banner, a foothold, a suspicious
finding) into a **source-grounded research packet**. Nothing in the packet is
synthesised: `signals` are controlled-vocabulary terms found literally in the
text, each `TechniqueBrief` reports whether the term resolved and which sources
the graph says document it (evidence-linked edges only), and `references` /
`related_cases` are real indexed chunks from a technique- and case-biased search.

**Input**

```json
{
  "observation": "Found a kerberos SPN on the AD domain; considering kerberoasting the service account with impacket.",
  "sources": ["hacktricks", "0xdf"],
  "platform": "windows",
  "techniques": ["kerberoast"],
  "limit": 6,
  "include_cases": true
}
```

**Output** — detected signals, per-technique briefs, and cited references/cases.
`techniques[].resolved` (mapped to the vocabulary) and `in_graph` (a graph
entity exists) are reported separately, exactly as in `knowledge_technique`.

```json
{
  "observation": "Found a kerberos SPN …",
  "signals": {
    "services": ["active directory", "kerberos"],
    "techniques": ["kerberoasting"],
    "tools": ["impacket"]
  },
  "techniques": [
    { "technique": "kerberoasting", "resolved": true, "in_graph": true,
      "documented_by": [ { "name": "HackTricks", "entity_type": "source",
        "predicate": "documented_by", "confidence": 0.9, "inferred": false,
        "evidence": { "doc_id": 87, "title": "Kerberoasting", "source": "hacktricks" } } ] }
  ],
  "references": [ { "title": "Kerberoasting", "source": "hacktricks", "ref": { "chunk_id": 1234, "doc_id": 87 } } ],
  "related_cases": [ { "title": "HTB: Forest", "source": "0xdf", "ref": { "chunk_id": 5678, "doc_id": 142 }, "techniques": ["as-rep roasting", "kerberoasting"] } ],
  "note": ""
}
```

### `knowledge_context`

Manage **local investigation state** — the one tool that writes, and only to the
local, user-authored case layer inside the same SQLite file. It never executes
anything, fetches anything, or touches a remote system, so it stays within the
read-only-w.r.t.-external-systems boundary. There is deliberately **no delete
action** — the tool cannot destroy state.

**Actions**

| `action` | Required fields | Effect |
|----------|-----------------|--------|
| `create` | `case` | Upsert a case by name (optional `target`, `platform`, `meta`). |
| `add` | `case`, `text` | Append an observation (`kind`: observation / finding / hypothesis / technique / note). |
| `update_observation` | `case`, `obs_id`, `status` | Set an observation's status (open / tested / confirmed / refuted / resolved). Refuses an `obs_id` from another case. |
| `get` | `case` | Return a case's full current state. |
| `list` | — | Summarise all cases (with observation counts). |

**Input** (add an observation)

```json
{ "action": "add", "case": "op-forest", "kind": "finding",
  "text": "svc-alfresco is AS-REP roastable" }
```

**Output** — the action, an `ok` flag, and the affected `CaseState` (or the
`cases` summary list for `list`). On a bad request `ok` is `false` and `note`
explains why; state is never partially mutated.

```json
{
  "action": "add",
  "ok": true,
  "case": {
    "case_id": 1, "name": "op-forest", "target": "10.10.10.161",
    "platform": "windows", "created_at": "…", "updated_at": "…",
    "observations": [
      { "obs_id": 1, "kind": "finding", "text": "svc-alfresco is AS-REP roastable",
        "status": "open", "created_at": "…" }
    ]
  },
  "cases": [],
  "note": ""
}
```
