"""Shared cybersecurity vocabulary for signal extraction and the knowledge graph.

This module is the single source of truth for the term lists used to spot
*services*, *techniques*, and *tools* in raw text. Both the 0xdf ingestion
adapter (which tags a writeup's inferred signals) and the Phase 4 knowledge
graph builder import these lists, so extraction stays consistent across the
pipeline.

The membership of the three lists is **behaviour-locked** by
``tests/unit/test_ingestion_zerodf.py`` — the exact same terms 0xdf used before
this module existed. Add terms deliberately; a change here changes both what a
writeup reports and what the graph extracts.

Nothing here executes anything or touches the network. It is pure text matching
over a controlled vocabulary — no fabricated entities can appear because a term
must literally occur in the source text to be extracted.
"""

from __future__ import annotations

# --- controlled vocabulary -------------------------------------------------
# Order is preserved for readability; extraction always returns a sorted set.

SERVICE_TERMS: list[str] = [
    "smb", "ldap", "kerberos", "winrm", "rdp", "mssql", "mysql", "postgres",
    "nginx", "apache", "iis", "ssh", "ftp", "dns", "smtp", "rpc", "nfs",
    "active directory", "adcs", "jenkins", "tomcat", "wordpress",
]

TECHNIQUE_TERMS: list[str] = [
    "kerberoasting", "as-rep roasting", "password spraying", "ntlm relay",
    "constrained delegation", "unconstrained delegation", "acl abuse",
    "privilege escalation", "sql injection", "lfi", "rfi", "ssrf", "ssti",
    "xss", "deserialization", "pass the hash", "dcsync", "golden ticket",
    "silver ticket", "rbcd", "shadow credentials", "certifried", "esc1",
]

TOOL_TERMS: list[str] = [
    "nmap", "ffuf", "gobuster", "feroxbuster", "burp", "impacket",
    "bloodhound", "responder", "hashcat", "john", "metasploit", "evil-winrm",
    "crackmapexec", "netexec", "rubeus", "mimikatz", "certipy", "wpscan",
    "sqlmap", "nikto", "hydra", "smbclient", "ldapsearch", "rpcclient",
]

#: Category tokens that mark a document as a hands-on writeup / case study
#: rather than reference documentation. Used to identify "Writeup" entities and
#: to bias ``case_similarity`` retrieval.
WRITEUP_CATEGORY_MARKERS: frozenset[str] = frozenset(
    {"htb", "hackthebox", "ctf", "writeup", "pg", "proving grounds", "oscp",
     "thm", "tryhackme", "vulnhub"}
)

#: Common surface-form aliases mapped onto a canonical technique term. Lets a
#: caller ask for "asreproast" and resolve to the indexed "as-rep roasting".
_TECHNIQUE_ALIASES: dict[str, str] = {
    "kerberoast": "kerberoasting",
    "asreproast": "as-rep roasting",
    "asrep roasting": "as-rep roasting",
    "as rep roasting": "as-rep roasting",
    "password spray": "password spraying",
    "pth": "pass the hash",
    "resource-based constrained delegation": "rbcd",
    "resource based constrained delegation": "rbcd",
    "sqli": "sql injection",
    "local file inclusion": "lfi",
    "remote file inclusion": "rfi",
    "server-side request forgery": "ssrf",
    "server-side template injection": "ssti",
    "cross-site scripting": "xss",
}

_TECHNIQUE_SET = frozenset(TECHNIQUE_TERMS)
_TOOL_SET = frozenset(TOOL_TERMS)
_SERVICE_SET = frozenset(SERVICE_TERMS)

#: Curated MITRE ATT&CK IDs for the vocabulary's technique terms. This is
#: hand-maintained reference metadata (not extracted from indexed sources);
#: techniques without a well-established mapping are deliberately absent so
#: the field can be trusted when present — an unmapped technique yields
#: ``None``, never a guessed ID.
TECHNIQUE_ATTACK_IDS: dict[str, str] = {
    "kerberoasting": "T1558.003",
    "as-rep roasting": "T1558.004",
    "golden ticket": "T1558.001",
    "silver ticket": "T1558.002",
    "pass the hash": "T1550.002",
    "dcsync": "T1003.006",
    "password spraying": "T1110.003",
    "ntlm relay": "T1557.001",
    "unconstrained delegation": "T1550.001",
    "constrained delegation": "T1550.001",
    "rbcd": "T1550.001",
    "shadow credentials": "T1550",
    "certifried": "T1649",
    "esc1": "T1649",
    "deserialization": "T1203",
    "privilege escalation": "T1068",
    "sql injection": "T1190",
    "ssrf": "T1190",
    "ssti": "T1190",
    "lfi": "T1190",
    "rfi": "T1190",
    "xss": "T1059.007",
}


def attack_id(technique: str) -> str | None:
    """MITRE ATT&CK ID for a technique term (or alias), or ``None``.

    Resolves through :func:`resolve_technique` first so aliases ("kerberoast",
    "pth") work. Never invents an ID: unmapped techniques return ``None``.
    """
    canonical = resolve_technique(technique)
    return TECHNIQUE_ATTACK_IDS.get(canonical) if canonical else None


def _found(terms: list[str], lowered: str) -> list[str]:
    """Return the sorted subset of ``terms`` that literally occur in ``lowered``."""
    return sorted({t for t in terms if t in lowered})


def extract_signals(text: str) -> tuple[list[str], list[str], list[str]]:
    """Extract ``(services, techniques, tools)`` mentioned in ``text``.

    This is the exact behaviour 0xdf's ``_extract_signals`` had — a substring
    scan over the controlled vocabulary, each result list sorted and de-duped.
    Matches are *inferred* signals; callers mark them as such.
    """
    lowered = text.lower()
    return (
        _found(SERVICE_TERMS, lowered),
        _found(TECHNIQUE_TERMS, lowered),
        _found(TOOL_TERMS, lowered),
    )


def extract_terms(text: str) -> dict[str, list[str]]:
    """Extract vocabulary hits from ``text`` as a ``{kind: [terms]}`` mapping.

    Convenience wrapper over :func:`extract_signals` for the graph builder,
    which prefers named access (``["technique"]``) over positional tuples.
    """
    services, techniques, tools = extract_signals(text)
    return {"service": services, "technique": techniques, "tool": tools}


def resolve_technique(name: str) -> str | None:
    """Map a free-form technique name onto a canonical vocabulary term.

    Returns the canonical term (e.g. ``"kerberoasting"``) when ``name`` matches
    a known technique or a known alias, else ``None``. Case/whitespace
    insensitive. Never invents a term that isn't in :data:`TECHNIQUE_TERMS`.
    """
    if not name:
        return None
    key = " ".join(name.lower().split())
    if key in _TECHNIQUE_SET:
        return key
    if key in _TECHNIQUE_ALIASES:
        return _TECHNIQUE_ALIASES[key]
    return None


def is_writeup_category(categories: list[str] | None) -> bool:
    """True when any category token marks the document as a writeup/case."""
    if not categories:
        return False
    return any(c.strip().lower() in WRITEUP_CATEGORY_MARKERS for c in categories)
