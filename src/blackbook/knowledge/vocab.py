"""Shared cybersecurity vocabulary for signal extraction and the knowledge graph.

This module is the single source of truth for the term lists used to spot
*services*, *techniques*, and *tools* in raw text. Both the 0xdf ingestion
adapter (which tags a writeup's inferred signals) and the Phase 4 knowledge
graph builder import these lists, so extraction stays consistent across the
pipeline.

The three lists began as the exact terms 0xdf used before this module existed,
and that core membership is pinned by ``tests/unit/test_ingestion_zerodf.py``
and ``tests/unit/test_graph.py``, which assert the *counts* and the extracted
signals rather than leaving them to drift. Add terms deliberately: a change here
changes both what a writeup reports and what the graph extracts.

``TECHNIQUE_TERMS`` has since outgrown that core. The original list was written
for Active Directory boxes, so a writeup tagging pass built on it alone is
silent on nearly everything a web or API engagement finds — an observation like
"GraphQL introspection enabled, IDOR on the user object, no rate limit on
password reset" extracted zero techniques. The web and bug-bounty terms added
below close that gap. They are grouped and commented separately so it stays
visible which half of the list is the frozen 0xdf core and which half is the
extension.

Nothing here executes anything or touches the network. It is pure text matching
over a controlled vocabulary — no fabricated entities can appear because a term
must literally occur in the source text to be extracted.
"""

from __future__ import annotations

import re

# --- controlled vocabulary -------------------------------------------------
# Order is preserved for readability; extraction always returns a sorted set.

SERVICE_TERMS: list[str] = [
    "smb", "ldap", "kerberos", "winrm", "rdp", "mssql", "mysql", "postgres",
    "nginx", "apache", "iis", "ssh", "ftp", "dns", "smtp", "rpc", "nfs",
    "active directory", "adcs", "jenkins", "tomcat", "wordpress",
]

TECHNIQUE_TERMS: list[str] = [
    # -- Active Directory / Kerberos: the original 0xdf core ---------------
    "kerberoasting", "as-rep roasting", "password spraying", "ntlm relay",
    "constrained delegation", "unconstrained delegation", "acl abuse",
    "privilege escalation", "pass the hash", "dcsync", "golden ticket",
    "silver ticket", "rbcd", "shadow credentials", "certifried", "esc1",
    # -- injection and server-side flaws -----------------------------------
    "sql injection", "nosql injection", "ldap injection", "command injection",
    "lfi", "rfi", "ssrf", "ssti", "xss", "xxe", "deserialization",
    # -- browser and client-side -------------------------------------------
    "csrf", "clickjacking", "open redirect", "prototype pollution",
    # -- access control and identity ---------------------------------------
    "idor", "insecure direct object reference", "bola",
    "broken object level authorization", "authentication bypass",
    "mass assignment",
    # -- web surface, protocols and file handling --------------------------
    "jwt", "oauth", "saml", "graphql", "host header injection",
    "subdomain takeover", "cache poisoning", "request smuggling",
    "path traversal", "directory traversal", "race condition",
    "arbitrary file upload", "rate limit",
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
#:
#: Two kinds of row live here, and they are not equally strong claims. A
#: *direct* row names the ATT&CK technique that describes the same behaviour;
#: the technique's own name is quoted beside it so the row can be checked
#: against the corpus. An *approximate* row is the nearest ATT&CK has for a
#: whole vulnerability class, which ATT&CK models as an outcome ("Exploit
#: Public-Facing Application") rather than as the class itself. The split is
#: kept visible because collapsing it would make a weak claim look like a
#: strong one.
#:
#: The quoted names are the ones the indexed corpus holds, which is the STIX
#: leaf name: a sub-technique is stored as "JavaScript", not as
#: "Command and Scripting Interpreter: JavaScript". attack.mitre.org prints the
#: longer "Parent: Child" form, and both resolve through
#: :class:`blackbook.knowledge.attack_index.AttackIndex`, which derives the
#: path from the indexed parent. The leaf is what is quoted here because the
#: leaf is what a grep of the corpus finds.
#:
#: Four rows were corrected after checking them against the indexed corpus.
#: ``unconstrained delegation``, ``constrained delegation`` and ``rbcd`` all
#: pointed at T1550.001, whose real name is "Application Access Token" — an
#: application and cloud token theft technique with nothing to do with Kerberos
#: delegation. ATT&CK has no Kerberos-delegation technique at all: searching
#: every indexed technique name for "delegat" returns only the Kerberos
#: *ticket* techniques plus T1098.002 "Additional Email Delegate Permissions".
#: The honest mapping for all three is therefore the class they sit in, T1558
#: "Steal or Forge Kerberos Tickets". ``shadow credentials`` pointed at the bare
#: superclass T1550 "Use Alternate Authentication Material"; the attack it names
#: is a write to the target object's key credentials, which is T1098 "Account
#: Manipulation".
#:
#: The web terms added alongside them are mostly left out on purpose. Terms
#: ATT&CK genuinely models are mapped (``command injection`` -> T1059, and the
#: injection family -> T1190); IDOR, CSRF, JWT, OAuth, GraphQL, race conditions,
#: request smuggling, subdomain takeover and the rest are absent because ATT&CK
#: has no technique that means them. Deriving an ID for those from the indexed
#: ATT&CK source is :class:`blackbook.knowledge.attack_index.AttackIndex`, which
#: resolves by exact technique name and never guesses either.
TECHNIQUE_ATTACK_IDS: dict[str, str] = {
    # -- direct: ATT&CK names the same behaviour ---------------------------
    "kerberoasting": "T1558.003",          # Kerberoasting
    "as-rep roasting": "T1558.004",        # AS-REP Roasting
    "golden ticket": "T1558.001",          # Golden Ticket
    "silver ticket": "T1558.002",          # Silver Ticket
    # No delegation technique exists; these map to the class that covers them.
    "unconstrained delegation": "T1558",   # Steal or Forge Kerberos Tickets
    "constrained delegation": "T1558",     # Steal or Forge Kerberos Tickets
    "rbcd": "T1558",                       # Steal or Forge Kerberos Tickets
    "pass the hash": "T1550.002",          # Pass the Hash
    "dcsync": "T1003.006",                 # DCSync
    "password spraying": "T1110.003",      # Password Spraying
    "ntlm relay": "T1557.001",             # Name Resolution Poisoning and SMB Relay
    "shadow credentials": "T1098",         # Account Manipulation
    "certifried": "T1649",                 # Steal or Forge Authentication Certificates
    "esc1": "T1649",                       # Steal or Forge Authentication Certificates
    "command injection": "T1059",          # Command and Scripting Interpreter
    # -- approximate: nearest ATT&CK has for a vulnerability class ----------
    "deserialization": "T1203",            # Exploitation for Client Execution
    "privilege escalation": "T1068",       # Exploitation for Privilege Escalation
    "sql injection": "T1190",              # Exploit Public-Facing Application
    "nosql injection": "T1190",            # Exploit Public-Facing Application
    "ldap injection": "T1190",             # Exploit Public-Facing Application
    "ssrf": "T1190",                       # Exploit Public-Facing Application
    "ssti": "T1190",                       # Exploit Public-Facing Application
    "lfi": "T1190",                        # Exploit Public-Facing Application
    "rfi": "T1190",                        # Exploit Public-Facing Application
    "xxe": "T1190",                        # Exploit Public-Facing Application
    "xss": "T1059.007",                    # JavaScript
}


def attack_id(technique: str) -> str | None:
    """MITRE ATT&CK ID for a technique term (or alias), or ``None``.

    Resolves through :func:`resolve_technique` first so aliases ("kerberoast",
    "pth") work. Never invents an ID: unmapped techniques return ``None``.
    """
    canonical = resolve_technique(technique)
    return TECHNIQUE_ATTACK_IDS.get(canonical) if canonical else None


#: Terms at or below this length are matched with a leading word boundary.
_BOUNDARY_MIN_LEN = 4

#: Boundary-anchored matchers for the technique terms short enough to hide
#: inside an unrelated word. A bare substring scan is what makes a four-letter
#: term match by accident — "bola" inside "ebola", or any short token buried in
#: a base64 blob, which writeups are full of. Requiring a non-alphanumeric
#: character (or the start of the text) before the term removes those without
#: touching how longer phrases match.
#:
#: The anchor is deliberately *leading-only*. A full ``\b``-style boundary would
#: also demand a word edge after the term, which would drop the plural and
#: possessive forms that matter ("two golden tickets", "the XSSes") in exchange
#: for a handful of trailing false positives. Plurals are worth more.
#:
#: Scoped to :data:`TECHNIQUE_TERMS`. The service and tool lists are matched by
#: plain scan, unchanged: their members are proper names and product names where
#: a substring accident is rare, and their extraction is pinned by tests whose
#: expectations were derived under the plain scan.
_ANCHORED: dict[str, re.Pattern[str]] = {
    term: re.compile(r"(?<![a-z0-9])" + re.escape(term))
    for term in TECHNIQUE_TERMS
    if len(term) <= _BOUNDARY_MIN_LEN
}


def _found(terms: list[str], lowered: str) -> list[str]:
    """Return the sorted subset of ``terms`` that occur in ``lowered``.

    A plain substring scan, except for the terms in :data:`_ANCHORED`, which
    must also start at a word edge.
    """
    hits = []
    for term in terms:
        anchored = _ANCHORED.get(term)
        matched = anchored.search(lowered) if anchored else term in lowered
        if matched:
            hits.append(term)
    return sorted(set(hits))


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
