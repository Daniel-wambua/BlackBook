"""The evaluation benchmark corpus.

A curated set of authored security documents, indexed through the **real**
chunker (:func:`blackbook.retrieval.chunking.chunk_markdown`) and the **real**
storage writes (``upsert_source`` / ``upsert_document`` / ``replace_chunks``).
There is no adapter and no network: :func:`build_eval_corpus` takes a
:class:`~blackbook.storage.database.Database` and populates it directly.

Design constraints that keep the benchmark honest:

* The content is accurate, not fabricated nonsense, so ranking behaviour is
  realistic — but the documents are explicitly **synthetic benchmark material**
  (their source name says so). They are never presented to a user as knowledge.
* Every chunk written is real text; a citation into this corpus resolves to a
  real excerpt exactly as it would for HackTricks or 0xdf. The benchmark
  measures the same code path production uses.
* Two sources — reference pages and hands-on writeups — so the ``technique`` and
  ``case_similarity`` intent-modes have distinct material to separate. Writeups
  carry the ``htb`` category marker the reranker keys on.

The corpus is deliberately small but includes *distractors* (multiple
overlapping AD / web documents) so recall@k and MRR actually discriminate a
good ranking from a bad one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from blackbook.ingestion.base import content_hash
from blackbook.retrieval.chunking import chunk_markdown, estimate_tokens
from blackbook.storage.database import Database
from blackbook.storage.models import Chunk, Document, Source

# Source ids are namespaced so a benchmark DB can never be confused with a real
# index, and so the names visibly announce the documents are synthetic.
REF_SOURCE = "eval_reference"
WRITEUP_SOURCE = "eval_writeups"


@dataclass(frozen=True)
class BenchmarkDoc:
    """One authored document in the benchmark corpus."""

    source_id: str
    external_id: str
    title: str
    body: str
    categories: list[str] = field(default_factory=list)
    url: str | None = None


EVAL_SOURCES: list[Source] = [
    Source(
        source_id=REF_SOURCE,
        name="Eval Benchmark — Reference (synthetic)",
        authority="trusted",
        source_type="filesystem",
    ),
    Source(
        source_id=WRITEUP_SOURCE,
        name="Eval Benchmark — Writeups (synthetic)",
        authority="trusted",
        source_type="filesystem",
    ),
]


# --- reference documents ---------------------------------------------------

_REF_DOCS: list[BenchmarkDoc] = [
    BenchmarkDoc(
        source_id=REF_SOURCE,
        external_id="ref/kerberoasting.md",
        title="Kerberoasting",
        categories=["active-directory", "kerberos", "windows"],
        body="""# Kerberoasting

Kerberoasting abuses the Kerberos protocol in Active Directory. Any
authenticated domain user can request a service ticket (TGS) for an account
that has a Service Principal Name (SPN) set.

## How it works

The TGS is encrypted with the service account's NTLM hash. Because a normal
user can request tickets for any SPN, an attacker requests RC4-encrypted
service tickets and cracks them offline to recover the service account
password. No elevated privileges are required to request the tickets.

## Tooling

Use GetUserSPNs.py from Impacket to enumerate SPNs and request tickets, or
Rubeus on Windows. Crack the resulting hashes with hashcat mode 13100.
""",
    ),
    BenchmarkDoc(
        source_id=REF_SOURCE,
        external_id="ref/asrep-roasting.md",
        title="AS-REP Roasting",
        categories=["active-directory", "kerberos", "windows"],
        body="""# AS-REP Roasting

AS-REP roasting targets Active Directory accounts that have Kerberos
pre-authentication disabled (the "Do not require Kerberos preauthentication"
flag).

## How it works

When pre-authentication is disabled, the domain controller returns an AS-REP
whose encrypted portion is derived from the user's password — without the
attacker ever proving they know it. The attacker collects this material and
cracks it offline.

## Tooling

GetNPUsers.py from Impacket enumerates preauth-disabled users and dumps the
AS-REP hashes; crack them with hashcat mode 18200.
""",
    ),
    BenchmarkDoc(
        source_id=REF_SOURCE,
        external_id="ref/password-spraying.md",
        title="Password Spraying",
        categories=["active-directory", "windows"],
        body="""# Password Spraying

Password spraying tries a small number of common passwords against a large set
of accounts, staying under lockout thresholds by using one password per round
across every user.

## Considerations

Enumerate the domain password policy first to learn the lockout threshold and
observation window. Spray slowly, one attempt per account per window. Tools
such as netexec (crackmapexec) and kerbrute automate spraying over SMB, LDAP,
or Kerberos.
""",
    ),
    BenchmarkDoc(
        source_id=REF_SOURCE,
        external_id="ref/ntlm-relay.md",
        title="NTLM Relay",
        categories=["active-directory", "smb", "windows"],
        body="""# NTLM Relay

NTLM relay captures an NTLM authentication and forwards it to another service,
authenticating as the victim without ever cracking a password.

## How it works

Responder poisons name-resolution requests to coerce authentication, while
ntlmrelayx from Impacket relays the captured NTLM to a target — for example
relaying to LDAP to grant an attacker-controlled account rights, or to SMB to
execute commands. SMB signing being disabled is the classic precondition.
""",
    ),
    BenchmarkDoc(
        source_id=REF_SOURCE,
        external_id="ref/dcsync.md",
        title="DCSync",
        categories=["active-directory", "windows", "credential-access"],
        body="""# DCSync

DCSync impersonates a domain controller and asks another DC to replicate
account secrets, yielding password hashes for any principal — including krbtgt.

## Requirements

The calling principal needs the replication rights (DS-Replication-Get-Changes
and DS-Replication-Get-Changes-All), typically held by Domain Admins. mimikatz
(lsadump::dcsync) and Impacket's secretsdump.py both implement it. Recovering
the krbtgt hash enables golden ticket forgery.
""",
    ),
    BenchmarkDoc(
        source_id=REF_SOURCE,
        external_id="ref/sql-injection.md",
        title="SQL Injection",
        categories=["web", "injection"],
        body="""# SQL Injection

SQL injection occurs when untrusted input is concatenated into a SQL query,
letting an attacker alter the query's logic.

## Types

Union-based injection extracts data through UNION SELECT; boolean- and
time-based blind injection infer data one bit at a time from the response or
its timing. Always test for injection in every parameter, including headers and
cookies.

## Tooling

sqlmap automates detection and exploitation across many database backends. The
durable fix is parameterised queries (prepared statements), never string
concatenation.
""",
    ),
    BenchmarkDoc(
        source_id=REF_SOURCE,
        external_id="ref/lfi.md",
        title="Local File Inclusion",
        categories=["web", "injection"],
        body="""# Local File Inclusion

Local file inclusion (LFI) lets an attacker read files on the web server by
controlling a path passed to an include/read routine.

## Exploitation

Directory traversal sequences (../../) reach files such as /etc/passwd. PHP
wrappers like php://filter can base64-encode source files for disclosure, and
under some conditions log poisoning or session files turn LFI into remote code
execution. Distinct from remote file inclusion (RFI), which pulls a remote URL.
""",
    ),
    BenchmarkDoc(
        source_id=REF_SOURCE,
        external_id="ref/ssrf.md",
        title="Server-Side Request Forgery",
        categories=["web"],
        body="""# Server-Side Request Forgery

Server-side request forgery (SSRF) coerces a server into making HTTP requests
of the attacker's choosing, reaching systems the attacker cannot address
directly.

## Impact

A common target is the cloud instance metadata service (169.254.169.254),
which can leak temporary credentials. Internal-only admin panels and port
scanning of the internal network are also reachable. Defences include strict
allow-lists of destinations and blocking link-local and internal ranges.
""",
    ),
    # -- distractors -------------------------------------------------------
    # These exist to make the benchmark discriminate. Each one shares heavy
    # vocabulary with a document above while being a *different* answer, so a
    # ranking that merely matches terms ranks them highly and loses. Before
    # these were added every gold query scored a perfect 1.0 and the benchmark
    # could not have detected a ranking regression at all.
    BenchmarkDoc(
        source_id=REF_SOURCE,
        external_id="ref/kerberos-delegation.md",
        title="Kerberos Delegation Abuse",
        categories=["active-directory", "kerberos", "windows"],
        body="""# Kerberos Delegation Abuse

Kerberos delegation lets a service impersonate a user to a *different* service,
which is the mechanism behind several Active Directory privilege escalations.

## Unconstrained delegation

A host configured for unconstrained delegation caches a forwardable TGT for
every user who authenticates to it. Compromising such a host and coercing a
domain controller to authenticate to it yields a domain controller machine
account ticket, which can be replayed.

## Constrained delegation

Constrained delegation restricts impersonation to a named service, and
resource-based constrained delegation moves the permission onto the target
object — where a write on the target's ``msDS-AllowedToActOnBehalfOfOtherIdentity``
attribute is enough to abuse it.
""",
    ),
    BenchmarkDoc(
        source_id=REF_SOURCE,
        external_id="ref/golden-ticket.md",
        title="Golden Ticket Forgery",
        categories=["active-directory", "kerberos", "credential-access", "windows"],
        body="""# Golden Ticket Forgery

A golden ticket is a forged Kerberos TGT signed with the krbtgt account's key,
granting its bearer domain-wide access as any principal.

## Obtaining the key

Forging one requires the krbtgt hash or AES key, which is why replication-based
attacks that extract account secrets are the usual precursor. Once the key is
held, no further authentication to the domain controller is needed.

## Detection and limits

Golden tickets bypass normal authentication logs; the practical detection is
anomalous TGT lifetimes and encryption types. Rotating the krbtgt password twice
invalidates outstanding tickets.
""",
    ),
    BenchmarkDoc(
        source_id=REF_SOURCE,
        external_id="ref/sql-injection-prevention.md",
        title="Preventing SQL Injection",
        categories=["web", "defence"],
        body="""# Preventing SQL Injection

Injection flaws are prevented at the point where a query is built, not by
filtering input.

## Parameterised queries

Bind every value as a parameter so the database parses the statement once and
treats the value as data rather than SQL. This is the only reliably complete
defence for SQL injection and it costs nothing at runtime.

## When parameters are not possible

Identifiers such as table names cannot be bound as parameters; map user input
to a known allow-list of identifiers instead. Escaping and blacklisting are
weaker and routinely bypassed by encoding tricks.
""",
    ),
]


# --- writeup documents (hands-on cases) ------------------------------------

_WRITEUP_DOCS: list[BenchmarkDoc] = [
    BenchmarkDoc(
        source_id=WRITEUP_SOURCE,
        external_id="wu/htb-forest.md",
        title="HTB: Forest",
        categories=["htb", "windows", "writeup"],
        body="""# HTB: Forest

Forest is a Windows Active Directory box. This walkthrough chains AS-REP
roasting into a foothold and finishes with a DCSync.

## Foothold

Enumeration with rpcclient lists domain users. The account svc-alfresco has
Kerberos pre-authentication disabled, so I AS-REP roast it with GetNPUsers.py
and crack the hash in hashcat to get a shell over evil-winrm.

## Privilege escalation

BloodHound shows the path to Domain Admin via account operators and a WriteDacl
on the domain, which I abuse to grant DCSync rights and run secretsdump.py for
the Administrator hash.
""",
    ),
    BenchmarkDoc(
        source_id=WRITEUP_SOURCE,
        external_id="wu/htb-sizzle.md",
        title="HTB: Sizzle",
        categories=["htb", "windows", "writeup"],
        body="""# HTB: Sizzle

Sizzle is a hard Windows box centred on Active Directory Certificate Services
and Kerberos.

## Foothold

An open SMB share allows me to plant a .scf file that coerces authentication;
Responder captures the hash. After getting a certificate I authenticate over
WinRM.

## Escalation

With a shell I kerberoast a service account using Rubeus, crack the ticket, and
then use the recovered credentials to reach the domain administrator.
""",
    ),
    BenchmarkDoc(
        source_id=WRITEUP_SOURCE,
        external_id="wu/htb-secnotes.md",
        title="HTB: SecNotes",
        categories=["htb", "web", "writeup"],
        body="""# HTB: SecNotes

SecNotes is a Windows box with a web front end.

## Foothold

The notes application is vulnerable to second-order SQL injection in the login
flow, letting me bypass authentication. From the authenticated area I upload a
webshell to an SMB-backed share and get code execution.

## Escalation

A stored PowerShell history file leaks credentials that grant administrator
access.
""",
    ),
    BenchmarkDoc(
        source_id=WRITEUP_SOURCE,
        external_id="wu/htb-cascade.md",
        title="HTB: Cascade",
        categories=["htb", "windows", "writeup"],
        body="""# HTB: Cascade

Cascade is a Windows Active Directory box solved without any exploit — pure
enumeration and credential reuse.

## Foothold

LDAP anonymous bind leaks a base64 password in a user attribute. That gets me a
foothold, and a deleted-objects AD recycle bin entry exposes another account's
password.

## Escalation

ACL abuse across nested groups leads to the AD Recycle Bin group and ultimately
to administrator credentials stored in a legacy application's configuration.
""",
    ),
    BenchmarkDoc(
        source_id=WRITEUP_SOURCE,
        external_id="wu/pg-clue.md",
        title="PG: Clue (Linux LFI to RCE)",
        categories=["pg", "linux", "writeup", "oscp"],
        body="""# PG: Clue (Linux LFI to RCE)

Clue is a Linux Proving Grounds box driven by a web vulnerability.

## Foothold

A parameter on the site is vulnerable to local file inclusion. I read
/etc/passwd to confirm, then poison the Apache access log through the
User-Agent header and include it to gain remote code execution as www-data.

## Escalation

A world-writable cron script running as root is my path to a root shell.
""",
    ),
    # -- cross-source near-duplicates --------------------------------------
    # Same subject matter as a reference page, told as a case. These are what
    # make the mode bonus observable: for a query whose terms match both about
    # equally, ``case_similarity`` must rank the writeup first while
    # ``technique`` ranks the reference page.
    #
    # "About equally" is load-bearing, and the measurements say exactly how
    # equal. With ``reranker._MODE_BONUS = 0.3`` the mode multiplier is 1.3, so
    # a writeup overtakes the reference page only when its base score is at
    # least ~1/1.3 = 0.77 of the reference's. Measured from this corpus:
    # blackfield/asrep 0.93 (flips), hawat/ssrf 1.22 (flips), forest/asrep 0.88
    # (flips). A query that matches a reference page's *title* pushes the ratio
    # far below that (a bare "kerberoasting" gives 0.00 for wu/htb-sizzle, whose
    # body says "kerberoast" — the index does not stem), and then no mode bonus
    # can flip it. The gold set therefore tests the bonus with the descriptive
    # queries below, not with bare terms.
    BenchmarkDoc(
        source_id=WRITEUP_SOURCE,
        external_id="wu/htb-blackfield.md",
        title="HTB: Blackfield",
        categories=["htb", "windows", "writeup"],
        body="""# HTB: Blackfield

Blackfield is a hard Windows Active Directory box.

## Foothold

An anonymous SMB share contains a backup archive of the domain controller,
which leaks a password hash. Cracking it gives a domain user. Enumeration then
shows an account with Kerberos pre-authentication disabled, so I AS-REP roast
it and crack the ticket offline.

## Escalation

The foothold account holds backup operator rights. Using those I dump the
domain's account secrets and recover the administrator hash, then log in over
WinRM.
""",
    ),
    BenchmarkDoc(
        source_id=WRITEUP_SOURCE,
        external_id="wu/pg-hawat.md",
        title="PG: Hawat (SSRF to internal service)",
        categories=["pg", "linux", "web", "writeup"],
        body="""# PG: Hawat (SSRF to internal service)

Hawat is a Linux Proving Grounds box whose web application fetches URLs on the
user's behalf.

## Foothold

The fetch feature is a server-side request forgery. I point it at the internal
port range and find a service bound to loopback that is not exposed publicly.
Its admin interface accepts a request created through the SSRF, which gets me
command execution.

## Escalation

A misconfigured systemd unit running as root is the path to a root shell.
""",
    ),
]


BENCHMARK_DOCS: list[BenchmarkDoc] = _REF_DOCS + _WRITEUP_DOCS


def build_eval_corpus(db: Database) -> dict[str, int]:
    """Populate ``db`` with the benchmark corpus and return summary counts.

    Uses the production chunker and storage writes so the benchmark exercises
    exactly the code path a real ingest would. Idempotent: content-hash change
    detection means rebuilding an unchanged corpus rewrites the same rows.
    """
    docs_written = 0
    chunks_written = 0
    with db.session():
        for src in EVAL_SOURCES:
            db.upsert_source(src)

        for bd in BENCHMARK_DOCS:
            doc_id = db.upsert_document(
                Document(
                    source_id=bd.source_id,
                    external_id=bd.external_id,
                    title=bd.title,
                    url=bd.url,
                    content_hash=content_hash(bd.body),
                    categories=list(bd.categories),
                    metadata={"benchmark": True},
                )
            )
            raw_chunks = chunk_markdown(
                bd.body, title_path=list(bd.categories) + [bd.title]
            )
            rows = [
                Chunk(
                    doc_id=doc_id,
                    ordinal=rc.ordinal,
                    text=rc.text,
                    section_path=rc.section_path,
                    page=rc.page,
                    token_estimate=estimate_tokens(rc.text),
                    content_hash=content_hash(rc.text),
                    metadata={"kind": rc.kind},
                )
                for rc in raw_chunks
            ]
            db.replace_chunks(doc_id, rows)
            docs_written += 1
            chunks_written += len(rows)

        # Leave the FTS index compact — the same maintenance a real ingest does.
        db.optimize_fts()

    return {
        "sources": len(EVAL_SOURCES),
        "documents": docs_written,
        "chunks": chunks_written,
    }
