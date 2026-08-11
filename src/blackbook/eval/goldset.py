"""The labelled gold query set.

Each :class:`GoldQuery` pairs a natural query with the set of benchmark
documents (by ``external_id``) that genuinely answer it, plus the retrieval
``mode`` and any filters to exercise. The harness runs every query through the
real retriever and scores the returned ranking against ``relevant``.

Labels are deliberately conservative — a query's ``relevant`` set lists only the
documents a knowledgeable analyst would call a correct answer, so a passing
recall/MRR score reflects real ranking quality rather than a loose label. The
queries are spread across every mode the reranker supports so a regression in
any one intent-bias is visible:

* ``keyword`` — pure FTS5/BM25 term matching against reference pages;
* ``technique`` — technique/reference material should surface first;
* ``case_similarity`` — hands-on writeups should be favoured over reference
  pages for the same subject (this is where the writeup mode-bonus is tested);
* ``hybrid`` — the default path (lexical when embeddings are disabled).

The relevant ``external_id``s must exist in :mod:`blackbook.eval.corpus`; a
consistency test asserts that, so a typo here fails loudly rather than silently
scoring zero.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldQuery:
    """One labelled evaluation query."""

    qid: str
    query: str
    relevant: tuple[str, ...]  # benchmark document external_ids
    mode: str = "hybrid"
    platform: str | None = None
    categories: tuple[str, ...] | None = None
    note: str = ""


GOLD_QUERIES: list[GoldQuery] = [
    # -- keyword / reference retrieval -------------------------------------
    GoldQuery(
        qid="kw-kerberoast",
        query="kerberoasting service principal name request TGS crack offline",
        relevant=("ref/kerberoasting.md",),
        mode="keyword",
    ),
    GoldQuery(
        qid="kw-asrep",
        query="as-rep roasting kerberos pre-authentication disabled GetNPUsers",
        relevant=("ref/asrep-roasting.md",),
        mode="keyword",
    ),
    GoldQuery(
        qid="kw-spray",
        query="password spraying lockout threshold observation window",
        relevant=("ref/password-spraying.md",),
        mode="keyword",
    ),
    GoldQuery(
        qid="kw-relay",
        query="ntlm relay responder ntlmrelayx smb signing disabled",
        relevant=("ref/ntlm-relay.md",),
        mode="keyword",
    ),
    GoldQuery(
        qid="kw-dcsync",
        query="dcsync replication rights krbtgt secretsdump golden ticket",
        relevant=("ref/dcsync.md",),
        mode="keyword",
    ),
    GoldQuery(
        qid="kw-sqli",
        query="sql injection union select parameterised prepared statements",
        relevant=("ref/sql-injection.md",),
        mode="keyword",
    ),
    GoldQuery(
        qid="kw-lfi",
        query="local file inclusion directory traversal etc passwd php filter",
        relevant=("ref/lfi.md",),
        mode="keyword",
    ),
    GoldQuery(
        qid="kw-ssrf",
        query="server side request forgery instance metadata credentials",
        relevant=("ref/ssrf.md",),
        mode="keyword",
    ),
    # -- technique-biased ---------------------------------------------------
    # In technique mode the canonical reference page should rank above the
    # writeup that merely demonstrates the technique.
    GoldQuery(
        qid="tech-kerberoast",
        query="kerberoasting",
        relevant=("ref/kerberoasting.md",),
        mode="technique",
    ),
    GoldQuery(
        qid="tech-dcsync",
        query="dcsync",
        relevant=("ref/dcsync.md",),
        mode="technique",
    ),
    # -- case-similarity (writeups favoured) --------------------------------
    GoldQuery(
        qid="case-forest",
        query="active directory as-rep roasting to domain admin foothold windows",
        relevant=("wu/htb-forest.md",),
        mode="case_similarity",
    ),
    GoldQuery(
        qid="case-clue",
        query="linux web local file inclusion log poisoning remote code execution",
        relevant=("wu/pg-clue.md",),
        mode="case_similarity",
    ),
    GoldQuery(
        qid="case-secnotes",
        query="sql injection login bypass upload webshell code execution",
        relevant=("wu/htb-secnotes.md",),
        mode="case_similarity",
    ),
    # -- filtered -----------------------------------------------------------
    # Platform filter must not drop the on-platform reference page.
    GoldQuery(
        qid="filt-relay-windows",
        query="ntlm relay coerce authentication",
        relevant=("ref/ntlm-relay.md",),
        mode="keyword",
        platform="windows",
    ),
]
