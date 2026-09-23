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

The ``case-*`` queries pair a writeup with the reference page covering the same
ground. Note what the mode bonus can and cannot do: it multiplies the base score
by ``1 + _MODE_BONUS`` (1.3), so a writeup wins only when its base score is
within ~77% of the reference page's. These queries are phrased with descriptive
vocabulary to land in that band; a bare one-word technique query lands far
outside it and the reference page wins in every mode. That boundary is measured
and pinned in the tests rather than left as an assumption.

The relevant ``external_id``s must exist in :mod:`blackbook.eval.corpus`; a
consistency test asserts that, so a typo here fails loudly rather than silently
scoring zero.

Two properties of this set are load-bearing and are tested explicitly, because
a benchmark that cannot fail measures nothing:

* **Labels are set by correctness, never by observed output.** A query whose
  retriever ranking is wrong keeps its correct label and simply scores lower.
  Tuning ``relevant`` to match what the retriever returns would make the scores
  meaningless, so it is not done here.
* **Some queries are hard on purpose.** Paraphrases avoid the document's own
  title words, several queries have more than one correct answer (so recall is
  genuinely partial), and the ``neg-*`` queries have no correct answer in the
  corpus at all. An earlier version of this file had 14 single-answer queries
  that every one of which scored a perfect 1.0, which meant no ranking
  regression could ever have been detected.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldQuery:
    """One labelled evaluation query."""

    qid: str
    query: str
    relevant: tuple[str, ...]  # benchmark document external_ids; empty if unanswerable
    mode: str = "hybrid"
    platform: str | None = None
    categories: tuple[str, ...] | None = None
    # True when the corpus genuinely cannot answer the query. Such a query has
    # an empty ``relevant`` set and is scored on the *precision* side: the
    # retriever must not present a confident-looking match. The FTS5 query is
    # OR-joined, so a weak match is returned anyway; what is asserted is that
    # its score is far below every answered query's.
    unanswerable: bool = False
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
        note="ref/sql-injection-prevention.md shares every term except UNION; "
             "ranking the prevention page first is a miss.",
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
    # -- hard paraphrases ---------------------------------------------------
    # These deliberately avoid the answer document's own title words, so BM25
    # has to rank on the body rather than a title shortcut.
    GoldQuery(
        qid="par-metadata-creds",
        query="reading the link-local address that serves temporary cloud credentials",
        relevant=("ref/ssrf.md",),
        mode="keyword",
        note="never says SSRF, request forgery, or metadata service by name",
    ),
    GoldQuery(
        qid="par-timing-inference",
        query="inferring values one bit at a time from response timing",
        relevant=("ref/sql-injection.md",),
        mode="keyword",
        note="describes blind injection without naming SQL injection",
    ),
    GoldQuery(
        qid="par-coerced-auth",
        query="forcing a machine to authenticate to a listener we control and forwarding it onward",
        relevant=("ref/ntlm-relay.md",),
        mode="keyword",
    ),
    GoldQuery(
        qid="par-log-poison",
        query="including a file we can write to in order to run code as the web user",
        relevant=("ref/lfi.md", "wu/pg-clue.md"),
        mode="keyword",
        note="both the reference page and the case describe this",
    ),
    # -- multi-answer queries (genuinely partial recall) --------------------
    GoldQuery(
        qid="multi-ticket-crack",
        query="obtaining kerberos ticket material from a domain controller and cracking it offline",
        relevant=("ref/kerberoasting.md", "ref/asrep-roasting.md"),
        mode="keyword",
        note="both ticket attacks are correct answers; recall is partial by design",
    ),
    GoldQuery(
        qid="multi-replication",
        query="abusing replication to extract account secrets and forge a domain-wide ticket",
        relevant=("ref/dcsync.md", "ref/golden-ticket.md"),
        mode="keyword",
    ),
    # -- technique-biased ---------------------------------------------------
    # In technique mode the canonical reference page ranks above anything else
    # covering the technique. These use bare terms, where the title weighting
    # makes the reference page dominant; the *flip* to writeups is tested by the
    # case-* queries below, which is where the mode bonus can actually act.
    GoldQuery(
        qid="tech-kerberoast",
        query="kerberoasting",
        relevant=("ref/kerberoasting.md",),
        mode="technique",
        note="no writeup in the corpus contains the exact token 'kerberoasting' "
             "(wu/htb-sizzle says 'kerberoast' and the index does not stem), so "
             "this query has exactly one match.",
    ),
    GoldQuery(
        qid="tech-dcsync",
        query="dcsync",
        relevant=("ref/dcsync.md",),
        mode="technique",
    ),
    GoldQuery(
        qid="tech-delegation",
        query="unconstrained kerberos delegation abuse",
        relevant=("ref/kerberos-delegation.md",),
        mode="technique",
    ),
    GoldQuery(
        qid="tech-golden",
        query="golden ticket krbtgt forgery",
        relevant=("ref/golden-ticket.md",),
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
    # Cross-source near-duplicates: the case must beat the reference page that
    # covers the same ground, which is the whole point of the mode bonus.
    GoldQuery(
        qid="case-blackfield",
        query="active directory kerberos pre-authentication disabled backup operator secrets dump",
        relevant=("wu/htb-blackfield.md",),
        mode="case_similarity",
        note="wu/htb-blackfield.md and ref/asrep-roasting.md overlap heavily",
    ),
    GoldQuery(
        qid="case-hawat",
        query="web application fetches a user supplied url reaching an internal service",
        relevant=("wu/pg-hawat.md",),
        mode="case_similarity",
        note="near-duplicate of ref/ssrf.md",
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
    # -- unanswerable (precision side) --------------------------------------
    # Nothing in the benchmark corpus is about these subjects. Because the FTS5
    # query is OR-joined, the retriever still returns *something*; the harness
    # therefore asserts the best hit is weak rather than absent.
    GoldQuery(
        qid="neg-kubernetes",
        query="kubernetes admission controller network policy enforcement",
        relevant=(),
        mode="keyword",
        unanswerable=True,
    ),
    GoldQuery(
        qid="neg-cloud-iam",
        query="gcp iam policy binding terraform state key rotation",
        relevant=(),
        mode="keyword",
        unanswerable=True,
    ),
    GoldQuery(
        qid="neg-wireless",
        query="wpa2 four way handshake deauthentication capture",
        relevant=(),
        mode="keyword",
        unanswerable=True,
    ),
]
