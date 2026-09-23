"""Cross-source diversity: one source should not own a result set.

The per-document cap already stops one page filling the list. This is the other
axis: a source with thousands of documents can do the same thing at a coarser
grain. HackTricks alone is over a thousand documents here, so a query it covers
heavily can return eight HackTricks chunks and nothing else, even when the
pdfs, the writeup repos and the cheatsheets hold comparable evidence.

The property that matters most, and the one most of these tests are about, is
that the cap is a *preference*. It reorders when another source has something
worth showing, and it never shortens the list when none does.
"""

from __future__ import annotations

from collections import Counter

from blackbook.config import Settings
from blackbook.retrieval.hybrid import HybridRetriever
from blackbook.retrieval.lexical import LexicalHit
from blackbook.retrieval.reranker import rerank
from blackbook.storage import Document, Source
from blackbook.storage.database import Database

# Distinct bodies on purpose. The reranker drops near-duplicates across
# documents, so shared wording would be filtered before the source cap ever
# came up and the test would prove nothing about diversity.
_BODIES = [
    "kerberoasting requests a service ticket for a service principal name",
    "asrep roasting targets accounts with preauthentication disabled",
    "dcsync abuses directory replication rights to dump the krbtgt hash",
    "unconstrained delegation lets a compromised host capture a domain admin tgt",
    "shadow credentials write a key credential to the target computer object",
    "adcs esc8 relays ntlm to the certificate authority web enrollment endpoint",
    "resource based constrained delegation edits msdsallowedtoact on the target",
    "golden ticket forges a tgt signed with the krbtgt account key",
]


def _hit(chunk_id, doc_id, source_id, *, score, text=None):
    return LexicalHit(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text or _BODIES[chunk_id % len(_BODIES)],
        title=f"Doc {doc_id}",
        source_id=source_id,
        source_name=source_id,
        authority="trusted",
        bm25=-1.0,
        score=score,
        metadata={"categories": []},
    )


# -- the cap reorders ------------------------------------------------------


def test_source_cap_prefers_a_second_source_over_a_fifth_from_the_first():
    # Four strong hits from source "big", then one weaker from "small". The
    # small hit's text is distinct, or the dedup pass would drop it before the
    # source cap got a say and this would prove nothing.
    hits = [_hit(i, i, "big", score=1.0 - i * 0.01) for i in range(1, 5)]
    hits.append(
        _hit(
            90,
            90,
            "small",
            score=0.60,
            text="a writeup walking through a full domain compromise end to end",
        )
    )

    capped = rerank(hits, query="q", limit=4, per_document_cap=10, per_source_cap=3)
    assert len(capped) == 4
    assert Counter(h.source_id for h in capped)["big"] == 3
    assert "small" in {h.source_id for h in capped}

    # Without the cap the weaker hit never gets a look in.
    plain = rerank(hits, query="q", limit=4, per_document_cap=10, per_source_cap=0)
    assert {h.source_id for h in plain} == {"big"}


def test_the_cap_never_shortens_the_result_set():
    """Slots the cap leaves empty are filled from what it set aside."""
    hits = [_hit(i, i, "only", score=1.0 - i * 0.01) for i in range(1, 9)]
    capped = rerank(hits, query="q", limit=5, per_document_cap=10, per_source_cap=2)
    assert len(capped) == 5
    # Two inside the cap, three backfilled, all from the one source that exists.
    assert all(h.source_id == "only" for h in capped)


def test_a_cap_of_zero_disables_the_preference():
    hits = [_hit(i, i, "big", score=1.0 - i * 0.01) for i in range(1, 5)]
    hits.append(
        _hit(
            90,
            90,
            "small",
            score=0.60,
            text="a writeup walking through a full domain compromise end to end",
        )
    )
    out = rerank(hits, query="q", limit=4, per_document_cap=10, per_source_cap=0)
    assert [h.chunk_id for h in out] == [1, 2, 3, 4]


def test_the_cap_does_not_reorder_when_the_other_source_is_worse():
    """Diversity is a preference for equal evidence, not a quota to fill."""
    strong = [_hit(i, i, "big", score=1.0) for i in range(1, 3)]
    weak = [_hit(9, 9, "small", score=0.10)]
    out = rerank(strong + weak, query="q", limit=2, per_document_cap=10, per_source_cap=1)
    # The cap would have admitted "small" into slot 2 only after backfill, and
    # backfill happens after the strong hits have had their turn.
    assert [h.source_id for h in out] == ["big", "big"]


# -- the two caps interact correctly ---------------------------------------


def test_the_document_cap_still_holds_during_backfill():
    """Relaxing the source cap must not let one page repeat past its own cap."""
    hits = [
        _hit(1, 1, "big", score=1.0),
        _hit(2, 1, "big", score=0.99),
        _hit(3, 1, "big", score=0.98),  # third chunk of doc 1: over the doc cap
        _hit(4, 2, "big", score=0.97),
    ]
    out = rerank(hits, query="q", limit=4, per_document_cap=2, per_source_cap=1)
    by_doc = Counter(h.doc_id for h in out)
    assert all(v <= 2 for v in by_doc.values())
    assert 3 not in [h.chunk_id for h in out]


def test_a_near_duplicate_is_dropped_for_good_not_deferred():
    """Backfill must not resurrect what the dedup pass rejected."""
    body = "always installd elevated runs an msi with system privileges"
    dup = "Always Installd Elevated runs an MSI with SYSTEM privileges!"
    hits = [
        _hit(1, 1, "big", score=1.0, text=body),
        _hit(2, 2, "big", score=0.99, text=dup),
        _hit(3, 3, "big", score=0.98),
    ]
    out = rerank(hits, query="q", limit=5, per_document_cap=10, per_source_cap=1)
    assert dup not in [h.text for h in out]


# -- through the retriever -------------------------------------------------


def _seed(db: Database) -> None:
    """Six matching chunks from "big", two from "small".

    Deliberately lopsided: that asymmetry is the situation the cap exists for.
    """
    from blackbook.storage import Chunk
    from blackbook.storage.database import sha256_text

    db.upsert_source(Source(source_id="big", name="Big"))
    db.upsert_source(Source(source_id="small", name="Small"))
    for i, body in enumerate(_BODIES):
        source_id = "big" if i < 6 else "small"
        doc_id = db.upsert_document(
            Document(
                source_id=source_id,
                external_id=f"d{i}",
                title=f"Doc {i}",
                content_hash=sha256_text(f"doc {i}"),
            )
        )
        db.replace_chunks(
            doc_id,
            [
                Chunk(
                    doc_id=doc_id,
                    ordinal=0,
                    text=body,
                    section_path=[f"Doc {i}"],
                    token_estimate=12,
                    content_hash=sha256_text(f"chunk {i}"),
                )
            ],
        )


def test_retriever_honours_the_configured_cap(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    try:
        _seed(db)
        # Every body names a different technique, so a query naming them all
        # matches across the whole set rather than only the lopsided source.
        query = (
            "kerberoasting preauthentication replication delegation certificate "
            "credential golden ticket"
        )

        # Two slots and a cap of one: both sources should be represented. This
        # is the whole point of the cap, and it only holds when the second
        # source has something to contribute.
        settings = Settings()
        settings.retrieval.per_source_cap = 1
        capped = HybridRetriever(db, settings).search(query, limit=2)
        assert len(capped) == 2
        assert len({r.source_id for r in capped}) == 2

        # A third slot has nowhere diverse to come from, so it is backfilled
        # rather than left empty: the cap prefers, it does not truncate.
        three = HybridRetriever(db, settings).search(query, limit=3)
        assert len(three) == 3

        settings.retrieval.per_source_cap = 0
        uncapped = HybridRetriever(db, settings).search(query, limit=3)
        assert uncapped  # still returns results with the cap off
    finally:
        db.close()
