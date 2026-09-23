import pytest

from blackbook.storage import Source, Document, Chunk
from blackbook.storage.database import sha256_text


def test_upsert_source_and_get(db):
    with db.session():
        db.upsert_source(Source(source_id="hacktricks", name="HackTricks", authority="trusted"))
    s = db.get_source("hacktricks")
    assert s is not None
    assert s["name"] == "HackTricks"
    assert s["authority"] == "trusted"


def test_document_upsert_change_detection(db):
    with db.session():
        db.upsert_source(Source(source_id="s1", name="S1"))
        h = sha256_text("v1")
        did1 = db.upsert_document(Document(source_id="s1", external_id="e1", title="T", content_hash=h))
        # Same hash -> same doc_id, no change
        did2 = db.upsert_document(Document(source_id="s1", external_id="e1", title="T", content_hash=h))
        assert did1 == did2
        # New hash -> update in place (same doc_id via ON CONFLICT)
        did3 = db.upsert_document(Document(source_id="s1", external_id="e1", title="T2", content_hash=sha256_text("v2")))
        assert did3 == did1
    doc = db.get_document(did1)
    assert doc["title"] == "T2"


def test_fts_search_and_source_filter(seeded_db):
    res = seeded_db.fts_search('"kerberoasting"', limit=10)
    assert res, "expected a hit for kerberoasting"
    assert any(r["source_id"] == "hacktricks" for r in res)

    # Filter to a source that has no matching chunk
    res_none = seeded_db.fts_search('"kerberoasting"', source_ids=["0xdf"], limit=10)
    assert all(r["source_id"] == "0xdf" for r in res_none)

    # Filter to the matching source
    res_ht = seeded_db.fts_search('"kerberoasting"', source_ids=["hacktricks"], limit=10)
    assert res_ht and all(r["source_id"] == "hacktricks" for r in res_ht)


def test_fts_triggers_keep_index_in_sync(db):
    with db.session():
        db.upsert_source(Source(source_id="s1", name="S1"))
        did = db.upsert_document(Document(source_id="s1", external_id="e", title="Doc", content_hash="h"))
        db.replace_chunks(did, [Chunk(doc_id=did, ordinal=0, text="xyzzy unique token", section_path=[], token_estimate=2, content_hash="c")])
    assert db.fts_search('"xyzzy"', limit=5)
    # Delete the chunk and confirm the FTS hit disappears
    with db.session():
        db.replace_chunks(did, [])
    assert db.fts_search('"xyzzy"', limit=5) == []


def test_counts(seeded_db):
    counts = seeded_db.counts()
    assert counts["sources"] == 2
    assert counts["documents"] == 2
    assert counts["chunks"] == 3


# -- counts_cached ---------------------------------------------------------
#
# Counting the chunk table costs a few milliseconds against a real corpus, and
# the poll-shaped readouts (/health, the landing page) ask for it on a timer.
# These pin the two properties that keep the cache honest: it is actually
# serving a cached reading, and a write this process commits clears it.


def _add_source_out_of_band(db, source_id="late"):
    """Insert a source without going through ``session()``.

    This is what a write made by another process looks like from here: it
    reaches the tables but never touches the invalidation hook. Used to prove
    the cache is being served rather than silently recomputed.
    """
    with db.conn:
        db.upsert_source(Source(source_id=source_id, name=source_id))


def test_counts_cached_agrees_with_counts(seeded_db):
    assert seeded_db.counts_cached() == seeded_db.counts()


def test_counts_cached_serves_the_cached_reading(seeded_db):
    first = seeded_db.counts_cached()
    _add_source_out_of_band(seeded_db)
    # The exact read sees the new row; the cached read is still inside its TTL.
    assert seeded_db.counts()["sources"] == 3
    assert seeded_db.counts_cached() == first


def test_a_committed_write_invalidates_the_cache(seeded_db):
    seeded_db.counts_cached()
    with seeded_db.session():
        seeded_db.upsert_source(Source(source_id="late", name="Late"))
    assert seeded_db.counts_cached()["sources"] == 3


def test_a_rolled_back_session_leaves_the_cache_alone(seeded_db):
    seeded_db.counts_cached()
    with pytest.raises(RuntimeError):
        with seeded_db.session():
            seeded_db.upsert_source(Source(source_id="ghost", name="Ghost"))
            raise RuntimeError("boom")
    assert seeded_db.get_source("ghost") is None
    # A rollback changed nothing, so the cache was left in place. Had the
    # rollback cleared it, this read would have recomputed and picked up the
    # out-of-band row below.
    _add_source_out_of_band(seeded_db)
    assert seeded_db.counts_cached()["sources"] == 2


def test_max_age_zero_always_recomputes(seeded_db):
    seeded_db.counts_cached()
    _add_source_out_of_band(seeded_db)
    assert seeded_db.counts_cached(max_age=0.0)["sources"] == 3


def test_the_cached_reading_is_handed_out_as_a_copy(seeded_db):
    got = seeded_db.counts_cached()
    got["chunks"] = -1
    assert seeded_db.counts_cached()["chunks"] == 3
