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
