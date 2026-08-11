"""Tests for near-duplicate detection (retrieval.dedup) and the reranker's
cross-document dedup."""

from blackbook.retrieval.dedup import (
    is_near_duplicate,
    normalize,
    normalized_hash,
    shingle_overlap,
    shingles,
)
from blackbook.retrieval.lexical import LexicalHit
from blackbook.retrieval.reranker import rerank


def _hit(chunk_id, doc_id, text, score=0.9, authority="trusted", title="Doc"):
    return LexicalHit(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        title=title,
        source_id="s",
        source_name="S",
        authority=authority,
        bm25=1.0,
        score=score,
        metadata={},
    )


# -- normalize / normalized_hash -------------------------------------------


def test_normalize_collapses_case_and_punctuation():
    a = "Kerberoasting:   GetUserSPNs.py  -request"
    b = "kerberoasting getuserspns py request"
    assert normalize(a) == normalize(b)


def test_normalized_hash_equal_across_formatting():
    a = "Always Install Elevated\nallows MSI escalation!"
    b = "always   install elevated allows msi escalation"
    assert normalized_hash(a) == normalized_hash(b)


def test_normalized_hash_differs_for_different_text():
    assert normalized_hash("kerberoasting asreproasting") != normalized_hash("pass the hash")


# -- shingles / shingle_overlap --------------------------------------------


def test_shingles_handles_short_text():
    # Fewer tokens than n yields a single shingle, not an error.
    assert shingles("one two") == {"one two"}
    assert shingles("") == set()


def test_shingle_overlap_identical_is_one():
    t = "the quick brown fox jumps over the lazy dog"
    assert shingle_overlap(t, t) == 1.0


def test_shingle_overlap_disjoint_is_zero():
    a = "alpha beta gamma delta epsilon zeta"
    b = "one two three four five six seven eight"
    assert shingle_overlap(a, b) == 0.0


def test_shingle_overlap_partial_between_zero_and_one():
    a = "the quick brown fox jumps over the lazy dog today"
    b = "the quick brown fox jumps over a different fence now"
    ov = shingle_overlap(a, b)
    assert 0.0 < ov < 1.0


# -- is_near_duplicate -------------------------------------------------------


def test_is_near_duplicate_identical_normalized():
    a = "Mimikatz sekurlsa::logonpasswords"
    b = "  mimikatz   sekurlsa logonpasswords "
    assert is_near_duplicate(a, b)


def test_is_near_duplicate_high_overlap():
    base = (
        "kerberoasting requests service tickets for accounts with spns then "
        "cracks them offline with hashcat"
    )
    variant = base + " easily"
    assert is_near_duplicate(base, variant, threshold=0.7)


def test_is_near_duplicate_distinct_texts():
    a = "pass the hash uses an ntlm hash to authenticate without a password"
    b = "sql injection exploits unsanitized user input in database queries"
    assert not is_near_duplicate(a, b)


# -- reranker cross-document dedup -----------------------------------------


def test_reranker_drops_cross_document_near_duplicate():
    text = (
        "always installd elevated lets a standard user run an msi with "
        "system privileges when both registry keys are set to one"
    )
    # Same words, only case/punctuation differ -> overlap is 1.0 after
    # normalization, so the reranker must treat it as a duplicate.
    near_dup = (
        "Always Installd Elevated lets a standard user run an MSI with "
        "SYSTEM privileges, when both registry keys are set to one!"
    )
    distinct = "unquoted service paths let you plant a binary in a writable directory"
    hits = [
        _hit(1, 100, text, score=0.95, title="HackTricks"),
        _hit(2, 200, near_dup, score=0.90, title="PDF"),   # near-dup, different doc
        _hit(3, 300, distinct, score=0.80, title="Other"),
    ]
    out = rerank(hits, query="privilege escalation", limit=10, per_document_cap=10)
    texts = [h.text for h in out]
    # The near-duplicate from the other document must be dropped.
    assert near_dup not in texts
    assert text in texts and distinct in texts


def test_reranker_keeps_distinct_documents():
    hits = [
        _hit(1, 100, "kerberoasting requests rc4 service tickets to crack offline", score=0.9),
        _hit(2, 200, "asreproasting targets accounts without preauthentication", score=0.85),
        _hit(3, 300, "dcsync abuses replication rights to dump password hashes", score=0.8),
    ]
    out = rerank(hits, query="active directory", limit=10, per_document_cap=10)
    assert len(out) == 3
