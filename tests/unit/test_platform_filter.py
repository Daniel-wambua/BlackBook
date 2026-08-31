"""Tests for platform/categories as *hard* filters.

A filter must actually restrict results (at the SQL level for lexical, at the
reranker for the semantic path), never silently widen or merely down-rank.
"""

from __future__ import annotations

import pytest

from blackbook.config import Settings
from blackbook.retrieval.hybrid import HybridRetriever
from blackbook.retrieval.lexical import LexicalHit, LexicalRetriever
from blackbook.retrieval.reranker import rerank
from blackbook.storage import Chunk, Document
from blackbook.storage.database import sha256_text


@pytest.fixture()
def mixed_db(seeded_db):
    """seeded_db plus a linux-tagged document so there is something to filter out."""
    with seeded_db.session():
        d3 = seeded_db.upsert_document(
            Document(
                source_id="hacktricks",
                external_id="linux/privesc.md",
                title="Linux Privilege Escalation",
                url="https://book.hacktricks.xyz/linux-hardening/privilege-escalation",
                content_hash=sha256_text("linux doc"),
                categories=["linux", "privesc"],
            )
        )
        seeded_db.replace_chunks(
            d3,
            [
                Chunk(
                    doc_id=d3,
                    ordinal=0,
                    text="Kerberoasting has no meaning here, but sudo misconfiguration does.",
                    section_path=["Linux", "Privilege Escalation"],
                    token_estimate=12,
                    content_hash=sha256_text("c-linux"),
                )
            ],
        )
    return seeded_db


def _hit(doc_id: int, title: str, categories: list[str], score: float = 0.5) -> LexicalHit:
    return LexicalHit(
        chunk_id=doc_id,
        doc_id=doc_id,
        text=f"text {doc_id} {title}",  # distinct: near-dup shingle check would drop clones
        title=title,
        source_id="s",
        source_name="s",
        authority="trusted",
        bm25=0.0,
        score=score,
        metadata={"categories": categories},
    )


# -- SQL level (fts_search) --------------------------------------------------


# A MATCH expression hitting chunks in all three seeded documents (the Forest
# chunk spells it "kerberoast", the others "kerberoasting").
_MATCH_ALL = '"kerberoasting" OR "kerberoast" OR "sudo"'


def test_fts_platform_filter_excludes(mixed_db):
    res = mixed_db.fts_search(_MATCH_ALL, limit=10)
    assert {r["title"] for r in res} == {"Kerberoasting", "HTB: Forest", "Linux Privilege Escalation"}
    res_linux = mixed_db.fts_search(_MATCH_ALL, limit=10, platform="linux")
    assert res_linux, "linux filter should still match the linux doc's chunk"
    assert {r["title"] for r in res_linux} == {"Linux Privilege Escalation"}


def test_fts_platform_filter_case_insensitive(mixed_db):
    res = mixed_db.fts_search(_MATCH_ALL, limit=10, platform="Linux")
    assert {r["title"] for r in res} == {"Linux Privilege Escalation"}


def test_fts_platform_filter_no_match(mixed_db):
    assert mixed_db.fts_search(_MATCH_ALL, limit=10, platform="solaris") == []


def test_fts_categories_filter_any_of(mixed_db):
    # Windows matches both windows-tagged docs; htb matches only Forest.
    res = mixed_db.fts_search(_MATCH_ALL, limit=10, categories=["htb"])
    assert {r["title"] for r in res} == {"HTB: Forest"}
    res_both = mixed_db.fts_search(_MATCH_ALL, limit=10, categories=["htb", "active-directory"])
    assert {r["title"] for r in res_both} == {"Kerberoasting", "HTB: Forest"}


def test_fts_no_filter_returns_all(mixed_db):
    res = mixed_db.fts_search(_MATCH_ALL, limit=10)
    # No filter must never restrict: all three documents are represented
    # (the Kerberoasting doc contributes more than one chunk).
    assert {r["title"] for r in res} == {
        "Kerberoasting", "HTB: Forest", "Linux Privilege Escalation",
    }


# -- retriever level ---------------------------------------------------------


def test_lexical_retriever_threads_platform(mixed_db):
    hits = LexicalRetriever(mixed_db).search("kerberoast kerberoasting sudo", platform="linux", limit=10)
    assert [h.title for h in hits] == ["Linux Privilege Escalation"]


def test_hybrid_platform_filter_restricts(mixed_db):
    settings = Settings()
    retriever = HybridRetriever(mixed_db, settings)
    results = retriever.search("kerberoast kerberoasting sudo", platform="linux")
    assert [r.title for r in results] == ["Linux Privilege Escalation"]


def test_hybrid_categories_filter_restricts(mixed_db):
    retriever = HybridRetriever(mixed_db, Settings())
    results = retriever.search("kerberoast kerberoasting sudo", categories=["htb"])
    assert [r.title for r in results] == ["HTB: Forest"]


# -- reranker hard filter (covers the semantic path) -------------------------


def test_rerank_drops_non_matching_platform():
    keep, drop = _hit(1, "Win doc", ["windows"]), _hit(2, "Lin doc", ["linux"])
    out = rerank([keep, drop], query="q", limit=10, platform="windows")
    assert [h.chunk_id for h in out] == [1]


def test_rerank_drops_non_matching_categories():
    keep, drop = _hit(1, "HTB doc", ["htb", "windows"]), _hit(2, "Other", ["vulnhub"])
    out = rerank([keep, drop], query="q", limit=10, categories=["htb"])
    assert [h.chunk_id for h in out] == [1]


def test_rerank_drops_hits_with_no_categories_when_filtered():
    """A hit with no category tags can't satisfy the filter — exclude it."""
    hit = _hit(1, "No tags doc", [])
    assert rerank([hit], query="q", limit=10, platform="windows") == []


def test_rerank_no_filter_keeps_all():
    hits = [_hit(1, "a", ["windows"]), _hit(2, "b", ["linux"])]
    assert len(rerank(hits, query="q", limit=10)) == 2
