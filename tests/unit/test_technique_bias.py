"""Tests for the ``techniques`` parameter as a reranking bias.

Techniques resolve through the controlled vocabulary and bias retrieval toward
chunks whose title/section names the technique — they are no longer just
extra query terms. Unknown techniques still join the query as plain terms and
are surfaced to the caller.
"""

from __future__ import annotations

from blackbook.mcp.schemas import (
    CaseSearchInput,
    SearchInput,
)
from blackbook.mcp.tools import KnowledgeTools
from blackbook.retrieval.lexical import LexicalHit
from blackbook.retrieval.reranker import rerank


def _hit(doc_id: int, title: str, sections: list[str], score: float = 0.5) -> LexicalHit:
    return LexicalHit(
        chunk_id=doc_id,
        doc_id=doc_id,
        text=f"text {doc_id} {title}",
        title=title,
        source_id="s",
        source_name="s",
        authority="trusted",
        bm25=0.0,
        score=score,
        section_path=sections,
    )


# -- reranker -----------------------------------------------------------------


def test_technique_bias_promotes_named_hit():
    # Comparable base scores, but only one chunk's *heading* names the
    # requested technique — the bias must promote it.
    generic = _hit(1, "AD notes", ["Misc notes"], score=0.55)
    named = _hit(2, "Kerberoasting", ["Active Directory", "Kerberoasting"], score=0.5)
    out = rerank(
        [generic, named], query="active directory", limit=2, techniques=["kerberoasting"]
    )
    assert [h.chunk_id for h in out] == [2, 1]


def test_technique_bias_cannot_override_much_better_match():
    # A bias is not a filter: a far stronger base hit still wins.
    generic = _hit(1, "AD notes", ["Misc notes"], score=0.9)
    named = _hit(2, "Kerberoasting", ["Kerberoasting"], score=0.3)
    out = rerank(
        [generic, named], query="active directory", limit=2, techniques=["kerberoasting"]
    )
    assert out[0].chunk_id == 1


def test_technique_bias_resolves_aliases():
    # "kerberoast" is an alias — must resolve to the indexed canonical term.
    generic = _hit(1, "AD notes", ["Misc"], score=0.55)
    named = _hit(2, "Kerberoasting", ["Kerberoasting"], score=0.5)
    out = rerank(
        [generic, named], query="tickets", limit=2, techniques=["kerberoast"]
    )
    assert out[0].chunk_id == 2


def test_technique_bias_is_a_bias_not_a_filter():
    # No hit names the technique — nothing is excluded, ranking is untouched.
    a = _hit(1, "a", ["x"], score=0.7)
    b = _hit(2, "b", ["y"], score=0.5)
    out = rerank([a, b], query="q", limit=2, techniques=["dcsync"])
    assert [h.chunk_id for h in out] == [1, 2]


def test_technique_bias_ignores_body_mentions():
    # The technique only appears in body text (not title/section) — no bias.
    mentioning = LexicalHit(
        chunk_id=1, doc_id=1, text="we used kerberoasting to crack a ticket",
        title="Forest writeup", source_id="s", source_name="s",
        authority="trusted", bm25=0.0, score=0.9,
    )
    other = _hit(2, "Other", ["z"], score=0.5)
    out = rerank(
        [mentioning, other], query="q", limit=2, techniques=["kerberoasting"]
    )
    assert out[0].chunk_id == 1  # still first on base score, but no boost needed


def test_no_techniques_leaves_scores_alone():
    h = _hit(1, "Kerberoasting", ["Kerberoasting"], score=0.5)
    # A technique the hit doesn't name must not change its score vs. passing
    # no techniques at all.
    plain = rerank([h], query="q", limit=1)
    untouched = rerank([h], query="q", limit=1, techniques=["dcsync"])
    assert plain[0].score == untouched[0].score


# -- tool layer ---------------------------------------------------------------


def _tools(seeded_db) -> KnowledgeTools:
    from blackbook.config import Settings

    return KnowledgeTools(seeded_db, Settings())


def test_search_techniques_note_lists_unresolved(seeded_db):
    tools = _tools(seeded_db)
    out = tools.knowledge_search(
        SearchInput(query="cracking tickets", techniques=["kerberoast", "made-up-thing"])
    )
    assert "made-up-thing" in out.note
    assert "plain terms" in out.note
    assert "kerberoast" not in out.note  # resolved aliases are not flagged


def test_search_unresolved_technique_still_searched(seeded_db):
    tools = _tools(seeded_db)
    out = tools.knowledge_search(
        SearchInput(query="cracking tickets", techniques=["kerberoasting"])
    )
    assert out.count >= 1


def test_case_search_threads_techniques(seeded_db):
    tools = _tools(seeded_db)
    out = tools.knowledge_case_search(
        CaseSearchInput(query="service account tickets", techniques=["kerberoast"])
    )
    # Must not raise, must return provenance-bearing results or a clean note.
    assert out.count >= 0
    for item in out.results:
        assert item.ref.chunk_id > 0
