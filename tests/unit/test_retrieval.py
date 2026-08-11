from blackbook.config import Settings, DatabaseConfig
from blackbook.retrieval import HybridRetriever
from blackbook.retrieval.lexical import LexicalHit, LexicalRetriever, normalize_query
from blackbook.retrieval.reranker import rerank


def make_settings(tmp_path):
    return Settings(home=tmp_path, database=DatabaseConfig(path=str(tmp_path / "d.db")))


def test_normalize_query_escapes_punctuation():
    q = normalize_query("kerberoasting; DROP TABLE chunks_fts; --")
    # Should only contain quoted word tokens joined by OR; no raw SQL survives
    assert "DROP" not in q or '"drop"' in q
    assert ";" not in q
    assert "--" not in q


def test_normalize_query_empty():
    assert normalize_query("!!! ???") == '""'


def test_lexical_search_returns_hits(seeded_db):
    lex = LexicalRetriever(seeded_db)
    hits = lex.search("kerberoasting SPN", limit=5)
    assert hits
    assert hits[0].title in ("Kerberoasting", "HTB: Forest")
    # snippet present
    assert hits[0].snippet


def test_hybrid_search_source_filter(tmp_path, seeded_db):
    settings = make_settings(tmp_path)
    r = HybridRetriever(seeded_db, settings)
    res = r.search("kerberoast", source_ids=["0xdf"], limit=5)
    assert res and all(x.source_id == "0xdf" for x in res)


def test_hybrid_search_respects_limit(tmp_path, seeded_db):
    settings = make_settings(tmp_path)
    r = HybridRetriever(seeded_db, settings)
    res = r.search("kerberoast OR service OR tickets", limit=1)
    assert len(res) <= 1


def test_per_document_diversity_cap(tmp_path, seeded_db):
    # Kerberoasting doc has 2 matching chunks; cap is 2, so both may appear but
    # no more than cap from a single doc.
    settings = make_settings(tmp_path)
    r = HybridRetriever(seeded_db, settings)
    res = r.search("kerberoasting service tickets SPN", limit=10)
    from collections import Counter

    by_doc = Counter(x.doc_id for x in res)
    assert all(v <= settings.retrieval.per_document_cap for v in by_doc.values())


def _hit(chunk_id, doc_id, *, title, section_path=None, categories=None,
         score=0.5, authority="trusted"):
    return LexicalHit(
        chunk_id=chunk_id, doc_id=doc_id, text=f"text {chunk_id}", title=title,
        source_id="s", source_name="S", authority=authority, bm25=-1.0,
        score=score, section_path=section_path or [],
        metadata={"categories": categories or []},
    )


def test_case_similarity_mode_favours_writeups():
    # Two equally-scored hits; only one is a writeup/case study by category.
    # Neutral query matches neither title, so the base scores tie and only the
    # mode bonus can break it.
    ref = _hit(1, 1, title="Reference page", categories=["active-directory"])
    case = _hit(2, 2, title="HTB: Forest", categories=["htb", "windows"])
    ranked = rerank([ref, case], query="zzz", limit=10, mode="case_similarity")
    # The writeup is boosted above the equally-scored reference page.
    assert ranked[0].chunk_id == 2
    # Without the mode, the tie is not broken in the writeup's favour.
    plain = rerank([ref, case], query="zzz", limit=10)
    assert plain[0].score == plain[1].score


def test_technique_mode_favours_technique_headings():
    # Both hits mention the term, but only one names it in the heading path.
    heading = _hit(1, 1, title="SQL Injection",
                   section_path=["Web", "SQL Injection"], categories=["web"])
    passing = _hit(2, 2, title="Some CTF box",
                   section_path=["Recon"], categories=["htb"])
    ranked = rerank([heading, passing], query="injection", limit=10,
                    mode="technique")
    assert ranked[0].chunk_id == 1


def test_mode_bonus_does_not_exclude_non_matching_hits():
    # A far stronger generic hit still outranks a weak on-intent one: the bonus
    # nudges, it does not gate.
    strong = _hit(1, 1, title="Generic page", score=1.0, categories=["web"])
    weak_case = _hit(2, 2, title="HTB: Box", score=0.1, categories=["htb"])
    ranked = rerank([strong, weak_case], query="x", limit=10,
                    mode="case_similarity")
    assert {h.chunk_id for h in ranked} == {1, 2}   # nothing dropped
    assert ranked[0].chunk_id == 1                    # strong hit still leads
