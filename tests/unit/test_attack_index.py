"""Tests for the corpus-derived MITRE ATT&CK lookup.

:mod:`blackbook.knowledge.attack_index` exists so a caller can name any ATT&CK
technique and get its record, without a second hand-maintained table that starts
going stale the moment the source is re-ingested. These tests pin the properties
that make it trustworthy rather than the size of the table it happens to build:

* it reads the indexed corpus and nothing else, so it cannot fabricate a
  mapping — an unknown name resolves to ``None``, never to a guess;
* an ID is exact in both directions, while a *name* ATT&CK reuses across
  techniques is treated as a question that has no single answer, not as a coin
  flip to hide;
* the lookup is rebuilt when the source changes, which is what lets a
  long-lived server pick up a re-ingest without a restart;
* the curated map in :mod:`blackbook.knowledge.vocab` and this index compose the
  way :meth:`blackbook.mcp.tools.KnowledgeTools._attack_ids` relies on: the
  curated judgement wins, the corpus answers what curation does not cover.

The ATT&CK fixture holds one live technique for Kerberoasting (T1558.003) plus a
revoked, a deprecated and a non-ATT&CK object, so it also confirms the index
only ever sees what the adapter actually indexed. Note that the fixture names
its sub-technique the way attack.mitre.org displays it ("Steal or Forge Kerberos
Tickets: Kerberoasting"), while the live enterprise bundle stores the STIX leaf
name alone ("Kerberoasting") — so the display-path tests build the live shape
explicitly rather than leaning on the fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from blackbook.config import Settings, SourceConfig
from blackbook.ingestion.attack import MitreAttackAdapter
from blackbook.ingestion.pipeline import IngestionPipeline
from blackbook.knowledge.attack_index import (
    ATTACK_SOURCE_ID,
    AttackIndex,
    normalize_name,
)
from blackbook.knowledge.vocab import (
    TECHNIQUE_ATTACK_IDS,
    TECHNIQUE_TERMS,
    _TECHNIQUE_ALIASES,
    attack_id,
    extract_signals,
)
from blackbook.mcp.schemas import TechniqueInput
from blackbook.mcp.tools import KnowledgeTools
from blackbook.storage.database import Database
from blackbook.storage.models import Document, Source

FIXTURES = Path(__file__).parent.parent / "fixtures"


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
def _seed_attack_source(db: Database, rows: list[tuple[str, str]]) -> None:
    """Insert ``(attack_id, name)`` pairs as indexed ATT&CK documents.

    Written straight through the storage layer rather than through the adapter,
    so a test can create shapes the fixture bundle does not contain — above all
    two techniques that share a name, which is the case the lookup has to refuse
    to answer.
    """
    with db.session():
        db.upsert_source(
            Source(
                source_id=ATTACK_SOURCE_ID,
                name="MITRE ATT&CK",
                authority="official",
                enabled=True,
                source_type="git",
            )
        )
        for attack_id_value, name in rows:
            db.upsert_document(
                Document(
                    source_id=ATTACK_SOURCE_ID,
                    external_id=attack_id_value,
                    title=f"{name} ({attack_id_value})",
                    url=f"https://attack.mitre.org/techniques/{attack_id_value}/",
                    content_hash=attack_id_value,
                    metadata={"format": "stix", "attack_id": attack_id_value},
                )
            )


@pytest.fixture()
def attack_db(tmp_path):
    """A database holding the real ATT&CK adapter output from the fixture bundle."""
    db = Database(str(tmp_path / "attack.db"))
    cfg = SourceConfig(
        id="attack", name="MITRE ATT&CK", type="git", authority="official",
        url="https://raw.githubusercontent.com/mitre-attack/attack-stix-data/"
            "master/enterprise-attack/enterprise-attack.json",
        ref="master",
    )
    adapter = MitreAttackAdapter(cfg, raw_dir=str(tmp_path))
    adapter.fetch = lambda force=False: None
    adapter._bundle_path = FIXTURES / "attack" / "bundle.json"
    with db.session():
        db.upsert_source(
            Source(
                source_id="attack", name="MITRE ATT&CK", authority="official",
                enabled=True, source_type="git", url=cfg.url,
            )
        )
    IngestionPipeline(db).run(adapter)
    return db


# --------------------------------------------------------------------------- #
# Name folding                                                                 #
# --------------------------------------------------------------------------- #
def test_normalize_name_folds_punctuation_and_case():
    assert normalize_name("Kerberoasting") == "kerberoasting"
    assert normalize_name("Scheduled Task/Job") == "scheduled task job"
    assert normalize_name("  Steal or Forge Kerberos Tickets: Kerberoasting ") == (
        "steal or forge kerberos tickets kerberoasting"
    )
    # Punctuation runs collapse to a single space, so spacing variants agree.
    assert normalize_name("Scheduled  Task / Job") == normalize_name("scheduled task/job")
    assert normalize_name("") == ""


# --------------------------------------------------------------------------- #
# Lookups against the real adapter output                                      #
# --------------------------------------------------------------------------- #
def test_index_sees_only_what_the_adapter_indexed(attack_db):
    """Revoked, deprecated and non-ATT&CK objects never reach the lookup."""
    index = AttackIndex(attack_db)
    assert len(index) == 2
    assert set(index.by_id) == {"T1558.003", "T1190"}
    assert index.resolve("T9999") is None  # revoked
    assert index.resolve("T9998") is None  # deprecated


def test_index_strips_the_appended_id_from_the_title(attack_db):
    """The adapter writes "Name (ID)"; the lookup must recover the bare name.

    The fixture's sub-technique carries the site's ``Parent: Child`` form as its
    stored name, so it is reachable under that name directly rather than through
    derivation, and it must appear exactly once — the two routes to a path must
    not produce two entries for one technique.
    """
    index = AttackIndex(attack_db)
    assert index.name("T1558.003") == "Steal or Forge Kerberos Tickets: Kerberoasting"
    assert index.name("t1190") == "Exploit Public-Facing Application"  # case-insensitive
    assert index.resolve("Exploit Public-Facing Application") == "T1190"
    assert index.resolve("Steal or Forge Kerberos Tickets: Kerberoasting") == "T1558.003"
    assert index.ids_for("Steal or Forge Kerberos Tickets: Kerberoasting") == ["T1558.003"]


def test_index_resolves_the_id_to_itself(attack_db):
    index = AttackIndex(attack_db)
    assert index.resolve("T1190") == "T1190"
    assert index.resolve("t1190") == "T1190"
    assert index.resolve("T1") is None  # malformed, never guessed at
    assert index.resolve("") is None
    assert index.resolve("not a technique") is None


def test_unmapped_name_is_none_not_a_guess(attack_db):
    """The whole point: absence of evidence is reported, not papered over."""
    index = AttackIndex(attack_db)
    assert index.resolve("SQL Injection") is None
    assert index.resolve("Kerberoasting") is None  # the fixture names it differently
    assert index.ids_for("SQL Injection") == []


# --------------------------------------------------------------------------- #
# Ambiguity: a name ATT&CK gives to more than one technique                    #
# --------------------------------------------------------------------------- #
def test_ambiguous_name_resolves_to_none_and_is_inspectable(tmp_path):
    """ATT&CK really does reuse names across distinct techniques.

    "Domains" is both acquiring infrastructure (T1583.001) and compromising
    someone else's (T1584.001) in the live corpus. Resolving that to one ID
    would present a coin flip as a fact, so the lookup declines and offers the
    candidates instead.
    """
    db = Database(str(tmp_path / "ambig.db"))
    try:
        _seed_attack_source(db, [("T1583.001", "Domains"), ("T1584.001", "Domains")])
        index = AttackIndex(db)
        assert len(index) == 2
        assert index.resolve("Domains") is None
        assert index.resolve("domains") is None  # folding does not rescue it
        assert index.ids_for("Domains") == ["T1583.001", "T1584.001"]
        # An ID is unique even where the name is not, so this direction is exact.
        assert index.name("T1583.001") == "Domains"
        assert index.name("T1584.001") == "Domains"
    finally:
        db.close()


def test_unique_name_still_resolves_alongside_an_ambiguous_one(tmp_path):
    db = Database(str(tmp_path / "ambig2.db"))
    try:
        _seed_attack_source(
            db, [("T1583.001", "Domains"), ("T1584.001", "Domains"), ("T1059", "Command and Scripting Interpreter")]
        )
        index = AttackIndex(db)
        assert index.resolve("Command and Scripting Interpreter") == "T1059"
        assert index.resolve("Domains") is None
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# Derived display paths: what attack.mitre.org prints                          #
# --------------------------------------------------------------------------- #
def test_derived_display_path_resolves_where_the_corpus_stores_only_the_leaf(tmp_path):
    """The corpus holds "JavaScript"; the site prints the path; both resolve.

    This is the gap the derivation closes. Every one of the live bundle's 475
    sub-techniques is stored under its leaf name, so the longer name a reader
    copies off attack.mitre.org used to be precisely the one that found nothing.
    """
    db = Database(str(tmp_path / "path.db"))
    try:
        _seed_attack_source(
            db,
            [("T1059", "Command and Scripting Interpreter"), ("T1059.007", "JavaScript")],
        )
        index = AttackIndex(db)
        # The stored name — what a search of the corpus itself turns up.
        assert index.resolve("JavaScript") == "T1059.007"
        # The displayed name — what a reader has in hand.
        assert index.resolve("Command and Scripting Interpreter: JavaScript") == "T1059.007"
        # Folding applies to the path like any other name, so the colon and the
        # spacing carry no weight.
        assert index.resolve("command and scripting interpreter javascript") == "T1059.007"
    finally:
        db.close()


def test_display_path_is_not_derived_when_the_parent_is_absent(tmp_path):
    """A parentless sub-technique yields no path alias, because nothing says what the parent is.

    The derivation reads the parent name out of an indexed row. With no such
    row there is no prefix to build, and inventing one would be exactly the
    fabrication the leaf-name design avoids.
    """
    db = Database(str(tmp_path / "orphan.db"))
    try:
        _seed_attack_source(db, [("T1059.007", "JavaScript")])
        index = AttackIndex(db)
        assert index.resolve("JavaScript") == "T1059.007"
        assert index.resolve("Command and Scripting Interpreter: JavaScript") is None
        assert index.ids_for("Command and Scripting Interpreter: JavaScript") == []
    finally:
        db.close()


def test_display_path_that_collides_with_a_stored_name_is_refused(tmp_path):
    """A derived path equal to some technique's own name is ambiguous, not a tie to break.

    No technique in the current corpus has a colon in its stored name, so this
    cannot arise today. A later release could introduce one, and resolving it to
    whichever entry was written first would report a coin flip as a fact, so the
    ordinary ambiguity rule is left to cover it.
    """
    db = Database(str(tmp_path / "collide.db"))
    try:
        _seed_attack_source(
            db,
            [
                ("T9001", "Foo: Bar"),   # a technique stored under a two-part name
                ("T9002", "Foo"),        # a parent...
                ("T9002.001", "Bar"),    # ...whose child derives the same path
            ],
        )
        index = AttackIndex(db)
        assert index.resolve("Foo: Bar") is None
        assert index.ids_for("Foo: Bar") == ["T9001", "T9002.001"]
        # Each half stays individually exact, which is what makes the refusal
        # informative rather than a dead end.
        assert index.resolve("Bar") == "T9002.001"
        assert index.resolve("T9001") == "T9001"
    finally:
        db.close()


def test_refresh_gains_a_display_path_once_its_parent_arrives(tmp_path):
    """The alias follows the same rebuild rule as every other lookup.

    A corpus that gains the parent gains the path on the next lookup, with no
    restart and no explicit invalidation.
    """
    db = Database(str(tmp_path / "late_parent.db"))
    try:
        _seed_attack_source(db, [("T1059.007", "JavaScript")])
        index = AttackIndex(db)
        assert index.resolve("Command and Scripting Interpreter: JavaScript") is None

        _seed_attack_source(db, [("T1059", "Command and Scripting Interpreter")])
        assert index.refresh() is True
        assert index.resolve("Command and Scripting Interpreter: JavaScript") == "T1059.007"
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# Staleness: the rebuild rule                                                  #
# --------------------------------------------------------------------------- #
def test_refresh_rebuilds_when_the_source_grows(tmp_path):
    db = Database(str(tmp_path / "grow.db"))
    try:
        _seed_attack_source(db, [("T1059", "Command and Scripting Interpreter")])
        index = AttackIndex(db)
        assert len(index) == 1
        assert index.refresh() is False  # nothing changed, no rebuild

        _seed_attack_source(db, [("T1190", "Exploit Public-Facing Application")])
        assert index.refresh() is True
        assert len(index) == 2
        assert index.resolve("Exploit Public-Facing Application") == "T1190"
    finally:
        db.close()


def test_refresh_notices_a_rewrite_without_a_count_change(tmp_path):
    """A correction, not an addition, is still picked up.

    The count does not move when an existing technique is rewritten, so the
    signature has to carry ``MAX(updated_at)`` as well. The timestamp is set
    explicitly here because two writes inside the same second are genuinely
    indistinguishable at SQLite's ``datetime('now')`` resolution, which is a
    stated limit of the rule rather than something a test can paper over.
    """
    db = Database(str(tmp_path / "rewrite.db"))
    try:
        _seed_attack_source(db, [("T1059", "Command and Scripting Interpreter")])
        index = AttackIndex(db)
        assert index.name("T1059") == "Command and Scripting Interpreter"

        with db.session():
            db.conn.execute(
                "UPDATE documents SET external_id = ?, title = ?, metadata = ?, "
                "updated_at = '2031-01-01 00:00:00' "
                "WHERE source_id = ? AND external_id = ?",
                (
                    "T1059.004",
                    "Command and Scripting Interpreter: Unix Shell (T1059.004)",
                    '{"attack_id": "T1059.004"}',
                    ATTACK_SOURCE_ID,
                    "T1059",
                ),
            )
        assert index.refresh() is True
        assert index.name("T1059.004") == "Command and Scripting Interpreter: Unix Shell"
        assert index.resolve("Command and Scripting Interpreter: Unix Shell") == "T1059.004"
    finally:
        db.close()


def test_empty_or_absent_source_yields_an_empty_index(tmp_path):
    """No ATT&CK source indexed is an empty lookup, not an error."""
    db = Database(str(tmp_path / "empty.db"))
    try:
        index = AttackIndex(db)
        assert len(index) == 0
        assert index.resolve("T1190") is None
        assert index.ids_for("Domains") == []
        assert index.name("T1190") is None
        assert index.refresh() is False
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# Curated map: corrections and invariants                                      #
# --------------------------------------------------------------------------- #
def test_delegation_terms_do_not_map_to_application_access_token():
    """The four corrected rows, pinned so they cannot silently come back.

    T1550.001 is "Application Access Token" — application and cloud token
    theft. It has nothing to do with Kerberos delegation, and the old table
    pointed all three delegation terms at it. ATT&CK has no Kerberos-delegation
    technique, so the honest mapping is the class itself, T1558.
    """
    assert attack_id("unconstrained delegation") == "T1558"
    assert attack_id("constrained delegation") == "T1558"
    assert attack_id("rbcd") == "T1558"
    assert attack_id("resource-based constrained delegation") == "T1558"  # alias
    assert attack_id("shadow credentials") == "T1098"
    assert "T1550.001" not in set(TECHNIQUE_ATTACK_IDS.values())


def test_every_mapped_term_is_a_vocabulary_term():
    """A map key that is not in ``TECHNIQUE_TERMS`` is unreachable dead weight.

    ``attack_id`` resolves through the vocabulary first, so a row keyed on a
    term the extractor cannot produce would never fire. This keeps the map and
    the term list from drifting apart.
    """
    assert set(TECHNIQUE_ATTACK_IDS) <= set(TECHNIQUE_TERMS)
    assert set(_TECHNIQUE_ALIASES.values()) <= set(TECHNIQUE_TERMS)


def test_new_web_terms_extract_from_an_observation():
    """The gap this closed: a web observation used to detect nothing at all."""
    _, techniques, _ = extract_signals(
        "GraphQL introspection enabled, IDOR on the user object, "
        "no rate limit on password reset"
    )
    assert {"graphql", "idor", "rate limit"} <= set(techniques)


# --------------------------------------------------------------------------- #
# Short-term matching: the leading-boundary rule                               #
# --------------------------------------------------------------------------- #
def test_short_terms_need_a_word_boundary():
    """Four-letter terms must not match inside an unrelated word.

    "bola" is an authenticated-API access-control flaw; it is also four letters
    that occur inside "ebola". Without the anchor a virus article is reported as
    a broken-object-level-authorization finding, and a base64 blob can hit any
    of the short tokens by chance.
    """
    _, techniques, _ = extract_signals("the ebola outbreak and a zzlfixx token")
    assert "bola" not in techniques
    assert "lfi" not in techniques


def test_boundary_rule_keeps_plurals_and_normal_matches():
    """Leading-only anchoring: the plural forms still land.

    A full word boundary would drop "IDORs" and "XSSes", which is a worse trade
    than the trailing false positives it would remove.
    """
    _, techniques, _ = extract_signals("two IDORs in the invoice endpoint")
    assert "idor" in techniques
    _, techniques, _ = extract_signals("an LFI in the download parameter")
    assert "lfi" in techniques


# --------------------------------------------------------------------------- #
# Composition: curated first, corpus second                                    #
# --------------------------------------------------------------------------- #
def test_tool_lookup_prefers_curation_then_falls_back(attack_db):
    tools = KnowledgeTools(attack_db, Settings())
    # Curated: our own term, a judgement the corpus cannot make for us.
    assert tools._attack_ids("kerberoast") == "T1558.003"
    # Corpus: a real ATT&CK name the curated map deliberately does not carry.
    assert tools._attack_ids("Exploit Public-Facing Application") == "T1190"
    assert tools._attack_ids("T1190") == "T1190"
    # Neither knows it, so nothing is claimed.
    assert tools._attack_ids("not-a-technique") is None


def test_index_is_built_lazily_and_reused(attack_db):
    tools = KnowledgeTools(attack_db, Settings())
    assert tools._attack_index is None  # nothing paid for until it is needed
    tools._attack_ids("Exploit Public-Facing Application")
    built = tools._attack_index
    assert built is not None
    tools._attack_ids("T1190")
    assert tools._attack_index is built  # reused, not rebuilt per call


def test_corpus_resolution_enriches_a_technique_dossier(attack_db):
    """End to end: naming a real ATT&CK technique is enough to get its record.

    Before the fallback existed this returned a dossier with no tactics, no
    platforms and no ATT&CK citation, because the curated map has no row for a
    technique named in ATT&CK's own words.
    """
    tools = KnowledgeTools(attack_db, Settings())
    out = tools.knowledge_technique(
        TechniqueInput(technique="Exploit Public-Facing Application")
    )
    assert out.attack_id == "T1190"
    assert out.tactics == ["initial-access"]
    assert out.platforms == ["Linux", "Windows"]
    assert out.mitre_url == "https://attack.mitre.org/techniques/T1190/"
