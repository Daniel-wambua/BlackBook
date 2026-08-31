"""Tests for the MITRE ATT&CK STIX adapter and the technique-tool enrichment
it enables."""

from pathlib import Path

import pytest

from blackbook.config import Settings, SourceConfig
from blackbook.ingestion.attack import MitreAttackAdapter
from blackbook.ingestion.pipeline import IngestionPipeline
from blackbook.knowledge.sources import get_chunk_excerpt
from blackbook.mcp.schemas import TechniqueInput
from blackbook.mcp.tools import KnowledgeTools
from blackbook.storage.database import Database

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture()
def attack_db(tmp_path):
    db = Database(str(tmp_path / "attack.db"))
    cfg = SourceConfig(
        id="attack", name="MITRE ATT&CK", type="git", authority="official",
        url="https://raw.githubusercontent.com/mitre-attack/attack-stix-data/"
            "master/enterprise-attack/enterprise-attack.json",
        ref="master",
    )
    adapter = MitreAttackAdapter(cfg, raw_dir=str(tmp_path))
    # The bundle is already present (fixture); fetching must stay offline.
    adapter.fetch = lambda force=False: None
    adapter._bundle_path = FIXTURES / "attack" / "bundle.json"
    # Register the source row exactly like the CLI does before ingesting.
    from blackbook.storage.models import Source

    with db.session():
        db.upsert_source(
            Source(
                source_id="attack", name="MITRE ATT&CK", authority="official",
                enabled=True, source_type="git", url=cfg.url,
            )
        )
    IngestionPipeline(db).run(adapter)
    return db


def _adapter(tmp_path) -> MitreAttackAdapter:
    cfg = SourceConfig(
        id="attack", name="MITRE ATT&CK", type="git", authority="official",
        url="https://raw.githubusercontent.com/mitre-attack/attack-stix-data/"
            "master/enterprise-attack/enterprise-attack.json",
    )
    adapter = MitreAttackAdapter(cfg, raw_dir=str(tmp_path))
    adapter.fetch = lambda force=False: None
    adapter._bundle_path = FIXTURES / "attack" / "bundle.json"
    return adapter


def test_attack_ingests_only_live_attack_patterns(tmp_path):
    docs = list(_adapter(tmp_path).iter_documents())
    by_id = {d.external_id: d for d in docs}
    assert set(by_id) == {"T1558.003", "T1190"}
    # revoked / deprecated / non-attack-pattern / no-ATT&CK-ID objects skipped
    assert "T9999" not in by_id and "T9998" not in by_id


def test_attack_document_shape(tmp_path):
    doc = {d.external_id: d for d in _adapter(tmp_path).iter_documents()}["T1558.003"]
    assert doc.title == "Steal or Forge Kerberos Tickets: Kerberoasting (T1558.003)"
    assert doc.url == "https://attack.mitre.org/techniques/T1558/003/"
    # Platforms land as lowercase category tags (platform hard filter works)
    assert "windows" in doc.categories and "iaas" in doc.categories
    # Tactics land as dashed ATT&CK slugs
    assert "credential-access" in doc.categories
    assert doc.metadata["tactics"] == ["credential-access"]
    assert doc.metadata["platforms"] == ["IaaS", "Windows"]
    # Body carries description and detection as citable sections
    assert "## Description" in doc.text and "## Detection" in doc.text
    assert any("Detection" in c.section_path for c in doc.chunks)
    # Modified date recorded for recency
    assert doc.metadata["date"] == "2025-04-02"


def test_attack_external_id_is_stable(attack_db):
    doc = attack_db.get_document_by_external("attack", "T1558.003")
    assert doc is not None
    assert attack_db.get_document_by_external("attack", "T9999") is None


def test_knowledge_technique_enriched_from_attack_source(attack_db):
    tools = KnowledgeTools(attack_db, Settings())
    out = tools.knowledge_technique(TechniqueInput(technique="kerberoast"))
    assert out.attack_id == "T1558.003"
    assert out.tactics == ["credential-access"]
    assert out.platforms == ["IaaS", "Windows"]
    assert out.mitre_url == "https://attack.mitre.org/techniques/T1558/003/"
    # The ATT&CK record is cited first, resolvable to a real indexed chunk
    attack_refs = [r for r in out.references if r.ref.source == "attack"]
    assert attack_refs, "expected an ATT&CK reference in the dossier"
    first = attack_refs[0]
    excerpt = get_chunk_excerpt(attack_db, first.ref.chunk_id)
    assert excerpt is not None
    assert "Kerberos" in excerpt.text


def test_knowledge_technique_without_attack_source_stays_clean(attack_db):
    # With an unmapped term there is nothing to enrich: fields stay empty.
    tools = KnowledgeTools(attack_db, Settings())
    out = tools.knowledge_technique(TechniqueInput(technique="not-a-technique"))
    assert out.attack_id is None
    assert out.tactics == [] and out.platforms == []
    assert out.mitre_url is None
