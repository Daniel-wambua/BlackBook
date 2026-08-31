"""Tests for the LOLBAS YAML adapter."""

from pathlib import Path

from blackbook.config import SourceConfig
from blackbook.ingestion.lolbas import LolbasAdapter

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _adapter() -> LolbasAdapter:
    cfg = SourceConfig(
        id="lolbas", name="LOLBAS", type="git", authority="trusted",
        url="https://github.com/LOLBAS-Project/LOLBAS.git", ref="master",
        site_url="https://lolbas-project.github.io",
    )
    adapter = LolbasAdapter(cfg, raw_dir=str(FIXTURES))
    adapter._extract_root = FIXTURES / "lolbas"
    return adapter


def test_lolbas_parses_yaml_entry():
    docs = list(_adapter().iter_documents())
    assert len(docs) == 1
    doc = docs[0]
    assert doc.title == "Certutil.exe"
    assert doc.external_id == "yml/OSBinaries/Certutil.yml"
    # Category from the repo layout, "yml/" plumbing stripped
    assert doc.categories == ["Osbinaries"]
    # Description preserved in the body
    assert "handling certificates" in doc.text
    # ATT&CK IDs collected from commands
    assert set(doc.metadata["mitre_ids"]) == {"T1105", "T1027.013"}
    # Full paths recorded
    assert "C:\\Windows\\System32\\certutil.exe" in doc.metadata["full_path"]


def test_lolbas_url_maps_to_site():
    doc = list(_adapter().iter_documents())[0]
    assert doc.url == "https://lolbas-project.github.io/osbinaries/certutil/"


def test_lolbas_commands_rendered_with_structure():
    doc = list(_adapter().iter_documents())[0]
    # Every command survives with its ATT&CK mapping
    assert "certutil.exe -urlcache -f" in doc.text
    assert "certutil -encode" in doc.text
    assert "T1027.013" in doc.text
    # Detection pointers preserved
    assert "https://example.com/sigma/certutil_download.yml" in doc.text
    # Chunks carry section breadcrumbs
    assert any("Commands" in c.section_path for c in doc.chunks)
    code = [c for c in doc.chunks if c.kind == "code"]
    assert code and any("urlcache" in c.text for c in code)
