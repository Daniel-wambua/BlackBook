from pathlib import Path

from blackbook.config import SourceConfig
from blackbook.ingestion.zerodf import ZeroDFAdapter

FIXTURE = Path(__file__).parent.parent / "fixtures" / "zerodf"


def make_adapter():
    cfg = SourceConfig(id="0xdf", name="0xdf", type="website", authority="trusted")
    adapter = ZeroDFAdapter(cfg, raw_dir=str(FIXTURE.parent.parent))
    return adapter


def test_zerodf_extracts_structured_metadata():
    adapter = make_adapter()
    # parse the fixture post directly
    doc = adapter._parse_post(FIXTURE / "htb-forest.html")
    assert doc is not None
    assert doc.title == "HTB: Forest"
    md = doc.metadata
    assert md["machine_name"] == "Forest"
    assert md["kind"] == "htb"
    assert md["os"] == "Windows"
    assert md["difficulty"] == "Easy"
    assert md["date"] == "2019-12-21"
    assert md["creator"] == "egre55"
    # summary captured from og:description
    assert md["summary"] and "DCSync" in md["summary"]
    # attack-chain narrative captured from H2 headings
    assert "Recon" in md["sections"]
    assert "Shell as svc-alfresco" in md["sections"]
    assert "Shell as SYSTEM" in md["sections"]


def test_zerodf_inferred_signals_marked():
    adapter = make_adapter()
    doc = adapter._parse_post(FIXTURE / "htb-forest.html")
    md = doc.metadata
    # services/techniques/tools are inferred and marked as such
    assert md["metadata_inferred"]["services"] is True
    assert "kerberos" in md["services"] or "smb" in md["services"]
    assert "kerberoasting" in md["techniques"]


def test_zerodf_chunks_have_sections_and_code():
    adapter = make_adapter()
    doc = adapter._parse_post(FIXTURE / "htb-forest.html")
    assert doc.chunks
    # nmap command should be present in some chunk
    assert any("nmap" in c.text for c in doc.chunks)
    # section breadcrumb present
    assert any("Recon" in c.section_path for c in doc.chunks)


def test_zerodf_title_split():
    name, kind = ZeroDFAdapter._split_title("HTB: Forest")
    assert name == "Forest" and kind == "htb"
    name, kind = ZeroDFAdapter._split_title("Some Random Post")
    assert name is None and kind == "unknown"
