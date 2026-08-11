from pathlib import Path

from blackbook.config import SourceConfig
from blackbook.ingestion.hacktricks import HackTricksAdapter

FIXTURE = Path(__file__).parent.parent / "fixtures" / "hacktricks"


def make_adapter():
    cfg = SourceConfig(id="hacktricks", name="HackTricks", type="git", authority="trusted")
    adapter = HackTricksAdapter(cfg, raw_dir=str(FIXTURE.parent.parent))
    # Point the adapter at the fixture as the extract root.
    adapter._extract_root = FIXTURE
    return adapter


def test_hacktricks_parses_markdown_and_hierarchy():
    adapter = make_adapter()
    docs = list(adapter.iter_documents())
    assert len(docs) == 1
    doc = docs[0]
    assert doc.title == "Kerberoasting"
    # category derived from directory name
    assert "Active Directory Methodology" in doc.categories
    # url mapped to the published book
    assert doc.url.startswith("https://book.hacktricks.xyz/")
    assert doc.url.endswith("kerberoasting")


def test_hacktricks_chunks_preserve_section_paths():
    adapter = make_adapter()
    doc = list(adapter.iter_documents())[0]
    paths = [c.section_path for c in doc.chunks]
    # Chunks under "Enumeration" carry the nested breadcrumb
    assert any("Enumeration" in p for p in paths)
    # code block captured as a code chunk
    code = [c for c in doc.chunks if c.kind == "code"]
    assert code and any("GetUserSPNs" in c.text for c in code)


def test_hacktricks_front_matter_stripped():
    adapter = make_adapter()
    doc = list(adapter.iter_documents())[0]
    # YAML front matter must not leak into the body/chunks
    assert "description: Kerberoasting reference" not in doc.text
