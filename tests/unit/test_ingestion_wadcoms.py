"""Tests for the WADComs front-matter cheat-sheet adapter."""

from pathlib import Path

from blackbook.config import SourceConfig, Settings
from blackbook.ingestion.wadcoms import WadcomsAdapter

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _adapter(**cfg_overrides) -> WadcomsAdapter:
    kwargs = dict(
        id="wadcoms", name="WADComs", type="git", authority="trusted",
        url="https://github.com/WADComs/WADComs.github.io.git", ref="master",
        content_root="_wadcoms", site_url="https://wadcoms.github.io",
    )
    kwargs.update(cfg_overrides)
    cfg = SourceConfig(**kwargs)
    adapter = WadcomsAdapter(cfg, raw_dir=str(FIXTURES))
    adapter._extract_root = FIXTURES / "github_wadcoms"
    return adapter


def test_wadcoms_document_shape():
    docs = list(_adapter().iter_documents())
    assert len(docs) == 1
    doc = docs[0]
    # The stem becomes a readable title; dashes to spaces
    assert doc.title == "Dementor"
    assert doc.external_id == "_wadcoms/Dementor.md"
    assert doc.url == "https://wadcoms.github.io/wadcoms/Dementor/"
    # OS values -> lowercase platform tags; attack types -> dashed slugs
    assert doc.categories == ["linux", "windows", "exploitation"]
    assert doc.metadata["services"] == ["RPC", "NTLM"]
    assert doc.metadata["items"] == ["Password", "Username"]


def test_wadcoms_front_matter_becomes_the_body():
    doc = list(_adapter().iter_documents())[0]
    # The whole payload lives in front matter; the adapter renders it
    assert "printer spooler" in doc.text
    assert "## Command" in doc.text
    assert "dementor.py -u john -p password123" in doc.text
    # The command is preserved as a code chunk
    code = [c for c in doc.chunks if c.kind == "code"]
    assert code and "dementor.py" in code[0].text
    # Meta line carries services / requirements / OS
    assert "Services: RPC, NTLM" in doc.text
    assert "Requires: Password, Username" in doc.text
    assert "## References" in doc.text


def test_wadcoms_file_without_front_matter_is_skipped(tmp_path):
    # A plain markdown file (no front matter) yields nothing, not a crash.
    content = tmp_path / "_wadcoms"
    content.mkdir()
    (content / "plain.md").write_text("# just markdown\n\nbody\n")
    adapter = _adapter()
    adapter._extract_root = tmp_path
    assert list(adapter.iter_documents()) == []


def test_wadcoms_in_default_sources():
    by_id = {s.id: s for s in Settings().sources}
    assert "wadcoms" in by_id
    assert by_id["wadcoms"].ref == "master"
    assert by_id["wadcoms"].content_root == "_wadcoms"
    assert by_id["wadcoms"].site_url == "https://wadcoms.github.io"
