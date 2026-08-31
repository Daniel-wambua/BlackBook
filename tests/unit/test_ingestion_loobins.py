"""Tests for the LOOBins macOS YAML adapter."""

from pathlib import Path

from blackbook.config import SourceConfig, Settings
from blackbook.ingestion.loobins import LooBinsAdapter

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _adapter(**cfg_overrides) -> LooBinsAdapter:
    kwargs = dict(
        id="loobins", name="LOOBins", type="git", authority="trusted",
        url="https://github.com/infosecB/LOOBins.git", ref="main",
        content_root="LOOBins", site_url="https://loobins.io",
    )
    kwargs.update(cfg_overrides)
    cfg = SourceConfig(**kwargs)
    adapter = LooBinsAdapter(cfg, raw_dir=str(FIXTURES))
    adapter._extract_root = FIXTURES / "github_loobins"
    return adapter


def test_loobins_document_shape():
    docs = list(_adapter().iter_documents())
    assert len(docs) == 1
    doc = docs[0]
    assert doc.title == "osascript"
    assert doc.external_id == "LOOBins/osascript.yml"
    assert doc.url == "https://loobins.io/binaries/osascript/"
    # macOS tag plus slugified ATT&CK tactics across all use cases
    assert doc.categories == [
        "macos", "collection", "credential-access", "discovery",
    ]
    assert doc.metadata["paths"] == ["/usr/bin/osascript"]
    assert doc.metadata["tactics"] == [
        "Collection", "Credential Access", "Discovery",
    ]
    assert "clipboard" in doc.metadata["tags"]


def test_loobins_rendered_structure():
    doc = list(_adapter().iter_documents())[0]
    assert "## Use Cases" in doc.text
    # Each use case is its own citable section with its code
    assert "### Use the osascript binary to gather sensitive clipboard data" in doc.text
    code = [c for c in doc.chunks if c.kind == "code"]
    assert any("the clipboard" in c.text for c in code)
    assert any("system info" in c.text for c in code)
    # Binary paths and detection pointers survive
    assert "## Paths" in doc.text and "`/usr/bin/osascript`" in doc.text
    assert "## Detection" in doc.text
    # Tactics meta line keeps the official ATT&CK names
    assert "Tactics: Collection, Credential Access" in doc.text


def test_loobins_no_site_url_falls_back_to_blob():
    adapter = _adapter(site_url=None)
    doc = list(adapter.iter_documents())[0]
    assert doc.url == (
        "https://github.com/infosecB/LOOBins/blob/main/LOOBins/osascript.yml"
    )


def test_loobins_in_default_sources():
    by_id = {s.id: s for s in Settings().sources}
    assert "loobins" in by_id
    assert by_id["loobins"].ref == "main"
    assert by_id["loobins"].content_root == "LOOBins"
    assert by_id["loobins"].site_url == "https://loobins.io"
