"""Tests for the GTFOBins YAML adapter."""

from pathlib import Path

from blackbook.config import SourceConfig
from blackbook.ingestion.gtfobins import GtfoBinsAdapter

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _adapter(**cfg_overrides) -> GtfoBinsAdapter:
    kwargs = dict(
        id="gtfobins", name="GTFOBins", type="git", authority="trusted",
        url="https://github.com/GTFOBins/GTFOBins.github.io.git", ref="master",
        content_root="_gtfobins", site_url="https://gtfobins.github.io",
    )
    kwargs.update(cfg_overrides)
    cfg = SourceConfig(**kwargs)
    adapter = GtfoBinsAdapter(cfg, raw_dir=str(FIXTURES))
    adapter._extract_root = FIXTURES / "github_gtfobins"
    return adapter


def test_alias_files_fold_into_target():
    docs = {d.title: d for d in _adapter().iter_documents()}
    # The alias-only file (awk -> mawk) is not a document of its own...
    assert set(docs) == {"find", "mawk"}
    # ...but the target advertises the alias so "awk" queries still land.
    assert docs["mawk"].metadata["aliases"] == ["awk"]
    assert "awk" in docs["mawk"].text


def test_gtfobins_document_shape():
    doc = {d.title: d for d in _adapter().iter_documents()}["find"]
    assert doc.external_id == "_gtfobins/find"
    assert doc.url == "https://gtfobins.github.io/find/"
    # Unix tags make platform filters hit
    assert doc.categories == ["linux", "unix"]
    assert doc.metadata["functions"] == [
        "file-read", "file-write", "shell", "sudo-enabled", "suid-enabled",
    ]
    assert set(doc.metadata["contexts"]) == {"sudo", "suid", "unprivileged"}


def test_gtfobins_rendered_structure():
    doc = {d.title: d for d in _adapter().iter_documents()}["find"]
    # Each function becomes a citable section
    assert "## shell" in doc.text and "## file-read" in doc.text
    # Code preserved as code chunks
    code = [c for c in doc.chunks if c.kind == "code"]
    assert any("find . -exec /bin/sh" in c.text for c in code)
    assert any("sudo find" in c.text for c in code)
    # The suid context carries its own -p variant
    assert "find . -exec /bin/sh -p" in doc.text
    # Comments survive for context
    assert "format string" in doc.text


def test_no_site_url_falls_back_to_github_blob():
    adapter = _adapter(site_url=None)
    doc = {d.title: d for d in adapter.iter_documents()}["find"]
    assert doc.url == (
        "https://github.com/GTFOBins/GTFOBins.github.io"
        "/blob/master/_gtfobins/find"
    )
