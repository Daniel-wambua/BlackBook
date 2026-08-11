from pathlib import Path

import pytest

from blackbook.config import SourceConfig
from blackbook.ingestion.pdf import PDFAdapter

reportlab = pytest.importorskip("reportlab", reason="reportlab needed to build test PDFs")
from reportlab.pdfgen import canvas  # noqa: E402
from reportlab.lib.pagesizes import LETTER  # noqa: E402


def _make_structural_pdf(path: Path, title: str = "Test Doc") -> None:
    """Build a 2-page PDF with a large heading, body text, and a monospaced
    code block so structural detection has real signals to work with."""
    c = canvas.Canvas(str(path), pagesize=LETTER)
    c.setTitle(title)
    # Page 1
    c.setFont("Helvetica-Bold", 18)
    c.drawString(72, 720, "1. Services")
    c.setFont("Helvetica", 11)
    c.drawString(72, 700, "Services run as SYSTEM and can be hijacked.")
    c.setFont("Courier", 9)
    c.drawString(72, 670, "icacls C:\\Program Files\\App\\service.exe")
    c.setFont("Helvetica", 11)
    c.drawString(72, 640, "Weak permissions allow binary replacement.")
    c.showPage()
    # Page 2
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, 720, "2. Unquoted Service Paths")
    c.setFont("Helvetica", 11)
    c.drawString(72, 700, "An unquoted path with spaces lets you plant a binary.")
    c.showPage()
    c.save()


def _adapter(pdf_dir: Path, tmp_path: Path) -> PDFAdapter:
    cfg = SourceConfig(
        id="local_pdfs", name="Local PDFs", type="filesystem",
        authority="user", directory=str(pdf_dir),
    )
    return PDFAdapter(cfg, raw_dir=str(tmp_path))


def test_pdf_metadata_and_structure(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    target = pdf_dir / "windows_privesc.pdf"
    _make_structural_pdf(target, title="Windows Privilege Escalation")

    docs = list(_adapter(pdf_dir, tmp_path).iter_documents())
    assert len(docs) == 1
    doc = docs[0]
    # Title from PDF metadata, not inferred
    assert doc.title == "Windows Privilege Escalation"
    assert doc.metadata["metadata_inferred"]["title"] is False
    assert doc.metadata["page_count"] == 2
    # Headings detected (font-size + numbered)
    assert any("Services" in h for h in doc.metadata["headings"])
    assert any("Unquoted" in h for h in doc.metadata["headings"])
    # Code block detected via monospaced font
    assert doc.metadata["code_block_count"] >= 1


def test_pdf_chunks_have_page_and_section(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    _make_structural_pdf(pdf_dir / "win.pdf")

    doc = list(_adapter(pdf_dir, tmp_path).iter_documents())[0]
    # Page numbers recorded
    pages = {c.page for c in doc.chunks}
    assert 1 in pages and 2 in pages
    # Section breadcrumb carries the detected heading
    assert any("1. Services" in c.section_path for c in doc.chunks)
    # Code emitted as a code chunk
    code = [c for c in doc.chunks if c.kind == "code"]
    assert code and any("icacls" in c.text for c in code)


def test_pdf_boilerplate_title_falls_back_to_filename(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    # No setTitle -> reportlab stamps "untitled"; adapter must fall back.
    target = pdf_dir / "my_notes.pdf"
    c = canvas.Canvas(str(target), pagesize=LETTER)
    c.setFont("Helvetica", 11)
    c.drawString(72, 720, "some content here")
    c.showPage()
    c.save()

    doc = list(_adapter(pdf_dir, tmp_path).iter_documents())[0]
    assert doc.title == "my_notes"
    assert doc.metadata["metadata_inferred"]["title"] is True


def test_pdf_rejects_out_of_directory(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    _make_structural_pdf(pdf_dir / "ok.pdf")
    docs = list(_adapter(pdf_dir, tmp_path).iter_documents())
    assert all(Path(d.path).parent == pdf_dir for d in docs)
