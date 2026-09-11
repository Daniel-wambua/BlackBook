from pathlib import Path

from blackbook.config import SourceConfig
from blackbook.ingestion.github_md import GithubMarkdownAdapter


def test_github_adapter_supports_multiple_globs_and_source_categories(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "guide.md").write_text("# Guide\n\nMarkdown guidance.", encoding="utf-8")
    (root / "report.txt").write_text("Plain text report details.", encoding="utf-8")
    (root / "ignored.json").write_text('{"skip": true}', encoding="utf-8")
    adapter = GithubMarkdownAdapter(
        SourceConfig(
            id="mixed",
            name="Mixed",
            type="git",
            authority="unknown",
            url="https://github.com/example/repo.git",
            ref="master",
            include_glob="**/*.md,**/*.txt",
            categories=["bug-bounty", "hackerone"],
        ),
        raw_dir=str(tmp_path),
    )
    adapter._extract_root = root

    documents = list(adapter.iter_documents())
    assert {document.external_id for document in documents} == {"guide.md", "report.txt"}
    assert all(document.categories[:2] == ["bug-bounty", "hackerone"] for document in documents)
