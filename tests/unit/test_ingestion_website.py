from pathlib import Path

from blackbook.config import Settings, SourceConfig
from blackbook.ingestion import ADAPTER_REGISTRY, adapter_for
from blackbook.ingestion.website import WebsiteAdapter


def test_website_sources_are_registered_and_scoped():
    settings = Settings()
    by_id = {source.id: source for source in settings.sources}
    expected = {
        "owasp_wstg",
        "owasp_asvs",
        "owasp_api_security",
        "bugbounty_cheatsheet",
        "portswigger",
        "google_bug_hunters",
        "hackerone_hacktivity",
        "bugcrowd_vrt",
        "github_security_lab",
    }
    assert expected <= set(by_id)
    assert by_id["portswigger"].path_prefix == "/web-security"
    assert by_id["google_bug_hunters"].path_prefix == "/learn"
    assert by_id["github_security_lab"].path_prefix == "/research"
    assert isinstance(adapter_for(by_id["portswigger"]), WebsiteAdapter)
    assert ADAPTER_REGISTRY["portswigger"] is WebsiteAdapter


def test_website_adapter_rejects_external_and_out_of_scope_links(tmp_path: Path):
    config = SourceConfig(
        id="docs",
        name="Docs",
        type="website",
        authority="trusted",
        url="https://example.test/docs",
        path_prefix="/docs",
    )
    adapter = WebsiteAdapter(config, raw_dir=str(tmp_path))

    assert adapter._canonical_url("https://example.test/docs/page#section") == (
        "https://example.test/docs/page"
    )
    assert adapter._canonical_url("https://example.test/other") is None
    assert adapter._canonical_url("https://other.test/docs/page") is None
    assert adapter._canonical_url("https://example.test/docs/file.pdf") is None


def test_website_adapter_parses_cached_html(tmp_path: Path):
    config = SourceConfig(
        id="docs",
        name="Docs",
        type="website",
        authority="trusted",
        url="https://example.test/docs",
        path_prefix="/docs",
    )
    adapter = WebsiteAdapter(config, raw_dir=str(tmp_path))
    workdir = tmp_path / "docs"
    pages = workdir / "pages"
    pages.mkdir(parents=True)
    url = "https://example.test/docs/page"
    cache_path = adapter._cache_path(url)
    cache_path.write_text(
        "<html><title>Example</title><body><main>"
        "<h1>Example</h1><p>Useful guidance.</p>"
        "<pre>curl https://example.test</pre>"
        "</main></body></html>",
        encoding="utf-8",
    )
    (workdir / "index_urls.json").write_text(f'["{url}"]', encoding="utf-8")

    documents = list(adapter.iter_documents())
    assert len(documents) == 1
    assert documents[0].title == "Example"
    assert "Useful guidance." in documents[0].text
    assert any(chunk.kind == "code" for chunk in documents[0].chunks)
