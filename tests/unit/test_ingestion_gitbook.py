import json
from pathlib import Path

import httpx
import pytest

from blackbook.config import Settings, SourceConfig
from blackbook.ingestion import ADAPTER_REGISTRY, adapter_for
from blackbook.ingestion.gitbook import GitbookAdapter

BASE = "https://example.gitbook.io/notes"

# Mirrors the shape GitBook publishes: a title, a section heading, then one
# ``- [Title](url.md)`` line per page, optionally with a ": description".
LLMS_TXT = """# Example Notebook

## Example Notebook

- [Example Notebook](https://example.gitbook.io/notes/readme.md)
- [Keeping it Real](https://example.gitbook.io/notes/training/keeping-it-real.md): Some advice.
- [Cybersecurity Domains](https://example.gitbook.io/notes/training/cyber-domains.md)
- [Cybersecurity Domains](https://example.gitbook.io/notes/training/cyber-domains.md)
- [Elsewhere](https://other.gitbook.io/notes/x.md)
- [Outside the space](https://example.gitbook.io/other/x.md)
"""

PAGE = """\
> For the complete documentation index, see [llms.txt](https://example.gitbook.io/notes/llms.txt). Markdown versions of documentation pages are available by appending `.md` to the URL.

# Cybersecurity Domains

### Governance

{% hint style="info" %}
Compliance work is a domain of its own.
{% endhint %}

<div data-full-width="true"><figure><img src="https://files.gitbook.io/x.png"></figure></div>

See [the index](/notes/training/keeping-it-real) for more.

# Agent Instructions

This documentation is published with GitBook. Perform an HTTP GET request on
the page URL with the `ask` query parameter.
"""


def _adapter(tmp_path: Path, **overrides) -> GitbookAdapter:
    kwargs = dict(
        id="notes",
        name="Notes",
        type="gitbook",
        authority="unknown",
        url=BASE,
        categories=["notes"],
    )
    kwargs.update(overrides)
    return GitbookAdapter(SourceConfig(**kwargs), raw_dir=str(tmp_path))


def _seed(tmp_path: Path, adapter: GitbookAdapter, url: str, body: str) -> None:
    cache = adapter._cache_path(url)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(body, encoding="utf-8")


def test_gitbook_adapter_requires_a_url(tmp_path):
    with pytest.raises(ValueError, match="requires a url"):
        GitbookAdapter(
            SourceConfig(id="notes", name="Notes", type="gitbook", authority="unknown"),
            raw_dir=str(tmp_path),
        )


def test_gitbook_index_keeps_only_in_space_links(tmp_path):
    adapter = _adapter(tmp_path)
    entries = adapter._parse_index(LLMS_TXT)

    assert [e["url"] for e in entries] == [
        f"{BASE}/readme.md",
        f"{BASE}/training/keeping-it-real.md",
        f"{BASE}/training/cyber-domains.md",
    ]
    assert entries[0]["title"] == "Example Notebook"
    # The ": description" suffix is not part of the title.
    assert entries[1]["title"] == "Keeping it Real"


def test_gitbook_clean_markdown_strips_chrome_but_keeps_content(tmp_path):
    adapter = _adapter(tmp_path)
    cleaned = adapter._clean_markdown(PAGE)

    # GitBook's per-page chrome is gone.
    assert "Markdown versions of documentation pages" not in cleaned
    assert "Agent Instructions" not in cleaned
    assert "Perform an HTTP GET request" not in cleaned
    assert "{% hint" not in cleaned and "{% endhint" not in cleaned
    assert "<img" not in cleaned and "<figure" not in cleaned
    # The author's text, the hint's body and the section heading survive.
    assert "# Cybersecurity Domains" in cleaned
    assert "Compliance work is a domain of its own." in cleaned
    assert "### Governance" in cleaned
    # Site-relative links are made absolute so citations resolve.
    assert f"]({BASE}/notes/training/keeping-it-real)" in cleaned


def test_gitbook_iter_documents_from_cache(tmp_path):
    adapter = _adapter(tmp_path)
    workdir = tmp_path / "notes"
    workdir.mkdir()
    url = f"{BASE}/training/cyber-domains.md"
    (workdir / "index_pages.json").write_text(
        json.dumps([{"title": "Cybersecurity Domains", "url": url}]), encoding="utf-8"
    )
    _seed(tmp_path, adapter, url, PAGE)

    documents = list(adapter.iter_documents())
    assert len(documents) == 1
    doc = documents[0]
    assert doc.title == "Cybersecurity Domains"
    assert doc.url == f"{BASE}/training/cyber-domains"
    assert doc.external_id == url
    # The page's own slug becomes the title, so the breadcrumb keeps only the
    # configured category plus the URL grouping.
    assert doc.categories == ["notes", "Training"]
    assert doc.chunks
    assert any(chunk.kind == "code" or "Compliance" in chunk.text for chunk in doc.chunks)
    assert "Agent Instructions" not in doc.text


def test_gitbook_skips_soft_404_pages(tmp_path):
    adapter = _adapter(tmp_path)
    workdir = tmp_path / "notes"
    workdir.mkdir()
    url = f"{BASE}/sections/cloud.md"
    (workdir / "index_pages.json").write_text(
        json.dumps([{"title": "Cloud", "url": url}]), encoding="utf-8"
    )
    _seed(
        tmp_path,
        adapter,
        url,
        "# Page Not Found\n\nThe URL `sections/cloud` does not exist. "
        "This page may have been moved, renamed, or deleted.\n",
    )

    assert list(adapter.iter_documents()) == []


def test_gitbook_categories_drop_the_page_slug(tmp_path):
    adapter = _adapter(tmp_path)
    assert adapter._categories_from_url(f"{BASE}/readme.md") == []
    assert adapter._categories_from_url(f"{BASE}/training/cyber-domains.md") == ["Training"]
    assert adapter._categories_from_url(f"{BASE}/a/b/page-name.md") == ["A", "B"]


def test_gitbook_fetch_caches_index_and_pages(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path)
    page_url = f"{BASE}/training/cyber-domains.md"

    def get(self, url, headers=None):
        if url.endswith("/llms.txt"):
            return httpx.Response(200, content=LLMS_TXT.encode(), request=httpx.Request("GET", url))
        return httpx.Response(200, content=PAGE.encode(), request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.Client, "get", get)
    adapter.fetch()

    workdir = tmp_path / "notes"
    assert (workdir / "llms.txt").read_text(encoding="utf-8") == LLMS_TXT
    assert (workdir / "index_pages.json").is_file()
    assert adapter._cache_path(page_url).is_file()
    assert len(list(adapter.iter_documents())) == 3

    # A second fetch with no force re-uses the cache and makes no requests.
    requested: list[str] = []

    def boom(self, url, headers=None):
        requested.append(url)
        raise AssertionError(f"unexpected refetch of {url}")

    monkeypatch.setattr(httpx.Client, "get", boom)
    adapter.fetch()
    assert requested == []


def test_new_sources_are_registered_and_typed():
    settings = Settings()
    by_id = {source.id: source for source in settings.sources}
    expected = {
        "highon_coffee",
        "martian1337_notes",
        "pentest_book",
        "shirajuki_blog",
        "arcanum_devops",
    }
    assert expected <= set(by_id)

    for source_id in ("martian1337_notes", "pentest_book"):
        source = by_id[source_id]
        assert source.type == "gitbook"
        assert source.authority == "unknown"
        assert isinstance(adapter_for(source), GitbookAdapter)
    assert ADAPTER_REGISTRY["pentest_book"] is GitbookAdapter

    # The crawl- and repo-backed entries keep their existing adapters.
    from blackbook.ingestion.github_md import GithubMarkdownAdapter
    from blackbook.ingestion.website import WebsiteAdapter

    assert by_id["highon_coffee"].path_prefix == "/blog"
    assert isinstance(adapter_for(by_id["highon_coffee"]), WebsiteAdapter)
    assert isinstance(adapter_for(by_id["arcanum_devops"]), WebsiteAdapter)
    assert isinstance(adapter_for(by_id["shirajuki_blog"]), GithubMarkdownAdapter)
    assert by_id["shirajuki_blog"].content_root == "src/content/blog"
