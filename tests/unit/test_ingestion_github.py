"""Tests for the generic GitHub-markdown adapter and the sources it enables
(PayloadsAllTheThings, The Hacker Recipes, GTFOBins, Internal All The Things,
Moamen Basel's HTB writeups)."""

from pathlib import Path

from blackbook.config import SourceConfig, Settings
from blackbook.ingestion import adapter_for, ADAPTER_REGISTRY
from blackbook.ingestion.github_base import GithubTarballAdapter
from blackbook.ingestion.github_md import GithubMarkdownAdapter

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _adapter(fixture: Path, **cfg_overrides) -> GithubMarkdownAdapter:
    kwargs = dict(
        id="testsrc", name="Test Source", type="git", authority="trusted",
        url="https://github.com/example/repo.git", ref="master",
    )
    kwargs.update(cfg_overrides)
    cfg = SourceConfig(**kwargs)
    adapter = GithubMarkdownAdapter(cfg, raw_dir=str(FIXTURES))
    adapter._extract_root = fixture
    return adapter


def test_hacker_recipes_content_root_and_site_url():
    adapter = _adapter(
        FIXTURES / "github_recipes",
        content_root="docs",
        site_url="https://thehacker.recipes",
    )
    docs = list(adapter.iter_documents())
    # Root README.md is outside the content root: not ingested.
    assert len(docs) == 1
    doc = docs[0]
    assert doc.title == "Kerberoasting"
    # categories come from the path *below* the content root
    assert "Ad" in doc.categories and "Movement" in doc.categories
    assert "Docs" not in doc.categories  # plumbing stripped
    # URL maps to the published site, .md dropped
    assert doc.url == "https://thehacker.recipes/ad/movement/kerberos/kerberoast"
    # front matter stripped from the body
    assert "description: Kerberos recipe front matter" not in doc.text
    assert any("GetUserSPNs" in c.text for c in doc.chunks)


def test_no_site_url_falls_back_to_github_blob():
    adapter = _adapter(FIXTURES / "github_recipes" / "docs")
    docs = list(adapter.iter_documents())
    assert docs
    assert docs[0].url.startswith(
        "https://github.com/example/repo/blob/master/"
    )


def test_repo_slug_parsing():
    adapter = _adapter(FIXTURES / "github_recipes")
    assert adapter._repo_slug() == ("example", "repo")


def test_default_sources_include_new_github_sources():
    s = Settings()
    ids = {src.id for src in s.sources}
    assert {"payloads", "hacker_recipes", "gtfobins", "lolbas", "attack"} <= ids
    by_id = {src.id: src for src in s.sources}
    assert by_id["gtfobins"].content_root == "_gtfobins"
    assert by_id["gtfobins"].site_url == "https://gtfobins.github.io"
    assert by_id["hacker_recipes"].ref == "main"
    assert by_id["hacker_recipes"].content_root == "docs"
    assert by_id["payloads"].url.endswith("PayloadsAllTheThings.git")
    assert by_id["attack"].url.startswith(
        "https://raw.githubusercontent.com/mitre-attack/"
    )


def test_adapter_registry_maps_new_sources():
    from blackbook.ingestion.gtfobins import GtfoBinsAdapter
    from blackbook.ingestion.lolbas import LolbasAdapter

    assert ADAPTER_REGISTRY["payloads"] is GithubMarkdownAdapter
    assert ADAPTER_REGISTRY["hacker_recipes"] is GithubMarkdownAdapter
    assert ADAPTER_REGISTRY["gtfobins"] is GtfoBinsAdapter
    assert ADAPTER_REGISTRY["lolbas"] is LolbasAdapter
    adapter = adapter_for(
        SourceConfig(
            id="gtfobins", name="GTFOBins", type="git",
            url="https://github.com/GTFOBins/GTFOBins.github.io.git", ref="master",
        )
    )
    assert isinstance(adapter, GtfoBinsAdapter)
    assert isinstance(adapter, GithubTarballAdapter)


def test_repo_slug_rejects_bad_url():
    import pytest

    adapter = _adapter(FIXTURES / "github_recipes", url="not-a-url")
    with pytest.raises(ValueError):
        adapter._repo_slug()


def test_htb_writeups_permalink_and_exclude_glob():
    adapter = _adapter(
        FIXTURES / "github_htb",
        site_url="https://www.moamenbasel.com/htb-writeups",
        exclude_glob="templates/**, 0xdf-htb-machines.md",
    )
    docs = {d.external_id: d for d in adapter.iter_documents()}
    # excluded plumbing / link-index files are not ingested
    assert "templates/machine-template.md" not in docs
    assert "0xdf-htb-machines.md" not in docs
    assert set(docs) == {"machines/README.md", "machines/easy/Code/README.md"}

    code = docs["machines/easy/Code/README.md"]
    # the Jekyll permalink wins over the path-derived URL (canonical case:
    # directory is "Code", permalink is lowercase)
    assert code.url == "https://www.moamenbasel.com/htb-writeups/machines/easy/code/"
    # title from the h1 (not "README"), categories from the path — including
    # the machine's own directory, so difficulty and machine name both tag
    # every chunk
    assert code.title == "Code"
    assert code.categories == ["Machines", "Easy", "Code"]
    # front matter never reaches the indexed body
    assert "permalink" not in code.text and "nav_order" not in code.text

    # a section README with a permalink maps to the directory URL
    assert docs["machines/README.md"].url == (
        "https://www.moamenbasel.com/htb-writeups/machines/"
    )


def test_internal_all_the_things_readme_collapses_to_site_root():
    adapter = _adapter(
        FIXTURES / "github_iatt",
        content_root="docs",
        site_url="https://swisskyrepo.github.io/InternalAllTheThings",
    )
    docs = {d.external_id: d for d in adapter.iter_documents()}
    # repo landing README (no Jekyll permalink) maps to the site root
    assert docs["docs/README.md"].url == (
        "https://swisskyrepo.github.io/InternalAllTheThings/"
    )
    assert docs["docs/active-directory/pwd-spraying.md"].url == (
        "https://swisskyrepo.github.io/InternalAllTheThings/active-directory/pwd-spraying"
    )


def test_permalink_ignored_without_site_url():
    # No site_url: the permalink is site-relative and unusable, so the
    # citation falls back to the always-resolvable GitHub blob URL.
    adapter = _adapter(FIXTURES / "github_htb")
    docs = {d.external_id: d for d in adapter.iter_documents()}
    assert docs["machines/easy/Code/README.md"].url == (
        "https://github.com/example/repo/blob/master/machines/easy/Code/README.md"
    )


def test_default_sources_include_internal_and_htb_writeups():
    s = Settings()
    by_id = {src.id: src for src in s.sources}

    iatt = by_id["internal_all_the_things"]
    assert iatt.ref == "main"
    assert iatt.content_root == "docs"
    assert iatt.site_url == "https://swisskyrepo.github.io/InternalAllTheThings"

    htb = by_id["htb_writeups"]
    assert htb.ref == "main"
    assert htb.site_url == "https://www.moamenbasel.com/htb-writeups"
    for pattern in ("templates/**", "0xdf-htb-machines.md"):
        assert pattern in htb.exclude_glob

    # both are pure configuration: they resolve to the generic adapter
    # through the git-type fallback, no registry entry needed
    assert isinstance(adapter_for(iatt), GithubMarkdownAdapter)
    assert isinstance(adapter_for(htb), GithubMarkdownAdapter)
