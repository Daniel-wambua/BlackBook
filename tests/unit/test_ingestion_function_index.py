import json
from pathlib import Path

import httpx
import pytest

from blackbook.config import Settings, SourceConfig
from blackbook.ingestion import ADAPTER_REGISTRY, adapter_for
from blackbook.ingestion.function_index import FunctionIndexAdapter

BASE = "https://likeagod.revers.engineering/nt"

# Mirrors the site's index: a search box, a flat list of items ordered by name
# (here deliberately alphabetically mixed so a prefix filter is observable),
# and a trailing inline script that contains no anchors.
INDEX = """\
<html><head>
<link rel="stylesheet" href="style.css">
</head><body>
<input type="text" id="search" onkeyup="searchFunction()" placeholder="Search for functions..">
<ul id="functionList">
<li><a href='aa11bb22cc33dd44ee55ff6600112233.html'>ADD_MAP_REGISTERS</a></li>
<li><a href='bb11bb22cc33dd44ee55ff6600112233.html'>AdtpAppendString</a></li>
<li><a href='cc11bb22cc33dd44ee55ff6600112233.html'>IoCreateDevice</a></li>
<li><a href='dd11bb22cc33dd44ee55ff6600112233.html'>MmMapIoSpace</a></li>
<li><a href='ee11bb22cc33dd44ee55ff6600112233.html'>NtCreateFile</a></li>
<li><a href='ff11bb22cc33dd44ee55ff6600112233.html'>ZwClose</a></li>
<li><a href='11000000000000000000000000000000.html'>NtCreateFile</a></li>
<li><a href='https://other.example/x.html'>Elsewhere</a></li>
<li><a href='style.css'>Stylesheet</a></li>
<li><a href='../nnyt/'>Parent</a></li>
</ul>
<script>
function searchFunction() { var li = document.getElementById("functionList"); }
</script>
</body></html>
"""


def _page(name: str, callees: str = "", callers: list[str] | None = None) -> str:
    refs = "".join(
        f"<a href='{i:032x}.html'>{c}</a><br>" for i, c in enumerate(callers or [])
    )
    body = (
        "<head><link rel='stylesheet' href='style.css'></head><body>\n"
        f"<h2>{name}</h2>\n"
        f"<pre>NTSTATUS __stdcall {name}(UINT64 a1){{\n"
        f"  int v3;\n  {callees}"
        "  return 0;\n}</pre><h3>Referenced by:</h3>\n"
        + (refs if refs else "No references.")
    )
    return body


def _adapter(tmp_path: Path, **overrides) -> FunctionIndexAdapter:
    kwargs = dict(
        id="nt_functions",
        name="NT",
        type="function_index",
        authority="unknown",
        url=BASE + "/",
        include_prefixes=["Nt", "Mm", "Io", "Zw"],
        max_files=8000,
        categories=["windows", "kernel"],
    )
    kwargs.update(overrides)
    return FunctionIndexAdapter(SourceConfig(**kwargs), raw_dir=str(tmp_path))


def _seed(tmp_path: Path, adapter: FunctionIndexAdapter, url: str, body: str) -> None:
    cache = adapter._cache_path(url)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(body, encoding="utf-8")


def test_function_index_adapter_requires_a_url(tmp_path):
    with pytest.raises(ValueError, match="requires a url"):
        FunctionIndexAdapter(
            SourceConfig(id="nt", name="NT", type="function_index", authority="unknown"),
            raw_dir=str(tmp_path),
        )


def test_index_keeps_only_same_origin_item_pages(tmp_path):
    adapter = _adapter(tmp_path)
    entries = adapter._parse_index(INDEX)

    names = [e["name"] for e in entries]
    # The external link, the stylesheet and the parent directory are all out;
    # the ALL_CAPS macro and the Adtp* name are out on the prefix filter.
    assert names == [
        "IoCreateDevice",
        "MmMapIoSpace",
        "NtCreateFile",
        "ZwClose",
        "NtCreateFile",
    ]
    assert entries[0]["url"] == f"{BASE}/cc11bb22cc33dd44ee55ff6600112233.html"


def test_index_prefix_filter_selects_before_the_cap(tmp_path):
    """The cap must bound the selected set, not the index's alphabetical order."""
    adapter = _adapter(tmp_path, include_prefixes=["Nt", "Mm"], max_files=2)
    entries = adapter._parse_index(INDEX)
    assert [e["name"] for e in entries] == ["MmMapIoSpace", "NtCreateFile", "NtCreateFile"]

    # With no prefixes configured every item page is eligible — including the
    # Adtp* and ALL_CAPS entries the prefix filter above dropped.
    unfiltered = _adapter(tmp_path, include_prefixes=[])
    assert len(unfiltered._parse_index(INDEX)) == 7


def test_page_becomes_pseudocode_plus_callers(tmp_path):
    adapter = _adapter(tmp_path)
    page = _page(
        "MmMapIoSpace",
        callees="<a href='4198ee1c5725bd242ae74a8277d8031f.html'>MmMapIoSpaceEx</a>(a1);\n",
        callers=["MiMapPages", "VfFilterAttach"],
    )
    title, code, references = adapter._parse_page(page, "fallback")

    assert title == "MmMapIoSpace"
    # The inline call link is unwrapped to its name, not dropped.
    assert "MmMapIoSpaceEx(a1);" in code
    assert "<a" not in code and "</pre>" not in code
    assert "NTSTATUS __stdcall MmMapIoSpace" in code
    assert references == ["MiMapPages", "VfFilterAttach"]


def test_less_than_comparisons_survive_code_extraction(tmp_path):
    """A literal ``<`` is not an HTML tag and must not be stripped as one.

    Decompiled C is full of ``if( a < b )``. A permissive tag pattern deletes
    everything from the ``<`` to the next real tag, silently corrupting the
    pseudocode, so the code path uses an allowlist of tag names instead.
    """
    adapter = _adapter(tmp_path)
    page = (
        "<h2>ZwClose</h2>\n<pre>NTSTATUS ZwClose(HANDLE h){\n"
        "  if( (unsigned __int8)t < (unsigned int)Max )\n"
        "    v = a<b && c>d;\n"
        "  if( n <i )\n"
        "    <a href='aa.html'>ObCloseHandle</a>(h);\n"
        "  return 0;\n}</pre><h3>Referenced by:</h3>\nNo references."
    )
    _title, code, _refs = adapter._parse_page(page, "ZwClose")

    assert "if( (unsigned __int8)t < (unsigned int)Max )" in code
    assert "v = a<b && c>d;" in code
    assert "if( n <i )" in code
    # ...while the real markup is still removed.
    assert "ObCloseHandle(h);" in code
    assert "<a href" not in code

    # Markup that IS in the allowlist still goes, in both spellings.
    assert adapter._parse_page(
        "<pre>x<br/>y<BR>z<span class='k'>w</span></pre>", "f"
    )[1] == "x\ny\nzw"


def test_iter_documents_from_cache(tmp_path):
    adapter = _adapter(tmp_path)
    workdir = tmp_path / "nt_functions"
    workdir.mkdir()
    mm_url = f"{BASE}/dd11bb22cc33dd44ee55ff6600112233.html"
    nt_url = f"{BASE}/ee11bb22cc33dd44ee55ff6600112233.html"
    (workdir / "index_items.json").write_text(
        json.dumps(
            [
                {"name": "MmMapIoSpace", "url": mm_url},
                {"name": "NtCreateFile", "url": nt_url},
            ]
        ),
        encoding="utf-8",
    )
    _seed(tmp_path, adapter, mm_url, _page("MmMapIoSpace"))
    _seed(tmp_path, adapter, nt_url, _page("NtCreateFile", callers=["IoCreateDevice"]))

    documents = list(adapter.iter_documents())
    assert [d.title for d in documents] == ["MmMapIoSpace", "NtCreateFile"]

    leaf = documents[0]
    assert leaf.external_id == mm_url
    assert leaf.url == mm_url
    # The subsystem facet is derived from the name prefix.
    assert leaf.categories == ["windows", "kernel", "mm"]
    assert leaf.metadata["subsystem"] == "mm"
    assert leaf.metadata["reference_count"] == 0
    # The pseudocode is one intact code chunk; a leaf emits no caller section.
    code_chunks = [c for c in leaf.chunks if c.kind == "code"]
    assert len(code_chunks) == 1
    assert code_chunks[0].text.startswith("```c")
    assert "MmMapIoSpace" in code_chunks[0].text
    assert "No references." not in leaf.text
    assert "## Referenced by" not in leaf.text

    called = documents[1]
    assert called.metadata["reference_count"] == 1
    assert "## Referenced by" in called.text
    assert "- IoCreateDevice" in called.text
    # The name is not duplicated into the body as a heading; it is the title.
    assert "# NtCreateFile" not in called.text


def test_iter_documents_skips_uncached_and_codless_pages(tmp_path):
    adapter = _adapter(tmp_path)
    workdir = tmp_path / "nt_functions"
    workdir.mkdir()
    missing = f"{BASE}/aa.html"
    empty = f"{BASE}/bb.html"
    (workdir / "index_items.json").write_text(
        json.dumps(
            [{"name": "Missing", "url": missing}, {"name": "Empty", "url": empty}]
        ),
        encoding="utf-8",
    )
    # A page that has no <pre> yields no document rather than an empty one.
    _seed(tmp_path, adapter, empty, "<h2>Empty</h2>\n<h3>Referenced by:</h3>\nNo references.")

    assert list(adapter.iter_documents()) == []


def test_fetch_caches_index_items_and_pages(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path)

    def get(self, url, headers=None):
        body = INDEX if url.rstrip("/") == BASE else _page("NtCreateFile")
        return httpx.Response(200, content=body.encode(), request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.Client, "get", get)
    adapter.fetch()

    workdir = tmp_path / "nt_functions"
    assert (workdir / "index.html").read_text(encoding="utf-8") == INDEX
    items = json.loads((workdir / "index_items.json").read_text(encoding="utf-8"))
    assert len(items) == 5  # 6 anchors, "NtCreateFile" linked twice to two pages
    assert len(list(adapter.iter_documents())) == 5

    # A second fetch with no force re-uses both caches and makes no requests.
    requested: list[str] = []

    def boom(self, url, headers=None):
        requested.append(url)
        raise AssertionError(f"unexpected refetch of {url}")

    monkeypatch.setattr(httpx.Client, "get", boom)
    adapter.fetch()
    assert requested == []


def test_fetch_picks_up_newly_selected_pages_without_force(tmp_path, monkeypatch):
    """Widening include_prefixes refetches the index from cache, not the wire."""
    adapter = _adapter(tmp_path, include_prefixes=["Nt"])
    monkeypatch.setattr(
        httpx.Client,
        "get",
        lambda self, url, headers=None: httpx.Response(
            200,
            content=(INDEX if url.rstrip("/") == BASE else _page("NtCreateFile")).encode(),
            request=httpx.Request("GET", url),
        ),
    )
    adapter.fetch()
    first = json.loads(
        (tmp_path / "nt_functions" / "index_items.json").read_text(encoding="utf-8")
    )
    assert [e["name"] for e in first] == ["NtCreateFile", "NtCreateFile"]

    wider = _adapter(tmp_path, include_prefixes=["Nt", "Mm", "Io", "Zw"])
    requested: list[str] = []

    def get(self, url, headers=None):
        requested.append(url)
        return httpx.Response(
            200, content=_page("x").encode(), request=httpx.Request("GET", url)
        )

    monkeypatch.setattr(httpx.Client, "get", get)
    wider.fetch()

    assert BASE + "/" not in requested  # index came from the cache
    assert len(json.loads(
        (tmp_path / "nt_functions" / "index_items.json").read_text(encoding="utf-8")
    )) == 5


def test_fetch_logs_and_stops_when_the_index_is_missing(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path)
    monkeypatch.setattr(
        httpx.Client,
        "get",
        lambda self, url, headers=None: httpx.Response(
            404, request=httpx.Request("GET", url)
        ),
    )
    adapter.fetch()
    assert not (tmp_path / "nt_functions" / "index_items.json").is_file()


def test_nt_functions_source_is_registered_and_typed():
    settings = Settings()
    by_id = {source.id: source for source in settings.sources}
    source = by_id["nt_functions"]

    assert source.type == "function_index"
    assert source.authority == "unknown"
    assert source.enabled is True
    assert source.url == "https://likeagod.revers.engineering/nt/"
    assert source.include_prefixes == [
        "Nt", "Zw", "Rtl", "Io", "Mm", "Ob", "Ps", "Se", "Cm", "Ki",
    ]
    # The selection is ~7.6k entries; a lower cap would silently truncate it.
    assert source.max_files == 8000
    assert "nt-internals" in source.categories

    assert isinstance(adapter_for(source), FunctionIndexAdapter)
    assert ADAPTER_REGISTRY["nt_functions"] is FunctionIndexAdapter
