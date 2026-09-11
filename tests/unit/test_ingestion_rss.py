import json
from pathlib import Path

import httpx

from blackbook.config import SourceConfig
from blackbook.ingestion.rss import RssAdapter


RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item><title>First report</title><guid>https://example.test/first</guid>
    <link>https://example.test/first</link>
    <description><![CDATA[<p>Broken access control guidance.</p>]]></description>
  </item>
</channel></rss>"""


def _adapter(tmp_path: Path) -> RssAdapter:
    return RssAdapter(
        SourceConfig(
            id="feed",
            name="Feed",
            type="rss",
            authority="trusted",
            url="https://example.test/",
            feed_url="https://example.test/feed.xml",
        ),
        raw_dir=str(tmp_path),
    )


def test_rss_adapter_parses_cached_entries(tmp_path):
    adapter = _adapter(tmp_path)
    workdir = tmp_path / "feed"
    workdir.mkdir()
    (workdir / "feed.xml").write_text(RSS, encoding="utf-8")

    documents = list(adapter.iter_documents())
    assert len(documents) == 1
    assert documents[0].title == "First report"
    assert "Broken access control guidance." in documents[0].text
    assert documents[0].url == "https://example.test/first"


def test_rss_adapter_sends_validators_and_reuses_304(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path)
    workdir = tmp_path / "feed"
    workdir.mkdir()
    feed_path = workdir / "feed.xml"
    feed_path.write_text(RSS, encoding="utf-8")
    (workdir / "feed-meta.json").write_text(
        json.dumps({"etag": '"abc"', "last_modified": "yesterday"}),
        encoding="utf-8",
    )
    requests = []

    def get(self, url, headers=None):
        requests.append((url, headers))
        return httpx.Response(304, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.Client, "get", get)
    adapter.fetch()
    assert requests == [
        (
            "https://example.test/feed.xml",
            {"If-None-Match": '"abc"', "If-Modified-Since": "yesterday"},
        )
    ]
    assert feed_path.read_text(encoding="utf-8") == RSS
