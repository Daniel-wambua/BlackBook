from blackbook.retrieval.chunking import chunk_markdown, chunk_plain_pages, estimate_tokens


def test_markdown_headings_create_section_paths():
    md = "# Top\n\nintro text\n\n## Sub\n\nbody text\n"
    chunks = chunk_markdown(md, title_path=["Root"])
    paths = [c.section_path for c in chunks]
    assert any("Top" in p for p in paths)
    assert any("Sub" in p for p in paths)
    # body text under Sub should carry the nested breadcrumb
    sub = [c for c in chunks if "body text" in c.text][0]
    assert sub.section_path == ["Root", "Top", "Sub"]


def test_code_blocks_kept_intact():
    md = "# A\n\nbefore\n\n```bash\nimpacket-GetUserSPNs x\n```\n\nafter\n"
    chunks = chunk_markdown(md, title_path=[])
    code = [c for c in chunks if c.kind == "code"]
    assert code, "expected a code chunk"
    assert "impacket-GetUserSPNs" in code[0].text
    assert "```" in code[0].text


def test_blank_line_splits_paragraphs():
    md = "# A\n\npara one is here.\n\npara two is here.\n"
    chunks = chunk_markdown(md, title_path=[])
    texts = [c.text for c in chunks]
    assert any("para one" in t for t in texts)
    assert any("para two" in t for t in texts)


def test_long_text_is_split_on_boundaries():
    para = "word " * 300
    md = f"# A\n\n{para}\n\n{para}\n"
    chunks = chunk_markdown(md, title_path=[], max_tokens=120)
    # Should produce multiple chunks, none wildly over budget
    assert len(chunks) >= 2
    for c in chunks:
        assert estimate_tokens(c.text) <= 120 * 2  # generous bound


def test_plain_pages_track_page_numbers():
    pages = ["Page one content.\n\nMore on one.", "Page two content."]
    chunks = chunk_plain_pages(pages, title_path=["Doc"])
    by_page = {}
    for c in chunks:
        by_page.setdefault(c.page, []).append(c)
    assert 1 in by_page and 2 in by_page
    assert any("Page two" in c.text for c in by_page[2])


def test_estimate_tokens_monotonic():
    assert estimate_tokens("one two three") < estimate_tokens("one two three four five six")
