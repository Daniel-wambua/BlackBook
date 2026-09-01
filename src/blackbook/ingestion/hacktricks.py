"""HackTricks ingestion adapter.

HackTricks is a GitBook/MkDocs-style markdown tree on GitHub. All the
fetching and parsing mechanics are shared with every other GitHub-markdown
source (see :class:`GithubMarkdownAdapter`); this subclass only pins the
source identity and the published-site URL mapping.
"""

from __future__ import annotations

from blackbook.ingestion.github_md import GithubMarkdownAdapter


class HackTricksAdapter(GithubMarkdownAdapter):
    """Ingests the HackTricks markdown book."""

    source_id = "hacktricks"

    def _source_url(self, root, path, front_matter: str = "") -> str:
        # The book is served at book.hacktricks.xyz with the repository's
        # directory structure mirrored and ``.md`` dropped. HackTricks pages
        # carry no Jekyll permalinks, so the front matter is unused here.
        rel = self._rel_to_repo(root, path)
        rel_str = str(rel).replace("\\", "/")
        if rel_str.endswith(".md"):
            rel_str = rel_str[:-3]
        return f"https://book.hacktricks.xyz/{rel_str}"
