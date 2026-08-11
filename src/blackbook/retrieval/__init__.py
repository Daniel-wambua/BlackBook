"""blackbook.retrieval subpackage."""

from blackbook.retrieval.chunking import RawChunk, chunk_markdown, chunk_plain_pages
from blackbook.retrieval.hybrid import HybridRetriever, SearchResult
from blackbook.retrieval.lexical import LexicalRetriever

__all__ = [
    "RawChunk",
    "chunk_markdown",
    "chunk_plain_pages",
    "HybridRetriever",
    "SearchResult",
    "LexicalRetriever",
]
