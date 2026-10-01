"""blackbook.ingestion subpackage."""

from blackbook.ingestion.base import IngestStats, ParsedDocument, SourceAdapter
from blackbook.ingestion.pipeline import IngestionPipeline, PipelineResult
from blackbook.ingestion.hacktricks import HackTricksAdapter
from blackbook.ingestion.zerodf import ZeroDFAdapter
from blackbook.ingestion.pdf import PDFAdapter
from blackbook.ingestion.github_base import GithubTarballAdapter
from blackbook.ingestion.github_md import GithubMarkdownAdapter
from blackbook.ingestion.lolbas import LolbasAdapter
from blackbook.ingestion.gtfobins import GtfoBinsAdapter
from blackbook.ingestion.loobins import LooBinsAdapter
from blackbook.ingestion.wadcoms import WadcomsAdapter
from blackbook.ingestion.attack import MitreAttackAdapter
from blackbook.ingestion.website import WebsiteAdapter
from blackbook.ingestion.rss import RssAdapter
from blackbook.ingestion.gitbook import GitbookAdapter
from blackbook.ingestion.function_index import FunctionIndexAdapter

ADAPTER_REGISTRY = {
    "hacktricks": HackTricksAdapter,
    "0xdf": ZeroDFAdapter,
    "local_pdfs": PDFAdapter,
    "payloads": GithubMarkdownAdapter,
    "hacker_recipes": GithubMarkdownAdapter,
    "gtfobins": GtfoBinsAdapter,
    "lolbas": LolbasAdapter,
    "loobins": LooBinsAdapter,
    "wadcoms": WadcomsAdapter,
    "attack": MitreAttackAdapter,
    "portswigger": WebsiteAdapter,
    "google_bug_hunters": RssAdapter,
    "github_security_lab": RssAdapter,
    "martian1337_notes": GitbookAdapter,
    "pentest_book": GitbookAdapter,
    "nt_functions": FunctionIndexAdapter,
}


def adapter_for(source_config, raw_dir: str | None = None) -> SourceAdapter:
    """Instantiate the adapter for a source config.

    Falls back to matching by ``type`` when no id-specific adapter exists, so
    future sources can reuse adapters (e.g. a new "filesystem" source uses the
    PDF adapter shape, a new GitHub-markdown source uses the generic adapter,
    a feed-backed source uses the RSS adapter, a new GitBook space uses the
    GitBook adapter, or a new name-indexed HTML site uses the function-index
    adapter).
    """
    cls = ADAPTER_REGISTRY.get(source_config.id)
    if cls is None:
        by_type = {
            "filesystem": PDFAdapter,
            "git": GithubMarkdownAdapter,
            "website": WebsiteAdapter,
            "rss": RssAdapter,
            "gitbook": GitbookAdapter,
            "function_index": FunctionIndexAdapter,
        }
        cls = by_type.get(source_config.type)
    if cls is None:
        raise ValueError(f"no adapter registered for source {source_config.id!r}")
    return cls(source_config, raw_dir=raw_dir)


__all__ = [
    "SourceAdapter",
    "ParsedDocument",
    "IngestStats",
    "IngestionPipeline",
    "PipelineResult",
    "HackTricksAdapter",
    "ZeroDFAdapter",
    "PDFAdapter",
    "GithubTarballAdapter",
    "GithubMarkdownAdapter",
    "LolbasAdapter",
    "GtfoBinsAdapter",
    "LooBinsAdapter",
    "WadcomsAdapter",
    "MitreAttackAdapter",
    "WebsiteAdapter",
    "RssAdapter",
    "GitbookAdapter",
    "FunctionIndexAdapter",
    "adapter_for",
]
