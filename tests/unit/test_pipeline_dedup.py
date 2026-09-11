from blackbook.config import SourceConfig
from blackbook.ingestion.base import ParsedDocument, SourceAdapter
from blackbook.ingestion.pipeline import IngestionPipeline
from blackbook.retrieval.chunking import RawChunk
from blackbook.storage import Chunk, Database, Document, Source
from blackbook.storage.database import sha256_text


class _Adapter(SourceAdapter):
    def __init__(self, config, documents, raw_dir=None):
        super().__init__(config, raw_dir)
        self.documents = documents

    def fetch(self, force=False):
        return None

    def iter_documents(self):
        yield from self.documents


def _document(external_id, text):
    return ParsedDocument(
        external_id=external_id,
        title=external_id,
        text=text,
        chunks=[RawChunk(text=text, section_path=[external_id], ordinal=0)],
    )


def test_opt_in_dedup_skips_chunks_already_in_other_sources(tmp_path):
    db = Database(tmp_path / "dedup.db")
    try:
        with db.session():
            db.upsert_source(Source(source_id="existing", name="Existing"))
            db.upsert_source(Source(source_id="new", name="New"))
            doc_id = db.upsert_document(
                Document(
                    source_id="existing",
                    external_id="old",
                    title="Old",
                    content_hash=sha256_text("shared guidance"),
                )
            )
            db.replace_chunks(
                doc_id,
                [Chunk(doc_id=doc_id, ordinal=0, text="shared guidance", content_hash=sha256_text("shared guidance"))],
            )
        config = SourceConfig(
            id="new",
            name="New",
            type="git",
            authority="trusted",
            url="https://github.com/example/new.git",
            deduplicate_chunks=True,
        )
        adapter = _Adapter(
            config,
            [_document("duplicate", "shared guidance"), _document("unique", "new guidance")],
        )
        result = IngestionPipeline(db).run(adapter)
        assert result.stats.parsed == 2
        assert result.stats.chunks_written == 1
        assert len(db.document_chunks(int(db.get_document_by_external("new", "unique")["doc_id"]))) == 1
        assert db.document_chunks(int(db.get_document_by_external("new", "duplicate")["doc_id"])) == []
    finally:
        db.close()
