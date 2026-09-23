"""Schema definition and migrations for the BlackBook SQLite database.

The schema is intentionally simple and single-file. FTS5 provides lexical
search; JSON1 (stored as TEXT) carries flexible metadata. A ``meta`` table
records the schema version and index version for staleness checks.
"""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 5
INDEX_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    source_id    TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    authority    TEXT NOT NULL DEFAULT 'unknown',
    enabled      INTEGER NOT NULL DEFAULT 1,
    source_type  TEXT NOT NULL DEFAULT 'filesystem',
    url          TEXT,
    last_fetched TEXT,
    version      TEXT,
    meta         TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id        INTEGER PRIMARY KEY,
    source_id     TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
    external_id   TEXT NOT NULL,
    title         TEXT NOT NULL,
    url           TEXT,
    path          TEXT,
    content_hash  TEXT NOT NULL,
    metadata      TEXT NOT NULL DEFAULT '{}',
    categories    TEXT NOT NULL DEFAULT '[]',
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (source_id, external_id)
);

CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_id);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id       INTEGER PRIMARY KEY,
    doc_id         INTEGER NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    ordinal        INTEGER NOT NULL,
    text           TEXT NOT NULL,
    section_path   TEXT NOT NULL DEFAULT '[]',
    page           INTEGER,
    token_estimate INTEGER NOT NULL DEFAULT 0,
    content_hash   TEXT NOT NULL,
    metadata       TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);

-- Dense vector embeddings for chunks (Phase 3, optional semantic layer).
-- One row per chunk. The vector is stored as raw float32 little-endian bytes;
-- ``dim`` and ``model`` let us detect staleness when the embedding model
-- changes. Rows are deleted with their chunk via the FK cascade.
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id  INTEGER PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    model     TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    vector    BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_embeddings_model ON chunk_embeddings(model);

-- Full-text index over chunk content. contentless so rows are managed by
-- triggers; we join back to chunks/documents for metadata at query time.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    title,
    section,
    text,
    content='',
    tokenize='unicode61 remove_diacritics 2'
);

-- Keep the FTS index in sync with chunks.
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, title, section, text)
    SELECT new.chunk_id,
           (SELECT title FROM documents WHERE doc_id = new.doc_id),
           (SELECT json_extract(new.section_path, '$[#-1]')),
           new.text;
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, title, section, text)
    VALUES ('delete', old.chunk_id,
            (SELECT title FROM documents WHERE doc_id = old.doc_id),
            (SELECT json_extract(old.section_path, '$[#-1]')),
            old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, title, section, text)
    VALUES ('delete', old.chunk_id,
            (SELECT title FROM documents WHERE doc_id = old.doc_id),
            (SELECT json_extract(old.section_path, '$[#-1]')),
            old.text);
    INSERT INTO chunks_fts(rowid, title, section, text)
    SELECT new.chunk_id,
           (SELECT title FROM documents WHERE doc_id = new.doc_id),
           (SELECT json_extract(new.section_path, '$[#-1]')),
           new.text;
END;

CREATE TABLE IF NOT EXISTS entities (
    entity_id   INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    meta        TEXT NOT NULL DEFAULT '{}',
    UNIQUE (name, entity_type)
);

CREATE TABLE IF NOT EXISTS relationships (
    rel_id           INTEGER PRIMARY KEY,
    subject_id       INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    predicate        TEXT NOT NULL,
    object_id        INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    evidence_doc_id  INTEGER REFERENCES documents(doc_id) ON DELETE SET NULL,
    confidence       REAL NOT NULL DEFAULT 1.0,
    inferred         INTEGER NOT NULL DEFAULT 0,
    support          INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_rel_subject ON relationships(subject_id);
CREATE INDEX IF NOT EXISTS idx_rel_object ON relationships(object_id);

-- Per-document knowledge-graph inputs: the vocabulary terms the document
-- contributed, keyed by a fingerprint of everything the extraction read
-- (content hash, title, source, categories, metadata). This is a cache, not
-- part of the graph: dropping the table costs one slower rebuild and nothing
-- else, since a rebuild derives the same terms again. It exists because term
-- extraction is a regex pass over every document's full text and dominates a
-- full rebuild, so an incremental rebuild skips it for the documents that have
-- not changed. Rows follow their document.
CREATE TABLE IF NOT EXISTS document_graph (
    doc_id      INTEGER PRIMARY KEY REFERENCES documents(doc_id) ON DELETE CASCADE,
    fingerprint TEXT NOT NULL,
    terms       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cases (
    case_id    INTEGER PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    target     TEXT NOT NULL DEFAULT '',
    platform   TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    meta       TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS case_observations (
    obs_id     INTEGER PRIMARY KEY,
    case_id    INTEGER NOT NULL REFERENCES cases(case_id) ON DELETE CASCADE,
    kind       TEXT NOT NULL,
    text       TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_obs_case ON case_observations(case_id);

-- Local query log: what was asked, and what came back. Written by the search
-- tools and `blackbook search`, read by `blackbook queries`. It exists to
-- answer the questions the corpus cannot answer about itself: which phrasings
-- return nothing, and what is being asked that is not indexed. Bounded by
-- pruning, so it is a recent-history log rather than an ever-growing one.
-- ``created_at`` is UTC, as every other timestamp in this schema.
CREATE TABLE IF NOT EXISTS query_log (
    query_id     INTEGER PRIMARY KEY,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    tool         TEXT NOT NULL,
    query        TEXT NOT NULL DEFAULT '',
    mode         TEXT NOT NULL DEFAULT '',
    sources      TEXT NOT NULL DEFAULT '[]',
    result_count INTEGER NOT NULL DEFAULT 0,
    top_score    REAL,
    latency_ms   REAL NOT NULL DEFAULT 0,
    backend      TEXT NOT NULL DEFAULT '',
    degraded     INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_query_log_created ON query_log(created_at);
"""


def migrate(conn) -> None:
    """Apply the schema and record versions. Idempotent.

    Commits before returning so migration does not leave a write transaction
    open. Python's sqlite3 opens an implicit transaction on the first write
    (the ``_set_meta`` INSERTs); without this commit that transaction — and the
    WAL writer lock it holds — would stay open for the life of the connection,
    blocking every other instance from writing (e.g. a second server started
    alongside the editor's stdio one). When the schema is already current the
    write is skipped entirely, so read-only peers open with no lock contention.
    """
    if (
        get_meta(conn, "schema_version") == str(SCHEMA_VERSION)
        and get_meta(conn, "index_version") == str(INDEX_VERSION)
    ):
        return
    conn.executescript(SCHEMA)
    # ``CREATE TABLE IF NOT EXISTS`` is a no-op on a table that already exists,
    # so it cannot deliver a new column to an existing database: the script above
    # leaves a pre-v3 ``relationships`` table exactly as it was. An explicit ALTER
    # is the only thing that reaches it, and it must be guarded because SQLite has
    # no ``ADD COLUMN IF NOT EXISTS``.
    _ensure_column(conn, "relationships", "support", "INTEGER NOT NULL DEFAULT 1")
    _set_meta(conn, "schema_version", str(SCHEMA_VERSION))
    _set_meta(conn, "index_version", str(INDEX_VERSION))
    conn.commit()


def _ensure_column(conn, table: str, column: str, decl: str) -> None:
    """Add ``column`` to ``table`` if it is not already there. Idempotent.

    The existence check is what makes this safe to run on every migration pass,
    including the first one against a fresh database where ``SCHEMA`` just
    created the column. Both the table and column names are code-supplied
    literals, never user input, so the interpolation below cannot be reached
    with anything a caller controls.
    """
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def _set_meta(conn, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def get_meta(conn, key: str) -> str | None:
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    except sqlite3.OperationalError:
        # meta table not created yet (fresh database) -> treat as unset so the
        # caller proceeds to build the schema.
        return None
    return row[0] if row else None
