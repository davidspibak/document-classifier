"""
Schema definitions for both storage engines.

- SQLite (SQL_SCHEMA): metadata that is small, structured, and queried by exact
  match / joins — one row per document, plus taxonomy and review-queue tables.
- LanceDB (pyarrow schemas): the embedding vectors. Two tables, matching the
  "matrix vs pooled vector" design discussed earlier:
    * doc_vectors   -> exactly one row per document (pooled/whole-doc embedding),
                       used by the classifier.
    * chunk_vectors -> N rows per document (one per chunk), used by search.
"""
import pyarrow as pa

# Embedding dimension for BAAI/bge-m3 dense vectors. If you swap embedding models,
# update this constant — it must match the model's output dimension exactly.
EMBEDDING_DIM = 1024

SQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    filename        TEXT NOT NULL,
    source_hash     TEXT NOT NULL,      -- content hash, used for dedup
    language        TEXT,
    category_path   TEXT,               -- e.g. "Science/Physics/Quantum"
    confidence      REAL,
    status          TEXT DEFAULT 'pending',  -- pending | auto_assigned | llm_assigned | needs_review
    upload_batch    TEXT,
    created_at      TEXT,
    -- metadata extraction fields (see metadata/ package)
    title_zh        TEXT,
    title_en        TEXT,
    authors_zh      TEXT,               -- JSON array string
    authors_pinyin  TEXT,
    keywords_zh     TEXT,               -- JSON array string
    keywords_en     TEXT,               -- JSON array string
    published_date  TEXT,
    summary_cache   TEXT                -- cached one-page summary, filled on-demand
);

CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category_path);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(source_hash);
CREATE INDEX IF NOT EXISTS idx_documents_batch ON documents(upload_batch);

CREATE TABLE IF NOT EXISTS taxonomy (
    category_id     TEXT PRIMARY KEY,
    parent_id       TEXT,               -- NULL for top-level (Domain) categories
    name            TEXT NOT NULL,
    description     TEXT NOT NULL,
    level            INTEGER NOT NULL,   -- 0 = Domain, 1 = Field, 2 = Subfield, ...
    embedding_id     TEXT                -- id of the matching row in LanceDB category_vectors
);

CREATE TABLE IF NOT EXISTS review_queue (
    doc_id          TEXT PRIMARY KEY,
    reason          TEXT,               -- e.g. "low_confidence", "llm_no_match", "low_ocr_confidence"
    candidate_categories TEXT,          -- JSON array of category_ids considered
    created_at      TEXT,
    resolved        INTEGER DEFAULT 0,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);

CREATE TABLE IF NOT EXISTS monthly_reports (
    batch_id        TEXT PRIMARY KEY,
    generated_at    TEXT,
    report_text     TEXT,               -- overall digest
    report_path     TEXT                -- path to the rendered docx/pdf, if exported
);
"""

# --- LanceDB (pyarrow) schemas ---

DOC_VECTORS_SCHEMA = pa.schema([
    pa.field("doc_id", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
    pa.field("category_path", pa.string()),
    pa.field("language", pa.string()),
])

CHUNK_VECTORS_SCHEMA = pa.schema([
    pa.field("chunk_id", pa.string()),
    pa.field("doc_id", pa.string()),
    pa.field("chunk_index", pa.int32()),
    pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
    pa.field("chunk_text", pa.string()),
    pa.field("language", pa.string()),
])

CATEGORY_VECTORS_SCHEMA = pa.schema([
    pa.field("category_id", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
])
