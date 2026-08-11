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

# --- document status vocabulary -------------------------------------------------
# Single source of truth for documents.status. The classifier reports its own
# intermediate outcomes (OUTCOME_*), which the pipeline maps to one of these
# persisted values via persisted_status() — that mapping is the only place the
# two vocabularies meet, so they can't drift apart again.
STATUS_PENDING = "pending"
STATUS_AUTO_ASSIGNED = "auto_assigned"
STATUS_LLM_ASSIGNED = "llm_assigned"
STATUS_HUMAN_ASSIGNED = "human_assigned"
STATUS_NEEDS_REVIEW = "needs_review"

PERSISTED_STATUSES = (
    STATUS_PENDING,
    STATUS_AUTO_ASSIGNED,
    STATUS_LLM_ASSIGNED,
    STATUS_HUMAN_ASSIGNED,
    STATUS_NEEDS_REVIEW,
)

# ClassificationResult.status values. The first two are also persisted verbatim;
# the last two mean "this document still has no confident home".
OUTCOME_AUTO_ASSIGNED = STATUS_AUTO_ASSIGNED
OUTCOME_LLM_ASSIGNED = STATUS_LLM_ASSIGNED
OUTCOME_AMBIGUOUS = "ambiguous"
OUTCOME_NO_MATCH = "no_match"


def persisted_status(outcome: str) -> str:
    """
    Maps a ClassificationResult.status onto the value stored in documents.status.
    Anything the classifier couldn't resolve becomes STATUS_NEEDS_REVIEW, which is
    what the Taxonomy view's review queue filters on.
    """
    if outcome in (OUTCOME_AUTO_ASSIGNED, OUTCOME_LLM_ASSIGNED):
        return outcome
    return STATUS_NEEDS_REVIEW


SQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    filename        TEXT NOT NULL,
    source_hash     TEXT NOT NULL,      -- content hash, used for dedup
    language        TEXT,
    category_path   TEXT,               -- e.g. "Science/Physics/Quantum"
    confidence      REAL,
    status          TEXT DEFAULT 'pending',  -- see PERSISTED_STATUSES above
    upload_batch    TEXT,
    created_at      TEXT,
    -- Metadata extraction fields (see metadata/extract.py). The *_zh suffix is
    -- historical: these hold the ORIGINAL-language values whatever that language
    -- is, and the *_en columns hold the English mirror (a copy for English
    -- documents, an LLM translation otherwise).
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
    reason          TEXT,               -- comma-separated, e.g. "llm_no_match,low_ocr_confidence"
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

# category_path is denormalized onto every chunk row on purpose: search filters by
# category, and LanceDB can only push a filter down if the column lives in the
# table being scanned. lancedb_store.update_doc_category() keeps it in sync when a
# document is re-classified.
CHUNK_VECTORS_SCHEMA = pa.schema([
    pa.field("chunk_id", pa.string()),
    pa.field("doc_id", pa.string()),
    pa.field("chunk_index", pa.int32()),
    pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
    pa.field("chunk_text", pa.string()),
    pa.field("language", pa.string()),
    pa.field("category_path", pa.string()),
])

CATEGORY_VECTORS_SCHEMA = pa.schema([
    pa.field("category_id", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
])
