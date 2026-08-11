"""
Thin access layer over the SQLite metadata database.
Every function opens a short-lived connection (SQLite handles this cheaply)
rather than holding one connection open for the app's lifetime, which keeps
this safe to call from multiple threads (e.g. UI thread + background worker).
"""
import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
 
from docclassify.config import CONFIG
from docclassify.storage.schema import SQL_SCHEMA
 
DB_PATH = CONFIG["storage"]["sqlite_path"]
 
 
@contextmanager
def get_connection():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
 
 
def init_db():
    """Create all tables if they don't exist yet. Safe to call on every app startup."""
    with get_connection() as conn:
        conn.executescript(SQL_SCHEMA)
 
 
def upsert_document(doc: dict):
    """
    Insert a new document row, or update it if doc_id already exists
    (e.g. re-classification after a taxonomy edit).

    A partial dict (doc_id plus only the columns you want to change) is the normal
    way to update an EXISTING row. Passing a partial dict for a doc_id that
    doesn't exist yet raises on the NOT NULL filename/source_hash columns — that
    is deliberate, since it means a caller is updating a document that was never
    ingested.
    """
    doc = dict(doc)  # don't mutate caller's dict
    doc.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    # JSON-encode any list fields (authors, keywords) before storing
    for key in ("authors_zh", "keywords_zh", "keywords_en"):
        if key in doc and isinstance(doc[key], list):
            doc[key] = json.dumps(doc[key], ensure_ascii=False)
 
    columns = ", ".join(doc.keys())
    placeholders = ", ".join(f":{k}" for k in doc.keys())
    updates = ", ".join(f"{k}=excluded.{k}" for k in doc.keys() if k != "doc_id")
 
    sql = f"""
        INSERT INTO documents ({columns}) VALUES ({placeholders})
        ON CONFLICT(doc_id) DO UPDATE SET {updates}
    """
    with get_connection() as conn:
        conn.execute(sql, doc)
 
 
def get_document(doc_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return dict(row) if row else None
 
 
def find_by_hash(source_hash: str) -> dict | None:
    """Used by ingestion/dedup.py to detect re-uploaded files before re-processing."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE source_hash = ?", (source_hash,)
        ).fetchone()
        return dict(row) if row else None
 
 
def documents_in_batch(upload_batch: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM documents WHERE upload_batch = ?", (upload_batch,)
        ).fetchall()
        return [dict(r) for r in rows]
 
 
def documents_in_category(category_path: str, limit: int | None = None) -> list[dict]:
    sql = "SELECT * FROM documents WHERE category_path = ?"
    params: tuple = (category_path,)
    if limit:
        sql += " LIMIT ?"
        params = (category_path, limit)
    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
 
 
# --- taxonomy ---
 
def find_category(name: str, parent_id: str | None) -> dict | None:
    """
    Looks up a category by (name, parent_id) rather than category_id, so
    taxonomy construction/import scripts can be re-run safely - a second run
    updates the existing node's description/embedding instead of creating a
    duplicate with a new random id.
    """
    with get_connection() as conn:
        if parent_id is None:
            row = conn.execute(
                "SELECT * FROM taxonomy WHERE name = ? AND parent_id IS NULL", (name,)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM taxonomy WHERE name = ? AND parent_id = ?", (name, parent_id)
            ).fetchone()
        return dict(row) if row else None
 
 
def save_category(category: dict):
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO taxonomy (category_id, parent_id, name, description, level, embedding_id)
            VALUES (:category_id, :parent_id, :name, :description, :level, :embedding_id)
            ON CONFLICT(category_id) DO UPDATE SET
                parent_id=excluded.parent_id, name=excluded.name,
                description=excluded.description, level=excluded.level,
                embedding_id=excluded.embedding_id
            """,
            category,
        )
 
 
def load_taxonomy() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM taxonomy ORDER BY level ASC").fetchall()
        return [dict(r) for r in rows]
 
 
def children_of(category_id: str | None) -> list[dict]:
    """category_id=None returns the top-level (Domain) categories."""
    with get_connection() as conn:
        if category_id is None:
            rows = conn.execute("SELECT * FROM taxonomy WHERE parent_id IS NULL").fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM taxonomy WHERE parent_id = ?", (category_id,)
            ).fetchall()
        return [dict(r) for r in rows]
 
 
# --- review queue ---
 
def add_to_review_queue(doc_id: str, reason: str, candidate_categories: list[str]):
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO review_queue (doc_id, reason, candidate_categories, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                reason=excluded.reason, candidate_categories=excluded.candidate_categories
            """,
            (doc_id, reason, json.dumps(candidate_categories), datetime.now(timezone.utc).isoformat()),
        )
 
 
def pending_review_items() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM review_queue WHERE resolved = 0").fetchall()
        return [dict(r) for r in rows]
 
 
def resolve_review_item(doc_id: str):
    with get_connection() as conn:
        conn.execute("UPDATE review_queue SET resolved = 1 WHERE doc_id = ?", (doc_id,))