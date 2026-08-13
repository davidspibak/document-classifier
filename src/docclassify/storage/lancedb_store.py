"""
Access layer over LanceDB (embedded, single-file vector store — no server process).
Owns three tables: doc_vectors, chunk_vectors, category_vectors.

Every filter string built here goes through storage/filters.py rather than raw
f-string interpolation, so values containing quotes can't break (or rewrite) the
predicate.
"""
import lancedb

from docclassify.config import CONFIG
from docclassify.storage.filters import sql_literal
from docclassify.storage.schema import (
    DOC_VECTORS_SCHEMA,
    CHUNK_VECTORS_SCHEMA,
    CATEGORY_VECTORS_SCHEMA,
)

_DB_PATH = CONFIG["storage"]["lancedb_path"]
_db = None  # lazy singleton connection


def get_db():
    global _db
    if _db is None:
        _db = lancedb.connect(_DB_PATH)
    return _db


def _get_or_create_table(name: str, schema):
    db = get_db()
    if name in db.table_names():
        return db.open_table(name)
    return db.create_table(name, schema=schema)


def doc_vectors_table():
    return _get_or_create_table("doc_vectors", DOC_VECTORS_SCHEMA)


def chunk_vectors_table():
    return _get_or_create_table("chunk_vectors", CHUNK_VECTORS_SCHEMA)


def category_vectors_table():
    return _get_or_create_table("category_vectors", CATEGORY_VECTORS_SCHEMA)


def upsert_doc_vector(doc_id: str, vector: list[float], category_path: str, language: str):
    table = doc_vectors_table()
    # LanceDB has no native upsert; delete-then-add is the standard pattern for
    # replacing a row (e.g. re-classifying a document after a taxonomy edit).
    table.delete(f"doc_id = {sql_literal(doc_id)}")
    table.add([{
        "doc_id": doc_id, "vector": vector,
        "category_path": category_path, "language": language,
    }])


def upsert_chunk_vectors(doc_id: str, chunks: list[dict], category_path: str = ""):
    """
    chunks: list of {"chunk_index": int, "vector": list[float], "chunk_text": str, "language": str}
    category_path is denormalized onto every chunk row so search can push a
    category filter down into LanceDB (see CHUNK_VECTORS_SCHEMA).

    Replaces ALL existing chunks for this doc_id before inserting the new set —
    important when re-processing a document (e.g. after an edit), so stale
    chunks don't linger and pollute search results.
    """
    table = chunk_vectors_table()
    table.delete(f"doc_id = {sql_literal(doc_id)}")
    rows = [
        {
            "chunk_id": f"{doc_id}_chunk_{c['chunk_index']:04d}",
            "doc_id": doc_id,
            "chunk_index": c["chunk_index"],
            "vector": c["vector"],
            "chunk_text": c["chunk_text"],
            "language": c.get("language", "unknown"),
            "category_path": category_path or "",
        }
        for c in chunks
    ]
    if rows:
        table.add(rows)


def update_doc_category(doc_id: str, category_path: str):
    """
    Re-points a document's denormalized category_path in BOTH vector tables after
    a re-classification (the LLM tie-break pass, or a human resolving a review
    item). Without this the vectors keep the stale category and search's category
    filter silently disagrees with SQLite.
    """
    where = f"doc_id = {sql_literal(doc_id)}"
    values = {"category_path": category_path or ""}
    doc_vectors_table().update(where=where, values=values)
    chunk_vectors_table().update(where=where, values=values)


def upsert_category_vector(category_id: str, vector: list[float]):
    table = category_vectors_table()
    table.delete(f"category_id = {sql_literal(category_id)}")
    table.add([{"category_id": category_id, "vector": vector}])


def get_doc_vector(doc_id: str) -> list[float] | None:
    table = doc_vectors_table()
    results = table.search().where(f"doc_id = {sql_literal(doc_id)}").limit(1).to_list()
    return results[0]["vector"] if results else None


def search_chunks(query_vector: list[float], top_k: int = 50, where: str | None = None) -> list[dict]:
    """Wide ANN candidate search over chunk_vectors, optional SQL-style metadata filter."""
    table = chunk_vectors_table()
    q = table.search(query_vector).limit(top_k)
    if where:
        q = q.where(where)
    return q.to_list()


def all_category_vectors() -> list[dict]:
    """Small table (one row per taxonomy node) — safe to pull entirely into memory."""
    table = category_vectors_table()
    return table.to_pandas().to_dict(orient="records")


def get_document_chunks(doc_id: str) -> list[dict]:
    """Every chunk of one document, in reading order."""
    table = chunk_vectors_table()
    rows = table.search().where(f"doc_id = {sql_literal(doc_id)}").limit(100_000).to_list()
    return sorted(rows, key=lambda r: r.get("chunk_index", 0))


def get_document_text(doc_id: str) -> str:
    """
    Reassembles a document's text from its stored chunks.

    The full text is never persisted anywhere — only chunks are — so anything that
    needs the whole document (the on-demand summary, a UI preview) has to rebuild
    it from here.

    The overlap-aware reassembly lives in ingestion.chunking.join_chunks, next to the
    chunk_text() it inverts, so the two cannot drift apart.
    """
    from docclassify.ingestion.chunking import join_chunks

    chunks = get_document_chunks(doc_id)
    return join_chunks([c.get("chunk_text", "") for c in chunks])
