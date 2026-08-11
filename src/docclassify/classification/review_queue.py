"""
Thin convenience wrapper over sqlite_store's review-queue functions, kept as
its own module so the UI layer (taxonomy_view.py) has a single, stable import
target rather than reaching into storage internals directly.
"""
from docclassify.storage import sqlite_store
from docclassify.storage.schema import STATUS_HUMAN_ASSIGNED


def list_pending() -> list[dict]:
    return sqlite_store.pending_review_items()


def resolve(doc_id: str, chosen_category_path: str):
    """
    Called from the UI when a human manually assigns a category to a flagged
    document. `chosen_category_path` must be the FULL slash-joined path
    ("Science/Physics/Quantum"), not a bare category name — see
    taxonomy_store.build_category_paths().

    Updates the vector tables as well as SQLite: search filters on the
    denormalized category_path stored alongside each chunk, so a resolution that
    only touched SQLite would leave the document unfindable under its new
    category.
    """
    from docclassify.storage import lancedb_store

    sqlite_store.upsert_document({"doc_id": doc_id, "category_path": chosen_category_path,
                                   "status": STATUS_HUMAN_ASSIGNED, "confidence": 1.0})
    lancedb_store.update_doc_category(doc_id, chosen_category_path)
    sqlite_store.resolve_review_item(doc_id)
