"""
Thin convenience wrapper over sqlite_store's review-queue functions, kept as
its own module so the UI layer (taxonomy_view.py) has a single, stable import
target rather than reaching into storage internals directly.
"""
from docclassify.storage import sqlite_store


def list_pending() -> list[dict]:
    return sqlite_store.pending_review_items()


def resolve(doc_id: str, chosen_category_path: str):
    """Called from the UI when a human manually assigns a category to a flagged document."""
    sqlite_store.upsert_document({"doc_id": doc_id, "category_path": chosen_category_path,
                                   "status": "human_assigned", "confidence": 1.0})
    sqlite_store.resolve_review_item(doc_id)
