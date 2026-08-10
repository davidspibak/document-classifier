"""
Content-hash based deduplication so re-uploading the same file doesn't create
duplicate rows or double-count in the monthly report.
"""
import hashlib


def hash_content(text: str) -> str:
    """
    Hash the *normalized text*, not the raw file bytes — this catches the same
    document re-saved in a different format or with different PDF metadata,
    which a byte-level file hash would miss.
    """
    normalized = " ".join(text.split()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def is_duplicate(text: str) -> tuple[bool, str]:
    """Returns (is_duplicate, source_hash). Caller checks sqlite_store.find_by_hash(source_hash)."""
    from docclassify.storage import sqlite_store  # local import avoids a circular import at module load time
    source_hash = hash_content(text)
    existing = sqlite_store.find_by_hash(source_hash)
    return existing is not None, source_hash
