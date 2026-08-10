"""
The single function that wires every module together for ONE document:
parse -> OCR (if needed) -> language detect -> chunk -> embed -> classify ->
store. Both the UI's "ingest" button and scripts/monthly_ingest.py call this
same function, so behavior never drifts between the two entry points.
"""
import uuid
from pathlib import Path
from datetime import datetime, timezone

from docclassify.ingestion.parsers import parse_document
from docclassify.ingestion.ocr import ocr_flagged_pages
from docclassify.ingestion.language import detect_language, detect_languages_per_chunk
from docclassify.ingestion.chunking import chunk_text, needs_pooling
from docclassify.ingestion.dedup import is_duplicate
from docclassify.embeddings.embedder import embed_text, embed_texts
from docclassify.embeddings.pooling import mean_pool
from docclassify.classification.classifier import classify_document
from docclassify.storage import sqlite_store, lancedb_store


def process_document(file_path: str, upload_batch: str | None = None) -> dict:
    """
    Runs the full pipeline for one file. Returns the final document record.
    Skips re-processing (returns the existing record) if the content is a
    detected duplicate of something already ingested.
    """
    path = Path(file_path)
    parsed = parse_document(str(path))

    # OCR fallback only kicks in for PDFs with pages flagged as text-less.
    if parsed.get("needs_ocr_pages"):
        # NOTE: language hint for Tesseract defaults to English here; if you
        # know your corpus mix in advance, route this per-document instead
        # (e.g. from a filename convention or a prior classification pass).
        parsed = ocr_flagged_pages(parsed, tesseract_lang="eng")

    text = parsed["text"].strip()
    if not text:
        raise ValueError(f"No extractable text from {file_path} even after OCR fallback.")

    duplicate, source_hash = is_duplicate(text)
    if duplicate:
        existing = sqlite_store.find_by_hash(source_hash)
        return existing  # already ingested; caller can decide whether to notify the user

    doc_id = str(uuid.uuid4())
    doc_language = detect_language(text)

    # --- classification path: whole-doc embedding if it fits, else chunk + pool ---
    if needs_pooling(text):
        chunks_for_pooling = chunk_text(text)
        chunk_vecs = embed_texts(chunks_for_pooling)
        doc_vector = mean_pool(chunk_vecs)
    else:
        doc_vector = embed_text(text)

    classification = classify_document(doc_id, doc_vector)

    # --- search path: always chunk-level, independent of the classification path above ---
    search_chunks_text = chunk_text(text)
    chunk_languages = detect_languages_per_chunk(search_chunks_text)
    chunk_vectors = embed_texts(search_chunks_text)
    chunk_records = [
        {"chunk_index": i, "vector": v, "chunk_text": t, "language": lang}
        for i, (v, t, lang) in enumerate(zip(chunk_vectors, search_chunks_text, chunk_languages))
    ]

    # --- persist everything ---
    lancedb_store.upsert_doc_vector(doc_id, doc_vector, classification.category_path or "", doc_language)
    lancedb_store.upsert_chunk_vectors(doc_id, chunk_records)

    doc_record = {
        "doc_id": doc_id,
        "filename": path.name,
        "source_hash": source_hash,
        "language": doc_language,
        "category_path": classification.category_path,
        "confidence": classification.confidence,
        "status": classification.status,
        "upload_batch": upload_batch or datetime.now(timezone.utc).strftime("%Y-%m"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    sqlite_store.upsert_document(doc_record)

    if classification.status == "no_match":
        sqlite_store.add_to_review_queue(doc_id, reason="unresolved_after_llm",
                                          candidate_categories=classification.candidate_category_ids)

    return doc_record


def process_folder(folder_path: str, upload_batch: str | None = None,
                    extensions: tuple[str, ...] = (".pdf", ".docx", ".pptx", ".html", ".htm", ".txt", ".md")) -> list[dict]:
    """Sequential convenience wrapper for a small/medium folder. For 10M-doc scale, use scripts/bulk_init_classify.py instead."""
    folder = Path(folder_path)
    results = []
    for file_path in sorted(folder.rglob("*")):
        if file_path.suffix.lower() in extensions:
            try:
                results.append(process_document(str(file_path), upload_batch=upload_batch))
            except Exception as e:  # noqa: BLE001 - log and continue; one bad file shouldn't kill the batch
                print(f"[pipeline] Failed on {file_path}: {e}")
    return results
