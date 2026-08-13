"""
The single function that wires every module together for ONE document:
parse -> OCR (if needed) -> language detect -> metadata -> chunk -> embed ->
classify -> store. Both the UI's "ingest" button and scripts/monthly_ingest.py
call this same function, so behavior never drifts between the two entry points.
"""
import uuid
from pathlib import Path
from datetime import datetime, timezone

from docclassify.config import CONFIG
from docclassify.ingestion.parsers import parse_document
from docclassify.ingestion.ocr import ocr_flagged_pages
from docclassify.ingestion.language import detect_language, detect_languages_per_chunk
from docclassify.ingestion.chunking import chunk_text, needs_pooling
from docclassify.ingestion.dedup import is_duplicate
from docclassify.embeddings.embedder import embed_text, embed_texts
from docclassify.embeddings.pooling import mean_pool
from docclassify.classification.classifier import classify_document
from docclassify.metadata.extract import classification_snippet, extract_metadata
from docclassify.storage import sqlite_store, lancedb_store
from docclassify.storage.schema import persisted_status

OCR_CONFIDENCE_FLAG_THRESHOLD = CONFIG["ocr"]["confidence_flag_threshold"]


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

    # Metadata comes before classification because the extracted title is the most
    # useful thing to show the LLM tie-breaker if the embedding path is unsure.
    metadata = extract_metadata(str(path), text, doc_language)

    # --- embedding: chunk ONCE and reuse ---
    # The search index needs chunk vectors regardless, so a long document's
    # document-level vector is pooled from those same vectors rather than
    # chunking and embedding the whole document a second time.
    search_chunks_text = chunk_text(text)
    chunk_vectors = embed_texts(search_chunks_text)

    if needs_pooling(text) and chunk_vectors:
        doc_vector = mean_pool(chunk_vectors)
    else:
        doc_vector = embed_text(text)

    classification = classify_document(
        doc_vector, document_snippet=classification_snippet(metadata, text)
    )

    chunk_languages = detect_languages_per_chunk(search_chunks_text)
    chunk_records = [
        {"chunk_index": i, "vector": v, "chunk_text": t, "language": lang}
        for i, (v, t, lang) in enumerate(zip(chunk_vectors, search_chunks_text, chunk_languages))
    ]

    # --- persist everything ---
    category_path = classification.category_path or ""
    lancedb_store.upsert_doc_vector(doc_id, doc_vector, category_path, doc_language)
    lancedb_store.upsert_chunk_vectors(doc_id, chunk_records, category_path=category_path)

    doc_record = {
        "doc_id": doc_id,
        "filename": path.name,
        "source_hash": source_hash,
        "language": doc_language,
        "category_path": classification.category_path,
        "confidence": classification.confidence,
        "status": persisted_status(classification.status),
        "upload_batch": upload_batch or datetime.now(timezone.utc).strftime("%Y-%m"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        **metadata,
    }
    sqlite_store.upsert_document(doc_record)

    # One review-queue entry per document, carrying every reason it needs a look.
    # Writing separately per reason would silently overwrite the previous one,
    # since the queue is keyed on doc_id.
    review_reasons = []
    if classification.review_reason:
        review_reasons.append(classification.review_reason)
    ocr_confidence = parsed.get("ocr_min_confidence")
    if ocr_confidence is not None and ocr_confidence < OCR_CONFIDENCE_FLAG_THRESHOLD:
        review_reasons.append("low_ocr_confidence")

    if review_reasons:
        sqlite_store.add_to_review_queue(
            doc_id, reason=",".join(review_reasons),
            candidate_categories=classification.candidate_category_ids,
        )

    return doc_record


DEFAULT_EXTENSIONS = (".pdf", ".docx", ".pptx", ".html", ".htm", ".txt", ".md")


def process_folder(folder_path: str, upload_batch: str | None = None,
                    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
                    on_progress=None, should_cancel=None) -> list[dict]:
    """
    Sequential convenience wrapper for a small/medium folder. For 10M-doc scale, use
    scripts/bulk_init_classify.py instead.

    on_progress(done, total, path, status, record, error) is called after every file,
    where status is "ingested", "duplicate" or "failed". A long-running ingestion is
    otherwise completely opaque to a caller — the UI in particular had nothing to
    report between "starting" and "finished".

    should_cancel() is polled before each file; return True to stop early. Already
    processed documents stay ingested, since each is committed as it completes.

    NOTE on "duplicate": process_document returns the EXISTING row when content is a
    duplicate, so it is recognised here by that row carrying a different
    upload_batch than the one requested. That inference needs upload_batch to be
    passed, and cannot distinguish a re-run of the very same batch id.
    """
    folder = Path(folder_path)
    candidates = [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in extensions]
    total = len(candidates)
    results: list[dict] = []

    for index, file_path in enumerate(candidates, start=1):
        if should_cancel is not None and should_cancel():
            print(f"[pipeline] Cancelled after {index - 1}/{total} documents.")
            break

        status, record, error = "failed", None, None
        try:
            record = process_document(str(file_path), upload_batch=upload_batch)
            results.append(record)
            if upload_batch and record.get("upload_batch") != upload_batch:
                status = "duplicate"
            else:
                status = "ingested"
        except Exception as e:  # noqa: BLE001 - log and continue; one bad file shouldn't kill the batch
            error = e
            print(f"[pipeline] Failed on {file_path}: {e}")

        if on_progress is not None:
            try:
                on_progress(index, total, file_path, status, record, error)
            except Exception as callback_error:  # noqa: BLE001
                # A broken progress callback must never abort an ingestion run.
                print(f"[pipeline] progress callback raised: {callback_error}")

    return results
