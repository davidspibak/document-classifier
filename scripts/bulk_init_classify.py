"""
One-time initial classification run for the FULL existing corpus (designed
for multi-million-document scale, e.g. 10M documents on an RTX 5090).

Key differences from the normal monthly pipeline (pipeline.process_folder):
  1. CPU-bound parsing/OCR runs in a multiprocessing pool, not sequentially.
  2. Embedding is batched across the whole GPU batch — every chunk of every
     document in the batch goes into ONE encode() call, rather than one call per
     document, which is where most of the GPU throughput comes from.
  3. The LLM tie-breaker runs as a separate pass over all ambiguous documents at
     the end instead of inline per document, so it can be batched via vLLM — at
     this scale that's the single biggest lever on total runtime (unbatched can
     be 10-40x slower for the ambiguous subset).
  4. Metadata extraction is left to config.yaml's `metadata:` block; turning off
     `use_llm_fallback` there avoids a per-document generation call across the
     whole corpus.

Usage: python scripts/bulk_init_classify.py --folder /path/to/corpus --workers 16
"""
import argparse
import sys
import uuid
from pathlib import Path
from multiprocessing import Pool

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Documents, category names and generated summaries in this project are multilingual
# by design. Force UTF-8 on the console: Windows defaults to cp1252, which raises
# UnicodeEncodeError the moment a CJK character is printed.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


from docclassify.config import CONFIG
from docclassify.ingestion.parsers import parse_document
from docclassify.ingestion.ocr import ocr_flagged_pages
from docclassify.ingestion.language import detect_language
from docclassify.ingestion.chunking import chunk_text, needs_pooling
from docclassify.ingestion.dedup import is_duplicate
from docclassify.embeddings.embedder import embed_texts
from docclassify.embeddings.pooling import mean_pool
from docclassify.storage import sqlite_store, lancedb_store
from docclassify.storage.schema import OUTCOME_AMBIGUOUS, OUTCOME_LLM_ASSIGNED, persisted_status
from docclassify.classification.classifier import (
    ClassificationResult, classify_top_down, resolve_with_llm,
)

EXTENSIONS = (".pdf", ".docx", ".pptx", ".html", ".htm", ".txt", ".md")
EMBED_BATCH_SIZE = 64          # tune upward if VRAM allows, for higher GPU throughput
UPLOAD_BATCH = "initial_bulk_load"
TIEBREAK_SNIPPET_CHARS = 800
OCR_CONFIDENCE_FLAG_THRESHOLD = CONFIG["ocr"]["confidence_flag_threshold"]


def _cpu_stage(file_path: str) -> dict | None:
    """Runs in a worker process: parse + OCR + dedup check. No GPU work here."""
    try:
        parsed = parse_document(file_path)
        if parsed.get("needs_ocr_pages"):
            parsed = ocr_flagged_pages(parsed, tesseract_lang="eng")
        text = parsed["text"].strip()
        if not text:
            return None
        duplicate, source_hash = is_duplicate(text)
        if duplicate:
            return None
        return {"file_path": file_path, "text": text, "source_hash": source_hash,
                "language": detect_language(text),
                "ocr_min_confidence": parsed.get("ocr_min_confidence")}
    except Exception as e:  # noqa: BLE001 - log and skip; a corrupt file shouldn't stop a 10M-doc run
        print(f"[cpu_stage] Failed on {file_path}: {e}")
        return None


def _embed_batch(parsed_batch: list[dict]) -> tuple[list[list[str]], list[list[list[float]]], list[list[float]]]:
    """
    Embeds a whole batch in as few encode() calls as possible.

    Returns (chunks_per_doc, chunk_vectors_per_doc, doc_vectors).

    Every document's chunks are concatenated into one encode() call and sliced
    back out afterwards. Document-level vectors are then either pooled from those
    chunk vectors (long documents) or produced by a second batched call over the
    whole texts (short ones) — matching pipeline.process_document, which the
    previous version of this script did not: it embedded every document whole,
    silently truncating anything past BGE-M3's context window.
    """
    chunks_per_doc: list[list[str]] = []
    flat_chunks: list[str] = []
    spans: list[tuple[int, int]] = []
    for d in parsed_batch:
        chunks = chunk_text(d["text"])
        chunks_per_doc.append(chunks)
        spans.append((len(flat_chunks), len(chunks)))
        flat_chunks.extend(chunks)

    flat_vectors = embed_texts(flat_chunks, batch_size=EMBED_BATCH_SIZE)
    chunk_vectors_per_doc = [flat_vectors[start:start + count] for start, count in spans]

    # Short documents are embedded whole (no pooling dilution); batch those too.
    whole_indices = [i for i, d in enumerate(parsed_batch)
                     if not needs_pooling(d["text"]) or not chunk_vectors_per_doc[i]]
    whole_vectors = embed_texts([parsed_batch[i]["text"] for i in whole_indices],
                                 batch_size=EMBED_BATCH_SIZE) if whole_indices else []

    doc_vectors: list[list[float] | None] = [None] * len(parsed_batch)
    for i, vec in zip(whole_indices, whole_vectors):
        doc_vectors[i] = vec
    for i, chunk_vectors in enumerate(chunk_vectors_per_doc):
        if doc_vectors[i] is None:
            doc_vectors[i] = mean_pool(chunk_vectors)

    return chunks_per_doc, chunk_vectors_per_doc, doc_vectors


def _gpu_and_write_stage(parsed_batch: list[dict], low_confidence_buffer: list[dict]):
    """
    Runs in the main process (GPU access). Embeds a batch of already-parsed
    documents, classifies via the fast embedding path, writes results, and
    collects ambiguous cases into low_confidence_buffer for a SEPARATE batched
    LLM pass afterward (rather than calling the LLM inline per document, which
    would serialize GPU work between embedding and generation).
    """
    chunks_per_doc, chunk_vectors_per_doc, doc_vectors = _embed_batch(parsed_batch)

    for d, chunks, chunk_vectors, doc_vector in zip(
        parsed_batch, chunks_per_doc, chunk_vectors_per_doc, doc_vectors
    ):
        doc_id = str(uuid.uuid4())
        result = classify_top_down(doc_vector)
        category_path = result.category_path or ""

        lancedb_store.upsert_chunk_vectors(doc_id, [
            {"chunk_index": i, "vector": v, "chunk_text": t, "language": d["language"]}
            for i, (v, t) in enumerate(zip(chunk_vectors, chunks))
        ], category_path=category_path)
        lancedb_store.upsert_doc_vector(doc_id, doc_vector, category_path, d["language"])

        sqlite_store.upsert_document({
            "doc_id": doc_id, "filename": Path(d["file_path"]).name,
            "source_hash": d["source_hash"], "language": d["language"],
            "category_path": result.category_path, "confidence": result.confidence,
            "status": persisted_status(result.status), "upload_batch": UPLOAD_BATCH,
        })

        # A page that only just survived OCR is worth a human's eyes regardless of
        # how the classifier did, so the reason travels with the document into the
        # tie-break pass rather than being written now and overwritten there.
        extra_reasons = []
        ocr_confidence = d.get("ocr_min_confidence")
        if ocr_confidence is not None and ocr_confidence < OCR_CONFIDENCE_FLAG_THRESHOLD:
            extra_reasons.append("low_ocr_confidence")

        if result.status == OUTCOME_AMBIGUOUS:
            low_confidence_buffer.append({
                "doc_id": doc_id,
                "candidates": result.candidate_category_ids,
                "base_path": result.category_path,
                "confidence": result.confidence,
                "snippet": d["text"][:TIEBREAK_SNIPPET_CHARS],
                "extra_reasons": extra_reasons,
            })
            continue

        reasons = ([result.review_reason] if result.review_reason else []) + extra_reasons
        if reasons:
            sqlite_store.add_to_review_queue(doc_id, reason=",".join(reasons),
                                              candidate_categories=result.candidate_category_ids)


def run_llm_tiebreak_batch(low_confidence_buffer: list[dict]):
    """
    LLM disambiguation for every document the fast path couldn't resolve, as a
    SEPARATE pass after the main embed/classify loop finishes.

    TODO (throughput): feed the whole buffer to vLLM in one batch instead of the
    per-item llama.cpp calls below. The prompts are already independent and the
    buffer is already materialized, so this is a drop-in replacement of the loop
    body; it's left explicit because the exact vLLM entry point depends on the
    installed version. See docs/architecture.md.
    """
    if not low_confidence_buffer:
        return

    print(f"[llm_tiebreak] {len(low_confidence_buffer)} documents need LLM disambiguation.")
    assigned = 0
    for item in low_confidence_buffer:
        base = ClassificationResult(item["base_path"], item["confidence"], OUTCOME_AMBIGUOUS,
                                     item["candidates"])
        try:
            resolved = resolve_with_llm(item["snippet"], base)
        except Exception as e:  # noqa: BLE001 - one bad generation shouldn't abort the pass
            print(f"[llm_tiebreak] Failed on {item['doc_id']}: {e}")
            sqlite_store.add_to_review_queue(item["doc_id"], reason="llm_error",
                                              candidate_categories=item["candidates"])
            continue

        extra_reasons = item.get("extra_reasons") or []
        if resolved.status == OUTCOME_LLM_ASSIGNED:
            # Full path, not the chosen node's bare name — resolve_with_llm composes
            # it from the ancestors the embedding walk already resolved.
            sqlite_store.upsert_document({
                "doc_id": item["doc_id"], "category_path": resolved.category_path,
                "confidence": resolved.confidence,
                "status": persisted_status(resolved.status),
            })
            lancedb_store.update_doc_category(item["doc_id"], resolved.category_path)
            assigned += 1
            if extra_reasons:
                sqlite_store.add_to_review_queue(item["doc_id"], reason=",".join(extra_reasons),
                                                  candidate_categories=item["candidates"])
        else:
            reasons = [resolved.review_reason or "llm_no_match"] + extra_reasons
            sqlite_store.add_to_review_queue(
                item["doc_id"], reason=",".join(reasons),
                candidate_categories=item["candidates"],
            )

    print(f"[llm_tiebreak] {assigned} assigned by the LLM, "
          f"{len(low_confidence_buffer) - assigned} queued for human review.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--workers", type=int, default=16, help="CPU worker processes for parsing/OCR")
    parser.add_argument("--gpu-batch-size", type=int, default=256, help="Documents per GPU embedding batch")
    args = parser.parse_args()

    sqlite_store.init_db()

    files = [str(p) for p in Path(args.folder).rglob("*") if p.suffix.lower() in EXTENSIONS]
    print(f"Found {len(files)} documents. Starting pipelined bulk classification with {args.workers} CPU workers ...")

    # Holds every ambiguous document until the tie-break pass at the end. Each entry
    # is ~1 KB (the snippet dominates), so budget roughly 1 GB per million ambiguous
    # documents — if that's too much for your corpus, flush the buffer through
    # run_llm_tiebreak_batch() every N batches instead of once at the end.
    low_confidence_buffer: list[dict] = []
    gpu_batch: list[dict] = []
    processed = 0
    skipped_duplicates = 0
    # The workers each check SQLite for duplicates, but two identical NEW files in
    # the same run both pass that check (neither is committed yet). Tracking hashes
    # here, where every result funnels through one process, closes that gap.
    seen_hashes: set[str] = set()

    with Pool(processes=args.workers) as pool:
        for parsed in pool.imap_unordered(_cpu_stage, files, chunksize=8):
            if parsed is None:
                continue
            if parsed["source_hash"] in seen_hashes:
                skipped_duplicates += 1
                continue
            seen_hashes.add(parsed["source_hash"])

            gpu_batch.append(parsed)
            if len(gpu_batch) >= args.gpu_batch_size:
                _gpu_and_write_stage(gpu_batch, low_confidence_buffer)
                processed += len(gpu_batch)
                print(f"  ... {processed}/{len(files)} processed")
                gpu_batch = []

        if gpu_batch:  # flush remainder
            _gpu_and_write_stage(gpu_batch, low_confidence_buffer)
            processed += len(gpu_batch)

    print(f"Main pass complete: {processed} documents classified via the embedding path"
          f" ({skipped_duplicates} in-run duplicates skipped).")
    run_llm_tiebreak_batch(low_confidence_buffer)
    print("Bulk initial classification finished.")


if __name__ == "__main__":
    main()
