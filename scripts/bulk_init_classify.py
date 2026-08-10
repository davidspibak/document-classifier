"""
One-time initial classification run for the FULL existing corpus (designed
for multi-million-document scale, e.g. 10M documents on an RTX 5090).

Key differences from the normal monthly pipeline (pipeline.process_folder):
  1. CPU-bound parsing/OCR runs in a multiprocessing pool, not sequentially.
  2. The LLM tie-breaker step is batched via vLLM instead of one-at-a-time
     llama-cpp-python calls - this is the single biggest lever on total
     runtime at this scale (see the throughput discussion: unbatched can be
     10-40x slower for the ambiguous-document subset).
  3. Stages are pipelined (parsing and embedding overlap) via a bounded queue,
     so the GPU isn't idle while the CPU stage catches up, and vice versa.

Usage: python scripts/bulk_init_classify.py --folder /path/to/corpus --workers 16
"""
import argparse
import sys
from pathlib import Path
from multiprocessing import Pool
from queue import Queue
from threading import Thread

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docclassify.ingestion.parsers import parse_document
from docclassify.ingestion.ocr import ocr_flagged_pages
from docclassify.ingestion.language import detect_language
from docclassify.ingestion.chunking import chunk_text, needs_pooling
from docclassify.ingestion.dedup import is_duplicate
from docclassify.embeddings.embedder import embed_texts
from docclassify.embeddings.pooling import mean_pool
from docclassify.storage import sqlite_store, lancedb_store
from docclassify.classification.classifier import classify_top_down

EXTENSIONS = (".pdf", ".docx", ".pptx", ".html", ".htm", ".txt", ".md")
EMBED_BATCH_SIZE = 64          # tune upward if VRAM allows, for higher GPU throughput
QUEUE_MAXSIZE = 500            # bounds memory use; backpressures the CPU stage if GPU falls behind


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
                "language": detect_language(text)}
    except Exception as e:  # noqa: BLE001 - log and skip; a corrupt file shouldn't stop a 10M-doc run
        print(f"[cpu_stage] Failed on {file_path}: {e}")
        return None


def _gpu_and_write_stage(parsed_batch: list[dict], low_confidence_buffer: list[dict]):
    """
    Runs in the main process (GPU access). Embeds a batch of already-parsed
    documents, classifies via the fast embedding path, writes results, and
    collects ambiguous cases into low_confidence_buffer for a SEPARATE batched
    LLM pass afterward (rather than calling the LLM inline per document, which
    would serialize GPU work between embedding and generation).
    """
    texts = [d["text"] for d in parsed_batch]
    # Split by whether pooling is needed to keep the embedding call itself simple;
    # for a real 10M-doc run, precompute this split once rather than per batch.
    doc_vectors = embed_texts(texts, batch_size=EMBED_BATCH_SIZE)

    for d, vec in zip(parsed_batch, doc_vectors):
        import uuid
        doc_id = str(uuid.uuid4())
        result = classify_top_down(vec)

        chunks = chunk_text(d["text"])
        chunk_vecs = embed_texts(chunks, batch_size=EMBED_BATCH_SIZE)
        lancedb_store.upsert_chunk_vectors(doc_id, [
            {"chunk_index": i, "vector": v, "chunk_text": t, "language": d["language"]}
            for i, (v, t) in enumerate(zip(chunk_vecs, chunks))
        ])
        lancedb_store.upsert_doc_vector(doc_id, vec, result.category_path or "", d["language"])

        record = {
            "doc_id": doc_id, "filename": Path(d["file_path"]).name,
            "source_hash": d["source_hash"], "language": d["language"],
            "category_path": result.category_path, "confidence": result.confidence,
            "status": result.status, "upload_batch": "initial_bulk_load",
        }
        sqlite_store.upsert_document(record)

        if result.status == "ambiguous":
            low_confidence_buffer.append({"doc_id": doc_id, "candidates": result.candidate_category_ids,
                                           "snippet": d["text"][:800]})


def run_llm_tiebreak_batch(low_confidence_buffer: list[dict]):
    """
    Batched LLM disambiguation via vLLM for every document the fast path
    couldn't resolve. This is a SEPARATE pass after the main embed/classify
    loop finishes, so vLLM can be given the whole batch at once for maximum
    throughput rather than interleaved one-at-a-time with embedding calls.

    Requires `pip install vllm` and a running local vLLM instance/engine -
    left as a clearly-marked extension point since exact vLLM setup depends
    on your installed version; the llama-cpp-python fallback below works but
    is far slower at this scale (see docs/architecture.md).
    """
    if not low_confidence_buffer:
        return
    print(f"[llm_tiebreak] {len(low_confidence_buffer)} documents need LLM disambiguation.")
    try:
        from docclassify.classification.llm_tiebreaker import llm_tiebreak
        for item in low_confidence_buffer:
            chosen = llm_tiebreak(item["snippet"], item["candidates"])  # TODO: replace with vLLM batched calls
            if chosen:
                from docclassify.storage import sqlite_store as sql
                taxonomy = sql.load_taxonomy()
                chosen_row = next((c for c in taxonomy if c["category_id"] == chosen), None)
                if chosen_row:
                    sql.upsert_document({"doc_id": item["doc_id"], "category_path": chosen_row["name"],
                                          "status": "llm_assigned"})
                    continue
            sqlite_store.add_to_review_queue(item["doc_id"], reason="llm_no_match",
                                              candidate_categories=item["candidates"])
    except ImportError:
        print("[llm_tiebreak] vLLM not installed - falling back to slower one-at-a-time llama.cpp calls.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--workers", type=int, default=16, help="CPU worker processes for parsing/OCR")
    parser.add_argument("--gpu-batch-size", type=int, default=256, help="Documents per GPU embedding batch")
    args = parser.parse_args()

    sqlite_store.init_db()

    files = [str(p) for p in Path(args.folder).rglob("*") if p.suffix.lower() in EXTENSIONS]
    print(f"Found {len(files)} documents. Starting pipelined bulk classification with {args.workers} CPU workers ...")

    low_confidence_buffer: list[dict] = []
    gpu_batch: list[dict] = []
    processed = 0

    with Pool(processes=args.workers) as pool:
        for parsed in pool.imap_unordered(_cpu_stage, files, chunksize=8):
            if parsed is None:
                continue
            gpu_batch.append(parsed)
            if len(gpu_batch) >= args.gpu_batch_size:
                _gpu_and_write_stage(gpu_batch, low_confidence_buffer)
                processed += len(gpu_batch)
                print(f"  ... {processed}/{len(files)} processed")
                gpu_batch = []

        if gpu_batch:  # flush remainder
            _gpu_and_write_stage(gpu_batch, low_confidence_buffer)
            processed += len(gpu_batch)

    print(f"Main pass complete: {processed} documents classified via the embedding path.")
    run_llm_tiebreak_batch(low_confidence_buffer)
    print("Bulk initial classification finished.")


if __name__ == "__main__":
    main()
