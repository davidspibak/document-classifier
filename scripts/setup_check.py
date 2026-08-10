"""
Run this once before going offline. Loads every local model/service and
reports pass/fail for each, so you catch a missing download or a broken
GPU install before it surfaces confusingly deep inside the pipeline.

Usage: python scripts/setup_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def check(name: str, fn):
    try:
        fn()
        print(f"[OK]   {name}")
        return True
    except Exception as e:  # noqa: BLE001 - intentionally broad; this is a diagnostic script
        print(f"[FAIL] {name}: {e}")
        return False


def main():
    results = []

    def check_embedding():
        from docclassify.embeddings.embedder import embed_text
        vec = embed_text("test sentence")
        assert len(vec) == 1024, f"expected 1024-dim vector, got {len(vec)}"
    results.append(check("BGE-M3 embedding model", check_embedding))

    def check_reranker():
        from docclassify.search.reranker import rerank
        rerank("test query", [{"chunk_text": "test candidate"}], top_n=1)
    results.append(check("bge-reranker-v2-m3", check_reranker))

    def check_llm():
        from docclassify.llm.local_llm import generate
        out = generate("Say OK.", max_tokens=5)
        assert out
    results.append(check("Local LLM (llama-cpp-python)", check_llm))

    def check_lancedb():
        from docclassify.storage import lancedb_store
        lancedb_store.doc_vectors_table()
    results.append(check("LanceDB", check_lancedb))

    def check_sqlite():
        from docclassify.storage import sqlite_store
        sqlite_store.init_db()
    results.append(check("SQLite", check_sqlite))

    def check_fasttext():
        from docclassify.ingestion.language import detect_language
        lang = detect_language("This is a test sentence in English.")
        assert lang == "en", f"expected 'en', got '{lang}'"
    results.append(check("fastText language ID", check_fasttext))

    def check_tesseract():
        import pytesseract
        pytesseract.get_tesseract_version()
    results.append(check("Tesseract binary", check_tesseract))

    def check_grobid():
        from docclassify.metadata.grobid_client import is_grobid_available
        if not is_grobid_available():
            raise RuntimeError("GROBID not reachable at localhost:8070 (start the service if you need academic PDF metadata extraction)")
    results.append(check("GROBID service (optional)", check_grobid))

    print(f"\n{sum(results)}/{len(results)} checks passed.")
    if not all(results[:-1]):  # GROBID is optional; don't fail the whole run on it alone
        sys.exit(1)


if __name__ == "__main__":
    main()
