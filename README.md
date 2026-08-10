# Document Auto-Classification & Semantic Search Project

Fully offline, single-GPU, Windows desktop application for hierarchical document
classification and multilingual semantic search.

## Layout

```
document-classifier-search/
├── config/                  # config.yaml — model paths, thresholds, DB paths
├── models/                  # downloaded model weights (gitignored, populated once, offline after)
├── data/
│   ├── inbox/                # drop new documents here for the app/watcher to pick up
│   ├── raw/                  # archived original files, content-addressed
│   ├── sqlite/                # app.db — documents, taxonomy, review queue, report metadata
│   └── lancedb/               # doc_vectors (1 row/doc) and chunk_vectors (N rows/doc)
├── src/docclassify/
│   ├── ingestion/            # parsing, OCR, language detection, chunking, dedup
│   ├── embeddings/           # BGE-M3 wrapper + pooling for doc-level vectors
│   ├── taxonomy/             # one-time: clustering + LLM labeling to build the fixed taxonomy
│   ├── classification/       # embedding-similarity classifier + LLM tie-breaker + review queue
│   ├── search/                # query embedding, ANN search, cross-encoder reranking
│   ├── reports/               # monthly batch report + on-demand per-document summaries
│   ├── storage/               # SQLite + LanceDB access layers and schemas
│   ├── llm/                   # local LLM wrapper (llama-cpp-python interactive / vLLM bulk)
│   ├── metadata/              # GROBID client, keyword extraction, field translation
│   └── ui/                    # PySide6 desktop app (views: ingest, search, taxonomy, reports)
├── scripts/
│   ├── build_taxonomy.py         # run once: build the fixed hierarchical taxonomy
│   ├── bulk_init_classify.py     # run once: classify the entire existing corpus
│   ├── monthly_ingest.py         # run monthly: ingest + classify + report new uploads
│   └── setup_check.py            # verify every local model/service loads before going offline
├── tests/
├── build/                    # PyInstaller/Nuitka packaging config
└── docs/architecture.md
```

## Build order

1. `scripts/setup_check.py` — confirm every model (BGE-M3, reranker, local LLM,
   Tesseract, GROBID, fastText) loads correctly while still online, so weights cache locally.
2. Implement `ingestion/` + `embeddings/` + `storage/` — validate on a small folder of documents.
3. `scripts/build_taxonomy.py` — cluster a sample of your existing corpus, label with the
   local LLM, review/edit, save the fixed taxonomy.
4. Implement `classification/` — embedding-similarity classifier + LLM tie-breaker.
5. `scripts/bulk_init_classify.py` — classify the full existing corpus (pipelined + batched;
   see docs/architecture.md for throughput notes on large corpora).
6. Implement `search/` — ANN search + reranking.
7. Implement `ui/` — PySide6 desktop app wrapping everything above.
8. Package with PyInstaller (or Nuitka) into a distributable `.exe`.

## Requirements

See `requirements.txt`. A handful of components need special installation
(GPU-enabled `llama-cpp-python`, the Tesseract binary itself, GROBID as a local
service, and model weight files) — see `docs/architecture.md` for exact commands.
