# Architecture notes

See the in-chat design discussion for full detail. Quick reference:

## Shared pipeline
document storage -> ingestion & preprocessing (parse, OCR, language detect, chunk)
-> multilingual embeddings (BGE-M3)

## Subsystem 1: classification
fixed taxonomy (built once via clustering + LLM labeling, then human-reviewed)
-> embedding-similarity match, top-down through hierarchy levels
-> LLM tie-breaker (constrained to top-N candidates) for low-confidence cases
-> human review queue for anything still unresolved
-> metadata store (SQLite) + monthly LLM-generated report

## Subsystem 2: semantic search
chunk-level vectors in LanceDB -> query embedded with same BGE-M3 model
-> wide ANN candidate search (~50) -> cross-encoder rerank (bge-reranker-v2-m3)
-> top 5-10 results

## Storage shape
- SQLite: `documents` (metadata, category_path, language, extracted fields)
- LanceDB `doc_vectors`: one pooled vector per document, used by the classifier
- LanceDB `chunk_vectors`: N rows per document, used by search

## Special install notes
- `llama-cpp-python`: Windows PowerShell, `$env:CMAKE_ARGS = "-DGGML_CUDA=on"` then
  `pip install llama-cpp-python` (needs VS Build Tools + CUDA Toolkit).
- `vllm`: for the bulk one-time classification of the full existing corpus -
  batching matters enormously at multi-million-document scale (10-40x speedup
  over unbatched llama.cpp calls for the LLM tie-breaker step).
- Tesseract: install via `winget install -e --id UB-Mannheim.TesseractOCR`.
- GROBID: run locally as a Java service / Docker container for academic-paper
  metadata extraction (title/authors/date/keywords), ahead of the general
  LLM-based extraction fallback.
- Model weights (BGE-M3, reranker, Qwen2.5 GGUF, fastText lid.176): download once
  while online, then set `HF_HUB_OFFLINE=1` so the app never attempts a network call.

## Packaging
PyInstaller (recommended first) or Nuitka, `--standalone` mode (not `--onefile` -
unpacking GB-scale ML dependencies on every launch is too slow). Model weight
files stay external in `models/`, not compiled into the executable.
