# Document Auto-Classification & Semantic Search

Fully offline, single-GPU, Windows desktop application for hierarchical
document classification and multilingual semantic search.

This README was generated from a full read-through of the codebase (`src/docclassify/`,
`scripts/`, `tests/`, `config/`) plus a live syntax check and dry-run of every
pure-logic module. It is meant to be the one document you hand to a new
machine to go from a fresh clone to a running app.

---

## 1. What this project does

Two subsystems share one ingestion pipeline:

| Subsystem | Purpose | Key models |
|---|---|---|
| **Classification** | Assigns every ingested document to a node in a fixed, hierarchical taxonomy (Domain → Field → Subfield → …) | BGE-M3 (embedding similarity) + Qwen2.5-7B (LLM tie-breaker for ambiguous cases) |
| **Semantic Search** | Multilingual, chunk-level search over the whole corpus | BGE-M3 (candidate retrieval) + bge-reranker-v2-m3 (precision reranking) |

Shared pipeline: **parse → OCR fallback → language ID → chunk → embed → (classify + index)**.

Everything runs locally after a one-time model download — no network calls
at runtime (`HF_HUB_OFFLINE=1`). It ships as a PySide6 desktop app with four
views: Ingest, Search, Taxonomy (+ review queue), Reports.

---

## 2. Project layout

```
document-classifier-search/
├── config/config.yaml         # model paths, thresholds, DB paths — single source of truth
├── models/                    # NOT included — model weights, downloaded once (see §4)
├── data/
│   ├── inbox/                 # drop new documents here for ingestion
│   ├── raw/                   # archived originals, content-addressed
│   ├── sqlite/                # app.db — documents, taxonomy, review queue, reports
│   └── lancedb/                # doc_vectors, chunk_vectors, category_vectors
├── src/docclassify/
│   ├── ingestion/             # parsing, OCR, language detection, chunking, dedup
│   ├── embeddings/            # BGE-M3 wrapper + pooling
│   ├── taxonomy/              # clustering + LLM labeling to build the fixed taxonomy
│   ├── classification/        # similarity classifier + LLM tie-breaker + review queue
│   ├── search/                # query embedding, ANN search, cross-encoder reranking
│   ├── reports/                # monthly batch report + on-demand document summaries
│   ├── storage/                # SQLite + LanceDB access layers and schemas
│   ├── llm/                    # local LLM wrapper (llama-cpp-python / vLLM)
│   ├── metadata/                # GROBID client, keyword extraction, translation
│   ├── ui/                      # PySide6 desktop app
│   └── pipeline.py             # the single function that wires one document end-to-end
├── scripts/
│   ├── setup_check.py          # verify every local model/service loads (run first)
│   ├── build_taxonomy.py       # one-time: build the fixed taxonomy
│   ├── bulk_init_classify.py   # one-time: classify the full existing corpus
│   ├── monthly_ingest.py       # recurring: ingest + classify + report new uploads
│   ├── fetch_offline_bundle.ps1 # ONLINE, once: pack every wheel/tool/weight (§5A)
│   ├── fetch_models.py         # ONLINE, once: download the four model artifacts
│   └── install_offline.ps1     # OFFLINE: build the venv from the vendored wheelhouse
├── packages/                   # the offline bundle payload (gitignored)
│   ├── wheels/                 # every runtime + build dependency as a .whl
│   ├── wheels-torch-cuda/      # CUDA torch (not on PyPI), if fetched with -TorchCuda
│   ├── tools/                  # the Tesseract OCR installer
│   └── MANIFEST.txt            # what was fetched, and for which Python version
├── tests/                      # unit tests for pure-logic modules (no GPU/model needed)
├── build/
│   ├── main.spec               # PyInstaller build spec (--onedir)
│   ├── build.ps1               # env check → clean → build → assemble
│   ├── README_BUILD.md         # packaging gotchas — read before debugging a build
│   └── hooks/
│       ├── hook-llama_cpp.py   # collects llama.cpp's ctypes-loaded backend DLLs
│       └── hook-lancedb.py     # collects lancedb's native Rust extension
├── dist/docclassify/           # build output (gitignored)
├── main.py                     # application entry point (what gets packaged into the .exe)
├── requirements.txt            # runtime dependencies
├── requirements-build.txt      # PyInstaller, pytest, venv bootstrap
├── pyproject.toml
└── docs/architecture.md
```

---

## 3. Code audit summary — is this ready to run?

I read every source file, syntax-checked the whole tree, and executed the
pure-logic modules (storage, chunking, dedup, pooling, classifier similarity
scoring, reranker sort order) against synthetic data. Results:

**Working / verified:**
- All 51 Python files compile with no syntax errors.
- `config.py` correctly loads `config.yaml` and resolves relative paths against the project root.
- `storage/sqlite_store.py` — schema creation, upsert/read round-trips, taxonomy CRUD, review queue all behave correctly (verified live).
- `ingestion/chunking.py`, `ingestion/dedup.py`, `embeddings/pooling.py` — correct output on live tests, including edge cases (empty text, zero vectors).
- `classification/classifier.py` cosine-similarity ranking and zero-vector handling — correct (unit tests pass).
- `search/reranker.py` sort order and `top_n` truncation — correct (unit tests pass).
- `tests/test_classification.py` and `tests/test_ingestion.py` pass out of the box.
- Design is internally consistent: the same `embed_text`/`embed_texts` functions are used for documents, queries, and taxonomy nodes (required for the classifier and search to share one vector space); ingestion, classification, and search modules are properly decoupled through `storage/`.

**Gaps to close before a full production run:**
1. **Nothing is installed yet.** There is no virtual environment in the repo. The
   offline bundle in `packages/` is what you install *from* — see §5A.
2. **The venv must be Python 3.12.** Not a preference: `fasttext` publishes no wheels
   at all (sdist only, needs a C++ compiler), so `requirements.txt` uses
   `fasttext-wheel`, whose newest binaries are `cp312`. Everything else in the stack
   is fine on newer Pythons. Full explanation in `build/README_BUILD.md`.
3. **The build has never been executed end to end.** The PyInstaller setup in `build/`
   is written and its dependency set is verified resolvable offline, but producing an
   actual `.exe` needs a 3.12 install, which hasn't happened yet.
4. **`scripts/bulk_init_classify.py`'s LLM tie-break pass is still one-at-a-time.** It runs as a correct, separate batch pass over all ambiguous documents (so the buffer is already materialized and the prompts are independent), but the loop body still makes per-item `llama-cpp` calls rather than the vLLM batched call the architecture doc calls for. Marked `TODO (throughput)` at the function it lives in. Note that **`vllm` has no Windows wheels**, so this path cannot be completed on Windows as things stand.
5. **No dependency lock file** (`requirements.txt` has no pinned versions) — for a reproducible offline build, pin versions once you've confirmed a working combination (§5.4). The wheelhouse in `packages/wheels/` is a de-facto lock: `packages/MANIFEST.txt` lists every exact filename fetched.
6. **`reports/doc_summary.py` is not reachable from the UI.** The logic is complete, but generating a summary needs the document's full text, which is only persisted as chunks in LanceDB — wiring it up means reassembling text from `chunk_vectors` first.

**Bottom line:** the application logic is real, coherent, and testable — this is not a
skeleton. The packaging and offline-install tooling is written and verified as far as
it can be without a 3.12 interpreter. What remains is running it.

---

## 4. Prerequisites

| Requirement | Notes |
|---|---|
| OS | Windows 10/11 (the app and its OCR/LLM install steps are Windows-targeted; Linux/macOS work for development but adjust the installer commands in §5.2) |
| Python | **3.12 exactly.** Not "3.11 or newer" — `fasttext` publishes no wheels at all, so `requirements.txt` uses `fasttext-wheel`, whose newest binaries are `cp312`. See `build/README_BUILD.md`. |
| GPU | NVIDIA GPU with CUDA support strongly recommended (BGE-M3, the reranker, and the 7B local LLM are all meant to run GPU-accelerated; CPU-only will work but is slow) |
| NVIDIA driver | Needed on the machine that *runs* the app. The CUDA runtime DLLs ship inside the wheels; the driver cannot be bundled. |
| CUDA Toolkit | **Not required** if you use the prebuilt `llama-cpp-python` wheel and CUDA torch (§5.3) — only needed if you choose to compile `llama-cpp-python` yourself. |
| VS Build Tools | **Not required** for the same reason. The whole point of the vendored wheelhouse is that nothing compiles at install time. |
| Disk space | ~13 GB: model weights ~8.8 GB (BGE-M3 2.2 GB, reranker 2.2 GB, Qwen2.5-7B Q4_K_M GGUF 4.4 GB across two shards, fastText lid.176 126 MB), wheelhouse ~1.2 GB, installed venv ~3 GB. Add ~2.5 GB if you also fetch CUDA torch. |
| Java or Docker | Only if you want GROBID (optional — academic PDF metadata extraction) |

---

## 5A. Fully offline installation (recommended)

Two commands. The first needs internet **once**, on any machine; the second needs
none, ever. Nothing compiles at install time, so the target machine needs no CUDA
Toolkit and no Visual Studio Build Tools.

```powershell
# ON A CONNECTED MACHINE, ONCE — packs wheels, the Tesseract installer and the
# model weights into the checkout (~13 GB, or ~10.5 GB without -TorchCuda):
powershell -ExecutionPolicy Bypass -File scripts\fetch_offline_bundle.ps1 -TorchCuda

# copy the whole folder to the offline machine, then THERE:
powershell -ExecutionPolicy Bypass -File scripts\install_offline.ps1 -TorchCuda
```

`install_offline.ps1` creates `.venv`, installs everything with `--no-index` (so pip
*cannot* silently reach PyPI), verifies the imports, and prints the two manual steps
left: running `packages\tools\tesseract-ocr-w64-setup-*.exe`, and `setx HF_HUB_OFFLINE 1`.

What the fetch step produces:

| Path | Contents |
|---|---|
| `packages\wheels\` | 118 wheels, ~1.2 GB — every runtime and build dependency, including a prebuilt CUDA `llama-cpp-python` |
| `packages\wheels-torch-cuda\` | CUDA torch (`-TorchCuda` only). PyPI's torch is **CPU-only**; without this everything runs correct but slow. |
| `packages\tools\` | The Tesseract OCR installer, plus a note on vendoring GROBID via `docker save` |
| `packages\MANIFEST.txt` | Every filename fetched, and the Python version targeted |
| `models\` | The four model artifacts, ~8.8 GB |

> **The wheelhouse is Python-version-specific.** `MANIFEST.txt` records the target
> (3.12) and `install_offline.ps1` refuses to run on a mismatch rather than failing
> halfway through. You do **not** need 3.12 on the machine doing the fetching —
> `pip download --python-version 3.12` cross-fetches binaries for another interpreter.

To verify a bundle without a 3.12 interpreter anywhere:

```powershell
python -m pip install --dry-run --ignore-installed --no-index `
    --find-links packages\wheels --only-binary=:all: `
    --python-version 3.12 --platform win_amd64 --target $env:TEMP\check `
    -r requirements.txt -r requirements-build.txt
```

A clean exit means every dependency resolves from local files with zero network.

---

## 5. Installation (online, for development)

### 5.1 Clone and set up a virtual environment

```powershell
git clone <your-repo-url> document-classifier-search
cd document-classifier-search
# 3.12 specifically — see §4. Newer Pythons have no fasttext-wheel binaries.
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

### 5.2 Install core dependencies

```powershell
pip install -r requirements.txt
pip install -r requirements-build.txt   # PyInstaller + pytest
```

### 5.3 Install the components that need special handling

**`llama-cpp-python`** — prefer the prebuilt wheel; it needs no compiler and no
CUDA Toolkit:

```powershell
pip install llama-cpp-python --index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```
Use `.../whl/cpu` for a CPU-only build. Only if no prebuilt wheel suits you should
you compile it, which *does* need the CUDA Toolkit and VS Build Tools (C++ workload):
```powershell
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
pip install llama-cpp-python
```

**CUDA torch** — PyPI's `torch` is CPU-only on Windows:
```powershell
pip install --force-reinstall --no-deps torch --index-url https://download.pytorch.org/whl/cu124
```

**`vllm`** — referenced by `scripts/bulk_init_classify.py`'s batching TODO, but
**there are no Windows wheels for it**, so that path can't currently be completed on
Windows. The per-item llama.cpp fallback in the same function is what runs.

**Tesseract OCR** (binary, not a pip package):
```powershell
winget install -e --id UB-Mannheim.TesseractOCR
```
Confirm `pytesseract` can find it — if not, set the path explicitly:
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

**GROBID** (optional — improves metadata extraction for academic PDFs; the
app works without it and falls back to LLM-based extraction):
```powershell
docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.8.0
```

### 5.4 (Recommended) Pin your working versions

Once every check in §6 passes, freeze the environment so it's reproducible:
```powershell
pip freeze > requirements.lock.txt
```

### 5.5 Download model weights (one time, while online)

```powershell
pip install huggingface_hub
python scripts\fetch_models.py                    # all four, into models\
python scripts\fetch_models.py --only lid         # or just one
```

The script uses `snapshot_download()` rather than the CLI on purpose — the CLI was
renamed from `huggingface-cli` to `hf` and its flags moved, whereas the Python API
has been stable for years. Re-running is cheap: transfers resume and existing files
are skipped. It also excludes bge-m3's ~2.2 GB ONNX export, which nothing here loads.

Resulting layout (8.8 GB):

```
models\bge-m3\                                        2.2 GB   (pytorch_model.bin — this repo has no safetensors)
models\bge-reranker-v2-m3\                            2.2 GB
models\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf 3.8 GB
models\qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf 658 MB
models\lid.176.bin                                    126 MB
```

> **The Qwen GGUF arrives SPLIT into two shards.** `config.yaml` names the logical
> single file `qwen2.5-7b-instruct-q4_k_m.gguf`, which does not exist on disk —
> `llm/local_llm.py`'s `resolve_gguf_path()` finds the `-00001-of-00002` shard, and
> llama.cpp reads `split.count` from its header and pulls in the second. Both a split
> and a merged single-file layout work with the shipped config, unchanged.

After this step, set the environment variable so the app never attempts a
network call again:
```powershell
setx HF_HUB_OFFLINE 1
```

---

## 6. Verify the environment

Run the built-in diagnostic script before doing anything else — it loads
every model/service and reports pass/fail individually, so a broken install
step surfaces immediately instead of deep inside a pipeline run:

```powershell
python scripts/setup_check.py
```

Expected output:
```
[OK]   BGE-M3 embedding model
[OK]   bge-reranker-v2-m3
[OK]   Local LLM (llama-cpp-python)
[OK]   LanceDB
[OK]   SQLite
[OK]   fastText language ID
[OK]   Tesseract binary
[OK]   GROBID service (optional)   <- may show [FAIL] if you skipped GROBID; this alone won't fail the script

7/7 checks passed.
```
The script exits non-zero if anything other than GROBID fails, since GROBID
is explicitly optional.

---

## 7. Running the application

### 7.1 Desktop app (primary entry point)

```powershell
python main.py
```
This opens the PySide6 window with four views (Ingest / Search / Taxonomy / Reports)
and calls `sqlite_store.init_db()` on startup, so the database schema is
created automatically on first run.

### 7.2 One-time setup workflow (run once, in order)

```powershell
# 1. Confirm every model/service loads
python scripts/setup_check.py

# 2. Build the fixed taxonomy from a sample of your existing corpus.
#    One number per hierarchy level, coarsest first: "6 20 60" means 6 Domains,
#    each split into up to 20 Fields, each split into up to 60 Subfields.
python scripts/build_taxonomy.py --folder "C:\path\to\existing\corpus" --sample-size 3000 --level-cuts 6 20 60

# Review/edit the generated data/taxonomy_review.json, then apply your edits
# (directly via taxonomy_store.py, or the Taxonomy view in the UI)

# 3. Classify the entire existing corpus in bulk
python scripts/bulk_init_classify.py --folder "C:\path\to\existing\corpus"
```
> **How the levels nest:** each level is built by clustering *within* its parent's
> own member documents, which is what guarantees a Field stays inside one Domain.
> Clusters smaller than `MIN_DOCUMENTS_TO_SPLIT` (10) are left as leaves rather
> than split into categories that describe individual documents.
>
> Pass `--sample-seed` to make the corpus sample reproducible across re-runs, and
> `--review-output` to put the review sheet somewhere other than `data/`.

### 7.3 Recurring workflow (run monthly, or on your own schedule)

```powershell
python scripts/monthly_ingest.py
```
This ingests everything currently in `data/inbox/`, classifies it, and
generates the monthly report — using the same `pipeline.process_document()`
function the UI's Ingest view calls, so behavior never drifts between the two entry points.

### 7.4 Running the test suite

The tests are split so none of them need a GPU, a model download, or a running
service:
```powershell
pip install pytest
pytest tests/ -v
```
| File | What it covers | Needs |
|---|---|---|
| `test_storage_filters.py` | SQL-literal quoting/escaping for LanceDB filters | nothing — runs on a bare Python install |
| `test_ingestion.py` | chunk overlap, pooling threshold, content hashing | `pyyaml` |
| `test_taxonomy.py` | full-path resolution, orphan/cycle handling | `pyyaml`, `pyarrow` |
| `test_classification.py` | cosine ranking, LLM tie-break path composition, status mapping, vector cache | `numpy`, `pyarrow`, `lancedb` |
| `test_search.py` | reranker sort order and `top_n` truncation | `FlagEmbedding` importable (the model itself is monkeypatched away) |

`test_classification.py` fakes the tie-breaker by substituting the module in
`sys.modules`, so llama-cpp is never imported.

---

## 8. Packaging into a fully offline Windows `.exe`

> **Status: written, not yet executed.** Every file below exists and the dependency
> set is verified installable with zero network, but producing an actual `.exe`
> requires a Python 3.12 install, which hasn't happened here yet.

```
build/
├── main.spec               # PyInstaller build spec (--onedir/"standalone" mode)
├── build.ps1               # wraps the whole process: env check → clean → build → assemble
├── README_BUILD.md         # gotchas specific to this project's dependency stack
└── hooks/
    ├── hook-llama_cpp.py   # collects llama_cpp's ctypes-loaded backend DLLs
    └── hook-lancedb.py     # collects lancedb's native Rust extension
```

The two hooks are the non-obvious part, and both fail *at runtime* rather than at
build time, which is what makes them nasty. PyInstaller's static analysis cannot see
a library loaded through `ctypes` (llama.cpp's `llama.dll`, resolved from a path
computed relative to the package) or a compiled Rust extension resolved by path
(lancedb's, which lives in the companion `lance`/`pylance` distribution). Without
the hooks the build succeeds, then the first LLM call dies with
`Shared library with base name 'llama' not found` and the first `lancedb.connect()`
raises ImportError.

Read `build/README_BUILD.md` before debugging a build — most first-build failures are
one of the known items listed there.

### 8.1 Required source fix (applied)

`src/docclassify/config.py` computed the project root as
`Path(__file__).resolve().parents[2]`. That only works when the file is on
disk at `src/docclassify/config.py` — once PyInstaller extracts it into its
own bundle layout, that no longer points at the folder containing `config/`,
`models/`, and `data/`. It now detects `sys.frozen` (set by
PyInstaller at runtime) and anchors to the `.exe`'s own directory instead:

```python
def _resolve_project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]
```
Verified directly: in dev mode this still resolves to the project root
exactly as before; with `sys.frozen`/`sys.executable` simulated, it correctly
resolves to the simulated `.exe`'s folder instead. If you pull `config.py`
from an older copy of this project, re-apply this fix first.

### 8.2 Build it

```powershell
.venv\Scripts\activate
powershell -ExecutionPolicy Bypass -File build\build.ps1 -Console -SkipEnvCheck
```

| Flag | Why |
|---|---|
| `-Console` | **Use this for a first build.** A windowed PyInstaller app discards stdout and stderr, so a missing import shows up as a window that simply never appears. |
| `-SkipEnvCheck` | Skips `setup_check.py`, which loads every model and so needs `models\` populated on the *build* machine. Skip it if you'll populate models on the target. |
| `-KeepWork` | Keeps `build\work\`, which holds `warn-docclassify.txt` (every unresolved module) and `xref-docclassify.html` (the import graph). These are what you read when something is missing. |
| `-Python <path>` | Build with an interpreter other than `.venv\Scripts\python.exe`. |

`build.ps1` runs `setup_check.py` (unless skipped), cleans previous output, runs
PyInstaller against `build/main.spec`, then assembles the distributable: `config/`
copied next to the `.exe`, and empty `data/` and `models/` trees created alongside
with a `MODELS_REQUIRED.txt` note listing exactly which files go where. It warns if
the interpreter isn't 3.12 and reports the final size.

Output: `dist\docclassify\` — a self-contained, `--onedir`-packaged app (not
`--onefile`; unpacking GB-scale ML dependencies on every launch is too slow
for a desktop app people open repeatedly).

### 8.3 Finish the offline app

```powershell
# Copy this checkout's models\ folder wholesale into the built app:
Copy-Item -Recurse -Force models\* dist\docclassify\models\
dist\docclassify\docclassify.exe
```
See `dist\docclassify\models\MODELS_REQUIRED.txt`, which `build.ps1` writes with the
exact expected layout.
Once `models/` is populated, the whole `dist\docclassify\` folder is
self-contained and needs no network access to run — zip it to move it to
another offline machine.

### 8.4 Things to know before you run it

- **First build is slow** (10–20+ minutes) — `torch`, `transformers`,
  `opencv`, and `PySide6` are large; subsequent builds are faster once caches
  are warm.
- **Disable UPX compression** in `main.spec` — compressing
  CUDA/torch DLLs with UPX is a common source of builds that work locally
  and silently break on another machine.
- **GPU DLLs**: if your build environment has CUDA-enabled `torch` and
  `llama-cpp-python`, the spec should pull in the matching CUDA runtime DLLs
  automatically; the *target* machine still needs a compatible NVIDIA
  driver — that part can't be bundled.
- **Make `collect_all()` failures for an optional package non-fatal** — warn and
  continue rather than aborting the whole build, and grep PyInstaller's output for
  those warnings afterwards.

---

## 9. Configuration reference (`config/config.yaml`)

| Key | Meaning | Default |
|---|---|---|
| `models.embedding` | BGE-M3 weights path | `./models/bge-m3` |
| `models.reranker` | bge-reranker-v2-m3 weights path | `./models/bge-reranker-v2-m3` |
| `models.llm_gguf` | Local LLM GGUF file path | `./models/qwen2.5-7b-instruct-q4_k_m.gguf` |
| `models.lang_id` | fastText language ID model | `./models/lid.176.bin` |
| `chunking.chunk_size_tokens` | Words per chunk (approximate) | `400` |
| `chunking.chunk_overlap_tokens` | Overlap between consecutive chunks | `50` |
| `classification.auto_assign_threshold` | Min cosine similarity to auto-assign | `0.72` |
| `classification.ambiguous_gap_threshold` | Min gap between top-1/top-2 to avoid escalation | `0.05` |
| `search.ann_candidate_count` | Candidates pulled before reranking | `50` |
| `search.final_result_count` | Results returned after reranking | `10` |
| `ocr.confidence_flag_threshold` | Below this, OCR output is flagged for review | `60` |
| `metadata.enabled` | Extract title/authors/keywords/date during ingestion at all | `true` |
| `metadata.use_grobid` | Try local GROBID first for PDFs | `true` |
| `metadata.use_llm_fallback` | LLM header extraction when GROBID is absent or finds no title | `true` |
| `metadata.extract_keywords` | KeyBERT keywords when the document lists none | `true` |
| `metadata.translate_to_english` | Fill the `*_en` mirror fields for non-English documents | `true` |
| `metadata.llm_snippet_chars` | How much of the document head the LLM extractor reads | `3000` |

The `metadata.*` LLM-backed steps cost one generation call per document. That is
fine for a monthly batch and far too expensive for a multi-million-document
initial load — turn `use_llm_fallback` (and optionally `translate_to_english`) off
before running `bulk_init_classify.py` over a large corpus.

All relative paths (`./...`) are resolved against the project root at
import time, so the app can be launched from any working directory.

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'pyarrow'` / `'lancedb'` / `'FlagEmbedding'` etc. | `pip install -r requirements.txt` wasn't run in the active venv | Re-activate the venv, reinstall |
| `llama-cpp-python` fails to build | Missing CUDA Toolkit or VS Build Tools | Install both, or drop `CMAKE_ARGS` for a CPU-only (slower) build |
| `setup_check.py` fails only on GROBID | Expected if you didn't start the GROBID container | Safe to ignore — it's optional |
| Embedding/reranker calls hang or OOM | GPU VRAM exhausted (BGE-M3 + reranker + 7B LLM concurrently) | Reduce `n_gpu_layers` in `llm/local_llm.py`, or run components sequentially |
| Language detection returns `unknown` for short strings | By design — `detect_language()` returns `unknown` for text under 5 characters | Not a bug |
| Re-uploading a document creates a duplicate row | `is_duplicate()` hashes normalized *text content*, not the file — different filenames with identical content ARE deduped; check `data/raw/` if you expect dedup and don't see it working | Confirm the file actually re-extracts to identical text (e.g. OCR nondeterminism can produce a near-duplicate that hashes differently) |

---

## 11. Requirements file (for reference)

See `requirements.txt` in the repo. Summary by category:
- **Embeddings/reranking:** FlagEmbedding
- **Parsing:** pymupdf, python-docx, python-pptx, beautifulsoup4
- **OCR:** pytesseract, easyocr, opencv-python
- **Language ID:** fasttext
- **Vector DB:** lancedb, pyarrow
- **Metadata:** keybert, pypinyin
- **Data handling:** numpy, pandas, pyyaml, requests
- **Desktop UI:** pyside6
- **Taxonomy construction:** scikit-learn, hdbscan
- **Installed separately:** llama-cpp-python, vllm, GROBID, Tesseract (see §5.3)
- **Dropped as unused:** `unstructured` (parsing goes through pymupdf/docx/pptx/bs4
  directly) and `umap-learn` (nothing imports it) — both were pulling a large
  dependency tree into the offline bundle for nothing.
