# Performance testing on a vast.ai RTX 5090

Headless, end-to-end evaluation of classification, semantic search and report
generation on a cloud GPU box. No UI involved.

Written for: Ubuntu, RTX 5090 (32 GB), ~30 test documents spanning economics,
biology and mathematics, with a hand-authored taxonomy.

---

## 0. Read this before you rent the instance

Three things will waste your money if you get them wrong at instance-selection time.

### 0.1 The RTX 5090 is Blackwell — pick a CUDA 12.8+ image

The 5090 is compute capability **sm_120**. Software built for older architectures
does not silently fall back; every kernel launch fails with
`no kernel image is available for execution on the device`.

| Component | What works | What silently breaks |
|---|---|---|
| PyTorch | Default PyPI `pip install torch` (CUDA 13 build) | `--index-url .../whl/cu124` — tops out at torch 2.6 / CUDA 12.4, **no sm_120 kernels** |
| llama-cpp-python | Compiled with `-DCMAKE_CUDA_ARCHITECTURES=120` | The prebuilt cu124 wheels — newest Linux index available, still no sm_120 |

> The Windows offline bundle in `packages/` uses the cu124 index. That is correct for
> the older cards it was built for and **wrong for a 5090**. `setup_linux.sh` does the
> right thing; don't port the Windows commands across.

When choosing the vast.ai instance, filter for a **CUDA driver version of 12.8 or
newer** and pick a PyTorch or CUDA-devel base image. You need `nvcc` on the box to
compile llama-cpp-python.

### 0.2 The Windows wheelhouse is useless here

`packages/wheels/` holds 118 `win_amd64` wheels. On Linux they cannot install.
A cloud box has internet, so install from PyPI — that is what `setup_linux.sh` does.

Do **not** copy `packages/` to the instance. It is 1.2 GB of dead weight.

### 0.3 Provision enough disk

| Item | Size |
|---|---|
| Model weights | 8.8 GB |
| Python environment (torch, CUDA libs, transformers) | ~8 GB |
| Test documents + database | small |
| Headroom for pip caches and the llama.cpp build | ~5 GB |
| **Minimum recommended** | **30 GB** |

vast.ai instances default to a small disk. Increase it on the rental screen — you
cannot resize afterwards without losing the instance.

### 0.4 Python version

On Linux the project runs on **3.10, 3.11 or 3.12**. `setup_linux.sh` picks the
newest available in that range.

(The stricter "3.12 exactly" rule in the main README is a Windows-only constraint:
`fasttext-wheel` publishes cp310–cp312 wheels for Linux but only up to cp312 for
Windows.)

---

## 1. Get the code and data onto the instance

From your local machine:

```bash
# vast.ai gives you a host and port; adjust to match
export VAST="ssh -p 12345 root@ssh5.vast.ai"
export VAST_SCP="scp -P 12345"

# The repo, minus the things that must not travel:
#   packages/  - Windows wheels, useless here
#   models/    - 8.8 GB, far faster to re-download on the instance
#   .venv/     - platform specific
rsync -av --progress \
      --exclude 'packages/' --exclude 'models/' --exclude '.venv/' \
      --exclude '__pycache__/' --exclude 'data/sqlite/' --exclude 'data/lancedb/' \
      -e "ssh -p 12345" \
      ./document-classifier/ root@ssh5.vast.ai:/workspace/document-classifier/
```

No `rsync`? `tar` works just as well:

```bash
tar --exclude=packages --exclude=models --exclude=.venv --exclude=__pycache__ \
    -czf dc.tar.gz document-classifier/
$VAST_SCP dc.tar.gz root@ssh5.vast.ai:/workspace/
$VAST "cd /workspace && tar xzf dc.tar.gz"
```

### Test document layout

Put the 30 documents in **per-category folders**. The benchmark derives ground
truth from the folder names, so this gets you accuracy measurement for free:

```
data/testset/
├── Economics/
│   ├── monetary_policy_review.pdf
│   └── ...
├── Biology/
│   └── ...
└── Mathematics/
    └── ...
```

For two-level ground truth, nest one deeper — `data/testset/Biology/Genetics and
Genomics/paper.pdf` yields the expected path `Biology/Genetics and Genomics`.
Folder names must match the taxonomy names exactly.

Prefer not to reorganise your files? Supply a CSV instead and pass `--labels`:

```csv
monetary_policy_review.pdf,Economics/Macroeconomics and Policy
gene_regulation_study.pdf,Biology/Molecular and Cell Biology
```

```bash
$VAST_SCP -r ./testset root@ssh5.vast.ai:/workspace/document-classifier/data/testset
```

---

## 2. Set up the environment

```bash
ssh -p 12345 root@ssh5.vast.ai
cd /workspace/document-classifier

nvidia-smi                 # confirm the GPU and driver version
nvcc --version             # confirm the CUDA toolkit is present

bash scripts/setup_linux.sh
```

The script installs system packages (`tesseract-ocr`, plus `libgl1` and
`libglib2.0-0`, which OpenCV links against — without them `import cv2` fails and
takes the whole ingestion path down), creates `.venv`, installs PyTorch **first**
and hard-fails if that build has no kernels for your GPU, installs the project
dependencies without PySide6, and compiles llama-cpp-python for your GPU's exact
architecture.

Expect 10–20 minutes, most of it the llama.cpp compile.

If `nvcc` is missing:

```bash
export PATH=/usr/local/cuda/bin:$PATH
bash scripts/setup_linux.sh
```

Falling back to a CPU-only LLM is survivable — classification and search stay on the
GPU, and only the tie-breaker and report generation slow down:

```bash
bash scripts/setup_linux.sh --cpu-llm
```

---

## 3. Download the models

```bash
source .venv/bin/activate
python scripts/fetch_models.py          # 8.8 GB
```

Then verify everything loads before going further:

```bash
python scripts/setup_check.py
```

Expect 7/8 passes — GROBID will fail unless you started the container, and that
alone does not fail the run. To skip its probe entirely, set
`metadata.use_grobid: false` in `config/config.yaml`.

Once the models are down, forbid any further network call:

```bash
export HF_HUB_OFFLINE=1
```

---

## 4. Create the taxonomy

You are supplying the taxonomy by hand rather than clustering it, which is the right
call for 30 documents — `build_taxonomy.py` needs roughly a thousand to produce
sensible clusters.

`config/taxonomy_manual.json` already contains a three-domain, nine-field tree for
economics, biology and mathematics. Check it and edit the names to match your
documents:

```bash
python scripts/import_taxonomy.py --file config/taxonomy_manual.json --dry-run
```

`--dry-run` validates without loading a model — it catches missing names, duplicate
siblings, unusably short descriptions and missing seed files. It also runs on a
machine with none of the dependencies installed, so you can check your edits locally
before uploading.

Then import for real:

```bash
python scripts/import_taxonomy.py --file config/taxonomy_manual.json --replace
```

### The one thing that most affects your results

With a hand-written taxonomy and no seed documents, **a category's vector is just
its description embedded**. A short description sits far from a real document in
vector space, which depresses every similarity score and pushes documents into the
LLM tie-breaker. Two mitigations, in order of effectiveness:

1. **Give each category two or three seed documents.** Even one helps a lot:
   ```json
   {
     "name": "Macroeconomics and Policy",
     "description": "...",
     "seed_documents": ["data/seeds/inflation_paper.pdf"]
   }
   ```
   The stored vector then blends the description embedding with the seed centroid.
   If you can spare 3 of your 30 documents as seeds, that is a good trade — just
   exclude them from the accuracy measurement afterwards.

2. **Write descriptions like an abstract, not a dictionary definition.** Several
   sentences, using the vocabulary the real documents use. The shipped file does
   this deliberately: every description is 44–67 words. Compare
   *"Study of money and markets"* against the shipped Economics entry.

Expect the default `auto_assign_threshold` of 0.72 to be too high for a
description-only taxonomy. Do not tune it blind — the benchmark reports the actual
score distribution and suggests a value (§6.2).

---

## 5. Run the benchmark

```bash
python scripts/benchmark.py --folder data/testset --batch 2026-08 --json results.json
```

One run measures everything:

| Phase | What you learn |
|---|---|
| 1. Environment | GPU, driver, compute capability, and **whether this torch build actually has kernels for your GPU** |
| 2. Model load | Load time and VRAM cost per model, from both torch's and the driver's view |
| 3. LLM throughput | Approximate tokens/second — the number that governs tie-break and report cost |
| 4. Ingestion | Per-document wall time, throughput in docs/minute, failures |
| 5. Stage breakdown | Where the time actually goes: parse / OCR / language / metadata / embed / classify / persist |
| 6. Classification | Status and category distribution, the similarity distribution, and accuracy against ground truth |
| 7. Search | Latency split into embed / ANN / rerank, with P50 and P95 |
| 8. Report | Monthly digest generation time and per-LLM-call cost, plus the digest itself |

Useful variations:

```bash
# Re-run search and report without re-ingesting
python scripts/benchmark.py --skip-ingest --folder data/testset

# Your own queries, one per line
python scripts/benchmark.py --folder data/testset --queries my_queries.txt

# Ground truth from a CSV rather than folder names
python scripts/benchmark.py --folder data/testset --labels labels.csv

# Ingestion only — fastest iteration when tuning thresholds
python scripts/benchmark.py --folder data/testset --skip-search --skip-report
```

`--json results.json` writes everything, including per-document timings and the list
of misclassifications, for comparing runs later.

---

## 6. Interpreting what you get

### 6.1 Stage breakdown

Stages nest — `classify` includes the LLM tie-breaker, and `metadata` may include an
LLM call — so the shares sum to more than 100%. That is expected and labelled in the
output.

What to look for:

- **`embed_chunks` dominating** is normal and healthy. It is the real work.
- **`metadata` dominating** means the LLM fallback is firing on most documents. Set
  `metadata.use_llm_fallback: false` and re-run to isolate it.
- **`classify` dominating** means most documents are escalating to the tie-breaker —
  go to §6.2, your threshold is mistuned for a description-only taxonomy.
- **`ocr` non-zero** means some PDFs have no text layer. Check those specific
  documents; OCR is by far the most expensive stage per page.

### 6.2 Choosing the threshold

Phase 6 prints the top-1 similarity distribution and how many documents fell below
`auto_assign_threshold`. If more than half escalated, it suggests a lower value.

```yaml
# config/config.yaml
classification:
  auto_assign_threshold: 0.55     # from the benchmark's suggestion
  ambiguous_gap_threshold: 0.05
```

Then re-run. Because the pipeline deduplicates on content hash, a straight re-run
will skip every document as a duplicate — clear the database first:

```bash
rm -rf data/sqlite/app.db data/lancedb
python scripts/import_taxonomy.py --file config/taxonomy_manual.json --replace
python scripts/benchmark.py --folder data/testset --json results_t055.json
```

Sweep a few values and compare the exact-accuracy figures. With 30 documents, treat
differences of one or two documents as noise, not signal.

### 6.3 What accuracy to expect

With 30 documents, three domains and a description-only taxonomy:

- **Top-level (domain) accuracy** should be high. Economics, biology and mathematics
  are far apart in embedding space; if this is poor, something is wrong — check that
  the category names in your folders match the taxonomy exactly.
- **Exact path accuracy** will be noticeably lower. Distinguishing
  "Macroeconomics and Policy" from "Econometrics and Finance" is genuinely hard from
  a description alone, and this is exactly where seed documents pay off.

Read the `misclassified` list in the JSON. If a document lands in a sibling of the
right category, that is a description-quality problem. If it lands in a different
domain entirely, look at whether the document parsed correctly at all.

### 6.4 Search

30 documents is a very small index. Take from the search phase:

- **Latency composition** — the split between embed, ANN and rerank is
  representative and will hold at larger scale. Rerank should dominate.
- **ANN latency is not representative.** With a few hundred chunks LanceDB does a
  brute-force scan; at a million chunks it uses an index. This number will not
  extrapolate.
- **Result quality** is meaningful. The default query set deliberately includes a
  Chinese query against an English corpus — BGE-M3 is multilingual, so relevant
  English documents should come back for it. If they do not, cross-lingual alignment
  is broken, which would be a significant finding.

### 6.5 Report generation

Cost is `(number of categories with documents) + 1` LLM calls. With 30 documents
across 9 categories that is up to 10 calls. Per-call time here is your best estimate
for what a real monthly batch will cost.

Read the digest. It is generated from per-category summaries rather than from raw
documents, so it is a fair test of whether that two-stage summarisation holds
together.

---

## 7. Testing the monthly workflow properly

The benchmark tags documents with `--batch`. To exercise the actual production path
instead:

```bash
rm -rf data/sqlite/app.db data/lancedb
python scripts/import_taxonomy.py --file config/taxonomy_manual.json --replace

cp -r data/testset/* data/inbox/
python scripts/monthly_ingest.py
```

This runs ingestion, classification and report generation exactly as a scheduled run
would, tagging the batch `YYYY-MM`. It is the closest thing to a production
rehearsal available without the UI.

---

## 8. Ad-hoc search

```bash
python scripts/search_cli.py "how does monetary policy affect inflation"
python scripts/search_cli.py "gene regulation" --top-k 10 --category Biology
python scripts/search_cli.py "货币政策对通货膨胀的影响" --show-text
python scripts/search_cli.py --interactive
```

Unlike the UI, this resolves `doc_id` to filenames, so results are readable.
`--category` takes a path prefix, so `--category Biology` also matches
`Biology/Genetics and Genomics`.

---

## 9. Inspecting the database directly

```bash
sqlite3 data/sqlite/app.db

.headers on
.mode column
SELECT filename, category_path, status, round(confidence,3) FROM documents;
SELECT status, count(*) FROM documents GROUP BY status;
SELECT r.doc_id, d.filename, r.reason FROM review_queue r
  JOIN documents d ON d.doc_id = r.doc_id WHERE r.resolved = 0;
SELECT filename, title_en, keywords_en FROM documents LIMIT 5;
```

The last query is the quickest check on whether metadata extraction actually worked.
If `title_en` is NULL everywhere, either `metadata.enabled` is false or the LLM
fallback is failing — check the ingestion output for `[metadata]` warnings.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `no kernel image is available for execution on the device` | torch or llama.cpp built without sm_120 | Reinstall torch from plain PyPI; rebuild llama-cpp-python with `-DCMAKE_CUDA_ARCHITECTURES=120` |
| `ImportError: libGL.so.1` | OpenCV needs system GL libraries | `apt-get install -y libgl1 libglib2.0-0` |
| `TesseractNotFoundError` | The binary is not installed | `apt-get install -y tesseract-ocr` |
| Benchmark exits: "taxonomy is empty" | No categories imported | Run `import_taxonomy.py` first |
| Every document returns `needs_review` | Threshold too high for a description-only taxonomy | §6.2 |
| Re-run ingests 0 documents | Content-hash dedup is working | Delete `data/sqlite/app.db` and `data/lancedb/` |
| llama.cpp compile fails, out of memory | Parallel compile exhausts RAM | `export CMAKE_BUILD_PARALLEL_LEVEL=4` and retry |
| `Shared library ... 'llama' not found` | The CUDA build failed and left a broken install | `pip uninstall -y llama-cpp-python`, then reinstall |
| Report generation very slow | LLM on CPU | Check phase 2 — if the LLM added no VRAM, it is not on the GPU |
| VRAM climbs across documents | EasyOCR loaded as an OCR fallback | Normal; it stays resident once loaded |

---

## 11. What this test does and does not establish

**Establishes:** that the pipeline runs end to end on real documents; per-stage cost
and where time goes; VRAM headroom with all three models resident; search latency
composition; cross-lingual retrieval; report coherence; and whether classification is
usable with a hand-authored taxonomy.

**Does not establish:**

- **Throughput at scale.** 30 documents will not surface the batching behaviour,
  the multiprocess parsing path, or the memory profile of `bulk_init_classify.py`.
  The monthly path (`process_folder`) is sequential and single-threaded by design.
- **ANN performance.** Too few chunks for LanceDB to build or use an index.
- **Classification accuracy in any statistical sense.** With 30 documents the
  confidence interval on an accuracy figure is very wide. Treat the result as a smoke
  test and a threshold-tuning aid, not a measurement.
- **OCR quality**, unless some of your PDFs genuinely lack a text layer. If none do,
  the OCR path — including the deskew correction, which has never been exercised
  against a real scan — remains untested. Deliberately including two or three scanned
  PDFs is worthwhile.
- **Long-document behaviour.** Pooling only engages above 6,000 words. If none of
  your documents cross that, `mean_pool` never runs. Worth including one long
  document on purpose.
