r"""
End-to-end performance and quality harness. Headless — no Qt, no display needed.

Measures, in one run:
  * environment: GPU, driver, CUDA capability, torch build
  * model load time and VRAM cost, per model
  * ingestion throughput with a per-stage timing breakdown (parse / OCR / language /
    metadata / embed / classify / persist)
  * classification quality against ground truth, plus the similarity-score
    distribution you need in order to choose thresholds
  * search latency, split into embed / ANN / rerank, with P50 and P95
  * LLM generation throughput in tokens per second
  * monthly report generation time

Usage:
    python scripts/benchmark.py --folder data/testset
    python scripts/benchmark.py --folder data/testset --queries queries.txt --json results.json
    python scripts/benchmark.py --folder data/testset --skip-ingest      # reuse a previous run

GROUND TRUTH for classification accuracy comes from either:
  * the folder layout — data/testset/Economics/paper1.pdf means the expected
    top-level category is "Economics" (nested folders become a path:
    testset/Biology/Genetics and Genomics/x.pdf -> "Biology/Genetics and Genomics"), or
  * --labels labels.csv, with rows "filename,expected/category/path".
If neither is available the run still reports timings and the score distribution,
just without an accuracy figure.

Stage timings work by wrapping the functions pipeline.py has already imported into
its own namespace. Nothing in the production path is modified.
"""
import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Documents, category names and generated summaries in this project are multilingual
# by design. Force UTF-8 on the console: Windows defaults to cp1252, which raises
# UnicodeEncodeError the moment a CJK character is printed.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


EXTENSIONS = (".pdf", ".docx", ".pptx", ".html", ".htm", ".txt", ".md")

# Collected by the wrappers installed in instrument_pipeline().
STAGE_TIMES: dict[str, list[float]] = defaultdict(list)
STAGE_COUNTS: dict[str, int] = defaultdict(int)


# ------------------------------------------------------------------ utilities

def human(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    return f"{int(seconds // 60)}m {seconds % 60:.1f}s"


def rule(title: str = "") -> None:
    print()
    if title:
        print(f"--- {title} " + "-" * max(0, 68 - len(title)))
    else:
        print("-" * 72)


def gpu_memory_mb() -> float | None:
    """Allocated VRAM via torch, if torch is present and CUDA is up."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.memory_allocated() / 1e6
    except Exception:  # noqa: BLE001
        return None


def gpu_memory_peak_mb() -> float | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.max_memory_allocated() / 1e6
    except Exception:  # noqa: BLE001
        return None


def nvidia_smi_used_mb() -> float | None:
    """
    Total VRAM in use on the device, from the driver. This includes llama.cpp's
    allocation, which torch cannot see — the two numbers together are what tell you
    whether three models actually fit alongside each other.
    """
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0:
            return float(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        pass
    return None


# ------------------------------------------------------------------ instrumentation

def instrument_pipeline() -> None:
    """
    Wraps the stage functions with timers.

    pipeline.py does `from X import y`, so the callable lives in pipeline's own
    module namespace — patching `docclassify.pipeline.y` is what takes effect, not
    patching X.y.
    """
    import docclassify.pipeline as pipeline

    def timed(module, attribute: str, label: str):
        original = getattr(module, attribute)

        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                STAGE_TIMES[label].append(time.perf_counter() - start)
                STAGE_COUNTS[label] += 1

        setattr(module, attribute, wrapper)

    timed(pipeline, "parse_document", "parse")
    timed(pipeline, "ocr_flagged_pages", "ocr")
    timed(pipeline, "detect_language", "language_id")
    timed(pipeline, "detect_languages_per_chunk", "language_id_chunks")
    timed(pipeline, "extract_metadata", "metadata")
    timed(pipeline, "chunk_text", "chunking")
    timed(pipeline, "embed_texts", "embed_chunks")
    timed(pipeline, "embed_text", "embed_document")
    timed(pipeline, "classify_document", "classify")
    timed(pipeline, "is_duplicate", "dedup")

    import docclassify.storage.lancedb_store as lancedb_store
    import docclassify.storage.sqlite_store as sqlite_store
    timed(lancedb_store, "upsert_doc_vector", "persist_vectors")
    timed(lancedb_store, "upsert_chunk_vectors", "persist_vectors")
    timed(sqlite_store, "upsert_document", "persist_metadata")


# ------------------------------------------------------------------ ground truth

def expected_labels(folder: Path, labels_csv: str | None) -> dict[str, str]:
    """Maps filename -> expected category path."""
    if labels_csv:
        mapping = {}
        with open(labels_csv, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.reader(f):
                if len(row) >= 2 and row[0].strip() and not row[0].lstrip().startswith("#"):
                    mapping[Path(row[0].strip()).name] = row[1].strip()
        return mapping

    # Otherwise derive from directory structure relative to the corpus root.
    # The folder may legitimately be absent under --skip-ingest.
    mapping = {}
    if not folder.is_dir():
        return mapping
    for path in folder.rglob("*"):
        if path.suffix.lower() not in EXTENSIONS:
            continue
        relative = path.relative_to(folder).parts[:-1]
        if relative:
            mapping[path.name] = "/".join(relative)
    return mapping


# ------------------------------------------------------------------ phases

def phase_environment() -> dict:
    rule("1. Environment")
    info: dict = {}

    import platform
    info["python"] = platform.python_version()
    info["platform"] = platform.platform()
    print(f"  Python           {info['python']}")
    print(f"  Platform         {info['platform']}")

    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        print(f"  torch            {torch.__version__}")
        print(f"  CUDA available   {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu"] = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            info["compute_capability"] = f"{capability[0]}.{capability[1]}"
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            info["vram_gb"] = round(total, 1)
            print(f"  CUDA (torch)     {torch.version.cuda}")
            print(f"  GPU              {info['gpu']}")
            print(f"  Compute cap.     sm_{capability[0]}{capability[1]}")
            print(f"  VRAM             {total:.1f} GB")

            supported = torch.cuda.get_arch_list()
            info["arch_list"] = supported
            arch_tag = f"sm_{capability[0]}{capability[1]}"
            if arch_tag not in supported and f"compute_{capability[0]}{capability[1]}" not in supported:
                print(f"  !! This torch build lists {supported}")
                print(f"  !! It does NOT include {arch_tag}. Kernels will fail to launch.")
                print("  !! Install a torch built for this GPU before trusting any number below.")
    except ImportError:
        print("  torch            NOT INSTALLED")

    used = nvidia_smi_used_mb()
    if used is not None:
        print(f"  VRAM in use now  {used:.0f} MB (driver view, before loading anything)")
        info["vram_used_at_start_mb"] = used
    return info


def phase_model_load() -> dict:
    rule("2. Model load time and VRAM cost")
    results = {}

    def load(label, fn):
        before_torch = gpu_memory_mb() or 0.0
        before_driver = nvidia_smi_used_mb() or 0.0
        start = time.perf_counter()
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"  {label:22} FAILED: {type(e).__name__}: {str(e)[:80]}")
            results[label] = {"error": f"{type(e).__name__}: {e}"}
            return
        elapsed = time.perf_counter() - start
        after_torch = gpu_memory_mb() or 0.0
        after_driver = nvidia_smi_used_mb() or 0.0
        results[label] = {
            "load_seconds": round(elapsed, 2),
            "torch_delta_mb": round(after_torch - before_torch, 1),
            "driver_delta_mb": round(after_driver - before_driver, 1),
        }
        print(f"  {label:22} {human(elapsed):>10}   "
              f"+{after_driver - before_driver:7.0f} MB VRAM (driver)")

    def load_embedder():
        from docclassify.embeddings.embedder import embed_text
        embed_text("warm up the embedding model")

    def load_reranker():
        from docclassify.search.reranker import rerank
        rerank("warm up", [{"chunk_text": "a candidate passage"}], top_n=1)

    def load_llm():
        from docclassify.llm.local_llm import generate
        generate("Reply with OK.", max_tokens=4)

    load("BGE-M3 embedder", load_embedder)
    load("bge-reranker-v2-m3", load_reranker)
    load("Qwen2.5-7B (llama.cpp)", load_llm)

    total = nvidia_smi_used_mb()
    if total is not None:
        print(f"\n  All three resident:    {total:.0f} MB VRAM total (driver view)")
        results["all_resident_mb"] = total
    return results


def phase_llm_throughput(max_tokens: int = 256) -> dict:
    rule("3. LLM generation throughput")
    from docclassify.llm.local_llm import generate

    prompt = (
        "Write a detailed paragraph explaining what a document classification "
        "taxonomy is and why a fixed hierarchy is useful. Be thorough."
    )
    start = time.perf_counter()
    output = generate(prompt, max_tokens=max_tokens)
    elapsed = time.perf_counter() - start

    # Word count is a proxy; llama.cpp does not expose token counts through this
    # wrapper. ~1.3 tokens per word is a reasonable English approximation.
    words = len(output.split())
    approx_tokens = int(words * 1.3)
    print(f"  generated ~{approx_tokens} tokens in {human(elapsed)}")
    print(f"  throughput      ~{approx_tokens / elapsed:.1f} tokens/s (approximate)")
    return {
        "seconds": round(elapsed, 2),
        "approx_tokens": approx_tokens,
        "approx_tokens_per_second": round(approx_tokens / elapsed, 1),
    }


def phase_ingest(folder: Path, batch: str) -> dict:
    rule("4. Ingestion")
    from docclassify.pipeline import process_document

    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not files:
        raise SystemExit(f"No documents with extensions {EXTENSIONS} under {folder}")

    print(f"  {len(files)} documents from {folder}\n")
    per_document = []
    failures = []
    records = []

    overall_start = time.perf_counter()
    for index, path in enumerate(files, start=1):
        start = time.perf_counter()
        try:
            record = process_document(str(path), upload_batch=batch)
            elapsed = time.perf_counter() - start
            records.append(record)
            per_document.append({"file": path.name, "seconds": elapsed,
                                  "category": record.get("category_path"),
                                  "status": record.get("status"),
                                  "confidence": record.get("confidence")})
            status = (record.get("status") or "?")[:14]
            category = (record.get("category_path") or "(none)")[:38]
            print(f"  [{index:3}/{len(files)}] {human(elapsed):>9}  {status:<14} {category:<38} {path.name[:30]}")
        except Exception as e:  # noqa: BLE001
            elapsed = time.perf_counter() - start
            failures.append({"file": path.name, "error": f"{type(e).__name__}: {e}"})
            print(f"  [{index:3}/{len(files)}] {human(elapsed):>9}  FAILED  {type(e).__name__}: {str(e)[:60]}")
    total_elapsed = time.perf_counter() - overall_start

    durations = [d["seconds"] for d in per_document]
    summary = {
        "documents_attempted": len(files),
        "documents_succeeded": len(per_document),
        "documents_failed": len(failures),
        "total_seconds": round(total_elapsed, 2),
        "failures": failures,
        "per_document": per_document,
    }
    if durations:
        summary.update({
            "mean_seconds": round(statistics.mean(durations), 3),
            "median_seconds": round(statistics.median(durations), 3),
            "min_seconds": round(min(durations), 3),
            "max_seconds": round(max(durations), 3),
            "docs_per_minute": round(60 * len(durations) / total_elapsed, 1),
        })
        print(f"\n  total          {human(total_elapsed)}")
        print(f"  per document   mean {human(summary['mean_seconds'])}, "
              f"median {human(summary['median_seconds'])}, "
              f"min {human(summary['min_seconds'])}, max {human(summary['max_seconds'])}")
        print(f"  throughput     {summary['docs_per_minute']} docs/minute")
        peak = gpu_memory_peak_mb()
        if peak:
            print(f"  torch VRAM peak {peak:.0f} MB")
            summary["torch_vram_peak_mb"] = round(peak, 1)
    return summary


def phase_stage_breakdown(document_count: int) -> dict:
    rule("5. Per-stage breakdown")
    if not STAGE_TIMES:
        print("  (no stage data — ingestion was skipped)")
        return {}

    rows = []
    grand_total = sum(sum(v) for v in STAGE_TIMES.values())
    for label, times in sorted(STAGE_TIMES.items(), key=lambda kv: -sum(kv[1])):
        total = sum(times)
        rows.append({
            "stage": label,
            "calls": len(times),
            "total_seconds": round(total, 3),
            "mean_ms": round(1000 * total / len(times), 1),
            "share_percent": round(100 * total / grand_total, 1) if grand_total else 0.0,
            "per_document_ms": round(1000 * total / document_count, 1) if document_count else 0.0,
        })

    print(f"  {'stage':<20} {'calls':>6} {'total':>10} {'mean':>10} {'/doc':>10} {'share':>7}")
    for row in rows:
        print(f"  {row['stage']:<20} {row['calls']:>6} "
              f"{row['total_seconds']:>9.2f}s {row['mean_ms']:>9.1f}ms "
              f"{row['per_document_ms']:>9.1f}ms {row['share_percent']:>6.1f}%")
    print("\n  Note: stages nest (classify includes the LLM tie-breaker; metadata may")
    print("  include an LLM call), so shares sum to more than 100%.")
    return {"stages": rows}


def phase_classification_quality(folder: Path, labels_csv: str | None) -> dict:
    rule("6. Classification quality")
    from docclassify.storage import sqlite_store

    documents = []
    with sqlite_store.get_connection() as conn:
        for row in conn.execute("SELECT * FROM documents").fetchall():
            documents.append(dict(row))
    if not documents:
        print("  No documents in the database.")
        return {}

    status_counts = Counter(d.get("status") or "?" for d in documents)
    print(f"  {len(documents)} documents classified")
    print("\n  by status:")
    for status, count in status_counts.most_common():
        print(f"    {status:<16} {count:>4}  ({100 * count / len(documents):.0f}%)")

    category_counts = Counter(d.get("category_path") or "(unassigned)" for d in documents)
    print("\n  by assigned category:")
    for category, count in category_counts.most_common():
        print(f"    {count:>4}  {category}")

    confidences = [d["confidence"] for d in documents if d.get("confidence") is not None]
    result: dict = {
        "total": len(documents),
        "by_status": dict(status_counts),
        "by_category": dict(category_counts),
    }
    if confidences:
        confidences.sort()
        result["confidence"] = {
            "min": round(min(confidences), 4),
            "p25": round(confidences[len(confidences) // 4], 4),
            "median": round(statistics.median(confidences), 4),
            "p75": round(confidences[3 * len(confidences) // 4], 4),
            "max": round(max(confidences), 4),
            "mean": round(statistics.mean(confidences), 4),
        }
        c = result["confidence"]
        print("\n  top-1 similarity distribution (this is what the threshold is compared against):")
        print(f"    min {c['min']}   p25 {c['p25']}   median {c['median']}   "
              f"p75 {c['p75']}   max {c['max']}")

        from docclassify.config import CONFIG
        threshold = CONFIG["classification"]["auto_assign_threshold"]
        below = sum(1 for x in confidences if x < threshold)
        print(f"\n    auto_assign_threshold is {threshold}; {below}/{len(confidences)} "
              f"documents scored below it")
        if below > len(confidences) * 0.5:
            suggestion = round(confidences[len(confidences) // 10] - 0.01, 2)
            print(f"    -> More than half escalated. With a hand-written taxonomy and no seed")
            print(f"       documents this is expected: a description embeds further from real")
            print(f"       documents than a cluster centroid does. Consider lowering the")
            print(f"       threshold to about {suggestion}, or adding seed documents.")
            result["suggested_threshold"] = suggestion

    # --- accuracy against ground truth ---
    expected = expected_labels(folder, labels_csv)
    if not expected:
        print("\n  No ground truth available (no per-category folders, no --labels file).")
        print("  Accuracy not computed.")
        return result

    matched = exact = top_level = 0
    confusion = []
    for document in documents:
        want = expected.get(document.get("filename") or "")
        if want is None:
            continue
        matched += 1
        got = document.get("category_path") or ""
        if got == want:
            exact += 1
            top_level += 1
        else:
            if got.split("/")[0] == want.split("/")[0] and got:
                top_level += 1
            confusion.append({"file": document["filename"], "expected": want,
                               "got": got or "(unassigned)",
                               "status": document.get("status")})

    if matched:
        print(f"\n  ground truth matched for {matched}/{len(documents)} documents")
        print(f"    exact path accuracy       {exact}/{matched}  ({100 * exact / matched:.0f}%)")
        print(f"    top-level (domain) only   {top_level}/{matched}  ({100 * top_level / matched:.0f}%)")
        result["accuracy"] = {
            "evaluated": matched,
            "exact": exact,
            "exact_percent": round(100 * exact / matched, 1),
            "top_level": top_level,
            "top_level_percent": round(100 * top_level / matched, 1),
        }
        if confusion:
            print("\n  misclassified:")
            for item in confusion[:20]:
                print(f"    {item['file'][:34]:<34} want {item['expected'][:24]:<24} "
                      f"got {item['got'][:24]}")
            result["misclassified"] = confusion
    return result


def phase_search(queries: list[str], top_k: int) -> dict:
    rule("7. Search latency")
    from docclassify.config import CONFIG
    from docclassify.search.ann_search import search_candidates
    from docclassify.search.query import embed_query
    from docclassify.search.reranker import rerank

    embed_times, ann_times, rerank_times, total_times = [], [], [], []
    samples = []

    for query in queries:
        start = time.perf_counter()
        vector, language = embed_query(query)
        t_embed = time.perf_counter() - start

        start = time.perf_counter()
        candidates = search_candidates(vector, top_k=CONFIG["search"]["ann_candidate_count"])
        t_ann = time.perf_counter() - start

        start = time.perf_counter()
        ranked = rerank(query, candidates, top_n=top_k)
        t_rerank = time.perf_counter() - start

        embed_times.append(t_embed)
        ann_times.append(t_ann)
        rerank_times.append(t_rerank)
        total_times.append(t_embed + t_ann + t_rerank)

        print(f"\n  \"{query[:58]}\"  [{language}]")
        print(f"    embed {t_embed*1000:6.0f} ms | ann {t_ann*1000:6.0f} ms "
              f"({len(candidates)} candidates) | rerank {t_rerank*1000:6.0f} ms "
              f"| total {(t_embed+t_ann+t_rerank)*1000:6.0f} ms")
        for hit in ranked[:3]:
            snippet = " ".join(hit.get("chunk_text", "").split())[:72]
            print(f"      {hit.get('rerank_score', 0):.4f}  {snippet}")
        samples.append({
            "query": query, "language": language, "candidates": len(candidates),
            "results": [{"score": h.get("rerank_score"), "doc_id": h.get("doc_id"),
                          "category": h.get("category_path")} for h in ranked[:top_k]],
        })

    def stats(values):
        values = sorted(values)
        return {
            "mean_ms": round(1000 * statistics.mean(values), 1),
            "p50_ms": round(1000 * statistics.median(values), 1),
            "p95_ms": round(1000 * values[min(len(values) - 1, int(0.95 * len(values)))], 1),
        }

    print(f"\n  over {len(queries)} queries:")
    result = {"queries": len(queries)}
    for label, values in (("embed", embed_times), ("ann", ann_times),
                           ("rerank", rerank_times), ("total", total_times)):
        s = stats(values)
        result[label] = s
        print(f"    {label:<8} mean {s['mean_ms']:>7.1f} ms   p50 {s['p50_ms']:>7.1f} ms   "
              f"p95 {s['p95_ms']:>7.1f} ms")
    result["samples"] = samples
    return result


def phase_report(batch: str) -> dict:
    rule("8. Monthly report generation")
    from docclassify.reports.monthly_report import generate_monthly_report

    start = time.perf_counter()
    report = generate_monthly_report(batch)
    elapsed = time.perf_counter() - start

    categories = report["stats"]["by_category"]
    print(f"  batch '{batch}': {report['stats']['total_documents']} documents, "
          f"{len(categories)} categories")
    print(f"  generated in {human(elapsed)} "
          f"({len(categories)} category summaries + 1 overall digest = "
          f"{len(categories) + 1} LLM calls)")
    if categories:
        print(f"  ~{elapsed / (len(categories) + 1):.1f} s per LLM call")
    print("\n  --- overall digest ---")
    for line in report["overall_digest"].splitlines():
        print(f"  {line}")
    print("\n  stats:", json.dumps(report["stats"], ensure_ascii=False))

    return {
        "seconds": round(elapsed, 2),
        "llm_calls": len(categories) + 1,
        "seconds_per_call": round(elapsed / (len(categories) + 1), 2) if categories else None,
        "stats": report["stats"],
        "overall_digest": report["overall_digest"],
    }


DEFAULT_QUERIES = [
    "How does monetary policy affect inflation?",
    "gene expression regulation in eukaryotic cells",
    "convergence properties of infinite series",
    "impact of biodiversity loss on ecosystem stability",
    "estimating causal effects with instrumental variables",
    "prime number distribution and the Riemann zeta function",
    "货币政策对通货膨胀的影响",
    "protein folding and enzyme catalysis",
]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--folder", default="data/testset", help="Folder of test documents")
    parser.add_argument("--batch", default="bench", help="upload_batch id to tag these documents with")
    parser.add_argument("--labels", help="CSV of filename,expected/category/path")
    parser.add_argument("--queries", help="Text file of search queries, one per line")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", help="Write the full results to this JSON file")
    parser.add_argument("--skip-ingest", action="store_true",
                         help="Reuse documents already in the database")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--skip-search", action="store_true")
    parser.add_argument("--llm-tokens", type=int, default=256,
                         help="Tokens to generate for the throughput measurement")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not args.skip_ingest and not folder.is_dir():
        raise SystemExit(f"No such folder: {folder}")

    queries = DEFAULT_QUERIES
    if args.queries:
        queries = [line.strip() for line in Path(args.queries).read_text(encoding="utf-8").splitlines()
                   if line.strip() and not line.startswith("#")]

    print("=" * 72)
    print("  docclassify benchmark")
    print("=" * 72)

    from docclassify.storage import sqlite_store
    sqlite_store.init_db()

    # Refuse to produce meaningless numbers against an empty taxonomy.
    taxonomy = sqlite_store.load_taxonomy()
    if not taxonomy:
        raise SystemExit(
            "The taxonomy is empty — every document would come back 'no_match' and the\n"
            "classification numbers would be meaningless.\n\n"
            "Import one first:\n"
            "    python scripts/import_taxonomy.py --file config/taxonomy_manual.json"
        )
    print(f"\n  taxonomy: {len(taxonomy)} categories, "
          f"{len({c['level'] for c in taxonomy})} levels")

    results: dict = {"taxonomy_size": len(taxonomy)}
    started = time.perf_counter()

    results["environment"] = phase_environment()
    results["model_load"] = phase_model_load()
    results["llm_throughput"] = phase_llm_throughput(args.llm_tokens)

    if args.skip_ingest:
        rule("4. Ingestion")
        print("  skipped (--skip-ingest)")
        document_count = len(sqlite_store.documents_in_batch(args.batch)) or 1
    else:
        instrument_pipeline()
        results["ingestion"] = phase_ingest(folder, args.batch)
        document_count = max(1, results["ingestion"]["documents_succeeded"])
        results["stage_breakdown"] = phase_stage_breakdown(document_count)

    results["classification"] = phase_classification_quality(folder, args.labels)

    if not args.skip_search:
        results["search"] = phase_search(queries, args.top_k)
    if not args.skip_report:
        try:
            results["report"] = phase_report(args.batch)
        except Exception as e:  # noqa: BLE001
            print(f"  report generation FAILED: {type(e).__name__}: {e}")
            results["report"] = {"error": f"{type(e).__name__}: {e}"}

    rule("Summary")
    total = time.perf_counter() - started
    print(f"  benchmark wall time   {human(total)}")
    if "ingestion" in results and results["ingestion"].get("docs_per_minute"):
        print(f"  ingestion throughput  {results['ingestion']['docs_per_minute']} docs/minute")
    if "search" in results:
        print(f"  search p50            {results['search']['total']['p50_ms']} ms")
    if results.get("llm_throughput"):
        print(f"  LLM throughput        ~{results['llm_throughput']['approx_tokens_per_second']} tok/s")
    if results.get("classification", {}).get("accuracy"):
        acc = results["classification"]["accuracy"]
        print(f"  classification        {acc['exact_percent']}% exact, "
              f"{acc['top_level_percent']}% top-level")
    peak = gpu_memory_peak_mb()
    if peak:
        print(f"  torch VRAM peak       {peak:.0f} MB")
    driver = nvidia_smi_used_mb()
    if driver:
        print(f"  total VRAM in use     {driver:.0f} MB")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str),
                                    encoding="utf-8")
        print(f"\n  full results written to {args.json}")


if __name__ == "__main__":
    main()
