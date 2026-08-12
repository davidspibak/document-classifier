r"""
Headless semantic search — the Search view's behaviour without Qt.

Usage:
    python scripts/search_cli.py "how does monetary policy affect inflation"
    python scripts/search_cli.py "gene regulation" --top-k 10 --category Biology
    python scripts/search_cli.py "货币政策" --language zh --show-text
    python scripts/search_cli.py --interactive

Calls exactly the same two-stage path the UI does — embed, wide ANN retrieval,
cross-encoder rerank — so latency measured here is representative of the app.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Documents, category names and generated summaries in this project are multilingual
# by design. Force UTF-8 on the console: Windows defaults to cp1252, which raises
# UnicodeEncodeError the moment a CJK character is printed.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")



def run_search(query: str, top_k: int, category: str | None, language: str | None,
                show_text: bool) -> None:
    from docclassify.config import CONFIG
    from docclassify.search.ann_search import search_candidates
    from docclassify.search.query import embed_query
    from docclassify.search.reranker import rerank
    from docclassify.storage import sqlite_store

    start = time.perf_counter()
    vector, detected = embed_query(query)
    t_embed = time.perf_counter() - start

    start = time.perf_counter()
    candidates = search_candidates(
        vector,
        top_k=CONFIG["search"]["ann_candidate_count"],
        category_filter=category,
        language_filter=language,
    )
    t_ann = time.perf_counter() - start

    start = time.perf_counter()
    results = rerank(query, candidates, top_n=top_k)
    t_rerank = time.perf_counter() - start

    filters = []
    if category:
        filters.append(f"category={category}")
    if language:
        filters.append(f"language={language}")
    filter_note = f"  [{', '.join(filters)}]" if filters else ""

    print(f"\nQuery: {query!r}  (detected language: {detected}){filter_note}")
    print(f"embed {t_embed*1000:.0f} ms | ann {t_ann*1000:.0f} ms ({len(candidates)} candidates) "
          f"| rerank {t_rerank*1000:.0f} ms | total {(t_embed+t_ann+t_rerank)*1000:.0f} ms")

    if not results:
        print("\nNo results. Is anything ingested? Check with:")
        print("  sqlite3 data/sqlite/app.db 'select count(*) from documents;'")
        return

    # Resolve doc_id -> filename so results are readable; the UI shows raw ids.
    filenames = {}
    for hit in results:
        doc_id = hit.get("doc_id")
        if doc_id and doc_id not in filenames:
            row = sqlite_store.get_document(doc_id)
            filenames[doc_id] = (row or {}).get("filename", doc_id)

    print()
    for rank, hit in enumerate(results, start=1):
        doc_id = hit.get("doc_id", "?")
        snippet = " ".join(hit.get("chunk_text", "").split())
        print(f"{rank:2}. [{hit.get('rerank_score', 0):.4f}] {filenames.get(doc_id, doc_id)}"
              f"   chunk {hit.get('chunk_index', '?')}   {hit.get('category_path') or '(uncategorised)'}")
        print(f"    {snippet[:160]}{'...' if len(snippet) > 160 else ''}")
        if show_text:
            print(f"    --- full chunk ---\n    {snippet}\n")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("query", nargs="*", help="Search query (omit with --interactive)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--category", help="Restrict to a category path prefix, e.g. 'Biology'")
    parser.add_argument("--language", help="Restrict to a language code, e.g. 'en' or 'zh'")
    parser.add_argument("--show-text", action="store_true", help="Print each matching chunk in full")
    parser.add_argument("--interactive", action="store_true", help="Prompt for queries in a loop")
    args = parser.parse_args()

    if not args.query and not args.interactive:
        parser.error("give a query, or pass --interactive")

    if args.query:
        run_search(" ".join(args.query), args.top_k, args.category, args.language, args.show_text)

    if args.interactive:
        print("\nInteractive search. Blank line or Ctrl-D to quit.")
        # The first query pays the model-load cost; warn so it isn't mistaken for latency.
        print("(the first query loads the embedding and reranker models)")
        while True:
            try:
                query = input("\nsearch> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not query:
                break
            run_search(query, args.top_k, args.category, args.language, args.show_text)


if __name__ == "__main__":
    main()
