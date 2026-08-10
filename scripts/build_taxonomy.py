"""
One-time taxonomy construction: sample the existing corpus, cluster
embeddings hierarchically, propose labels with the local LLM, print a review
sheet for you to edit, then save the approved taxonomy.

Usage: python scripts/build_taxonomy.py --folder /path/to/corpus --sample-size 3000
"""
import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docclassify.ingestion.parsers import parse_document
from docclassify.embeddings.embedder import embed_texts
from docclassify.taxonomy.cluster import cluster_hierarchy, cluster_centroid
from docclassify.taxonomy.label import extract_cluster_keywords, propose_category_label
from docclassify.taxonomy.taxonomy_store import create_category
from docclassify.storage import sqlite_store


def sample_corpus(folder: str, sample_size: int) -> list[tuple[str, str]]:
    """Returns [(file_path, text), ...] for up to sample_size documents."""
    extensions = (".pdf", ".docx", ".pptx", ".html", ".htm", ".txt", ".md")
    files = [p for p in Path(folder).rglob("*") if p.suffix.lower() in extensions]
    import random
    random.shuffle(files)
    files = files[:sample_size]

    samples = []
    for f in files:
        try:
            parsed = parse_document(str(f))
            if parsed["text"].strip():
                samples.append((str(f), parsed["text"]))
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {f}: {e}")
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Folder containing your existing document corpus")
    parser.add_argument("--sample-size", type=int, default=3000)
    parser.add_argument("--level-cuts", type=int, nargs="+", default=[6, 20],
                         help="Number of clusters per hierarchy level, coarsest first, e.g. 6 20")
    args = parser.parse_args()

    sqlite_store.init_db()

    print(f"Sampling up to {args.sample_size} documents from {args.folder} ...")
    samples = sample_corpus(args.folder, args.sample_size)
    print(f"Parsed {len(samples)} documents. Embedding ...")

    texts = [t for _, t in samples]
    embeddings = embed_texts(texts, batch_size=32)

    print(f"Clustering into levels {args.level_cuts} ...")
    level_labels = cluster_hierarchy(embeddings, args.level_cuts)

    # --- Level 0 (Domain) ---
    top_labels = level_labels[0]
    review_sheet = []
    domain_ids = {}

    for cluster_id in sorted(set(top_labels)):
        member_texts = [texts[i] for i in range(len(texts)) if top_labels[i] == cluster_id]
        keywords = extract_cluster_keywords(member_texts)
        proposal = propose_category_label(member_texts, keywords) or {
            "name": f"Cluster {cluster_id} (keywords: {', '.join(keywords[:5])})",
            "description": f"Auto-generated placeholder - keywords: {', '.join(keywords)}",
        }
        centroid = cluster_centroid(embeddings, top_labels, cluster_id)
        seed_docs = member_texts[:10]

        category_id = create_category(
            name=proposal["name"], description=proposal["description"],
            level=0, parent_id=None, seed_documents=seed_docs,
        )
        domain_ids[cluster_id] = category_id
        review_sheet.append({
            "level": 0, "cluster_id": int(cluster_id), "category_id": category_id,
            "name": proposal["name"], "description": proposal["description"],
            "document_count": len(member_texts), "keywords": keywords,
        })

    # NOTE: extending this to level 1+ (Field, Subfield) follows the same
    # pattern - cluster within each domain's member documents, propose labels,
    # create_category(..., parent_id=domain_ids[cluster_id]). Left as a
    # straightforward extension once you've reviewed the Domain level below.

    review_path = Path(args.folder).parent / "taxonomy_review.json"
    with open(review_path, "w", encoding="utf-8") as f:
        json.dump(review_sheet, f, ensure_ascii=False, indent=2)

    print(f"\nDone. {len(review_sheet)} top-level categories created.")
    print(f"Review sheet written to {review_path} - edit names/descriptions there,")
    print("then re-run with your edits applied (or edit directly via taxonomy_store.py / the UI).")


if __name__ == "__main__":
    main()
