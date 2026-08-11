"""
One-time taxonomy construction: sample the existing corpus, cluster embeddings
into a NESTED hierarchy, propose labels with the local LLM, write a review sheet
for you to edit, then save the approved taxonomy.

The hierarchy is built recursively: the whole sample is clustered into Domains,
then each Domain's own member documents are clustered into Fields, and so on for
as many --level-cuts as you pass. Clustering the parent's subset is what makes the
levels genuinely nest — see the note at the top of taxonomy/cluster.py.

Usage:
    python scripts/build_taxonomy.py --folder /path/to/corpus --sample-size 3000 --level-cuts 6 20 60
"""
import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docclassify.config import PROJECT_ROOT
from docclassify.ingestion.parsers import parse_document
from docclassify.embeddings.embedder import embed_texts
from docclassify.taxonomy.cluster import cluster_documents, compute_centroid
from docclassify.taxonomy.label import extract_cluster_keywords, propose_category_label
from docclassify.taxonomy.taxonomy_store import create_category
from docclassify.storage import sqlite_store

# A cluster smaller than this isn't worth splitting further — sub-clustering four
# documents produces categories that describe individual papers, not fields.
MIN_DOCUMENTS_TO_SPLIT = 10

# How many of a cluster's documents seed the category's stored vector.
SEED_DOCUMENTS_PER_CATEGORY = 10

EXTENSIONS = (".pdf", ".docx", ".pptx", ".html", ".htm", ".txt", ".md")


def sample_corpus(folder: str, sample_size: int, seed: int | None = None) -> list[tuple[str, str]]:
    """Returns [(file_path, text), ...] for up to sample_size documents."""
    import random

    files = [p for p in Path(folder).rglob("*") if p.suffix.lower() in EXTENSIONS]
    rng = random.Random(seed)  # seedable so a re-run can reproduce the same sample
    rng.shuffle(files)
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


def build_level(texts: list[str], embeddings: list[list[float]], indices: list[int],
                 level: int, level_cuts: list[int], review_sheet: list[dict],
                 parent_id: str | None = None, parent_name: str | None = None) -> None:
    """
    Clusters the documents at `indices` into level_cuts[level] categories, creates
    a taxonomy node per cluster under `parent_id`, then recurses into each cluster
    for the next level down.

    `indices` are positions into the full `texts`/`embeddings` lists, so the
    recursion always clusters a genuine subset of its parent's members.
    """
    if level >= len(level_cuts):
        return

    subset_embeddings = [embeddings[i] for i in indices]
    labels = cluster_documents(subset_embeddings, level_cuts[level])

    indent = "  " * level
    for cluster_id in sorted(set(labels)):
        member_indices = [indices[j] for j, label in enumerate(labels) if label == cluster_id]
        member_texts = [texts[i] for i in member_indices]
        member_embeddings = [embeddings[i] for i in member_indices]

        keywords = extract_cluster_keywords(member_texts)
        proposal = propose_category_label(member_texts, keywords, parent_category_name=parent_name) or {
            "name": f"Cluster {cluster_id} (keywords: {', '.join(keywords[:5])})",
            "description": f"Auto-generated placeholder - keywords: {', '.join(keywords)}",
        }

        category_id = create_category(
            name=proposal["name"], description=proposal["description"],
            level=level, parent_id=parent_id,
            # Already embedded above — pass the vectors so create_category doesn't
            # re-embed the same documents.
            seed_vectors=member_embeddings[:SEED_DOCUMENTS_PER_CATEGORY],
        )

        print(f"{indent}[L{level}] {proposal['name']} ({len(member_texts)} docs)")
        review_sheet.append({
            "level": level,
            "category_id": category_id,
            "parent_id": parent_id,
            "parent_name": parent_name,
            "name": proposal["name"],
            "description": proposal["description"],
            "document_count": len(member_texts),
            "keywords": keywords,
            "centroid_preview": compute_centroid(member_embeddings)[:8],
        })

        if len(member_indices) >= MIN_DOCUMENTS_TO_SPLIT:
            build_level(texts, embeddings, member_indices, level + 1, level_cuts,
                         review_sheet, parent_id=category_id, parent_name=proposal["name"])
        elif level + 1 < len(level_cuts):
            print(f"{indent}  (only {len(member_texts)} docs - not split further)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Folder containing your existing document corpus")
    parser.add_argument("--sample-size", type=int, default=3000)
    parser.add_argument("--level-cuts", type=int, nargs="+", default=[6, 20],
                         help="Clusters per hierarchy level, coarsest first: '6 20' -> 6 Domains, "
                              "each split into up to 20 Fields")
    parser.add_argument("--sample-seed", type=int, default=None,
                         help="Seed the sampling RNG so a re-run picks the same documents")
    parser.add_argument("--review-output", default=str(PROJECT_ROOT / "data" / "taxonomy_review.json"),
                         help="Where to write the review sheet")
    args = parser.parse_args()

    sqlite_store.init_db()

    print(f"Sampling up to {args.sample_size} documents from {args.folder} ...")
    samples = sample_corpus(args.folder, args.sample_size, seed=args.sample_seed)
    if not samples:
        print("No parseable documents found - nothing to cluster.")
        return
    print(f"Parsed {len(samples)} documents. Embedding ...")

    texts = [t for _, t in samples]
    embeddings = embed_texts(texts, batch_size=32)

    print(f"Building {len(args.level_cuts)} taxonomy levels {args.level_cuts} ...")
    review_sheet: list[dict] = []
    build_level(texts, embeddings, list(range(len(texts))), 0, args.level_cuts, review_sheet)

    review_path = Path(args.review_output)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    with open(review_path, "w", encoding="utf-8") as f:
        json.dump(review_sheet, f, ensure_ascii=False, indent=2)

    by_level: dict[int, int] = {}
    for entry in review_sheet:
        by_level[entry["level"]] = by_level.get(entry["level"], 0) + 1
    print(f"\nDone. {len(review_sheet)} categories created: " +
          ", ".join(f"level {lvl}: {count}" for lvl, count in sorted(by_level.items())))
    print(f"Review sheet written to {review_path} - edit names/descriptions there,")
    print("then re-run with your edits applied (or edit directly via taxonomy_store.py / the UI).")


if __name__ == "__main__":
    main()
