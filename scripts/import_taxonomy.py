r"""
Builds the fixed taxonomy from a hand-written JSON file instead of by clustering.

scripts/build_taxonomy.py derives categories from the corpus itself, which needs a
sample of roughly a thousand documents to produce sensible clusters. When you
already know the category structure you want — or you only have a small evaluation
set — define it by hand and import it with this script instead.

Usage:
    python scripts/import_taxonomy.py --file config/taxonomy_manual.json
    python scripts/import_taxonomy.py --file my_tree.json --replace
    python scripts/import_taxonomy.py --file my_tree.json --dry-run

JSON format — a list of nodes, or {"categories": [...]}, nested arbitrarily deep:

    [
      {
        "name": "Economics",
        "description": "A paragraph describing what belongs here.",
        "children": [
          {"name": "Macroeconomics", "description": "..."},
          {"name": "Microeconomics",  "description": "..."}
        ]
      }
    ]

Each node may also carry:
    "seed_documents": ["path/to/a.pdf", "path/to/b.pdf"]   parsed, then embedded
    "seed_texts":     ["raw text of a representative doc"]  embedded directly

WHY SEEDS MATTER: with no seeds, a category's vector is just its description
embedded. A short description sits in a very different part of the vector space
from the long, concrete documents it is meant to attract, which depresses every
similarity score and pushes documents into the LLM tie-breaker. Two ways to fix
that, in order of effectiveness:
  1. give each category two or three seed documents, or
  2. write the description like an ABSTRACT of a typical document in the category
     — several sentences using the vocabulary the real documents use — rather than
     a dictionary definition.

Re-running is safe: categories are matched on (name, parent_id), so editing a
description and re-importing updates that node in place rather than creating a
duplicate with a new id.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Documents, category names and generated summaries in this project are multilingual
# by design. Force UTF-8 on the console: Windows defaults to cp1252, which raises
# UnicodeEncodeError the moment a CJK character is printed.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


# docclassify is imported lazily inside the functions that need it, so --dry-run can
# validate a taxonomy file on a machine with none of the dependencies installed —
# useful for checking your JSON before shipping it to a GPU box.


def _load_nodes(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("categories", data.get("taxonomy"))
    if not isinstance(data, list):
        raise ValueError(
            "Expected a JSON list of category nodes, or an object with a "
            '"categories" key holding one.'
        )
    return data


def _validate(nodes: list[dict], path: str = "") -> list[str]:
    """Collects every structural problem up front rather than failing on the first."""
    problems = []
    seen_names = set()
    for index, node in enumerate(nodes):
        where = f"{path}[{index}]"
        if not isinstance(node, dict):
            problems.append(f"{where}: expected an object, got {type(node).__name__}")
            continue

        name = node.get("name")
        if not name or not str(name).strip():
            problems.append(f"{where}: missing a non-empty 'name'")
        else:
            key = str(name).strip().lower()
            if key in seen_names:
                problems.append(
                    f"{where}: duplicate name {name!r} among siblings. Sibling names must "
                    "differ, since categories are identified by (name, parent)."
                )
            seen_names.add(key)

        description = node.get("description")
        if not description or not str(description).strip():
            problems.append(f"{where} ({name}): missing a non-empty 'description'")
        elif len(str(description).split()) < 8:
            word_count = len(str(description).split())
            problems.append(
                f"{where} ({name}): description is only {word_count} "
                f"{'word' if word_count == 1 else 'words'}. Very short descriptions classify "
                "poorly - see the note at the top of this script."
            )

        for seed_path in node.get("seed_documents", []) or []:
            if not Path(seed_path).is_file():
                problems.append(f"{where} ({name}): seed document not found: {seed_path}")

        children = node.get("children") or []
        if children:
            problems.extend(_validate(children, path=f"{where}.children"))
    return problems


def _seed_texts_for(node: dict) -> list[str]:
    texts = list(node.get("seed_texts", []) or [])
    seed_documents = node.get("seed_documents", []) or []
    if seed_documents:
        from docclassify.ingestion.parsers import parse_document
        for seed_path in seed_documents:
            try:
                parsed = parse_document(str(seed_path))
                if parsed["text"].strip():
                    texts.append(parsed["text"])
            except Exception as e:  # noqa: BLE001
                print(f"    [warn] could not parse seed {seed_path}: {e}")
    return texts


def _import_level(nodes: list[dict], level: int, parent_id: str | None,
                   dry_run: bool, counts: dict) -> None:
    if not dry_run:
        from docclassify.taxonomy.taxonomy_store import create_category

    indent = "  " * level
    for node in nodes:
        name = str(node["name"]).strip()
        description = str(node["description"]).strip()
        seed_texts = [] if dry_run else _seed_texts_for(node)

        seed_note = f"  [{len(seed_texts)} seed docs]" if seed_texts else "  [description only]"
        print(f"{indent}L{level}  {name}{seed_note}")

        if dry_run:
            category_id = f"dry-run-{name}"
        else:
            category_id = create_category(
                name=name,
                description=description,
                level=level,
                parent_id=parent_id,
                seed_documents=seed_texts or None,
            )
        counts[level] = counts.get(level, 0) + 1

        children = node.get("children") or []
        if children:
            _import_level(children, level + 1, category_id, dry_run, counts)


def _clear_taxonomy() -> None:
    """
    Deletes every category and its vector. Documents are left untouched: their
    category_path becomes stale text until they are re-classified, which is what
    you want when re-importing a corrected tree mid-experiment.
    """
    from docclassify.storage import lancedb_store, sqlite_store

    existing = sqlite_store.load_taxonomy()
    print(f"Removing {len(existing)} existing categories ...")
    for category in existing:
        try:
            lancedb_store.category_vectors_table().delete(
                f"category_id = '{category['category_id']}'"
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] could not delete vector for {category['name']}: {e}")
    with sqlite_store.get_connection() as conn:
        conn.execute("DELETE FROM taxonomy")

    from docclassify.classification.classifier import invalidate_category_vector_cache
    invalidate_category_vector_cache()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--file", required=True, help="JSON file describing the taxonomy")
    parser.add_argument("--replace", action="store_true",
                         help="Delete the existing taxonomy first (documents are left alone)")
    parser.add_argument("--dry-run", action="store_true",
                         help="Validate and print the tree without writing anything or loading a model")
    args = parser.parse_args()

    source = Path(args.file)
    if not source.is_file():
        raise SystemExit(f"No such file: {source}")

    nodes = _load_nodes(source)
    problems = _validate(nodes)
    if problems:
        print("Taxonomy file has problems:\n")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)

    if args.dry_run:
        print(f"{source} is valid. Tree that would be created:\n")
        counts: dict[int, int] = {}
        _import_level(nodes, 0, None, dry_run=True, counts=counts)
        total = sum(counts.values())
        print(f"\n{total} categories across {len(counts)} levels "
              f"({', '.join(f'L{k}: {v}' for k, v in sorted(counts.items()))}).")
        print("Nothing was written. Drop --dry-run to import.")
        return

    from docclassify.storage import sqlite_store

    sqlite_store.init_db()
    if args.replace:
        _clear_taxonomy()

    print(f"Importing taxonomy from {source} (this loads the embedding model) ...\n")
    counts = {}
    _import_level(nodes, 0, None, dry_run=False, counts=counts)

    total = sum(counts.values())
    print(f"\nDone. {total} categories across {len(counts)} levels "
          f"({', '.join(f'L{k}: {v}' for k, v in sorted(counts.items()))}).")
    print("\nVerify with:")
    print("  python -c \"import sys; sys.path.insert(0,'src'); "
          "from docclassify.taxonomy.taxonomy_store import build_category_paths, get_full_tree; "
          "[print(' ', p) for p in sorted(build_category_paths(get_full_tree()).values())]\"")


if __name__ == "__main__":
    main()
