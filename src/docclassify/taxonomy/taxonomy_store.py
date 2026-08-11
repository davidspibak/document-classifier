"""
Reads/writes the finalized, FIXED taxonomy. Once you've reviewed and approved
the categories (via build_taxonomy.py + manual edits), this module is what the
classifier reads from at runtime — the taxonomy is treated as read-mostly
after construction, matching the "firmly structured, then fixed" requirement.

The embedding model and LanceDB are imported lazily inside the writing functions:
the UI imports this module just to render the taxonomy tree, and it shouldn't pull
BGE-M3 into memory to do that.
"""
import uuid

from docclassify.storage import sqlite_store


def create_category(name: str, description: str, level: int,
                     parent_id: str | None = None,
                     seed_documents: list[str] | None = None,
                     seed_vectors: list[list[float]] | None = None) -> str:
    """
    Creates (or updates) one taxonomy node. Looks up an existing node by
    (name, parent_id) first — this makes both build_taxonomy.py and the JSON
    importer safe to re-run: editing a description and re-running updates the
    existing category's embedding in place instead of creating a duplicate
    with a new random id (which would silently orphan the old one).

    If seed documents are provided (e.g. hand-picked or from a cluster), the
    category's stored vector is a blend of the description embedding and the
    seed-document centroid, which — as discussed — tends to classify more
    accurately than a description-only embedding.

    Pass `seed_vectors` instead of `seed_documents` when the caller has already
    embedded those documents (build_taxonomy.py has), to skip re-embedding them.
    """
    import numpy as np

    from docclassify.classification.classifier import invalidate_category_vector_cache
    from docclassify.embeddings.embedder import embed_text, embed_texts
    from docclassify.storage import lancedb_store

    existing = sqlite_store.find_category(name, parent_id)
    category_id = existing["category_id"] if existing else str(uuid.uuid4())
    description_vec = embed_text(description)

    vectors = seed_vectors
    if vectors is None and seed_documents:
        vectors = embed_texts(seed_documents)

    if vectors:
        centroid = np.mean(np.array(vectors, dtype="float32"), axis=0)
        blended = (np.array(description_vec, dtype="float32") + centroid) / 2
        norm = np.linalg.norm(blended)
        final_vec = (blended / norm if norm > 0 else blended).tolist()
    else:
        final_vec = description_vec

    sqlite_store.save_category({
        "category_id": category_id, "parent_id": parent_id, "name": name,
        "description": description, "level": level, "embedding_id": category_id,
    })
    lancedb_store.upsert_category_vector(category_id, final_vec)
    # The classifier caches these vectors for the life of the process; without
    # this a taxonomy edit wouldn't take effect until restart.
    invalidate_category_vector_cache()
    return category_id


def get_full_tree() -> list[dict]:
    """Flat list of all categories; caller reconstructs the tree via parent_id if needed (e.g. for the UI)."""
    return sqlite_store.load_taxonomy()


def get_children(category_id: str | None) -> list[dict]:
    return sqlite_store.children_of(category_id)


def build_category_paths(nodes: list[dict]) -> dict[str, str]:
    """
    Maps every category_id to its full slash-joined path ("Science/Physics/Quantum").

    This is the form the rest of the system stores and filters on — documents.
    category_path, the LanceDB category filter, and the monthly report all use the
    full path, so anything that assigns a category (including a human resolving a
    review item) has to resolve one rather than using a bare node name.

    Pure function over the flat node list so it can be unit-tested without a
    database. A node whose parent_id points at a missing or cyclic ancestor is
    treated as a root rather than looping forever.
    """
    by_id = {n["category_id"]: n for n in nodes}
    paths: dict[str, str] = {}

    for node in nodes:
        parts = [node["name"]]
        seen = {node["category_id"]}
        parent = by_id.get(node.get("parent_id"))
        while parent is not None and parent["category_id"] not in seen:
            parts.append(parent["name"])
            seen.add(parent["category_id"])
            parent = by_id.get(parent.get("parent_id"))
        paths[node["category_id"]] = "/".join(reversed(parts))

    return paths


def category_path_for(category_id: str) -> str | None:
    """Full path of one category, resolved against the current taxonomy."""
    return build_category_paths(sqlite_store.load_taxonomy()).get(category_id)
