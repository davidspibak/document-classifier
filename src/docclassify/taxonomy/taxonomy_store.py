"""
Reads/writes the finalized, FIXED taxonomy. Once you've reviewed and approved
the categories (via build_taxonomy.py + manual edits), this module is what the
classifier reads from at runtime — the taxonomy is treated as read-mostly
after construction, matching the "firmly structured, then fixed" requirement.
"""
import uuid
 
from docclassify.storage import sqlite_store, lancedb_store
from docclassify.embeddings.embedder import embed_text
 
 
def create_category(name: str, description: str, level: int,
                     parent_id: str | None = None, seed_documents: list[str] | None = None) -> str:
    """
    Creates (or updates) one taxonomy node. Looks up an existing node by
    (name, parent_id) first — this makes both build_taxonomy.py and the JSON
    importer safe to re-run: editing a description and re-running updates the
    existing category's embedding in place instead of creating a duplicate
    with a new random id (which would silently orphan the old one).
 
    If seed_documents are provided (e.g. hand-picked or from a cluster), the
    category's stored vector is a blend of the description embedding and the
    seed-document centroid, which — as discussed — tends to classify more
    accurately than a description-only embedding.
    """
    existing = sqlite_store.find_category(name, parent_id)
    category_id = existing["category_id"] if existing else str(uuid.uuid4())
    description_vec = embed_text(description)
 
    if seed_documents:
        from docclassify.taxonomy.cluster import cluster_centroid
        import numpy as np
        seed_vecs = [embed_text(d) for d in seed_documents]
        centroid = np.mean(np.array(seed_vecs, dtype="float32"), axis=0)
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
    return category_id
 
 
def get_full_tree() -> list[dict]:
    """Flat list of all categories; caller reconstructs the tree via parent_id if needed (e.g. for the UI)."""
    return sqlite_store.load_taxonomy()
 
 
def get_children(category_id: str | None) -> list[dict]:
    return sqlite_store.children_of(category_id)