"""
One-time taxonomy construction, step 1: cluster a sample of the existing
corpus's document embeddings into a hierarchy.

The hierarchy is built by clustering RECURSIVELY — cluster the whole sample into
Domains, then cluster each Domain's own member documents into Fields, and so on.
Cutting a single global dendrogram at several heights is the obvious alternative
and is wrong for this: it gives no guarantee that a Field-level cluster stays
inside one Domain-level cluster, so the "hierarchy" it produces isn't actually
nested. See scripts/build_taxonomy.py for the recursion.
"""
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def cluster_documents(embeddings: list[list[float]], n_clusters: int) -> np.ndarray:
    """
    Single-level clustering over one set of embeddings — the whole sample for the
    Domain level, or just one Domain's member documents for the Field level
    beneath it. Agglomerative with cosine distance, to match the metric the
    classifier scores with at runtime.
    """
    n_clusters = min(n_clusters, len(embeddings))
    if n_clusters < 2:
        return np.zeros(len(embeddings), dtype=int)  # too few documents to split further
    arr = np.array(embeddings, dtype=np.float32)
    model = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
    return model.fit_predict(arr)


def cluster_documents_hdbscan(embeddings: list[list[float]], min_cluster_size: int = 15):
    """
    Alternative to agglomerative clustering: HDBSCAN doesn't require specifying
    the number of clusters upfront and naturally marks outliers (label -1)
    instead of forcing every document into a cluster. Useful as a sanity check
    against the agglomerative result, or if your corpus has a lot of genuinely
    miscellaneous documents that shouldn't be forced into a category.
    """
    import hdbscan
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    # HDBSCAN's default metric doesn't support cosine directly; normalize vectors
    # to unit length first so Euclidean distance behaves like cosine distance.
    arr = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    normalized = arr / np.clip(norms, 1e-8, None)
    return clusterer.fit_predict(normalized)


def compute_centroid(embeddings: list[list[float]]) -> list[float]:
    """
    Mean, L2-normalized centroid of a cluster's member embeddings — becomes half
    of the category's stored vector (blended with its description embedding in
    taxonomy_store.create_category).
    """
    arr = np.array(embeddings, dtype=np.float32)
    centroid = arr.mean(axis=0)
    norm = np.linalg.norm(centroid)
    return (centroid / norm if norm > 0 else centroid).tolist()
