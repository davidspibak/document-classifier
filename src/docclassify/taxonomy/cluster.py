"""
One-time taxonomy construction, step 1: cluster a sample of the existing
corpus's document embeddings into a hierarchy. Agglomerative clustering is
used because it naturally produces a dendrogram (tree), which we cut at
multiple heights to get "Domain -> Field -> Subfield" levels — a single flat
clustering pass wouldn't give hierarchy for free the way this does.
"""
import numpy as np
from sklearn.cluster import AgglomerativeClustering
 
 
def cluster_hierarchy(embeddings: list[list[float]], level_cuts: list[int]) -> dict:
    """
    embeddings: one vector per sampled document.
    level_cuts: number of clusters to cut at for each hierarchy level, coarsest
                first, e.g. [6, 20, 60] -> 6 Domains, 20 Fields, 60 Subfields.
    Returns {level_index: cluster_labels_array} so you can trace which
    documents fall into which cluster at each level, and how finer clusters
    nest inside coarser ones (by checking which coarse cluster each document's
    fine cluster's documents mostly belong to).
    """
    arr = np.array(embeddings, dtype=np.float32)
    results = {}
    for level, n_clusters in enumerate(level_cuts):
        n_clusters = min(n_clusters, len(arr))  # can't ask for more clusters than samples
        model = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
        labels = model.fit_predict(arr)
        results[level] = labels
    return results
 
 
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
    labels = clusterer.fit_predict(normalized)
    return labels
 
 
def cluster_documents(embeddings: list[list[float]], n_clusters: int) -> np.ndarray:
    """
    Single-level clustering on an arbitrary SUBSET of embeddings (e.g. just
    the documents that fell under one Domain, for building the Field level
    beneath it). This is what makes recursive/nested taxonomy construction
    correct — clustering the same subset at each level naturally nests,
    unlike cutting one global dendrogram at multiple heights, which doesn't
    guarantee Field-level clusters stay inside their parent Domain-level cluster.
    """
    n_clusters = min(n_clusters, len(embeddings))
    if n_clusters < 2:
        return np.zeros(len(embeddings), dtype=int)  # too few documents to split further
    arr = np.array(embeddings, dtype=np.float32)
    model = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
    return model.fit_predict(arr)
 
 
def compute_centroid(embeddings: list[list[float]]) -> list[float]:
    """Mean, L2-normalized centroid of an arbitrary list of embeddings (a cluster's member subset)."""
    arr = np.array(embeddings, dtype=np.float32)
    centroid = arr.mean(axis=0)
    norm = np.linalg.norm(centroid)
    return (centroid / norm if norm > 0 else centroid).tolist()
 
 
def cluster_centroid(embeddings: list[list[float]], labels: np.ndarray, cluster_id: int) -> list[float]:
    """Average embedding of all documents assigned to one cluster — becomes the category's centroid vector."""
    arr = np.array(embeddings, dtype=np.float32)
    members = arr[labels == cluster_id]
    centroid = members.mean(axis=0)
    norm = np.linalg.norm(centroid)
    return (centroid / norm if norm > 0 else centroid).tolist()