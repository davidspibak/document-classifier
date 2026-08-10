"""
Combines multiple chunk-level vectors into a single document-level vector,
used only when a document is too long to embed whole (see
ingestion.chunking.needs_pooling). For documents that fit BGE-M3's context
window, prefer embedding the whole text directly — pooling is a fallback,
not the default path.
"""
import numpy as np


def mean_pool(vectors: list[list[float]]) -> list[float]:
    """Simple average across chunk vectors — the standard, well-tested default."""
    arr = np.array(vectors, dtype=np.float32)
    pooled = arr.mean(axis=0)
    # Re-normalize to unit length so cosine similarity comparisons downstream
    # behave consistently with individually-embedded (non-pooled) vectors.
    norm = np.linalg.norm(pooled)
    if norm > 0:
        pooled = pooled / norm
    return pooled.tolist()


def weighted_pool(vectors: list[list[float]], weights: list[float]) -> list[float]:
    """
    Weighted average — e.g. weight the first chunk (title/abstract) more
    heavily than a chunk from deep in an appendix.
    """
    arr = np.array(vectors, dtype=np.float32)
    w = np.array(weights, dtype=np.float32).reshape(-1, 1)
    pooled = (arr * w).sum(axis=0) / w.sum()
    norm = np.linalg.norm(pooled)
    if norm > 0:
        pooled = pooled / norm
    return pooled.tolist()
