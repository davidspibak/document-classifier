"""
Tests for the classification module's pure-logic parts (similarity scoring,
threshold behavior) using synthetic vectors - no real embedding model needed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docclassify.classification.classifier import _cosine_similarities


def test_cosine_similarities_ranks_closest_first():
    doc_vector = [1.0, 0.0, 0.0]
    candidates = [
        {"category_id": "a", "vector": [1.0, 0.0, 0.0]},   # identical -> similarity 1.0
        {"category_id": "b", "vector": [0.0, 1.0, 0.0]},   # orthogonal -> similarity 0.0
        {"category_id": "c", "vector": [0.7, 0.7, 0.0]},   # partial match
    ]
    scored = _cosine_similarities(doc_vector, candidates)
    assert scored[0][0]["category_id"] == "a"
    assert scored[-1][0]["category_id"] == "b"
    assert scored[0][1] > scored[1][1] > scored[2][1]


def test_cosine_similarities_handles_zero_vector_gracefully():
    doc_vector = [0.0, 0.0, 0.0]
    candidates = [{"category_id": "a", "vector": [1.0, 0.0, 0.0]}]
    scored = _cosine_similarities(doc_vector, candidates)
    assert len(scored) == 1  # should not raise a division-by-zero error


if __name__ == "__main__":
    test_cosine_similarities_ranks_closest_first()
    test_cosine_similarities_handles_zero_vector_gracefully()
    print("All classification tests passed.")
