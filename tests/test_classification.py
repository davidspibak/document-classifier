"""
Tests for the classification module's pure-logic parts (similarity scoring,
threshold behavior, LLM path composition, status mapping) using synthetic vectors
and a fake tie-breaker - no real embedding model or LLM needed.
"""
import sys
import types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import docclassify.classification.classifier as classifier_module
from docclassify.classification.classifier import (
    ClassificationResult, _cosine_similarities, resolve_with_llm,
)
from docclassify.storage.schema import (
    OUTCOME_AMBIGUOUS, OUTCOME_AUTO_ASSIGNED, OUTCOME_LLM_ASSIGNED, OUTCOME_NO_MATCH,
    STATUS_HUMAN_ASSIGNED, STATUS_NEEDS_REVIEW, persisted_status,
)

TAXONOMY = [
    {"category_id": "d1", "parent_id": None, "name": "Science", "description": "", "level": 0},
    {"category_id": "f1", "parent_id": "d1", "name": "Physics", "description": "", "level": 1},
    {"category_id": "f2", "parent_id": "d1", "name": "Biology", "description": "", "level": 1},
]


def _fake_tiebreaker(monkeypatch, chosen_id):
    """
    Stands in for classification.llm_tiebreaker without importing it - the real
    module pulls in llama-cpp at import time.
    """
    fake = types.ModuleType("docclassify.classification.llm_tiebreaker")
    fake.llm_tiebreak = lambda snippet, candidate_ids: chosen_id
    monkeypatch.setitem(sys.modules, "docclassify.classification.llm_tiebreaker", fake)
    monkeypatch.setattr(classifier_module.sqlite_store, "load_taxonomy", lambda: TAXONOMY)


# --- similarity scoring ---

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


# --- LLM tie-break path composition ---

def test_resolve_with_llm_composes_full_path(monkeypatch):
    """
    The regression this guards: storing only the chosen node's bare name
    ("Physics") instead of the full path ("Science/Physics"), which no longer
    matches what search filters and the reports group on.
    """
    _fake_tiebreaker(monkeypatch, "f1")
    ambiguous = ClassificationResult("Science", 0.61, OUTCOME_AMBIGUOUS, ["f1", "f2"])

    resolved = resolve_with_llm("a document about quantum optics", ambiguous)

    assert resolved.status == OUTCOME_LLM_ASSIGNED
    assert resolved.category_path == "Science/Physics"
    assert resolved.review_reason is None


def test_resolve_with_llm_at_top_level_has_no_leading_separator(monkeypatch):
    _fake_tiebreaker(monkeypatch, "d1")
    ambiguous = ClassificationResult(None, 0.5, OUTCOME_AMBIGUOUS, ["d1"])

    resolved = resolve_with_llm("something", ambiguous)

    assert resolved.category_path == "Science"


def test_resolve_with_llm_declining_flags_for_review(monkeypatch):
    _fake_tiebreaker(monkeypatch, None)  # LLM says none of the candidates fit
    ambiguous = ClassificationResult("Science", 0.4, OUTCOME_AMBIGUOUS, ["f1", "f2"])

    resolved = resolve_with_llm("unrelated text", ambiguous)

    assert resolved.status == OUTCOME_NO_MATCH
    assert resolved.review_reason == "llm_no_match"
    assert resolved.category_path == "Science"  # keeps the ancestors we were sure about


def test_resolve_with_llm_rejects_a_category_outside_the_taxonomy(monkeypatch):
    _fake_tiebreaker(monkeypatch, "hallucinated-id")
    ambiguous = ClassificationResult("Science", 0.4, OUTCOME_AMBIGUOUS, ["hallucinated-id"])

    resolved = resolve_with_llm("text", ambiguous)

    assert resolved.status == OUTCOME_NO_MATCH
    assert resolved.review_reason == "llm_invalid_choice"


# --- status vocabulary ---

def test_persisted_status_passes_through_assigned_outcomes():
    assert persisted_status(OUTCOME_AUTO_ASSIGNED) == OUTCOME_AUTO_ASSIGNED
    assert persisted_status(OUTCOME_LLM_ASSIGNED) == OUTCOME_LLM_ASSIGNED


def test_persisted_status_maps_unresolved_outcomes_to_needs_review():
    assert persisted_status(OUTCOME_AMBIGUOUS) == STATUS_NEEDS_REVIEW
    assert persisted_status(OUTCOME_NO_MATCH) == STATUS_NEEDS_REVIEW


def test_human_assignment_is_a_distinct_status():
    # A human resolving a review item must not look like the classifier's own work.
    assert STATUS_HUMAN_ASSIGNED not in (OUTCOME_AUTO_ASSIGNED, OUTCOME_LLM_ASSIGNED)


# --- category vector cache ---

def test_category_vector_cache_is_invalidated(monkeypatch):
    calls = []

    def fake_all_category_vectors():
        calls.append(1)
        return [{"category_id": "d1", "vector": [1.0, 0.0]}]

    monkeypatch.setattr(classifier_module.lancedb_store, "all_category_vectors",
                        fake_all_category_vectors)
    classifier_module.invalidate_category_vector_cache()

    classifier_module._category_vectors()
    classifier_module._category_vectors()
    assert len(calls) == 1  # second lookup served from cache

    classifier_module.invalidate_category_vector_cache()
    classifier_module._category_vectors()
    assert len(calls) == 2  # a taxonomy edit forces a re-read

    classifier_module.invalidate_category_vector_cache()  # don't leak state into other tests
