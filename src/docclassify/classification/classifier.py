"""
Top-down, embedding-similarity classification against the FIXED taxonomy.
This is the fast path that handles the large majority of documents without
ever calling the LLM — see classification/llm_tiebreaker.py for what happens
when this path isn't confident enough.

Nothing in this module writes to the database. It reports an outcome plus, when
a document couldn't be placed, the reason it should go to human review; the
caller (pipeline.py, or scripts/bulk_init_classify.py) owns all persistence.
That keeps the review queue from being written twice with conflicting reasons.
"""
import numpy as np

from docclassify.config import CONFIG
from docclassify.storage import sqlite_store, lancedb_store
from docclassify.storage.schema import (
    OUTCOME_AMBIGUOUS,
    OUTCOME_AUTO_ASSIGNED,
    OUTCOME_LLM_ASSIGNED,
    OUTCOME_NO_MATCH,
)

AUTO_ASSIGN_THRESHOLD = CONFIG["classification"]["auto_assign_threshold"]
AMBIGUOUS_GAP_THRESHOLD = CONFIG["classification"]["ambiguous_gap_threshold"]

# The taxonomy is read-mostly by design, and every document classified walks the
# same category vectors — pulling the whole (small) table out of LanceDB per
# document was the single biggest avoidable cost in a bulk run.
_category_vectors_cache: dict[str, list[float]] | None = None


def invalidate_category_vector_cache() -> None:
    """
    Drops the cached category vectors. taxonomy_store.create_category() calls this
    after any write, so a taxonomy edit takes effect without restarting the app.
    """
    global _category_vectors_cache
    _category_vectors_cache = None


def _category_vectors() -> dict[str, list[float]]:
    global _category_vectors_cache
    if _category_vectors_cache is None:
        _category_vectors_cache = {
            c["category_id"]: c["vector"] for c in lancedb_store.all_category_vectors()
        }
    return _category_vectors_cache


def _cosine_similarities(doc_vector: list[float], candidates: list[dict]) -> list[tuple[dict, float]]:
    doc_arr = np.array(doc_vector, dtype=np.float32)
    doc_arr = doc_arr / (np.linalg.norm(doc_arr) + 1e-8)
    scored = []
    for c in candidates:
        cand_arr = np.array(c["vector"], dtype=np.float32)
        cand_arr = cand_arr / (np.linalg.norm(cand_arr) + 1e-8)
        score = float(np.dot(doc_arr, cand_arr))
        scored.append((c, score))
    return sorted(scored, key=lambda x: -x[1])


class ClassificationResult:
    def __init__(self, category_path: str | None, confidence: float, status: str,
                 candidate_category_ids: list[str] | None = None,
                 review_reason: str | None = None):
        self.category_path = category_path
        self.confidence = confidence
        # One of the OUTCOME_* values in storage/schema.py. Use
        # schema.persisted_status() to turn this into a documents.status value.
        self.status = status
        self.candidate_category_ids = candidate_category_ids or []
        # Set only when the document needs human review; the caller passes it
        # straight to sqlite_store.add_to_review_queue().
        self.review_reason = review_reason


def classify_top_down(doc_vector: list[float]) -> ClassificationResult:
    """
    Walks the taxonomy tree level by level (Domain -> Field -> Subfield -> ...),
    picking the best match at each level and only continuing to that node's
    children. Stops and returns "ambiguous" the moment any level's confidence
    doesn't clear the threshold, rather than forcing a guess deeper into a
    branch it isn't sure about.
    """
    category_vectors = _category_vectors()
    path_names: list[str] = []
    current_parent_id = None
    last_confidence = 0.0

    while True:
        children = sqlite_store.children_of(current_parent_id)
        if not children:
            break  # reached a leaf category

        candidates = [
            {"category_id": c["category_id"], "name": c["name"], "vector": category_vectors[c["category_id"]]}
            for c in children if c["category_id"] in category_vectors
        ]
        if not candidates:
            break

        scored = _cosine_similarities(doc_vector, candidates)
        best_candidate, best_score = scored[0]
        second_score = scored[1][1] if len(scored) > 1 else 0.0
        gap = best_score - second_score

        if best_score < AUTO_ASSIGN_THRESHOLD or gap < AMBIGUOUS_GAP_THRESHOLD:
            # Not confident enough at this level -> escalate, returning the
            # candidates at THIS level so the LLM tie-breaker has focused options.
            candidate_ids = [c["category_id"] for c, _ in scored[:CONFIG["classification"]["llm_candidate_count"]]]
            return ClassificationResult(
                category_path="/".join(path_names) if path_names else None,
                confidence=best_score, status=OUTCOME_AMBIGUOUS,
                candidate_category_ids=candidate_ids,
            )

        path_names.append(best_candidate["name"])
        current_parent_id = best_candidate["category_id"]
        last_confidence = best_score

    if not path_names:
        # Nothing to match against at all — an empty taxonomy, or every top-level
        # category is missing its vector. Worth a human's attention.
        return ClassificationResult(None, 0.0, OUTCOME_NO_MATCH, review_reason="no_taxonomy_match")
    return ClassificationResult("/".join(path_names), last_confidence, OUTCOME_AUTO_ASSIGNED)


def resolve_with_llm(document_snippet: str, result: ClassificationResult) -> ClassificationResult:
    """
    Hands an ambiguous result to the constrained LLM tie-breaker and composes the
    final category path from the ancestors the embedding walk already resolved
    plus the LLM's choice.

    Shared by the interactive pipeline and the bulk script so both compose the
    SAME full path — the bulk script previously stored only the chosen node's
    bare name, which doesn't match what search and the reports filter on.
    """
    from docclassify.classification.llm_tiebreaker import llm_tiebreak

    llm_choice = llm_tiebreak(document_snippet, result.candidate_category_ids)
    if llm_choice is None:
        return ClassificationResult(result.category_path, result.confidence, OUTCOME_NO_MATCH,
                                     result.candidate_category_ids, review_reason="llm_no_match")

    chosen_row = next(
        (c for c in sqlite_store.load_taxonomy() if c["category_id"] == llm_choice), None
    )
    if chosen_row is None:
        return ClassificationResult(result.category_path, result.confidence, OUTCOME_NO_MATCH,
                                     result.candidate_category_ids, review_reason="llm_invalid_choice")

    full_path = f"{result.category_path}/{chosen_row['name']}" if result.category_path else chosen_row["name"]
    return ClassificationResult(full_path, result.confidence, OUTCOME_LLM_ASSIGNED,
                                 result.candidate_category_ids)


def classify_document(doc_vector: list[float], document_snippet: str = "") -> ClassificationResult:
    """
    Full classification flow for one document: try the fast embedding path, then
    fall back to the LLM tie-breaker on ambiguity.

    `document_snippet` is the text the LLM tie-breaker reasons over (title plus
    the head of the document works well). It is passed in rather than read back
    from SQLite because ingestion classifies a document BEFORE its row exists —
    the old lookup therefore always came back empty and the LLM was asked to
    choose a category with no document content at all.
    """
    result = classify_top_down(doc_vector)

    if result.status != OUTCOME_AMBIGUOUS:
        return result  # confidently assigned, or nothing to disambiguate against

    return resolve_with_llm(document_snippet, result)
