"""
Top-down, embedding-similarity classification against the FIXED taxonomy.
This is the fast path that handles the large majority of documents without
ever calling the LLM — see classification/llm_tiebreaker.py for what happens
when this path isn't confident enough.
"""
import numpy as np

from docclassify.config import CONFIG
from docclassify.storage import sqlite_store, lancedb_store

AUTO_ASSIGN_THRESHOLD = CONFIG["classification"]["auto_assign_threshold"]
AMBIGUOUS_GAP_THRESHOLD = CONFIG["classification"]["ambiguous_gap_threshold"]


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
                 candidate_category_ids: list[str] | None = None):
        self.category_path = category_path
        self.confidence = confidence
        self.status = status  # "auto_assigned" | "ambiguous" | "no_match"
        self.candidate_category_ids = candidate_category_ids or []


def classify_top_down(doc_vector: list[float]) -> ClassificationResult:
    """
    Walks the taxonomy tree level by level (Domain -> Field -> Subfield -> ...),
    picking the best match at each level and only continuing to that node's
    children. Stops and returns "ambiguous" the moment any level's confidence
    doesn't clear the threshold, rather than forcing a guess deeper into a
    branch it isn't sure about.
    """
    category_vectors = {c["category_id"]: c["vector"] for c in lancedb_store.all_category_vectors()}
    path_names = []
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
                confidence=best_score, status="ambiguous",
                candidate_category_ids=candidate_ids,
            )

        path_names.append(best_candidate["name"])
        current_parent_id = best_candidate["category_id"]
        last_confidence = best_score

    if not path_names:
        return ClassificationResult(None, 0.0, "no_match")
    return ClassificationResult("/".join(path_names), last_confidence, "auto_assigned")


def classify_document(doc_id: str, doc_vector: list[float]) -> ClassificationResult:
    """
    Full classification flow for one document: try the fast embedding path,
    fall back to the LLM tie-breaker on ambiguity, and queue for human review
    if even the LLM can't resolve it. This is the function ingestion calls.
    """
    result = classify_top_down(doc_vector)

    if result.status == "auto_assigned":
        return result

    # Ambiguous or no match -> LLM tie-breaker among the candidates at the
    # level where confidence broke down.
    from docclassify.classification.llm_tiebreaker import llm_tiebreak
    from docclassify.storage import sqlite_store as sql

    doc_row = sql.get_document(doc_id)
    doc_text_snippet = (doc_row.get("title_en") or doc_row.get("filename", "")) if doc_row else ""

    llm_choice = llm_tiebreak(doc_text_snippet, result.candidate_category_ids)
    if llm_choice is None:
        sql.add_to_review_queue(doc_id, reason="llm_no_match",
                                 candidate_categories=result.candidate_category_ids)
        return ClassificationResult(result.category_path, result.confidence, "no_match")

    chosen = sql.load_taxonomy()
    chosen_row = next((c for c in chosen if c["category_id"] == llm_choice), None)
    if chosen_row is None:
        sql.add_to_review_queue(doc_id, reason="llm_invalid_choice",
                                 candidate_categories=result.candidate_category_ids)
        return ClassificationResult(result.category_path, result.confidence, "no_match")

    full_path = f"{result.category_path}/{chosen_row['name']}" if result.category_path else chosen_row["name"]
    return ClassificationResult(full_path, result.confidence, "llm_assigned")
