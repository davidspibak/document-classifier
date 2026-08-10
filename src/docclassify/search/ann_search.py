"""
Stage 1 of search: wide approximate-nearest-neighbor candidate retrieval over
chunk_vectors. Intentionally over-fetches (default top_k=50) since this is a
cheap, approximate first pass — the reranker in stage 2 is what produces the
final precise ordering.
"""
from docclassify.config import CONFIG
from docclassify.storage import lancedb_store

DEFAULT_CANDIDATE_COUNT = CONFIG["search"]["ann_candidate_count"]


def search_candidates(query_vector: list[float], top_k: int = DEFAULT_CANDIDATE_COUNT,
                       category_filter: str | None = None, language_filter: str | None = None) -> list[dict]:
    """
    category_filter: exact or prefix match on category_path, e.g. "Science/Physics"
    language_filter: exact language code, e.g. "ko"
    Filters are pushed down to LanceDB rather than applied after retrieval, so
    the candidate pool is drawn from the already-filtered subset (avoids
    wasting the top_k budget on irrelevant results before filtering).
    """
    where_clauses = []
    if category_filter:
        where_clauses.append(f"category_path LIKE '{category_filter}%'")
    if language_filter:
        where_clauses.append(f"language = '{language_filter}'")
    where = " AND ".join(where_clauses) if where_clauses else None

    return lancedb_store.search_chunks(query_vector, top_k=top_k, where=where)
