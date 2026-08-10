"""
Stage 2 of search: cross-encoder reranking (bge-reranker-v2-m3) of the
ANN candidate pool into a final precise ranking. Unlike the bi-encoder used
for stage-1 retrieval, this model looks at the query and each candidate
JOINTLY, which is far more accurate but too slow to run against the whole
corpus — hence only running it on the ~50 survivors from stage 1.
"""
from FlagEmbedding import FlagReranker

from docclassify.config import CONFIG

_reranker = None


def get_reranker() -> FlagReranker:
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker(CONFIG["models"]["reranker"], use_fp16=True)
    return _reranker


def rerank(query: str, candidates: list[dict], top_n: int = CONFIG["search"]["final_result_count"]) -> list[dict]:
    """
    candidates: list of dicts from ann_search.search_candidates (must contain "chunk_text").
    Returns the same dicts, sorted by reranker score descending, truncated to top_n,
    with a "rerank_score" field added to each.
    """
    if not candidates:
        return []
    reranker = get_reranker()
    pairs = [[query, c["chunk_text"]] for c in candidates]
    scores = reranker.compute_score(pairs, normalize=True)
    if isinstance(scores, float):  # compute_score returns a scalar for a single pair
        scores = [scores]

    for c, score in zip(candidates, scores):
        c["rerank_score"] = float(score)

    ranked = sorted(candidates, key=lambda c: -c["rerank_score"])
    return ranked[:top_n]


def search(query: str, category_filter: str | None = None, language_filter: str | None = None) -> list[dict]:
    """End-to-end convenience function: embed -> ANN search -> rerank."""
    from docclassify.search.query import embed_query
    from docclassify.search.ann_search import search_candidates

    query_vector, _ = embed_query(query)
    candidates = search_candidates(query_vector, category_filter=category_filter, language_filter=language_filter)
    return rerank(query, candidates)
