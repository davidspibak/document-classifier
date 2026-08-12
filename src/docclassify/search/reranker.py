"""
Stage 2 of search: cross-encoder reranking (bge-reranker-v2-m3) of the
ANN candidate pool into a final precise ranking. Unlike the bi-encoder used
for stage-1 retrieval, this model looks at the query and each candidate
JOINTLY, which is far more accurate but too slow to run against the whole
corpus — hence only running it on the ~50 survivors from stage 1.

We drive the cross-encoder directly through transformers rather than
FlagEmbedding's FlagReranker: FlagReranker 1.4.0 pre-tokenizes each side and
merges them with tokenizer.prepare_for_model(), a method transformers 5.x
removed, so it raises "XLMRobertaTokenizer has no attribute prepare_for_model"
on this stack. Encoding the (query, passage) pair straight through the
tokenizer is the canonical path and lets transformers assemble XLM-R's special
tokens itself. `normalize=True` applies a sigmoid to the raw logits, matching
FlagReranker's normalized-score semantics so the 0..1 scores are unchanged.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from docclassify.config import CONFIG

_reranker = None

# The cross-encoder truncates the joined (query, passage) pair to this many
# tokens; bge-reranker-v2-m3 is trained at 512 and search chunks are ~400 tokens.
_MAX_LENGTH = 512
_BATCH_SIZE = 32


class _NativeReranker:
    """
    Thin wrapper exposing the same compute_score(pairs, normalize=True) surface
    the rest of the module (and the tests) expect from FlagReranker, but backed
    by a plain transformers sequence-classification model.
    """

    def __init__(self, model_path: str):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        if self._device == "cuda":
            model = model.half()  # fp16 on GPU, mirroring the embedder's use_fp16
        self._model = model.to(self._device).eval()

    def compute_score(self, pairs: list[list[str]], normalize: bool = True) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(pairs), _BATCH_SIZE):
            batch = pairs[start:start + _BATCH_SIZE]
            queries = [p[0] for p in batch]
            passages = [p[1] for p in batch]
            enc = self._tokenizer(
                queries,
                passages,
                padding=True,
                truncation="only_second",  # never truncate the query, only the passage
                max_length=_MAX_LENGTH,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                logits = self._model(**enc).logits.view(-1).float()
                if normalize:
                    logits = torch.sigmoid(logits)
            scores.extend(logits.tolist())
        return scores


def get_reranker() -> _NativeReranker:
    global _reranker
    if _reranker is None:
        _reranker = _NativeReranker(CONFIG["models"]["reranker"])
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
