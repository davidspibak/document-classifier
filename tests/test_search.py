"""
Tests for the search module's pure-logic parts (reranker ordering) using a
fake reranker so this runs without loading the real cross-encoder model.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import docclassify.search.reranker as reranker_module


class _FakeReranker:
    """Deterministic fake: score = length of chunk_text, just to test sort order."""
    def compute_score(self, pairs, normalize=True):
        return [float(len(text)) for _, text in pairs]


def test_rerank_sorts_by_score_descending(monkeypatch):
    monkeypatch.setattr(reranker_module, "get_reranker", lambda: _FakeReranker())
    candidates = [
        {"chunk_text": "short"},
        {"chunk_text": "a much longer chunk of text here"},
        {"chunk_text": "medium length chunk"},
    ]
    ranked = reranker_module.rerank("query", candidates, top_n=3)
    scores = [r["rerank_score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rerank_respects_top_n(monkeypatch):
    monkeypatch.setattr(reranker_module, "get_reranker", lambda: _FakeReranker())
    candidates = [{"chunk_text": f"chunk {i}" * i} for i in range(1, 10)]
    ranked = reranker_module.rerank("query", candidates, top_n=3)
    assert len(ranked) == 3


def test_rerank_empty_candidates():
    assert reranker_module.rerank("query", [], top_n=5) == []


if __name__ == "__main__":
    # Simple manual monkeypatch since this block doesn't run under pytest
    class _Ctx:
        def setattr(self, module, name, value):
            setattr(module, name, value)
    test_rerank_sorts_by_score_descending(_Ctx())
    test_rerank_respects_top_n(_Ctx())
    test_rerank_empty_candidates()
    print("All search tests passed.")
