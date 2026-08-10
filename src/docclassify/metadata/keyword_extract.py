"""
KeyBERT-based keyword extraction, reusing the same BGE-M3 model already loaded
for embeddings — effectively free to add since no separate model needs to be
loaded. Useful as a supplement when GROBID/LLM extraction doesn't find
explicit author-listed keywords, or for non-academic documents that don't
have a "keywords" field at all.
"""
from keybert import KeyBERT

from docclassify.embeddings.embedder import get_model

_kw_model = None


def get_keyword_model() -> KeyBERT:
    global _kw_model
    if _kw_model is None:
        # KeyBERT wraps any object exposing .encode(); BGEM3FlagModel doesn't
        # match that interface exactly, so we adapt it with a tiny shim.
        class _EmbedderAdapter:
            def encode(self, texts, **kwargs):
                bge = get_model()
                return bge.encode(texts)["dense_vecs"]
        _kw_model = KeyBERT(model=_EmbedderAdapter())
    return _kw_model


def extract_keywords(text: str, top_n: int = 8) -> list[str]:
    model = get_keyword_model()
    pairs = model.extract_keywords(text, keyphrase_ngram_range=(1, 2), top_n=top_n)
    return [kw for kw, _score in pairs]
