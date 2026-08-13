"""
KeyBERT keyword extraction driven by the BGE-M3 model already loaded for
embeddings, rather than by a second model of KeyBERT's own choosing.

WHY THIS IS A BaseEmbedder SUBCLASS AND NOT A DUCK-TYPED SHIM
-------------------------------------------------------------
KeyBERT passes whatever you hand it as `model` through
keybert.backend.select_backend(), which inspects the object's TYPE. It recognises
BaseEmbedder subclasses, sentence-transformers, model2vec, flair, spacy, gensim,
USE and transformers pipelines — and if it recognises none of them, the last line
of that function is:

    return SentenceTransformerBackend("paraphrase-multilingual-MiniLM-L12-v2")

which downloads ~471 MB from the Hugging Face Hub on first use and silently
ignores the model you passed.

An object that merely exposes `.encode()` matches none of those checks, so an
earlier version of this file — a small adapter class with an `encode` method —
was discarded by KeyBERT at runtime. Every ingested document quietly pulled a
second embedding model over the network, which both broke the project's offline
guarantee and made the "reuses the already-loaded model, effectively free" claim
false: two models were resident, and keywords were computed by the wrong one.

`isinstance(embedding_model, BaseEmbedder)` is the first check select_backend
makes, and it returns the object untouched. So the inheritance below is load
bearing. Do not replace it with a plain class.
"""
import numpy as np
from keybert import KeyBERT
from keybert.backend import BaseEmbedder

_kw_model = None


class _BGEBackend(BaseEmbedder):
    """Presents BGE-M3 through the interface KeyBERT calls: embed() -> ndarray."""

    def embed(self, documents, verbose: bool = False) -> np.ndarray:
        # Imported lazily so constructing the backend doesn't force the embedding
        # model to load before it is actually needed.
        from docclassify.embeddings.embedder import embed_texts

        texts = [str(d) for d in documents]
        if not texts:
            return np.zeros((0, 1024), dtype=np.float32)
        return np.asarray(embed_texts(texts), dtype=np.float32)


def get_keyword_model() -> KeyBERT:
    global _kw_model
    if _kw_model is None:
        model = KeyBERT(model=_BGEBackend())
        # Fail loudly if a future KeyBERT stops honouring the backend, rather than
        # letting it fall back to downloading paraphrase-multilingual-MiniLM-L12-v2.
        if not isinstance(getattr(model, "model", None), _BGEBackend):
            raise RuntimeError(
                "KeyBERT did not accept the BGE-M3 backend "
                f"(it selected {type(getattr(model, 'model', None)).__name__}). "
                "It would download its own embedding model, breaking offline operation. "
                "Check keybert.backend.select_backend() against _BGEBackend, or set "
                "metadata.extract_keywords: false in config/config.yaml."
            )
        _kw_model = model
    return _kw_model


def extract_keywords(text: str, top_n: int = 8) -> list[str]:
    """
    Top-N keyphrases for one document.

    COST: KeyBERT embeds the document AND every candidate n-gram it generates,
    which for a long document is several hundred short strings pushed through
    BGE-M3. That is cheap on a GPU and slow on CPU. If ingestion throughput
    matters more than keywords, set metadata.extract_keywords: false in
    config/config.yaml — the field is optional everywhere downstream.
    """
    model = get_keyword_model()
    pairs = model.extract_keywords(text, keyphrase_ngram_range=(1, 2), top_n=top_n)
    return [kw for kw, _score in pairs]
