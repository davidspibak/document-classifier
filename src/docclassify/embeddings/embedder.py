"""
Wrapper around BAAI/bge-m3 (multilingual, local, GPU-accelerated).
This is the single embedding entry point used by both subsystems — the
classifier embeds whole documents/pooled chunks through here, and search
embeds both documents-at-ingestion-time and queries-at-search-time through
the exact same function, which is what makes cross-lingual retrieval work
(query and documents land in the same vector space).
"""
from FlagEmbedding import BGEM3FlagModel

from docclassify.config import CONFIG

_model = None


def _use_fp16() -> bool:
    """
    fp16 is a straight win on GPU and broken on CPU: PyTorch has no half-precision
    matmul kernel for CPU, so a half-cast model raises
    "addmm_impl_cpu_ not implemented for 'Half'" on the first encode. Decide from the
    hardware rather than hard-coding it, so the same code runs on a CPU-only box.
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_model() -> BGEM3FlagModel:
    global _model
    if _model is None:
        _model = BGEM3FlagModel(CONFIG["models"]["embedding"], use_fp16=_use_fp16())
    return _model


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """
    Encodes a batch of texts into dense vectors. Batching matters a lot for
    throughput on GPU — always prefer calling this once with many texts over
    calling it in a loop with one text at a time.
    """
    if not texts:
        return []
    model = get_model()
    output = model.encode(texts, batch_size=batch_size, max_length=8192)
    return output["dense_vecs"].tolist()


def embed_text(text: str) -> list[float]:
    """Convenience single-text wrapper — used for one-off calls like query embedding."""
    return embed_texts([text])[0]
