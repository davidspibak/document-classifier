"""
Splits document text into overlapping chunks for the search index, and
prepares a single "whole document" text for the classifier.

Chunking here is word-based rather than tokenizer-based to avoid pulling in a
heavy tokenizer dependency just for this step — it's an approximation of token
count (English: ~0.75 words/token; CJK: word-splitting is less meaningful, see
note below). If you need exact token boundaries matching BGE-M3's tokenizer,
swap this for `transformers.AutoTokenizer` — worth doing once you have real
throughput numbers and want to tune chunk size precisely.
"""
from docclassify.config import CONFIG

CHUNK_SIZE = CONFIG["chunking"]["chunk_size_tokens"]
OVERLAP = CONFIG["chunking"]["chunk_overlap_tokens"]

# BGE-M3 supports up to 8192 tokens — documents under this rough word-count
# estimate can be embedded whole for classification, skipping pooling entirely
# (avoids the "averaging dilutes the signal" problem discussed earlier).
WHOLE_DOC_WORD_LIMIT = 6000


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """
    Simple sliding-window word-based chunking with overlap.
    NOTE: for CJK languages without whitespace word boundaries, this degrades to
    roughly character-based chunking since `.split()` will return long "words" —
    functional, but consider a language-aware splitter (e.g. jieba for Chinese)
    if chunk quality matters a lot for your corpus.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start = end - overlap  # step back by `overlap` words so chunks share context
    return chunks


def join_chunks(chunk_texts: list[str], overlap: int = OVERLAP) -> str:
    """
    Inverse of chunk_text(): rebuilds the original word sequence from its chunks.

    Needed because the full document text is never persisted — only chunks are — so
    anything wanting the whole document (the on-demand summary, a UI preview) has to
    reconstruct it.

    Consecutive chunks share `overlap` words by construction, so a naive join would
    repeat those words at every boundary. Since chunking used a fixed stride, dropping
    the leading `overlap` words of every chunk after the first restores the original
    sequence exactly. Whitespace is normalised to single spaces, because that is all
    the chunks preserved.
    """
    if not chunk_texts:
        return ""

    words = chunk_texts[0].split()
    for chunk in chunk_texts[1:]:
        chunk_words = chunk.split()
        words.extend(chunk_words[overlap:] if overlap > 0 else chunk_words)
    return " ".join(words)


def needs_pooling(text: str) -> bool:
    """True if the document is too long to embed whole and must be chunked + pooled instead."""
    return len(text.split()) > WHOLE_DOC_WORD_LIMIT
