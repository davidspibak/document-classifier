"""
Basic tests for the ingestion module. These don't require GPU/model access,
so they can run in CI or as a quick sanity check without the heavier ML deps loaded.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docclassify.ingestion.chunking import chunk_text, join_chunks, needs_pooling
from docclassify.ingestion.dedup import hash_content


def test_chunk_text_respects_overlap():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    # consecutive chunks should share some words due to overlap
    first_chunk_words = chunks[0].split()
    second_chunk_words = chunks[1].split()
    assert any(w in second_chunk_words for w in first_chunk_words[-20:])


def test_chunk_text_empty_input():
    assert chunk_text("") == []


def test_needs_pooling_short_doc():
    assert needs_pooling("short document") is False


def test_needs_pooling_long_doc():
    long_text = " ".join(["word"] * 10000)
    assert needs_pooling(long_text) is True


def test_join_chunks_round_trips_a_chunked_document():
    """
    The property that matters: chunking then rejoining must give back the original
    word sequence. The document summary and the UI preview both rebuild text this
    way, and an off-by-one in the overlap would silently duplicate or drop words at
    every chunk boundary.
    """
    original = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(original, chunk_size=100, overlap=20)
    assert len(chunks) > 1, "test needs a multi-chunk document to be meaningful"
    assert join_chunks(chunks, overlap=20) == original


def test_join_chunks_round_trips_when_last_chunk_is_short():
    # 1000 words at stride 80 does not divide evenly, so the final chunk is partial.
    original = " ".join(f"w{i}" for i in range(1000))
    chunks = chunk_text(original, chunk_size=100, overlap=20)
    assert len(chunks[-1].split()) < 100
    assert join_chunks(chunks, overlap=20) == original


def test_join_chunks_with_no_overlap():
    original = " ".join(f"t{i}" for i in range(300))
    chunks = chunk_text(original, chunk_size=50, overlap=0)
    assert join_chunks(chunks, overlap=0) == original


def test_join_chunks_single_and_empty():
    assert join_chunks([]) == ""
    assert join_chunks(["only one chunk here"], overlap=20) == "only one chunk here"


def test_hash_content_normalizes_whitespace():
    a = hash_content("Hello   World")
    b = hash_content("hello world")
    assert a == b  # case + whitespace differences should not change the hash


def test_hash_content_detects_real_difference():
    a = hash_content("Hello World")
    b = hash_content("Hello There")
    assert a != b


if __name__ == "__main__":
    test_chunk_text_respects_overlap()
    test_chunk_text_empty_input()
    test_needs_pooling_short_doc()
    test_needs_pooling_long_doc()
    test_hash_content_normalizes_whitespace()
    test_hash_content_detects_real_difference()
    print("All ingestion tests passed.")
