"""
Basic tests for the ingestion module. These don't require GPU/model access,
so they can run in CI or as a quick sanity check without the heavier ML deps loaded.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docclassify.ingestion.chunking import chunk_text, needs_pooling
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
