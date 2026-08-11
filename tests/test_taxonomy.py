"""
Tests for taxonomy path resolution — the pure function that turns the flat
taxonomy rows into the full slash-joined paths the rest of the system stores and
filters on. No database or embedding model needed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docclassify.taxonomy.taxonomy_store import build_category_paths


def _node(category_id, name, parent_id=None, level=0):
    return {"category_id": category_id, "name": name, "parent_id": parent_id,
            "description": "", "level": level}


def test_root_path_is_just_its_name():
    paths = build_category_paths([_node("d1", "Science")])
    assert paths["d1"] == "Science"


def test_nested_path_is_joined_root_first():
    nodes = [
        _node("d1", "Science"),
        _node("f1", "Physics", parent_id="d1", level=1),
        _node("s1", "Quantum", parent_id="f1", level=2),
    ]
    paths = build_category_paths(nodes)
    assert paths["s1"] == "Science/Physics/Quantum"
    assert paths["f1"] == "Science/Physics"


def test_same_leaf_name_under_two_parents_stays_distinct():
    # This is why a bare category name can't be used as the stored category:
    # "Optics" alone is ambiguous, the full path isn't.
    nodes = [
        _node("d1", "Science"),
        _node("d2", "Engineering"),
        _node("a", "Optics", parent_id="d1", level=1),
        _node("b", "Optics", parent_id="d2", level=1),
    ]
    paths = build_category_paths(nodes)
    assert paths["a"] == "Science/Optics"
    assert paths["b"] == "Engineering/Optics"
    assert paths["a"] != paths["b"]


def test_missing_parent_is_treated_as_root():
    nodes = [_node("orphan", "Stray", parent_id="deleted-id", level=1)]
    assert build_category_paths(nodes)["orphan"] == "Stray"


def test_cyclic_parents_do_not_hang():
    # Shouldn't be reachable through the UI, but a hand-edited taxonomy table can
    # produce it and an unguarded parent walk would loop forever.
    nodes = [
        _node("a", "A", parent_id="b"),
        _node("b", "B", parent_id="a"),
    ]
    paths = build_category_paths(nodes)
    assert paths["a"] == "B/A"
    assert paths["b"] == "A/B"


def test_every_node_gets_a_path():
    nodes = [_node(f"n{i}", f"Name{i}", parent_id=f"n{i-1}" if i else None, level=i) for i in range(5)]
    paths = build_category_paths(nodes)
    assert len(paths) == 5
    assert paths["n4"] == "Name0/Name1/Name2/Name3/Name4"


if __name__ == "__main__":
    test_root_path_is_just_its_name()
    test_nested_path_is_joined_root_first()
    test_same_leaf_name_under_two_parents_stays_distinct()
    test_missing_parent_is_treated_as_root()
    test_cyclic_parents_do_not_hang()
    test_every_node_gets_a_path()
    print("All taxonomy tests passed.")
