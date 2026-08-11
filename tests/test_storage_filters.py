"""
Tests for the LanceDB filter-string builders. Deliberately dependency-free —
storage/filters.py imports nothing, so these run on a bare Python install.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docclassify.storage.filters import like_prefix_literal, sql_literal


def test_sql_literal_quotes_plain_value():
    assert sql_literal("Physics") == "'Physics'"


def test_sql_literal_escapes_embedded_quote():
    # The bug this guards: an unescaped apostrophe closes the literal early and
    # the rest of the value is parsed as SQL.
    assert sql_literal("O'Brien") == "'O''Brien'"


def test_sql_literal_neutralizes_injected_predicate():
    injected = "x' OR 1=1 --"
    literal = sql_literal(injected)
    assert literal == "'x'' OR 1=1 --'"
    # Exactly one opening and one closing quote survive, so the whole thing stays
    # a single string operand.
    assert literal.count("'") == 4  # 2 delimiters + the doubled inner quote


def test_sql_literal_accepts_non_strings():
    assert sql_literal(7) == "'7'"


def test_like_prefix_literal_appends_wildcard():
    assert like_prefix_literal("Science/Physics") == "'Science/Physics%'"


def test_like_prefix_literal_escapes_quotes():
    assert like_prefix_literal("Rock 'n' Roll") == "'Rock ''n'' Roll%'"


if __name__ == "__main__":
    test_sql_literal_quotes_plain_value()
    test_sql_literal_escapes_embedded_quote()
    test_sql_literal_neutralizes_injected_predicate()
    test_sql_literal_accepts_non_strings()
    test_like_prefix_literal_appends_wildcard()
    test_like_prefix_literal_escapes_quotes()
    print("All storage-filter tests passed.")
