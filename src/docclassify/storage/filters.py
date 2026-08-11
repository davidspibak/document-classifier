"""
Helpers for building the SQL-string filters LanceDB takes (`table.delete(...)`,
`table.search(...).where(...)`).

Those filters are plain SQL strings, so every value interpolated into one has to
be quoted properly — a category name containing an apostrophe would otherwise
produce a syntax error or, with user-supplied search filters, let a caller inject
arbitrary predicates. Deliberately dependency-free so it can be unit-tested
without pyarrow/lancedb installed.
"""


def sql_literal(value) -> str:
    """
    Renders a value as a single-quoted SQL string literal, doubling any embedded
    single quotes (the standard SQL escape).

        sql_literal("O'Brien")  ->  "'O''Brien'"
    """
    return "'" + str(value).replace("'", "''") + "'"


def like_prefix_literal(value) -> str:
    """
    Literal for a `LIKE` prefix match, e.g. matching "Science/Physics" against
    "Science/Physics/Quantum".

    NOTE: `%` and `_` inside `value` keep their LIKE-wildcard meaning here. That
    is harmless for taxonomy paths (a category named with an underscore just
    matches slightly more broadly than asked) and avoids depending on LanceDB's
    support for a `LIKE ... ESCAPE` clause.
    """
    return sql_literal(f"{value}%")
