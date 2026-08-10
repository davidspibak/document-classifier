"""
Embeds an incoming search query. Deliberately just a thin wrapper around the
same embedder used for documents — using a DIFFERENT embedding call here would
break cross-lingual alignment, so this file exists mainly to make that
intention explicit and to be the one place query-specific preprocessing
(e.g. trimming, language detection for logging) would go.
"""
from docclassify.embeddings.embedder import embed_text
from docclassify.ingestion.language import detect_language


def embed_query(query: str) -> tuple[list[float], str]:
    """Returns (query_vector, detected_language) — language is used for UI display / logging only."""
    query = query.strip()
    return embed_text(query), detect_language(query)
