"""
Metadata extraction for one document — the module that wires the rest of this
package into the ingestion pipeline.

Order of preference, each step falling back to the next:
  1. GROBID, for PDFs, when the local service is reachable. Purpose-built for
     scholarly headers, so it beats an LLM on standard academic formatting and
     costs no GPU time.
  2. The local LLM, reading the head of the document.
  3. KeyBERT, only to fill in keywords the document never listed.
Then, for non-English documents, the *_en mirror fields are filled by translation.

Every step is gated by the `metadata:` block in config.yaml: the LLM-backed steps
cost one generation call per document, which is fine for a monthly batch and far
too expensive for a multi-million-document initial load.

The heavy imports (KeyBERT, the LLM) are deliberately made lazily inside the
functions that need them, so importing this module doesn't drag the embedding
model or llama.cpp into a process that only wanted to read a title.
"""
import json
from pathlib import Path

from docclassify.config import CONFIG

_META = CONFIG.get("metadata", {}) or {}
ENABLED = _META.get("enabled", True)
USE_GROBID = _META.get("use_grobid", True)
USE_LLM_FALLBACK = _META.get("use_llm_fallback", True)
EXTRACT_KEYWORDS = _META.get("extract_keywords", True)
TRANSLATE_TO_ENGLISH = _META.get("translate_to_english", True)
LLM_SNIPPET_CHARS = _META.get("llm_snippet_chars", 3000)

# fastText's codes -> names the LLM understands when asked to translate.
_LANGUAGE_NAMES = {
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ru": "Russian",
    "de": "German", "fr": "French", "es": "Spanish", "pt": "Portuguese",
    "it": "Italian", "ar": "Arabic", "hi": "Hindi", "nl": "Dutch",
}

# GROBID availability is a network round-trip; probe it once per process rather
# than once per document.
_grobid_available: bool | None = None


def _grobid_is_up() -> bool:
    global _grobid_available
    if _grobid_available is None:
        from docclassify.metadata.grobid_client import is_grobid_available
        _grobid_available = is_grobid_available()
    return _grobid_available


def _llm_header_metadata(text: str) -> dict | None:
    """LLM fallback for title/authors/date/keywords. Returns None if the call fails."""
    from docclassify.llm.local_llm import generate_json

    snippet = text[:LLM_SNIPPET_CHARS]
    prompt = f"""Extract bibliographic metadata from the beginning of this document.
Use null (or an empty list) for any field that is genuinely absent — never invent
an author, a date, or a title that isn't there.

Document:
{snippet}

Respond with ONLY this JSON format:
{{"title": "<title or null>", "authors": ["<name>", "..."], "published_date": "<YYYY-MM-DD or null>", "keywords": ["<keyword>", "..."]}}
"""
    result = generate_json(prompt, max_tokens=400)
    if not isinstance(result, dict):
        return None
    return _normalize(result)


def _normalize(raw: dict) -> dict:
    """Coerces whatever GROBID or the LLM returned into consistent types."""
    def clean_str(value):
        if not isinstance(value, str):
            return None
        value = value.strip()
        return value if value and value.lower() != "null" else None

    def clean_list(value):
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [s.strip() for s in value if isinstance(s, str) and s.strip()]

    return {
        "title": clean_str(raw.get("title")),
        "authors": clean_list(raw.get("authors")),
        "published_date": clean_str(raw.get("published_date")),
        "keywords": clean_list(raw.get("keywords")),
        "abstract": clean_str(raw.get("abstract")),
    }


def _merge(primary: dict | None, secondary: dict | None) -> dict:
    """Field-by-field fill-in: keep whatever the higher-priority source found."""
    primary = primary or {}
    secondary = secondary or {}
    merged = {}
    for key in ("title", "published_date", "abstract"):
        merged[key] = primary.get(key) or secondary.get(key)
    for key in ("authors", "keywords"):
        merged[key] = primary.get(key) or secondary.get(key) or []
    return merged


def _keybert_keywords(text: str) -> list[str]:
    try:
        from docclassify.metadata.keyword_extract import extract_keywords
        return extract_keywords(text)
    except Exception as e:  # noqa: BLE001 - keywords are a nice-to-have, never fatal
        print(f"[metadata] KeyBERT keyword extraction failed: {e}")
        return []


def extract_metadata(file_path: str, text: str, language: str) -> dict:
    """
    Returns a dict of `documents` columns to merge into the row being written —
    only the keys it actually resolved, so nothing overwrites an existing value
    with NULL. Never raises: a document with no extractable metadata is still a
    perfectly good document to classify and index.
    """
    if not ENABLED:
        return {}

    try:
        return _extract(file_path, text, language)
    except Exception as e:  # noqa: BLE001 - metadata must never break ingestion
        print(f"[metadata] Extraction failed for {file_path}: {e}")
        return {}


def _extract(file_path: str, text: str, language: str) -> dict:
    grobid_result = None
    if USE_GROBID and Path(file_path).suffix.lower() == ".pdf" and _grobid_is_up():
        from docclassify.metadata.grobid_client import extract_header_metadata
        raw = extract_header_metadata(file_path)
        if raw:
            grobid_result = _normalize(raw)

    resolved = grobid_result or {}
    # Fall back to the LLM when GROBID is absent, or when it came back without the
    # one field that matters most downstream (the title feeds the classification
    # tie-breaker and the monthly report).
    if USE_LLM_FALLBACK and not resolved.get("title"):
        resolved = _merge(grobid_result, _llm_header_metadata(text))

    title = resolved.get("title")
    authors = resolved.get("authors") or []
    keywords = resolved.get("keywords") or []

    if not keywords and EXTRACT_KEYWORDS:
        keywords = _keybert_keywords(text)

    record: dict = {}
    if title:
        record["title_zh"] = title
    if authors:
        record["authors_zh"] = authors  # sqlite_store JSON-encodes list values
        if language == "zh":
            from docclassify.metadata.translate import romanize_chinese_name
            record["authors_pinyin"] = "; ".join(romanize_chinese_name(a) for a in authors)
    if keywords:
        record["keywords_zh"] = keywords
    if resolved.get("published_date"):
        record["published_date"] = resolved["published_date"]

    if not title and not keywords:
        return record

    if language == "en":
        # Already English — mirror rather than pay for a translation round-trip.
        if title:
            record["title_en"] = title
        if keywords:
            record["keywords_en"] = keywords
    elif TRANSLATE_TO_ENGLISH:
        from docclassify.metadata.translate import translate_title_and_keywords
        translated = translate_title_and_keywords(
            title or "", keywords,
            source_language_hint=_LANGUAGE_NAMES.get(language, language or "the source language"),
        )
        if translated.get("title_en"):
            record["title_en"] = translated["title_en"]
        if translated.get("keywords_en"):
            record["keywords_en"] = translated["keywords_en"]

    return record


def classification_snippet(record: dict, text: str, max_chars: int = 800) -> str:
    """
    Builds the text handed to the LLM classification tie-breaker: the title (in
    English where we have it) followed by the head of the document. A title alone
    is often too terse to disambiguate between sibling categories, and raw body
    text alone buries the subject.
    """
    title = record.get("title_en") or record.get("title_zh")
    body = text[:max_chars].strip()
    return f"{title}\n\n{body}" if title else body


def decode_list_field(value) -> list[str]:
    """
    Reads back a column that upsert_document() JSON-encoded (authors_zh,
    keywords_zh, keywords_en). Tolerates plain strings from hand-edited rows.
    """
    if not value:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return [str(value)]
    if isinstance(decoded, list):
        return [str(v) for v in decoded]
    return [str(decoded)]
