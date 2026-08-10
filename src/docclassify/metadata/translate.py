"""
Translates extracted metadata fields to English using the local LLM, while
preserving the original-language fields. Author names are NOT translated —
see the note below.
"""
from docclassify.llm.local_llm import generate_json

try:
    from pypinyin import pinyin, Style
except ImportError:
    pinyin = None  # only needed if source language is Chinese


def romanize_chinese_name(name: str) -> str:
    """
    Author names should never be "translated" (there's no English equivalent
    of a name) — romanize to pinyin instead, purely for sorting/searching by
    Latin-script users. The original Chinese characters remain the
    authoritative field regardless.
    """
    if pinyin is None:
        return name
    syllables = pinyin(name, style=Style.NORMAL)
    return " ".join(s[0] for s in syllables).title()


def translate_title_and_keywords(title: str, keywords: list[str], source_language_hint: str = "Chinese") -> dict:
    """Returns {"title_en": str, "keywords_en": list[str]}. Falls back to originals if the LLM call fails."""
    prompt = f"""Translate the following academic title and keywords from {source_language_hint} to English.
Respond with ONLY this JSON format.

Title: {title}
Keywords: {', '.join(keywords)}

{{"title_en": "...", "keywords_en": ["...", "..."]}}
"""
    result = generate_json(prompt, max_tokens=300)
    if result is None:
        return {"title_en": title, "keywords_en": keywords}  # fail-safe: keep originals rather than lose data
    return {
        "title_en": result.get("title_en", title),
        "keywords_en": result.get("keywords_en", keywords),
    }
