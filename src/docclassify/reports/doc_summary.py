"""
On-demand, cached one-page summary for a single document. Cheap to call
repeatedly since the result is cached in documents.summary_cache after the
first generation — see the design discussion on batch-vs-on-demand generation.
"""
from docclassify.storage import sqlite_store
from docclassify.llm.local_llm import generate
from docclassify.config import CONFIG

REPORT_LANGUAGE = CONFIG["report"]["output_language"]


def get_or_generate_summary(doc_id: str, document_text: str, force_regenerate: bool = False) -> str:
    doc = sqlite_store.get_document(doc_id)
    if doc and doc.get("summary_cache") and not force_regenerate:
        return doc["summary_cache"]

    title = (doc.get("title_en") or doc.get("filename")) if doc else doc_id
    category = doc.get("category_path", "") if doc else ""

    prompt = f"""Summarize the following document into a one-page report.

Title: {title}
Category: {category}

Document text (truncated):
{document_text[:6000]}

Produce:
1. A 3-4 sentence executive summary
2. 4-6 key points, as bullets
3. 3-5 keywords

Respond in {REPORT_LANGUAGE}.
"""
    summary = generate(prompt, max_tokens=500)
    sqlite_store.upsert_document({"doc_id": doc_id, "summary_cache": summary})
    return summary
