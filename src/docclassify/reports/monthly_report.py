"""
Generates the monthly digest: per-category stats + LLM summaries, then one
overall LLM digest built from those (already-condensed) per-category
summaries — not from the raw documents again, which keeps this final pass cheap.
"""
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone

from docclassify.config import CONFIG
from docclassify.storage import sqlite_store
from docclassify.llm.local_llm import generate

SAMPLES_PER_CATEGORY = CONFIG["report"]["samples_per_category"]
REPORT_LANGUAGE = CONFIG["report"]["output_language"]


def _category_summary(category_path: str, docs: list[dict]) -> str:
    sample_docs = docs[:SAMPLES_PER_CATEGORY]
    formatted = "\n---\n".join(
        f"Title: {d.get('title_en') or d.get('title_zh') or d.get('filename')}" for d in sample_docs
    )
    prompt = f"""Category: {category_path}
{len(docs)} new documents were added this month.
Below are {len(sample_docs)} representative documents:

{formatted}

Write a 3-4 sentence summary of the themes in this batch. Note anything unusual.
Respond in {REPORT_LANGUAGE}.
"""
    return generate(prompt, max_tokens=250)


def generate_monthly_report(upload_batch: str) -> dict:
    """
    Returns {"batch_id", "overall_digest", "category_summaries": {path: text},
    "stats": {...}} and also persists the report row to SQLite.
    """
    docs = sqlite_store.documents_in_batch(upload_batch)

    by_category = defaultdict(list)
    for d in docs:
        by_category[d.get("category_path") or "Uncategorized"].append(d)

    status_counts = Counter(d.get("status", "unknown") for d in docs)
    language_counts = Counter(d.get("language", "unknown") for d in docs)

    category_summaries = {
        category: _category_summary(category, category_docs)
        for category, category_docs in by_category.items()
    }

    digest_input = "\n\n".join(f"{cat}: {summary}" for cat, summary in category_summaries.items())
    overall_prompt = f"""Below are per-category summaries from this month's document upload batch.
Write a short executive summary (5-6 sentences): which categories grew the most,
overall document count and language mix, and anything worth flagging.

Total documents: {len(docs)}
Status breakdown: {dict(status_counts)}
Language breakdown: {dict(language_counts)}

Per-category summaries:
{digest_input}

Respond in {REPORT_LANGUAGE}.
"""
    overall_digest = generate(overall_prompt, max_tokens=400)

    report = {
        "batch_id": upload_batch,
        "overall_digest": overall_digest,
        "category_summaries": category_summaries,
        "stats": {
            "total_documents": len(docs),
            "by_category": {k: len(v) for k, v in by_category.items()},
            "by_status": dict(status_counts),
            "by_language": dict(language_counts),
        },
    }

    with sqlite_store.get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO monthly_reports (batch_id, generated_at, report_text, report_path) "
            "VALUES (?, ?, ?, ?)",
            (upload_batch, datetime.now(timezone.utc).isoformat(), overall_digest, None),
        )

    return report
