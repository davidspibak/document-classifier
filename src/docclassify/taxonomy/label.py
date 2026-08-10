"""
One-time taxonomy construction, step 2: given a cluster of documents, propose
a human-readable category name + description. Combines a cheap keyword
extraction pass (fast, no LLM) with an LLM labeling pass (more coherent
naming), so the LLM has concrete keywords to ground its proposal in rather
than guessing from raw document snippets alone.
"""
from sklearn.feature_extraction.text import TfidfVectorizer

from docclassify.llm.local_llm import generate_json


def extract_cluster_keywords(cluster_documents: list[str], top_n: int = 12) -> list[str]:
    """
    c-TF-IDF-style keyword extraction: treat the whole cluster as one document
    and compare its word frequencies against a generic corpus-wide baseline
    (approximated here by fitting TF-IDF across the cluster's own documents as
    separate rows, which still surfaces terms distinctive to this cluster).
    """
    if len(cluster_documents) < 2:
        return []
    vectorizer = TfidfVectorizer(max_features=200, stop_words="english")
    tfidf = vectorizer.fit_transform(cluster_documents)
    scores = tfidf.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(terms, scores), key=lambda x: -x[1])
    return [t for t, _ in ranked[:top_n]]


def propose_category_label(cluster_documents: list[str], keywords: list[str],
                            parent_category_name: str | None = None) -> dict | None:
    """
    Asks the local LLM to propose a name + description for this cluster.
    Returns {"name": str, "description": str} or None if the LLM call failed
    (caller should fall back to a keyword-based placeholder name for human review).
    """
    samples = "\n---\n".join(doc[:400] for doc in cluster_documents[:5])
    parent_context = f'This is a sub-category under "{parent_category_name}".\n' if parent_category_name else ""

    prompt = f"""You are helping design a document classification taxonomy.
{parent_context}Below are sample documents and extracted keywords from one cluster.
Propose a short category name (2-5 words) and a one-sentence description of what belongs in it.

Keywords: {', '.join(keywords)}

Sample documents:
{samples}

Respond with ONLY this JSON format:
{{"name": "...", "description": "..."}}
"""
    return generate_json(prompt, max_tokens=200)
