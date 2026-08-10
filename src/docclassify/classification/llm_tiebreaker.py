"""
Constrained LLM disambiguation among a SMALL set of candidate categories
(typically 3-5) that the embedding-similarity step couldn't confidently
choose between. Deliberately never lets the LLM invent a category outside
this candidate set — that's what keeps a 7B local model reliable for this
job, versus asking it to classify freely against the whole taxonomy.
"""
from docclassify.llm.local_llm import generate_json
from docclassify.storage import sqlite_store


def llm_tiebreak(document_snippet: str, candidate_category_ids: list[str]) -> str | None:
    """
    Returns the chosen category_id, or None if the LLM says none of the
    candidates fit well (caller should then queue for human review).
    """
    if not candidate_category_ids:
        return None

    taxonomy = sqlite_store.load_taxonomy()
    candidates = [c for c in taxonomy if c["category_id"] in candidate_category_ids]
    if not candidates:
        return None

    options_text = "\n".join(
        f'{i+1}. id="{c["category_id"]}" name="{c["name"]}" - {c["description"]}'
        for i, c in enumerate(candidates)
    )

    prompt = f"""You are classifying a document into exactly one of the following categories.
Choose the single best fit. If NONE of them fit reasonably well, say so explicitly.

Document (title or snippet): {document_snippet[:800]}

Candidate categories:
{options_text}

Respond with ONLY this JSON format:
{{"chosen_category_id": "<id from the list above, or null if none fit>", "reasoning": "<one sentence>"}}
"""
    result = generate_json(prompt, max_tokens=200)
    if result is None:
        return None

    chosen_id = result.get("chosen_category_id")
    if not chosen_id or chosen_id == "null":
        return None

    # Guard against the LLM hallucinating an id not in the candidate list.
    if chosen_id not in candidate_category_ids:
        return None

    return chosen_id
