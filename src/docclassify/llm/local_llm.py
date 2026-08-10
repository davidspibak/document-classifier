"""
Wrapper around the local LLM (Qwen2.5-7B-Instruct, GGUF, via llama-cpp-python).

This wrapper is for INTERACTIVE / low-volume use (taxonomy labeling, the
per-document classification tie-breaker during normal monthly ingestion,
translation, report generation). For bulk jobs at multi-million-document
scale (e.g. the one-time initial classification run), use vLLM instead —
see scripts/bulk_init_classify.py — since llama.cpp's request-at-a-time
model does not batch efficiently enough for that scale.
"""
import json
import re

from llama_cpp import Llama

from docclassify.config import CONFIG

_llm = None


def get_llm() -> Llama:
    global _llm
    if _llm is None:
        _llm = Llama(
            model_path=CONFIG["models"]["llm_gguf"],
            n_ctx=8192,
            n_gpu_layers=-1,   # offload all layers to GPU; reduce if you hit VRAM limits
            verbose=False,
        )
    return _llm


def generate(prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
    """Plain text generation, e.g. for report summaries."""
    llm = get_llm()
    result = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["</s>", "<|im_end|>"],
    )
    return result["choices"][0]["text"].strip()


def generate_json(prompt: str, max_tokens: int = 512, retries: int = 2) -> dict | None:
    """
    Generation with JSON parsing + one retry. LLMs occasionally wrap JSON in
    ```json fences or add stray text despite instructions — this strips common
    wrappers before parsing, and retries once with a stricter reminder if
    parsing still fails. Returns None if all attempts fail (caller should treat
    this as "escalate to human review", not crash).
    """
    current_prompt = prompt
    for attempt in range(retries + 1):
        raw = generate(current_prompt, max_tokens=max_tokens, temperature=0.1)
        cleaned = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            current_prompt = prompt + "\n\nIMPORTANT: respond with ONLY the raw JSON object, no other text."
    return None
