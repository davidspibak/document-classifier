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
from pathlib import Path

from llama_cpp import Llama

from docclassify.config import CONFIG

_llm = None


def resolve_gguf_path(configured_path: str) -> str:
    """
    Resolves the configured GGUF path to a file that actually exists.

    Large GGUF models are frequently published SPLIT into shards named
    `<name>-00001-of-0000N.gguf` — Qwen2.5-7B-Instruct-GGUF is, for instance. A
    split model is loaded by handing llama.cpp the FIRST shard, which then reads
    `split.count` from the header and pulls in the rest; but that means the
    single-file path in config.yaml matches nothing on disk, and llama.cpp's own
    error for a missing model is unhelpfully terse.

    Accepts either layout:
      1. the configured path, if it exists (single file, or an explicitly-named shard)
      2. the first shard of a split model with that base name
      3. a lone *.gguf sitting in the configured directory
    """
    path = Path(configured_path)
    if path.is_file():
        return str(path)

    directory = path.parent
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Model directory does not exist: {directory}\n"
            f"Configured models.llm_gguf = {configured_path}\n"
            "Download the weights first (scripts/fetch_models.py) or fix the path in config/config.yaml."
        )

    # A split model published under this exact base name.
    shards = sorted(directory.glob(f"{path.stem}-*-of-*.gguf"))
    if shards:
        return str(shards[0])

    # Exactly one .gguf in the folder — unambiguous, so use it.
    present = sorted(p for p in directory.glob("*.gguf"))
    first_shards = [p for p in present if "-00001-of-" in p.name]
    if len(first_shards) == 1:
        return str(first_shards[0])
    if len(present) == 1:
        return str(present[0])

    listing = "\n".join(f"  - {p.name}" for p in present) if present else "  (none)"
    raise FileNotFoundError(
        f"Could not resolve a GGUF model.\n"
        f"Configured models.llm_gguf = {configured_path}\n"
        f"That file does not exist, and no single obvious candidate was found in {directory}.\n"
        f"GGUF files present:\n{listing}\n"
        "Point models.llm_gguf at one of these (for a split model, at its -00001-of-* shard)."
    )


def get_llm() -> Llama:
    global _llm
    if _llm is None:
        _llm = Llama(
            model_path=resolve_gguf_path(CONFIG["models"]["llm_gguf"]),
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
