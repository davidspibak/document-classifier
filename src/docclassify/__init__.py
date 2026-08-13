"""
Document Auto-Classification & Semantic Search Project - core package.

This module exists to do ONE thing before anything else in the package runs:
switch the Hugging Face libraries into offline mode.

Why it has to happen here, and this early
-----------------------------------------
The project's central promise is that no code path touches the network at
runtime. That promise cannot be kept by documentation alone — a dependency deep
in the stack can decide to fetch a model, and the only symptom is a progress bar
in the middle of an ingestion run. That has already happened twice:

  * KeyBERT silently ignored a duck-typed embedding adapter and downloaded
    paraphrase-multilingual-MiniLM-L12-v2 (~471 MB) for every run. Fixed in
    metadata/keyword_extract.py, but only because someone noticed the progress bar.
  * EasyOCR downloads its detector and recogniser weights from GitHub on first
    use, with download_enabled defaulting to True. Fixed in ingestion/ocr.py.

Setting HF_HUB_OFFLINE turns that class of mistake from a silent download into an
immediate, loud error — so the next one is caught in seconds instead of being
discovered mid-benchmark.

huggingface_hub and transformers read these variables ONCE, at import time, into
module-level constants. Setting them later has no effect. Since importing any
docclassify submodule runs this file first, and the heavy libraries are imported
by those submodules, this is the earliest point that reliably wins the race.

Set offline.enforce: false in config/config.yaml to opt out (needed only if you
are deliberately downloading models through code that imports this package;
scripts/fetch_models.py does not, so it is unaffected).
"""
import os

_OFFLINE_VARIABLES = (
    "HF_HUB_OFFLINE",       # huggingface_hub: no Hub requests
    "TRANSFORMERS_OFFLINE",  # transformers: local files only
    "HF_DATASETS_OFFLINE",   # datasets: local files only
)


def _enforce_offline() -> None:
    try:
        from docclassify.config import CONFIG
    except Exception:  # noqa: BLE001
        # A missing or malformed config must not make the package unimportable —
        # config.py raises a far clearer error when something actually needs it.
        return

    if not (CONFIG.get("offline", {}) or {}).get("enforce", True):
        return

    for variable in _OFFLINE_VARIABLES:
        # Never override an explicit choice already in the environment: a user who
        # exported HF_HUB_OFFLINE=0 on purpose is telling us something.
        os.environ.setdefault(variable, "1")


_enforce_offline()
