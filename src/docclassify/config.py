"""
Loads config/config.yaml once and exposes it as a simple dict-like object.
Every other module imports `CONFIG` from here rather than re-reading the YAML file,
so there is exactly one source of truth for paths/thresholds across the app.
"""
import sys
from pathlib import Path
import yaml


def _resolve_project_root() -> Path:
    """
    In development the project root is two levels up from this file
    (src/docclassify/config.py -> project root). Once PyInstaller packages the
    app it extracts the source tree into its own private bundle directory, so
    that walk no longer points at the folder holding config/, models/ and
    data/ — anchor to the .exe's own directory instead.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _resolve_project_root()
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Resolve every relative path in the config against PROJECT_ROOT so the app
    # can be launched from any working directory (important once packaged as .exe).
    def resolve(d: dict, keys: list[str]):
        for k in keys:
            if k in d and isinstance(d[k], str) and d[k].startswith("./"):
                d[k] = str(PROJECT_ROOT / d[k][2:])

    resolve(raw.get("models", {}), ["embedding", "reranker", "llm_gguf", "lang_id"])
    resolve(raw.get("storage", {}), ["sqlite_path", "lancedb_path", "inbox_dir", "raw_dir"])
    resolve(raw.get("ocr", {}), ["easyocr_model_dir"])
    return raw


CONFIG = _load_config()
