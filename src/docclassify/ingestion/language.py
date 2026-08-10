"""
Offline language identification via fastText's lid.176 model.
Run per-chunk, not just per-document, so mixed-language documents (e.g. an
English abstract with a Chinese body) are handled correctly rather than
collapsed into one dominant-language guess.
"""
import fasttext

from docclassify.config import CONFIG

_model = None


def _get_model():
    global _model
    if _model is None:
        # fasttext prints a harmless warning about load_model vs the deprecated
        # FastText() constructor; safe to ignore.
        _model = fasttext.load_model(CONFIG["models"]["lang_id"])
    return _model


def detect_language(text: str) -> str:
    """Returns an ISO 639-1-ish language code, e.g. 'en', 'zh', 'ko'. 'unknown' if text is empty/too short."""
    text = text.strip().replace("\n", " ")
    if len(text) < 5:
        return "unknown"
    model = _get_model()
    # fastText expects single-line input and returns labels like '__label__en'
    labels, _ = model.predict(text[:1000], k=1)  # first 1000 chars is plenty for a reliable guess
    return labels[0].replace("__label__", "")


def detect_languages_per_chunk(chunks: list[str]) -> list[str]:
    return [detect_language(c) for c in chunks]
