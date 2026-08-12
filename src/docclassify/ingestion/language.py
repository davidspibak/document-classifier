"""
Offline language identification via fastText's lid.176 model.
Run per-chunk, not just per-document, so mixed-language documents (e.g. an
English abstract with a Chinese body) are handled correctly rather than
collapsed into one dominant-language guess.
"""
import fasttext

from docclassify.config import CONFIG

_model = None


def _patch_fasttext_numpy2() -> None:
    """
    fasttext-wheel 0.9.2's _FastText.predict ends with
    `np.array(probs, copy=False)`. Under NumPy 2.x `copy=False` means "raise if a
    copy is unavoidable" rather than the old "copy only if needed", so it fails
    with "Unable to avoid copy while creating an array as requested."

    We swap the numpy reference *inside fasttext's own module namespace* for a
    thin proxy that translates only that one `copy=False` call to copy-if-needed
    (`copy=None`) semantics — behaviour-identical on NumPy 1.x. Scoping it to
    fasttext (rather than mutating the global numpy module) keeps every other
    numpy caller, including concurrent worker threads, untouched.
    """
    from fasttext import FastText as _ft

    if getattr(_ft, "_docclassify_numpy2_patched", False):
        return

    import numpy as np

    class _NumpyCompat:
        def __getattr__(self, name):
            return getattr(np, name)

        @staticmethod
        def array(obj, *a, **kw):
            if kw.get("copy") is False:
                kw["copy"] = None  # NumPy 2.x: copy only when necessary
            return np.array(obj, *a, **kw)

    _ft.np = _NumpyCompat()
    _ft._docclassify_numpy2_patched = True


_patch_fasttext_numpy2()


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
