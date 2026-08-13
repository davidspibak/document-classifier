r"""
Proves the project makes no network calls at runtime, by blocking the network and
then exercising every code path that loads a model.

Run this once after installing, and again after any dependency upgrade. It is the
only reliable way to catch a library that has decided to fetch something: two such
leaks have already shipped in this project and both were found by noticing a
progress bar mid-run, not by reading code.

    python scripts/audit_offline.py
    python scripts/audit_offline.py --allow-localhost      # permit a local GROBID
    python scripts/audit_offline.py --skip llm easyocr     # skip the slow probes

How it works: socket.socket.connect and socket.create_connection are replaced with
versions that raise on any outbound address, then each subsystem is loaded in turn.
A probe that completes proves that subsystem read only local files. A probe that
raises NetworkAccessAttempted names the exact leak.

Exit code 0 means everything ran offline. Exit code 1 means at least one path tried
to reach the network, or failed for another reason worth looking at.
"""
import argparse
import socket
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}


class NetworkAccessAttempted(RuntimeError):
    """Raised in place of an outbound connection."""


def install_network_block(allow_localhost: bool) -> None:
    original_connect = socket.socket.connect
    original_create = socket.create_connection

    def _host_of(address) -> str:
        if isinstance(address, tuple) and address:
            return str(address[0])
        return str(address)

    def _permitted(address) -> bool:
        return allow_localhost and _host_of(address) in LOCAL_HOSTS

    def guarded_connect(self, address, *args, **kwargs):
        if _permitted(address):
            return original_connect(self, address, *args, **kwargs)
        raise NetworkAccessAttempted(f"outbound connection to {_host_of(address)} blocked")

    def guarded_create_connection(address, *args, **kwargs):
        if _permitted(address):
            return original_create(address, *args, **kwargs)
        raise NetworkAccessAttempted(f"outbound connection to {_host_of(address)} blocked")

    socket.socket.connect = guarded_connect
    socket.create_connection = guarded_create_connection


# --------------------------------------------------------------------- probes

SAMPLE_TEXT = (
    "Monetary policy transmission operates through interest rates, credit supply and "
    "exchange rates. This paper estimates the response of aggregate demand to a "
    "contractionary shock using quarterly panel data across eighteen economies, and "
    "finds that the pass-through to consumer prices is slower than to asset prices."
)


def probe_environment():
    import os
    from docclassify.config import CONFIG  # noqa: F401  (import triggers enforcement)

    flags = {name: os.environ.get(name) for name in
             ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")}
    unset = [name for name, value in flags.items() if value not in ("1", "true", "True")]
    detail = ", ".join(f"{k}={v}" for k, v in flags.items())
    if unset:
        raise AssertionError(
            f"offline flags not set: {', '.join(unset)}. "
            "Is offline.enforce false in config.yaml? ({detail})"
        )
    return detail


def probe_embedder():
    from docclassify.embeddings.embedder import embed_text
    vector = embed_text(SAMPLE_TEXT)
    assert len(vector) == 1024, f"expected a 1024-dim vector, got {len(vector)}"
    return f"1024-dim vector from local weights"


def probe_reranker():
    from docclassify.search.reranker import rerank
    ranked = rerank("monetary policy", [{"chunk_text": SAMPLE_TEXT},
                                         {"chunk_text": "photosynthesis in C4 plants"}], top_n=2)
    assert ranked and "rerank_score" in ranked[0]
    return f"top score {ranked[0]['rerank_score']:.4f}"


def probe_language_id():
    from docclassify.ingestion.language import detect_language
    code = detect_language(SAMPLE_TEXT)
    assert code == "en", f"expected 'en', got {code!r}"
    return "detected 'en'"


def probe_llm():
    from docclassify.llm.local_llm import generate, resolve_gguf_path
    from docclassify.config import CONFIG
    path = resolve_gguf_path(CONFIG["models"]["llm_gguf"])
    output = generate("Reply with the single word OK.", max_tokens=8)
    assert output.strip(), "the model returned nothing"
    return f"{Path(path).name} responded"


def probe_keywords():
    """
    The regression test for the KeyBERT leak. KeyBERT silently ignores an
    unrecognised embedding object and downloads paraphrase-multilingual-MiniLM-L12-v2
    instead; metadata/keyword_extract.py subclasses BaseEmbedder to prevent that.
    If this probe reports blocked network access, that inheritance has been broken.
    """
    from docclassify.metadata.keyword_extract import extract_keywords, get_keyword_model
    from docclassify.metadata.keyword_extract import _BGEBackend

    model = get_keyword_model()
    assert isinstance(model.model, _BGEBackend), (
        f"KeyBERT is using {type(model.model).__name__}, not the BGE-M3 backend"
    )
    keywords = extract_keywords(SAMPLE_TEXT, top_n=5)
    return f"BGE-M3 backend, keywords: {', '.join(keywords[:5])}"


def probe_easyocr():
    """
    The other historical leak: EasyOCR downloads its weights from GitHub on first
    use unless download_enabled is False and the files are already local.
    """
    from docclassify.ingestion.ocr import EASYOCR_ALLOW_DOWNLOAD, EASYOCR_MODEL_DIR, _get_easyocr_reader

    if EASYOCR_ALLOW_DOWNLOAD:
        raise AssertionError(
            "ocr.easyocr_allow_download is true in config.yaml — EasyOCR is permitted "
            "to download at runtime, which breaks offline operation."
        )
    reader = _get_easyocr_reader(["en"])
    if reader is None:
        return ("SKIPPED - weights absent from "
                f"{EASYOCR_MODEL_DIR}; OCR will keep Tesseract's result. "
                "Fetch with: python scripts/fetch_models.py --only easyocr")
    return f"reader built from {EASYOCR_MODEL_DIR}"


def probe_grobid():
    """Optional, and the only probe permitted to touch the network - on localhost."""
    from docclassify.metadata.grobid_client import is_grobid_available
    return "reachable" if is_grobid_available() else "not running (fine, it is optional)"


PROBES = [
    ("environment", probe_environment),
    ("embedder", probe_embedder),
    ("reranker", probe_reranker),
    ("language-id", probe_language_id),
    ("llm", probe_llm),
    ("keywords", probe_keywords),
    ("easyocr", probe_easyocr),
    ("grobid", probe_grobid),
]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--allow-localhost", action="store_true",
                         help="Permit connections to localhost (needed to probe GROBID)")
    parser.add_argument("--skip", nargs="+", default=[],
                         choices=[name for name, _ in PROBES],
                         help="Probes to skip (llm and keywords are the slow ones on CPU)")
    parser.add_argument("--verbose", action="store_true", help="Print full tracebacks")
    args = parser.parse_args()

    print("=" * 72)
    print("  offline audit - the network is blocked for the duration of this run")
    print("=" * 72)
    if args.allow_localhost:
        print("  localhost is permitted (--allow-localhost)")
    print()

    install_network_block(allow_localhost=args.allow_localhost)

    leaks, failures, skipped = [], [], []
    for name, probe in PROBES:
        if name in args.skip:
            print(f"  {name:<14} SKIP")
            skipped.append(name)
            continue
        try:
            detail = probe()
        except NetworkAccessAttempted as e:
            print(f"  {name:<14} NETWORK ACCESS  <-- {e}")
            leaks.append(name)
            if args.verbose:
                traceback.print_exc()
            continue
        except Exception as e:  # noqa: BLE001
            # A blocked connection can surface wrapped in a library's own exception,
            # so look for our marker anywhere in the chain before calling it a
            # plain failure.
            chain, cursor = [], e
            while cursor is not None:
                chain.append(cursor)
                cursor = cursor.__cause__ or cursor.__context__
            if any(isinstance(link, NetworkAccessAttempted) for link in chain):
                print(f"  {name:<14} NETWORK ACCESS  <-- wrapped in {type(e).__name__}: {str(e)[:70]}")
                leaks.append(name)
            else:
                print(f"  {name:<14} FAIL  {type(e).__name__}: {str(e)[:70]}")
                failures.append(name)
            if args.verbose:
                traceback.print_exc()
            continue

        if isinstance(detail, str) and detail.startswith("SKIPPED"):
            print(f"  {name:<14} SKIP  {detail[8:].strip()}")
            skipped.append(name)
        else:
            print(f"  {name:<14} OK    {detail}")

    print()
    print("-" * 72)
    if leaks:
        print(f"  FAILED: {len(leaks)} path(s) tried to reach the network: {', '.join(leaks)}")
        print("  This project cannot run disconnected until these are fixed.")
    else:
        print("  No path attempted a network call.")
    if failures:
        print(f"  {len(failures)} probe(s) failed for other reasons: {', '.join(failures)}")
        print("  Re-run with --verbose for tracebacks.")
    if skipped:
        print(f"  {len(skipped)} skipped: {', '.join(skipped)}")

    sys.exit(1 if (leaks or failures) else 0)


if __name__ == "__main__":
    main()
