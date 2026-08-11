r"""
Downloads the four model artifacts into models/. ONLINE step, run once.

Called by scripts/fetch_offline_bundle.ps1, but usable on its own:

    python scripts/fetch_models.py                 # everything
    python scripts/fetch_models.py --only lid      # just the language-ID model
    python scripts/fetch_models.py --dest D:\weights

Uses huggingface_hub's Python API rather than the CLI on purpose: the CLI was
renamed from `huggingface-cli` to `hf` and its flags moved, whereas
snapshot_download() has been stable for years.

Re-running is cheap — hugging face downloads resume and skip files already
present, and the direct download checks the existing size first.
"""
import argparse
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Skipped for every HF repo. These are real weights or assets, just not ones this
# project can use:
#   * bge-m3 ships a full ONNX export (~2.2 GB) alongside the PyTorch weights.
#     FlagEmbedding loads the model through transformers and never touches ONNX.
#   * both repos ship README illustrations.
# Deliberately NOT excluded: *.bin (bge-m3 publishes NO safetensors, so
# pytorch_model.bin is the only copy of the weights) and *.pt (colbert_linear.pt /
# sparse_linear.pt are part of BGE-M3 proper).
HF_IGNORE = [
    "onnx/*", "onnx/**", "*.onnx", "*.onnx_data",
    "imgs/*", "imgs/**", "assets/*", "assets/**",
    "*.jpg", "*.jpeg", "*.png", "*.gif",
]

# Kept as a literal table rather than read from config.yaml: this script must run
# before the project's dependencies (pyyaml included) are installed.
HF_MODELS = {
    "embedding": {
        "repo": "BAAI/bge-m3",
        "subdir": "bge-m3",
        "allow": None,
        "approx": "~2.3 GB",
    },
    "reranker": {
        "repo": "BAAI/bge-reranker-v2-m3",
        "subdir": "bge-reranker-v2-m3",
        "allow": None,
        "approx": "~2.2 GB",
    },
    "llm": {
        "repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "subdir": ".",  # the .gguf lands directly in models/
        # Tolerant pattern: the repo's exact casing has changed before. Whatever it
        # matches must agree with models.llm_gguf in config/config.yaml.
        "allow": ["*q4_k_m*.gguf"],
        "approx": "~4.5 GB",
    },
}

LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
LID_FILENAME = "lid.176.bin"
LID_APPROX_BYTES = 131_266_198  # ~131 MB; used only to detect a truncated download


def _human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def download_hf_model(key: str, dest_root: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub is not installed. This is the one ONLINE step, so just:\n"
            "    pip install huggingface_hub"
        )

    spec = HF_MODELS[key]
    target = dest_root if spec["subdir"] == "." else dest_root / spec["subdir"]
    target.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "repo_id": spec["repo"],
        "local_dir": str(target),
        "allow_patterns": spec["allow"],
        "ignore_patterns": HF_IGNORE,
    }

    # `local_dir_use_symlinks=False` and `resume_download=True` were the way to get
    # real files (not symlinks into the HF cache) and resumable transfers. Both are
    # now the default for local_dir downloads, deprecated in huggingface_hub 0.26+,
    # and REMOVED in 1.x — passing them unconditionally raises TypeError on new
    # versions, while omitting them on a pre-0.23 version would silently produce a
    # symlink farm that breaks the moment models/ is copied to another machine.
    # So: pass them only if this version still accepts them.
    import inspect

    accepted = inspect.signature(snapshot_download).parameters
    for legacy_kwarg, value in (("local_dir_use_symlinks", False), ("resume_download", True)):
        if legacy_kwarg in accepted:
            kwargs[legacy_kwarg] = value

    print(f"\n[{key}] {spec['repo']} -> {target}  ({spec['approx']})")
    snapshot_download(**kwargs)
    print(f"[{key}] done")


def download_lid(dest_root: Path) -> None:
    dest_root.mkdir(parents=True, exist_ok=True)
    target = dest_root / LID_FILENAME

    if target.exists() and target.stat().st_size >= LID_APPROX_BYTES * 0.99:
        print(f"\n[lid] already present ({_human(target.stat().st_size)}) - skipping")
        return

    print(f"\n[lid] {LID_URL} -> {target}  (~131 MB)")
    partial = target.with_suffix(target.suffix + ".part")

    def progress(block_count, block_size, total_size):
        if total_size <= 0:
            return
        done = min(block_count * block_size, total_size)
        pct = 100.0 * done / total_size
        print(f"\r[lid] {pct:5.1f}%  {_human(done)} / {_human(total_size)}", end="", flush=True)

    urllib.request.urlretrieve(LID_URL, partial, reporthook=progress)
    print()
    os.replace(partial, target)
    print(f"[lid] done ({_human(target.stat().st_size)})")


def main():
    default_dest = Path(__file__).resolve().parents[1] / "models"
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", default=str(default_dest),
                         help=f"Where to write the weights (default: {default_dest})")
    parser.add_argument("--only", nargs="+", choices=["embedding", "reranker", "llm", "lid"],
                         help="Fetch only these artifacts (default: all four)")
    args = parser.parse_args()

    dest_root = Path(args.dest).resolve()
    wanted = args.only or ["embedding", "reranker", "llm", "lid"]

    print(f"Model destination: {dest_root}")
    print(f"Fetching: {', '.join(wanted)}")

    for key in wanted:
        if key == "lid":
            download_lid(dest_root)
        else:
            download_hf_model(key, dest_root)

    total = sum(f.stat().st_size for f in dest_root.rglob("*") if f.is_file())
    print(f"\nAll requested models present. models/ is now {_human(total)}.")
    print("\nVerify the paths line up with config/config.yaml's models: section, then set")
    print("HF_HUB_OFFLINE=1 so nothing attempts a network call again:")
    print("    setx HF_HUB_OFFLINE 1")


if __name__ == "__main__":
    main()
