#!/usr/bin/env bash
#
# Environment setup for a Linux GPU box (tested target: vast.ai, Ubuntu, RTX 5090).
#
# This is the Linux counterpart to install_offline.ps1. It installs from PyPI rather
# than the vendored wheelhouse, because packages/wheels/ holds win_amd64 wheels and
# is useless here. A cloud GPU box has internet, so that is the right trade.
#
# Usage:
#   bash scripts/setup_linux.sh                 # full setup
#   bash scripts/setup_linux.sh --skip-apt      # if you lack sudo / already did it
#   bash scripts/setup_linux.sh --cpu-llm       # skip the CUDA llama.cpp compile
#
set -euo pipefail

SKIP_APT=0
CPU_LLM=0
for arg in "$@"; do
  case "$arg" in
    --skip-apt) SKIP_APT=1 ;;
    --cpu-llm)  CPU_LLM=1 ;;
    *) echo "unknown argument: $arg"; exit 1 ;;
  esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
VENV="$PROJECT_ROOT/.venv"

step() { echo; echo "=== $* ==="; }

# ---------------------------------------------------------------- 1. system packages
if [ "$SKIP_APT" -eq 0 ]; then
  step "System packages"
  SUDO=""
  if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi
  $SUDO apt-get update -qq
  # tesseract-ocr        : the OCR binary pytesseract shells out to (not a pip package)
  # libgl1, libglib2.0-0 : opencv-python links against these. Without them `import cv2`
  #                        fails with "libGL.so.1: cannot open shared object file",
  #                        which takes down the whole ingestion path on a headless box.
  # build-essential,cmake: needed to compile llama-cpp-python for Blackwell
  $SUDO apt-get install -y -qq \
      tesseract-ocr \
      libgl1 libglib2.0-0 \
      build-essential cmake git curl \
      python3-venv python3-dev
  echo "  tesseract: $(tesseract --version 2>&1 | head -1)"
else
  step "Skipping apt (--skip-apt)"
fi

# ---------------------------------------------------------------- 2. interpreter
step "Python interpreter"
PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    version="$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    case "$version" in
      3.10|3.11|3.12) PYTHON="$candidate"; break ;;
    esac
  fi
done

if [ -z "$PYTHON" ]; then
  echo "ERROR: need Python 3.10, 3.11 or 3.12."
  echo "The binding constraint is fasttext-wheel, whose newest Linux wheels are cp312."
  echo "Install one, e.g.:  sudo apt-get install -y python3.12 python3.12-venv"
  exit 1
fi
echo "  using $PYTHON ($("$PYTHON" --version))"

# ---------------------------------------------------------------- 3. virtualenv
step "Virtual environment"
if [ ! -x "$VENV/bin/python" ]; then
  "$PYTHON" -m venv "$VENV"
  echo "  created $VENV"
else
  echo "  reusing $VENV"
fi
PY="$VENV/bin/python"
"$PY" -m pip install --quiet --upgrade pip setuptools wheel

# ---------------------------------------------------------------- 4. torch first
step "PyTorch (installed first, so nothing else pulls a wrong build)"
# IMPORTANT: plain PyPI torch, NOT the download.pytorch.org/whl/cu124 index.
# That index tops out at torch 2.6 / CUDA 12.4, which predates Blackwell (sm_120):
# on a 5090 every kernel launch fails with "no kernel image is available for
# execution on the device". The default PyPI build is CUDA 13 and covers sm_120.
"$PY" -m pip install --upgrade torch
"$PY" - <<'PYEOF'
import torch
print(f"  torch {torch.__version__}  CUDA {torch.version.cuda}  available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    archs = torch.cuda.get_arch_list()
    print(f"  GPU: {name}  (sm_{major}{minor})")
    print(f"  this build supports: {archs}")
    if f"sm_{major}{minor}" not in archs and f"compute_{major}{minor}" not in archs:
        raise SystemExit(
            f"\nFATAL: torch has no kernels for sm_{major}{minor}.\n"
            "Every GPU operation would fail. Install a newer torch build."
        )
    print("  OK: this torch build has kernels for the installed GPU.")
else:
    print("  WARNING: CUDA not available — everything will run on CPU and be slow.")
PYEOF

# ---------------------------------------------------------------- 5. project deps
step "Project dependencies"
# PySide6 is the desktop UI. Nothing in the headless test path imports it, and it is
# ~170 MB plus X11 libraries, so it is dropped here.
grep -v -i '^pyside6' requirements.txt > /tmp/requirements-headless.txt
echo "  installing requirements.txt without pyside6 (headless box)"
"$PY" -m pip install --quiet -r /tmp/requirements-headless.txt
"$PY" -m pip install --quiet -r requirements-build.txt 2>/dev/null || true
"$PY" -m pip install --quiet pytest huggingface_hub

# ---------------------------------------------------------------- 6. llama-cpp-python
step "llama-cpp-python"
if [ "$CPU_LLM" -eq 1 ]; then
  echo "  CPU build (--cpu-llm)"
  "$PY" -m pip install --upgrade llama-cpp-python
else
  # There are no prebuilt Blackwell wheels — abetlen's newest Linux CUDA index is
  # cu124, whose binaries carry no sm_120 kernels. So compile, targeting this GPU's
  # architecture specifically (compiling for "all" takes far longer).
  ARCH="$("$PY" -c 'import torch;c=torch.cuda.get_device_capability(0);print(f"{c[0]}{c[1]}")' 2>/dev/null || echo "")"
  if [ -z "$ARCH" ]; then
    echo "  could not detect the GPU architecture; falling back to a CPU build"
    "$PY" -m pip install --upgrade llama-cpp-python
  else
    echo "  compiling for CUDA arch $ARCH (this takes 5-15 minutes)"
    if ! command -v nvcc >/dev/null 2>&1; then
      echo "  WARNING: nvcc not on PATH. The CUDA toolkit is required to compile."
      echo "           Try: export PATH=/usr/local/cuda/bin:\$PATH"
    fi
    CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${ARCH}" \
      "$PY" -m pip install --upgrade --no-cache-dir llama-cpp-python || {
        echo
        echo "  CUDA build failed. Falling back to CPU so the rest of the test can run."
        echo "  The LLM tie-breaker and report generation will be slow but correct."
        "$PY" -m pip install --upgrade llama-cpp-python
      }
  fi
fi

# ---------------------------------------------------------------- 7. verify
step "Verifying imports"
"$PY" - <<'PYEOF'
import importlib
modules = ["yaml","numpy","pandas","pyarrow","lancedb","fasttext","fitz","docx","pptx",
           "bs4","cv2","pytesseract","sklearn","keybert","pypinyin","requests","torch",
           "FlagEmbedding","easyocr"]
missing = []
for name in modules:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append((name, type(exc).__name__, str(exc)[:100]))
for name, kind, message in missing:
    print(f"  MISSING {name}: {kind}: {message}")
try:
    import llama_cpp
    print("  llama_cpp: ok")
except Exception as exc:
    print(f"  llama_cpp: UNAVAILABLE ({type(exc).__name__})")
print(f"\n  {len(modules) - len(missing)}/{len(modules)} core imports OK")
PYEOF

step "Done"
cat <<EOF
Next:
  source .venv/bin/activate
  python scripts/fetch_models.py                                  # ~8.8 GB
  python scripts/setup_check.py                                   # all models load?
  python scripts/import_taxonomy.py --file config/taxonomy_manual.json --replace
  python scripts/benchmark.py --folder data/testset --json results.json
EOF
