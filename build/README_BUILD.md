# Packaging notes

Gotchas specific to this project's dependency stack. Read this before debugging a
build; most first-build failures here are one of the known items below.

## The two-command version

```powershell
# on a connected machine, once:
powershell -ExecutionPolicy Bypass -File scripts\fetch_offline_bundle.ps1 -TorchCuda

# on the target machine (no network needed from here on):
powershell -ExecutionPolicy Bypass -File scripts\install_offline.ps1 -TorchCuda
powershell -ExecutionPolicy Bypass -File build\build.ps1 -Console -SkipEnvCheck
```

## Python must be 3.12

Not a preference — a constraint, and the reason is a single package:

`fasttext` (language identification) publishes **no wheels at all**, only an sdist,
so installing it compiles C++ and needs Visual Studio Build Tools. That is fatal for
an offline install. `requirements.txt` therefore uses **`fasttext-wheel`**, the same
library with prebuilt binaries and the same `import fasttext` module name — and its
newest wheels are `cp312`.

Everything else in the stack (torch, lancedb, PySide6, the CUDA `llama-cpp-python`
build, which ships as a version-independent `py3-none-win_amd64` wheel) is fine on
newer Pythons. If `fasttext-wheel` ever publishes cp313+, or you swap in a different
language-ID library, this constraint lifts.

A wheelhouse is **not portable across Python minor versions**.
`packages\MANIFEST.txt` records which version it was built for and
`install_offline.ps1` refuses to proceed on a mismatch rather than failing halfway.

You do **not** need Python 3.12 on the machine that *builds* the bundle:
`pip download --python-version 3.12 --platform win_amd64 --only-binary=:all:`
cross-fetches binaries for another interpreter. Only the target machine needs 3.12.

## What the two custom hooks are for

Both cover the same class of failure — a library loaded by a mechanism
PyInstaller's static analysis cannot see — and both fail *at runtime*, not at build
time, which is what makes them nasty.

**`hook-llama_cpp.py`** — `llama_cpp` doesn't import its backend; it `ctypes.CDLL`s
`llama_cpp/lib/llama.dll` (plus the ggml and CUDA DLLs) from a path computed
relative to the package. Without the hook the build succeeds and the first LLM call
dies with `Shared library with base name 'llama' not found`.

**`hook-lancedb.py`** — lancedb is a thin Python layer over a compiled Rust core
living in the `lance`/`pylance` package. Analysis collects the Python modules and
leaves the extension behind, so `lancedb.connect()` raises ImportError the first
time anything touches the vector store.

## `--onedir`, not `--onefile`

A onefile build unpacks the whole GB-scale dependency tree into a temp directory on
**every launch**. For a desktop app people open repeatedly that is unacceptable, and
the unpack also duplicates the payload on disk each run.

## UPX stays off

`upx=False` in `main.spec`, deliberately. Compressing CUDA and torch DLLs with UPX
is a well-known cause of builds that work on the build machine and fail silently
elsewhere. Do not "optimize" this back on.

## What stays outside the .exe

`config/`, `models/` and `data/` sit next to the executable, not inside the bundle.
This only works because `config.py`'s `_resolve_project_root()` detects `sys.frozen`
and anchors to `sys.executable`'s directory:

```python
def _resolve_project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]
```

If you ever pull `config.py` from an older copy of this project, re-apply that or
the packaged app will look for `config/` inside PyInstaller's private temp dir.

Rationale: `config/` stays user-editable without a rebuild, `models/` is ~8 GB and
swapping a model shouldn't mean re-packaging, and `data/` is written at runtime.

## Debugging a build

**Always use `-Console` for a first build.** A windowed PyInstaller app discards
stdout and stderr entirely, so a missing import shows up as a window that never
appears and nothing else.

**`-KeepWork`** preserves `build\work\`, which holds the two files worth reading:

| File | Use |
|---|---|
| `warn-docclassify.txt` | every module PyInstaller couldn't resolve. Most entries are harmless optional imports; search it for your failing module name. |
| `xref-docclassify.html` | the full import graph — shows *why* something got pulled in. |

If a module is genuinely missing, add it to `hiddenimports` in `main.spec` rather
than reaching for `collect_all` on its whole distribution.

**The spec swallows `collect_all` failures on purpose** (an optional package that
isn't installed shouldn't abort a 20-minute build), so grep the build log for
`[main.spec] WARNING:` before shipping a build.

## Known quirks

- **First build takes 10–20+ minutes.** torch, transformers, opencv and PySide6 are
  all large. Later builds are faster once caches warm.
- **Two copies of OpenCV.** `opencv-python` is a direct dependency and easyocr pulls
  `opencv-python-headless`; both provide `cv2`. Harmless — PyInstaller bundles one —
  but it explains a chunk of the output size.
- **GPU DLLs can't be fully bundled.** With CUDA-enabled torch and
  `llama-cpp-python` on the build machine, the spec picks up the matching CUDA
  runtime DLLs, but the *target* machine still needs a compatible NVIDIA driver.
- **PyPI's torch is CPU-only.** The GPU build comes from `download.pytorch.org`,
  which is why `fetch_offline_bundle.ps1` has a separate `-TorchCuda` switch and
  keeps those wheels in their own directory (same version number as the CPU wheel,
  so mixing them in one `--find-links` folder resolves unpredictably).
- **`vllm` is not in the bundle and cannot be.** It has no Windows wheels. The
  `TODO (throughput)` batching path in `scripts/bulk_init_classify.py` therefore
  can't be exercised on Windows as things stand; the per-item llama.cpp fallback in
  that same function is what actually runs.
- **GROBID isn't vendored** — it's a Docker image / Java service, not a file. See
  `packages\tools\GROBID_NOTE.txt` for the `docker save` / `docker load` route, or
  set `metadata.use_grobid: false` in `config/config.yaml` to skip the probe.
- **Tesseract is a separate installer**, in `packages\tools\`. If `pytesseract`
  can't find it after installation, set the path explicitly (README §5.3).

## Verifying a bundle without a 3.12 interpreter

This resolves the whole dependency set against the local wheelhouse with the network
disabled, and confirms every package came from a `file://` URL:

```powershell
python -m pip install --dry-run --ignore-installed --no-index `
    --find-links packages\wheels --only-binary=:all: `
    --python-version 3.12 --platform win_amd64 --target $env:TEMP\check `
    -r requirements.txt -r requirements-build.txt
```

A clean exit means the offline install will resolve. It does not prove the wheels
*run* — only a real 3.12 install can do that.
