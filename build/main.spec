# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the offline desktop app.

Mode: --onedir ("standalone"), NOT --onefile. A onefile build unpacks the entire
GB-scale ML dependency tree into a temp directory on every launch, which is far
too slow for an app people open repeatedly.

Three things stay OUTSIDE the bundle on purpose, sitting next to the .exe:
  * config/  - the user edits thresholds and model paths here; keeping it external
               means a config change doesn't need a rebuild.
  * models/  - ~15-20 GB of weights. Bundling them would make the executable
               unmanageable and force a rebuild to swap a model.
  * data/    - the SQLite database and LanceDB tables the app writes at runtime.

config.py's _resolve_project_root() is what makes that layout work: it detects
sys.frozen and anchors those three paths to the .exe's own directory instead of
walking up from __file__ (which points into PyInstaller's private bundle dir).
build.ps1 puts them in place after the build.

Build with:  build\\build.ps1
or directly: pyinstaller build\\main.spec --noconfirm --distpath dist --workpath build\\work
"""
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

PROJECT_ROOT = Path(SPECPATH).parent
SRC_DIR = PROJECT_ROOT / "src"

# A console window is invaluable while getting a first build to work — a traceback
# from a windowed app goes nowhere at all — and unwanted in the shipped app.
# Set DOCCLASSIFY_CONSOLE=1 (or pass -Console to build.ps1) to keep it.
CONSOLE = os.environ.get("DOCCLASSIFY_CONSOLE", "0").lower() not in ("0", "", "false", "no")

datas = []
binaries = []
hiddenimports = []

# Packages whose data files or dynamically-referenced submodules PyInstaller's
# static analysis does not find on its own. Everything NOT listed here is left to
# PyInstaller's own bundled hooks (torch, transformers, scikit-learn, scipy, cv2,
# PIL, numpy, pandas and pyarrow all ship one) — re-collecting those here would
# only duplicate work and slow the build down.
#
# llama_cpp and lancedb are deliberately absent: they need more than collect_all
# and have dedicated hooks in build/hooks/.
COLLECT_PACKAGES = [
    "pypinyin",       # pinyin dictionaries (plain data files)
    "docx",           # python-docx ships a default .docx template it opens at runtime
    "pptx",           # python-pptx ships the same for .pptx
    "easyocr",        # character-set lists and per-language model config
    "FlagEmbedding",  # imports its model classes dynamically
    "keybert",
    "fasttext",
    "hdbscan",
]

for package in COLLECT_PACKAGES:
    try:
        package_datas, package_binaries, package_hidden = collect_all(package)
    except Exception as exc:  # noqa: BLE001
        # Non-fatal on purpose: an optional package that isn't installed shouldn't
        # abort a 20-minute build. Grep the build log for these before shipping.
        print(f"[main.spec] WARNING: collect_all({package!r}) failed, continuing without it: {exc}")
        continue
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

# PyMuPDF is importable under two names depending on version; take whichever works.
for pymupdf_name in ("pymupdf", "fitz"):
    try:
        package_datas, package_binaries, package_hidden = collect_all(pymupdf_name)
    except Exception as exc:  # noqa: BLE001
        print(f"[main.spec] WARNING: collect_all({pymupdf_name!r}) failed: {exc}")
        continue
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden
    break

# Modules reached only at runtime. The first group is the classic scikit-learn set
# PyInstaller misses (they're pulled in by compiled extensions, not by an import
# statement); the rest are our own lazily-imported modules, listed explicitly so a
# missing one fails the BUILD rather than a user's click three views in.
hiddenimports += [
    "sklearn.utils._typedefs",
    "sklearn.utils._heap",
    "sklearn.utils._sorting",
    "sklearn.utils._vector_sentinel",
    "sklearn.neighbors._partition_nodes",
    "scipy._lib.array_api_compat.numpy.fft",
    "docclassify.classification.llm_tiebreaker",
    "docclassify.metadata.extract",
    "docclassify.metadata.grobid_client",
    "docclassify.metadata.keyword_extract",
    "docclassify.metadata.translate",
    "docclassify.reports.doc_summary",
    "docclassify.reports.monthly_report",
    "docclassify.taxonomy.cluster",
    "docclassify.taxonomy.label",
    "docclassify.taxonomy.taxonomy_store",
]

# Trimmed because nothing imports them and they are large or pull in a GUI toolkit
# we don't use. Keep this list short: over-excluding produces a build that only
# fails at runtime.
excludes = [
    "tkinter",
    "pytest",
    "_pytest",
]

a = Analysis(
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(SRC_DIR)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(Path(SPECPATH) / "hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="docclassify",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX stays OFF. Compressing CUDA/torch DLLs is a well-known source of builds
    # that run on the build machine and fail silently on another one.
    upx=False,
    console=CONSOLE,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="docclassify",
)
