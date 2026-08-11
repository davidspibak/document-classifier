"""
PyInstaller hook for LanceDB.

Why this is needed: lancedb is a thin Python layer over a compiled Rust core. The
work is done by the native extension in the `lance`/`pylance` package (and
lancedb's own `_lancedb` extension), which PyInstaller's analysis under-collects —
it picks up the Python modules and leaves the extension or its sibling data files
behind. The failure shows up as an ImportError on `lancedb.connect()`, i.e. the
first time anything touches the vector store.

`lance` is collected separately and tolerantly: it's a distinct distribution
(pylance) whose presence and internal layout vary by lancedb version.
"""
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

datas, binaries, hiddenimports = collect_all("lancedb")

for companion in ("lance", "lancedb.remote", "lancedb.embeddings"):
    try:
        companion_datas, companion_binaries, companion_hidden = collect_all(companion)
    except Exception as exc:  # noqa: BLE001 - optional across lancedb versions
        print(f"[hook-lancedb] note: collect_all({companion!r}) skipped: {exc}")
        continue
    datas += companion_datas
    binaries += companion_binaries
    hiddenimports += companion_hidden

# The compiled extensions themselves, in case collect_all treated them as data.
for package in ("lancedb", "lance"):
    try:
        binaries += collect_dynamic_libs(package)
    except Exception as exc:  # noqa: BLE001
        print(f"[hook-lancedb] note: collect_dynamic_libs({package!r}) skipped: {exc}")

hiddenimports += [
    "lancedb._lancedb",
    "lancedb.table",
    "lancedb.query",
    "lancedb.db",
]
