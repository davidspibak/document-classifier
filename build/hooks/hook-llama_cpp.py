"""
PyInstaller hook for llama-cpp-python.

Why this is needed: llama_cpp does not `import` its backend. It loads
`llama_cpp/lib/llama.dll` (plus ggml*.dll, and the CUDA variants when built with
GPU support) through `ctypes.CDLL` at runtime, using a path computed relative to
the package. PyInstaller's static analysis sees no import and bundles none of it,
so the build succeeds and the app dies on first LLM call with a bare
"Shared library with base name 'llama' not found".

collect_dynamic_libs() finds the DLLs; the explicit collect_data_files pass is a
belt-and-braces for versions that ship the libraries somewhere other than lib/,
and preserves the package-relative layout ctypes expects.
"""
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

binaries = collect_dynamic_libs("llama_cpp")

# Catch backend libraries that collect_dynamic_libs misses because of where they
# sit in the wheel. include_py_files=False keeps this to the native artifacts.
binaries += [
    (source, destination)
    for source, destination in collect_data_files(
        "llama_cpp", include_py_files=False,
        includes=["**/*.dll", "**/*.so", "**/*.dylib", "**/*.metal"],
    )
]

datas = collect_data_files("llama_cpp", excludes=["**/*.dll", "**/*.so", "**/*.dylib"])

hiddenimports = [
    "llama_cpp.llama_cpp",
    "llama_cpp._internals",
    "llama_cpp._ctypes_extensions",
    "llama_cpp.llama_chat_format",
    "llama_cpp.llama_grammar",
    "llama_cpp.llama_types",
]
