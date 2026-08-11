<#
.SYNOPSIS
    OFFLINE step: builds the virtual environment entirely from packages\wheels\,
    with no network access.

.DESCRIPTION
    Run this on the disconnected machine after copying the checkout across. It
    creates .venv, installs every dependency from the vendored wheelhouse with
    --no-index (so pip cannot silently reach out to PyPI), and then tells you what
    still needs doing by hand.

    Requires the Python version recorded in packages\MANIFEST.txt — a wheelhouse is
    not portable across Python minor versions, so this is checked and refused
    rather than left to fail confusingly halfway through.

.PARAMETER PythonExe
    Interpreter to build the venv from. Default: whatever `py -3.12` resolves to,
    falling back to `python`.

.PARAMETER TorchCuda
    After the main install, force-reinstall torch from packages\wheels-torch-cuda\
    (only present if the bundle was fetched with -TorchCuda).

.PARAMETER SkipVerify
    Don't run the post-install import check.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install_offline.ps1 -TorchCuda
#>
[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [switch]$TorchCuda,
    [switch]$SkipVerify
)

$ErrorActionPreference = "Stop"

$ProjectRoot  = Split-Path -Parent $PSScriptRoot
$PackagesDir  = Join-Path $ProjectRoot "packages"
$WheelDir     = Join-Path $PackagesDir "wheels"
$TorchCudaDir = Join-Path $PackagesDir "wheels-torch-cuda"
$ToolsDir     = Join-Path $PackagesDir "tools"
$ManifestPath = Join-Path $PackagesDir "MANIFEST.txt"
$VenvDir      = Join-Path $ProjectRoot ".venv"
$VenvPython   = Join-Path $VenvDir "Scripts\python.exe"

function Write-Step($message) {
    Write-Host ""
    Write-Host "=== $message ===" -ForegroundColor Cyan
}

function Assert-LastExitCode($what) {
    if ($LASTEXITCODE -ne 0) { throw "$what failed with exit code $LASTEXITCODE" }
}

# --- 0. sanity ------------------------------------------------------------------
Write-Step "Checking the bundle"

if (-not (Test-Path $WheelDir)) {
    throw "No wheelhouse at $WheelDir. Run scripts\fetch_offline_bundle.ps1 on a connected machine first."
}
$wheelCount = (Get-ChildItem -File -Filter *.whl $WheelDir).Count
if ($wheelCount -eq 0) {
    throw "$WheelDir contains no .whl files."
}
Write-Host "  $wheelCount wheels found"

$requiredPython = "3.12"
if (Test-Path $ManifestPath) {
    $line = Select-String -Path $ManifestPath -Pattern '^TARGET_PYTHON:\s*(\S+)' | Select-Object -First 1
    if ($line) {
        $requiredPython = $line.Matches[0].Groups[1].Value
        Write-Host "  manifest says the wheelhouse targets Python $requiredPython"
    }
} else {
    Write-Warning "No MANIFEST.txt - assuming the wheelhouse targets Python $requiredPython"
}

# --- 1. locate a matching interpreter -------------------------------------------
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $candidate = $null
    try {
        $resolved = & py "-$requiredPython" -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0) { $candidate = $resolved }
    } catch {
        $candidate = $null
    }
    if ([string]::IsNullOrWhiteSpace($candidate)) { $PythonExe = "python" } else { $PythonExe = $candidate }
}

Write-Step "Interpreter"
$actual = & $PythonExe -c "import sys; print('%d.%d' % sys.version_info[:2])"
Assert-LastExitCode "querying $PythonExe"
Write-Host "  $PythonExe -> Python $actual"

if ($actual -ne $requiredPython) {
    throw @"
Python version mismatch.

  wheelhouse built for : $requiredPython
  interpreter found    : $actual

A wheelhouse only contains binaries for one Python minor version, so installing
from it with $actual cannot work. Either install Python $requiredPython on this
machine and pass -PythonExe, or re-run fetch_offline_bundle.ps1 on the connected
machine with -PythonVersion $actual.
"@
}

# --- 2. venv --------------------------------------------------------------------
Write-Step "Creating the virtual environment"
if (Test-Path $VenvPython) {
    Write-Host "  $VenvDir already exists - reusing it"
} else {
    # --without-pip because ensurepip's bundled pip may be older than the wheelhouse
    # expects; the correct pip is installed from the bundle in the next step.
    & $PythonExe -m venv --without-pip $VenvDir
    Assert-LastExitCode "python -m venv"
    Write-Host "  created $VenvDir"
}

Write-Step "Bootstrapping pip from the bundle"
$pipWheel = Get-ChildItem -File -Filter "pip-*.whl" $WheelDir | Select-Object -First 1
if ($null -eq $pipWheel) {
    throw "No pip wheel in $WheelDir - requirements-build.txt should have put one there."
}
# A venv made --without-pip has no pip to invoke, so run the wheel itself: a pip
# wheel is a valid zipimport target.
& $VenvPython $pipWheel.FullName install --no-index --find-links $WheelDir pip setuptools wheel
Assert-LastExitCode "pip bootstrap"

# --- 3. dependencies ------------------------------------------------------------
Write-Step "Installing runtime and build dependencies (no network)"
& $VenvPython -m pip install --no-index --find-links $WheelDir `
    --requirement (Join-Path $ProjectRoot "requirements.txt") `
    --requirement (Join-Path $ProjectRoot "requirements-build.txt")
Assert-LastExitCode "pip install requirements"

Write-Step "Installing llama-cpp-python"
$llamaWheel = Get-ChildItem -File -Filter "llama_cpp_python-*.whl" $WheelDir | Select-Object -First 1
if ($null -eq $llamaWheel) {
    Write-Warning "No llama-cpp-python wheel in the bundle. The local LLM (tie-breaker,"
    Write-Warning "report generation, metadata fallback) will be unavailable. Everything"
    Write-Warning "else - ingestion, embedding classification, search - still works."
} else {
    & $VenvPython -m pip install --no-index --find-links $WheelDir $llamaWheel.FullName
    Assert-LastExitCode "pip install llama-cpp-python"
    Write-Host "  installed $($llamaWheel.Name)"
}

if ($TorchCuda) {
    Write-Step "Replacing torch with the CUDA build"
    if (-not (Test-Path $TorchCudaDir)) {
        Write-Warning "$TorchCudaDir does not exist - the bundle was fetched without -TorchCuda."
        Write-Warning "Staying on CPU torch."
    } else {
        & $VenvPython -m pip install --no-index --find-links $TorchCudaDir --force-reinstall --no-deps torch
        Assert-LastExitCode "pip install CUDA torch"
    }
}

# --- 4. verify ------------------------------------------------------------------
if (-not $SkipVerify) {
    Write-Step "Verifying imports (models not required yet)"
    $probe = @'
import importlib, sys
mods = ["yaml","numpy","pandas","pyarrow","lancedb","fasttext","fitz","docx","pptx","bs4",
        "cv2","pytesseract","PySide6","sklearn","keybert","pypinyin","requests","torch",
        "FlagEmbedding","easyocr"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append((m, type(e).__name__, str(e)[:90]))
for m, kind, msg in missing:
    print(f"  MISSING {m}: {kind}: {msg}")
try:
    import llama_cpp
    print("  llama_cpp: ok")
except Exception as e:
    print(f"  llama_cpp: unavailable ({type(e).__name__})")
try:
    import torch
    print(f"  torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
except Exception:
    pass
print(f"\n{len(mods) - len(missing)}/{len(mods)} core imports OK")
sys.exit(1 if missing else 0)
'@
    $probePath = Join-Path $env:TEMP "docclassify_probe.py"
    Set-Content -Path $probePath -Value $probe -Encoding utf8
    & $VenvPython $probePath
    $probeExit = $LASTEXITCODE
    Remove-Item $probePath -Force -ErrorAction SilentlyContinue
    if ($probeExit -ne 0) {
        Write-Warning "Some imports failed - see above. The wheelhouse may be incomplete."
    }
}

# --- 5. what's left -------------------------------------------------------------
Write-Step "Remaining manual steps"

$tesseract = Get-ChildItem -File -Filter "tesseract-ocr-w64-setup-*.exe" $ToolsDir -ErrorAction SilentlyContinue | Select-Object -First 1
if ($null -eq $tesseract) {
    Write-Host "  [ ] Install Tesseract (no installer found in packages\tools\)"
} else {
    Write-Host "  [ ] Install Tesseract: $($tesseract.FullName)"
    Write-Host "      If pytesseract can't find it afterwards, set the path explicitly -"
    Write-Host "      see README section 5.3."
}

$modelsDir = Join-Path $ProjectRoot "models"
$modelFiles = 0
if (Test-Path $modelsDir) {
    $modelFiles = (Get-ChildItem -Recurse -File $modelsDir -ErrorAction SilentlyContinue).Count
}
if ($modelFiles -eq 0) {
    Write-Host "  [ ] Populate models\ - it is empty. Copy it from the machine that ran"
    Write-Host "      fetch_offline_bundle.ps1."
} else {
    Write-Host "  [x] models\ has $modelFiles files"
}

Write-Host "  [ ] setx HF_HUB_OFFLINE 1   (belt-and-braces: forbids any HF network call)"
Write-Host ""
Write-Host "Then:"
Write-Host "  .venv\Scripts\activate"
Write-Host "  python scripts\setup_check.py       # confirms every model loads"
Write-Host "  python main.py                     # run the app"
Write-Host "  powershell -ExecutionPolicy Bypass -File build\build.ps1 -Console -SkipEnvCheck"
