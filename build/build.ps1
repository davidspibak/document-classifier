<#
.SYNOPSIS
    Builds the offline Windows distributable: env check -> clean -> PyInstaller -> assemble.

.DESCRIPTION
    Produces dist\docclassify\, a self-contained --onedir app. After the build it
    assembles the runtime layout the app expects next to the .exe: config\ copied
    in, and empty data\ and models\ trees created with a note listing exactly which
    model files have to be dropped in.

    Once models\ is populated the whole dist\docclassify\ folder needs no network
    access — zip it and move it to the offline machine.

.PARAMETER Python
    Interpreter to build with. Defaults to the project venv.

.PARAMETER Console
    Keep a console window attached to the built .exe. Do this for a first build:
    a windowed app swallows tracebacks entirely.

.PARAMETER SkipEnvCheck
    Skip scripts\setup_check.py. That check loads every model, so it needs models\
    populated on the BUILD machine — skip it if you are only producing the binary
    and will populate models on the target.

.PARAMETER KeepWork
    Don't delete build\work\ afterwards. PyInstaller's warn-docclassify.txt and
    xref-docclassify.html live there and are what you read when an import is missing.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File build\build.ps1 -Console -SkipEnvCheck
#>
[CmdletBinding()]
param(
    [string]$Python = "",
    [switch]$Console,
    [switch]$SkipEnvCheck,
    [switch]$KeepWork
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BuildDir    = Join-Path $ProjectRoot "build"
$SpecFile    = Join-Path $BuildDir   "main.spec"
$WorkDir     = Join-Path $BuildDir   "work"
$DistDir     = Join-Path $ProjectRoot "dist"
$AppDir      = Join-Path $DistDir    "docclassify"

function Write-Step($message) {
    Write-Host ""
    Write-Host "=== $message ===" -ForegroundColor Cyan
}

function Assert-LastExitCode($what) {
    if ($LASTEXITCODE -ne 0) {
        throw "$what failed with exit code $LASTEXITCODE"
    }
}

# --- 0. locate the interpreter -------------------------------------------------
if ([string]::IsNullOrWhiteSpace($Python)) {
    $venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        $Python = $venvPython
    } else {
        Write-Warning "No .venv found at $venvPython - falling back to 'python' on PATH."
        $Python = "python"
    }
}

Write-Step "Environment"
& $Python --version
Assert-LastExitCode "python --version"

$pyVersion = & $Python -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ($pyVersion -ne "3.12") {
    Write-Warning "Building with Python $pyVersion. This project targets 3.12 - several"
    Write-Warning "dependencies (fasttext-wheel, the CUDA llama-cpp-python wheels) have no"
    Write-Warning "prebuilt binaries beyond cp312. Continuing, but expect trouble."
}

& $Python -c "import PyInstaller; print('PyInstaller', PyInstaller.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller is not installed in $Python. Install it with: pip install pyinstaller (or, offline: pip install --no-index --find-links packages\wheels pyinstaller)"
}

if (-not (Test-Path $SpecFile)) {
    throw "Spec file not found: $SpecFile"
}

# --- 1. optional pre-flight check ----------------------------------------------
if ($SkipEnvCheck) {
    Write-Step "Skipping setup_check.py (-SkipEnvCheck)"
} else {
    Write-Step "Verifying every model and service loads"
    & $Python (Join-Path $ProjectRoot "scripts\setup_check.py")
    if ($LASTEXITCODE -ne 0) {
        throw "setup_check.py failed. Fix the reported item, or re-run with -SkipEnvCheck to build anyway."
    }
}

# --- 2. clean previous output ---------------------------------------------------
Write-Step "Cleaning previous build output"
foreach ($path in @($WorkDir, $AppDir)) {
    if (Test-Path $path) {
        Write-Host "  removing $path"
        Remove-Item -Recurse -Force $path -Confirm:$false
    }
}

# --- 3. build -------------------------------------------------------------------
Write-Step "Running PyInstaller (first build takes 10-20+ minutes)"
if ($Console) {
    $env:DOCCLASSIFY_CONSOLE = "1"
    Write-Host "  console window: ENABLED"
} else {
    $env:DOCCLASSIFY_CONSOLE = "0"
    Write-Host "  console window: disabled (use -Console for a debuggable build)"
}

$buildStart = Get-Date
& $Python -m PyInstaller $SpecFile --noconfirm --distpath $DistDir --workpath $WorkDir
Assert-LastExitCode "PyInstaller"
$elapsed = (Get-Date) - $buildStart
Write-Host ("  built in {0:N1} minutes" -f $elapsed.TotalMinutes)

if (-not (Test-Path $AppDir)) {
    throw "PyInstaller reported success but $AppDir does not exist."
}

# Surface the collect_all warnings the spec deliberately swallows.
$warnFile = Join-Path $WorkDir "docclassify\warn-docclassify.txt"
if (Test-Path $warnFile) {
    $missing = Select-String -Path $warnFile -Pattern "^missing module named" -ErrorAction SilentlyContinue
    if ($missing) {
        Write-Host "  $($missing.Count) 'missing module' notes in warn-docclassify.txt (most are harmless)" -ForegroundColor DarkYellow
    }
}

# --- 4. assemble the runtime layout --------------------------------------------
Write-Step "Assembling the distributable"

Copy-Item -Recurse -Force (Join-Path $ProjectRoot "config") (Join-Path $AppDir "config")
Write-Host "  copied config\"

foreach ($sub in @("data\inbox", "data\raw", "data\sqlite", "data\lancedb", "models")) {
    $target = Join-Path $AppDir $sub
    New-Item -ItemType Directory -Force -Path $target | Out-Null
}
Write-Host "  created data\ and models\"

$modelsNote = @'
Drop the model weights in this folder before running docclassify.exe.

Expected layout (must match the models: section of ..\config\config.yaml):

  models\bge-m3\                                 BAAI/bge-m3            ~2.2 GB
  models\bge-reranker-v2-m3\                     BAAI/bge-reranker-v2-m3 ~1.1 GB
  models\qwen2.5-7b-instruct-q4_k_m.gguf         Qwen2.5-7B-Instruct    ~4.5 GB
  models\lid.176.bin                             fastText lid.176       ~130 MB

If you built the offline bundle with scripts\fetch_offline_bundle.ps1, copy the
contents of that checkout's models\ folder in here wholesale.

Once this folder is populated the app needs no network access at all.
'@
Set-Content -Path (Join-Path $AppDir "models\MODELS_REQUIRED.txt") -Value $modelsNote -Encoding utf8
Write-Host "  wrote models\MODELS_REQUIRED.txt"

if (-not $KeepWork) {
    if (Test-Path $WorkDir) {
        Remove-Item -Recurse -Force $WorkDir -Confirm:$false
    }
} else {
    Write-Host "  kept $WorkDir (-KeepWork) - see warn-docclassify.txt for import diagnostics"
}

# --- 5. report ------------------------------------------------------------------
Write-Step "Done"
$sizeBytes = (Get-ChildItem -Recurse -Force $AppDir | Measure-Object -Property Length -Sum).Sum
Write-Host ("Output: {0}" -f $AppDir)
Write-Host ("Size:   {0:N2} GB (excluding model weights)" -f ($sizeBytes / 1GB))
Write-Host ""
Write-Host "Next: populate $AppDir\models\ (see MODELS_REQUIRED.txt), then run:"
Write-Host "  $AppDir\docclassify.exe"
