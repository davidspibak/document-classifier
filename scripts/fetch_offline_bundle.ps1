<#
.SYNOPSIS
    ONLINE step, run once: packs every wheel, tool and model weight needed to
    install, run and build this project on a machine with no internet.

.DESCRIPTION
    Produces:
      packages\wheels\              every runtime + build dependency as a .whl
      packages\wheels-torch-cuda\   (optional) CUDA torch, which is NOT on PyPI
      packages\tools\               the Tesseract OCR installer
      packages\MANIFEST.txt         what was fetched, and for which Python
      models\                       the four model artifacts (~8 GB)

    Then copy the whole checkout to the offline machine and run
    scripts\install_offline.ps1 there.

    IMPORTANT - a wheelhouse is Python-version specific. -PythonVersion decides
    which interpreter the offline machine must have; it does NOT have to match the
    interpreter running this script, because pip cross-fetches binaries for another
    target. 3.12 is the default and effectively mandatory: fasttext-wheel and the
    CUDA llama-cpp-python builds publish nothing newer than cp312.

.PARAMETER PythonVersion
    Target interpreter for the wheelhouse. Default 3.12.

.PARAMETER CudaTag
    Which prebuilt llama-cpp-python to fetch: cu124, cu122, or cpu.
    The CUDA builds only exist up to cp312.

.PARAMETER TorchCuda
    Also fetch CUDA-enabled torch from download.pytorch.org into a separate
    folder. PyPI's torch is CPU-only, so without this the embedding model,
    reranker and EasyOCR all run on CPU (correct, but slow).

.PARAMETER SkipWheels
.PARAMETER SkipModels
.PARAMETER SkipTools
    Fetch only part of the bundle.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\fetch_offline_bundle.ps1 -TorchCuda
#>
[CmdletBinding()]
param(
    [string]$PythonVersion = "3.12",
    [ValidateSet("cu124", "cu122", "cpu")]
    [string]$CudaTag = "cu124",
    [string]$Platform = "win_amd64",
    [switch]$TorchCuda,
    [string]$TorchCudaTag = "cu124",
    [switch]$SkipWheels,
    [switch]$SkipModels,
    [switch]$SkipTools
)

$ErrorActionPreference = "Stop"

$ProjectRoot   = Split-Path -Parent $PSScriptRoot
$PackagesDir   = Join-Path $ProjectRoot "packages"
$WheelDir      = Join-Path $PackagesDir "wheels"
$TorchCudaDir  = Join-Path $PackagesDir "wheels-torch-cuda"
$ToolsDir      = Join-Path $PackagesDir "tools"
$ModelsDir     = Join-Path $ProjectRoot "models"
$ManifestPath  = Join-Path $PackagesDir "MANIFEST.txt"

$LlamaIndex = "https://abetlen.github.io/llama-cpp-python/whl/$CudaTag"
$TorchIndex = "https://download.pytorch.org/whl/$TorchCudaTag"
$TesseractListing = "https://digi.bib.uni-mannheim.de/tesseract/"
$BrowserUserAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"

function Write-Step($message) {
    Write-Host ""
    Write-Host "=== $message ===" -ForegroundColor Cyan
}

function Assert-LastExitCode($what) {
    if ($LASTEXITCODE -ne 0) { throw "$what failed with exit code $LASTEXITCODE" }
}

function Get-DirSize($path) {
    if (-not (Test-Path $path)) { return 0 }
    $measured = Get-ChildItem -Recurse -Force -File $path | Measure-Object -Property Length -Sum
    if ($null -eq $measured.Sum) { return 0 }
    return $measured.Sum
}

Write-Step "Plan"
Write-Host "  target Python : $PythonVersion ($Platform)"
Write-Host "  llama-cpp     : $CudaTag"
if ($TorchCuda) { Write-Host "  torch         : CUDA ($TorchCudaTag) + CPU fallback" }
else            { Write-Host "  torch         : CPU only (pass -TorchCuda for GPU)" }
$fetchingWith = & python --version
Write-Host "  fetching with : $fetchingWith"

if ($PythonVersion -ne "3.12") {
    Write-Warning "PythonVersion is $PythonVersion, not 3.12. fasttext-wheel publishes no"
    Write-Warning "wheels past cp312 and neither do the CUDA llama-cpp-python builds, so the"
    Write-Warning "download will very likely fail to resolve. Continuing as asked."
}

New-Item -ItemType Directory -Force -Path $PackagesDir | Out-Null

# --- 1. wheelhouse --------------------------------------------------------------
if ($SkipWheels) {
    Write-Step "Skipping wheels (-SkipWheels)"
} else {
    Write-Step "Downloading wheels for Python $PythonVersion / $Platform"
    New-Item -ItemType Directory -Force -Path $WheelDir | Out-Null

    # --only-binary=:all: is mandatory when cross-targeting a different interpreter,
    # and is also exactly what we want: an sdist in the bundle would need a compiler
    # on the offline machine, which defeats the purpose.
    python -m pip download `
        --only-binary=:all: `
        --python-version $PythonVersion `
        --platform $Platform `
        --dest $WheelDir `
        --requirement (Join-Path $ProjectRoot "requirements.txt") `
        --requirement (Join-Path $ProjectRoot "requirements-build.txt")
    Assert-LastExitCode "pip download (requirements)"

    Write-Host ""
    Write-Host "  fetching prebuilt llama-cpp-python from $LlamaIndex"
    # PyPI ships llama-cpp-python as an sdist only (it compiles llama.cpp), so it
    # comes from abetlen's wheel index instead. --index-url, not --extra-index-url:
    # with PyPI in the mix the resolver picks PyPI's newer sdist and then rejects it.
    # --no-deps because its dependencies already came from the main wheelhouse.
    python -m pip download `
        --only-binary=:all: `
        --python-version $PythonVersion `
        --platform $Platform `
        --no-deps `
        --index-url $LlamaIndex `
        --dest $WheelDir `
        llama-cpp-python
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Could not fetch a prebuilt llama-cpp-python for cp$($PythonVersion.Replace('.','')) from $LlamaIndex."
        Write-Warning "The offline machine will need CUDA Toolkit + VS Build Tools to compile it,"
        Write-Warning "or try -CudaTag cpu for a CPU-only build."
    }

    Write-Host ("  wheelhouse: {0} files, {1:N2} GB" -f `
        (Get-ChildItem -File $WheelDir).Count, ((Get-DirSize $WheelDir) / 1GB))
}

# --- 2. CUDA torch --------------------------------------------------------------
if ($TorchCuda) {
    Write-Step "Downloading CUDA torch from $TorchIndex"
    New-Item -ItemType Directory -Force -Path $TorchCudaDir | Out-Null
    # Kept in its own folder: it shares a version number with PyPI's CPU torch, and
    # two same-version wheels in one --find-links directory resolve unpredictably.
    python -m pip download `
        --only-binary=:all: `
        --python-version $PythonVersion `
        --platform $Platform `
        --no-deps `
        --index-url $TorchIndex `
        --dest $TorchCudaDir `
        torch
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "CUDA torch download failed. The bundle still works on CPU."
    } else {
        Write-Host ("  CUDA torch: {0:N2} GB" -f ((Get-DirSize $TorchCudaDir) / 1GB))
    }
}

# --- 3. external tools ----------------------------------------------------------
if ($SkipTools) {
    Write-Step "Skipping tools (-SkipTools)"
} else {
    Write-Step "Downloading the Tesseract OCR installer"
    New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null
    try {
        # The Mannheim server 403s PowerShell's default user agent.
        $listing = Invoke-WebRequest -Uri $TesseractListing -UseBasicParsing `
                                      -UserAgent $BrowserUserAgent -TimeoutSec 60

        # Only stable releases: the listing also holds alphas, betas, rcs, a
        # "missing-dlls" respin and two conference builds, none of which we want.
        # Sorting the filenames as strings picks the wrong one - "v5.3.0" sorts above
        # "5.4.0" because of the 'v' - so parse the version and date and compare
        # numerically.
        $candidates = @()
        foreach ($match in [regex]::Matches($listing.Content, 'tesseract-ocr-w64-setup-v?(\d+)\.(\d+)\.(\d+)\.(\d{8})\.exe')) {
            $candidates += [pscustomobject]@{
                Name  = $match.Value
                Major = [int]$match.Groups[1].Value
                Minor = [int]$match.Groups[2].Value
                Patch = [int]$match.Groups[3].Value
                Date  = [int64]$match.Groups[4].Value
            }
        }
        if ($candidates.Count -eq 0) { throw "no stable installer found in the directory listing" }

        $newest = $candidates | Sort-Object Major, Minor, Patch, Date | Select-Object -Last 1
        $installer = $newest.Name
        $target = Join-Path $ToolsDir $installer

        if (Test-Path $target) {
            Write-Host "  $installer already present - skipping"
        } else {
            Write-Host "  selected $installer (newest stable of $($candidates.Count))"
            Invoke-WebRequest -Uri ($TesseractListing + $installer) -OutFile $target `
                              -UseBasicParsing -UserAgent $BrowserUserAgent -TimeoutSec 1800
            Write-Host ("  saved {0:N1} MB" -f ((Get-Item $target).Length / 1MB))
        }
    } catch {
        Write-Warning "Tesseract installer download failed: $($_.Exception.Message)"
        Write-Warning "Fetch it manually from $TesseractListing into $ToolsDir"
    }

    # GROBID is optional and ships as a Docker image / Java service, so there is no
    # single file to vendor. Record that rather than pretend otherwise.
    $grobidNote = @'
GROBID is NOT vendored here.

It is optional (the app falls back to LLM-based metadata extraction) and ships as
a Docker image or a Java service, neither of which is a single file that belongs
in a wheelhouse.

To use it on an offline machine, on a connected machine run:
    docker pull lfoppiano/grobid:0.8.0
    docker save lfoppiano/grobid:0.8.0 -o grobid-0.8.0.tar
copy that tar into this folder, then on the offline machine:
    docker load -i grobid-0.8.0.tar
    docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.8.0

Without it, set metadata.use_grobid: false in config/config.yaml to skip the
availability probe entirely.
'@
    Set-Content -Path (Join-Path $ToolsDir "GROBID_NOTE.txt") -Value $grobidNote -Encoding utf8
}

# --- 4. models ------------------------------------------------------------------
if ($SkipModels) {
    Write-Step "Skipping models (-SkipModels)"
} else {
    Write-Step "Downloading model weights (~8 GB, slow)"
    python -m pip install --quiet --upgrade huggingface_hub
    Assert-LastExitCode "pip install huggingface_hub"
    python (Join-Path $ProjectRoot "scripts\fetch_models.py") --dest $ModelsDir
    Assert-LastExitCode "fetch_models.py"
}

# --- 5. manifest ----------------------------------------------------------------
Write-Step "Writing manifest"
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("docclassify offline bundle")
$lines.Add("generated:        $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
$lines.Add("TARGET_PYTHON:    $PythonVersion")
$lines.Add("target_platform:  $Platform")
$lines.Add("llama_cpp_index:  $LlamaIndex")
if ($TorchCuda) { $lines.Add("torch_cuda_index: $TorchIndex") } else { $lines.Add("torch_cuda_index: (none - CPU torch only)") }
$lines.Add("")
$lines.Add("The offline machine MUST run Python $PythonVersion. A wheelhouse is not")
$lines.Add("portable across Python minor versions; install_offline.ps1 enforces this.")
$lines.Add("")
$lines.Add(("wheels:           {0} files, {1:N2} GB" -f (Get-ChildItem -File -ErrorAction SilentlyContinue $WheelDir).Count, ((Get-DirSize $WheelDir) / 1GB)))
$lines.Add(("wheels-torch-cuda:{0} files, {1:N2} GB" -f (Get-ChildItem -File -ErrorAction SilentlyContinue $TorchCudaDir).Count, ((Get-DirSize $TorchCudaDir) / 1GB)))
$lines.Add(("tools:            {0} files, {1:N2} MB" -f (Get-ChildItem -File -ErrorAction SilentlyContinue $ToolsDir).Count, ((Get-DirSize $ToolsDir) / 1MB)))
$lines.Add(("models:           {0:N2} GB" -f ((Get-DirSize $ModelsDir) / 1GB)))
$lines.Add("")
$lines.Add("--- wheels ---")
if (Test-Path $WheelDir) {
    Get-ChildItem -File $WheelDir | Sort-Object Name | ForEach-Object { $lines.Add("  $($_.Name)") }
}
Set-Content -Path $ManifestPath -Value $lines -Encoding utf8
Write-Host "  $ManifestPath"

# --- 6. report ------------------------------------------------------------------
Write-Step "Done"
$total = (Get-DirSize $WheelDir) + (Get-DirSize $TorchCudaDir) + (Get-DirSize $ToolsDir) + (Get-DirSize $ModelsDir)
Write-Host ("Bundle total: {0:N2} GB" -f ($total / 1GB))
Write-Host ""
Write-Host "Next, on the OFFLINE machine (which needs Python $PythonVersion installed):"
Write-Host "  1. copy this entire folder across"
Write-Host "  2. powershell -ExecutionPolicy Bypass -File scripts\install_offline.ps1"
Write-Host "  3. run the Tesseract installer from packages\tools\"
Write-Host "  4. powershell -ExecutionPolicy Bypass -File build\build.ps1 -Console -SkipEnvCheck"
