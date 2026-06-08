# alpha.66 publish RETRY.
#
# The alpha.66 topo order in publish_alpha66.ps1 is already correct (no new
# crates; cutlass-kernels-sys before kernels-sys; cublas/ozimmu before cutlass
# before kernels). So unlike the alpha.65 retry there is no topo bug to work
# around — this script exists only to re-drive crates that failed the main run
# on a transient sparse-index race or network blip.
#
# It re-walks the SAME dependency order as the main script and skips anything
# already on crates.io, so it's safe to run repeatedly. Between attempts on a
# failing crate it refreshes the index and backs off 45s so a just-published
# dependency becomes visible.

$ErrorActionPreference = "Stop"

$order = @(
    "baracuda-build",
    "baracuda-cutlass-sys",
    "baracuda-forge",
    "baracuda-types-derive",
    "baracuda-cutlass-kernels-sys",
    "baracuda-kernels-sys",
    "baracuda-types",
    "baracuda-core",
    "baracuda-cuda-sys",
    "baracuda-cudf-sys",
    "baracuda-cudnn-sys",
    "baracuda-cufft-sys",
    "baracuda-cufile-sys",
    "baracuda-cupti-sys",
    "baracuda-curand-sys",
    "baracuda-cusolver-sys",
    "baracuda-cusparse-sys",
    "baracuda-cutensor-sys",
    "baracuda-cuvs-sys",
    "baracuda-cvcuda-sys",
    "baracuda-flashinfer-sys",
    "baracuda-nccl-sys",
    "baracuda-npp-sys",
    "baracuda-nvcomp-sys",
    "baracuda-nvimagecodec-sys",
    "baracuda-nvjitlink-sys",
    "baracuda-nvjpeg-sys",
    "baracuda-nvml-sys",
    "baracuda-nvrtc-sys",
    "baracuda-nvshmem-sys",
    "baracuda-tensorrt-sys",
    "baracuda-cublas-sys",
    "baracuda-ozimmu-sys",
    "baracuda-transformer-engine-sys",
    "baracuda-driver",
    "baracuda-runtime",
    "baracuda-kernels-types",
    "baracuda-cudf",
    "baracuda-cudnn",
    "baracuda-cufft",
    "baracuda-cufile",
    "baracuda-cupti",
    "baracuda-curand",
    "baracuda-cusolver",
    "baracuda-cusparse",
    "baracuda-cutensor",
    "baracuda-cublas",
    "baracuda-ozimmu",
    "baracuda-cutlass",
    "baracuda-cuvs",
    "baracuda-cvcuda",
    "baracuda-transformer-engine",
    "baracuda-nccl",
    "baracuda-nvshmem",
    "baracuda-optim",
    "baracuda-megatron",   # MUST precede kernels (kernels optional-deps megatron via megatron_tp)
    "baracuda-kernels",
    "baracuda-flashinfer",
    "baracuda-npp",
    "baracuda-nvcomp",
    "baracuda-nvimagecodec",
    "baracuda-nvjitlink",
    "baracuda-nvjpeg",
    "baracuda-nvml",
    "baracuda-nvrtc",
    "baracuda-tensorrt",
    "baracuda"
)

$logFile = "target/publish_alpha66_retry.log"
New-Item -ItemType Directory -Force -Path "target" | Out-Null
Set-Content -Path $logFile -Value "alpha.66 RETRY run started $(Get-Date)" -NoNewline:$false

$published = 0
$skipped = 0
$failed = @()

foreach ($crate in $order) {
    Write-Host "$crate ..."
    Add-Content -Path $logFile -Value "`n=== $crate ==="
    $exit = -1
    $skippedThis = $false
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        $out = (cargo publish -p $crate --no-verify --allow-dirty 2>&1) -join "`n"
        Add-Content -Path $logFile -Value $out
        $exit = $LASTEXITCODE
        if ($exit -eq 0) { break }
        if ($out -match "already (uploaded|exists)") { $skippedThis = $true; $exit = 0; break }
        if ($out -match "failed to select a version|no matching package|Could not resolve host|spurious network|connection (reset|refused)") {
            Write-Host "    [retry $attempt/5] dep not visible / transient - refresh + 45s backoff"
            cargo update --workspace 2>&1 | Out-Null
            Start-Sleep -Seconds 45
            continue
        }
        break
    }
    if ($exit -ne 0) {
        Write-Host "    FAILED - see $logFile"
        $failed += $crate
    } elseif ($skippedThis) {
        $skipped++
    } else {
        Write-Host "    published"
        $published++
    }
}

Write-Host ""
Write-Host "=== Retry summary ==="
Write-Host "Published: $published"
Write-Host "Skipped:   $skipped"
Write-Host "Failed:    $($failed.Count)"
foreach ($f in $failed) { Write-Host "  - $f" }
