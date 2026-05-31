# Phase 62 (alpha.62) publish script.
# alpha.62 ships strided in-place op support + comprehensive test
# investment: 11 new affine in-place FFI symbols (4 contig int dtype
# backfill + 7 strided variants matching forward affine matrix) +
# documented stride-equality aliasing contract on unary/binary/ternary
# strided trailblazers + new `baracuda_kernels_types::strides_equal`
# helper + 38 new tests (14 host-only + 24 GPU).
# Crate roster unchanged from alpha.62.
#
# Bursts the first 28 publishes (crates.io's burst window), then sleeps 61s
# between each subsequent publish. See feedback_publish_pacing.md.
# Skipped (already-uploaded) crates do NOT consume burst credits.

$ErrorActionPreference = "Stop"

$order = @(
    "baracuda-build",
    "baracuda-cutlass-sys",
    "baracuda-forge",
    "baracuda-kernels-sys",
    "baracuda-types-derive",
    "baracuda-cutlass-kernels-sys",
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
    "baracuda-cvcuda-sys",
    "baracuda-driver",
    "baracuda-kernels-types",
    "baracuda-nccl-sys",
    "baracuda-npp-sys",
    "baracuda-nvcomp-sys",
    "baracuda-nvjitlink-sys",
    "baracuda-nvjpeg-sys",
    "baracuda-nvml-sys",
    "baracuda-nvrtc-sys",
    "baracuda-runtime",
    "baracuda-tensorrt-sys",
    "baracuda-cublas-sys",
    "baracuda-ozimmu-sys",                # NEW (Phase 44)
    "baracuda-transformer-engine-sys",    # NEW (Phase 55)
    "baracuda-cudf",
    "baracuda-cudnn",
    "baracuda-cufft",
    "baracuda-cufile",
    "baracuda-cupti",
    "baracuda-curand",
    "baracuda-cusolver",
    "baracuda-cusparse",
    "baracuda-cutensor",
    "baracuda-cutlass",
    "baracuda-cvcuda",
    "baracuda-cublas",                    # MOVED earlier (was at end)
    "baracuda-ozimmu",                    # NEW (Phase 44)
    "baracuda-transformer-engine",        # NEW (Phase 55)
    "baracuda-nccl",
    "baracuda-optim",                     # NEW (Phase 49)
    "baracuda-kernels",
    "baracuda-megatron",                  # NEW (Phase 57)
    "baracuda-npp",
    "baracuda-nvcomp",
    "baracuda-nvjitlink",
    "baracuda-nvjpeg",
    "baracuda-nvml",
    "baracuda-nvrtc",
    "baracuda-tensorrt",
    "baracuda"
)

$logFile = "target/publish_alpha60.log"
New-Item -ItemType Directory -Force -Path "target" | Out-Null
Set-Content -Path $logFile -Value "alpha.62 publish run started $(Get-Date)" -NoNewline:$false

$burstBudget = 28
$published = 0
$skipped = 0
$failed = @()

for ($i = 0; $i -lt $order.Count; $i++) {
    $crate = $order[$i]
    $idx = $i + 1
    Write-Host "[$idx/$($order.Count)] $crate ..."
    Add-Content -Path $logFile -Value "`n=== [$idx/$($order.Count)] $crate ==="

    $maxRetries = 3
    $exit = -1
    $skippedThis = $false
    for ($attempt = 1; $attempt -le $maxRetries; $attempt++) {
        $out = (cargo publish -p $crate --no-verify --allow-dirty 2>&1) -join "`n"
        Add-Content -Path $logFile -Value $out
        $exit = $LASTEXITCODE
        if ($exit -eq 0) { break }
        if ($out -match "already (uploaded|exists)") {
            $skippedThis = $true
            $exit = 0
            break
        }
        if ($out -match "Could not resolve host|spurious network error|connection (reset|refused)") {
            Write-Host "    [retry $attempt/$maxRetries] transient network - backing off 30s"
            Start-Sleep -Seconds 30
            continue
        }
        break
    }

    if ($exit -ne 0) {
        Write-Host "    FAILED - see $logFile"
        $failed += $crate
        continue
    }

    if ($skippedThis) {
        Write-Host "    skipped (already on crates.io)"
        $skipped++
    } else {
        Write-Host "    published"
        $published++
        if ($published -ge $burstBudget) {
            Write-Host "    sleeping 61s (post-burst pacing) ..."
            Start-Sleep -Seconds 61
        }
    }
}

Write-Host ""
Write-Host "=== Publish summary ==="
Write-Host "Published: $published"
Write-Host "Skipped:   $skipped"
Write-Host "Failed:    $($failed.Count)"
if ($failed.Count -gt 0) {
    Write-Host "Failed crates:"
    foreach ($f in $failed) { Write-Host "  - $f" }
}
