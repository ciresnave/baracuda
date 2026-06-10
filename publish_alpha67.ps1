# alpha.67 publish script.
#
# alpha.67 ships Phase 74 — the Fuel dense-FP-GEMM + reduce-to facade
# closure (Fuel ask 2026-06-10; consumer reply in
# docs/fuel-reply-fp-gemm-reduce-to-2026-06-10.md):
#
#   - NEW gemm_dense_cublas_facade in baracuda-kernels-sys: 12 flat C
#     symbols baracuda_kernels_gemm_dense_{f32,f64,f16,bf16}_{run,
#     can_implement,workspace_size} — cuBLAS-backed, RRR/RCR/CRR layout
#     tags, flexible leading dims, strided-batch folded into the base
#     symbol, lock-free context-keyed handle pool.
#   - NEW DenseGemmPlan<T> + ReduceToPlan<T, N> + UnaryKind::Step plan
#     facades; gelu flavor-disambiguation docs.
#   - Regression: 2180/0 across 513 baracuda-kernels test binaries on
#     RTX 4070 (sm89 + cudnn).
#
# No NEW crates and no new inter-crate deps this release (the facade
# uses kernels-sys's own cublas extern declarations — no Cargo.toml
# changes). Crate order matches alpha.66 (already topo-correct:
# cutlass-kernels-sys before kernels-sys; cublas/ozimmu before cutlass;
# megatron before kernels).
#
# Bursts the first 28 publishes (crates.io's burst window), then sleeps 61s
# between each subsequent publish. See feedback_publish_pacing.md.
# Skipped (already-uploaded) crates do NOT consume burst credits.

$ErrorActionPreference = "Stop"

$order = @(
    # --- Foundation (no inter-crate deps) ---
    "baracuda-build",
    "baracuda-cutlass-sys",
    "baracuda-forge",
    # NOTE: baracuda-cutlass-kernels-sys MUST precede baracuda-kernels-sys —
    # kernels-sys gained a normal dep on cutlass-kernels-sys (Phase 24).
    "baracuda-types-derive",
    "baracuda-cutlass-kernels-sys",
    "baracuda-kernels-sys",
    "baracuda-types",
    "baracuda-core",

    # --- Raw FFI -sys crates (depend only on baracuda-core / baracuda-build) ---
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

    # --- Driver + Runtime (depend on baracuda-cuda-sys) ---
    "baracuda-driver",
    "baracuda-runtime",

    # --- kernels-types + bench have no other-sibling deps ---
    "baracuda-kernels-types",

    # --- Safe wrappers (each depends on its -sys + baracuda-driver) ---
    # Inter-wrapper deps (topo MUST honor):
    #   baracuda-cutlass     -> baracuda-cublas
    #   baracuda-kernels     -> baracuda-cutlass (+ many other sibling safe wrappers)
    #   baracuda-flashinfer  -> baracuda-kernels
    #   baracuda-megatron    -> baracuda-kernels
    "baracuda-cudf",
    "baracuda-cudnn",
    "baracuda-cufft",
    "baracuda-cufile",
    "baracuda-cupti",
    "baracuda-curand",
    "baracuda-cusolver",
    "baracuda-cusparse",
    "baracuda-cutensor",
    "baracuda-cublas",                      # before cutlass (cutlass depends on cublas)
    "baracuda-ozimmu",                      # before cutlass (cutlass `ozimmu` feature depends on it)
    "baracuda-cutlass",                     # before kernels (kernels depends on cutlass)
    "baracuda-cuvs",
    "baracuda-cvcuda",
    "baracuda-transformer-engine",
    "baracuda-nccl",
    "baracuda-nvshmem",
    "baracuda-optim",
    # baracuda-megatron MUST precede baracuda-kernels: kernels has an OPTIONAL
    # dep on megatron via the `megatron_tp` feature, and cargo resolves optional
    # deps against the index at publish time. The alpha.65 order had megatron
    # AFTER kernels — it only "worked" because the retry republished kernels
    # once megatron had landed. megatron's own deps (cublas, nccl) already
    # precede this point, so moving it up is safe.
    "baracuda-megatron",
    "baracuda-kernels",                     # before flashinfer
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

$logFile = "target/publish_alpha67.log"
New-Item -ItemType Directory -Force -Path "target" | Out-Null
Set-Content -Path $logFile -Value "alpha.67 publish run started $(Get-Date)" -NoNewline:$false

# Strip baracuda-* lines from [dev-dependencies] section, write to dst.
function Strip-BaracudaDevDeps {
    param([string]$Src, [string]$Dst)
    $lines = Get-Content $Src
    $out = New-Object System.Collections.Generic.List[string]
    $inDev = $false
    foreach ($line in $lines) {
        if ($line -match '^\[dev-dependencies(\..*)?\]') { $inDev = $true; $out.Add($line); continue }
        if ($line -match '^\[') { $inDev = $false; $out.Add($line); continue }
        if ($inDev -and $line -match '^baracuda-') { continue }
        $out.Add($line)
    }
    Set-Content -Path $Dst -Value $out
}

$burstBudget = 28
$published = 0
$skipped = 0
$failed = @()

for ($i = 0; $i -lt $order.Count; $i++) {
    $crate = $order[$i]
    $idx = $i + 1
    Write-Host "[$idx/$($order.Count)] $crate ..."
    Add-Content -Path $logFile -Value "`n=== [$idx/$($order.Count)] $crate ==="

    $cargoToml = "crates/$crate/Cargo.toml"
    $backupToml = "$cargoToml.bak"

    $maxRetries = 4
    $exit = -1
    $skippedThis = $false
    $stripped = $false
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
        $devCycle = $out -match "failed to select a version for the requirement ``baracuda-" -or `
                     $out -match "no matching package named ``baracuda-"
        if ($devCycle -and -not $stripped) {
            Write-Host "    [retry $attempt/$maxRetries] dev-dep cycle - stripping baracuda-* from [dev-dependencies]"
            Copy-Item $cargoToml $backupToml -Force
            Strip-BaracudaDevDeps $cargoToml "$cargoToml.tmp"
            Move-Item "$cargoToml.tmp" $cargoToml -Force
            $stripped = $true
            continue
        }
        if ($out -match "no matching package named .* found") {
            Write-Host "    [retry $attempt/$maxRetries] dep not yet visible in index - refreshing + backing off 30s"
            cargo update --workspace --aggressive 2>&1 | Out-Null
            Start-Sleep -Seconds 30
            continue
        }
        break
    }

    if ($stripped -and (Test-Path $backupToml)) {
        Move-Item $backupToml $cargoToml -Force
        Write-Host "    restored $cargoToml"
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
    Write-Host ""
    Write-Host "Re-run publish_alpha67_retry.ps1 (or re-run this script — it skips already-published crates) to retry failures only."
}
