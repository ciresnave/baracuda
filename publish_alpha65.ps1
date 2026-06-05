# Phase 73 follow-up (alpha.65) publish script.
#
# alpha.65 is a Phase 73 follow-up release — perf-focused rather than new
# op families. The biggest wins are at LLM decode shapes:
#
#   - **FlashDecodingPlan (NEW op family)** — split-K parallel attention
#     decode for seq_q=1. Replaces the legacy FlashSdpaPlan at decode:
#     17-33× faster at MHA shapes (B=1, H=32, K∈{1024..8192}, D=128 f16)
#     and 4× faster again at GQA shapes (Llama-3-8B/70B/qwen2-14b)
#     via the new explicit `num_kv_heads` descriptor field.
#       commits: 7fbcd5d (kernel), 460a018 (warp-coop QK^T),
#                9327f6d (GQA-aware API), 89b74e1 (WMMA v2 reference)
#
#   - **fa2 is now a default cargo feature** on baracuda-kernels — closes
#     the ~108× perf gap vs PyTorch at the standard MHA shape (Hq=Hkv=32,
#     Q=K=2048, D=128) by auto-routing through Tri Dao FA2's tensor-core
#     MMA path. Standard MHA shape is now **50% faster than PyTorch**.
#     Opt out with `default-features = false, features = ["sm80"]`.
#       commit: 833f862
#
#   - **FlashSdpaPlan GQA-broadcast routing** — long-standing ROADMAP
#     entry closed. `FlashSdpaPlan` now accepts stride-0 K/V (full MQA
#     broadcast) and auto-routes to FlashSdpaSm89Plan. End-user MQA
#     inference no longer requires manual plan selection.
#
#   - **ConcatPlan contig fast path** — 13× speedup on the KV-cache
#     decode shape (4.4ms → 338μs at the BH=32 / K=2047+1 / D=128
#     bench case). Pre-1.0 release blocker removed.
#       commit: 9d6d825
#
#   - **reduce_axis block-per-row rewrite** — 2.6-15.4× speedup across
#     the reductions bench sweep. baracuda now BEATS PyTorch at large
#     reduce shapes (R4096_H4096 f32: baracuda 297μs vs PyTorch 365μs).
#       commit: e8e5257
#
#   - **WMMA tensor-core kernel for decode (negative result, shipped as
#     reference)** — two iterations written, smoke-tested, and benched.
#     Loses to SIMT-GQA by 1.24-2.87× at every benchmarked GQA shape
#     because decode is bandwidth-bound + M-tile underutilized at
#     single-batch. Dispatch disabled; kernel + helpers stay in tree
#     as documented reference for future multi-batch contig decode.
#       commits: 9327f6d (v1), 89b74e1 (v2 — 1.13-1.41× better than v1)
#
# No NEW crates this release. Crate order matches alpha.64.
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
    "baracuda-kernels-sys",
    "baracuda-types-derive",
    "baracuda-cutlass-kernels-sys",
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
    "baracuda-cutlass",                     # before kernels (kernels depends on cutlass)
    "baracuda-cuvs",
    "baracuda-cvcuda",
    "baracuda-ozimmu",
    "baracuda-transformer-engine",
    "baracuda-nccl",
    "baracuda-nvshmem",
    "baracuda-optim",
    "baracuda-kernels",                     # before flashinfer + megatron
    "baracuda-flashinfer",
    "baracuda-megatron",
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

$logFile = "target/publish_alpha65.log"
New-Item -ItemType Directory -Force -Path "target" | Out-Null
Set-Content -Path $logFile -Value "alpha.65 publish run started $(Get-Date)" -NoNewline:$false

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
    Write-Host "Re-run with publish_alpha65_retry.ps1 to retry failures only."
}
