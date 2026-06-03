# Phase 64 (alpha.64) publish script.
#
# alpha.64 is the largest single release in the alpha series. It rolls up:
#   - Phase 64 in-place aliasing docs extended (Cast/Where/Triu/Tril/Fill/
#     Activation BW marked safe; Flip/Roll/Permute/RoPE marked NOT safe)
#   - Phase 65a-d SMEM-staged normalizer in-place + reusable SMEM helpers
#   - Phase 65d-ext f64 in-place dispatch on RMS/Layer/Softmax/LogSoftmax
#   - Phase 66 FlashInfer Tier-2 closure (paged prefill, spec-decode, FP8 KV,
#     ragged prefill); 20/20 GPU smoke tests
#   - Phase 67a-f reusable kernel-helper headers
#   - Phase 68 TensorRT vtable-dispatch C++ shim
#   - Phase 69 NVSHMEM host-side wrapper pair (NEW)
#   - Phase 70 nvImageCodec wrapper pair (NEW)
#   - Phase 71 RAPIDS cuVS wrapper pair (NEW)
#   - `_can_implement` companion contract: full ~2030-symbol fanout, every
#     `_run` FFI symbol now has a host-side validator
#   - Tier-2 docs sweep: ~6900 one-line docs across all -sys crates;
#     workspace lints promoted from warn → deny on missing_docs
#   - 95 clippy warnings closed; `cargo fix` Rust 2024 unsafe migration
#
# NEW crates this release (Phase 66/69/70/71):
#   - baracuda-flashinfer{,-sys}
#   - baracuda-nvshmem{,-sys}
#   - baracuda-nvimagecodec{,-sys}
#   - baracuda-cuvs{,-sys}
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
    "baracuda-cuvs-sys",                    # NEW (Phase 71)
    "baracuda-cvcuda-sys",
    "baracuda-flashinfer-sys",              # NEW (Phase 66)
    "baracuda-nccl-sys",
    "baracuda-npp-sys",
    "baracuda-nvcomp-sys",
    "baracuda-nvimagecodec-sys",            # NEW (Phase 70)
    "baracuda-nvjitlink-sys",
    "baracuda-nvjpeg-sys",
    "baracuda-nvml-sys",
    "baracuda-nvrtc-sys",
    "baracuda-nvshmem-sys",                 # NEW (Phase 69)
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
    # Inter-wrapper deps observed in alpha.64 publish run (topo MUST honor):
    #   baracuda-cutlass     -> baracuda-cublas
    #   baracuda-kernels     -> baracuda-cutlass (+ many other sibling safe wrappers)
    #   baracuda-flashinfer  -> baracuda-kernels
    #   baracuda-megatron    -> baracuda-kernels
    # Wrappers below are ordered so each precedes its consumer.
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
    "baracuda-cuvs",                        # NEW (Phase 71)
    "baracuda-cvcuda",
    "baracuda-ozimmu",
    "baracuda-transformer-engine",
    "baracuda-nccl",
    "baracuda-nvshmem",                     # NEW (Phase 69; depends on -sys)
    "baracuda-optim",
    "baracuda-kernels",                     # before flashinfer + megatron
    "baracuda-flashinfer",                  # NEW (Phase 66; depends on baracuda-kernels)
    "baracuda-megatron",
    "baracuda-npp",
    "baracuda-nvcomp",
    "baracuda-nvimagecodec",                # NEW (Phase 70; depends on -sys + driver)
    "baracuda-nvjitlink",
    "baracuda-nvjpeg",
    "baracuda-nvml",
    "baracuda-nvrtc",
    "baracuda-tensorrt",
    "baracuda"
)

$logFile = "target/publish_alpha64.log"
New-Item -ItemType Directory -Force -Path "target" | Out-Null
Set-Content -Path $logFile -Value "alpha.64 publish run started $(Get-Date)" -NoNewline:$false

# Strip baracuda-* lines from [dev-dependencies] section, write to dst.
# Used to work around the dev-dep cycle (e.g. baracuda-types <-> baracuda-types-
# derive) where cargo publish's manifest-resolution check tries to find the
# sibling on crates.io before it's been uploaded. Restored after each crate.
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
        # Dev-dep cycle: a [dev-dependencies] line references a sibling
        # baracuda-* crate that hasn't been published yet at this version.
        # Strip those lines + retry. Restored at end-of-iteration.
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
            # Non-baracuda dep just-published but index not refreshed yet.
            Write-Host "    [retry $attempt/$maxRetries] dep not yet visible in index - refreshing + backing off 30s"
            cargo update --workspace --aggressive 2>&1 | Out-Null
            Start-Sleep -Seconds 30
            continue
        }
        break
    }

    # Restore the original Cargo.toml if we stripped.
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
    Write-Host "Re-run with publish_alpha64_retry.ps1 to retry failures only."
}
