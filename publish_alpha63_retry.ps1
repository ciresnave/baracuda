# Phase 63 (alpha.63) retry script — handles dev-dep cycles by stripping
# `baracuda-*` lines from `[dev-dependencies]` sections.
#
# **Retry order in actual topological dependency order**, per
# feedback_publish_retry_order.md.
#
# **Phase 63 update — replaces the alpha.62 dep-bridge sleep**: alpha.62
# tried `Start-Sleep -Seconds 30` before publishing baracuda-cutlass /
# baracuda-kernels (the two crates that depend on a freshly-published
# direct dep in the previous slot) and STILL had cutlass + kernels fail.
# The root cause is cargo's LOCAL index cache, not crates.io itself
# (which propagates within ~1-2s of upload). The fix that works:
# `cargo update -p <just-published-dep>` between dep-bridge publishes
# forces a per-crate index re-fetch — near-instant and reliable,
# replaces the unreliable 30s sleep. alpha.62 round-3 used this trick
# manually and worked first-try; now baked into the retry script.

$ErrorActionPreference = "Stop"

$failed = @(
    # Layer 1: sys-crates that depend only on cutlass-kernels-sys (now live).
    "baracuda-kernels-sys",
    # Layer 2: nvrtc is independent of driver/runtime.
    "baracuda-nvrtc",
    # Layer 3: driver (depends on cuda-sys).
    "baracuda-driver",
    # Layer 4: runtime (depends on driver).
    "baracuda-runtime",
    # Layer 5: kernels-types (depends on driver). MUST come after driver.
    "baracuda-kernels-types",
    # Layer 6: CUDA-library safe wrappers (depend on their -sys + runtime).
    "baracuda-cudnn",
    "baracuda-cufft",
    "baracuda-cufile",
    "baracuda-curand",
    "baracuda-cusolver",
    "baracuda-cusparse",
    "baracuda-cutensor",
    "baracuda-cvcuda",
    "baracuda-cublas",
    # Layer 7: cutlass (depends on cublas). MUST come after cublas.
    "baracuda-cutlass",
    # Layer 8: optional-backend safe wrappers (depend on their -sys + cublas).
    "baracuda-ozimmu",
    "baracuda-transformer-engine",
    "baracuda-nccl",
    # Layer 9: optim (depends on kernels-types + cublas).
    "baracuda-optim",
    # Layer 10: kernels (depends on cutlass + cublas + many CUDA libs). MUST come after cutlass.
    "baracuda-kernels",
    # Layer 11: megatron (depends on kernels + cublas + nccl).
    "baracuda-megatron",
    # Layer 12: high-level safe wrappers (depend on kernels).
    "baracuda-npp",
    "baracuda-nvcomp",
    "baracuda-nvjitlink",
    "baracuda-nvjpeg",
    # Layer 13: meta-crate (depends on everything).
    "baracuda"
)

# Strip baracuda-* lines from [dev-dependencies] section, write to dst.
function Strip-DevDeps {
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

$logFile = "target/publish_alpha61_retry.log"
Add-Content -Path $logFile -Value "`n=== alpha.63 publish retry started $(Get-Date) ==="

$published = 0
$alreadyOk = 0
$stillFailed = @()

for ($i = 0; $i -lt $failed.Count; $i++) {
    $crate = $failed[$i]
    $idx = $i + 1
    $cargoToml = "crates/$crate/Cargo.toml"
    $backup = "$cargoToml.bak"

    Write-Host "[$idx/$($failed.Count)] $crate ..."
    Add-Content -Path $logFile -Value "`n--- [$idx/$($failed.Count)] $crate ---"

    # Phase 63 update: alpha.63 confirmed the actual root cause of
    # baracuda-cutlass / baracuda-kernels still-failing after their
    # direct dep just published is NOT crates.io propagation (which is
    # ~1-2s) but cargo's LOCAL index cache. The previous publish's
    # "waiting for available" loop checks a specific endpoint; the next
    # crate's `cargo update` reads the local cache which is stale.
    # `cargo update -p <dep>` forces a per-crate refresh — near-instant
    # and reliable, replacing the alpha.63 30s sleep. See
    # feedback_publish_retry_order memory.
    if ($crate -eq "baracuda-cutlass") {
        Write-Host "    refresh local index for baracuda-cublas"
        cargo update -p baracuda-cublas 2>&1 | Out-Null
    }
    if ($crate -eq "baracuda-kernels") {
        Write-Host "    refresh local index for baracuda-cutlass"
        cargo update -p baracuda-cutlass 2>&1 | Out-Null
    }

    # First try without modifications.
    $out = (cargo publish -p $crate --no-verify --allow-dirty 2>&1) -join "`n"
    $exit = $LASTEXITCODE
    Add-Content -Path $logFile -Value $out

    if ($out -match "already (uploaded|exists)") {
        Write-Host "    already published"
        $alreadyOk++
        continue
    }

    if ($exit -ne 0) {
        # Check for dev-dep cycle / propagation error.
        $cycleHit = $out -match "no matching package named ``baracuda-" -or `
                    $out -match "failed to select a version for the requirement ``baracuda-"
        if ($cycleHit) {
            Write-Host "    dev-dep cycle detected; stripping baracuda-* from [dev-dependencies]"
            Copy-Item $cargoToml $backup -Force
            try {
                Strip-DevDeps $cargoToml "$cargoToml.tmp"
                Move-Item "$cargoToml.tmp" $cargoToml -Force
                $out = (cargo publish -p $crate --no-verify --allow-dirty 2>&1) -join "`n"
                $exit = $LASTEXITCODE
                Add-Content -Path $logFile -Value "`n[after strip]`n$out"
            } finally {
                Move-Item $backup $cargoToml -Force
                Write-Host "    restored $cargoToml"
            }
            if ($out -match "already (uploaded|exists)") {
                Write-Host "    already published (race)"
                $alreadyOk++
                continue
            }
        }
    }

    if ($exit -eq 0) {
        Write-Host "    published"
        $published++
        # Post-burst pacing - 61s between successful publishes.
        if ($i -lt $failed.Count - 1) {
            Write-Host "    sleeping 61s ..."
            Start-Sleep -Seconds 61
        }
    } else {
        Write-Host "    STILL FAILED - see $logFile"
        $stillFailed += $crate
    }
}

Write-Host ""
Write-Host "=== Retry summary ==="
Write-Host "Published: $published"
Write-Host "Already on crates.io: $alreadyOk"
Write-Host "Still failed: $($stillFailed.Count)"
if ($stillFailed.Count -gt 0) {
    Write-Host "Still-failed crates:"
    foreach ($f in $stillFailed) { Write-Host "  - $f" }
}
