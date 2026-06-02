# Phase 62 (alpha.62) retry script — handles dev-dep cycles by stripping
# `baracuda-*` lines from `[dev-dependencies]` sections.
#
# **Retry order in actual topological dependency order**, per
# feedback_publish_retry_order.md.
#
# **Phase 62 update**: alpha.61 confirmed dep-order alone is necessary
# but NOT sufficient — baracuda-cutlass still failed after baracuda-cublas
# was just published because crates.io's index propagation lagged behind
# cargo publish's own "Published" signal. To bridge that gap we add an
# extra 30s pre-attempt sleep at the dep-bridge slots (cutlass after
# cublas; kernels after cutlass) on top of the base 61s post-publish
# pacing.

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
Add-Content -Path $logFile -Value "`n=== alpha.62 publish retry started $(Get-Date) ==="

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

    # Phase 62: extra index-propagation buffer for crates whose direct
    # dep was just published in the previous slot. crates.io's index
    # propagation lags cargo publish's "Published" signal by ~5-15s;
    # the base 61s post-publish pacing isn't always enough.
    if ($crate -eq "baracuda-cutlass" -or $crate -eq "baracuda-kernels") {
        Write-Host "    dep-bridge sleep 30s (index propagation buffer)"
        Start-Sleep -Seconds 30
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
