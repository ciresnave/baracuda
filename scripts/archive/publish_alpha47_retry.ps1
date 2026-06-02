# Phase 32 (alpha.47) retry script — handles dev-dep cycles by stripping
# `baracuda-*` lines from `[dev-dependencies]` sections.
# Note: baracuda-kernels-sys is FIRST in the retry list because Phase 32
# added a normal dep on baracuda-cutlass-kernels-sys, which means
# kernels-sys was out of order during the main run — re-publish it now
# that cutlass-kernels-sys is live.

$ErrorActionPreference = "Stop"

$failed = @(
    "baracuda-kernels-sys",
    "baracuda-kernels-types",
    "baracuda-nvrtc",
    "baracuda-driver",
    "baracuda-runtime",
    "baracuda-cudnn",
    "baracuda-cufft",
    "baracuda-cufile",
    "baracuda-curand",
    "baracuda-cusolver",
    "baracuda-cusparse",
    "baracuda-cutensor",
    "baracuda-cutlass",
    "baracuda-cvcuda",
    "baracuda-kernels",
    "baracuda-nccl",
    "baracuda-npp",
    "baracuda-nvcomp",
    "baracuda-nvjitlink",
    "baracuda-nvjpeg",
    "baracuda-cublas",
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

$logFile = "target/publish_alpha47_retry.log"
Add-Content -Path $logFile -Value "`n=== Phase 22 publish retry started $(Get-Date) ==="

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
        # Check for dev-dep cycle error.
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
        # Post-burst pacing — 61s between successful publishes.
        if ($i -lt $failed.Count - 1) {
            Write-Host "    sleeping 61s ..."
            Start-Sleep -Seconds 61
        }
    } else {
        Write-Host "    STILL FAILED — see $logFile"
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
