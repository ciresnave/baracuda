# Retry pass for the 3 crates that lost to a topo-order cascade during the main
# alpha.64 publish run:
#   - baracuda-cutlass     (needs baracuda-cublas published; happened at slot 50)
#   - baracuda-kernels     (needs baracuda-cutlass)
#   - baracuda-flashinfer  (needs baracuda-kernels)
#
# Each is published in order, with a 61s sleep between (post-burst pacing —
# this run is well past the 28-publish burst budget).

$ErrorActionPreference = "Stop"

$order = @(
    "baracuda-cutlass",
    "baracuda-kernels",
    "baracuda-flashinfer"
)

$logFile = "target/publish_alpha64_retry.log"
New-Item -ItemType Directory -Force -Path "target" | Out-Null
Set-Content -Path $logFile -Value "alpha.64 retry run started $(Get-Date)" -NoNewline:$false

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
        # Always pace 61s (we're past the burst budget in this retry run).
        if ($i -lt $order.Count - 1) {
            Write-Host "    sleeping 61s (post-burst pacing) ..."
            Start-Sleep -Seconds 61
        }
    }
}

Write-Host ""
Write-Host "=== Retry summary ==="
Write-Host "Published: $published"
Write-Host "Skipped:   $skipped"
Write-Host "Failed:    $($failed.Count)"
if ($failed.Count -gt 0) {
    Write-Host "Failed crates:"
    foreach ($f in $failed) { Write-Host "  - $f" }
}
