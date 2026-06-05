# alpha.65 publish RETRY — the 5 crates that failed the main run due to
# two topo-order bugs in publish_alpha65.ps1 (both now harmless because
# the missing deps are published):
#
#   - baracuda-kernels-sys depends on baracuda-cutlass-kernels-sys, but
#     the main script published kernels-sys (#4) BEFORE cutlass-kernels-sys
#     (#6). On retry the dep is already in the index.
#   - baracuda-cutlass depends on baracuda-ozimmu (optional `ozimmu`
#     feature), but the main script published cutlass before ozimmu.
#     Also already in the index now.
#
# The other 3 (flashinfer-sys, kernels, flashinfer) were pure cascade
# failures off kernels-sys / cutlass.
#
# Published in dependency tiers with an index-refresh wait between tiers
# so each tier's deps are visible before the next tier resolves them.

$ErrorActionPreference = "Stop"
$logFile = "target/publish_alpha65_retry.log"
Set-Content -Path $logFile -Value "alpha.65 RETRY run started $(Get-Date)" -NoNewline:$false

# Tier A: deps (cutlass-kernels-sys, ozimmu, cublas) already long-published.
# Tier B: depend on tier A's kernels-sys / cutlass.
# Tier C: depends on tier B's kernels.
$tiers = @(
    @("baracuda-kernels-sys", "baracuda-cutlass"),
    @("baracuda-flashinfer-sys", "baracuda-kernels"),
    @("baracuda-flashinfer")
)

$published = 0
$skipped = 0
$failed = @()

for ($t = 0; $t -lt $tiers.Count; $t++) {
    $tier = $tiers[$t]
    foreach ($crate in $tier) {
        Write-Host "[tier $($t+1)] $crate ..."
        Add-Content -Path $logFile -Value "`n=== [tier $($t+1)] $crate ==="
        $exit = -1
        for ($attempt = 1; $attempt -le 5; $attempt++) {
            $out = (cargo publish -p $crate --no-verify --allow-dirty 2>&1) -join "`n"
            Add-Content -Path $logFile -Value $out
            $exit = $LASTEXITCODE
            if ($exit -eq 0) { break }
            if ($out -match "already (uploaded|exists)") { $exit = 0; $skipped++; break }
            # Dep not yet visible in the sparse index — refresh + back off.
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
        } elseif ($skipped -eq 0 -or $out -notmatch "already") {
            Write-Host "    published"
            $published++
        }
    }
    # Index-refresh pause between tiers so the next tier sees this tier.
    if ($t -lt $tiers.Count - 1) {
        Write-Host "  ... waiting 45s for index refresh before next tier ..."
        Start-Sleep -Seconds 45
    }
}

Write-Host ""
Write-Host "=== Retry summary ==="
Write-Host "Published: $published"
Write-Host "Failed:    $($failed.Count)"
foreach ($f in $failed) { Write-Host "  - $f" }
