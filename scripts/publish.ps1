# Version- and crate-agnostic crates.io publish script.
#
# Replaces the per-version `publish_alphaXX.ps1` proliferation (one script was
# copied per release, differing only in the version string + log name). This one
# reads the publishable crate set AND the lockstep version from `cargo metadata`,
# so it adapts automatically to any version bump and any crate-set change: a new
# workspace crate is picked up, and a `publish = false` crate
# (baracuda-kernels-bench / xtask / baracuda-examples) is excluded — no manual
# list to maintain.
#
# Why per-crate (not `cargo publish --workspace`): the whole-graph resolve trips
# the baracuda-types <-> baracuda-types-derive DEV-dep cycle / prerelease req.
# `--no-verify` skips the per-crate rebuild (kernels need the full CUDA env);
# single-crate publish doesn't enforce dev-dep availability, which lets
# types-derive publish before types.
#
# Why no explicit topo order: the loop is RESILIENT — it defers a crate whose
# dependency isn't on the index yet and retries it on a later pass, so the
# dependency ordering resolves itself across passes. It also retries transient
# network errors, sleeps on rate-limit, and treats an already-uploaded version as
# success. Re-running is safe (idempotent).
#
# Usage:
#   pwsh scripts/publish.ps1            # publish the current workspace version
#   pwsh scripts/publish.ps1 -DryRun    # list what would publish; upload nothing

[CmdletBinding()]
param([switch]$DryRun)
$ErrorActionPreference = "Stop"

Write-Host "Reading workspace metadata..."
$meta = cargo metadata --no-deps --format-version 1 | ConvertFrom-Json

# Publishable: `publish` is null (any registry) or a non-empty allowlist.
# Excluded:    `publish = false` -> an empty array in cargo metadata.
$pub = @($meta.packages | Where-Object {
    $null -eq $_.publish -or @($_.publish).Count -gt 0
})
$excluded = @($meta.packages | Where-Object {
    $null -ne $_.publish -and @($_.publish).Count -eq 0
} | ForEach-Object { $_.name })

if ($pub.Count -eq 0) { throw "no publishable crates found in workspace metadata" }
$order = @($pub | ForEach-Object { $_.name })
$version = $pub[0].version

# Sanity: every publishable crate should share the one lockstep version.
$distinct = @($pub | ForEach-Object { $_.version } | Sort-Object -Unique)
if ($distinct.Count -ne 1) {
    Write-Warning "publishable crates are NOT on one version: $($distinct -join ', ')"
}

Write-Host "Version:  $version"
Write-Host "Publish:  $($order.Count) crate(s)"
Write-Host "Excluded: $($excluded -join ', ') (publish=false)"

if ($DryRun) {
    Write-Host "`n-DryRun: would publish (order resolves via the retry loop):"
    $order | ForEach-Object { Write-Host "  $_" }
    return
}

$logFile = "target/publish_$($version -replace '[^\w.-]','_').log"
New-Item -ItemType Directory -Force -Path "target" | Out-Null
Set-Content -Path $logFile -Value "publish run for $version started $(Get-Date)`n"

$burstBudget = 28   # crates.io burst window; pace after this many uploads
$published = 0
$skipped = 0
$pending = [System.Collections.Generic.List[string]]@($order)

for ($pass = 1; $pass -le 12 -and $pending.Count -gt 0; $pass++) {
    Write-Host "===== pass $pass : $($pending.Count) crate(s) pending ====="
    Add-Content -Path $logFile -Value "`n===== PASS $pass ($($pending.Count) pending) ====="
    $deferred = [System.Collections.Generic.List[string]]@()

    foreach ($crate in $pending) {
        Write-Host "  $crate ..."
        Add-Content -Path $logFile -Value "`n--- $crate ---"

        $done = $false
        for ($attempt = 1; $attempt -le 4 -and -not $done; $attempt++) {
            $out = (cargo publish -p $crate --no-verify --allow-dirty 2>&1) -join "`n"
            $exit = $LASTEXITCODE
            Add-Content -Path $logFile -Value $out

            if ($exit -eq 0) {
                Write-Host "    published"; $published++; $done = $true
                if ($published -ge $burstBudget) { Write-Host "    (post-burst pacing 61s)"; Start-Sleep -Seconds 61 }
            }
            elseif ($out -match "already (uploaded|exists)|crate version .* is already") {
                Write-Host "    skipped (already on crates.io)"; $skipped++; $done = $true
            }
            elseif ($out -match "rate limit|429|Too Many Requests|published too many") {
                $wait = 65
                if ($out -match "after (\d+) seconds") { $wait = [int]$Matches[1] + 5 }
                Write-Host "    rate-limited; sleeping ${wait}s"; Start-Sleep -Seconds $wait
            }
            elseif ($out -match "Could not resolve host|spurious network|connection (reset|refused)|timed out|failed to get") {
                Write-Host "    [net retry $attempt] backing off 30s"; Start-Sleep -Seconds 30
            }
            elseif ($out -match "failed to select a version|no matching package|cannot find") {
                Write-Host "    dep not on index yet -> defer to next pass"; break
            }
            else {
                Write-Host "    unexpected error (see log) -> defer"; break
            }
        }

        if (-not $done) { $deferred.Add($crate) }
    }

    if ($deferred.Count -eq $pending.Count) {
        Write-Host "no progress this pass; sleeping 30s for index propagation before next pass"
        Start-Sleep -Seconds 30
    }
    $pending = $deferred
}

$failed = @($pending)
Write-Host ""
Write-Host "=== Publish summary ($version) ==="
Write-Host "Published: $published"
Write-Host "Skipped:   $skipped"
Write-Host "Failed:    $($failed.Count)"
if ($failed.Count -gt 0) { foreach ($f in $failed) { Write-Host "  - $f" } }
"SUMMARY version=$version published=$published skipped=$skipped failed=$($failed.Count)" | Add-Content -Path $logFile
