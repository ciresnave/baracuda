# alpha.74 publish script.
#
# Prefix scan + JIT-envelope conform: baracuda-kernelgen gains Access::Scan
# (cumsum/prod/max/min, serial base + forward FP block-scan variant, on-device
# validated, adversarially reviewed), and BaracudaSynthesizer conforms to Fuel's
# frozen JIT envelope (fuel-kernel-seam 0.10.3: take_kernel on the trait, light
# Synthesized{entry_point} handle, envelope SynthArtifact, loadable-only
# ArtifactKind, per-request budget). The seam surface bump unblocks Fuel's JIT
# adoption path. See CHANGELOG.md (0.0.1-alpha.74) and the docs/fuel-* channel.
# baracuda-kernelgen + baracuda-kernels-bench are publish=false (not on the
# index), so the topo order is unchanged from alpha.71/.72/.73 (68 crates).
#
# Per-crate publish in dependency order (topo-sorted from `cargo metadata`,
# dev-deps excluded so the baracuda-types <-> baracuda-types-derive dev cycle
# doesn't deadlock). `cargo publish --workspace` does NOT work for this repo:
# its whole-graph resolve trips over that dev-dep cycle / prerelease req.
#
# --no-verify: skip the per-crate rebuild (kernels need the full CUDA env);
# single-crate publish does not enforce dev-dependency availability, which is
# what lets types-derive publish before types.
#
# Resilient: retries transient network + "dependency not yet on the index"
# (propagation lag / ordering) across multiple passes, sleeps on rate-limit,
# and treats already-uploaded versions as success. Re-running is safe.

$ErrorActionPreference = "Stop"

$order = @(
    "baracuda-build", "baracuda-seam", "baracuda-cutlass-sys", "baracuda-types-derive",
    "baracuda-forge", "baracuda-types", "baracuda-core", "baracuda-cutlass-kernels-sys",
    "baracuda-cuda-sys", "baracuda-cufile-sys", "baracuda-cupti-sys", "baracuda-cutensor-sys",
    "baracuda-cvcuda-sys", "baracuda-kernels-sys", "baracuda-nvcomp-sys", "baracuda-nvjitlink-sys",
    "baracuda-nvml-sys", "baracuda-nvrtc-sys", "baracuda-cublas-sys", "baracuda-cudf-sys",
    "baracuda-cudnn-sys", "baracuda-cufft-sys", "baracuda-cufile", "baracuda-cupti",
    "baracuda-curand-sys", "baracuda-cusolver-sys", "baracuda-cusparse-sys", "baracuda-cutensor",
    "baracuda-cuvs-sys", "baracuda-cvcuda", "baracuda-driver", "baracuda-flashinfer-sys",
    "baracuda-nccl-sys", "baracuda-npp-sys", "baracuda-nvcomp", "baracuda-nvimagecodec-sys",
    "baracuda-nvjitlink", "baracuda-nvjpeg-sys", "baracuda-nvml", "baracuda-nvrtc",
    "baracuda-nvshmem-sys", "baracuda-tensorrt-sys", "baracuda-transformer-engine-sys",
    "baracuda-cublas", "baracuda-cudf", "baracuda-cudnn", "baracuda-cufft",
    "baracuda-curand", "baracuda-cusolver", "baracuda-cusparse", "baracuda-cuvs",
    "baracuda-kernels-types", "baracuda-nccl", "baracuda-npp", "baracuda-nvimagecodec",
    "baracuda-nvjpeg", "baracuda-nvshmem", "baracuda-ozimmu-sys", "baracuda-runtime",
    "baracuda-tensorrt", "baracuda-transformer-engine", "baracuda", "baracuda-megatron",
    "baracuda-optim", "baracuda-ozimmu", "baracuda-cutlass", "baracuda-kernels",
    "baracuda-flashinfer"
)

$logFile = "target/publish_alpha74.log"
New-Item -ItemType Directory -Force -Path "target" | Out-Null
Set-Content -Path $logFile -Value "alpha.74 publish run started $(Get-Date)`n"

$burstBudget = 28        # crates.io burst window; pace after this many uploads
$published = 0
$skipped = 0
$pending = [System.Collections.Generic.List[string]]@($order)
$failed = @()

for ($pass = 1; $pass -le 8 -and $pending.Count -gt 0; $pass++) {
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
                Write-Host "    dep not on index yet -> defer to next pass"; break  # defer
            }
            else {
                Write-Host "    unexpected error (see log) -> defer"; break          # defer, inspect later
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
Write-Host "=== Publish summary ==="
Write-Host "Published: $published"
Write-Host "Skipped:   $skipped"
Write-Host "Failed:    $($failed.Count)"
if ($failed.Count -gt 0) { foreach ($f in $failed) { Write-Host "  - $f" } }
"SUMMARY published=$published skipped=$skipped failed=$($failed.Count)" | Add-Content -Path $logFile
