#!/usr/bin/env bash
# Resume the crates.io publish chain once the "new crates per day" rate
# limit resets (retry-after: 2026-04-22 18:37:22 GMT, roughly 24h after the
# initial attempt).
#
# Run from the repo root. Each publish auto-waits for index propagation
# before returning, so there's no need for explicit sleeps.
#
# If you hit the rate limit again mid-chain, note which crate failed and
# re-run starting from that crate onward.

set -e

# -sys crates (only depend on baracuda-core/types/cuda-sys, which are
# already on crates.io).
SYS_CRATES=(
    baracuda-nvrtc-sys
    baracuda-nvjitlink-sys
    baracuda-cublas-sys
    baracuda-curand-sys
    baracuda-cufft-sys
    baracuda-cusparse-sys
    baracuda-cusolver-sys
    baracuda-cudnn-sys
    baracuda-nccl-sys
    baracuda-nvml-sys
    baracuda-nvjpeg-sys
    baracuda-npp-sys
    baracuda-nvcomp-sys
    baracuda-cvcuda-sys
    baracuda-cufile-sys
    baracuda-cupti-sys
    baracuda-cutensor-sys
    baracuda-tensorrt-sys
    baracuda-cudf-sys
)

# Safe crates depend on their -sys and on the foundation. Driver +
# Runtime first, then everything else in parallel-ish order.
SAFE_FOUNDATION=(
    baracuda-driver
    baracuda-runtime
)

SAFE_CRATES=(
    baracuda-nvrtc
    baracuda-nvjitlink
    baracuda-cublas
    baracuda-curand
    baracuda-cufft
    baracuda-cusparse
    baracuda-cusolver
    baracuda-cudnn
    baracuda-nccl
    baracuda-nvml
    baracuda-nvjpeg
    baracuda-npp
    baracuda-nvcomp
    baracuda-cvcuda
    baracuda-cufile
    baracuda-cupti
    baracuda-cutensor
    baracuda-tensorrt
    baracuda-cudf
)

# Umbrella crate last — depends on (optionally) everything above.
UMBRELLA=baracuda

publish() {
    local crate=$1
    echo "=== publishing $crate ==="
    cargo publish -p "$crate"
}

for c in "${SYS_CRATES[@]}";        do publish "$c"; done
for c in "${SAFE_FOUNDATION[@]}";   do publish "$c"; done
for c in "${SAFE_CRATES[@]}";       do publish "$c"; done
publish "$UMBRELLA"

echo "=== all 37 remaining crates published ==="
