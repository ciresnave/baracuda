#!/usr/bin/env bash
# Resume the crates.io publish chain once the "new crates per day" rate
# limit resets (retry-after: 2026-04-22 18:37:22 GMT, roughly 24h after
# the initial attempt).
#
# Since this session added new API (ValidAsZeroBits / HostSlice on
# baracuda-types, DevicePtr traits on baracuda-driver, cuBLASLt
# Activation enum, CudnnDataType trait, NVRTC CompileOptions struct) the
# workspace is now at 0.0.1-alpha.2. The five already-published crates
# at 0.0.1-alpha.1 need to be re-published at the new version before
# their dependents can build against it.
#
# Run from the repo root. Each publish auto-waits for index propagation
# before returning, so there's no need for explicit sleeps.
#
# If you hit the rate limit mid-chain, note which crate failed and
# re-run starting from that crate onward by editing the arrays below.

set -e

# Foundation crates that were already published at alpha.1 — re-publish at
# alpha.2 so dependents resolve cleanly.
FOUNDATION=(
    baracuda-types-derive
    baracuda-types
    baracuda-build
    baracuda-core
    baracuda-cuda-sys
)

# -sys crates (only depend on baracuda-core/types/cuda-sys).
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

# Safe-API core.
SAFE_FOUNDATION=(
    baracuda-driver
    baracuda-runtime
)

# Safe wrappers for each library.
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

# Umbrella crate last — depends (optionally) on everything above.
UMBRELLA=baracuda

publish() {
    local crate=$1
    echo "=== publishing $crate ==="
    cargo publish -p "$crate"
}

for c in "${FOUNDATION[@]}";      do publish "$c"; done
for c in "${SYS_CRATES[@]}";       do publish "$c"; done
for c in "${SAFE_FOUNDATION[@]}";  do publish "$c"; done
for c in "${SAFE_CRATES[@]}";      do publish "$c"; done
publish "$UMBRELLA"

echo "=== all 42 crates published at 0.0.1-alpha.2 ==="
