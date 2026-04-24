#!/usr/bin/env bash
# Resume the baracuda crates.io publish chain.
#
# --- Progress so far (0.0.1-alpha.2) ---
# Round 1 (2026-04-22): 5 foundation crates went out at 0.0.1-alpha.1 before
#                       hitting the "new crates per day" rate limit.
# Round 2 (2026-04-23): re-published the foundation at alpha.2 and got 5 more
#                       -sys crates through before hitting the daily cap again.
# Round 3 (2026-04-24): 5 more -sys crates through (cusparse, cusolver, cudnn,
#                       nccl, nvml). Blocked at baracuda-nvjpeg-sys
#                       (retry-after: 2026-04-24 12:57:22 GMT).
#
# Already on crates.io at 0.0.1-alpha.2 (15/46):
#   baracuda-types-derive, baracuda-types, baracuda-build, baracuda-core,
#   baracuda-cuda-sys, baracuda-nvrtc-sys, baracuda-nvjitlink-sys,
#   baracuda-cublas-sys, baracuda-curand-sys, baracuda-cufft-sys,
#   baracuda-cusparse-sys, baracuda-cusolver-sys, baracuda-cudnn-sys,
#   baracuda-nccl-sys, baracuda-nvml-sys
#
# crates.io's "new crate" quota is ~5 fresh names per ~12h per publisher.
# Re-publishing an existing name at a new version does NOT count toward
# that quota. Each ~12h window should unlock another 5 names until we finish.
#
# Run this script from the repo root whenever the retry window opens.
# It uses `set -e` so it stops at the first 429 — remove any
# successfully-published crates from the arrays below before the next
# run, or request a quota bump at:
#   https://github.com/rust-lang/crates.io/issues/new?template=rate_limit_increase.yml

set -e

# -sys crates still pending. baracuda-nvjpeg-sys is the first retry target.
SYS_CRATES_PENDING=(
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

# Safe wrappers — publish only once their matching -sys crate is live.
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

for c in "${SYS_CRATES_PENDING[@]}"; do publish "$c"; done
for c in "${SAFE_FOUNDATION[@]}";     do publish "$c"; done
for c in "${SAFE_CRATES[@]}";         do publish "$c"; done
publish "$UMBRELLA"

echo "=== all remaining baracuda crates published at 0.0.1-alpha.2 ==="
