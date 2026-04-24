#!/usr/bin/env bash
# Resume the baracuda crates.io publish chain.
#
# --- Progress so far (0.0.1-alpha.2) ---
# Round 1 (2026-04-22): foundation-5 at alpha.1.
# Round 2 (2026-04-23): alpha.2 re-pubs of the foundation + 5 more -sys.
# Round 3 (2026-04-24 ~02h): cusparse/cusolver/cudnn/nccl/nvml-sys.
# Round 4 (2026-04-24 ~13h): nvjpeg/npp/nvcomp/cvcuda/cufile-sys.
#                            Blocked at baracuda-cupti-sys
#                            (retry-after: 2026-04-24 13:47:22 GMT — <1h away).
#
# Already on crates.io at 0.0.1-alpha.2 (20/46):
#   baracuda-types-derive, baracuda-types, baracuda-build, baracuda-core,
#   baracuda-cuda-sys, baracuda-nvrtc-sys, baracuda-nvjitlink-sys,
#   baracuda-cublas-sys, baracuda-curand-sys, baracuda-cufft-sys,
#   baracuda-cusparse-sys, baracuda-cusolver-sys, baracuda-cudnn-sys,
#   baracuda-nccl-sys, baracuda-nvml-sys, baracuda-nvjpeg-sys,
#   baracuda-npp-sys, baracuda-nvcomp-sys, baracuda-cvcuda-sys,
#   baracuda-cufile-sys
#
# crates.io's "new crate" quota is ~5 fresh names per ~1h per publisher,
# which turned out to be more forgiving than I first thought. Re-publishing
# an existing name does NOT count toward that quota.
#
# Run this script from the repo root whenever the retry window opens.
# It uses `set -e` so it stops at the first 429 — remove any
# successfully-published crates from the arrays below before the next
# run, or request a quota bump at:
#   https://github.com/rust-lang/crates.io/issues/new?template=rate_limit_increase.yml

set -e

# -sys crates still pending. baracuda-cupti-sys is the first retry target.
SYS_CRATES_PENDING=(
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
