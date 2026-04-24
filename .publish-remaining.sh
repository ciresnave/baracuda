#!/usr/bin/env bash
# Resume the baracuda crates.io publish chain.
#
# --- Progress so far (0.0.1-alpha.2) ---
# Round 1 (2026-04-22): foundation-5 at alpha.1.
# Round 2 (2026-04-23): alpha.2 re-pubs of the foundation + 5 more -sys.
# Round 3 (2026-04-24 ~02h): cusparse/cusolver/cudnn/nccl/nvml-sys.
# Round 4 (2026-04-24 ~13h): nvjpeg/npp/nvcomp/cvcuda/cufile-sys.
# Round 5 (2026-04-24 ~13:45h): cupti-sys only — blocked at
#                               baracuda-cutensor-sys
#                               (retry-after: 2026-04-24 13:57:22 GMT
#                               — 10 min away).
#
# Already on crates.io at 0.0.1-alpha.2 (21/46):
#   baracuda-types-derive, baracuda-types, baracuda-build, baracuda-core,
#   baracuda-cuda-sys, baracuda-nvrtc-sys, baracuda-nvjitlink-sys,
#   baracuda-cublas-sys, baracuda-curand-sys, baracuda-cufft-sys,
#   baracuda-cusparse-sys, baracuda-cusolver-sys, baracuda-cudnn-sys,
#   baracuda-nccl-sys, baracuda-nvml-sys, baracuda-nvjpeg-sys,
#   baracuda-npp-sys, baracuda-nvcomp-sys, baracuda-cvcuda-sys,
#   baracuda-cufile-sys, baracuda-cupti-sys
#
# The rate-limit window is a sliding one — it can re-open after just
# 10 minutes when the previous round uploaded fewer new names. Re-running
# this script every time the retry-after passes is the expected flow.
#
# Run this script from the repo root whenever the retry window opens.
# It uses `set -e` so it stops at the first 429 — remove any
# successfully-published crates from the arrays below before the next
# run, or request a quota bump at:
#   https://github.com/rust-lang/crates.io/issues/new?template=rate_limit_increase.yml

set -e

# -sys crates still pending. baracuda-cutensor-sys is the first retry target.
SYS_CRATES_PENDING=(
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
