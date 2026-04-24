#!/usr/bin/env bash
# Auto-publish baracuda to crates.io.
#
# Start it once — it will:
#   1. Walk the dependency-ordered list of 46 crates.
#   2. Skip any crate whose version is already on crates.io.
#   3. Call `cargo publish -p <crate>` on the next pending one.
#   4. If crates.io responds 429, parse the retry-after timestamp,
#      sleep until it passes (+ 30 s slack), and retry the same crate.
#   5. If `cargo publish` fails for any other reason, print the log and
#      exit with the underlying exit code.
#
# Hit Ctrl-C at any time to stop cleanly. Re-running picks up where it
# left off because already-published crates are detected via the sparse
# index (no local state file needed).
#
# Requires: bash 4+, curl, GNU `date -d` (default on Linux / macOS /
# Git Bash on Windows).

set -u -o pipefail

VERSION="0.0.1-alpha.2"

# Dependency-ordered list. Foundation → -sys crates → safe foundation
# (driver, runtime) → safe wrappers → umbrella.
ALL_CRATES=(
    baracuda-types-derive
    baracuda-types
    baracuda-build
    baracuda-core
    baracuda-cuda-sys
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
    baracuda-driver
    baracuda-runtime
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
    baracuda
)

# Map a crate name to its sparse-index path.
# crates.io layout:
#   1-char  → "1/<name>"
#   2-char  → "2/<name>"
#   3-char  → "3/<first>/<name>"
#   4+-char → "<first-two>/<next-two>/<name>"
sparse_index_path() {
    local name=$1
    local n=${#name}
    case $n in
        1) printf '1/%s' "$name" ;;
        2) printf '2/%s' "$name" ;;
        3) printf '3/%s/%s' "${name:0:1}" "$name" ;;
        *) printf '%s/%s/%s' "${name:0:2}" "${name:2:2}" "$name" ;;
    esac
}

# Returns 0 if <crate> <version> is already on crates.io, 1 otherwise.
already_published() {
    local crate=$1
    local version=$2
    local path
    path=$(sparse_index_path "$crate")
    local url="https://index.crates.io/$path"
    local body
    # --fail-with-body so a 404 returns 22; curl exits 0 on 200 only.
    if body=$(curl -sS --fail "$url" 2>/dev/null); then
        grep -q "\"vers\":\"$version\"" <<< "$body"
        return $?
    fi
    return 1
}

# Sleep until the timestamp in stdin ("Fri, 24 Apr 2026 13:57:22 GMT"
# or similar), with a 30-second slack buffer. Clamps to [15, 3600] s
# for safety.
sleep_until() {
    local ts=$1
    local target_epoch
    if ! target_epoch=$(date -d "$ts" +%s 2>/dev/null); then
        echo ">> couldn't parse '$ts'; defaulting to 60s sleep" >&2
        sleep 60
        return
    fi
    local now_epoch sleep_for
    now_epoch=$(date +%s)
    sleep_for=$(( target_epoch - now_epoch + 30 ))
    if (( sleep_for < 15 )); then sleep_for=15; fi
    if (( sleep_for > 3600 )); then sleep_for=3600; fi
    echo ">> rate-limited; sleeping ${sleep_for}s (until $ts + 30s slack)"
    sleep "$sleep_for"
}

# Publish a single crate, retrying past any 429s. Returns cargo's exit
# status on non-rate-limit failures. Automatically falls back to
# `--no-verify` when the failure is a dev-dep cycle with another
# baracuda crate that isn't yet on crates.io.
try_publish() {
    local crate=$1
    local extra_args=()
    while :; do
        echo "=== publishing $crate ${extra_args[*]}==="
        local log status
        # Capture combined stdout+stderr while also showing it live.
        log=$(cargo publish -p "$crate" "${extra_args[@]}" 2>&1 | tee /dev/stderr)
        status=${PIPESTATUS[0]}
        if (( status == 0 )); then
            return 0
        fi
        # 429 → sleep and retry.
        local retry_after
        retry_after=$(sed -n 's/.*Please try again after \(.*GMT\).*/\1/p' <<< "$log" | head -n1)
        if [[ -n $retry_after ]]; then
            sleep_until "$retry_after"
            continue
        fi
        # "no matching package named `baracuda-...` found" → dev-dep cycle,
        # retry with --no-verify.  We detect any missing baracuda-* package
        # since they'll all be on crates.io eventually; skipping local
        # re-compile against a not-yet-published sibling is safe because
        # the workspace build already verified the crate compiles.
        if [[ " ${extra_args[*]-} " != *" --no-verify "* ]] && \
           grep -q 'no matching package named `baracuda-' <<< "$log"; then
            echo ">> $crate has a dev-dep cycle on an unpublished baracuda crate;" \
                 "retrying with --no-verify" >&2
            extra_args+=(--no-verify)
            continue
        fi
        echo "!! $crate failed (exit $status) with a non-rate-limit error; aborting" >&2
        return "$status"
    done
}

# ---- main --------------------------------------------------------------

published_count=0
skipped_count=0
remaining_count=0

for crate in "${ALL_CRATES[@]}"; do
    if already_published "$crate" "$VERSION"; then
        echo "-- $crate v$VERSION already on crates.io; skipping"
        (( skipped_count++ )) || true
        continue
    fi
    (( remaining_count++ )) || true
done

echo ">> $skipped_count already published, $remaining_count to go."
echo

for crate in "${ALL_CRATES[@]}"; do
    if already_published "$crate" "$VERSION"; then
        continue
    fi
    try_publish "$crate" || exit $?
    (( published_count++ )) || true
done

echo
echo "=== done: $published_count crates published this run, all ${#ALL_CRATES[@]} baracuda crates are now live at v$VERSION ==="
