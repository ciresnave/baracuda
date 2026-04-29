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

VERSION="0.0.1-alpha.4"

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

# Strip any `baracuda-*` lines from the [dev-dependencies] section of
# a crate's Cargo.toml. Writes to a fresh file; caller swaps it in.
strip_dev_deps() {
    local src=$1
    local dst=$2
    awk '
        /^\[dev-dependencies(\..*)?\]/ { in_dev=1; print; next }
        /^\[/                         { in_dev=0; print; next }
        in_dev && /^baracuda-/        { next }
        { print }
    ' "$src" > "$dst"
}

# Publish a single crate, retrying past any 429s. Returns cargo's exit
# status on non-rate-limit failures. Automatically strips `baracuda-*`
# dev-dep lines when the failure is a dev-dep cycle with another
# baracuda crate that isn't yet on crates.io, then restores the
# original Cargo.toml.
try_publish() {
    local crate=$1
    local cargo_toml="crates/$crate/Cargo.toml"
    local backup="$cargo_toml.bak.$$"
    local dev_deps_stripped=0
    # Safety net: if we die mid-publish (Ctrl-C, hard crash, etc.), restore.
    cleanup() {
        if (( dev_deps_stripped == 1 )) && [[ -f $backup ]]; then
            mv "$backup" "$cargo_toml"
            echo ">> restored $cargo_toml" >&2
        fi
    }
    trap cleanup RETURN INT TERM

    while :; do
        echo "=== publishing $crate $([[ $dev_deps_stripped -eq 1 ]] && echo '(dev-deps stripped)')==="
        local extra_args=()
        if (( dev_deps_stripped == 1 )); then
            extra_args+=(--allow-dirty)
        fi
        local log status
        log=$(cargo publish -p "$crate" "${extra_args[@]}" 2>&1 | tee /dev/stderr)
        status=${PIPESTATUS[0]}
        if (( status == 0 )); then
            trap - RETURN INT TERM
            cleanup
            return 0
        fi
        # 429 → sleep and retry (the stripped state, if any, persists).
        local retry_after
        retry_after=$(sed -n 's/.*Please try again after \(.*GMT\).*/\1/p' <<< "$log" | head -n1)
        if [[ -n $retry_after ]]; then
            sleep_until "$retry_after"
            continue
        fi
        # Dev-dep cycle with an unpublished baracuda crate → strip & retry.
        # Two formats appear in cargo's output:
        #   "no matching package named `baracuda-X`"            (older / fresh index miss)
        #   "failed to select a version for the requirement     (newer / stale-index miss)
        #      `baracuda-X = "^0.0.1-alpha.4"` ..."
        if (( dev_deps_stripped == 0 )) && \
           { grep -q 'no matching package named `baracuda-' <<< "$log" \
             || grep -q 'failed to select a version for the requirement `baracuda-' <<< "$log" ; }; then
            echo ">> $crate has a dev-dep cycle on an unpublished baracuda crate;" \
                 "stripping baracuda-* lines from [dev-dependencies] and retrying" >&2
            cp "$cargo_toml" "$backup"
            strip_dev_deps "$backup" "$cargo_toml"
            dev_deps_stripped=1
            continue
        fi
        echo "!! $crate failed (exit $status) with a non-rate-limit error; aborting" >&2
        trap - RETURN INT TERM
        cleanup
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
