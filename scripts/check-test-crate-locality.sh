#!/usr/bin/env bash
# Guard: a test in `crates/A/tests/` runs under `cargo test -p A` and NEVER under
# `cargo test -p B`, regardless of whose code it exercises. A split moves code and
# leaves tests where they were, and nothing in the toolchain objects — so a test
# that exercises crate B's behaviour but lives in crate A's `tests/` runs in a
# context whose property it does not guard, and its count is right while its
# coverage is misplaced (workspace totals never move).
#
# This fails any `crates/*/tests/*.rs` that never names its own crate's lib name
# — with `//` comments STRIPPED first, because a crate name mentioned only in a
# doc comment is documentation, not usage (an unstripped check calls the original
# defect compliant; that is the exact trap `neutral_spelling.rs` recorded upstream).
#
# Two honesty checks keep the guard itself from going vacuous:
#   1. it asserts it scanned > 0 files (an empty walk must not read as clean);
#   2. it asserts every ALLOWLIST-exempt file is STILL a would-be violation — if an
#      exempt file starts naming its own crate, the exemption is stale and must be
#      removed, so the exemption can never silently cover a now-compliant file.
set -euo pipefail

# Consumer-side / convention exemptions — files that legitimately do not name
# their host crate. Each MUST stay a would-be violation (honesty check 2).
exempt=(
  # backend_declines.rs WAS here (CpuC/Slang decline contracts, consumer-side) —
  # removed 2026-08-26: it now also carries the Nextafter@f16 CUDA wiring test,
  # which uses baracuda_cuda_emit::Cuda, so it legitimately names its own crate and
  # is no longer a would-be violation. The guard's honesty-check-2 caught the stale
  # exemption the moment the file stopped qualifying for it — as designed.
  "crates/baracuda-types-derive/tests/derive_device_repr.rs" # proc-macro test drives the consumer crate
  "crates/baracuda-runtime/tests/external_smoke.rs"          # external-binding smoke
)
is_exempt() { local f="$1" e; for e in "${exempt[@]}"; do [ "$f" = "$e" ] && return 0; done; return 1; }
libname() { local c; c=$(basename "$(dirname "$(dirname "$1")")"); echo "${c//-/_}"; }

# Strip `//` comments and report whether $2 (the lib name) survives. This runs
# sed to completion into a variable and greps a here-string; it is NOT `sed … |
# grep -q …`, and re-piping it is a measured regression, not a stylistic one:
#
#   MEASURED (this guard's first live CI run, PR #27): the pipe form false-flagged
#   8 tests on ubuntu-latest that DO name their crate, while passing on
#   windows-latest and the local Git-Bash box — an 8/0/0 split by platform.
#
# Mechanism: under `set -o pipefail`, `grep -q` exits at the first match and
# closes the pipe; sed, still writing a file larger than the pipe buffer, takes
# SIGPIPE (exit 141); pipefail promotes 141 to the pipeline status even though
# grep SUCCEEDED, and `if !` inverts it into a violation. It fires only where the
# OS delivers SIGPIPE to a producer still writing after the reader closes — Linux
# does, MSYS/Git-Bash masks it — so it is invisible on every Windows box and reds
# only a Linux runner. The here-string has no sed→grep pipe, so nothing can
# SIGPIPE and pipefail has nothing to promote. Re-piping reintroduces the split.
names_crate() { local stripped; stripped=$(sed 's|//.*||' "$1"); grep -q "$2" <<<"$stripped"; }

scanned=0
violations=()
for d in crates/*/; do
  c=$(basename "$d"); lib=${c//-/_}
  for t in "$d"tests/*.rs; do
    [ -f "$t" ] || continue
    scanned=$((scanned + 1))
    if ! names_crate "$t" "$lib"; then
      is_exempt "$t" && continue
      violations+=("$t")
    fi
  done
done

# Honesty 1: an empty walk is a broken guard, not a clean tree.
if [ "$scanned" -eq 0 ]; then
  echo "GUARD BROKEN: scanned 0 test files — the walk found nothing (wrong cwd, or crates/ layout changed)."
  exit 2
fi

# Honesty 2: every exemption must still be earning its place.
for e in "${exempt[@]}"; do
  if [ ! -f "$e" ]; then
    echo "GUARD STALE: exempt file '$e' no longer exists — remove it from the allowlist."
    exit 2
  fi
  if names_crate "$e" "$(libname "$e")"; then
    echo "GUARD STALE: exempt file '$e' now names its own crate — the exemption is unnecessary, remove it."
    exit 2
  fi
done

if [ "${#violations[@]}" -gt 0 ]; then
  echo "TEST-IN-WRONG-CRATE — these tests/*.rs never name their own crate (comments stripped):"
  printf '  %s\n' "${violations[@]}"
  echo "A test that does not exercise its own crate runs under the wrong 'cargo test -p'."
  echo "Move it to the crate it exercises, or add it to the exempt list in this script with justification."
  exit 1
fi

echo "test-crate-locality OK: scanned $scanned files, ${#exempt[@]} exempt (each still a would-be violation), 0 unaccounted."
