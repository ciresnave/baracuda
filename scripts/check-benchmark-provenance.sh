#!/usr/bin/env bash
# Guard: `BENCHMARKS.md` states the machine it was measured on TWICE, and only one
# of the two can ever be updated.
#
# `tools/build_benchmarks_table.py` rewrites only what lies BETWEEN its markers.
# The generated preamble's `Hardware:` line is inside, so every regen refreshes it.
# The file's top-level `**Hardware**:` header is outside, so the generator is
# structurally incapable of touching it — no amount of care at regen time reaches
# it, and it drifts silently for as long as the file exists.
#
# Measured 2026-09-06 at main `33c58b48`:
#
#   line 16, header, OUTSIDE   **Hardware**: RTX 4070 (sm_89), CUDA 13.0, cuDNN 9.x.
#   preamble, INSIDE           Hardware: RTX 4070 Laptop GPU (sm_89).
#
# The header had carried that text unchanged since the file was created on
# 2026-05-28 — through every regeneration, including the full-suite one in #74 that
# corrected the INNER line on the same night. RTX 4070 and RTX 4070 Laptop are
# different parts with different SM counts and memory bandwidth, so a reader
# comparing these numbers against published desktop figures compares against the
# wrong GPU. The numbers were fresh; the sentence saying what produced them was not,
# which is the combination a reader is least able to catch.
#
# This is the same shape as #73/#77 (two claims about one gate, ~25 lines apart,
# disagreeing) and #78 (a SMEM header true at d=64 and false at the d=128 the code
# below it declares supported). Three in one evening is a property of how these
# headers are maintained, not a run of bad luck — so it wants a guard, not a fix.
#
# NOT in scope: the dated `### Measured on … 2026-05-28` section. That records a
# specific past run rather than describing the current rollup, and rewriting a
# measurement nobody here took would be worse than leaving it stated as history.
#
# Op-family drops are NOT checked here — `build_benchmarks_table.py` refuses those
# at regen time since #79, which is a better place for it: it fires at the moment
# of the mistake rather than a push later.
set -euo pipefail

MD="crates/baracuda-kernels-bench/BENCHMARKS.md"
BEGIN_MARKER="<!-- BEGIN auto-generated phase29 rollup -->"
END_MARKER="<!-- END auto-generated phase29 rollup -->"

[ -f "$MD" ] || { echo "GUARD BROKEN: $MD not found — wrong cwd?"; exit 2; }

# Honesty 1: both markers must be present. Without them the region below is empty
# and every comparison in this guard passes having compared nothing.
grep -qF -- "$BEGIN_MARKER" "$MD" || { echo "GUARD BROKEN: BEGIN marker missing from $MD."; exit 2; }
grep -qF -- "$END_MARKER"   "$MD" || { echo "GUARD BROKEN: END marker missing from $MD."; exit 2; }

# Extracted to a variable, not piped into a consumer that can exit early: under
# `set -o pipefail` a downstream `grep -q` closing the pipe gives the producer
# SIGPIPE (141), which pipefail promotes to the pipeline's status. That exact
# failure was measured on ubuntu-latest in PR #27 and is recorded at length in
# scripts/check-test-crate-locality.sh.
region=$(awk -v b="$BEGIN_MARKER" -v e="$END_MARKER" \
  'index($0,b){f=1;next} index($0,e){f=0} f' "$MD")

# Honesty 2: an empty or truncated region is a broken guard, not a clean file.
region_lines=$(printf '%s\n' "$region" | grep -c '' || true)
if [ "$region_lines" -lt 10 ]; then
  echo "GUARD BROKEN: the generated region holds $region_lines lines — markers matched but the body is empty or truncated."
  exit 2
fi

header_hw=$(grep -m1 -E '^\*\*Hardware\*\*:' "$MD" || true)
gen_hw=$(printf '%s\n' "$region" | grep -m1 -E '^Hardware:' || true)

# Honesty 3: both claims must be FOUND. If either line is renamed away, this guard
# must fail loudly rather than quietly stop checking the thing it exists for.
if [ -z "$header_hw" ]; then
  echo "GUARD BROKEN: no '**Hardware**:' header line in $MD — renamed, or removed."
  exit 2
fi
if [ -z "$gen_hw" ]; then
  echo "GUARD BROKEN: no 'Hardware:' line inside the generated region — the generator's preamble changed shape."
  exit 2
fi

# Compare the GPU PART only. The generated line deliberately carries no toolchain
# versions — #74 dropped them rather than restate them per-run — so comparing whole
# lines would fail for a reason that is not drift.
#
# Markdown emphasis is stripped first. The header is prose a human formats and the
# generated line is plain text a script writes, so `RTX 4070 **Laptop GPU**` and
# `RTX 4070 Laptop GPU` are the same claim differently marked up. Without the strip
# this guard reports a hardware disagreement for a pair of asterisks — which it did,
# on the very commit that fixed the header, so the failure mode is not theoretical.
# ⚠ The trailing `|| true` is load-bearing, not defensive noise. Under
# `set -euo pipefail` a `grep` that matches nothing exits 1, pipefail promotes it
# to the pipeline, and the command substitution below then aborts the WHOLE SCRIPT
# — before honesty 4 can run. Measured while exercising these paths: renaming the
# GPU in both lines killed the guard with exit 1 and NO message, so the check
# written to catch an unparseable part was unreachable. A guard that cannot report
# its own bad case is the defect this file exists to find, one level in.
part_of() {
  printf '%s\n' "$1" \
    | tr -d '*_' \
    | grep -oE 'RTX [0-9]+([[:space:]]+[A-Za-z]+)*' \
    | head -1 \
    | tr -s ' ' || true
}
h_part=$(part_of "$header_hw")
g_part=$(part_of "$gen_hw")

# Honesty 4: a part must parse out of BOTH. Two empty strings compare equal, which
# would be a pass produced by a broken parser rather than by agreement.
if [ -z "$h_part" ] || [ -z "$g_part" ]; then
  echo "GUARD BROKEN: could not parse a GPU part from one of the two hardware lines:"
  echo "  header:    $header_hw"
  echo "  generated: $gen_hw"
  exit 2
fi

if [ "$h_part" != "$g_part" ]; then
  echo "HARDWARE CLAIM DISAGREES — two claims about one machine, in one file:"
  echo "  header (hand-maintained, OUTSIDE the markers): $h_part"
  echo "      $header_hw"
  echo "  generated (rewritten by every regen, INSIDE):  $g_part"
  echo "      $gen_hw"
  echo
  echo "The generator rewrites only between its markers, so it cannot reach the"
  echo "header and the header cannot self-correct. The generated line is the fresh"
  echo "one; make the header agree with it."
  exit 1
fi

echo "benchmark-provenance OK: header and generated preamble agree on the GPU part ($h_part); region $region_lines lines."
