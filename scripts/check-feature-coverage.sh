#!/usr/bin/env bash
# Guard: every crate with `#[cfg(feature = "…")]`-gated code must have that code
# built by SOME CI job, or be explicitly recorded as CUDA-only (uncovered because
# no CI CUDA runner exists). The failure this defends against is the one that broke
# Fuel's `--features vulkan` for 72 minutes with every check green: a feature gates
# code, the code builds without the toolkit, and NO job compiles it — so a rename
# or API drift there ships invisible. A partially-covered dimension is MORE
# dangerous than an uncovered one, because the presence of some `--features` steps
# tells a reader the dimension is handled.
#
# The gate discriminates on its OWN configuration, not just on code:
#   - the GATED set is DERIVED fresh (grep), so a NEW feature-gated crate can't
#     silently join without being classified — it reds until triaged;
#   - it asserts over the SET, not a count, so add-one-drop-one can't pass;
#   - it asserts the ALL_FEATURES steps actually appear in ci.yml (the list can't
#     lie about coverage it doesn't have);
#   - it asserts no ALL_FEATURES crate's `--all-features` graph pulls an
#     UNCONDITIONAL-nvcc crate — because on a no-CUDA runner that crate's build.rs
#     SKIPS (it gates on toolkit presence, not features), so the step would go
#     GREEN while forging nothing and appearing to cover the crate. Vacuous-green
#     is invisible in exactly the environment that can't verify CUDA-freeness.
set -euo pipefail

# ── Coverage classification (the three lists partition the GATED set) ──────────
# CUDA-FREE crates CI clippies with `--all-features` (a new feature auto-joins).
ALL_FEATURES=(
  baracuda baracuda-build baracuda-types baracuda-cudnn baracuda-nccl
  baracuda-megatron baracuda-runtime baracuda-driver baracuda-cuvs
  baracuda-cuvs-sys baracuda-cuda-emit
)
# MIXED crates: some features need the toolkit, so `--all-features` can't be used;
# the CUDA-FREE feature(s) are enumerated in CI and pinned here. Any OTHER feature
# that gates code must be either listed here or CUDA-denylisted below, or it reds.
declare -A ENUMERATED=( [baracuda-optim]="distributed_optim" )
declare -A CUDA_FEATURES=( [baracuda-optim]="sm80 sm89 sm90a" )
# Crates whose feature-gated code genuinely needs nvcc/CUDA headers → uncovered by
# design (no CI CUDA runner); recorded so the absence is a stated blindness, not a
# gap someone "fixes" by adding a step that then skips-green on the runner.
CUDA_ONLY=(
  baracuda-cutlass baracuda-cutlass-kernels-sys baracuda-flashinfer-sys
  baracuda-flashinfer baracuda-kernels-sys baracuda-kernels baracuda-tensorrt-sys
)
# Crates whose build.rs invokes nvcc UNCONDITIONALLY (graph membership is enough —
# no feature guard). If an ALL_FEATURES `--all-features` graph pulls one of these,
# the step is vacuous on a no-CUDA runner. (optim only-with-arch and tensorrt-sys
# only-with-shim are conditional and excluded here.)
ALWAYS_NVCC="baracuda-kernels-sys baracuda-cutlass-sys baracuda-cutlass-kernels-sys baracuda-ozimmu-sys baracuda-transformer-engine-sys baracuda-kernels-bench"

CI=".github/workflows/ci.yml"
fail() { echo "FEATURE-COVERAGE GUARD: $*"; exit 1; }
in_list() { local x="$1"; shift; local e; for e in "$@"; do [ "$e" = "$x" ] && return 0; done; return 1; }

# ── Derive the GATED set fresh (crates with cfg(feature=) code) ────────────────
gated=()
for d in crates/*/; do
  c=$(basename "$d")
  # Match feature= inside ANY cfg form — cfg(feature=), cfg(any(feature=…)),
  # cfg(all(test, feature=…)). The narrow `cfg\(feature` pattern missed
  # cfg(any(...)) and made a whole nvcc crate (cutlass-kernels-sys) invisible to
  # this classifier — the exact vocabulary-narrower-than-the-population defect
  # this guard exists to catch, so don't re-narrow it.
  grep -rqE 'cfg\(.*feature *= *"' "$d"src/ 2>/dev/null && gated+=("$c")
done
[ "${#gated[@]}" -gt 0 ] || fail "GUARD BROKEN: found 0 crates with cfg(feature=) code — the grep is wrong or the layout changed."

# ── 1+2. Every GATED crate is in exactly one list; lists don't overlap ─────────
for c in "${gated[@]}"; do
  n=0
  in_list "$c" "${ALL_FEATURES[@]}" && n=$((n+1))
  [ -n "${ENUMERATED[$c]+x}" ] && n=$((n+1))
  in_list "$c" "${CUDA_ONLY[@]}" && n=$((n+1))
  [ "$n" -eq 1 ] || fail "'$c' has feature-gated code but is in $n coverage lists (want exactly 1). A new/changed feature-gated crate must be classified: ALL_FEATURES (CUDA-free, --all-features'd), ENUMERATED (mixed), or CUDA_ONLY (needs the toolkit)."
done

# ── 3. ENUMERATED crates: every cfg-gating feature is listed or CUDA-denylisted ─
for c in "${!ENUMERATED[@]}"; do
  listed=" ${ENUMERATED[$c]} ${CUDA_FEATURES[$c]:-} "
  # feature names in ANY cfg form (feature="X" appears in every cfg gate form;
  # in .rs a bare `feature = "…"` is a cfg predicate) — same broadening as the
  # GATED grep, so cfg(any(feature=…)) is not silently skipped here either.
  for f in $(grep -rhoE 'feature *= *"[^"]+"' "crates/$c/src/" 2>/dev/null | sed -E 's/.*"([^"]+)".*/\1/' | sort -u); do
    [[ "$listed" == *" $f "* ]] || fail "$c gates code on feature '$f', which is neither CI-covered (ENUMERATED='${ENUMERATED[$c]}') nor CUDA-denylisted (CUDA_FEATURES='${CUDA_FEATURES[$c]:-}'). A new non-CUDA feature has silently rejoined the uncovered set — add it to the CI step + ENUMERATED, or to CUDA_FEATURES if it needs the toolkit."
  done
done

# ── 4. The ALL_FEATURES list isn't lying: there IS an `--all-features` clippy step
#      and each ALL_FEATURES crate is named as a `-p` target in the workflow.
grep -qE "cargo clippy.*--all-features|--all-features.*cargo clippy" "$CI" \
  || grep -Pzoq "(?s)cargo clippy(\s|\\\\\n|[^\n])*--all-features" "$CI" 2>/dev/null \
  || fail "no 'cargo clippy … --all-features' step found in $CI — the ALL_FEATURES coverage does not exist."
for c in "${ALL_FEATURES[@]}"; do
  # `-p <crate>` followed by a non-name char or end-of-line, so `baracuda` does
  # not match `-p baracuda-build`.
  grep -qE "\-p $c([^-a-z0-9]|\$)" "$CI" \
    || fail "'$c' is classified ALL_FEATURES but is not a '-p $c' target anywhere in $CI — the classification claims coverage the workflow doesn't provide."
done

# ── 5. ENUMERATED pins aren't lying: each (crate,feature) appears in ci.yml ─────
for c in "${!ENUMERATED[@]}"; do
  for f in ${ENUMERATED[$c]}; do
    grep -qE "\-p $c .*--features $f|--features $f .*-p $c" "$CI" \
      || fail "ENUMERATED pin '$c --features $f' is not built in $CI."
  done
done

# ── 6. Anti-vacuous: no ALL_FEATURES --all-features graph pulls an always-nvcc crate
for c in "${ALL_FEATURES[@]}"; do
  # Do NOT `|| true`: a `cargo tree` failure means the anti-vacuous check could
  # not run, and a guard that swallows its own inability to run reports success
  # on a run it knows was incomplete. Fail loud instead.
  tree=$(cargo tree -p "$c" --all-features -e no-dev 2>&1) \
    || fail "GUARD BROKEN: 'cargo tree -p $c --all-features' failed, so the anti-vacuous check could not run: $tree"
  for n in $ALWAYS_NVCC; do
    # POSIX-safe word boundary: `\b` is a GNU extension, undefined on BSD/macOS
    # grep — on such a runner it would silently make THIS anti-vacuous check never
    # fire, on the one guard whose job is catching things that never fire. Match
    # start-of-line or a non-name char so a suffix like x-$n doesn't false-match.
    echo "$tree" | grep -qE "(^|[^-a-z0-9])$n v" && fail "'$c --all-features' pulls '$n' (unconditional-nvcc) into its graph — on a no-CUDA runner its build.rs SKIPS, so the CI step passes GREEN while covering nothing (vacuous). Either the crate gained a CUDA dep, or it belongs in CUDA_ONLY."
  done
done

echo "feature-coverage OK: ${#gated[@]} feature-gated crates, all classified; ALL_FEATURES steps present in ci.yml; ENUMERATED features accounted for; no ALL_FEATURES step pulls an unconditional-nvcc crate."
