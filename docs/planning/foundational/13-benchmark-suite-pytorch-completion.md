# 13 — Benchmark suite completion: the frozen PyTorch baseline

> Plan for the ROADMAP 1.0-freeze gate **"Cross-implementation benchmark
> suite vs PyTorch / cuDNN / cuBLAS"** (`ROADMAP.md`, Pre-1.0 must-haves).
> Written 2026-09-01 after mapping the existing harness. The PyTorch
> comparison strategy is **candidate 2 (frozen reference values on disk)**,
> ratified by the portfolio PM over candidate 1 (`tch-rs` — a LibTorch build +
> distribution dependency) and candidate 3 (out-of-process Python — already
> rejected in Phase 29 as too slow).

## The state is further along than the roadmap said

The roadmap text says "~10 ops with cuBLAS/cuDNN references." **Measured, the
harness already covers far more:**

- `crates/baracuda-kernels-bench/` — ~23 `[[bench]]` binaries over a shared
  library (`src/lib.rs`): device setup, the CUDA-event timing primitive
  (`time_with_events`), `warmup`, `measure_median_ns`, the shape-sweep consts,
  the FLOP helpers, the `PhaseTwentyNineRow` result row + CSV emitter, and the
  `PytorchBaseline` frozen-JSON loader.
- **Candidate 2 is already IMPLEMENTED as a timing baseline.**
  `bench-baselines/pytorch_rtx4070_2.11.0_cu130.json` carries **58 distinct
  ops / 319 rows** of PyTorch timing references, loaded in-process by
  `PytorchBaseline` and surfaced through the `pytorch_ns` column that
  `tools/build_benchmarks_table.py` already joins into the rollup.

So this gate is **harden + extend**, not build-from-scratch.

## The frozen-ref contract (the two acceptance conditions)

The baseline is a *checkable claim* only if it carries its provenance and its
staleness is detectable. Two conditions, both first-class in the schema:

1. **Provenance per baseline.** The metadata block records `torch_version`,
   `cuda_version`, `device_name`, `device_capability`, `generated_at_utc`,
   the sample/inner/warmup counts, a methodology string, and (v2)
   `generator_git_sha`. A timing reference generated on different hardware is
   not a reference — `device_name` + git SHA + `generated_at_utc` is what makes
   the comparison attributable. **A consumer must confirm `device_name` matches
   its own target before trusting the numbers.**
2. **A stated regeneration trigger (a condition, never a date).** The metadata
   `regen_trigger` block states the conditions under which the baseline must be
   regenerated: (1) a PyTorch MAJOR bump; (2) a documented PyTorch numerics
   change in a covered op; (3) a target-device / CUDA-toolchain change; (4) new
   ops added. **Condition (1) is auto-detected** by the scheduled
   `.github/workflows/pytorch-baseline-liveness.yml` (WARNS, opens a dedup'd
   issue, **never reds a build** — the `cuda-vocab-pin-liveness` pattern);
   (2)-(4) are documented conditions a human evaluates, printed by the same job
   as a reminder. An unconditioned frozen ref is a permanent hold in a frozen
   artifact's clothes.

### Timing-only, plus a liveness check that is NOT a reference

The baseline stores **timing only** (`median_ns`), no numerical values.
Numerical correctness is a **separate authority** — `kiss-ref-diff` (the
logical oracle) + on-device parity tests. A second numerical authority in the
bench JSON would be a *divergence generator*: two references that can disagree
with nothing to arbitrate. A weak second authority (a checksum) is worse than
none, for the same reason.

But a benchmark that times garbage is meaningless and currently silent — an op
returning NaN, the wrong shape, or exiting early on a degenerate input still
produces a fast number. So each benched cell gets a **liveness assertion** —
output is finite, correct element count, correct dtype — stated explicitly as a
liveness check and **not** a numerical reference. It distinguishes *"this op
ran"* from *"this op agrees with PyTorch"*; only the second belongs to the
oracle. One check per cell, outside the timed loop, no stored value, no second
regeneration policy. (It checks baracuda's own output, so it is independent of
PyTorch — the rollout wave needs no torch install.)

## Attribution of the existing 319 rows

The existing baseline's `device_name` matches this box's GPU exactly and it was
committed by Eric on 2026-06-04, but the artifact carries no host fingerprint
beyond the GPU MODEL, and thermal/power state at generation is unrecorded. So it
is **provisional**: its v2 `attribution` field records the rows as
device-model-matched but physical-machine + thermal/power-state UNVERIFIED, with
no `generator_git_sha` (they predate the field). v2 does not falsely imply
attribution. The first fully-attributed baseline is the same-box regeneration.

## Phasing

- **Phase 1 — foundation (this PR, no PyTorch needed):** discharge both
  conditions. v1→v2 schema (`generator_git_sha` + `attribution` +
  `regen_trigger`), backward-compatible loader (v1 still parses; a missing
  regen-trigger warns at load), refresh-script v2 emit, the drift-detection
  workflow, this plan, and the ROADMAP "~10 ops" correction.
- **Phase 1b — liveness helper + rollout (no PyTorch needed):** add
  `assert_cell_live` to the harness and apply it per cell across the existing
  benches. Independent of the torch install; unblocked today.
- **Phase 1c — auto-generate `BENCHMARKS.md`** from the per-bench CSVs (the
  roadmap's second open seam): wire `tools/build_benchmarks_table.py` into a
  committed target so the rollup stops being hand-maintained.
- **Phase 2+ — coverage waves (needs a PyTorch env on the target box):** extend
  the refresh script + benches from 58 ops toward the full `OP-MATRIX.md`
  surface (167 Plan rows across 21 categories). Ordered by value: Loss (20),
  Linalg-Dense (15), Quantization (15), Sort/Order (11), Segment/Scatter-Reduce
  (10), Image/Spatial (8), FFT (8), Indexing (6), then backward passes.

### Infra gate for Phase 2+

PyTorch is not installed on the target GPU box. A timing ref from different
hardware is not a reference (it measures a different machine), and there is no
GPU CI runner — so refs from another box and a CI-generated baseline are both
invalid for *timing*. The only valid source is a PyTorch install **on the target
box**; that is Eric's call (a multi-GB CUDA-torch install on his machine) and
does not gate the foundation.
