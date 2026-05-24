# baracuda Roadmap

The live backlog of outstanding work — deferrals from completed phases
plus the long-arc 1.0-freeze items. Categories below are roughly
ordered by current priority; specific items are ordered by impact ×
effort within each category. Authoritative status per op lives in
[`OP-MATRIX.md`](OP-MATRIX.md); historical phase summaries live in
[`ARCHITECTURE.md`](ARCHITECTURE.md).

The current tag is **v0.0.1-alpha.31** with **1890 GPU tests
passing** on RTX 4070 (sm_89) across **602 binary targets**.

---

## How this file gets updated

- New deferral discovered during a phase → add to the matching
  category below with a brief "what's missing + why deferred" line.
- Item ships in a release → remove from this file, add a line to
  the relevant op row in `OP-MATRIX.md`, and append a memory entry
  under `~/.claude/projects/c--Users-cires-OneDrive-Documents-projects-baracuda/memory/`.
- Priority shifts (Fuel asks for something + something gets bumped) →
  reorder within the relevant category.

---

## Phase 15 — quick wins + correctness bugs (complete; shipped alpha.32)

- **MMVQ `w_start_byte_offset` alignment guard** ✓ Phase 15.1. Debug-
  build assertion in `GgufMmvqPlan::select()` gated on
  `#[cfg(debug_assertions)]` (release builds elide the check).
  Per-block-format alignment requirements: 2 bytes for
  Q4_0/Q5_0/Q8_0/Q3_K/Q6_K; 4 bytes for Q4_1/Q5_1/Q2_K/Q4_K/Q5_K/Q8_K.
- **OneHot / Nonzero i64 Rust wrappers** ✓ Phase 15.2. Both
  refactored to take `I: IndexElement = i32` (default preserves
  source-compat). MaskedFill correctly out of scope — it takes a
  `u8` Bool mask, not an index tensor (Phase 11.5's grouping was
  wrong; no `masked_fill_i64idx_*` FFI symbols exist).
- **MoE fixture race fix** ✓ Phase 15.3. The previous `top_k=2`
  test fixture had two experts writing to the same output address
  with no kernel-side synchronization. Rewrote to `top_k=1` with
  `token t → expert t % num_experts`. Reference math was already
  correct; the kernel's no-synchronization contract is now
  surfaced in `OP-MATRIX.md` (MoE row) so callers know.

Items pre-emptively listed here in an earlier ROADMAP revision but
no longer outstanding:

- ~~**CTC backward γ-accumulation bug**~~ — already fixed
  2026-05-16 (see `loss_ctc_backward_smoke.rs` header comment). The
  earlier ROADMAP entry was stale agent-memory inheritance. Hand-
  computed and PyTorch-invariant tests cover f32/f16/bf16/f64.
- ~~**MoE CPU-reference math fix**~~ — turned out to be a fixture
  race, fixed in 15.3 above.

## Phase 16 — pool completion (proposed)

- **Bit-exact AdaptiveAvgPool / AdaptiveMaxPool** — current cuDNN
  approximation (`kernel = ceil(in/out); stride = floor(in/out);
  pad = 0`) diverges from PyTorch by ±1 input cell when
  `in_i % out_i != 0`. cuDNN doesn't expose a true adaptive pool —
  needs a bespoke CUDA kernel that computes per-output-cell kernel
  bounds. PyTorch convention: `start = floor(i * in_i / out_i)`,
  `end = ceil((i+1) * in_i / out_i)`.
- **LpPool 1d / 2d** — `LpPool(p, kernel) = (avg_pool(input^p))^(1/p)`.
  Could be a composite plan (`pow → avg_pool → pow`); blocked on a
  parameterized `Pow(p)` unary plan that doesn't exist yet (today's
  Pow is binary; the unary parameterized PowI from Phase 12 takes
  integer exponents only). Two paths: add `PowFloat(p)` parameterized
  unary, then compose; OR ship a bespoke fused kernel.
- **FractionalMaxPool 2d / 3d** — bespoke kernel needed (cuDNN
  doesn't support this). PyTorch uses pseudorandom sampling to
  decide kernel-window offsets. Effort: medium — design pass needed
  on the RNG interface.

## Phase 17 — SDPA / attention completion (proposed)

- **Flash SDPA sm_89 strided sibling** — the existing Phase 10
  `FlashSdpaSm89Plan` hardcodes `bh_off * q_len * d_k +
  qbase * d_k` offsets internally; the kernel is not stride-driven.
  Rewriting it as stride-driven would unlock strided attention on
  the perf-critical sm_89 path (the user's RTX 4070). Effort:
  medium-large — needs to thread strides through the `cp.async`
  double-buffer + SMEM-tile loops without losing the existing
  tuning. The generic naive `SdpaPlan` strided sibling already
  shipped in Phase 14.4; this would bring the sm_89 Flash variant
  to feature parity.
- **SDPA backward + GQA broadcast** — Phase 14.4 plan layer rejects
  `stride_k[head_axis] == 0` / `stride_v[head_axis] == 0` for the
  backward path with `Error::Unsupported`. Lifting this requires an
  atomicAdd-based dK / dV accumulation kernel for the broadcast
  axis (the `atomicAdd_via_cas` helper from Phase 11.3 already
  exists for the bf16 / f16 case). Effort: medium — design pass
  needed on the accumulation pattern; smoke tests need a reference
  that fans out the broadcast and compares against the naive
  expanded version.
- **Paged FlashAttention** — original Phase 6 milestone never landed.
  Effort: large — separate design effort.

## Phase 18 — sub-byte + quantized completeness

- **f16 / bf16 activations for `GgufMmvqPlan`** — f32 only today.
  llama.cpp upstream has these paths; need to vendor + adapt.
- **MMVQ multi-dim activation strides** — Phase 14.5 chose a single
  `stride_y: i64` model (MMVQ is rank-1 at the kernel surface).
  If Fuel hits a case where they need rank > 1 at the FFI level
  (instead of host-loop batching), revisit and add the multi-dim
  shape + stride params.

## Phase 19 — segment + embedding BW completion

- **Segment `Max` / `Min` BW** — needs argmax / argmin tracking in
  the FW (output a paired index tensor alongside the value output;
  BW gathers gradient into the recorded index).
- **Segment `Prod` BW** — needs numerically stable `prod / x_n` for
  the gradient (division-by-self for each contributing element).
  Care needed for zero inputs.
- **Unsorted Segment Max / Min BW** — same as sorted variants plus
  the non-determinism caveat already documented in OP-MATRIX.
- **Unsorted Segment `Prod`** — has no FW today either (no native
  FP `atomicMul`; would need an `atomicCAS` retry loop).
- **`EmbeddingBag Max` mode** — needs per-feature argmax tracking
  (same pattern as Segment Max).

## Phase 20 — linalg completion

- **`BatchedOrmqrWyPlan` complex variants** — real `{f32, f64}` ship
  today; complex needs `cunmqr` rather than `cusolverDnXormqr`.

## Phase 21+ — long-arc roadmap items

These were the original "Phase 11" and "Phase 12" items in the
comprehensive plan before Fuel-driven work pre-empted the numbering.
They remain valid 1.0-freeze gates.

- **sm_90a (Hopper async) specialization** — sibling plans for
  Hopper's WGMMA + async tensor cores + cluster-launch. The
  sibling-plan + arch-dispatcher pattern Phase 10 established for
  sm_89 generalizes here.
- **Blackwell forward-compat** — verify the kernel set compiles +
  runs on sm_100+ once a Blackwell test machine is available.
- **API freeze + 1.0 stability** — review all `pub` surfaces;
  document breaking-change policy; cut `0.1.0-beta.0` once the
  surface settles. Likely involves consolidating the `T: Element`
  vs `T: DeviceRepr + Copy` trait-bound split (currently split by
  whether sub-byte dtypes need to participate; see Phase 13
  pragma).
- **Benchmark suite vs. PyTorch / cuDNN / cuBLAS references** —
  extend the Phase 10 `baracuda-kernels-bench` criterion harness to
  cover the full op matrix with reference comparisons; publish
  perf-vs-baseline tables per release.

---

## Cross-cutting carry-forwards

Not phase-specific; tracked here so they don't fall off the radar.

- **Sparsemax for extents > 1024** — Phase 11.6 lifted the cap from
  64 → 1024 via `cub::BlockRadixSort` + `BlockScan`. Larger rows
  would need a multi-block / global sort pipeline. Low priority
  unless a use case shows up.
- **CTC bespoke flake under parallel test execution** —
  `cudnn_ctc_f32_uniform_t2_c2` intermittently fails when the full
  test suite runs in parallel; passes deterministically in
  isolation. Likely cuDNN handle contention. Not a correctness
  issue — pure test infrastructure flake.
- **Documentation lifecycle hooks** — the
  `feedback_readme_badges_on_publish` memory entry captures the
  "bump README badges on release" gotcha. A pre-commit hook or
  release script could automate this rather than relying on memory.
