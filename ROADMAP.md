# baracuda Roadmap

The live backlog of outstanding work — deferrals from completed phases
plus the long-arc 1.0-freeze items. Categories below are roughly
ordered by current priority; specific items are ordered by impact ×
effort within each category. Authoritative status per op lives in
[`OP-MATRIX.md`](OP-MATRIX.md); historical phase summaries live in
[`ARCHITECTURE.md`](ARCHITECTURE.md).

The current tag is **v0.0.1-alpha.36** with **1957 GPU tests
passing** on RTX 4070 (sm_89) across **614 binary targets**.

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

## Phase 16 — pool completion (complete; shipped alpha.33)

- **Bit-exact AdaptiveAvgPool / AdaptiveMaxPool {1,2,3}d** ✓
  Phase 16.1. Replaces cuDNN approximation with bespoke rank-
  agnostic CUDA kernel implementing PyTorch's exact
  `start = floor(i*in/out); end = ceil((i+1)*in/out)`. 16 FFI
  symbols (FW + BW × 4 fp dtypes). MaxPool BW recomputes argmax
  from saved `x` to preserve `*BwArgs` API source-compat. Existing
  callers depending on the approximation will see ±1 input cell
  behavior change on non-divisible cases.
- **LpPool {1,2}d** ✓ Phase 16.2. Bespoke fused kernel —
  `y = (Σ|x|^p)^(1/p)` over the window in one launch. FW + BW × 4
  dtypes. `p == 1` simplifies naturally; `p == ∞` rejected (use
  MaxPool). **Breaking**: descriptor gained `ceil_mode: bool` field.
- **FractionalMaxPool {2,3}d** ✓ Phase 16.3. Bespoke kernel with
  caller-supplied `random_samples: TensorRef<f32, 3>`
  (`[N, C, num_axes]`); no internal RNG state. Window placement
  formula: evenly-spaced base + α perturbation. **Documented
  divergence** from PyTorch's exact `start_index/end_index`
  derivation — bit-exact PyTorch match is a future item if needed.

Carry-forward from Phase 16:
- FractionalMaxPool exact PyTorch formula (current is approximation).
- LpPool 3d (current scope was 1d/2d only).

## Phase 17 — SDPA / attention completion (complete; shipped alpha.34)

- **Flash SDPA sm_89 strided FW sibling** ✓ Phase 17.1. Rewrote the
  hardcoded contiguous addressing (`bh_off * q_len * d_k`) to take
  caller-provided per-dim strides. Touches 5 offset lines + Q-load
  loop + Y-finalize loop + 4 `cp.async` call sites. 2 new FFI
  symbols (f16 + bf16). GQA broadcast works via strides.
  `head_dim` must stay `stride==1` (SMEM tile layout). LSE output
  stays contig (BW routes through sm_80 baseline).
- **SDPA backward + GQA broadcast** ✓ Phase 17.2. Lifted the Phase
  14.4 `Error::Unsupported` rejection. Added `template <bool
  gqa_broadcast>` to `sdpa_dV_strided_kernel` + `sdpa_dK_strided_
  kernel`; host launcher dispatches based on stride detection;
  `if constexpr` routes to `baracuda::atomic::add<T>` (Phase 11.3
  helper) on broadcast. 0 new FFI symbols. Caller MUST pre-zero
  dK/dV under broadcast.

Carry-forward from Phase 17:

- **Flash SDPA sm_89 BW strided** — sm_89 plan remains FW-only
  (BW routes through sm_80 baseline). Future work would add an
  sm_89 Flash BW kernel using `cp.async` + tensor cores.
- **Flash SDPA sm_89 mask support** — sm_89 FW doesn't accept a
  mask at all (in contig or strided form). Separate feature.
- **Paged FlashAttention** — original Phase 6 milestone never
  landed. Separate design effort.

## Phase 18 — sub-byte + quantized completeness (complete; shipped alpha.35)

- **f16 / bf16 activations for `GgufMmvqPlan`** ✓ Phase 18.1. All
  11 block formats × 2 new activation dtypes × contig + strided =
  44 new FFI symbols. Dst dtype matches activation (PyTorch
  convention); f32 accumulator. Type-0/1 formats use templated
  `dequantize_mul_mat_vec<ActT, DstT, BlockT>`; k-quants got
  per-format mechanical rewrites matching the Phase 14.5 actstrided
  pattern. Existing f32-only FFI untouched.

Carry-forward from Phase 18:

- **MMVQ multi-dim activation strides** — Phase 14.5 chose a single
  `stride_y: i64` model (MMVQ is rank-1 at the kernel surface).
  Still no signal from Fuel that multi-dim FFI strides are needed.
  Defer until concrete use case.
- **Mixed-dtype paths** (f16 activation → f32 dst, or vice versa) —
  not implemented; callers can post-cast if they need the alternative.
  Output dtype always matches activation dtype.

## Phase 19 — Fuel retirement asks: pool/conv FFI facade + im2col (complete; shipped alpha.36)

Supersedes the original "segment + embedding BW" plan in this slot —
those move to Phase 21+. Phase 19 addresses items 1 + 2 + 3
from Fuel's 2026-05-25 comprehensive retirement ask.

- **Non-adaptive Pool FFI surface** (Item 1) — Avg/MaxPool 1/2/3d
  exist as cuDNN-backed Rust plans (Phase 7 + 11.8) but have NO
  `baracuda-kernels-sys` FFI symbols. Add thin C-ABI wrappers
  exposing `(kernel_size, stride, padding)` params; route through
  the existing cuDNN handle setup.
- **Conv FFI surface** (Item 2) — same gap for Conv1/2/3d +
  ConvTranspose1/2/3d (Phase 11.7) and Upsample (Nearest2d +
  Bilinear2d from `InterpolatePlan`). Add FFI wrappers.
- **NEW im2col / im2col1d / col2im1d** (Item 2) — baracuda has
  none today; bespoke kernels needed. Fuel uses these for the
  conv-via-im2col-and-GEMM fallback lowering + the conv backward
  filter-gradient path.
- **Vendor Fuel's Q8_1 staging kernels for inspection** (Item 3) —
  copy `fuel-cuda-kernels/src/quantized.cu` into
  `crates/baracuda-kernels-sys/vendor/fuel-q8_1/` before Fuel
  deletes it. Not built; inspection-only. Evaluate against
  baracuda's existing MMVQ for shape-specific perf wins.
  Companion task: actually do the comparison; if a Q8_1-staging
  inner-loop or SMEM-tile structure beats the current MMVQ
  implementation at some shape class, port it into the
  corresponding `baracuda_kernels_mmvq_<qtype>_run` kernel.

## Design correction surfaced during Phase 19

The Phase 19 recon revealed a meta-architectural flaw: **only
bespoke kernels in `baracuda-kernels-sys` get the FFI facade
treatment.** Plans backed by NVIDIA libraries (cuDNN, cuBLAS,
cuSOLVER, cuFFT, cuRAND, cuSPARSE, cuTENSOR, NPP, CV-CUDA) only
exist as Rust plans — there's no `baracuda-kernels-sys` FFI symbol
a caller can use directly. This breaks baracuda's "single unified
CUDA-stack facade" premise.

Phase 19 closes the gap for the Fuel-blocking subset (pool +
conv + upsample). **Pre-1.0 freeze task**: audit every other
library-backed Rust plan and add the corresponding
`baracuda-kernels-sys` FFI wrapper. Rough inventory:

- **cuSOLVER-backed**: Cholesky, LU, QR (real + complex, batched +
  non-batched), SVD (real + complex, batched + non-batched), eigh,
  eig, lstsq, solve, inverse, ormqr (real ships; complex pending).
- **cuFFT-backed**: fft 1d/2d/3d, fftshift.
- **cuRAND-backed**: random sampling (uniform, normal, gamma, etc.).
- **cuSPARSE-backed**: sparse ops (limited scope today).
- **cuTENSOR-backed**: einsum-style ops (if any landed).
- **NPP-backed**: image transforms (if any landed beyond what
  shipped via bespoke).
- **CV-CUDA-backed**: image/spatial transforms (if any landed).
- **Cutlass-backed**: GEMM is the big one; `baracuda-cutlass` is
  itself an FFI-shaped crate but the unified `baracuda-kernels-sys`
  re-export surface is missing.

This is mechanical work but substantial in volume (rough estimate:
40-60 new FFI wrappers across the library-backed families). Tracked
as a 1.0-freeze prerequisite — no caller should have to wonder
which crate hosts a given kernel.

## Phase 20 — Fuel retirement asks: MoE (planned, alpha.37)

Item 4 from Fuel's 2026-05-25 ask. Ship **both** Option 1 and
Option 2 together.

- **Option 1: batched MMVQ × N-experts** — new kernel family with
  a single launch processing all (token, expert) pairs via a
  routing triple `(sorted_token_ids, expert_offsets, topk_weights)`.
  Mirror the existing MMVQ matrix: 11 GGUF block formats + FP
  variants for f16/bf16. ~22+ new FFI symbols.
- **Option 2: refresh + expose existing MoE kernels** — baracuda's
  `MoeVariant::{ScalarGguf, Wmma, WmmaGguf}` (Phase 8.5) was
  originally vendored from Fuel's `moe/*.cu` files. Refresh
  against Fuel's current state (in case improvements landed since
  the vendor), and add direct `baracuda-kernels-sys` FFI surface
  so Fuel can call them without going through the Rust plan layer
  (consistent with the Phase 19 FFI facade work).

Once Phase 20 ships, Fuel's `fuel-cuda-kernels` crate retires.

## Phase 21 — segment + embedding BW completion (deferred from previous Phase 19 slot)

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

## Phase 22 — linalg completion

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
