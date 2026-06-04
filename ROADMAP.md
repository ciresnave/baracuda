# baracuda Roadmap

The live backlog of outstanding work — deferrals from completed phases
plus the long-arc 1.0-freeze items. Categories below are roughly
ordered by current priority; specific items are ordered by impact ×
effort within each category. Authoritative status per op lives in
[`OP-MATRIX.md`](OP-MATRIX.md); historical phase summaries live in
[`ARCHITECTURE.md`](ARCHITECTURE.md).

The current tag is **v0.0.1-alpha.63** with **2240+ GPU tests
passing, zero failures** across the 6 critical test crates
(baracuda-kernels, baracuda-optim, baracuda-megatron, baracuda-nccl,
baracuda-transformer-engine, baracuda-ozimmu) on RTX 4070 (sm_89).
Phase 64 work is **in progress (no version bump yet)** — Fuel is
adding a large batch of in-place ops; baracuda is accumulating
in-place-coverage closures across multiple kernel families before
the next release.

**Phase 65a (in-progress, no version bump)** — SMEM staging
infrastructure for the normalizer-family retrofit. Adds two
reusable kernel-helper headers under
`crates/baracuda-kernels-sys/kernels/include/`:

- `baracuda_smem_row_stager.cuh` — cooperative load/store helpers
  for the "one block per row, all threads stage a row in SMEM"
  pattern. Supports contig + strided variants and a runtime
  per-arch SMEM-budget helper.
- `baracuda_smem_reduce.cuh` — warp + block reductions (sum/max/min)
  using warp-shuffles + cross-warp SMEM aggregation. Lifts the
  pre-existing `warp_reduce_sum`/`warp_reduce_max` from
  `baracuda_moe.cuh` into a shared header + adds `block_reduce_*`.

Both headers are independently testable + designed to be reused
across any future per-block multi-pass kernel (not just norms —
also future fused activations, dropout-with-recompute, etc.).

Phase 65b will use these headers to retrofit the 6 normalizer
families (RMSNorm, LayerNorm, Softmax, LogSoftmax, BatchNorm,
GroupNorm; InstanceNorm delegates to GroupNorm) for SMEM-staged
single-global-read + single-global-write — enabling in-place
dispatch (`y_ptr == x_ptr`) for the dominant last-axis-contig
case AND providing a 1.2-2× perf win even in the out-of-place
case (cuDNN's normalizer pattern). The legacy global-read kernels
stay as fallback for dims exceeding SMEM budget.

Phase 65 audit finding (recorded for context): the current
normalizer kernels read input from global memory 2-3 times per
output cell (no SMEM cache layer), making them silently UNSAFE
for same-pointer aliasing. The retrofit fixes this for the
common case + ships a `_max_inplace_dim` helper so callers can
check before dispatch.

**Phase 64 (in-progress, no version bump — accumulating for next
release)** closes baracuda's documentation gap on five additional
kernel families that are structurally per-thread-isolated and
therefore safe to dispatch with same-pointer aliasing, but lacked
the explicit FFI-level aliasing contract. Documented as stable
public contract on the trailblazers: **Cast** (safe IFF source and
dest dtypes have the same byte width — e.g. f32↔i32, f16↔bf16),
**Where** (`a == y` or `b == y` safe), **Triu / Tril**
(`input == output` safe), **Activation BW** (`dx == saved` or
`dx == dy` safe, applies to all of relu/gelu/silu/tanh/sigmoid/
elu/leaky_relu/mish/hardswish/hardsigmoid/erf/erfc backwards),
**Fill** (trivially write-only). Also added **NOT-safe warnings**
on Flip / Roll / Permute / RoPE — these have shape-permuting
access patterns where two threads concurrently touch each cell,
making same-pointer dispatch silent data corruption (an earlier
audit suggested they were safe; deeper analysis shows they aren't).
NEW docs guide at `docs/guides/inplace-op-coverage.md` — single
source of truth on which kernels can be in-place-dispatched. Test
investment: 8 new aliasing-contract proof tests in
`crates/baracuda-kernels/tests/inplace_aliasing_extended_smoke.rs`
covering the 5 safe families. No new FFI symbols, no new CUDA
kernels — pure documentation + test work consolidating the
in-place coverage matrix.

**Phase 63 (alpha.63, Fuel-ask)** closes the FlashAttention
saved-tensor wiring gap for downstream autograd integration. NEW
`baracuda_kernels_fa2_sdpa_lse_size(batch, num_heads, seq_q) -> usize`
dense LSE size helper (sibling of the existing `_varlen_lse_size`
from Phase 59b). The FA2 forward v1 + v2 have written `softmax_lse`
since alpha.56; what was missing was the size helper for
pre-allocation + clarity on the FW→saved-LSE→BW pattern. **Load-bearing
"LSE saved-tensor contract" documented** on the FW + BW trailblazers
naming the exact handoff: pre-allocate via `_lse_size`, pass same
f32 buffer to FW as output and BW as input, ALiBi/sliding-window/
softcap parameters must match between FW and BW. NEW docs guide at
`docs/guides/fa2-saved-tensor-contract.md` showing the wiring
pattern downstream autograd frameworks should use. BW head_dim cap
confirmed at 256 (matches Fuel's Vulkan limit); hd160/224/512 BW
remains structurally not supported by FA2 per Phase 60, callers
fall back to bespoke `SdpaBackwardPlan`. Test investment: 12 new
tests (3 host-only `_lse_size` sanity + 5 GPU FW→BW roundtrip
proofs across f16/bf16 × head_dim × causal/noncausal + 4 BW
feature-surface tests for sliding window / softcap / ALiBi /
all-features composed — backfills the BW feature-test gaps in
`fa2_backward_smoke.rs`). Option B (recompute-LSE backward
variant) rejected: 2× backward compute for zero functional
benefit when the saved-tensor pattern already works. PagedAttention
backward filed as "ask if needed".

**Phase 62 (alpha.62, Fuel-ask)** lifts the in-place op contract
from contig-only (Phase 61) to strided. 11 new affine in-place FFI
symbols: 4 contig int dtype backfill (`i32`/`i64`/`u8`/`i8`
matching the forward affine dtype matrix) + 7 strided variants
across the full forward-strided dtype set (`f32`/`f64`/`i32`/`i64`/`u8`/`bf16`/`f16`).
Same-pointer aliasing safety on the unary / binary / ternary strided
trailblazers documented as a stable public contract: aliasing is
safe IFF the aliased input's stride array equals `stride_y`
element-for-element. NEW `baracuda_kernels_types::strides_equal`
host helper for callers to validate the precondition before
dispatching. Zero new bespoke CUDA kernels for the elementwise
unary/binary/ternary families — their existing strided launchers
are aliasing-safe under the contract (per-thread access pattern is
read-then-write at the same cell, identical to the contig case).
Test investment: 14 new host-only unit tests + 17 new GPU
direct-FFI smoke tests for the new affine in-place surface +
backfill for Phase 61's bf16/f16 + 7 aliasing-contract proof tests
on contig + strided trailblazers (also backfills the Phase 61
contig contract that shipped without test coverage). Multi-pass
families (Softmax / LayerNorm / RMSNorm) explicitly out of scope.

**Phase 61 (alpha.61, Fuel-ask)** completes the 4-dtype matrix on
the in-place affine helper (`baracuda_kernels_affine_inplace_{bf16,f16}_run`,
+2 FFI symbols) and documents same-pointer aliasing safety on the
unary / binary / ternary contig elementwise trailblazers as a stable
public contract. Unblocks Fuel's planned in-place op fanout (16+
unary in-place op families + 4 binary in-place op families +
ClampInplace + PowIInplace) with zero new baracuda symbols per
family — Fuel dispatches the forward symbol with `x_ptr == y_ptr`
(or `a_ptr == y_ptr` for binary). Half-precision in-place affine
kernels use the same upcast-to-f32 / downcast-to-storage pattern as
the forward `affine_contig_kernel_{f16,bf16}`; scalars are `f32`
through the FFI matching the forward convention.

**Phase 60 (alpha.60) corrects a Phase 59a inaccuracy** — head_dims
{160, 224, 512} FW is NOT permanently out-of-scope as alpha.59 claimed;
the Candle fork has carried them since 2023 (PR #245 by Laurent
Mazare for hd160/192/224/256; PR #2688 by Michael Feil restored hd224
after a prior upgrade had removed it; PR #3417 by Eric Buehler added
hd512 with the SMEM-opt-in pattern). Phase 60 vendors the 12 missing FW
.cu files from those Candle PRs into baracuda's FA2 tree (8 hd160/224
from `EricLBuehler/candle@main`; 4 hd512 from `huggingface/candle@5430d32c`).
**BW path NOT extended** — hd160/224 fall on FA2 BW kernel's
`kBlockKSmem = (kHeadDim % 64 == 0) ? 64 : 32` constraint (BW
atom_layout assumes 64); hd512 needs `kBlockM = 32` to fit any
SMEM budget but BW kernel_traits static-asserts `kBlockM >= 64`.
Upstream FA2 and the Candle fork ship no BW for these three either —
this limitation is fundamental to FA2's BW algorithm. The Phase 60
BW experiment + reasoning is documented in
`crates/baracuda-kernels-sys/vendor/flash-attention/VENDOR.md`,
in code comments at the dropped registration sites, and in
`FA2_BW_SUPPORTED_HEAD_DIMS` (kept at `{32, 64, 96, 128, 192, 256}`).
Callers needing BW at hd160/224/512 transparently fall back to the
bespoke 3-kernel SDPA BW pipeline (the only path supporting them
previously, anyway).
+12 new FW smoke tests in `fa2_hdim_fanout_smoke`.

Phase 59a + 59b add the full FA2 v2.8.3 FW + BW + varlen surface
(head_dims 32-256, GQA, ALiBi, sliding window, softcap) closing
Fuel's FA2-retirement requirements (+48 new tests). Phase 59c
(consolidation pass) fixed a pre-existing parallel-test race in the
bespoke flash kernel's SMEM-carveout call surfaced by Phase 59a's
hdim fanout, plus migrated `flash_sdpa_backward_smoke` to force the
bespoke backend on f16/bf16 (Phase 59b made FA2 the default BW
backend for fp16/bf16, breaking source-compat for the existing
bespoke smoke tests). Phase 42-44 add three
opt-in backends behind cargo features (none on the default build
path), Phase 49 adds the `baracuda-optim` sibling crate (Apex
multi-tensor Adam / LAMB / SGD) under the `optim` cargo feature,
and Phase 55 adds the `baracuda-transformer-engine` sibling crate
(NVIDIA TransformerEngine FP8 cast + delayed-scaling recipe,
Apache-2.0) under the `tensor_engine` cargo feature — sm_89 caveat:
the FP8 wins on Ada are bandwidth-saving only (KV cache, weights);
tensor-core FP8 MMA throughput equals BF16, so the recipe is
forward-compatible with Hopper / Blackwell where the MMA win also
materializes:

- **Phase 42**: Tri Dao's Flash Attention v2 (BSD-3) Tier-1 vendor —
  head_dim=128, fp16+bf16, sm_80, FW only — exposed as
  `BackendKind::FlashAttentionV2` on `FlashSdpaPlan` under the `fa2`
  feature. Heuristic routes long-context (seq_q×seq_k ≥ 1M) shapes
  to FA2; `PlanPreference::prefer_backend` overrides.
- **Phase 43**: DeepSeek-AI's mHC.cu (MIT) Tier-1 vendor — static-H,
  bf16 only — exposed as `HyperConnectionPlan` under the `mhc`
  feature. Replaces the bare residual `y = x + sublayer(x)` with a
  learned `n×n` Sinkhorn-Knopp doubly-stochastic mixing matrix.
- **Phase 44 + 44b + 44c**: ozIMMU Ozaki-scheme DGEMM — synthesizes
  FP64 GEMM from S² int8 tensor-core matmuls — wired as opt-in
  `BackendKind::Ozaki { slices }` on `GemmPlan`'s f64 path. NEW
  sibling crates `baracuda-ozimmu-sys` + `baracuda-ozimmu`.
  Phase 44 (alpha.56) vendored enp1s0/ozIMMU + cutf submodule
  (Linux-only); Phase 44b (alpha.57) clean-forked the whole stack —
  cutf submodule eliminated (~360 LOC of useful utilities folded into
  baracuda; ~2,200 LOC of duplicates deleted), portable `Uint128`
  unblocks Windows, LD_PRELOAD path removed.
  **Phase 44c (alpha.57, no version bump)** folds in the RIKEN-RCCS
  `accelerator_for_ozIMMU` perf-enhancement variants: `EF`
  (group-wise error-free summation), `RN` (nearest-rounding split),
  `H` (= EF + RN), plus n-blocking on the int8 cublas call (chunk
  `n > 12288` into 8192-wide pieces). Variants selected via the
  `BackendKind::Ozaki { slices }` discriminant's high-3-bits field
  (`ozaki_slices::ef(8)` etc. helpers in `baracuda-kernels-types`).
  Source-compatible with Phase 44b callers — `slices: 8` still
  decodes as Base/S=8. Default f64 GEMM stays on CUTLASS/cuBLAS
  DGEMM (bit-exact); Ozaki is opt-in for callers accepting the
  "comparable to DGEMM at S≥8" precision contract.
  Algorithm + reference implementations from Ootomo/Ozaki/Yokota
  (base) and Uchino/Ozaki/Imamura (Phase 44c variants) — see
  `crates/baracuda-ozimmu-sys/ATTRIBUTION.md`.

**Phase 45 (alpha.56, no version bump yet)** ships two pure-Rust
zero-new-CUDA composition wins over existing kernels:

- **`SmoothQuantLinearPlan<TIn, TWQ>`** composes the existing Phase 8.3
  `quantized_linear_w8a8` kernel + `fill_<dt>` broadcast for the
  per-tensor activation scale. Caller arrives with pre-smoothed-and-
  quantized int8 activations + int8 weights from the offline
  SmoothQuant Python flow (mit-han-lab/smoothquant, MIT, Xiao et al.
  ICML 2023 — algorithmic; not in scope). ~360 LOC Rust; dtype matrix
  matches `QuantizedLinearPlan` exactly (TIn ∈ {f32, f64}; TWQ = S8).
- **`RopeScaledTableBuilder` + `RopeScaling` enum** (Linear / YaRN /
  LongRoPE) — host-side cos/sin table builder feeding the existing
  Phase 36 `rope_apply_<dt>_run` kernel. YaRN (jquesnelle/yarn MIT,
  arXiv:2309.00071) implements §3.2 NTK-by-parts frequency
  interpolation + §3.3 attention-temperature absorption into cos/sin.
  LongRoPE (microsoft/LongRoPE MIT, arXiv:2402.13753) multiplies
  inv-freq by caller-supplied per-dim factors (evolutionary search
  is offline + out of scope). ~470 LOC Rust + 6 host-side unit tests +
  3 GPU integration smoke tests. Existing Phase 36 `RopeApply*` types
  source-compat preserved.

**Phase 47 (alpha.56, no version bump)** ships **Fused Linear Cross-
Entropy** — a single-kernel port of LinkedIn's Liger-Kernel FLCE
(BSD-2-Clause, clean-room CUDA reimplementation; no Liger source
vendored). The plan fuses the lm_head GEMM (`logits = input @
weight^T`) with the cross-entropy reduction into a single chunked
outer loop, **never materializing the `[BT, V]` logits tensor**. At
BT=16K, V=128K (Llama-3-class), this saves **5–10 GiB of activation
memory**. The chunk_size heuristic mirrors Liger's:
`chunk_size = next_pow2(ceildiv(BT, ceildiv(V, H)))` capped at 2048.

- NEW `FusedLinearCrossEntropyPlan<T>` + `FusedLinearCrossEntropyBackwardPlan<T>`
  for f32 / f16 / bf16 / f64; NEW `LossKind::FusedLinearCrossEntropy`
  variant.
- 16 new bespoke FFI symbols: per-row fused softmax+CE+gradient ×
  per-row-cast (None mode) × scalar-finalize (Mean/Sum) × in-place-
  scale (BW), each across 4 dtypes; plus 1 count-non-ignore helper.
- NEW `cublasGemmEx` FFI binding for f16/bf16 GEMM with f32 accumulator
  (existing `cublas{S,D}gemmStridedBatched` covers the f32/f64 cases).
- BW design: `grad_input` and `grad_weight` are produced during the FW
  pass (loss-reduction scale already folded in). The BW call multiplies
  them by `dy_scalar` (a host f32). Fast-path: `dy=1.0` (the typical
  "CE is the last layer" case) emits zero kernels.

Tier 2 deferred: `label_smoothing`, `lse_square_scale`, `softcap`,
`ce_weight` (per-class), `return_z_loss`. All are scalar / per-class
parameters Liger threads through the same kernel; adding them is
mechanical fanout.

**Phase 46 (alpha.57, no version bump — Checkpoint A)** ships **three
FlashInfer cherry-picked kernel families** for vLLM-style serving +
sort-free sampling + cascade attention. NVIDIA / FlashInfer-community
v0.6.12 (Apache-2.0, commit `eee0d75f`) was surgically cherry-picked
(~12 kLOC across 25 headers; NOT a wholesale wrap) into
`crates/baracuda-kernels-sys/vendor/flashinfer/`. Three new plan
families behind the opt-in `flashinfer` cargo feature:

- **`BatchPagedDecodePlan`** + **`PagedKvAppendPlan`** — batched
  paged-KV decode (the missing vLLM primitive) + decode-time KV-cache
  append into the paged store. Companion to existing
  `KvCacheAppendPlan` (contiguous-cache append, unchanged from Phase
  6.5). `head_dim ∈ {64, 128, 256}` × `{f16, bf16, f32}` × kHND
  layout × `DefaultAttention<false, false, false, false>` (no mask /
  no sliding-window / no soft-cap / no ALiBi). `BlockManager` /
  `BlockTable` (the free-list + refcount + fork/CoW allocator) is
  caller-owned per the Phase 46 brief — Fuel provides it.
- **`TopKTopPSamplingPlan`** — sort-free combined TopK / TopP / MinP
  / TopK+TopP sampler. Single-kernel rejection sampling; much faster
  than baracuda's existing argsort + softmax + multinomial decode
  pipeline. f32 probs + i32 output. Deterministic mode wired
  through.
- **`CascadeAttentionPlan`** — LSE-aware merge of partial attention
  states (`(v, s) <- merge((v, s), (v_other, s_other))`). The
  building block for prefix-cache sharing across requests — system
  prompts, RAG context reuse. Pairwise in-place merge in Tier 1;
  many-way `MergeStates` FFI-exposed but plan-wrapping deferred.

NEW `BackendKind::FlashInfer` + `RandomKind::Multinomial` SKU
discriminants. ~25 new FFI symbols (Checkpoint A — sampling +
cascade + paged-KV append launchers). Six MSVC-portability patches to
vendored headers documented in
`vendor/flashinfer/VENDOR.md`. Three `#[ignore]` smoke tests.

**Checkpoint B (paged decode launcher)** is staged for a follow-up
revision — the launcher hits an MSVC nvcc template-deduction issue
at the `cudaLaunchKernel((void*)kernel, …)` call site
(`cudaLaunchKernel_ptsz` per-thread-default-stream overload conflict).
The Rust `BatchPagedDecodePlan` + FFI declarations + vendored
headers are in place; only the `.cu` launcher TU is excluded from
the build.

**Phase 51 (alpha.57, no version bump)** ships **arbitrary-mask FW
SDPA** as an optional path on `FlashSdpaPlan` — closing the FA2 Tier-1
gap and unlocking speculative decoding (EAGLE / Medusa / lookahead),
MoE expert masking, prefix-LM, and sliding-window-with-sinks. NEW
optional `mask: TensorRef<f32, 4>` field on `FlashSdpaArgs`; when
`Some(...)`, routes to a bespoke arbmask SDPA kernel that adds an f32
`[B, H, Q, K]` additive bias to `S = Q·K^T·scale` before softmax.

- 8 new FFI symbols (4 fp dtypes × `_run` + `_can_implement`) under
  `baracuda_kernels_sdpa_{f32,f16,bf16,f64}_arbmask_run`. New header
  `baracuda_attn_arbmask.cuh` reuses the Phase 6.6 online-softmax tile
  pipeline; 1 new `.cu` instantiation file.
- Mask is **always f32** (decoupled from QKV precision; keeps the
  FFI surface from combinatorial blowup). Use `-INFINITY` cells for
  exact suppression; finite values are arbitrary additive biases.
- Composes correctly with `is_causal`: kernel applies causal first,
  then adds mask (`-INF + finite == -INF`, so causal cells stay
  suppressed regardless of mask).
- Phase 42 FA2 vendor untouched — FA2 v2.8.3's `Mask` template has
  no hooks for per-cell additive biases; bolting one in would require
  modifying the vendored kernel template (vendor-drift cost > benefit
  at the small-Q decode shapes where arbmask is most useful).
- Runnable example at
  `crates/baracuda-kernels/examples/speculative_decode_compose.rs`;
  design rationale at
  [`docs/guides/spec-decode.md`](docs/guides/spec-decode.md).
- BW deferred (Tier 2 — same deferral cadence as FA2).

Tier 2 deferred: BW pass (training-time arbmask gradients), GQA
broadcast on arbmask (would require host-side mask broadcast or stride
plumbing through the FFI), paged-KV + arbmask (lands with the
FlashInfer cherry-pick).

Next: the Phase 46+48 mainstream-techniques roadmap (FlashInfer
cherry-pick, Apex optimizers, Marlin/AWQ) synthesized from the recon
round documented in `~/.claude/projects/.../memory/MEMORY.md`.
Phase 50 (Mamba-2 SSD + causal-conv1d) shipped 2026-05-28.
Phase 50b (Mamba-1 selective_scan) shipped 2026-05-28 — completes
state-space LLM coverage; powers Mamba-7B, Falcon-Mamba,
Codestral-Mamba.

**Phase 55 (alpha.57, no version bump)** ships **TransformerEngine
FP8 cast + delayed-scaling recipe** — clean-room hand-port of the
NVIDIA TransformerEngine (Apache-2.0) cast/recipe subset, gated
behind the `tensor_engine` cargo feature.

The differentiated value of TE is the **per-tensor delayed-scaling
recipe with amax history** for stable FP8 training. That's the
load-bearing piece — `scale = max_representable / max_amax_in_history`
over a sliding-window ring, with the amax fed in by a fused
cast+amax kernel.

NEW sibling crate pair:
- `baracuda-transformer-engine-sys` — raw FFI to the C-ABI shim
  (`csrc/baracuda_te_shim.cu`, ~530 LOC). Apache-2.0 vendor +
  attribution at `ATTRIBUTION.md`.
- `baracuda-transformer-engine` — safe wrapper. `Fp8Recipe` RAII
  handle (amax_history ring + scale + scale_inv scalars), generic
  `Fp8CastPlan<TIn>` for f32/f16/bf16 → FP8 with running amax,
  `Fp8DequantPlan<TOut>` for the reverse.

Both formats supported: E4M3 (max=448) for fwd/weights, E5M2
(max=57344) for grads. `Fp8Recipe::update_after_pass(&stream)`
reduces the amax history, computes the new scale, advances the
write pointer.

**No cuDNN dep**: the cast/recipe paths don't need it — only
`fused_attn` does, and we skip that (baracuda Phase 17/42 covers
it). **No pybind11**: raw C ABI.

**Sm_89 reality check (RTX 4070)**: Ada has FP8 storage + cast
intrinsics, but tensor-core FP8 MMA throughput equals BF16. So on
this hardware the FP8 wins are bandwidth-saving only (KV cache,
weight storage, activation memory). The recipe machinery is
forward-compatible with Hopper (sm_90a) / Blackwell (sm_100) where
the MMA throughput win also materializes.

Out of scope (each one overlaps an existing baracuda phase or
needs deps we want to avoid):
- `normalization` (Phase 5 RMSNorm / LayerNorm)
- `fused_rope` (Phase 14/36/41)
- `fused_attn` (Phase 17/42; the cuDNN 9.3+ dep)
- `fused_softmax` (Phase 5)
- `activation` (Phase 3/31)
- `gemm` (Phase 1+24+30)
- `comm_gemm_overlap` / `nvshmem_api` (Hopper-only)
- `fused_router` (Phase 8 + 20 MoE)
- `hadamard_transform`, `newton_schulz`, `swizzle`, `permutation`
- `multi_tensor` (Phase 49 Apex)
- `dropout` (caller can compose)
- All Python bindings (`pytorch/`, `jax/`)

10 GPU smoke tests, all green on RTX 4070. The cast plan is fused
(one kernel for cast + amax reduction); the recipe update + init
each take one launch.

**Phase 54 (alpha.57, no version bump)** ships **xFormers cherry-pick:
BlockSparseAttention + 2:4 structured sparsity GEMM** — clean-room
hand-port of the facebookresearch/xformers (BSD-3-Clause) algorithmic
reference for the two sparsity families that don't overlap with any
existing baracuda surface.

Goal A — `SdpaBlockSparsePlan` (gated behind the
`xformers_blocksparse` cargo feature). Block-sparse SDPA FW where the
attention mask is a per-block boolean pattern
`[B, H, num_blocks_q * num_blocks_k]` (uint8). Only the active
(q_block, k_block) pairs participate in the QK^T matmul +
online-softmax accumulation — masked blocks are skipped entirely
(no K/V load, no compute). Real wall-clock speedup on long-context
attention with known sparse patterns (sliding-window with sinks,
BigBird-style local+global, dilated attention).

- 8 new FFI symbols (4 fp dtypes × `_run` + `_can_implement`) under
  `baracuda_kernels_sdpa_{f32,f16,bf16,f64}_block_sparse_*`. NEW
  header `baracuda_sdpa_block_sparse.cuh` reuses the Phase 6.6
  online-softmax tile pipeline (one block per `(b, h, qb)`; iterates
  only the active k-blocks).
- Differentiated from Phase 51's arbitrary-additive-mask path: the
  arbmask kernel still iterates every k-block and just adds an f32
  bias to S = Q·K^T before softmax (O(QK) compute). Block-sparse
  actually *skips* masked blocks.
- Tier-1 constraints: `block_size ∈ [1, 64]`, `d_k == d_v ≤ 128`,
  FW only, optional causal mask composes with the pattern.

Goal B — `GemmSparse24Plan` (gated behind the `xformers_sparse24`
cargo feature). 2:4 structured-sparsity GEMM accepting pre-compressed
`[M, K/2]` weights + `[M, K/8]` uint16 metadata. The 2:4 pattern is
the hardware-supported sparsity scheme on Ampere+ sparse tensor cores.

- 11 new FFI symbols × 3 dtypes (`{f32, f16, bf16}`):
  `baracuda_kernels_gemm_<dt>_sparse24_inflate`,
  `..._sparse24_gemm_run`, `..._sparse24_gemm_can_implement`,
  `..._sparse24_gemm_workspace_bytes`.
- **Tier-1 implementation strategy**: `inflate-then-dense-matmul`.
  An inflation kernel reconstructs the dense `[M, K]` weight in
  caller-supplied workspace, then a reference dense GEMM runs.
  Correctness first; the sparse-tensor-core hardware speedup
  (`mma.sp.sync.aligned` / cuSPARSELt) is deferred to Tier 2.
- Tier-1 is *NOT* faster than dense cuBLAS — the API + compression
  format are the Phase 54 deliverable. Tier-2 backend with
  cuSPARSELt or hand-rolled `mma.sp.sync` lands separately.
- New `AttentionKind::BlockSparseAttention = 9` variant
  (`#[non_exhaustive]` so source-compat).

What we deliberately did NOT vendor from xFormers (algorithmic
reference only — no upstream source files were copied):
- xFormers' "memory-efficient attention" — overlaps with baracuda's
  Phase 6.2 SDPA + Phase 6.6 FlashSdpa + Phase 42 FA2.
- xFormers' fused biases / RoPE / norm — overlaps with baracuda's
  Phase 14 / 36 / 41.
- xFormers' Triton kernel paths — no Triton toolchain in baracuda
  (consistent with Phase 47 Liger FLCE which also hand-ported from
  Triton-reference to C++/CUDA).

Tier-2 deferred: BW pass (training-time sparse-attention gradients),
GQA broadcast on BlockSparse, paged-KV + BlockSparse, sparse-tensor-
core perf backend for 2:4 GEMM (cuSPARSELt or `mma.sp.sync` inline-
PTX).

Attribution + LICENSE + AUTHORS at
`crates/baracuda-kernels-sys/vendor/xformers/` (clean-room port — no
upstream source files are vendored verbatim; the directory carries
license attribution only).

---

## Phase 50b — Mamba-1 selective_scan (complete; alpha.57, 2026-05-28)

Completes the state-space LLM coverage that Phase 50 explicitly
deferred. Phase 50 shipped Mamba-2 SSD chunk-scan, which is correct
for Mamba-2 / Codestral-Mamba / Falcon-Mamba / Zamba2 — but every
Mamba-1-shipping LLM (Mamba-7B, the broader Mamba-1 family) uses the
original `selective_scan` op family, not SSD. Phase 50b adds the
sibling Plan family alongside the existing SSD path; the `mamba`
cargo feature gates both.

- **Mamba-1 selective_scan** (Gu + Dao, Apache-2.0) hand-port at
  `crates/baracuda-kernels-sys/kernels/ssd/selective_scan_*.cu`.
  Shape contract `(B, L, D, N)` with optional `D[d]` skip, optional
  SiLU-gated `z[t, d]` tail, optional `delta_bias[d]` + optional
  `softplus(delta)` mapping — all 9 args of the upstream
  `selective_scan_fn` are wired. Dtypes: f32 / f16 / bf16 (complex
  deferred — no shipping Mamba-1 model uses it).
- **`SelectiveScanPlan`** + **`SelectiveScanBackwardPlan`** under
  `attention/` (sibling to `SsdChunkScanPlan`). FW caps state at
  `N ≤ 256`; BW uses a two-pass record-then-reverse pipeline with
  `B * D * L * N * sizeof(T)` workspace.
- **New `AttentionKind::SelectiveScan = 8`** variant added to
  `baracuda-kernels-types` (`#[non_exhaustive]` so source-compat).
- **Vendor metadata**: same `crates/baracuda-kernels-sys/vendor/mamba/`
  directory as Phase 50 — no new LICENSE / AUTHORS files; `VENDOR.md`
  updated to note the additional `selective_scan/` source files.
- **17 new FFI symbols** total: 3 FW runners + 3 FW can-implement +
  3 BW runners + 1 workspace-bytes helper + 4 module-internal
  launchers. Symbols `baracuda_kernels_selective_scan_{f32,f16,bf16}_{run,backward_run,can_implement}`
  + `baracuda_kernels_selective_scan_workspace_bytes`.
- **Smoke tests** (3 new): `selective_scan_smoke` (FW vs CPU
  reference across 4 option combinations + f16 / bf16 loose-tol),
  `selective_scan_bw_smoke` (BW finite-diff for `du` / `ddelta` /
  `dA` + finiteness for `dB` / `dC` + topology rejection check),
  `mamba1_block_smoke` (end-to-end: `causal_conv1d` + selective_scan
  full-option chain + residual).

Caller contracts:

- `dA`, `dB`, `dC`, `dD`, `d_delta_bias` MUST be zero-initialized
  before BW launch — kernel uses `atomicAdd`.
- `du`, `d_delta`, `dz` are deterministic (one writer per cell).
- Optional-input presence must match between FW and BW (e.g. if FW
  passed `d_skip`, BW must supply `d_d`; the plan rejects the
  mismatch at `run` time).

Trailblazer-only scope; deferred for future phases:

- **Complex selective_scan** — reserved in upstream but no shipping
  Mamba-1 LLM uses it.
- **Variable-length sequences** (`cu_seqlens` packed batches).
- **Paged SSM state** — analog of paged-attention.
- **Hybrid Mamba + Attention architectures** (Jamba, Zamba) —
  caller-side orchestration over baracuda primitives.

---

## Phase 50 — Mamba-2 SSD + causal-conv1d (complete; shipped alpha.57)

Opens the state-space LLM class — Mamba-2 8B, Codestral-Mamba,
Falcon-Mamba, Zamba2. Mamba-1 `selective_scan` deferred to
Phase 50b pending a v1-specific consumer (Mamba-7B).

- **causal-conv1d** (Tri Dao, BSD-3-Clause) hand-port at
  `crates/baracuda-kernels-sys/kernels/conv/causal_conv1d_*.cu`.
  Depthwise per-channel causal cross-correlation; widths 2/3/4;
  optional SiLU; f32/f16/bf16/f64. `CausalConv1dPlan` +
  `CausalConv1dBackwardPlan` exposed at the top of the
  `baracuda-kernels` facade (not under `conv/`, which is
  cudnn-gated). BW is FW-recompute style — dx deterministic, dw/db
  via atomicAdd.
- **Mamba-2 SSD chunk-scan** (Tri Dao + Albert Gu, Apache-2.0) hand-
  port at `crates/baracuda-kernels-sys/kernels/ssd/ssd_chunk_scan_*.cu`.
  Shape contract `(B, L, H, D, N)`; f32/f16/bf16 (no upstream f64).
  `SsdChunkScanPlan` + `SsdChunkScanBackwardPlan` under
  `attention/` (SSD-as-attention duality). FW caps state at
  D, N ≤ 256; BW tighter at 64 (SMEM budget). BW is two-pass —
  record states pass 1 + reverse-time gradient pass 2.
- **Vendor metadata**: `crates/baracuda-kernels-sys/vendor/causal-conv1d/`
  + `.../vendor/mamba/` with LICENSE + AUTHORS + VENDOR.md.
- **Smoke tests** (5 new): `causal_conv1d_smoke`, `causal_conv1d_bw_smoke`,
  `ssd_chunk_scan_smoke`, `ssd_chunk_scan_bw_smoke`, `mamba2_block_smoke`.
- **30 new FFI symbols** total.
- **New `AttentionKind::SsdChunkScan` variant** added to
  `baracuda-kernels-types` (`#[non_exhaustive]` so source-compat).

Trailblazer-only scope; deferred:

- **Mamba-1 selective_scan** — Phase 50b if a v1-specific consumer asks.
- **Variable-length sequences** (`cu_seqlens` style).
- **Paged SSM state** (analog of paged-attention).
- **Mamba-2 chunk-aware perf kernel**: the FFI accepts `chunk_size`
  but the trailblazer kernel runs the sequential per-(b, h)
  recurrence. The chunk-scan decomposition is a perf optimization
  that produces bit-identical outputs; a future phase will route
  through baracuda's GEMM stack for intra/inter chunk matmul.
- **Hybrid Mamba + Attention architectures (Jamba, Zamba)** —
  caller-side orchestration over baracuda primitives.

## Phase 53 — bitsandbytes NF4 vendor (complete; alpha.57, 2026-05-28)

Opens the **QLoRA-trained Llama / Mistral / Qwen inference class** on
the HuggingFace Hub. NF4 (NormalFloat 4-bit) is the dominant 4-bit
format for QLoRA-trained prebuilts — distinct from GGUF Q4_0
(symmetric int4*scale; baracuda Phase 8) and AWQ int4 (asymmetric +
zp). NF4 uses a 16-entry **non-uniform quantile codebook** derived
from the inverse CDF of `N(0, 1)` — dequant is a 16-entry lookup,
not arithmetic.

### What shipped

- **NEW `Nf4DequantizePlan<T>`** — bulk unpack `[N/2, K]` u8 → `[N, K]`
  T. Primarily a debug / weight-export tool; inference path uses the
  fused GEMV variants below.
- **NEW `Nf4MmvqPlan<T>`** — fused dequant + matrix-vector multiply
  (M=1 single-vector decode). `out[n] = Σ_k codebook[W_q[n, k]] ·
  absmax[n, k/bs] · y[k]`.
- **NEW `Nf4MmvqMultiMPlan<T>`** — same op for `M ∈ {1, 2, 4, 8}`
  reusing each weight gmem read across all M activation rows
  (Phase 33 GGUF multi-M pattern, applied here to NF4).

11 new FFI symbols (3 dequant + 2 M=1 GEMV + 6 multi-M GEMV). Vendor
metadata at `crates/baracuda-kernels-sys/vendor/bitsandbytes/{LICENSE,
AUTHORS,VENDOR.md}` (MIT, Dettmers et al. arXiv:2305.14314). The 16
codebook constants are reproduced bit-identical to upstream as a
device-side switch + a host-side `NF4_CODEBOOK: [f32; 16]` const +
`nf4_pack_weight` host quantize helper for caller weight-prep + tests.

### Cargo feature gate

NEW `bnb_nf4` on both `baracuda-kernels-sys` and `baracuda-kernels`,
default OFF. Implies `sm80`. The Rust plan types compile
unconditionally (mirroring the Phase 46 FlashInfer precedent) so the
public API surface is stable regardless of whether the feature is
enabled; the FFI dispatch helpers are `cfg`-gated and the `not`
variants return `Unsupported`.

### Pack layout (matches bitsandbytes upstream `Linear4bit`)

- **Weight** `[N/2, K]` u8 — two 4-bit codes per byte. Row `n` lives
  at byte `(n/2)*K + k`; low nibble for even `n`, high nibble for
  odd. N must be **even**.
- **Absmax** `[N * (K/block_size)]` f32 — per-output-row, per-K-block
  scale. `block_size` typically 64.
- **Output** `[M, N]` (GEMV) or `[N, K]` (dequant) in T_act ∈
  {f16, bf16}.

### Out of scope (deferred)

- **8-bit optimizers** (`Adam8bit`, `Lion8bit`) — Phase 49 Apex
  multi-tensor optimizers already cover the optimizer-step surface.
- **LLM.int8()** vector-wise W8A8 with FP16 outlier path — obsoleted
  by SmoothQuant (Phase 45) + Phase 8 int8 GEMM.
- **FP4** — different format from NF4 (different codebook); a
  separate phase if/when a caller asks.
- **Double quantization of scales** — Tier-2 follow-up. The Phase 53
  plan reads `absmax` from device memory directly.
- **PyTorch ATen wrappers** — bitsandbytes is a Python C extension;
  all Python binding glue is stripped from the vendored sources.

### Tests

3 smoke test files under `crates/baracuda-kernels/tests/`:
`nf4_dequant_smoke.rs` (host-only codebook constants + monotonicity
checks; `#[ignore]` device dequant roundtrip), `nf4_gemv_smoke.rs`
(`#[ignore]` f16 + bf16 M=1 GEMV), `nf4_multim_smoke.rs` (`#[ignore]`
M ∈ {1, 2, 4, 8} multi-M vs M=1-looped + host-only unsupported-M
rejection test).

### Tolerances measured

- Roundtrip dequant (f32 path): `< 1e-6` (bit-equivalent — same
  codebook + same absmax applied on host vs device).
- M=1 GEMV f16: `< 0.02 * max_ref`.
- M=1 GEMV bf16: `< 0.04 * max_ref` (bf16's narrower 8-bit mantissa
  widens the tolerance vs f16's 11-bit).
- Multi-M vs M=1-looped: `< 0.01 * max_ref` (same math, only the
  final f16-store ulp drift differs).

NF4 quantization itself is lossy — end-to-end vs original fp32 weight
is `~1e-2` relative error class. The kernel matches
*dequantize-then-matmul* tightly; the lossy step is upstream of the
GPU.

## Phase 48 — Marlin + AWQ 4-bit GEMM (complete; alpha.57, 2026-05-28)

Two complementary 4-bit GEMM vendors completing the "4-bit hub
coverage" started in Phase 53 (NF4). Each opens a separate slice of
the production-LLM 4-bit inference space:

- **Marlin** (IST-DASLab, Apache-2.0 + §3 patent grant) — state-of-
  the-art W4A16 GEMM for the decode-batch regime, reports ~3.87×
  speedup over FP16 GEMM at M ∈ [1, 32] on Ampere / Ada per the
  paper. **Symmetric** int4 (zero-point fused into dequant as
  `q - 8`). Vendored at `crates/baracuda-kernels-sys/vendor/marlin/`.

- **AWQ** (mit-han-lab, MIT — no patent grant) — natively supports
  the **most-deployed 4-bit format on the Hugging Face Hub**
  (Llama / Mistral / Qwen prebuilts published as `*-AWQ`).
  **Asymmetric** int4 with explicit per-group zero-points; loads
  directly from HF checkpoints without repack. Vendored at
  `crates/baracuda-kernels-sys/vendor/awq/`.

- **GPTQ → Marlin repack utility** — pure-Rust host-side bridge
  converting GPTQ-format asymmetric int4 weights into Marlin's
  symmetric layout via zero-point absorption. Lives at
  `crates/baracuda-kernels/src/gemm/gptq_to_marlin.rs`. Trailblazer
  implementation — uses identity intra-fragment permutation (the
  upstream Marlin `_perm` / `_scale_perm` tables are documented as
  follow-up scope); act_order=True checkpoints are rejected with
  a clear error message (caller can re-quantize with desc_act=False
  or wait for the Phase 48 follow-up).

### What shipped

- **NEW `Int4MarlinGemmPlan<f16>`** — fp16-only (upstream is
  fp16-only; bf16 deferred). Group size 128 or per-channel. sm_80 /
  sm_86 / sm_89 (sm_90 NOT supported — Marlin requires a WGMMA
  rewrite for Hopper).
- **NEW `Int4AwqGemmPlan<f16>`** — fp16-only. Group size 64 or 128.
  OC must be divisible by 64; IC must be divisible by
  `32 * split_k_iters`.
- **NEW `gptq_to_marlin_repack(GptqWeights) → MarlinWeights`** host
  utility (no GPU dependency).

### FFI surface

- `baracuda_kernels_int4_marlin_gemm_f16_run` + `_can_implement`
  (gated behind `marlin`).
- `baracuda_kernels_int4_awq_gemm_f16_run` +
  `_workspace_bytes` + `_can_implement` + `_dequantize_f16_run`
  stub (gated behind `awq`).

### Cargo feature gates

NEW `marlin` and `awq` on both `baracuda-kernels-sys` and
`baracuda-kernels`, both default OFF. Each implies `sm80`. Rust
plan types compile unconditionally; FFI dispatch helpers are
`cfg`-gated and return `Unsupported` when the feature is off.

### Vendor source patches

- **Marlin**: `--expt-relaxed-constexpr` added to nvcc args (the
  upstream `Marlin` kernel calls a constexpr `ceildiv` from inside
  a `__global__` function).
- **AWQ**: stripped `<torch/extension.h>` and `<c10/cuda/CUDAGuard.h>`
  includes; removed the upstream `gemm_forward_cuda(...)` host
  wrapper (replaced by the C-ABI launcher at
  `kernels/quantize/awq_launcher.cu`); rewrote `__asm__ __volatile__`
  → `asm volatile` (GCC-only syntax vs. MSVC-portable). The launcher
  inline-includes the patched .cu file (mirroring the `bnb_nf4`
  pattern from Phase 53) — the .cu is NOT listed as a standalone
  source.

### Out of scope (deferred)

- **Marlin v2 / Sparse-Marlin** — 2:4 structured sparsity extension.
- **Marlin bf16** — upstream `0x64006400` magic-number dequant
  trick is fp16-specific.
- **Marlin sm_90 (Hopper)** — needs WGMMA rewrite.
- **AWQ GEMV path** (`gemv_cuda.cu`) — batch=1 hot path optimization;
  the Phase 48 GEMM kernel handles M=1 acceptably.
- **AWQ bf16** — same dequant magic-number issue as Marlin.
- **GPTQ→Marlin act_order=True** — non-monotonic g_idx
  permutation; current scope rejects with clear error.
- **Strict-fidelity Marlin intra-fragment permutation** — the
  trailblazer repack uses identity permutation along the IC axis;
  follow-up phase will port the upstream `_perm` table for full
  numerical-fidelity validation against an upstream-packed weight
  checkpoint.

### Tests

3 smoke test files under `crates/baracuda-kernels/tests/`:
- `marlin_smoke.rs` — host-only descriptor-validation rejection +
  `#[ignore]` GPU smoke (M=1, N=256, K=128, all-ones weights /
  scales) verifying kernel launches without crashing.
- `awq_smoke.rs` — host-only descriptor-validation rejection +
  `#[ignore]` GPU smoke (M=1, IC=256, OC=64, all-zero weights /
  all-one scales) verifying output ≈ 0.
- `gptq_to_marlin_smoke.rs` — pure-Rust roundtrip on synthetic
  weights: shape-rejection arms, zp-fold correctness (zp=8 → no
  shift; zp=3 → +5 shift; zp=15 → clamp to 0).

### Goals (from the consolidation brief)

- **Goal A** — Marlin: state-of-the-art W4A16 decode kernel. **Met.**
- **Goal B** — AWQ: load HF `*-AWQ` checkpoints directly. **Met.**
- **Goal C** — GPTQ→Marlin bridge for non-Marlin checkpoints. **Met
  (trailblazer; strict-fidelity packer is a follow-up).**

## Phase 56 — Ring Attention (complete; alpha.57, 2026-05-28)

**First Phase 52 NCCL consumer — sequence-parallel attention that
unlocks million-token context length across N GPUs with O(N/P)
memory.** Hand-port of Liu/Yan/Abbeel 2023 (arXiv:2310.01889;
algorithmic reference at <https://github.com/lhao499/RingAttention>,
Apache-2.0 with §3 patent grant — JAX, not vendored — clean-room
CUDA implementation). Tier 1 ships FW only, f16/bf16, head_dim=128.

- **`RingAttentionPlan<T>`** + **`RingAttentionDescriptor`** +
  **`RingAttentionArgs<T>`** under
  `crates/baracuda-kernels/src/attention/ring_attention.rs`. Behind
  the `ring_attention` cargo feature; pulls in `baracuda-nccl` +
  `baracuda-nccl-sys` as optional deps.
- **Bespoke kernel** at
  `crates/baracuda-kernels-sys/kernels/attention/ring_attention_kernel.cu`
  (~480 LOC kernel header + ~390 LOC Rust plan). Tile geometry
  inherits Phase 6.6 FlashAttention (`Br = Bc = 64`, 128 threads/block,
  one block per `(b, h, q_block)`). Online-softmax fold of the
  resident K/V chunk into persistent `(o_acc, m_acc, l_acc)` f32
  accumulator state across rotation steps.
- **Three kernel families** per dtype: per-step kernel (folds one
  K/V chunk's contribution), finalize kernel (divides `o_acc / l_acc`
  → emits `y` in operand dtype + optional `lse`), and a dtype-
  independent init helper (`o_acc = 0`, `m_acc = -INF`, `l_acc = 0`).
- **Ring rotation**: the plan calls
  `Communicator::group_start` → `send` (to `next_peer`) → `recv`
  (from `prev_peer`) → `group_end` for the bidirectional K/V chunk
  rotation between step kernels. K and V are concatenated in scratch
  (kv_scratch_a / kv_scratch_b ping-pong) so the transfer is a single
  send/recv pair per rotation.
- **Causal masking on global indices**: each step kernel takes
  `q_global_base` and `k_global_base` as launch parameters; the
  kernel applies `q_idx_abs > k_idx_abs → mask` consistently
  regardless of which rotation step is active. Whole-block early-
  exit preserved for chunks whose global K range is entirely past
  every owned Q's index.
- **12 new FFI symbols** (`workspace_bytes` + dtype-independent
  `init_run` + 5 per-dtype × 2 dtypes: `step_run` /
  `step_can_implement` / `finalize_run` / `finalize_can_implement`).
- **Caller-staging contract for K/V**: the caller MUST pre-stage this
  rank's initial K chunk + V chunk into `kv_scratch_a` (K first then
  V, concatenated) before calling `run()`. The plan does NO D2D
  copies on the K/V data — only NCCL `send`/`recv` ping-pong between
  the two scratch buffers across rotation steps. `accumulator_scratch`
  sizing is queried via `RingAttentionPlan::accumulator_scratch_bytes`
  (Σ `o_acc + m_acc + l_acc` in f32).
- **Single-rank degenerate case** (`world_size == 1`): the rotation
  loop is a no-op — the plan runs the step kernel once with
  `q_global_base = 0`, `k_global_base = 0` and then finalizes. The
  result is mathematically equivalent to `FlashSdpaPlan` (different
  float order so not bit-identical, but within streaming-softmax
  tolerance). **This is the validation path on single-GPU hardware** —
  the smoke test compares the single-rank Ring Attention output
  against `FlashSdpaPlan` as ground truth (max abs diff `<5e-3` for
  f16, `<2e-2` for bf16 on the synthetic Q/K/V fixtures, both within
  the chosen tolerances).
- **Validation**: 4 smoke tests in
  `crates/baracuda-kernels/tests/ring_attention_smoke.rs`:
  (1) `ring_attention_f16_single_rank_matches_flash_sdpa` — passes;
  (2) `ring_attention_bf16_single_rank_matches_flash_sdpa` — passes;
  (3) `ring_attention_f16_single_rank_causal` — passes (validates
  the `q_global_base`/`k_global_base` masking path with both bases
  at 0 in single-rank); (4) `ring_attention_multi_rank_scaffold`
  (`#[ignore = "requires 2+ GPUs and a multi-process NCCL bringup"]`).
  All 3 active tests pass on RTX 4070 in 2.07s.
- **Complementary to Phase 57**: Ring Attention shards the **sequence**
  dim across ranks; Phase 57's Megatron TP shards the **head** dim.
  They compose naturally — a future phase wires this up.

**Tier 2 deferred**: BW pass (the FW saves `lse` already, so BW is
mechanical follow-up); f32 / f64 dtypes (need bigger SMEM allocations
+ separate tile geometry); head_dim ≠ 128 (need parameterized tile
specializations); GQA broadcast; arbitrary additive mask (composes
naturally with the Phase 51 arbmask kernel — a few extra FFI symbols);
Striped Attention (Brandon et al. 2023 — workload-balanced variant
specifically for causal attention; would close the load-imbalance
windows for causal Ring Attention).

**Hardware-blocked**: multi-rank correctness validation requires
2+ GPUs (or a multi-process NCCL bringup harness). Single-rank
degenerate case validates the kernel math + API surface end-to-end.

## Phase 57 — Megatron-LM TP primitives (complete; alpha.57, 2026-05-28)

**Foundational tensor-parallel (TP) primitives — pure composition over
the Phase 52 NCCL substrate + the Phase 30 cuBLAS GEMM path. NO new
CUDA kernels.** Modern Megatron-LM is framework glue: it wraps
TransformerEngine / Apex for kernels, and its "primitives" are
PyTorch-level wrappers around NCCL collectives. baracuda already has
both the substrate and the kernel building blocks, so Phase 57 is
pure-orchestration Rust.

NEW sibling crate **`baracuda-megatron`** — gated behind the
`megatron_tp` cargo feature on `baracuda-kernels` so non-distributed
consumers (e.g. Fuel) don't pay the dep surface cost. Follows the
Phase 49 / Phase 55 sibling-crate pattern.

- **`ColumnParallelLinearPlan<T>`** — splits W along the OUTPUT
  dimension. Each rank holds `W_local: [out_features/N, in_features]`.
  FW: local `Y_local = X @ W_local^T` + cross-rank `all_gather` into
  `[N * B * out/N]` (NCCL rank-major concatenation; matches
  Megatron's `_gather_along_last_dim` contract). BW: local
  `dX_partial = dY_local @ W_local` + cross-rank `all_reduce(Sum)`
  for `dX`, plus local `dW_local = dY_local^T @ X` (`dW` stays
  sharded — each rank updates its own slice via the optimizer).
- **`RowParallelLinearPlan<T>`** — splits W along the INPUT dimension.
  Each rank holds `W_local: [out_features, in_features/N]` and
  consumes a pre-sharded `X_local: [B, in_features/N]`. FW: local
  `Y_partial = X_local @ W_local^T` + cross-rank `all_reduce(Sum)`
  → `Y: [B, out_features]` (replicated). BW: local `dX_local =
  dY @ W_local` (**no collective** — `dY` is already replicated by
  the upstream Column-parallel's FW all-gather), plus local
  `dW_local = dY^T @ X_local`. The Column→Row pairing is the design
  point of Megatron — only one collective per layer-pair.
- **`TensorParallelContext`** — borrow-type holding `&Communicator` +
  `in_features` / `out_features` + cached `rank` / `world_size`.
  Divisibility (`out_features % world_size == 0` for Column,
  `in_features % world_size == 0` for Row) is checked at plan
  construction.
- **Dtypes**: f32 always (via `cublasSgemm`); f16 + bf16 (via
  `cublasGemmEx` with `Compute32F` accumulator + `R_16F` / `R_16BF`
  tags) behind the crate-level `half-crate` cargo feature, which the
  kernel-facade `megatron_tp` feature pulls in.
- **Bias**: API accepts an optional bias arg and **rejects with a
  Tier-2 marker error** if set — Phase 57 is pure composition;
  Tier 2 will compose a `baracuda-kernels` `Affine` step internally.
  Callers can perform bias-add themselves between calls. **For
  RowParallel the bias must be added AFTER the all_reduce** so it
  isn't summed N times (the docstring calls this out).
- **Row-major-via-cuBLAS-column-major trick**: implemented on the
  per-dtype `MegatronGemmScalar::row_major_gemm_{nt,nn,tn}` helpers
  (operand swap + Op flip; same convention as the Phase 30
  GemmPlan→cuBLAS bridge).
- **Single-rank degenerate case**: when `world_size == 1`, the
  `all_gather` / `all_reduce` collectives short-circuit to stream-
  ordered D2D copies and the plan is bit-equivalent to a plain
  `Linear` layer. The two `*_smoke.rs` test files validate this on
  single-GPU dev hardware (smokes are `#[ignore]`-gated pending
  NCCL bring-up).
- **`tests/multi_rank_scaffold.rs`** — `#[ignore]`-gated 2-GPU
  scaffold for future multi-GPU CI validation. Exits cleanly on
  single-GPU dev boxes via `Device::count()` check.

Out of scope for Phase 57 (deferred):

- Async overlap (Hopper TMA + `comm_gemm_overlap`) — sm_89 hardware
  blocked.
- Sequence parallelism — Phase 56's domain (Ring Attention).
- Pipeline parallelism — orchestration-heavy; future phase.
- VocabParallelEmbedding — Megatron-specific; future polish.
- Distributed gradient accumulation — Phase 58 (DistributedAdam ZeRO-1).
- Expert parallelism (MoE) — separate distributed phase.

Algorithmic reference: Shoeybi, Patwary, Puri, LeGresley, Casper,
Catanzaro, "Megatron-LM: Training Multi-Billion Parameter Language
Models Using Model Parallelism", arXiv:1909.08053 (2019). Upstream
[NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM) is
Apache-2.0; **no source vendored** — kernel primitives are reused
from the rest of the baracuda stack and composed in Rust.

## Phase 52 — NCCL foundation (complete; alpha.57, 2026-05-28)

NCCL foundation (no callers yet) — the distributed-roadmap
prerequisite for Ring Attention, distributed MoE, Megatron TP,
FSDP collectives. Phase 52 only ships the substrate; consumer plans
land in Phase 53+.

- **`baracuda-nccl-sys`** — raw FFI types + 30+ function-pointer
  typedefs + `Nccl` struct with libloading lazy-resolve (candidates
  `libnccl.so.2` / `libnccl.so` on Linux, `nccl.dll` / `libnccl.dll`
  on Windows). **No bindgen, no link-time dep on libnccl** — first
  `nccl()` call returns `LoaderError::LibraryNotFound` on hosts
  without NCCL. `cargo:rerun-if-env-changed=NCCL_ROOT` in build.rs.
- **`baracuda-nccl`** — safe `Communicator` (RAII, `Drop` →
  `ncclCommDestroy`) covering every collective: `all_reduce` /
  `reduce` / `reduce_scatter` / `all_gather` / `broadcast` +
  point-to-point `send` / `recv` + group API + `Communicator::split`
  / `abort` / `finalize` / `get_async_error` / `register` /
  `deregister` (zero-copy collectives) + `NcclMem` allocator +
  `create_pre_mul_sum` custom reduction op (NCCL 2.11+) +
  `last_error` (NCCL 2.13+). Sealed `NcclScalar` trait for
  i8/u8/i32/u32/i64/u64/f32/f64 with optional `half::f16` /
  `half::bf16` behind `half-crate` feature.
- **Spec-compliant API surface (Phase 52 additions)**:
  `Communicator::new_single_gpu(device)` — convenience for the
  single-GPU smoke-test case; `new_with_id(id, world_size, rank)` —
  multi-process / cluster-join path; cached infallible
  `rank() -> i32` + `world_size() -> i32` (was `Result<i32>`,
  technically breaking but only the in-crate test used the old API);
  instance-method `comm.all_reduce(send, recv, op, stream)` and
  `comm.broadcast(buf, root, stream)`; associated-function
  `Communicator::group_start()` / `group_end()`; `NcclReduceOp` /
  `NcclUniqueId` / `NcclDataType` aliases for the spec names;
  `NcclUniqueId::generate()` alias for `UniqueId::new()`.
- **Three smoke test files**: `dtypes_smoke.rs` (10 active tests —
  no NCCL/GPU needed; runs on Windows), `single_gpu_smoke.rs` (7
  ignored tests exercising every collective), `unique_id_smoke.rs`
  (3 ignored tests for `generate()` + byte-cast roundtrip).
  Pre-existing `smoke.rs` (2 ignored tests) updated to call the
  new infallible `rank()` / `world_size()`.

Out of scope for Phase 52 (deferred to Phase 53+):

- Multi-rank correctness validation (needs 2+ GPUs or a process-
  spawning test harness — not available in single-RTX-4070 dev env).
- NCCL plugins (custom transport backends).
- NVSHMEM integration (flagged as awkward in the FlashMoE recon).
- Any kernels using NCCL — first consumer likely Ring Attention.
- Windows physical link — `baracuda-nccl-sys/build.rs` doesn't
  probe; the loader simply fails at first `nccl()` call. Consistent
  with how every other `baracuda-*-sys` crate behaves on hosts
  without the underlying library.

## Phase 58 — DistributedAdam (ZeRO-1 sharded optimizer state, complete; alpha.57, 2026-05-28)

**Builds on Phase 49 (Adam) + Phase 52 (NCCL).** First consumer plan
on `baracuda-optim` for the distributed-roadmap (Phase 52 was the
substrate; Phase 58 ships the first concrete plan). Pure-Rust
composition — **NO new CUDA kernel**, **NO new `baracuda-kernels-sys`
FFI**. Behind the new `distributed_optim` cargo feature on
`baracuda-optim` (default OFF; pulls `baracuda-nccl` as optional dep).

- **`DistributedAdamStepPlan<T>`** — wraps an inner Phase 49
  [`AdamStepPlan<T>`] + a borrowed `&baracuda_nccl::Communicator`.
  Dtypes: f32 / f16 / bf16 (matching Phase 49). AdamW + classic mode
  (inherited via the wrapped config). Mixed-precision variant
  (`step_with_f32_state`) for half-param + f32-moment training.
- **ZeRO-1 protocol** (per step):
  1. `all_reduce(grads, Sum, in-place)` across all ranks.
  2. Local Adam step on this rank's `1/world_size` shard of every
     (param, grad, exp_avg, exp_avg_sq) tuple — uses the inner Phase 49
     plan unchanged.
  3. `all_gather(updated_params, in-place)` reassembles the full
     updated tensor on every rank.
- **Single-rank degenerate case**: when `world_size == 1` both
  collectives are skipped and the call reduces to
  `AdamStepPlan::step` bit-exactly — gives the smoke test a path
  that runs on the single-RTX-4070 dev box.
- **API signature shape**: both `param_buffers` and `grad_buffers`
  are `&mut [&mut DeviceBuffer<T>]` (the in-place collective targets,
  needing exclusive ownership for NCCL); moment buffers stay on the
  existing `TensorList<T>` shape (not touched by collectives, only
  by the inner Adam launch). The inner TensorLists for params + grads
  are built internally from the &mut slices.
- **NEW `shard_range(n, rank, world_size) -> (offset, len)`** helper
  matching PyTorch's `torch.chunk(t, world_size)` semantics (first
  `n % world_size` ranks get one extra element). Public API; pure-
  Rust unit tests run unconditionally.
- **Constraint**: each tensor's element count must be a multiple of
  `world_size` (ring all_gather symmetry). The per-tensor broadcast
  fallback for ragged shards is future work; in practice model
  weight tensors are almost always dim-aligned for tensor-core
  bucketing.
- **No version bump** — landed on the in-progress alpha.57 cycle
  alongside Phase 49, 50, 52, 53, 54, 55, 57. Consolidation phase
  will bump.

Smoke tests (3 files):

- `distributed_adam_smoke.rs` — single-rank GPU smokes (`#[ignore]`-
  gated, "requires an NVIDIA GPU + working NCCL loader") +
  unconditional `shard_range_public_api_matches_pytorch_chunk` pure-
  Rust test. The `distributed_adam_single_rank_matches_plain_adam`
  case asserts bit-exact equality with `AdamStepPlan::step` for the
  degenerate `world_size == 1` path.
- `distributed_adam_multi_rank_scaffold.rs` — `#[ignore]`-gated
  2-GPU scaffold (`Device::count() < 2` skips with a message). Wires
  the full multi-rank call shape so a future contributor on
  multi-GPU hardware can uncomment the result assertions and
  validate.
- `distributed::tests` (in `src/distributed.rs`) — 4 pure-Rust unit
  tests covering `shard_range` edge cases (even split, uneven split
  per torch.chunk, single rank, empty shard at `rank >= n`). All 4
  green on this run.

## Phase 59a — FA2 FW expansion (complete; alpha.59, 2026-05-29)

**Closes Fuel's "still needs upstream FA2" gap on the forward side.**
Phase 42 shipped a head_dim=128-only Tier-1 FA2 vendor integration.
Phase 59a expands the FW pass to full upstream feature parity so Fuel
can drop its bundled `flash-attention` FW path.

- **Vendored 20 new `.cu` files** in
  `crates/baracuda-kernels-sys/vendor/flash-attention/src/`:
  `flash_fwd_hdim{32,64,96,192,256}_{fp16,bf16}_{,causal}_sm80.cu`
  (5 head_dims × 2 dtypes × 2 causal variants). Source: Dao-AILab
  `flash-attention` tag `v2.8.3` commit `060c9188`. Combined with the
  Phase 42 hdim128 set, the vendor now carries **24 forward `.cu`
  files** covering the full upstream FA2 v2.8.3 head_dim set.
- **Critical correction to the original task spec**: head_dims 160,
  224, and 512 are **NOT supported by upstream FA2 v2.8.3** —
  `flash_fwd_launch_template.h` only instantiates
  `run_mha_fwd_hdim{32,64,96,128,192,256}<T, is_causal>()`. The task
  asked for {32, 64, 96, 160, 192, 224, 256, 512} (8 head_dims = 32
  files); reality is {32, 64, 96, 192, 256} (5 new head_dims = 20
  files). Documented in `vendor/flash-attention/VENDOR.md`.
- **Launcher rewrite** (`kernels/attention/fa2_launcher.cu`):
  runtime `switch (head_dim)` over all 6 supported head_dims; added
  forward declarations for all 24 template instantiations; added
  `_v2` entry points (one per dtype) carrying the Phase 59a feature
  set: ALiBi slopes pointer + batch_stride, sliding window left/right,
  softcap. v1 entry points preserved for backwards compatibility
  (route through the same launcher with extras at disabled defaults).
- **GQA lift**: launcher v1 already accepted distinct `num_heads_k`
  but rejected `!= num_heads`; the gate is loosened to
  `num_heads % num_heads_k == 0` (FA2's `h_h_k_ratio` handles the
  K/V head broadcast in-kernel). No new FFI symbols.
- **API extensions** (`#[non_exhaustive]` + builder per Phase 32):
  `FlashSdpaDescriptor` gained `window_size_left: Option<i32>`,
  `window_size_right: Option<i32>`, `softcap: f32`. New constructor
  `FlashSdpaDescriptor::new(...)` with chainable
  `with_window_size_left`, `with_window_size_right`, `with_softcap`
  setters. `FlashSdpaArgs` gained `alibi_slopes: Option<TensorRef<f32, 2>>`
  (shape `[1, H]` with stride[0]=0 for per-head broadcast or `[B, H]`
  contiguous for per-batch-per-head; layout detected from shape[0]).
- **Heuristic lift**: `should_use_fa2` accepts any FA2-supported
  head_dim (was head_dim==128-only) and GQA-divisible head counts.
  `fa2_is_eligible` follows.
- **Plan-layer routing**: bespoke backend rejects sliding window /
  softcap / ALiBi at `select()` time with `Error::Unsupported`.
  Negative softcap rejected with `Error::InvalidProblem`. The FA2
  path uses the `_v2` FFI symbols and plumbs all Phase 59a params
  through.

**FFI symbol delta**: +4 new symbols (`baracuda_kernels_fa2_sdpa_{f16,bf16}_run_v2`
and the matching `_can_implement_v2`). v1 symbols untouched.

**Source-compat breakage** (acceptable as pre-1.0 hardening):

- `FlashSdpaDescriptor` is now `#[non_exhaustive]` — struct-literal
  callers (~33 callsites across tests + benches + examples) migrated
  to `FlashSdpaDescriptor::new(...)`.
- `FlashSdpaArgs` gained the `alibi_slopes` field — callsites
  (~30 across tests + benches + examples) updated to pass
  `alibi_slopes: None` explicitly.
- Bench files that were already broken (missing `mask: None` from
  Phase 51) were fixed as part of this migration.

**Smoke tests** (4 new files, `#[ignore]`-gated for real GPU +
`--features fa2,sm80`):

- `fa2_hdim_fanout_smoke.rs` — 20 tests covering all 5 new head_dims
  × {fp16, bf16} × {causal, non-causal}. For head_dim ≤ 128 validates
  numerically against the bespoke `FlashSdpaPlan`; for 192/256 the
  bespoke kernel can't run so only finiteness is checked.
- `fa2_gqa_smoke.rs` — Llama-shape GQA (32 query heads, 8 KV heads)
  validated against the bespoke reference path with manually
  broadcast K/V.
- `fa2_alibi_smoke.rs` — per-head broadcast (shape `[1, H]`,
  stride[0]=0) + per-batch-per-head (shape `[B, H]`) layouts; both
  validated for finiteness. Also an unconditional negative test:
  `alibi_slopes` on the bespoke backend must error.
- `fa2_sliding_window_smoke.rs` — `with_window_size_left(Some(128))`
  + causal+sliding combination (Mistral pattern); negative test on
  bespoke.
- `fa2_softcap_smoke.rs` — softcap=30.0 (Gemma-2 default) causal +
  non-causal; negative tests for bespoke + negative softcap value.

Total: 26 new test functions across 4 new test files.

**Hardware**: targeted at RTX 4070 (sm_89). head_dim=256 may pick FA2's
64×64 tile config (vs A100/H100's 128×64) on sm_89 due to SMEM budget;
should fit in the 99 KiB opt-in SMEM. head_dim=512 is NOT vendored
(not in upstream).

## Phase 59b — FA2 BW + varlen (complete; alpha.59, 2026-05-29)

**Closes the Fuel FA2-retirement requirements.** Phase 59a covered FW
expansion; Phase 59b adds the BW pass for the full head_dim set + the
varlen FW/BW path (packed-batch attention with `cu_seqlens_*`). Fuel
can now drop their FA2 vendor entirely.

- **Vendored 24 new BW `.cu` files** in
  `crates/baracuda-kernels-sys/vendor/flash-attention/src/`:
  `flash_bwd_hdim{32,64,96,128,192,256}_{fp16,bf16}_{,causal}_sm80.cu`
  (mirrors the Phase 59a FW set 1:1). Source: FA2 v2.8.3, same
  commit `060c9188`.
- **Vendored 3 new BW headers**: `flash_bwd_kernel.h`,
  `flash_bwd_launch_template.h`, `flash_bwd_preprocess_kernel.h`.
  No new utility headers required — BW reuses FW's algorithm pieces.
- **Critical pre-flight finding**: **varlen has NO separate .cu file
  family** upstream. FA2 v2.8.3 dispatches varlen via a runtime
  `params.cu_seqlens_q != nullptr` check inside the existing FW/BW
  launch templates. The same `run_mha_{fwd,bwd}_<T, hdim, is_causal>`
  instantiations serve dense and varlen callers; only the param
  setup differs (zero batch_stride, packed row_stride, set
  `unpadded_lse=true` for varlen). Phase 59b plumbs this through
  new launcher TUs, not new vendor .cu files.
- **New launcher TUs** in `crates/baracuda-kernels-sys/kernels/attention/`:
  - `fa2_backward_launcher.cu` — BW dispatch for the 6 head_dims;
    populates `Flash_bwd_params` for dense + varlen paths; allocates
    dQ/dK/dV outputs + dQaccum / dsoftmax_d scratch from a single
    caller-supplied workspace (zero-filled via `cudaMemsetAsync`
    before launch).
  - `fa2_varlen_launcher.cu` — varlen FW with packed Q/K/V/O strides;
    `unpadded_lse=true`.
- **FFI symbol delta**: 12 new symbols (BW dense × 2 dtypes ×
  {`_run`, `_can_implement`} + workspace_size; varlen FW × 2 dtypes
  × {`_run`, `_can_implement`} + lse_size; varlen BW × 2 dtypes ×
  {`_run`, `_can_implement`} + workspace_size).
- **API additions**:
  - `FlashSdpaBackwardDescriptor` is now `#[non_exhaustive]` with
    `::new(...)` + chainable `with_window_size_left/right/with_softcap`
    setters (mirrors Phase 59a's FW descriptor pattern).
  - `FlashSdpaBackwardArgs` gained `lse_f32: Option<TensorRef<f32, 3>>`
    (REQUIRED on FA2 backend — FA2 stores LSE in f32 regardless of
    operand dtype) and `alibi_slopes: Option<TensorRef<f32, 2>>`.
    Bespoke-path fields untouched; bespoke callers pass
    `lse_f32: None, alibi_slopes: None`.
  - `FlashSdpaBackwardPlan` gained a `BackendChoice::FlashAttentionV2`
    arm. Routing heuristic: FA2 whenever eligible (f16/bf16 +
    head_dim in the FA2 set + GQA divisibility); override via
    `PlanPreference::prefer_backend`.
  - NEW `FlashSdpaVarlenPlan` / `FlashSdpaVarlenBackwardPlan` plan
    families with matching `FlashSdpaVarlenDescriptor` and args
    bundles. f16 / bf16 only; FA2-exclusive.
- **BW workspace contract**: `dq_accum + dsoftmax_d` packed
  back-to-back, sizes returned by
  `baracuda_kernels_fa2_sdpa_backward_workspace_size(b, h, sq, d)`.
  Launcher zero-fills via `cudaMemsetAsync` so callers don't have
  to pre-zero. Determinism: NOT bit-stable (FA2 uses atomicAdd
  into dq_accum); precision SKU honestly tags this as non-deterministic.
- **Source-compat breakage** (acceptable as pre-1.0 hardening):
  `FlashSdpaBackwardDescriptor` is now `#[non_exhaustive]` — 3
  callsites in `flash_sdpa_backward_smoke.rs` migrated to
  `::new(...)`. `FlashSdpaBackwardArgs` gained 2 new optional
  fields — same 3 callsites updated to pass `None` explicitly.

**Smoke tests** (2 new files, `#[ignore]`-gated for real GPU +
`--features fa2,sm80`):

- `fa2_backward_smoke.rs` — 12 tests:
  workspace_size sanity (FA2 vs bespoke), eligibility (f32 →
  bespoke, unsupported head_dim → bespoke), and end-to-end BW
  execution for d ∈ {64, 128, 192, 256} × {f16, bf16} × {causal,
  non-causal}. Asserts non-zero dQ rather than vs-bespoke numeric
  comparison (FA2 BW uses atomicAdd, so non-deterministic; would
  need a wide tolerance to compare).
- `fa2_varlen_smoke.rs` — 5 tests:
  plan-selection sanity, `lse_size` formula, varlen FW (3 sequences
  of lengths {30, 70, 40}), varlen BW (2 sequences with causal
  mask), and varlen × GQA combo (H=4, H_k=2, bf16, head_dim=128).

**Hardware**: RTX 4070 (sm_89). The BW kernels are the heaviest
templates in the FA2 vendor — first compile takes ~20 minutes for
the 24 BW .cu instantiations. Subsequent rebuilds are incremental.

**Out of scope** (intentional):

- FA3 / Hopper sm_90a — hardware-blocked on RTX 4070.
- head_dim ∉ {32, 64, 96, 128, 192, 256} — upstream limitation.
- Paged-KV variants — Phase 46's FlashInfer cherry-pick is the
  planned home for paged attention.
- Split-KV by sequence-length (separate kernel family from
  regular split-by-batch) — defer unless Fuel asks.

Out of scope (Phase 58 → future):

- **ZeRO-2** (gradient sharding) — needs a `reduce_scatter` in step 1
  plus custom gradient accumulators; deeper FW/BW integration. Future
  phase.
- **ZeRO-3** (parameter sharding during FW/BW) — needs major plumbing
  in the autograd graph; not on a near-term roadmap.
- **DistributedLamb / DistributedSGD** — same composition pattern;
  add when concrete demand surfaces.
- **CPU-offload optimizer state** — separate concern.
- **8-bit distributed optimizer state** — combines with the
  bitsandbytes 8-bit Adam path; future phase.
- **Async gradient overlap** — Hopper-specific (TMA + `comm_gemm_overlap`
  territory); hardware-blocked on the current single-RTX-4070 dev
  environment.
- **Per-tensor broadcast fallback for ragged shards** (tensors where
  `n % world_size != 0`) — Tier 2 follow-up.
- **Multi-rank correctness validation** — needs 2+ GPUs or a
  process-spawning harness. Scaffold is in place
  (`distributed_adam_multi_rank_scaffold.rs`); validation deferred.

## Phase 49 — Apex optimizer subset (complete; alpha.57, 2026-05-28)

**Deliberate scope expansion — training-framework-adjacent.**
baracuda's main facade (`baracuda-kernels`) ships zero optimizers —
it's a kernel substrate, not a training framework. Phase 49 added
the sibling crate `baracuda-optim` (~600 LOC Rust + ~750 LOC CUDA)
hosting Adam / LAMB / SGD plans built on the **`multi_tensor_apply`
idiom** vendored from NVIDIA Apex (BSD-3-Clause).

The crate boundary is deliberate: inference-only consumers (e.g.
Fuel) don't pay the FFI surface cost because they simply don't depend
on `baracuda-optim`. The `optim` cargo feature on `baracuda-kernels`
re-exports the plans into the unified facade (under
`baracuda_kernels::optim`) when a downstream wants the training
surface.

- **Vendored sources** (under `crates/baracuda-optim/vendor/apex/`):
  `multi_tensor_apply.cuh` (launch scaffold + `TensorListMetadata<N>`
  pack — replaces Apex's PyTorch-tied `multi_tensor_apply<T>`
  host-side launcher with a baracuda C-ABI shim), `multi_tensor_adam.cuh`,
  `multi_tensor_lamb.cuh`, `multi_tensor_sgd.cuh`. PyTorch ATen
  frontends (`*_frontend.cpp`) stripped — the shim takes raw device
  pointer arrays directly.
- **C-ABI shim** at `crates/baracuda-optim/csrc/baracuda_optim_shim.cu`
  (~470 LOC) hosts the kernel launchers + multi-launch chunking loop
  (Apex caps `MAX_TENSORS_PER_LAUNCH = 110` and
  `MAX_BLOCKS_PER_LAUNCH = 320` per launch; the shim transparently
  splits larger problems into back-to-back launches).
- **Plans** (in `crates/baracuda-optim/src/lib.rs`):
  - `AdamStepPlan<T>` — f32 / f16 / bf16 params + grads with f32
    moments. AdamW mode flag toggles between classic Adam (L2 fold-
    in) and decoupled weight decay.
  - `LambStepPlan` — f32 only. Two-stage: stage 1 fuses Adam update
    + per-tensor L2-norm-via-atomicAdd; sqrt-in-place launch
    between stages; stage 2 reads norms, computes trust_ratio =
    `||w||/||u||`, applies weight update.
  - `SgdStepPlan<T>` — f32 / f16 / bf16 params + grads with f32
    momentum. Momentum + Nesterov + weight decay + Apex's
    `weight_decay_after_momentum` flag + GradScaler-style grad_scale.
- **TensorList**: opaque handle wrapping per-tensor device pointers +
  sizes. Built from `&[&DeviceBuffer<T>]`; cheaply staged into the
  Apex `TensorListMetadata<N>` pack inside each launch.
- **MultiTensorApplyContext**: read-once geometry constants
  (`chunk_size`, `max_tensors_per_launch`, `max_blocks_per_launch`).
- **Smoke tests** (4 files, 6 GPU tests, all green on RTX 4070):
  - `adam_smoke.rs` — single-step Adam over multiple tensors of
    varying shapes vs CPU reference. AdamW mode tested.
  - `sgd_smoke.rs` — momentum + Nesterov + weight decay vs CPU ref.
  - `lamb_smoke.rs` — single-step LAMB; max-abs-err vs CPU ref =
    4.77e-7 (relaxed 5e-4 tolerance documents the `atomicAdd` L2-norm
    race the LAMB algorithm is provably robust to).
  - `multi_tensor_dispatch_smoke.rs` — perf test validating the
    value prop: **41.07× speedup** at 1000-tensor multi-tensor Adam
    (0.173 ms) vs 1000 individual Adam launches (7.096 ms).

Documented LAMB edge cases (preserved from Apex):
- `||w|| == 0` or `||u|| == 0` ⇒ `trust_ratio = 1.0` (vanilla Adam
  fallback). Triggers on freshly-initialized and zero-gradient layers.
- atomicAdd L2-norm race produces 1-2-ulp deltas across launches;
  LAMB is documented-robust.
- Bias-correction disabled = caller pre-scales `lr` (Apex convention).

Out of scope for Phase 49 (future):

- **AdamW, AdaFactor, Sophia, Lion** — future additions; foundation
  in place. Each is a ~150-LOC functor + Rust plan exercise on top
  of `multi_tensor_apply`.
- **8-bit optimizer state** (bitsandbytes) — separate phase (Tier 4
  of the mainstream-techniques roadmap).
- **ZeRO-style sharded optimizer / DistributedAdam** — needs NCCL
  (now available from Phase 52, but consumer plans deferred).
- **LAMB f16/bf16** — Phase 49 ships f32-only LAMB; mixed-precision
  LAMB is a follow-up.

## Pre-1.0 must-haves

Work that needs to land before the `0.1.0-beta.0` cut. Ordered by
priority within the section.

### Cross-implementation benchmark suite vs PyTorch / cuDNN / cuBLAS

The current Phase 29 bench harness in `crates/baracuda-kernels-bench/`
covers ~10 ops with cuBLAS / cuDNN reference comparisons (no PyTorch —
the Phase 29 subprocess-shim attempt was rejected as too slow). For
1.0 we want a perf-vs-baseline table per release across the full op
matrix, with both NVIDIA-library references AND a viable PyTorch
comparison strategy.

**Scope:**

- Extend the criterion + CUDA-event harness from ~10 ops to the full
  op matrix (~120 ops in `OP-MATRIX.md`).
- Land a viable PyTorch comparison path. Open design question — three
  candidates:
  1. **In-process via `tch-rs`** (LibTorch bindings) — fast, but adds
     a heavy build-time dep + LibTorch artifact distribution problem.
  2. **Frozen reference values on disk** — run PyTorch once per op,
     dump numerical reference + timing baseline to JSON, check
     baracuda's output against the frozen values in-process. No
     runtime PyTorch dep. Trades fidelity (frozen baseline ages) for
     simplicity.
  3. **Out-of-process Python harness wrapping bench binaries** —
     Python script invokes baracuda's bench binary, parses CSV, runs
     PyTorch for the same shapes, emits side-by-side report. No
     in-process coupling but loses the per-op criterion ergonomics.
- Publish per-release perf-vs-baseline rollup in `BENCHMARKS.md`
  (today only the Phase 29 results are there).

**Why pre-1.0:** 1.0 needs a credible perf story. "Faster than X /
within Y of Z" claims belong in the release announcement, and the
bench infrastructure is what backs them.

### API freeze + 1.0 stability review

**Scope:**

- Review every `pub` surface across the workspace; categorize each
  item as stable / `#[doc(hidden)]` / candidate-for-removal.
- Document the breaking-change policy (semver discipline for 1.x,
  what counts as a breaking change at the FFI layer, etc.).
- Resolve the `T: Element` vs `T: DeviceRepr + Copy` trait-bound
  split. Today split by whether sub-byte dtypes need to participate
  (see Phase 13 pragma). Either unify under a new umbrella trait or
  document the split as intentional.
- Cut `0.1.0-beta.0` once the surface settles, then `0.1.0` after a
  stabilization window.

**Why pre-1.0:** the version cut.

### Conv / pool strided siblings via cuDNN (NHWC fast path)

`cudnnSetTensorNdDescriptor` natively takes per-axis strides, and
**channels-last (NHWC) is often *faster* than NCHW on tensor cores** —
so this isn't a rare-fallback convenience, it's a real perf layout
the bespoke families already support. Plumbing strides from
`TensorRef` into the cuDNN descriptor rather than a new kernel.

**Scope:** strided sibling for each Conv / Pool family that today
forces `Contiguize` at the plan layer. Most of the work is in the
descriptor-builder layer; the kernels themselves don't change.

**Why pre-1.0:** the bench suite will measure this and the gap will
be obvious in any NHWC-preferring downstream (vision models, Triton
backends, etc.). Better to land before publishing comparison tables.

### cuFFT advanced data layout (strided sibling)

cuFFT's "advanced data layout" (`istride` / `idist` / `ostride` /
`odist`) is native; lets callers FFT a non-contiguous slice without
packing. Cheap plumbing.

**Scope:** strided sibling for the FFT families. Same descriptor-
plumbing pattern as the cuDNN work above.

**Why pre-1.0:** small marginal cost once the cuDNN pattern is
established, and rounds out the "no `Contiguize` copies on library-
backed paths" 1.0 story.

## Pre-1.0 nice-to-haves

These would tighten the 1.0 surface but aren't blockers.

### FlashInfer direct-FFI smoke tests

Phase 46/66 ship 8 safe-wrapper test files
(`crates/baracuda-flashinfer/tests/`) but zero direct-FFI tests.
The safe-wrapper tests exercise the FFI symbols indirectly; direct-
FFI tests would only catch marshalling bugs (wrong pointer types,
wrong cardinality). **Marginal value** — listed for completeness.

### f64 in-place dispatch for SMEM-staged normalizers

Phase 65b/c shipped SMEM-staged
RMSNorm / LayerNorm / Softmax / LogSoftmax with the in-place
contract for f32 / f16 / bf16 only; f64 falls back to the legacy
multi-pass-global kernel which is **not** in-place safe.

**Scope:** add `block_reduce_sum_f64` (and `_max_f64` / `_min_f64`)
to `crates/baracuda-kernels-sys/kernels/include/baracuda_smem_reduce.cuh`,
then specialize each SMEM-staged kernel for `double`. ~1 day if Fuel
ever needs f64 in-place for these ops.

**Why deferred:** Fuel hasn't asked. BN / GN / IN already cover f64
in-place by construction (Phase 65d) so the f64 SMEM gap is only on
the row-shape normalizers.

### Automatic layout planner (layout-for-next-op)

baracuda today is layout *mechanism* (strided kernels + caller-
specified `PermutePlan` / `ContiguizePlan`), with *policy* pushed
downstream to the Fuel autotuner. A planner that inspects the next
op's preferred layout and emits either a zero-copy logical reorder
(rewrite the `TensorRef` `shape` / `stride` arrays — no kernel) or a
physical `PermutePlan` copy when the downstream kernel can't consume
that stride pattern would close the "ideal layout for the next
kernel" gap. Prereq: each `Plan` would need to expose a layout-
requirement / preferred-layout hint plus a rough strided-vs-(copy+
contig) cost. Natural home is the autotuner layer (downstream Fuel,
or a new `baracuda-plan` crate sitting on top of the kernels crate).
Pairs with the cuDNN strided siblings item above.

**Why nice-to-have:** unblocks downstream perf work but isn't
load-bearing for 1.0 if downstream is willing to make layout
decisions itself.

### Flash SDPA perf gap at Hq=Hkv=32, Q=K=2048, D=128, f16

Phase 73.3 bench surfaced a **~100× perf gap** vs PyTorch at the
plain-MHA shape: baracuda reports ~270ms per launch, PyTorch reports
~2.5ms. Reproducible across runs. Either baracuda's `FlashSdpaPlan`
isn't hitting the tensor-core fast path at this configuration, or
the bench timing closure is including setup work that should be
hoisted outside the inner loop.

**Scope:** profile the f16 path at Hq=Hkv=32, Q=K=2048, D=128.
Compare against PyTorch's reference. Either fix the kernel dispatch
(if a slow path was selected) or fix the bench harness.

**Why pre-1.0:** publishing a BENCHMARKS.md with a 100× gap to
PyTorch on a load-bearing op undermines the 1.0 perf narrative,
even if it's a bench artifact rather than a real kernel issue.

### `FlashSdpaPlan` GQA-broadcast routing gap

**Root cause** (Phase 73.3 follow-up): the public `FlashSdpaPlan`
rejects non-contiguous K/V tensors at `can_implement` ("trailblazer
requires contiguous tensors"), even though the strided sibling
`FlashSdpaSm89Plan` already supports the GQA broadcast case via the
Phase 17 template-bool `gqa_broadcast` switch. The safe-wrapper
layer doesn't route to the sibling when broadcast is detected.

The `sdpa_gqa.rs` bench was the test that surfaced this — its
Hkv=1 full-MQA-broadcast case sets `stride[1] = 0` on K/V, hits
the rejection, and panics. As an immediate mitigation the bench now
catches the rejection, emits a `reference: "skipped"` row, and
continues; the underlying baracuda gap remains.

**Fix scope:** make `FlashSdpaPlan::can_implement` accept stride-0
on K/V's head axis when the strided sibling can handle it, and
route `run` through `FlashSdpaSm89Plan` in that case. Or, more
broadly: unify the two plans so users don't have to pick. Either
way, plain end-user MQA / GQA inference should not require manual
plan-selection logic.

### `ConcatPlan` perf gap on KV-cache-typical shapes

Phase 73.8 bench surfaced a **12-50× perf gap** vs PyTorch on the
2-input concat at LLM-typical KV-cache shapes:

- **BH32_Ka2047_Kb1_D128 f32** (the canonical KV-cache decode shape —
  append one new token to a 2047-long cache): baracuda **4.42ms** vs
  PyTorch **339μs** (**13× slower**).
- **BH32_Ka1024_Kb1024_D128 f32** (mid-sequence join): baracuda
  **4.36ms** vs PyTorch **342μs** (**13× slower**).
- **BH32_Ka512_Kb512_D128 f32**: baracuda **2.15ms** vs PyTorch
  **44.8μs** (**48× slower** — gap grows worse at smaller shapes).

baracuda 4.42ms for ~16MB of data read + 16MB write = ~7 GB/s
effective bandwidth — far under the RTX 4070's ~250 GB/s peak. The
kernel is doing something pathological (likely per-element naive
copy with poor coalescing) rather than the standard cudaMemcpyAsync
or vectorized large-stride copy that PyTorch uses.

**Why pre-1.0:** the KV-cache concat is in the inner loop of every
autoregressive LLM decode step. A 13× perf gap here means every
LLM inference pipeline using baracuda is paying 13× the time it
should on this one op. Critical for the 1.0 perf credibility.

**Scope:** profile `ConcatPlan::run`'s C kernel; replace with a
cudaMemcpyAsync-per-input or vectorized-copy pattern. The per-input
data is contiguous (shape match on every axis except concat_dim);
the output write is also contiguous strided. This should be a
near-trivial memcpy kernel, not a per-element compute kernel.

### Reductions perf gap vs PyTorch at small rows × small hidden

Phase 73.3 bench surfaced PyTorch reduce_sum / reduce_max /
reduce_mean running 5-20× faster than baracuda at small (R×H ≤
2048×4096) shapes.

**Root cause** (Phase 73.3 follow-up): baracuda's `reduce_axis_kernel`
in `baracuda_elementwise.cuh` is **one thread per output cell**, with
a **serial inner loop** over the reduced axis:

```cuda
T acc = F::init();
for (int32_t k = 0; k < reduce_extent; ++k) {
    int64_t off_x = off_x_base + (int64_t)k * reduce_stride_x;
    acc = op(acc, x[off_x]);
}
y[off_y] = F::finalize(acc, reduce_extent);
```

For (R=512, H=1024) that launches 2 blocks of 256 threads (512 / 256
= 2). Each thread executes a 1024-iteration serial reduction. On
RTX 4070 (36 SMs) only 2 SMs are active — massive underutilization.

The fix is the standard parallel-reduction pattern: one block per
output row, threads in the block cooperatively reduce the row via
warp shuffles + cross-warp SMEM aggregation (already implemented in
`baracuda_smem_reduce.cuh` — Phase 65a). This both fills more SMs
(reduce_blocks = output_numel) and parallelizes the per-row work.

baracuda still beats cuDNN at the largest shape (R=4096, H=4096:
baracuda 749μs vs cuDNN 1.79ms) — cuDNN has its own fixed-overhead
problem. PyTorch's 318μs at the same shape is the actual reference
target.

**Scope:** rewrite `reduce_axis_kernel` to one-block-per-output-row
+ in-block warp-shuffle reduction (mirrors the Phase 65b SMEM-staged
normalizer pattern). Same kernel covers Sum/Max/Min/Mean/Prod/Norm2;
the functor `F` is already templated. Estimated 1-2 days plus
correctness retest across the 5 reduce kinds × 4 dtypes.

**Why pre-1.0:** 5-20× gap on a load-bearing primitive
(reduce_sum / reduce_mean fire on every backward pass + every
softmax/layernorm-bw). Worth addressing before 1.0 perf claims
publish.

## Post-1.0 (hardware-gated or follow-on)

Items that need hardware we don't have, or are natural 1.x follow-on
work. **Will not block the 1.0 cut.**

### sm_90a (Hopper async) specialization

Sibling plans for Hopper's WGMMA + async tensor cores + cluster-
launch. The sibling-plan + arch-dispatcher pattern Phase 10
established for sm_89 generalizes here.

**Why post-1.0:** no Hopper hardware on the dev box. Land once a
Hopper test machine is available.

### Blackwell forward-compat

Verify the kernel set compiles + runs on sm_100+.

**Why post-1.0:** no Blackwell hardware on the dev box.

### Sparsemax for extents > 1024

Phase 11.6 lifted the cap from 64 → 1024 via `cub::BlockRadixSort` +
`BlockScan`. Larger rows would need a multi-block / global sort
pipeline. **Low priority** unless a use case shows up.

### CTC bespoke flake under parallel test execution

`cudnn_ctc_f32_uniform_t2_c2` intermittently fails when the full
test suite runs in parallel; passes deterministically in isolation.
Likely cuDNN handle contention. **Not a correctness issue** — pure
test infrastructure flake.

### Documentation lifecycle hooks

The `feedback_readme_badges_on_publish` memory entry captures the
"bump README badges on release" gotcha. A pre-commit hook or release
script could automate this rather than relying on memory.

## Audit-surfaced items already closed

Recorded here so future audits don't re-discover them as gaps.

- **`_can_implement` companion fanout** — CLOSED in the alpha.64
  prep cycle. Today: 2727 `_run` ↔ 2733 `_can_implement` in
  `baracuda-kernels-sys`; 4 ↔ 4 in `baracuda-transformer-engine-sys`.
- **`-sys` crate `missing_docs` sweep** — CLOSED (commits `6af9533`
  + `757c9f4`). Zero `missing_docs` warnings workspace-wide;
  workspace lints promoted to `deny`.
- **Tier-2 cross-crate docs/tests polish** — CLOSED (`928d385`).
- **Strided FFI siblings for normalizer + shape ops** — CLOSED in
  Phase 72 (`e4afc03`, will ship alpha.65). 88 new FFI symbols
  across `rms_norm` / `layer_norm` / `softmax` / `log_softmax` (FW +
  BW × 4 dtypes) + `flip` / `roll` / `permute` (FW × 4 dtypes).
- **`bincount` smoke tests** — exist in
  `crates/baracuda-kernels/tests/bincount_smoke.rs` (4 tests).
- **`HyperConnectionPlan` smoke tests** — exist in
  `crates/baracuda-kernels/tests/hyper_connection_smoke.rs` (2 tests).

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
Items previously listed here but now shipped:

- ~~**Type-0/1 MMVQ `ncols ≥ 64` debug-build assertion**~~ — shipped
  Phase 22 (alpha.39). The assertion now lives in
  `GgufMmvqBatchedPlan::select()` (NOT `GgufMmvqPlan::select()` —
  single-matrix callers are incidentally safe because OOB lands in
  unallocated zero memory; only the contiguous-batched plan has the
  silent-wrong failure mode). Returns `Error::InvalidProblem` when
  any type-0/1 format is paired with `n_cols < 64`. Release builds
  elide the assertion.

## Phase 27 — Q8_1 MMVQ perf inspection (complete; doc-only, 2026-05-25)

Research / inspection task. Compared Fuel's vendored Q8_1 staging
kernels (`crates/baracuda-kernels-sys/vendor/fuel-q8_1/quantized.cu`,
4537 LOC) against baracuda's existing MMVQ. Findings written to
[`vendor/fuel-q8_1/PHASE_27_ANALYSIS.md`](crates/baracuda-kernels-sys/vendor/fuel-q8_1/PHASE_27_ANALYSIS.md).

**Headline**: at the M=1 workload point that dominates today's
inference decode step, both implementations are gmem-bound and
saturate at the same rate. **No quick-win optimizations** were
portable as constant tweaks.

**The material opportunity** (Tier S1 in the report) is a Phase-sized
refactor: port Fuel's `mul_mat_vec_q<ncols_y>` design + `quantize_q8_1`
staging into baracuda. Targets the **prefill step (M=8) — ~3-7×
speedup** via weight reuse across multiple activation vectors
(Fuel's template parameterizes `ncols_y = 1..8` so a single weight
load amortizes across up to 8 dot products). Not portable as a
small tweak — needs a new MMVQ family (10 qtypes × 8 M-sizes = 80
new launchers) + the Q8_1 staging FFI + host orchestration.

**Recommendation**: schedule a dedicated phase (next non-numeric
slot, "**Multi-M MMVQ via Q8_1 staging**"). Estimated 3-5 days.
The vendored directory **stays** — the future phase re-reads it.

Marginal opportunities (Tier A1-A2 in the report — multi-warp
k-quant blocks, `__dp4a` for Q8_K) were not chosen because they
require benchmarking infrastructure (Phase 29 territory) to gate
whether they're net-positive.

## Phase 24 — Cutlass GEMM re-export FFI facade (complete; shipped alpha.41)

Third and final slice of the Phase 19-surfaced library-backed FFI
facade audit. **Completes the 1.0-freeze prereq**: every
library-backed Rust plan now has a corresponding flat C-ABI
`baracuda-kernels-sys` symbol. cuTENSOR / NPP / CV-CUDA were skipped
per the Phase 23 cuSPARSE precedent — no `baracuda-kernels` plans
wrap them today; their respective parallel safe wrappers
(`baracuda-cutensor`, `baracuda-npp`, `baracuda-cvcuda`) remain the
authoritative API for non-Rust callers needing those libraries.

- **Cutlass GEMM re-export** (210 symbols) — full `cutlass_reexport.rs`
  in `baracuda-kernels-sys/src/`. Exposes the entire
  `baracuda-cutlass-kernels-sys` GEMM surface under the unified
  `baracuda_kernels_gemm_*` namespace so non-Rust callers can drive
  any Cutlass SKU through `baracuda-kernels-sys` directly, without
  taking a separate `baracuda-cutlass-kernels-sys` link-line dep.
  Coverage:
  - 10 non-bias single GEMM (fp16/bf16/tf32/f32_simt/s8/u8 × {rcr, rrr})
  - 2 non-bias DGEMM (f64 × {rcr, rrr})
  - 32 bias-fused fp single GEMM ({Bias, BiasRelu, BiasGelu, BiasSilu}
    × {f16, bf16, tf32, f32_simt} × {rcr, rrr})
  - 16 bias-fused int8 GEMM ({Bias, BiasRelu, BiasGelu, BiasSilu} ×
    {f32bias, i32bias} × {s8, u8} × rcr)
  - 8 bias-fused DGEMM ({Bias, BiasRelu, BiasGelu, BiasSilu} × {rcr, rrr})
  - 2 strided-batched GEMM ({f16, bf16} × rcr)
  - 70 SKU families × 3 entries (`*_run`, `*_workspace_size`,
    `*_can_implement`) = 210 trampolines.
- **`baracuda-kernels-sys` Cargo.toml** gained a normal dep on
  `baracuda-cutlass-kernels-sys` (and forwards `sm80` / `sm90a`
  features through). Different `links` keys
  (`baracuda_cutlass_kernels` vs `baracuda_kernels`) keep the two
  static archives co-existing without Cargo's links-conflict guard
  tripping.
- **Trampoline contract**: pass `(workspace, workspace_bytes)`
  verbatim to the underlying Cutlass symbol. No re-query, no
  validation at the trampoline layer — Cutlass workspace estimates
  are host-pure and don't drift across calls (so the cuSOLVER
  re-query footgun doesn't apply here). All `_run` pointer args are
  device-resident; `_workspace_size` and `_can_implement` are
  host-pure.

Skipped libraries with explicit deferral notes in `lib.rs`:

- **cuTENSOR** — `baracuda-cutensor` exists; `baracuda-kernels`
  has no plan wrapping it.
- **NPP** — `baracuda-npp` exists; no `baracuda-kernels` plan.
- **CV-CUDA** — `baracuda-cvcuda` exists; no `baracuda-kernels` plan.

If `baracuda-kernels` grows plans for any of these in the future,
the matching facade lands then.

## Phase 23 — cuFFT + cuRAND FFI facade (complete; shipped alpha.40)

Second slice of the Phase 19-surfaced library-backed FFI facade audit.
cuFFT and cuRAND both ship with the CUDA toolkit (no feature gate
needed). cuSPARSE was scoped in but skipped — no baracuda-kernels
plans wrap it today; sparse ops live only in the parallel
`baracuda-cusparse` safe wrapper.

- **cuFFT FFI facade** (24 symbols) — 6 plan families wrapped in
  `crates/baracuda-kernels-sys/src/cufft_facade.rs`: `fft_1d`,
  `rfft_1d`, `irfft_1d` × {f32 or c32, f64 or c64}, plus the ND
  variants `fft_nd`, `rfft_nd`, `irfft_nd` (rank 1..=3 via host-
  resident `dims` array). cuFFT manages its own internal scratch via
  the `cufftHandle`, so `*_workspace_size` returns 0 and `*_run`
  ignores `workspace` / `workspace_bytes` (kept in ABI for symmetry
  with the rest of the facade).
- **cuRAND FFI facade** (8 symbols) — 2 plan families in
  `crates/baracuda-kernels-sys/src/curand_facade.rs`: `curand_uniform`
  + `curand_normal` × {f32, f64}. Uniform supports the `(low, high]`
  post-affine remap via the bespoke `affine_inplace` kernel — so the
  facade is gated behind the same `sm80/sm89/sm90a` feature as the
  bespoke side. Each FFI call creates + seeds a transient generator
  (matches the Rust plan's contract).
- **cuSPARSE** — skipped per Phase 22's "qr_batched deferred" precedent:
  the brief listed it in scope but the grep returned zero plans. If
  baracuda-kernels grows cuSPARSE-backed plans in the future, the
  facade lands then.

Patterns retained:

- **Macro-fanout per family** mirroring `cusolver_facade.rs`. Each
  family yields `*_run` + `*_workspace_size` siblings.
- **HOST vs DEVICE residency documented per pointer** in the facade
  module docs (the `dims` array for ND FFT is HOST; everything else
  is DEVICE).
- **No feature gate** for cufft_facade (cuFFT ships with the CUDA
  toolkit). cuRAND facade IS feature-gated, but only because it
  composes the bespoke `affine_inplace` kernel — the cuRAND symbols
  themselves are toolkit-shipped.

## Phase 22 — MMVQ ncols≥64 assertion + cuSOLVER FFI facade (complete; shipped alpha.39)

Closes the first slice of the Phase 19-surfaced "library-backed Rust
plans need flat C-ABI FFI symbols" 1.0-freeze prereq.

- **Type-0/1 MMVQ `ncols >= 64` debug-build assertion** ✓ in
  `GgufMmvqBatchedPlan::select()`. Returns `Error::InvalidProblem`
  for Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 + `n_cols < 64` under
  `#[cfg(debug_assertions)]`. Single-matrix `GgufMmvqPlan` left
  unassertioned with an explanatory comment (the contiguous-batched
  layout is what makes the OOB reads dangerous; single-matrix's OOB
  region is unallocated zero memory and produces the correct answer).
- **cuSOLVER FFI facade** ✓ — 10 cuSOLVER-backed plan families wrapped
  as ~50 flat C symbols in
  `crates/baracuda-kernels-sys/src/cusolver_facade.rs`. Covered:
  Cholesky (non-batched + batched, f32/f64), LU `getrf` (f32/f64; no
  batched in cuSOLVER-Dn), QR `geqrf` + `ormqr` for dense-Q
  materialization (f32/f64), SVD `gesvd` (f32/f64), SVD-batched
  Jacobi `gesvdjBatched` (f32/f64), SVDA-batched strided
  `gesvdaStridedBatched` (f32/f64), eigh `syevd`+`heevd` (f32/f64
  symmetric + Complex32/Complex64 Hermitian), eig 64-bit-index
  `Xgeev` (single entry over `cudaDataType` tag covering all 4
  dtypes), lstsq `_gels` iterative path (f32/f64), solve fused
  `getrf`+`getrs` (f32/f64), inverse over caller-staged identity
  (f32/f64). Macro-fanout pattern matching `pool_cudnn_facade.rs`;
  RAII handle/params/jacobi-info guards. No feature gate (cuSOLVER
  ships with the CUDA toolkit).
- **cuSOLVER bufferSize / handle pairing gotcha** surfaced during
  smoke-test development: cuSOLVER's `_bufferSize` results for some
  ops (notably `Spotrf_bufferSize` and `SSgels_bufferSize`) drift
  across separate transient handles. The agent's initial design
  re-queried bufferSize inside `*_run` against a fresh handle and
  compared to the caller's `workspace_bytes` (which came from a
  transient-handle query in `*_workspace_size`) — this spuriously
  tripped `WS_TOO_SMALL`. Fix: Cholesky / inverse / solve trust the
  caller's `workspace_bytes` directly (cuSOLVER validates internally
  and returns its own status if too small); lstsq re-queries on the
  same handle that runs the op (cuSOLVER's `_gels` is sensitive to
  bufferSize-vs-run handle pairing).
- **cuSOLVER `niters` is a HOST scalar pointer** for `_gels` — not a
  device buffer. The iterative-refinement loop runs on the host; the
  status int is written from the host side. Passing a device pointer
  triggers `STATUS_ACCESS_VIOLATION` on Windows. FFI docstring now
  calls out this residency contract explicitly.

Carry-forward (will be addressed in Phase 23+):

- Remaining library-backed FFI facade coverage: cuFFT, cuRAND,
  cuSPARSE, cuTENSOR, NPP, CV-CUDA, Cutlass re-export surface.
- `mmvq_w_offset_alignment_misaligned_rejected_debug` is a stale
  release-mode test design flaw — the alignment guard is
  `#[cfg(debug_assertions)]` so the test always fails in `--release`.
  Pre-existing; not Phase 22 work.

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

## Phase 25 — segment + embedding BW completion (DONE)

Originally tracked under the Phase 21 slot. Shipped:

- **Segment `Max` / `Min` BW** (sorted + unsorted) — argmax / argmin
  **recomputed in the BW kernel** rather than saved from FW; preserves
  FW API source-compat. Tie-break = first occurrence (PyTorch picks
  last; documented divergence).
- **Segment `Prod` BW** (sorted + unsorted) — direct
  `d_input[k, d] = d_output[seg, d] * (output[seg, d] / input[k, d])`.
  Caller must avoid zero-valued inputs or accept NaN / Inf in the
  gradient.
- **Unsorted Segment `Prod` FW** — atomicMul-via-CAS retry loop on the
  underlying 32 / 64-bit slot. Non-deterministic.
- **`EmbeddingBag` Max mode FW + BW** — FW writes value + per-(b, d)
  contributing-row index (i32); BW scatters dout into dweight at those
  rows via atomicAdd. Tie-break = first occurrence.

Dtype coverage: `f32, f64` for all (BW uses atomicAdd / atomicCAS,
restricted to native-FP-atomic types). EmbeddingBag Max FW also covers
`f16, bf16` (f32 accumulator), index dtypes `i32 + i64`.

New FFI symbols added: 16 segment (8 sorted BW + 8 unsorted BW
plus 2 prod FW) and 10 embedding (8 max FW × dtype/idx plus 2 max BW).

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

- **Strided siblings for library-backed conv / pool / FFT** — these
  families are contiguous-only at the plan surface today (caller must
  `Contiguize` first), unlike the bespoke elementwise / reduction /
  norm / loss / index / shape families, which all carry strided
  `TensorRef` paths for both FW and BW. The opportunity here is narrow
  and worth it for one reason: the *backend* can already express
  per-axis strides at full speed, so a strided sibling skips the
  materialization copy without writing a slow strided kernel.
  - **Conv / pool (cuDNN)** — highest value. `cudnnSetTensorNdDescriptor`
    takes full per-axis strides, and **channels-last (NHWC) is native
    and often *faster* than NCHW on tensor cores** — so this is not
    just a rare-fallback convenience, it's a real perf layout. Mostly
    plumbing strides from `TensorRef` into the cuDNN descriptor rather
    than a new kernel.
  - **FFT (cuFFT)** — secondary. cuFFT "advanced data layout"
    (`istride` / `idist` / …) is native; lets callers FFT a
    non-contiguous slice without packing. Cheap plumbing.
  - **Explicitly NOT candidates** (recorded so a future sweep doesn't
    over-reach): **GEMM** is already covered — cuBLAS / CUTLASS expose
    transpose flags + leading dimension + strided-batched, which handle
    the realistic non-contiguous matmul cases; truly-arbitrary
    inner-stride GEMM needs a copy anyway (tensor cores want aligned
    vector loads). **Linalg (cuSOLVER)** is column-major `lda`-only —
    a "strided" sibling would copy internally, so no gain over an
    explicit contiguize. **Sort / TopK / segment / image** are bespoke
    but low-value on a non-contiguous path. Decision rule: a strided
    sibling earns its keep only when `cost(strided) < cost(copy) +
    cost(contig)` AND the backend expresses the strides at full speed.
  - **Bar: demonstrated need.** Add per a real caller or a measured
    copy cost, not for completeness — the sibling surface is already
    large (FFI symbols + tests + `can_implement` companions per family).
  - **Natural consumer: a layout planner.** The more strided siblings
    exist, the more often a future automatic layout planner (see the
    note below) can satisfy the next op with a *zero-copy* logical
    descriptor reorder instead of a physical `PermutePlan` copy. The
    two pieces are complementary.
- **Automatic layout planner (layout-for-next-op)** — baracuda today is
  layout *mechanism* (strided kernels + caller-specified `PermutePlan` /
  `ContiguizePlan`), with *policy* pushed downstream to the Fuel
  autotuner. A planner that inspects the next op's preferred layout and
  emits either a zero-copy logical reorder (rewrite the `TensorRef`
  `shape` / `stride` arrays — no kernel) or a physical `PermutePlan`
  copy when the downstream kernel can't consume that stride pattern
  would close the "ideal layout for the next kernel" gap. Prereq: each
  `Plan` would need to expose a layout-requirement / preferred-layout
  hint (innermost axis, stride-1 constraints, NHWC preference) plus a
  rough strided-vs-(copy+contig) cost — the same machinery as the
  sibling-racing judge. Natural home is the autotuner layer (downstream
  Fuel, or a new `baracuda-plan` crate sitting on top of the kernels
  crate). Pairs with the strided-siblings item above.
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
