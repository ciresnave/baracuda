# Baracuda backlog audit — 2026-08-27

A point-in-time inventory of **everything on the roadmap that is not yet completed**,
extracted from `ROADMAP.md` (the live backlog), `OP-MATRIX.md` (per-op status), and
`docs/planning/foundational/12-ir-expansion-roadmap.md`. Completed/shipped work is
excluded. This is a **dated snapshot** — `ROADMAP.md` remains the authoritative source;
re-audit against it rather than trusting this file once it ages.

~95 distinct incomplete items. The shape: the bulk of recent high-velocity work is
**attention** and **quantization**, each shipped as a "Tier-1 trailblazer" with a
consistent tail of Tier-2 deferrals (backward passes, GQA broadcast, paged-KV, bf16).
The second-largest theme is **distributed/training-time** work, where nearly everything
multi-rank is blocked on not having 2+ GPUs on the dev box. A third is **hardware-gated
forward-compat** (sm_90a Hopper / Blackwell), uniformly parked post-1.0. Finally there is
a cluster of genuine **1.0-freeze gates** (benchmark suite, API freeze, cuDNN/cuFFT
strided siblings). Recurring deferral reasons: *Tier-2 cadence / trailblazer-only*,
*Fuel hasn't asked / needs a v1 consumer*, *hardware-blocked*, *training-time only*,
*upstream limitation*.

## ⚠ Reconciliation flags (the doc set disagrees with itself — verify before acting)

1. **Stale release version.** `ROADMAP.md` header + `OP-MATRIX.md` both say the current
   release is **v0.0.1-alpha.77 (2026-07-09)**. The workspace is at alpha.79+. The
   headers are ~1.5 months stale; several "planned" phase headings below are already
   shipped as a result — a refresh pass is warranted.
2. **OP-MATRIX ⟂ ROADMAP conflicts (do NOT treat as open work):** OP-MATRIX lists two
   items as still-deferred that ROADMAP marks CLOSED — (a) the `_can_implement`-companion
   -per-`_run` fanout (ROADMAP: closed in the alpha.64 prep cycle), (b) strided FFI
   siblings for normalizer/shape ops (ROADMAP: closed Phase 72, 88 symbols). Also
   OP-MATRIX's Pooling header still says FractionalMaxPool/LpPool are stubbed but its own
   table rows show them shipped (Phase 16.x); and ROADMAP's "Phase 20 MoE (planned)" /
   "Phase 22 linalg (complex ORMQR)" headings are stale (both shipped).
3. **One real conflict to verify (a correctness claim):** `CtcLoss` backward
   γ-accumulation — OP-MATRIX says it is an **open** correctness bug (smoke-tested only);
   ROADMAP says it was **fixed 2026-05-16**. These disagree and need a real check; not
   resolved here (a correctness claim should not be guessed at).

---

## A. IR / Kernelgen

The ramp (increments 0–8: vocabulary → multi-output → RowReduce roles → layout/shape →
gather → scatter → scan → window → sort) is **shipped in alpha.68–77**. What remains is
by-design out-of-scope coverage + dispatch machinery.

1. **Dispatch-table population via measurement (v2 / "item 08").** The on-device bench
   gate is v1; the **Fuel `dispatch_record` feed (v2)** populator is not yet landed.
   Gating: waits on the Fuel dispatch_record feed.
2. **TILED_TC class (16 ops) — out of scope for the generator.** Flash/FA2/FlashInfer,
   int8/fp8 tensor-core GEMM: MMA-scheduling wins, not coverage. The generator provides
   only the decomposed no-cliff fallback; absorbing tile/MMA scheduling is a separate
   bench-gated program (`AccumSpec` reservation exists).
3. **SUBBYTE_DTYPE class (15 ops) — out of scope.** GGUF block-quant AoS, nibble packing,
   fp8 storage — storage-format codecs, kept bespoke.
4. **CROSS_ROW_STATE class (7 ops) — out of scope.** Mamba scans, ring attention, ormqr
   recurrences — kernel-launch-spanning / matrix-state recurrences.
5. **DATA_DEP_CONTROL class (3 ops) — out of scope.** `nonzero`, `unique_consecutive`,
   `nms` (data-dependent output cardinality).
6. **HOST_LOGIC class (2 ops) — out of scope.** cuFFT, mHC.
7. **In-kernel RNG — out of scope.** Philox-in-kernel sampling stays vendored.
8. **GEMM layout-vocabulary unification.** `DenseGemmPlan`'s `Crr` layout has no
   `LayoutSku` equivalent; unifying the two vocabularies is on the layout-planner backlog.

## B. Attention (largest cluster — ~25 items)

Nearly every family ships FW-only Tier-1 and defers BW + GQA-broadcast + paged-KV.

**Ring Attention (Phase 56):** 9. BW pass (FW saves `lse`, "mechanical"). 10. f32/f64
dtypes. 11. head_dim ≠ 128. 12. GQA broadcast. 13. Arbitrary additive mask.
14. Striped Attention (causal load-balance). 15. Multi-rank correctness validation —
hardware-blocked. 16. Ring × Megatron-TP composition.

**Arbitrary-mask FW SDPA (Phase 51):** 17. BW pass (training-time). 18. GQA broadcast on
arbmask. 19. Paged-KV + arbmask ("lands with the FlashInfer cherry-pick").

**Block-Sparse Attention (Phase 54):** 20. BW pass. 21. GQA broadcast. 22. Paged-KV.

**FlashInfer (Phase 46):** 23. **Paged-decode launcher (`BatchPagedDecodePlan`)** — Rust
plan + FFI + headers in place, but the `.cu` launcher TU is **build-excluded** on an MSVC
nvcc template-deduction issue (per-thread-default-stream overload conflict at
`cudaLaunchKernel`). 24. CascadeAttention many-way `MergeStates` — FFI-exposed,
plan-wrapping deferred.

**FA2 (Phase 59b/60):** 25. head_dim {160,224,512} **BW** — structurally unsupported
upstream; callers fall back to the bespoke 3-kernel SDPA BW. 26. head_dim outside
{32,64,96,128,192,256} — upstream v2.8.3 limitation. 27. FA3 / Hopper sm_90a — hardware.
28. Paged-KV FA2 (→ FlashInfer). 29. Split-KV by seq-length ("defer unless Fuel asks").
30. PagedAttention BW ("ask if needed").

**sm_89 Flash SDPA (Phase 17 carry):** 31. sm_89 Flash BW strided (routes through sm_80
baseline today). 32. sm_89 FW mask support (accepts no mask today). 33. Reference
`SdpaPlan` BW + GQA broadcast (still `Error::Unsupported` per OP-MATRIX — reconcile vs
ROADMAP's Phase 17.2 note).

## C. Quantization (~12 items)

**Marlin/AWQ/GPTQ (Phase 48):** 34. Marlin v2 / Sparse-Marlin (2:4). 35. Marlin bf16
(upstream fp16-magic-number). 36. Marlin sm_90 (WGMMA rewrite). 37. AWQ GEMV batch=1
path. 38. AWQ bf16. 39. GPTQ act_order=True (rejected w/ clear error today). 40.
Strict-fidelity Marlin intra-fragment permutation tables (`_perm`/`_scale_perm`).

**bitsandbytes NF4 (Phase 53):** 41. FP4 format (needs a consumer). 42. Double
quantization of scales.

**MMVQ/GGUF (Phase 18):** 43. MMVQ multi-dim activation strides (waits on Fuel). 44.
MMVQ mixed-dtype paths. 45. Stale release-mode test flaw
(`mmvq_w_offset_alignment_misaligned_rejected_debug` — guard is `#[cfg(debug_assertions)]`).

## D. Distributed / Megatron / NCCL (~16 items — mostly hardware-blocked)

**Megatron TP (Phase 57):** 46. Bias-add inside TP plans (rejected w/ Tier-2 marker
today). 47. Async overlap (Hopper TMA) — hardware. 48. Pipeline parallelism. 49.
VocabParallelEmbedding. 50. Expert parallelism (MoE).

**NCCL (Phase 52):** 51. NCCL plugins. 52. NVSHMEM integration. 53. Windows NCCL physical
link (build.rs doesn't probe). 54. Multi-rank NCCL correctness validation — hardware.

**DistributedAdam / ZeRO (Phase 58):** 55. ZeRO-2 (gradient sharding). 56. ZeRO-3
(parameter sharding). 57. DistributedLamb/SGD. 58. CPU-offload optimizer state. 59. 8-bit
distributed state. 60. Per-tensor broadcast fallback for ragged shards. 61. Multi-rank
correctness validation — hardware.

## E. Optimizer (Phase 49)

62. AdaFactor / Sophia / Lion (~150 LOC each). 63. LAMB f16/bf16 (f32-only today). 64.
8-bit optimizer state (bitsandbytes Adam8bit/Lion8bit).

## F. Normalization

65. **f64 in-place SMEM-staged normalizers** (RMSNorm/LayerNorm/Softmax/LogSoftmax) —
f32/f16/bf16 done; f64 falls back to a non-in-place-safe legacy kernel. Needs
`block_reduce_*_f64` + per-kernel `double` specialization (~1 day). "Fuel hasn't asked."
(BN/GN/IN already f64-in-place-safe.)

## G. SSD / Mamba (Phase 50/50b)

66. Complex selective_scan (no shipping consumer). 67. Variable-length sequences
(`cu_seqlens`). 68. Paged SSM state. 69. Mamba-2 chunk-aware perf kernel (perf-only,
bit-identical). 70. Hybrid Mamba+Attention (Jamba, Zamba — caller orchestration).

## H. Sparse / MoE

71. **2:4 sparse tensor-core perf backend** (`mma.sp.sync` / cuSPARSELt) — Tier-1 is
inflate-then-dense (correctness-first, not faster than dense cuBLAS). 72. Expert
parallelism (also §D). 73. Phase 20 Option 2 — refresh + FFI-expose existing MoE kernels
(`MoeVariant::{ScalarGguf,Wmma,WmmaGguf}`) — verify if shipped.

## I. Loss (Phase 47 FLCE)

74–78. `label_smoothing`, `lse_square_scale`, `softcap`, `ce_weight` (per-class),
`return_z_loss` — all "mechanical fanout" through the same kernel. 79. **CtcLoss BW
γ-accumulation** — open per OP-MATRIX / fixed per ROADMAP; see reconciliation flag #3. 80.
CTC bespoke flake under parallel tests (`cudnn_ctc_f32_uniform_t2_c2` — cuDNN handle
contention; test-infra, not correctness).

## J. Shape / Layout / Pad / Repeat

81. `PadPlan` bf16 for Reflect/Replicate/Circular (f16∪f32∪f64 only today). 82.
`RepeatPlan` bf16 (`{f32,f16,f64}` only).

## K. Pooling

83. FractionalMaxPool exact-PyTorch formula (documented divergence today). 84. LpPool 3d
(1d/2d only today).

## L. Convolution (1.0 must-have)

85. **Conv/Pool strided siblings via cuDNN (NHWC fast path).** Plumb `TensorRef` strides
into `cudnnSetTensorNdDescriptor` to avoid forced `Contiguize`; NHWC is often *faster* on
tensor cores. Descriptor-builder work; kernels unchanged. Bar: demonstrated need.

## M. FFT (1.0 must-have)

86. **cuFFT advanced-layout strided sibling** (`istride`/`idist`/`ostride`/`odist`) — FFT
a non-contiguous slice without packing. Same descriptor pattern as the cuDNN work.

## N. Linalg

87. **cuSPARSE / cuTENSOR / NPP / CV-CUDA FFI facades** — the "every library-backed Rust
plan needs a flat C-ABI `-sys` symbol" 1.0-freeze task. Conditionally deferred: no
`baracuda-kernels` plan currently wraps them; the facade lands if a plan grows. (Safe
wrappers already exist as the API for non-Rust callers.)

## O. TransformerEngine (Phase 55 out-of-scope)

88. `comm_gemm_overlap` / `nvshmem_api` — Hopper, hardware. 89. `hadamard_transform`,
`newton_schulz`, `swizzle`, `permutation` — TE utility kernels, no consumer. 90.
`fused_router` (considered covered by Phase 8+20 MoE).

---

## Long-arc / 1.0-freeze items

### 1.0-freeze gates (pre-1.0 must-haves)

91. **Cross-implementation benchmark suite vs PyTorch/cuDNN/cuBLAS.** Extend the Phase 29
    criterion+CUDA-event harness from ~10 ops to the full ~120-op matrix + land a PyTorch
    comparison path (candidates: in-proc `tch-rs`, frozen on-disk refs, out-of-proc Python
    harness). Publish per-release perf-vs-baseline in `BENCHMARKS.md`. 1.0 needs a credible
    perf story.
92. **API freeze + 1.0 stability review.** Audit every `pub` surface; document
    semver/FFI breaking-change policy; **resolve the `T: Element` vs `T: DeviceRepr + Copy`
    trait-bound split**; then cut `0.1.0-beta.0` → `0.1.0`.
    (Items 85 Conv/Pool and 86 cuFFT strided siblings are also pre-1.0 must-haves.)

### Pre-1.0 nice-to-haves

93. FlashInfer direct-FFI smoke tests ("marginal value"). 94. **Automatic layout planner**
    (layout-for-next-op: zero-copy logical reorder vs physical `PermutePlan`; prereq — each
    `Plan` exposes a preferred-layout hint + strided-vs-copy cost; natural home is the
    autotuner layer / a `baracuda-plan` crate; pairs with the cuDNN strided-siblings item).
    (Item 65 f64 in-place normalizers also filed here.)

### Post-1.0 / hardware-gated (will NOT block the 1.0 cut)

95. **sm_90a Hopper** (WGMMA + async tensor cores + cluster-launch) — no Hopper hardware.
96. **Blackwell sm_100+ forward-compat** — no Blackwell hardware. 97. **Sparsemax for
    extents > 1024** (multi-block/global sort pipeline) — low priority. 98. **Documentation
    lifecycle hooks** (automate README-badge bumps on release).
