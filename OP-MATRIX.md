# Op Coverage Matrix

Generated for **baracuda-kernels v0.0.1-alpha.31** on **RTX 4070 (sm_89)** —
2026-05-21 sweep at 1890/0 regression across 602 binaries.

This file is the authoritative reference for "what ops are implemented and at
what dtypes / shapes / backends." Each row corresponds to a `pub struct *Plan`
in `crates/baracuda-kernels/src/`. The dispatch decision (bespoke vs. NVIDIA
library) lives behind a single Plan-based Rust surface; backend choice is an
internal detail driven by `select()`. Detailed per-op API docs are in the
rustdoc.

## Legend
- ✓ — implemented
- ✗ — not implemented
- partial — some dtypes / shapes covered; see Notes
- deferred — on the roadmap, see [comprehensive plan](~/.claude/plans/baracuda-kernels-comprehensive.md)
- N/A — backward not meaningful for this op (e.g. integer inference, indexing,
  set-valued ops, RNG)
- **Bespoke** — `.cu` kernel in `baracuda-kernels-sys`
- **Cutlass** — `baracuda-cutlass` (CUTLASS 4.2.0)
- **Cublas** / **Cudnn** / **Cusolver** / **Cufft** / **Curand** — NVIDIA library wrapper

Dtype shorthand:
- FP-family = `{f32, f16, bf16, f64}` (the four standard FP types this crate
  consumes)
- `s8`/`u8`/`s4`/`u4`/`Bin` — `IntElement` family
- `Complex32`/`Complex64` — interleaved-real-imag complex floats

---

## OpCategory: Gemm

| Op | Backend | Dtypes | Shapes / Limits | FW | BW | Notes |
|----|---------|--------|-----------------|----|----|-------|
| `GemmPlan` | Cutlass | `{f16, bf16, f32 (TF32), F32Strict (SIMT), f64}` | M/N/K ≥ 1 | ✓ | N/A (inference) | Re-export of `baracuda_cutlass::GemmPlan`. Layouts `{Rcr, Rrr}`. Full bias-family epilogue: `{Identity, Bias, BiasRelu, BiasGelu, BiasSilu}`. F64 routes to cuBLAS DGEMM. |
| `BatchedGemmPlan` | Cutlass | same as `GemmPlan` | uniform M/N/K across batch | ✓ | N/A | Same SKU surface as `GemmPlan` with a fixed batch stride. |
| `GroupedGemmPlan` | Cutlass | `{f16, bf16, f32, f64}` | per-problem M/N/K | ✓ | N/A | Variable-shape grouped GEMM. Three scheduling modes (`Device`, `Host`, `Persistent`). |
| `IntGemmPlan<T, BT>` | Bespoke (RRR) + Cutlass (RCR) | `T ∈ {s8, u8}` × `BT ∈ {f32, i32}` | M/N/K ≥ 1, 8B-aligned | ✓ | N/A | W8A8 dispatcher. RCR delegates to `baracuda-cutlass`; RRR uses `mma.sync.m16n8k32.row.col.satfinite`. Full bias-family epilogue. |
| `Fp8GemmPlan<T>` | Bespoke | `T ∈ {Fp8E4M3, Fp8E5M2}`, bias always `f32` | M/N/K ≥ 1 | ✓ | N/A | sm_89 only. Full 20-SKU matrix: `{E4M3, E5M2} × {Rcr, Rrr} × {Identity, Bias, BiasRelu, BiasGelu, BiasSilu}`. F32 accumulator, saturating-cast on store. |
| `Int4GemmPlan<T, BT>` | Bespoke | `T ∈ {s4, u4}` × `BT ∈ {f32, i32}` | K, N must be even (packed-pair) | ✓ | N/A | sm_89. Full 36-SKU matrix: `{S4, U4} × {Rcr, Rrr} × full bias-family`. Storage `ld` in bytes (= packed slots). S32 accumulator, sat-cast back to int4. |
| `BinGemmPlan` | Bespoke | A/B: `Bin` (packed-bit); D: `i32` | K % 8 == 0; RRR also requires N % 8 == 0 | ✓ | N/A | sm_89. Identity-only — no `α`/`β`/bias/activation. PTX `mma.sync.m16n8k256.xor.popc`. RRR uses bit-gather B-load (bandwidth-heavy). |

---

## OpCategory: Elementwise

| Op | Backend | Dtypes | Shapes / Limits | FW | BW | Notes |
|----|---------|--------|-----------------|----|----|-------|
| `UnaryPlan<T, N>` | Bespoke | FP-family | rank N (compile-time) | ✓ | ✓ via `UnaryBackwardPlan` | ~50 kinds: `{Neg, Abs, Sign, Reciprocal, Square, Cube, Sqrt, Rsqrt, Cbrt, Exp, Exp2, Expm1, Log, Log2, Log10, Log1p, Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh, Asinh, Acosh, Atanh, Floor, Ceil, Round, Trunc, Frac, Relu, Gelu, GeluTanh, Silu, Mish, Sigmoid, Softplus, Hardswish, Hardsigmoid, Hardtanh, Erf, Erfc, Lgamma, Logit, Softsign, Tanhshrink, Relu6, Selu, LeakyRelu, Elu, Hardshrink, Softshrink}`. Both contig + strided. Activation BWs use weighted tolerance (cancellation-aware). |
| `UnaryParamPlan<T, N>` | Bespoke | FP-family | rank N | ✓ | ✓ via `UnaryParamBackwardPlan` | Unary with one scalar param: `{Threshold, LeakyReluA, EluA, HardtanhAB, PowI, …}`. PowI added in Phase 12.1 (integer-exponent power-by-squaring; FW + BW; correct on negative bases — no NaN). Phase 14.2 added strided FFI sibling. |
| `BinaryPlan<T, N>` | Bespoke | FP-family + `{i32, i64}` for bitwise + `Bool` for logical | broadcast-compatible (axis match or `dim==1 && stride==0`) | ✓ | ✓ via `BinaryBackwardPlan` | FP kinds: `{Add, Sub, Mul, Div, Pow, Atan2, Hypot, Copysign, Nextafter, Fmin, Fmax, Maximum, Minimum, FloorDivide, Mod, Remainder}`. Int kinds: `{BitwiseAnd/Or/Xor/LeftShift/RightShift}`. Bool kinds: `{LogicalAnd, LogicalOr, LogicalXor}` (contig only). |
| `BinaryParamPlan<T, N>` | Bespoke | FP-family | rank N | ✓ | ✓ via `BinaryParamBackwardPlan` | Binary with one scalar param: `{Lerp(weight), …}`. |
| `BinaryCmpPlan<T, N>` | Bespoke | FP-family | broadcast-compatible | ✓ | N/A | Comparison ops: `{Eq, Ne, Lt, Le, Gt, Ge}`. Output is `Bool`. |
| `TernaryPlan<T, N>` | Bespoke | FP-family | rank N | ✓ | ✓ via `TernaryBackwardPlan` | Kinds: `{Clamp, Fma, Addcmul, Addcdiv}`. |
| `WherePlan<T, N>` | Bespoke | FP-family (cond: `Bool`) | broadcast-compatible | ✓ | ✓ via `WhereBackwardPlan` | `y = cond ? x : y_alt`. |
| `CastPlan<TIn, TOut>` | Bespoke | full FP-family × `{i32, i64}` cross product | rank N | ✓ | N/A | 36-cell cross product; no Bool input/output. F32 accumulator for FP→FP. |
| `CastSubBytePlan<TIn, TOut>` | Bespoke | sub-byte endpoints: `Bool`, `Fp8E4M3`, `Fp8E5M2`, `S4`, `U4` ↔ FP / int peers | rank N | ✓ | N/A | Phase 13.3 sibling to `CastPlan` (relaxed trait bound `T: DeviceRepr + Copy` since S4/U4/Fp8 don't impl `Element`). 34 FFI symbols. Uses `__nv_cvt_fp8_to_halfraw` / `__nv_cvt_float_to_fp8`. S4/U4 pack+unpack both directions. |
| `AffinePlan<T, N>` | Bespoke | `{f32, f64, i32, i64, u8, f16, bf16}` | rank N | ✓ | N/A | `y = mul * x + add`. Phase 11.x core; Phase 14.1 added strided FFI sibling (canonical-contig fast-path → existing contig FFI; non-canonical → strided). |
| `AffinePlan<T>` | Bespoke | `{f32, f64, f16, bf16, i32, i64}` | rank-1 contig | ✓ | N/A | Fused `y = a·x + b` with scalar `a, b`. Contig only. |
| `PReluPlan<T, N>` | Bespoke | FP-family | rank N, per-channel `α` | ✓ | ✓ via `PReluBackwardPlan` | Parametric ReLU with per-channel slope. |
| `GatedActivationPlan<T, N>` | Bespoke | FP-family | rank N | ✓ | ✓ via `GatedActivationBackwardPlan` | Kinds: `{SwiGlu, Glu, ReGlu, GeGlu}`. Input is `[..., 2D]` halved along last axis. |

---

## OpCategory: Shape / Layout

| Op | Backend | Dtypes | Shapes / Limits | FW | BW | Notes |
|----|---------|--------|-----------------|----|----|-------|
| `FlipPlan<T, N>` | Bespoke | FP-family | rank N | ✓ | ✓ via `FlipBackwardPlan` | BW is involution (reuses FW kernel with same flip axes). |
| `RollPlan<T, N>` | Bespoke | FP-family | rank N | ✓ | ✓ via `RollBackwardPlan` | BW reuses FW kernel with negated shifts. |
| `PermutePlan<T, N>` | Bespoke | FP-family | rank N | ✓ | ✓ via `PermuteBackwardPlan` | BW reuses FW kernel with inverse permutation. |
| `PadPlan<T, N>` | Bespoke | FP-family | rank N | ✓ | ✓ via `PadBackwardPlan` (Constant mode only) | Modes: `{Constant, Reflect, Replicate, Circular}` (Reflect/Replicate/Circular = f16 ∪ f32 ∪ f64 only — bf16 deferred). BW only for Constant (slice operation). |
| `RepeatPlan<T, N>` | Bespoke | `{f32, f16, f64}` (no bf16) | rank N | ✓ | ✓ via `RepeatBackwardPlan` | f16/f64 added in Phase 3.5 fanout. |
| `ConcatPlan<T, N>` | Bespoke | FP-family | rank N | ✓ | ✓ via `ConcatBackwardPlan` | Multi-input concat along one axis. |
| `FillPlan<T>` | Bespoke | `{f32, f64, f16, bf16, i32, i64}` | numel ≥ 0 | ✓ | N/A | Scalar fill. No Bool. |
| `WriteSlicePlan<T, N>` | Bespoke | byte-aligned dtypes (5 byte-width symbols) + S4/U4 nibble | rank 1-8 | ✓ | N/A | Phase 13.1. In-place rectangular slab assignment (not add). Fast paths: full-dest cuMemcpyDtoDAsync + KV-cache contiguous-chunk cuMemcpyDtoDAsync. S4/U4 requires even start/end on innermost axis. No BW (non-differentiable per Fuel's IR). |
| `ContiguizePlan<T, N>` | Bespoke | byte-aligned + nibble | rank 1-8 | ✓ | N/A | Phase 13.2. Strided → contiguous copy. Signed i64 strides (Flip support) + zero strides (broadcast). Fast paths: already-contig early-out + innermost-stride-1 row-memcpy. S4/U4 requires innermost stride ∈ {1, -1, 2}. |
| `TriuPlan<T, N>` | Bespoke | `{f16, bf16, f32, f64, i32, i64, Bool}` | rank ≥ 2 | ✓ | ✓ via `TriuBackwardPlan` (reuses FW kernel; mask is self-adjoint) | Phase 13.4. Phase 14.3 added strided sibling FFI. |
| `TrilPlan<T, N>` | Bespoke | `{f16, bf16, f32, f64, i32, i64, Bool}` | rank ≥ 2 | ✓ | ✓ via `TrilBackwardPlan` (reuses FW kernel) | Phase 13.4. Phase 14.3 added strided sibling FFI. |

---

## OpCategory: Reduction

| Op | Backend | Dtypes | Shapes / Limits | FW | BW | Notes |
|----|---------|--------|-----------------|----|----|-------|
| `ReducePlan<T, N>` | Bespoke | FP-family | rank ≤ 8, single axis | ✓ | ✓ via `ReduceBackwardPlan` | Kinds: `{Sum, Mean, Max, Min, Prod, Norm2, LogSumExp, Var, Std}`. BWs cover all 9 kinds (Prod/Norm2/Var/Std added 2026-05-15). |
| `ArgReducePlan<T, N, I>` | Bespoke | FP-family input → `I ∈ {u32, i32, i64}` output | rank ≤ 8 | ✓ | N/A | Kinds: `{ArgMax, ArgMin}`. Reduce axis must be non-empty. Phase 12.2 added u32/i32 output dtypes via new `IndexOutputElement` sealed trait (default `I = i64` preserves source-compat). |
| `BoolReducePlan<T, N>` | Bespoke | `{f32, f16, bf16, f64, i32, i64, Bool}` input → `Bool` output | rank ≤ 8 | ✓ | N/A | Kinds: `{Any, All}`. Pure integer AND/OR — bit-stable, deterministic. |
| `CountReducePlan<T, N>` | Bespoke | `{f32, f16, bf16, f64, i32, i64, Bool}` input → `i64` output | rank ≤ 8 | ✓ | N/A | Kind: `CountNonzero`. i64 accumulator. |
| `TracePlan<T>` | Bespoke | FP-family | rank-2 only | ✓ | N/A | Sum of diagonal (both axes reduced). Scalar output. |

---

## OpCategory: Scan

| Op | Backend | Dtypes | Shapes / Limits | FW | BW | Notes |
|----|---------|--------|-----------------|----|----|-------|
| `ScanPlan<T, N>` | Bespoke | FP-family | rank ≤ 8, single axis | ✓ | ✓ via `ScanBackwardPlan` | Kinds: `{Cumsum, Cumprod, Cummax, Cummin, LogCumsumExp}`. Single-thread-per-cell sequential scan. |

---

## OpCategory: Softmax / Probability

| Op | Backend | Dtypes | Shapes / Limits | FW | BW | Notes |
|----|---------|--------|-----------------|----|----|-------|
| `SoftmaxPlan<T, N>` | Bespoke | FP-family | rank ≤ 8, single axis | ✓ | ✓ via `SoftmaxBackwardPlan` | Kinds: `{Softmax, LogSoftmax}`. Length-preserving, numerically stable (max-shift). |
| `GumbelSoftmaxPlan<T, N>` | Bespoke | FP-family | rank ≤ 8 | ✓ | ✓ via `GumbelSoftmaxBackwardPlan` | Uses cuRAND for Gumbel noise; BW reuses `softmax_backward_*` symbol (shape match). |
| `SparsemaxPlan<T, N>` | Bespoke | FP-family | rank ≤ 8, extent along softmax axis ≤ `SPARSEMAX_MAX_EXTENT = 1024` | ✓ | ✓ via `SparsemaxBackwardPlan` | Phase 11.6 block-cooperative `cub::BlockRadixSort` + `cub::BlockScan` pipeline (one block per row, 2 compiled tile specs: 256 / 1024). |

---

## OpCategory: Normalization

| Op | Backend | Dtypes | Shapes / Limits | FW | BW | Notes |
|----|---------|--------|-----------------|----|----|-------|
| `RMSNormPlan<T, N>` | Bespoke | FP-family | rank ≤ 8 | ✓ | ✓ via `RMSNormBackwardPlan` | Optional per-feature `γ`. Deterministic affine BW via warp-shuffle (no atomicAdd). Welford for non-f32. |
| `LayerNormPlan<T, N>` | Bespoke | FP-family | rank ≤ 8 | ✓ | ✓ via `LayerNormBackwardPlan` | Optional per-feature `γ`, `β`. Deterministic affine BW via warp-shuffle. |
| `BatchNormPlan<T, N>` | Bespoke | FP-family | rank ≤ 8 | ✓ | ✓ via `BatchNormBackwardPlan` | Train + eval mode; tracks running mean / var. |
| `GroupNormPlan<T, N>` | Bespoke | FP-family | rank ≤ 8 | ✓ | ✓ via `GroupNormBackwardPlan` | Per-group statistics. |
| `InstanceNormPlan<T, N>` | Bespoke | FP-family | rank ≤ 8 | ✓ | ✓ via `InstanceNormBackwardPlan` | Per-sample, per-channel statistics. |

---

## OpCategory: Loss

All loss plans share dtype scope `{f32, f16, bf16, f64}` via
`loss::common::check_supported_dtype`. Rank ≤ 8 via `validate_shape`. All
support `{None, Mean, Sum}` reduction. CtcLossCudnnPlan is the exception
(f32 + f64 only).

| Op | Backend | Dtypes | Shapes / Limits | FW | BW | Notes |
|----|---------|--------|-----------------|----|----|-------|
| `MseLossPlan<T, N>` | Bespoke | FP-family | rank ≤ 8 | ✓ | ✓ | |
| `L1LossPlan<T, N>` | Bespoke | FP-family | rank ≤ 8 | ✓ | ✓ | |
| `HuberLossPlan<T, N>` | Bespoke | FP-family | rank ≤ 8 | ✓ | ✓ | |
| `SmoothL1LossPlan<T, N>` | Bespoke | FP-family | rank ≤ 8 | ✓ | ✓ | |
| `NllLossPlan<T>` | Bespoke | FP-family | rank-2 | ✓ | ✓ | `(input[N,C], target[N])`. |
| `CrossEntropyLossPlan<T>` | Bespoke | FP-family | rank-2 | ✓ | ✓ | Fused logsoftmax + nll. |
| `BceLossPlan<T, N>` | Bespoke | FP-family | rank ≤ 8 | ✓ | ✓ | |
| `BceWithLogitsLossPlan<T, N>` | Bespoke | FP-family | rank ≤ 8 | ✓ | ✓ | Fused sigmoid + BCE. |
| `KlDivLossPlan<T, N>` | Bespoke | FP-family | rank ≤ 8 | ✓ | ✓ | |
| `PoissonNllLossPlan<T, N>` | Bespoke | FP-family | rank ≤ 8 | ✓ | ✓ | |
| `GaussianNllLossPlan<T, N>` | Bespoke | FP-family | rank ≤ 8 | ✓ | ✓ | |
| `CosineEmbeddingLossPlan<T>` | Bespoke | FP-family | rank-2 pairs | ✓ | ✓ | |
| `HingeEmbeddingLossPlan<T, N>` | Bespoke | FP-family | rank ≤ 8 | ✓ | ✓ | |
| `MarginRankingLossPlan<T, N>` | Bespoke | FP-family | rank ≤ 8 | ✓ | ✓ | |
| `MultiMarginLossPlan<T>` | Bespoke | FP-family | rank-2 | ✓ | ✓ | |
| `MultilabelMarginLossPlan<T>` | Bespoke | FP-family | rank-2 | ✓ | ✓ | |
| `MultilabelSoftMarginLossPlan<T>` | Bespoke | FP-family | rank-2 | ✓ | ✓ | |
| `TripletMarginLossPlan<T>` | Bespoke | FP-family | rank-2 triplets | ✓ | ✓ | |
| `CtcLossPlan<T>` | Bespoke | FP-family | variable-length DP | ✓ | partial | FW validated; BW has documented γ-accumulation correctness bug (smoke-tested only). FD helper code retained for re-validation. |
| `CtcLossCudnnPlan<T>` | Cudnn | `{f32, f64}` | variable-length | ✓ | ✓ | Phase 7 sibling; Fuel's autotuner races against bespoke. Gated by `cudnn` feature. |

---

## OpCategory: Random / Sampling

| Op | Backend | Dtypes | Shapes / Limits | FW | BW | Notes |
|----|---------|--------|-----------------|----|----|-------|
| `RandomPlan<T, N>` | Curand | `Uniform/Normal: {f32, f64}` · `Bernoulli: Bool` | rank ≤ 8 | ✓ | N/A | Kinds: `{Uniform, Normal, Bernoulli}`. cuRAND-backed. |
| `DropoutPlan<T, N>` | Bespoke (on top of cuRAND) | `{f32, f64}` | rank ≤ 8 | ✓ | ✓ via `DropoutBackwardPlan` | Generates Bool mask + scales by `1/(1-p)`. |

---

## OpCategory: Attention

| Op | Backend | Dtypes | Shapes / Limits | FW | BW | Notes |
|----|---------|--------|-----------------|----|----|-------|
| `SdpaPlan<T>` | Bespoke | FP-family | `[B, H, S, D]` | ✓ | ✓ via `SdpaBackwardPlan` | Reference (non-flash) scaled-dot-product attention. Phase 14.4 added strided FFI sibling. **SDPA BW + GQA broadcast (`stride_k[head_axis] == 0`) is `Error::Unsupported`** — would need atomicAdd for dK/dV accumulation. SDPA FW accepts it normally. |
| `FlashSdpaPlan<T>` | Bespoke (sm_80) | FP-family | `head_dim ≤ FLASH_SDPA_MAX_D = 128` | ✓ | ✓ via `FlashSdpaBackwardPlan` | Tile-streamed online softmax. Paged variant deferred. |
| `FlashSdpaSm89Plan<T>` | Bespoke (sm_89) | `{f16, bf16}` only | `head_dim ≤ 128` | ✓ | N/A | Phase 10.3 — Ada Lovelace variant with `cp.async` double-buffered K/V loads. Gated by `sm89` feature. **Strided sibling deferred** — kernel hardcodes offsets internally (not stride-driven). |
| `RopePlan<T>` | Bespoke | FP-family | `head_dim` must be even | ✓ | ✓ via `RopeBackwardPlan` | Rotary positional encoding (Llama / Mistral / Gemma). Default base 10000.0. Phase 14.4 added strided FFI sibling — head_dim axis must remain stride=1 (RoPE rotates adjacent pairs); enforced at plan layer. |
| `AlibiPlan<T>` | Bespoke | FP-family | `[B, H, S, S]` bias | ✓ | ✓ via `AlibiBackwardPlan` | Linear-bias attention (MPT / BLOOM). |
| `KvCacheAppendPlan<T>` | Bespoke | FP-family | append along seq axis | ✓ | N/A | Inference-time KV append. |

---

## OpCategory: Linalg (Dense)

cuSOLVER-backed; cuSOLVER's dense API does not expose f16/bf16 for these ops.

| Op | Backend | Dtypes | Shapes / Limits | FW | BW | Notes |
|----|---------|--------|-----------------|----|----|-------|
| `CholeskyPlan<T>` | Cusolver | `{f32, f64}` | 2-D, SPD | ✓ | N/A | |
| `LuPlan<T>` | Cusolver | `{f32, f64}` | 2-D | ✓ | N/A | Returns L, U, pivot. |
| `QrPlan<T>` | Cusolver | `{f32, f64}` | 2-D | ✓ | N/A | Returns A (R packed in upper) + tau. |
| `BatchedQrPlan<T>` | Cusolver | `{f32, f64, Complex32, Complex64}` | uniform batch | ✓ | N/A | All 4 fp dtypes (real + complex). |
| `BatchedQrMaterializePlan<T>` | Cusolver | `{f32, f64}` | 2-D batched | ✓ | N/A | Materializes explicit Q from packed tau output. |
| `BatchedOrmqrPlan<T>` | Cusolver | `{f32, f64, Complex32, Complex64}` | reflector-by-reflector | ✓ | N/A | `Q^T · C` / `Q · C` family. Side ∈ `{Left, Right}`, Op ∈ `{NoTrans, Trans, ConjTrans}`. |
| `BatchedOrmqrWyPlan<T>` | Cusolver | `{f32, f64}` | WY-blocked, `WY_NB` = block size | ✓ | N/A | WY-blocked variant for cache locality. Complex variants deferred to Phase 10.4. |
| `SvdPlan<T>` | Cusolver | `{f32, f64}` | 2-D, `m ≥ n` | ✓ | N/A | `cusolverDnSgesvd` / `Dgesvd`. |
| `BatchedSvdPlan<T>` | Cusolver | `{f32, f64}` | square `[N, N]` per slot | ✓ | N/A | Batched-Jacobi-SVD; cuSOLVER requires square matrices. |
| `BatchedSvdaPlan<T>` | Cusolver | `{f32, f64}` | strided batched | ✓ | N/A | `gesvdaStridedBatched`. Supports approximate / truncated SVD via `rank` cap. |
| `EighPlan<T>` | Cusolver | `{f32, f64, Complex32, Complex64}` | 2-D, single matrix | ✓ | N/A | Real symmetric → `syevd`; complex Hermitian → `heevd`. |
| `EigPlan<T>` | Cusolver | `{f32, f64, Complex32, Complex64}` | 2-D | ✓ | N/A | Non-symmetric eigendecomposition. |
| `SolvePlan<T>` | Cusolver | `{f32, f64}` | 2-D RHS | ✓ | N/A | `Ax = b` via LU. |
| `InversePlan<T>` | Cusolver | `{f32, f64}` | 2-D, square | ✓ | N/A | LU + `getrs` against identity. |
| `LstSqPlan<T>` | Cusolver | `{f32, f64}` | `m ≥ n` (full-rank) | ✓ | N/A | `_gels` mixed-precision iterative refinement. A is destroyed per call. |

---

## OpCategory: FFT

cuFFT-backed. f32 + f64 only (cuFFT's main API does not expose f16 / bf16).

| Op | Backend | Dtypes | Shapes / Limits | FW | BW | Notes |
|----|---------|--------|-----------------|----|----|-------|
| `FftPlan<T>` | Cufft | `{Complex32, Complex64}` | 1-D | ✓ | N/A | C2C 1-D forward FFT. |
| `RfftPlan<T>` | Cufft | `{f32, f64}` | 1-D | ✓ | N/A (paired with `IrfftPlan`) | R2C 1-D. |
| `IrfftPlan<T>` | Cufft | `{f32, f64}` | 1-D | ✓ | N/A (paired with `RfftPlan`) | C2R 1-D. |
| `FftNdPlan<T>` | Cufft | `{Complex32, Complex64}` | rank ≤ `MAX_RANK = 4` | ✓ | N/A | C2C N-D. |
| `RfftNdPlan<T>` | Cufft | `{f32, f64}` | rank ≤ 4 | ✓ | N/A | R2C N-D. |
| `IrfftNdPlan<T>` | Cufft | `{f32, f64}` | rank ≤ 4 | ✓ | N/A | C2R N-D. |
| `FftShiftPlan<T>` | Bespoke | `{f32, f64, Complex32, Complex64}` | 1-D | ✓ | N/A | Pure index permutation. |
| `FftShiftNdPlan<T, N>` | Bespoke | `{f32, f64, Complex32, Complex64}` | rank ≤ `FFTSHIFT_ND_MAX_RANK = 8`, num_shift_axes ≤ `FFTSHIFT_ND_MAX_SHIFT_AXES = 4` | ✓ | N/A | |

---

## OpCategory: Convolution

cuDNN-backed. Gated by `cudnn` cargo feature. Conv2d shipped in Phase 7;
Phase 11.7 added the 1D / 3D / Transpose / depthwise fanout. `Conv2dDescriptor`
gained a `groups: i32` field in Phase 11.7 (breaking change to literal-init
sites — depthwise = `groups == c_in == c_out`). 1D plans pad rank-3 → rank-4
internally because cuDNN's `cudnnSetTensorNdDescriptor` rejects `nb_dims < 4`.

| Op | Backend | Dtypes | Shapes / Limits | FW | BW | Notes |
|----|---------|--------|-----------------|----|----|-------|
| `Conv1dPlan<T>` | Cudnn | FP-family | NCL, rank-3 (padded to rank-4 NCLW with W=1 internally) | ✓ | ✓ (data + filter) | Phase 11.7. |
| `Conv2dPlan<T>` | Cudnn | FP-family | NCHW, 4-D | ✓ | ✓ (data + filter) | Phase 7. Phase 11.7 added `groups` field (depthwise). One cuDNN handle + 4 lazy descriptors per plan. BW data = `IMPLICIT_GEMM` algo 1; BW filter = `ALGO_1`. F64 math through F64; F16/Bf16 math via F32 promotion (cuDNN convention). |
| `Conv3dPlan<T>` | Cudnn | FP-family | NCDHW, rank-5 | ✓ | ✓ (data + filter) | Phase 11.7. |
| `ConvTranspose1dPlan<T>` | Cudnn | FP-family | NCL, rank-3 | ✓ | ✓ | Phase 11.7. Implemented via cuDNN's `BackwardData` with input/output roles swapped (standard transposed-conv trick). |
| `ConvTranspose2dPlan<T>` | Cudnn | FP-family | NCHW, 4-D | ✓ | ✓ | Phase 11.7. |
| `ConvTranspose3dPlan<T>` | Cudnn | FP-family | NCDHW, rank-5 | ✓ | ✓ | Phase 11.7. |

---

## OpCategory: Pooling

cuDNN-backed. Gated by `cudnn` cargo feature. Conv2d / Pool2d shipped in
Phase 7; Phase 11.8 added 1D / 3D + Adaptive fanout via the rank-agnostic
`cudnnPoolingNdDescriptor`. Adaptive pool uses the
`kernel = ceil(in/out); stride = floor(in/out); pad = 0` approximation —
diverges from PyTorch by ±1 input cell when `in_i % out_i != 0` (bit-exact
adaptive pool deferred — see [`ROADMAP.md`](ROADMAP.md)). FractionalMaxPool
and LpPool are stubbed (`Error::Unsupported` at `select()`) pending bespoke
kernels.

| Op | Backend | Dtypes | Shapes / Limits | FW | BW | Notes |
|----|---------|--------|-----------------|----|----|-------|
| `MaxPool1dPlan<T>` | Cudnn | FP-family | NCL, rank-3 (padded internally) | ✓ | ✓ | Phase 11.8. |
| `MaxPool2dPlan<T>` | Cudnn | FP-family | NCHW, 4-D | ✓ | ✓ | Phase 7. Shares pool descriptor with `AvgPool2d`. |
| `MaxPool3dPlan<T>` | Cudnn | FP-family | NCDHW, rank-5 | ✓ | ✓ | Phase 11.8. |
| `AvgPool1dPlan<T>` | Cudnn | FP-family | NCL, rank-3 | ✓ | ✓ | Phase 11.8. |
| `AvgPool2dPlan<T>` | Cudnn | FP-family | NCHW, 4-D | ✓ | ✓ | Phase 7. |
| `AvgPool3dPlan<T>` | Cudnn | FP-family | NCDHW, rank-5 | ✓ | ✓ | Phase 11.8. |
| `AdaptiveAvgPool{1,2,3}dPlan<T>` | Cudnn | FP-family | various ranks | ✓ | ✓ | Phase 11.8. cuDNN approximation — not bit-exact PyTorch for non-divisible cases. |
| `AdaptiveMaxPool{1,2,3}dPlan<T>` | Cudnn | FP-family | various ranks | ✓ | ✓ | Phase 11.8. Same approximation caveat. |
| `FractionalMaxPool{2,3}dPlan<T>` | — | — | — | ✗ stub | — | Phase 11.8 stub. Returns `Error::Unsupported` at `select()` pending bespoke kernel (cuDNN doesn't expose this). |
| `LpPool{1,2}dPlan<T>` | — | — | — | ✗ stub | — | Phase 11.8 stub. Would be a composite `pow → avg_pool → pow`; needs parameterized `Pow(p)` unary plan first. |

---

## OpCategory: Indexing / Scatter / Gather

Index dtype: `i32` (default) and `i64` (Phase 11.5 — opt-in via the
`IndexElement` sealed trait; default `I = i32` preserves source-compat).
Out-of-bounds + negative indices are silently skipped (no PyTorch-style
wrap-around). BWs use `atomicAdd`. As of Phase 11.3, bf16 / f16 indexing
& segment BW route through `atomicAdd_via_cas` (deterministic, available
on every supported arch); `{f32, f64}` continue to use native atomicAdd.

| Op | Backend | Dtypes | Shapes / Limits | FW | BW | Notes |
|----|---------|--------|-----------------|----|----|-------|
| `GatherPlan<T, N>` | Bespoke | FW: `{f32, f64, i32}` | rank ≤ 8 | ✓ | ✓ via `GatherBackwardPlan` (`{f32, f64}` only) | BW non-deterministic (atomicAdd). |
| `ScatterAddPlan<T, N>` | Bespoke | `{f32, f64}` | rank ≤ 8 | ✓ | N/A (this **is** the scatter BW for gather) | atomicAdd → non-deterministic. |
| `IndexSelectPlan<T, N>` | Bespoke | FW: `{f32, f64, i32}` | rank ≤ 8 | ✓ | ✓ via `IndexSelectBackwardPlan` (`{f32, f64}` only) | BW uses atomicAdd. |
| `MaskedFillPlan<T, N>` | Bespoke | `{f32, f64, i32, Bool}` | rank ≤ 8 | ✓ | ✓ via `MaskedFillBackwardPlan` (same dtypes) | |
| `OneHotPlan<T, N>` | Bespoke | output: `{f32, f64, i32, Bool}` | rank ≤ 8 | ✓ | N/A | |
| `NonzeroPlan<T, N>` | Bespoke | input: `{f32, f64, i32, Bool}` | rank ≤ 8 | ✓ | N/A | Atomic-counter index emission; non-deterministic across launches. |

---

## OpCategory: Embedding

Index dtype (Phase 11.5): `i32` (default) or `i64` (opt-in via `IndexElement`).
As of Phase 11.3, all BW dtypes are wired (bf16 / f16 route through
`atomicAdd_via_cas`).

| Op | Backend | Dtypes | Shapes / Limits | FW | BW | Notes |
|----|---------|--------|-----------------|----|----|-------|
| `EmbeddingPlan<T, I>` | Bespoke | FW + BW: FP-family | `[N, embedding_dim]` weight; `I ∈ {i32, i64}` | ✓ | ✓ via `EmbeddingBackwardPlan` | Optional `padding_idx` skip. BW atomicAdd (non-deterministic). |
| `EmbeddingBagPlan<T, I>` | Bespoke | FW + BW: FP-family | `[N, dim]` weight, `[bag_size]` offsets; `I ∈ {i32, i64}` | ✓ | ✓ via `EmbeddingBagBackwardPlan` | Modes: `{Sum, Mean}`. `Max` mode deferred (needs per-feature argmax tracking). |

---

## OpCategory: Segment / Scatter-Reduce

All segment plans share dtype scope `{f32, f64}` (atomicAdd /
atomicMax-via-CAS / atomicMin-via-CAS restricted to native-FP-atomic types).
Sorted variants are deterministic (single thread per output cell, in-order
sweep); unsorted variants are non-deterministic (atomic accumulation order).

| Op | Backend | Dtypes | Shapes / Limits | FW | BW | Notes |
|----|---------|--------|-----------------|----|----|-------|
| `SegmentSumPlan<T>` | Bespoke | `{f32, f64}` | rank-2 `[N, D]` | ✓ | ✓ via `SegmentSumBackwardPlan` | Sorted. |
| `SegmentMeanPlan<T>` | Bespoke | `{f32, f64}` | rank-2 | ✓ | ✓ via `SegmentMeanBackwardPlan` | Sorted. Workspace = `num_segments * sizeof(i32)` for counts. |
| `SegmentMaxPlan<T>` | Bespoke | `{f32, f64}` | rank-2 | ✓ | deferred | BW needs argmax tracking from FW. |
| `SegmentMinPlan<T>` | Bespoke | `{f32, f64}` | rank-2 | ✓ | deferred | BW needs argmin tracking. |
| `SegmentProdPlan<T>` | Bespoke | `{f32, f64}` | rank-2 | ✓ | deferred | BW needs numerically stable `prod / x_n`. |
| `UnsortedSegmentSumPlan<T>` | Bespoke | `{f32, f64}` | rank-2 | ✓ | ✓ via `UnsortedSegmentSumBackwardPlan` | atomicAdd. Non-deterministic FW; deterministic BW (pure gather). |
| `UnsortedSegmentMeanPlan<T>` | Bespoke | `{f32, f64}` | rank-2 | ✓ | ✓ via `UnsortedSegmentMeanBackwardPlan` | atomicAdd. |
| `UnsortedSegmentMaxPlan<T>` | Bespoke | `{f32, f64}` | rank-2 | ✓ | deferred | atomicMax-via-CAS. |
| `UnsortedSegmentMinPlan<T>` | Bespoke | `{f32, f64}` | rank-2 | ✓ | deferred | atomicMin-via-CAS. |
| `UnsortedSegmentProd` | — | — | — | ✗ | — | Deferred — no native FP atomicMul; would need atomicCAS retry loop. |

---

## OpCategory: Quantization

Phase 8. Dtype matrix: input `TIn ∈ FP-family` × quantized `TOut ∈ {s8, u8}`.
BW via STE (straight-through estimator) for `quantize_*` and straight-through
scaling for `dequantize_*`.

| Op | Backend | Dtypes | Shapes / Limits | FW | BW | Notes |
|----|---------|--------|-----------------|----|----|-------|
| `QuantizePerTensorPlan<TIn, TOut>` | Bespoke | FP-family × `{s8, u8}` | rank ≤ 8 | ✓ | ✓ via `QuantizePerTensorBackwardPlan` | Single global scale + zero-point. |
| `DequantizePerTensorPlan<TIn, TOut>` | Bespoke | FP-family × `{s8, u8}` | rank ≤ 8 | ✓ | ✓ via `DequantizePerTensorBackwardPlan` | |
| `QuantizePerChannelPlan<TIn, TOut>` | Bespoke | FP-family × `{s8, u8}` | per-channel along one axis | ✓ | ✓ via `QuantizePerChannelBackwardPlan` | |
| `DequantizePerChannelPlan<TIn, TOut>` | Bespoke | FP-family × `{s8, u8}` | per-channel | ✓ | ✓ via `DequantizePerChannelBackwardPlan` | |
| `QuantizePerTokenPlan<TIn, TOut>` | Bespoke | FP-family × `{s8, u8}` | per-token rows | ✓ | ✓ via `QuantizePerTokenBackwardPlan` | LLM W8A8 activation quant. |
| `DequantizePerTokenPlan<TIn, TOut>` | Bespoke | FP-family × `{s8, u8}` | per-token | ✓ | ✓ via `DequantizePerTokenBackwardPlan` | |
| `QuantizePerGroupPlan<TIn, TOut>` | Bespoke | FP-family × `{s8, u8}` | per-group along axis | ✓ | ✓ via `QuantizePerGroupBackwardPlan` | INT4 GPTQ-style. |
| `DequantizePerGroupPlan<TIn, TOut>` | Bespoke | FP-family × `{s8, u8}` | per-group | ✓ | ✓ via `DequantizePerGroupBackwardPlan` | |
| `FakeQuantizePlan<TIn>` | Bespoke | FP-family | rank ≤ 8 | ✓ | ✓ via `FakeQuantizeBackwardPlan` | Quant-then-dequant in FP (no int storage). STE BW. |
| `DynamicRangeQuantizePlan<TIn, TOut>` | Bespoke | `TIn ∈ {f32, f64}`, `TOut = S8` | rank ≤ 8 | ✓ | N/A | One-shot min/max-driven dynamic-range W8 quant. |
| `QuantizedLinearPlan<TIn, TWQ>` | Bespoke | `TIn ∈ {f32, f64}`, `TWQ = S8` | matmul shape | ✓ | N/A | Composing quant op — fused dequant + matmul. |
| `GgufDequantizePlan` | Bespoke (vendored llama.cpp) | block → `f32` (`f16` for Q8K) | per-block-format size | ✓ | N/A | Block formats: `{Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2K, Q3K, Q4K, Q5K, Q6K, Q8K}`. Q8K is dequant-only (no MMVQ). |
| `GgufMmvqPlan` | Bespoke (Q8_K is bespoke baracuda; rest vendored llama.cpp) | activation: `f32`; weights: GGUF block | matrix-vector | ✓ | N/A | Fused dequant + matrix-vector multiply. **All 11 block formats wired** as of Phase 11.4 (Q8_K added bespoke; no llama.cpp upstream). Phase 14.5 added an activation-strided sibling FFI (single `stride_y: i64` + `w_start_byte_offset` for "multiple matrices in one allocation"). f16 / bf16 activation deferred. |

---

## OpCategory: MoE (Mixture-of-Experts)

| Op | Backend | Dtypes | Shapes / Limits | FW | BW | Notes |
|----|---------|--------|-----------------|----|----|-------|
| `MoePlan` | Bespoke (vendored from `attention.rs`) | varies by variant (see below) | `[num_tokens, d_model]` | ✓ | N/A | Phase 8.5. Three variants: `ScalarGguf` (scalar dequant + GEMV per expert, `f32` activations), `Wmma` (sm_70+ tensor cores, `{f16, bf16}` activations), `WmmaGguf` (fused tensor-core + GGUF dequant; `{f16, bf16}` activations, `f32` output). |

---

## OpCategory: Sort / Order Statistics

Block-bitonic sort + topk: one block per row. `row_len ≤ SORT_MAX_ROW = 1024`,
`k ≤ TOPK_MAX_K = 64`. Sort/topk BW use the saved-indices scatter contract
(FW emits indices as a required output; BW reads them verbatim).

| Op | Backend | Dtypes | Shapes / Limits | FW | BW | Notes |
|----|---------|--------|-----------------|----|----|-------|
| `SortPlan<T>` | Bespoke | FW: `{f32, f64, i32, i64}` | `row_len ≤ 1024` | ✓ | ✓ via `SortBackwardPlan` (`{f32, f64}` only) | Block-bitonic. |
| `ArgsortPlan<T>` | Bespoke | FW: `{f32, f64, i32, i64}` | `row_len ≤ 1024` | ✓ | N/A | Returns sorted indices. |
| `MsortPlan<T>` | Bespoke | `{f32, f64}` | `row_len ≤ 1024` | ✓ | ✓ via `MsortBackwardPlan` | Multi-segment sort. |
| `TopkPlan<T>` | Bespoke | `{f32, f64}` | `k ≤ 64`, `row_len ≤ 1024` | ✓ | ✓ via `TopkBackwardPlan` | |
| `KthvaluePlan<T>` | Bespoke | `{f32, f64}` | `k+1 ≤ 64`, `row_len ≤ 1024` | ✓ | ✓ via `KthvalueBackwardPlan` | |
| `UniquePlan<T>` | Bespoke | `{f32, f64, i32}` | per-row | ✓ | N/A | Set-valued op. |
| `UniqueConsecutivePlan<T>` | Bespoke | `{f32, f64, i32}` | per-row | ✓ | N/A | Run-length unique. |
| `SearchsortedPlan<T>` | Bespoke | `{f32, f64, i32, i64}` | per-query binary search | ✓ | N/A | |
| `HistogramPlan<T>` | Bespoke | `{f32, f64}` | atomic-bin | ✓ | N/A | |
| `HistogramddPlan<T>` | Bespoke | `{f32, f64}` | rank-2 + edges | ✓ | N/A | Multi-dim histogram. |
| `BincountPlan<T>` | Bespoke | `{i32, i64}` | dense int counts | ✓ | N/A | |

---

## OpCategory: Image / Spatial Transforms

NCHW. f32 + f64 for math-bearing ops; `pixel_shuffle` / `pixel_unshuffle` add
f16 + bf16 (memory-bound, no math).

| Op | Backend | Dtypes | Shapes / Limits | FW | BW | Notes |
|----|---------|--------|-----------------|----|----|-------|
| `InterpolatePlan<T>` | Bespoke | `{f32, f64}` | NCHW 4-D | ✓ | ✓ via `InterpolateBackwardPlan` | Modes: bilinear 2-D. |
| `GridSamplePlan<T>` | Bespoke | `{f32, f64}` | NCHW 4-D + `[N, H_out, W_out, 2]` grid | ✓ | ✓ via `GridSampleBackwardPlan` | |
| `AffineGridPlan<T>` | Bespoke | `{f32, f64}` | `[N, 2, 3]` theta → `[N, H, W, 2]` grid | ✓ | N/A | Pairs with `GridSample`. |
| `PixelShufflePlan<T>` | Bespoke | FP-family | NCHW 4-D | ✓ | N/A (use `PixelUnshufflePlan`) | Memory-bound; FP-family supported. |
| `PixelUnshufflePlan<T>` | Bespoke | FP-family | NCHW 4-D | ✓ | N/A (use `PixelShufflePlan`) | The two are exact inverses. |
| `RoiAlignPlan<T>` | Bespoke | `{f32, f64}` | NCHW + `[N_rois, 5]` rois | ✓ | ✓ via `RoiAlignBackwardPlan` | |
| `RoiPoolPlan<T>` | Bespoke | `{f32, f64}` | NCHW + `[N_rois, 5]` rois | ✓ | ✓ via `RoiPoolBackwardPlan` | |
| `NmsPlan<T>` | Bespoke | `{f32, f64}` | `[N, 4]` boxes + `[N]` scores | ✓ | N/A | Non-max suppression. |

---

## Deferred / Out of Scope

The live backlog with priority + effort estimates lives in
[`ROADMAP.md`](ROADMAP.md). The summary by category:

- **Attention**: Flash SDPA sm_89 strided sibling (kernel hardcodes
  offsets internally; rewrite needed). SDPA BW + GQA broadcast
  (atomicAdd path for dK/dV accumulation across broadcast Q-head
  groups). Paged FlashAttention.
- **Pool completeness**: bit-exact AdaptiveAvgPool / AdaptiveMaxPool
  (current cuDNN approximation diverges ±1 input cell when
  `in_i % out_i != 0`). LpPool 1d/2d composite (needs parameterized
  `Pow(p)` unary plan first). FractionalMaxPool 2d/3d bespoke kernel.
- **Sparsemax for extents > 1024** — would need multi-block / global
  sort pipeline.
- **Indexing wrappers**: `OneHotPlan` / `NonzeroPlan` /
  `MaskedFillPlan` Rust plan wrappers are still i32-only;
  Phase 11.5 FFI symbols (`*_i64idx_*`) exist but the Rust wrapper
  doesn't expose `I: IndexElement` yet.
- **Segment**: `Max` / `Min` / `Prod` BW (needs argmax / argmin
  tracking from FW; numerically stable `prod / x_n` for Prod).
- **EmbeddingBag**: `Max` mode (per-feature argmax tracking).
- **`BatchedOrmqrWyPlan` complex variants** — `{f32, f64}` ship;
  complex would need `cunmqr` rather than `cusolverDnXormqr`.
- **f16 / bf16 activations for `GgufMmvqPlan`** — f32 only today.
- **Correctness bugs**:
  - CtcLossBackward γ-accumulation (FW validated; BW only smoke-
    tested; FD helper retained for re-validation after fix).
  - MoE CPU-reference math mismatch (Phase 8 vendor; kernel
    smoke-tested but the host-side reference is wrong, so assertions
    are disabled — pure test-layer follow-up).
  - MMVQ `w_start_byte_offset` alignment guard (Phase 14.5 added the
    offset parameter but no debug-build assertion for block formats
    that require alignment — Q4_K is 16-byte aligned; misuse is
    silent-wrong today).
- **Long-arc roadmap items** (from the original comprehensive plan,
  pre-empted by Fuel-driven Phase 11+ work):
  - sm_90a (Hopper async) specialization + Blackwell forward-compat
    (was the original "Phase 11").
  - API freeze + 1.0 stability + benchmark suite against
    PyTorch / cuDNN / cuBLAS references (was the original
    "Phase 12").
