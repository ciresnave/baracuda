//! # baracuda-kernels
//!
//! Unified ML op facade for the baracuda CUDA ecosystem.
//!
//! Exposes every primitive an ML framework would expect (union of
//! PyTorch `torch.*` + `nn.functional` and JAX `lax.*` / `numpy` ops)
//! through a single Plan-based Rust surface, internally dispatching to:
//!
//! 1. An NVIDIA-library wrapper crate when one already covers the op
//!    (`baracuda-cublas`, `baracuda-cudnn`, `baracuda-cufft`,
//!    `baracuda-cusparse`, `baracuda-cusolver`, `baracuda-curand`,
//!    `baracuda-cutensor`, `baracuda-npp`, `baracuda-cvcuda`,
//!    `baracuda-cutlass`).
//! 2. A bespoke `.cu` kernel shipped in
//!    [`baracuda-kernels-sys`](https://docs.rs/baracuda-kernels-sys)
//!    when no NVIDIA library covers it (or covers it poorly at relevant
//!    shapes).
//!
//! Callers import **one** crate and reach for **one** API style; the
//! dispatch decision is an internal detail driven by `select`.
//!
//! ## Status
//!
//! Phase 0 scaffolding: the facade currently re-exports the existing
//! `baracuda-cutlass` plan types so downstream callers can switch their
//! import paths now (`use baracuda_kernels::IntGemmPlan;` instead of
//! `use baracuda_cutlass::IntGemmPlan;`) and gain the new layouts /
//! dtypes as later phases land — no API breakage at the switch.
//!
//! The first bespoke kernels (int8 GEMM RRR — `LayoutSku::Rrr` over
//! `{S8, U8} × {Identity, Bias, BiasRelu, BiasGelu, BiasSilu} × {f32, i32}` bias)
//! land in workspace alpha.16.

#![deny(missing_docs)]

// Re-export the shared type vocabulary.
pub use baracuda_kernels_types::{
    contiguous_stride, ActivationKind, ArchSku, ArgReduceKind, AttentionKind, BackendKind,
    BiasElement, BiasElementKind, Bin, BinElement, BinaryCmpKind, BinaryKind, Bool, Complex32,
    Complex64, CrossEntropyTargetKind, Element, ElementKind, EmbeddingKind, EpilogueKind,
    F32Strict, FftKind, FillMode, Fp8E4M3, Fp8E5M2, FpElement, GatedActivationKind,
    GgufBlockFormat, IndexingKind,
    IntElement, KernelSku, LayoutSku, LinalgKind, LossKind, LossReduction, MathPrecision,
    MatrixMut, MatrixRef, MoeKind, NormalizationKind, OpCategory, PadMode, PlanPreference, PoolKind,
    PrecisionGuarantee, QuantizeKind, RandomKind, ReduceKind, S4, S8, ScalarType, ScanKind,
    SegmentKind,
    ShapeLayoutKind, SoftmaxKind, TensorMut, TensorRef, TernaryKind, U4, U8, UnaryKind, VectorRef,
    Workspace,
};

// Re-export the float-GEMM plan types from baracuda-cutlass unchanged —
// no bespoke path exists for float GEMM yet, the CUTLASS surface is
// the one true entry.
pub use baracuda_cutlass::{
    BatchedGemmArgs, BatchedGemmDescriptor, BatchedGemmPlan, Error, GemmArgs, GemmDescriptor,
    GemmPlan, GemmSku, GroupedGemmPlan, GroupedPlanPreference, GroupedProblem, GroupedScheduleMode,
    PreparedGroupedGemm, Result,
};

// Unified GEMM plan dispatchers. Today exposes only `IntGemmPlan` (RCR
// → CUTLASS, RRR → bespoke); float GEMM and the FP8 / int4 / bin
// dispatchers join later.
pub mod gemm;

pub use gemm::{
    BinGemmArgs, BinGemmDescriptor, BinGemmPlan, Fp8GemmArgs, Fp8GemmDescriptor, Fp8GemmPlan,
    Int4GemmArgs, Int4GemmDescriptor, Int4GemmPlan, IntGemmArgs, IntGemmDescriptor, IntGemmPlan,
};

// Elementwise op family — Phase 3 trailblazer surface. See module docs
// for the per-category Plan layout.
pub mod elementwise;

pub use elementwise::{
    AffineArgs, AffineDescriptor, AffinePlan, BinaryArgs, BinaryBackwardArgs,
    BinaryBackwardDescriptor, BinaryBackwardPlan, BinaryCmpArgs,
    BinaryCmpDescriptor, BinaryCmpPlan, BinaryDescriptor, BinaryParamArgs,
    BinaryParamBackwardArgs, BinaryParamBackwardDescriptor, BinaryParamBackwardPlan,
    BinaryParamDescriptor, BinaryParamPlan, BinaryPlan, CastArgs, CastDescriptor, CastPlan,
    GatedActivationArgs,
    GatedActivationBackwardArgs, GatedActivationBackwardDescriptor, GatedActivationBackwardPlan,
    GatedActivationDescriptor, GatedActivationPlan, TernaryArgs, TernaryBackwardArgs,
    TernaryBackwardDescriptor, TernaryBackwardPlan, TernaryDescriptor, TernaryPlan, UnaryArgs,
    UnaryBackwardArgs, UnaryBackwardDescriptor, UnaryBackwardPlan, UnaryDescriptor,
    UnaryParamArgs, UnaryParamBackwardArgs, UnaryParamBackwardDescriptor, UnaryParamBackwardPlan,
    UnaryParamDescriptor, UnaryParamPlan, UnaryPlan, WhereArgs, WhereBackwardArgs,
    WhereBackwardDescriptor, WhereBackwardPlan, WhereDescriptor, WherePlan,
};

pub use elementwise::{
    PReluArgs, PReluBackwardArgs, PReluBackwardDescriptor, PReluBackwardPlan, PReluDescriptor,
    PReluPlan,
};

// Shape / layout op family — Category N. Plan-per-op because each
// op's descriptor / args shape differs.
pub mod shape_layout;

pub use shape_layout::{
    ConcatArgs, ConcatBackwardArgs, ConcatBackwardDescriptor, ConcatBackwardPlan,
    ConcatDescriptor, ConcatPlan, FillArgs, FillDescriptor, FillPlan, FlipArgs,
    FlipBackwardArgs, FlipBackwardDescriptor,
    FlipBackwardPlan, FlipDescriptor, FlipPlan, PadArgs, PadBackwardArgs,
    PadBackwardDescriptor, PadBackwardPlan, PadDescriptor, PadPlan, PermuteArgs,
    PermuteBackwardArgs, PermuteBackwardDescriptor, PermuteBackwardPlan, PermuteDescriptor,
    PermutePlan, RepeatArgs, RepeatBackwardArgs, RepeatBackwardDescriptor,
    RepeatBackwardPlan, RepeatDescriptor, RepeatPlan, RollArgs, RollBackwardArgs,
    RollBackwardDescriptor, RollBackwardPlan, RollDescriptor, RollPlan,
};

// Reduction op family — Phase 4 (Category E). Output shape differs
// from input by the reduced axes.
pub mod reduce;

pub use reduce::{
    ArgReduceArgs, ArgReduceDescriptor, ArgReducePlan, BoolReduceArgs, BoolReduceDescriptor,
    BoolReducePlan, CountReduceArgs, CountReduceDescriptor, CountReducePlan, ReduceArgs,
    ReduceBackwardArgs, ReduceBackwardDescriptor, ReduceBackwardPlan, ReduceDescriptor, ReducePlan,
    TraceArgs, TraceDescriptor, TracePlan,
};

// Scan (associative prefix) op family — Phase 4 (Category F).
// Length-preserving along the scan axis.
pub mod scan;

pub use scan::{
    ScanArgs, ScanBackwardArgs, ScanBackwardDescriptor, ScanBackwardPlan, ScanDescriptor,
    ScanPlan,
};

// Softmax family — Phase 5 (Category H). Length-preserving stable
// softmax / log-softmax / sparsemax along a single axis.
pub mod softmax;

pub use softmax::{
    GumbelSoftmaxArgs, GumbelSoftmaxBackwardArgs, GumbelSoftmaxBackwardDescriptor,
    GumbelSoftmaxBackwardPlan, GumbelSoftmaxDescriptor, GumbelSoftmaxPlan, SoftmaxArgs,
    SoftmaxBackwardArgs, SoftmaxBackwardDescriptor, SoftmaxBackwardPlan, SoftmaxDescriptor,
    SoftmaxPlan, SparsemaxArgs, SparsemaxBackwardArgs, SparsemaxBackwardDescriptor,
    SparsemaxBackwardPlan, SparsemaxDescriptor, SparsemaxPlan, SPARSEMAX_MAX_EXTENT,
};

// Normalization family — Phase 5 (Category G). Per-row stable
// normalization along a single axis with optional per-feature affine
// (gamma / beta) parameters. Today wired: RMSNorm + LayerNorm × FW + BW.
pub mod norm;

pub use norm::{
    BatchNormArgs, BatchNormBackwardArgs, BatchNormBackwardDescriptor, BatchNormBackwardPlan,
    BatchNormDescriptor, BatchNormPlan, GroupNormArgs, GroupNormBackwardArgs,
    GroupNormBackwardDescriptor, GroupNormBackwardPlan, GroupNormDescriptor, GroupNormPlan,
    InstanceNormArgs, InstanceNormBackwardArgs, InstanceNormBackwardDescriptor,
    InstanceNormBackwardPlan, InstanceNormDescriptor, InstanceNormPlan, LayerNormArgs,
    LayerNormBackwardArgs, LayerNormBackwardDescriptor, LayerNormBackwardPlan, LayerNormDescriptor,
    LayerNormPlan, RMSNormArgs, RMSNormBackwardArgs, RMSNormBackwardDescriptor,
    RMSNormBackwardPlan, RMSNormDescriptor, RMSNormPlan,
};

// Loss family — Phase 5 (Category R). MSE / NLL / CrossEntropy / BCE
// / KLDiv (FW + BW × 4 FP dtypes × {None, Mean, Sum} reduction).
pub mod loss;

pub use loss::{
    BceLossArgs, BceLossBackwardArgs, BceLossBackwardDescriptor, BceLossBackwardPlan,
    BceLossDescriptor, BceLossPlan, BceWithLogitsLossArgs, BceWithLogitsLossBackwardArgs,
    BceWithLogitsLossBackwardDescriptor, BceWithLogitsLossBackwardPlan,
    BceWithLogitsLossDescriptor, BceWithLogitsLossPlan, CrossEntropyLossArgs,
    CrossEntropyLossBackwardArgs, CrossEntropyLossBackwardDescriptor,
    CrossEntropyLossBackwardPlan, CrossEntropyLossDescriptor, CrossEntropyLossPlan,
    GaussianNllLossArgs, GaussianNllLossBackwardArgs, GaussianNllLossBackwardDescriptor,
    GaussianNllLossBackwardPlan, GaussianNllLossDescriptor, GaussianNllLossPlan, HuberLossArgs,
    HuberLossBackwardArgs, HuberLossBackwardDescriptor, HuberLossBackwardPlan,
    HuberLossDescriptor, HuberLossPlan, KlDivLossArgs, KlDivLossBackwardArgs,
    KlDivLossBackwardDescriptor, KlDivLossBackwardPlan, KlDivLossDescriptor, KlDivLossPlan,
    L1LossArgs, L1LossBackwardArgs, L1LossBackwardDescriptor, L1LossBackwardPlan,
    L1LossDescriptor, L1LossPlan, MseLossArgs, MseLossBackwardArgs, MseLossBackwardDescriptor,
    MseLossBackwardPlan, MseLossDescriptor, MseLossPlan, NllLossArgs, NllLossBackwardArgs,
    NllLossBackwardDescriptor, NllLossBackwardPlan, NllLossDescriptor, NllLossPlan,
    PoissonNllLossArgs, PoissonNllLossBackwardArgs, PoissonNllLossBackwardDescriptor,
    PoissonNllLossBackwardPlan, PoissonNllLossDescriptor, PoissonNllLossPlan, SmoothL1LossArgs,
    SmoothL1LossBackwardArgs, SmoothL1LossBackwardDescriptor, SmoothL1LossBackwardPlan,
    SmoothL1LossDescriptor, SmoothL1LossPlan,
};

pub use loss::{
    CosineEmbeddingLossArgs, CosineEmbeddingLossBackwardArgs,
    CosineEmbeddingLossBackwardDescriptor, CosineEmbeddingLossBackwardPlan,
    CosineEmbeddingLossDescriptor, CosineEmbeddingLossPlan, HingeEmbeddingLossArgs,
    HingeEmbeddingLossBackwardArgs, HingeEmbeddingLossBackwardDescriptor,
    HingeEmbeddingLossBackwardPlan, HingeEmbeddingLossDescriptor, HingeEmbeddingLossPlan,
    MarginRankingLossArgs, MarginRankingLossBackwardArgs, MarginRankingLossBackwardDescriptor,
    MarginRankingLossBackwardPlan, MarginRankingLossDescriptor, MarginRankingLossPlan,
    MultiMarginLossArgs, MultiMarginLossBackwardArgs, MultiMarginLossBackwardDescriptor,
    MultiMarginLossBackwardPlan, MultiMarginLossDescriptor, MultiMarginLossPlan,
    MultilabelMarginLossArgs, MultilabelMarginLossBackwardArgs,
    MultilabelMarginLossBackwardDescriptor, MultilabelMarginLossBackwardPlan,
    MultilabelMarginLossDescriptor, MultilabelMarginLossPlan, MultilabelSoftMarginLossArgs,
    MultilabelSoftMarginLossBackwardArgs, MultilabelSoftMarginLossBackwardDescriptor,
    MultilabelSoftMarginLossBackwardPlan, MultilabelSoftMarginLossDescriptor,
    MultilabelSoftMarginLossPlan, TripletMarginLossArgs, TripletMarginLossBackwardArgs,
    TripletMarginLossBackwardDescriptor, TripletMarginLossBackwardPlan,
    TripletMarginLossDescriptor, TripletMarginLossPlan,
};

// CTCLoss (Phase 5 Milestone 5.5) — DP-based sequence loss for
// variable-length inputs/targets.
pub use loss::{
    CtcLossArgs, CtcLossBackwardArgs, CtcLossBackwardDescriptor, CtcLossBackwardPlan,
    CtcLossDescriptor, CtcLossPlan,
};

// CTCLoss cuDNN sibling (Phase 7 Milestone 7.4) — same op, distinct
// backend; Fuel's autotuner races this against the bespoke plan.
// Gated behind the `cudnn` cargo feature.
#[cfg(feature = "cudnn")]
pub use loss::{CtcLossCudnnArgs, CtcLossCudnnDescriptor, CtcLossCudnnPlan};

// Random / sampling family — Phase 4.5 (Category Q). Uniform / Normal
// pass through cuRAND; Bernoulli + Dropout use bespoke kernels on top
// of cuRAND-uniform.
pub mod random;

pub use random::{
    DropoutArgs, DropoutBackwardArgs, DropoutBackwardDescriptor, DropoutBackwardPlan,
    DropoutDescriptor, DropoutPlan, RandomArgs, RandomBoolArgs, RandomDescriptor, RandomPlan,
};

// Attention family — Phase 6 (Category K). Milestone 6.1 ships the two
// positional-encoding ops: RoPE (rotary, Llama / Mistral / Gemma) and
// ALiBi (linear biases, MPT / BLOOM). FW + BW × 4 FP dtypes.
pub mod attention;

pub use attention::{
    AlibiArgs, AlibiBackwardArgs, AlibiBackwardDescriptor, AlibiBackwardPlan, AlibiDescriptor,
    AlibiPlan, FlashSdpaArgs, FlashSdpaBackwardArgs, FlashSdpaBackwardDescriptor,
    FlashSdpaBackwardPlan, FlashSdpaDescriptor, FlashSdpaPlan, KvCacheAppendArgs,
    KvCacheAppendDescriptor, KvCacheAppendPlan, RopeArgs, RopeBackwardArgs, RopeBackwardDescriptor,
    RopeBackwardPlan, RopeDescriptor, RopePlan, SdpaArgs, SdpaBackwardArgs, SdpaBackwardDescriptor,
    SdpaBackwardPlan, SdpaDescriptor, SdpaPlan, FLASH_SDPA_MAX_D, ROPE_DEFAULT_BASE,
};

// Dense linalg family — Milestone 6.3 (Category Linalg). Wraps
// cuSOLVER for Cholesky / LU / QR / SVD. f32 + f64 only (cuSOLVER's
// dense API does not expose f16 / bf16 for these ops).
pub mod linalg;

pub use linalg::{
    BatchedOrmqrArgs, BatchedOrmqrDescriptor, BatchedOrmqrOp, BatchedOrmqrPlan, BatchedOrmqrSide,
    BatchedOrmqrWyArgs, BatchedOrmqrWyDescriptor, BatchedOrmqrWyPlan, BatchedQrArgs,
    BatchedQrDescriptor, BatchedQrMaterializeArgs, BatchedQrMaterializeDescriptor,
    BatchedQrMaterializePlan, BatchedQrPlan, BatchedSvdArgs, BatchedSvdDescriptor, BatchedSvdPlan,
    BatchedSvdaArgs, BatchedSvdaDescriptor, BatchedSvdaPlan, CholeskyArgs, CholeskyDescriptor,
    CholeskyPlan, EigArgs, EigDescriptor, EigPlan, EighArgs, EighDescriptor, EighPlan, InverseArgs,
    InverseDescriptor, InversePlan, LstSqArgs, LstSqDescriptor, LstSqPlan, LuArgs, LuDescriptor,
    LuPlan, QrArgs, QrDescriptor, QrPlan, SolveArgs, SolveDescriptor, SolvePlan, SvdArgs,
    SvdDescriptor, SvdPlan, WY_NB,
};

// Convolution family — Phase 7 Milestone 7.1 (Category Convolution).
// Wraps cuDNN's legacy descriptor-based API. Today wired: NCHW Conv2d
// FW + BW data + BW filter × {f32, f64, f16, bf16}. 1-D / 3-D /
// transposed / depthwise variants follow in fanout milestones. Gated
// behind the `cudnn` cargo feature — cuDNN is a separate NVIDIA
// download not bundled with the stock CUDA toolkit.
#[cfg(feature = "cudnn")]
pub mod conv;

#[cfg(feature = "cudnn")]
pub use conv::{
    Conv2dArgs, Conv2dBwArgs, Conv2dDescriptor, Conv2dDwArgs, Conv2dPlan,
};

// Pooling family — Phase 7 Milestone 7.2 (Category Pooling). Wraps
// cuDNN's legacy pooling API. Today wired: NCHW MaxPool2d + AvgPool2d
// (FW + BW) × {f32, f64, f16, bf16}. 1-D / 3-D / adaptive / LP-pool /
// fractional-max-pool follow in fanout milestones. Gated behind the
// `cudnn` cargo feature.
#[cfg(feature = "cudnn")]
pub mod pool;

#[cfg(feature = "cudnn")]
pub use pool::{
    AvgPool2dPlan, MaxPool2dPlan, Pool2dBwArgs, Pool2dDescriptor, Pool2dFwArgs, PoolMode,
};

// FFT family — Milestone 6.4 (Category Fft). Wraps cuFFT for the four
// canonical 1-D PyTorch / JAX FFTs (FFT / IFFT / RFFT / IRFFT) plus
// the two bespoke index-permutation helpers (fftshift / ifftshift).
// f32 + f64 only (cuFFT's main API does not expose f16 / bf16).
pub mod fft;

pub use fft::{
    FftArgs, FftDescriptor, FftNdArgs, FftNdDescriptor, FftNdPlan, FftPlan, FftShiftArgs,
    FftShiftDescriptor, FftShiftNdArgs, FftShiftNdDescriptor, FftShiftNdPlan, FftShiftPlan,
    IrfftArgs, IrfftDescriptor, IrfftNdArgs, IrfftNdDescriptor, IrfftNdPlan, IrfftPlan, RfftArgs,
    RfftDescriptor, RfftNdArgs, RfftNdDescriptor, RfftNdPlan, RfftPlan, FFTSHIFT_ND_MAX_RANK,
    FFTSHIFT_ND_MAX_SHIFT_AXES,
};

// Indexing / scatter / gather family — Phase 7 Milestone 7.3 (Category L).
// Bespoke kernels for gather + gather_backward, scatter_add, index_select
// + index_select_backward, masked_fill + masked_fill_backward, one_hot,
// nonzero. Index dtype is i32 only (i64 deferred); out-of-bounds + negative
// indices are skipped (no PyTorch-style wrap-around).
pub mod indexing;

pub use indexing::{
    GatherArgs, GatherBackwardArgs, GatherBackwardDescriptor, GatherBackwardPlan,
    GatherDescriptor, GatherPlan, IndexSelectArgs, IndexSelectBackwardArgs,
    IndexSelectBackwardDescriptor, IndexSelectBackwardPlan, IndexSelectDescriptor,
    IndexSelectPlan, MaskedFillArgs, MaskedFillBackwardArgs, MaskedFillBackwardDescriptor,
    MaskedFillBackwardPlan, MaskedFillDescriptor, MaskedFillPlan, NonzeroArgs,
    NonzeroDescriptor, NonzeroPlan, OneHotArgs, OneHotDescriptor, OneHotPlan, ScatterAddArgs,
    ScatterAddDescriptor, ScatterAddPlan,
};

// Embedding family — Phase 7 Milestone 7.5 (Category M). Bespoke
// kernels for `embedding` (FW + BW) with optional `padding_idx` and
// `embedding_bag` (FW + BW × Sum / Mean modes). FW dtypes: f32 / f64 /
// f16 / bf16 (pure copy / accumulator-typed reduce); BW dtypes: f32 /
// f64 only (atomicAdd is native-FP). Max-mode for `embedding_bag` is
// deferred (needs per-feature argmax tracking).
pub mod embedding;

pub use embedding::{
    EmbeddingArgs, EmbeddingBackwardArgs, EmbeddingBackwardDescriptor, EmbeddingBackwardPlan,
    EmbeddingBagArgs, EmbeddingBagBackwardArgs, EmbeddingBagBackwardDescriptor,
    EmbeddingBagBackwardPlan, EmbeddingBagDescriptor, EmbeddingBagMode, EmbeddingBagPlan,
    EmbeddingDescriptor, EmbeddingPlan,
};

// Segment / scatter-reduce family — Phase 7 Milestone 7.6 (Category S).
// Sorted (binary-search single-pass sweep) and unsorted (atomicAdd /
// atomicMax-via-CAS / atomicMin-via-CAS) variants for sum / mean / max
// / min / prod. BW shipped for sum + mean (sorted and unsorted share
// the BW launcher); max / min / prod BW deferred (argmax tracking +
// stable prod-div). f32 + f64 only.
pub mod segment;

pub use segment::{
    SegmentMaxArgs, SegmentMaxDescriptor, SegmentMaxPlan, SegmentMeanArgs,
    SegmentMeanBackwardArgs, SegmentMeanBackwardDescriptor, SegmentMeanBackwardPlan,
    SegmentMeanDescriptor, SegmentMeanPlan, SegmentMinArgs, SegmentMinDescriptor, SegmentMinPlan,
    SegmentProdArgs, SegmentProdDescriptor, SegmentProdPlan, SegmentSumArgs,
    SegmentSumBackwardArgs, SegmentSumBackwardDescriptor, SegmentSumBackwardPlan,
    SegmentSumDescriptor, SegmentSumPlan, UnsortedSegmentMaxArgs, UnsortedSegmentMaxDescriptor,
    UnsortedSegmentMaxPlan, UnsortedSegmentMeanArgs, UnsortedSegmentMeanBackwardArgs,
    UnsortedSegmentMeanBackwardDescriptor, UnsortedSegmentMeanBackwardPlan,
    UnsortedSegmentMeanDescriptor, UnsortedSegmentMeanPlan, UnsortedSegmentMinArgs,
    UnsortedSegmentMinDescriptor, UnsortedSegmentMinPlan, UnsortedSegmentSumArgs,
    UnsortedSegmentSumBackwardArgs, UnsortedSegmentSumBackwardDescriptor,
    UnsortedSegmentSumBackwardPlan, UnsortedSegmentSumDescriptor, UnsortedSegmentSumPlan,
};

// Quantization family — Phase 8 (Category P). Split across two parallel
// milestones: 8.1 ships per-tensor / per-channel / fake_quantize plans;
// 8.2 ships per-token / per-group plans for LLM-style activation +
// weight quantization (W8A8 and INT4 GPTQ). Dtype coverage:
// {f32, f64, f16, bf16} × {s8, u8}. Backwards via STE for `quantize_*`
// and straight-through scaling for `dequantize_*`.
pub mod quantize;

pub use quantize::{
    DequantizePerGroupArgs, DequantizePerGroupBackwardArgs,
    DequantizePerGroupBackwardDescriptor, DequantizePerGroupBackwardPlan,
    DequantizePerGroupDescriptor, DequantizePerGroupPlan, DequantizePerTokenArgs,
    DequantizePerTokenBackwardArgs, DequantizePerTokenBackwardDescriptor,
    DequantizePerTokenBackwardPlan, DequantizePerTokenDescriptor, DequantizePerTokenPlan,
    QuantizePerGroupArgs, QuantizePerGroupBackwardArgs, QuantizePerGroupBackwardDescriptor,
    QuantizePerGroupBackwardPlan, QuantizePerGroupDescriptor, QuantizePerGroupPlan,
    QuantizePerTokenArgs, QuantizePerTokenBackwardArgs, QuantizePerTokenBackwardDescriptor,
    QuantizePerTokenBackwardPlan, QuantizePerTokenDescriptor, QuantizePerTokenPlan,
};

// Milestone 8.1 — per-tensor + per-channel + fake_quantize plan types.
pub use quantize::{
    DequantizePerChannelArgs, DequantizePerChannelBackwardArgs,
    DequantizePerChannelBackwardDescriptor, DequantizePerChannelBackwardPlan,
    DequantizePerChannelDescriptor, DequantizePerChannelPlan, DequantizePerTensorArgs,
    DequantizePerTensorBackwardArgs, DequantizePerTensorBackwardDescriptor,
    DequantizePerTensorBackwardPlan, DequantizePerTensorDescriptor, DequantizePerTensorPlan,
    FakeQuantizeArgs, FakeQuantizeBackwardArgs, FakeQuantizeBackwardDescriptor,
    FakeQuantizeBackwardPlan, FakeQuantizeDescriptor, FakeQuantizePlan, QuantizePerChannelArgs,
    QuantizePerChannelBackwardArgs, QuantizePerChannelBackwardDescriptor,
    QuantizePerChannelBackwardPlan, QuantizePerChannelDescriptor, QuantizePerChannelPlan,
    QuantizePerTensorArgs, QuantizePerTensorBackwardArgs, QuantizePerTensorBackwardDescriptor,
    QuantizePerTensorBackwardPlan, QuantizePerTensorDescriptor, QuantizePerTensorPlan,
};

// Milestone 8.3 — composing quantization ops (DynamicRangeQuantize +
// QuantizedLinear).
pub use quantize::{
    DynamicRangeMode, DynamicRangeQuantizeArgs, DynamicRangeQuantizeDescriptor,
    DynamicRangeQuantizePlan, DynamicRangeScope, QuantizedLinearArgs,
    QuantizedLinearDescriptor, QuantizedLinearPlan,
};

// Milestone 8.4 — GGUF block-format dequant + MMVQ (Category P).
// Vendored from llama.cpp via fuel-cuda-kernels.
pub use quantize::{
    BlockQ2K, BlockQ3K, BlockQ4_0, BlockQ4_1, BlockQ4K, BlockQ5_0, BlockQ5_1, BlockQ5K, BlockQ6K,
    BlockQ8_0, BlockQ8K, GgufDequantizeArgs, GgufDequantizeDescriptor, GgufDequantizePlan,
    GgufMmvqArgs, GgufMmvqDescriptor, GgufMmvqPlan,
};

// Milestone 8.5 — Mixture-of-Experts inference forward (Category V).
// Vendored from attention.rs via fuel-cuda-kernels.
pub mod moe;
pub use moe::{MoeArgs, MoeDescriptor, MoePlan, MoeVariant};
