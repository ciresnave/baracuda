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
    Complex64, CrossEntropyTargetKind, Element, ElementKind, EpilogueKind, F32Strict, FftKind,
    FillMode, Fp8E4M3, Fp8E5M2, FpElement, GatedActivationKind, IntElement, KernelSku, LayoutSku,
    LinalgKind, LossKind, LossReduction, MathPrecision, MatrixMut, MatrixRef, NormalizationKind,
    OpCategory, PadMode, PlanPreference, PrecisionGuarantee, RandomKind, ReduceKind, S4, S8,
    ScalarType, ScanKind, ShapeLayoutKind, SoftmaxKind, TensorMut, TensorRef, TernaryKind, U4, U8,
    UnaryKind, VectorRef, Workspace,
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
    BinaryArgs, BinaryBackwardArgs, BinaryBackwardDescriptor, BinaryBackwardPlan, BinaryCmpArgs,
    BinaryCmpDescriptor, BinaryCmpPlan, BinaryDescriptor, BinaryParamArgs,
    BinaryParamBackwardArgs, BinaryParamBackwardDescriptor, BinaryParamBackwardPlan,
    BinaryParamDescriptor, BinaryParamPlan, BinaryPlan, GatedActivationArgs,
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
    ConcatDescriptor, ConcatPlan, FlipArgs, FlipBackwardArgs, FlipBackwardDescriptor,
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
