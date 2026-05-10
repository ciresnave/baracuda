//! Value types for the CUTLASS plan-based API.
//!
//! All types here are pure data — no hidden device allocations, no
//! handles, no driver state. Plans cache *selection metadata* on top of
//! these descriptors but never own device memory.

use baracuda_driver::{DeviceSlice, DeviceSliceMut};
use baracuda_types::DeviceRepr;
use half::{bf16, f16};

mod sealed {
    pub trait Sealed {}
}

/// Element types supported by the v0 CUTLASS kernel set.
///
/// Implemented for `half::f16` and `half::bf16`. Sealed to prevent
/// downstream `impl`s — adding a new dtype requires shipping a new kernel
/// instantiation in `baracuda-cutlass-kernels-sys`.
pub trait CutlassElement: DeviceRepr + sealed::Sealed + Copy + 'static {
    /// Runtime tag for this element type.
    const KIND: ElementKind;
}

impl sealed::Sealed for f16 {}
impl sealed::Sealed for bf16 {}
impl sealed::Sealed for f32 {}

impl CutlassElement for f16 {
    const KIND: ElementKind = ElementKind::F16;
}

impl CutlassElement for bf16 {
    const KIND: ElementKind = ElementKind::Bf16;
}

/// `f32` GEMM routes through TF32 tensor cores — see
/// [`PrecisionGuarantee::math_precision`] (returns
/// [`MathPrecision::Tf32`]). Inputs are full F32; the math instruction
/// reduces to TF32 (10-bit mantissa) and accumulates into F32.
impl CutlassElement for f32 {
    const KIND: ElementKind = ElementKind::F32;
}

/// Runtime tag for a [`CutlassElement`].
///
/// Includes variants for element types that don't yet have a
/// [`CutlassElement`] impl (e.g. `F32`) so that
/// [`PrecisionGuarantee::accumulator`] can name them without a
/// cross-version churn when the corresponding input-side kernels land.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ElementKind {
    /// IEEE 754 binary16.
    F16,
    /// Brain-float 16.
    Bf16,
    /// IEEE 754 binary32. No `CutlassElement` impl yet (the f32 input
    /// kernel lands in a follow-up alpha); used today only as the
    /// accumulator tag in [`PrecisionGuarantee`].
    F32,
}

/// Math precision used by the FMA / tensor-core instruction.
///
/// Distinct from the *input* element type because tensor cores can take
/// inputs at one precision and reduce through an instruction at a
/// different precision (most notably TF32: F32 inputs, 10-bit-mantissa
/// math).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum MathPrecision {
    /// IEEE 754 binary16 multiply-add.
    F16,
    /// Brain-float 16 multiply-add.
    Bf16,
    /// TensorFloat-32 (10-bit mantissa) multiply-add. Inputs are stored
    /// as F32 but reduced through TF32 tensor cores.
    Tf32,
    /// IEEE 754 binary32 multiply-add (CUDA cores, no tensor cores).
    F32,
}

/// Numerical guarantees a CUTLASS GEMM kernel provides.
///
/// Surfaces the salient numerical properties consumers (e.g. Fuel's
/// per-decision-point alternatives layer) need to decide whether a
/// kernel SKU satisfies an op's precision contract — without having to
/// re-derive them from the README per kernel.
///
/// All fields are intentionally cheap to compare so this struct can be
/// hashed into selection / autotuner caches.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct PrecisionGuarantee {
    /// Bit-precision used inside the math instruction.
    pub math_precision: MathPrecision,
    /// Element type of the multiply-accumulate accumulator. All current
    /// kernels accumulate into F32 regardless of input dtype.
    pub accumulator: ElementKind,
    /// Whether the kernel produces bit-identical results across runs on
    /// the same hardware with the same inputs.
    ///
    /// `false` for tensor-core kernels (F16, BF16, TF32) because the
    /// warp-level reduction order isn't fixed by the spec — adjacent
    /// runs can differ in the last bit even with the same inputs.
    /// `true` for SIMT F32.
    pub bit_stable_on_same_hardware: bool,
    /// Whether the kernel produces bit-identical results across runs
    /// from a single thread within a process — i.e. it has no internal
    /// nondeterminism (no atomic accumulation across blocks, no random
    /// tile-schedule decisions).
    ///
    /// All current kernels are deterministic in this sense; the
    /// distinction from `bit_stable_on_same_hardware` is about
    /// cross-driver-version stability on the same input.
    pub deterministic: bool,
}


/// Layout SKU. Describes the row/column orientation of A, B, C, and D.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum LayoutSku {
    /// `A` row-major `[M, K]`, `B` column-major `[K, N]`, `C/D` row-major `[M, N]`.
    ///
    /// Useful when a row-major weight tensor stored as `[N, K]` is
    /// reinterpreted as logical column-major `B = [K, N]` without a
    /// transpose copy.
    Rcr,
    /// `A` row-major `[M, K]`, `B` row-major `[K, N]`, `C/D` row-major `[M, N]`.
    ///
    /// The natural shape for activation-row-major @ weight-row-major
    /// matmul (the typical ML graph layout). No transpose pass needed
    /// before launch — both operands stored in their native row-major
    /// form.
    Rrr,
}

/// Compute capability bucket the selected kernel was compiled for.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ArchSku {
    /// Ampere (also runs on Ada and as forward-compatible fallback on Hopper).
    Sm80,
    /// Hopper-specialized (requires `sm90a` feature).
    Sm90a,
}

/// Epilogue applied after the matrix-multiply accumulation.
///
/// The four `Bias*` variants share one kernel family: they all fuse the
/// bias add into the output epilogue via
/// `cutlass::gemm::device::GemmUniversalWithBroadcast`, and additionally
/// apply the named activation function before the store. `BiasRelu`,
/// `BiasGelu`, and `BiasSilu` therefore deliver the full
/// `y = activation(W·x + b)` transformer-Linear pipeline in a single
/// kernel pass — no extra memory traffic vs plain `Bias`.
///
/// [`GemmArgs::bias`] is required (`Some`) for any `Bias*` variant and
/// must be `None` for `Identity`. See [`EpilogueKind::requires_bias`]
/// and [`EpilogueKind::activation`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum EpilogueKind {
    /// `D = α · (A · B) + β · C` (no activation, no bias).
    Identity,
    /// `D = α · (A · B) + β · C + bias_broadcast(N)`. The bias vector
    /// has length `N` (one element per output column) and is broadcast
    /// across rows. v1 ships kernels for `Rcr × {F16, Bf16}` only.
    Bias,
    /// `D = relu(α · (A · B) + β · C + bias_broadcast(N))`.
    /// `relu(x) = max(x, 0)`. Same SKU coverage as [`Bias`](Self::Bias).
    BiasRelu,
    /// `D = gelu(α · (A · B) + β · C + bias_broadcast(N))` using the
    /// exact (erf-based) GELU — matches PyTorch's default `nn.GELU()`.
    /// For the `'tanh'` approximation file a follow-up; not yet shipped.
    /// Same SKU coverage as [`Bias`](Self::Bias).
    BiasGelu,
    /// `D = silu(α · (A · B) + β · C + bias_broadcast(N))` where
    /// `silu(x) = x · sigmoid(x)`. Also known as Swish-1.
    /// Same SKU coverage as [`Bias`](Self::Bias).
    BiasSilu,
}

impl EpilogueKind {
    /// `true` if [`GemmArgs::bias`] must be `Some` for this epilogue.
    /// Equivalent to "any `Bias*` variant".
    #[inline]
    pub const fn requires_bias(self) -> bool {
        matches!(
            self,
            Self::Bias | Self::BiasRelu | Self::BiasGelu | Self::BiasSilu,
        )
    }

    /// Activation function this epilogue applies after the linear
    /// combination, if any.
    ///
    /// Returns `None` for [`Identity`](Self::Identity) and
    /// [`Bias`](Self::Bias) (both apply no activation); returns the
    /// corresponding [`ActivationKind`] for the `Bias*Activation`
    /// variants.
    #[inline]
    pub const fn activation(self) -> Option<ActivationKind> {
        match self {
            Self::Identity | Self::Bias => None,
            Self::BiasRelu => Some(ActivationKind::Relu),
            Self::BiasGelu => Some(ActivationKind::Gelu),
            Self::BiasSilu => Some(ActivationKind::Silu),
        }
    }
}

/// Activation functions implemented by the `Bias*Activation`
/// [`EpilogueKind`] variants. Surfaced for telemetry and selector
/// logic; the kernel selection itself is driven by the enum variant.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ActivationKind {
    /// `relu(x) = max(x, 0)`.
    Relu,
    /// Exact (erf-based) Gaussian Error Linear Unit. Matches
    /// PyTorch's default `nn.GELU()`.
    Gelu,
    /// `silu(x) = x · sigmoid(x)`. Also known as Swish-1.
    Silu,
}

/// Caller-supplied workspace for a launch.
///
/// CUTLASS plans never own device memory in baracuda — pass scratch in at
/// `run` time. Pass [`Workspace::None`] for plans whose
/// [`workspace_size`](crate::GemmPlan::workspace_size) is zero.
#[derive(Debug)]
pub enum Workspace<'a> {
    /// No workspace (only valid when the plan reports zero bytes needed).
    None,
    /// Borrowed device scratch. Length must be at least the plan's
    /// reported `workspace_size`.
    Borrowed(DeviceSliceMut<'a, u8>),
}

/// Read-only view of a device-resident matrix.
///
/// `ld` is the leading dimension in **elements** (not bytes), measured
/// along the major axis dictated by the layout: row-stride for row-major
/// matrices, column-stride for column-major matrices.
#[derive(Debug)]
pub struct MatrixRef<'a, T: CutlassElement> {
    /// Device-resident element storage.
    pub data: DeviceSlice<'a, T>,
    /// Number of rows.
    pub rows: i32,
    /// Number of columns.
    pub cols: i32,
    /// Leading dimension in elements.
    pub ld: i64,
}

/// Mutable view of a device-resident matrix (used for the output `D`).
#[derive(Debug)]
pub struct MatrixMut<'a, T: CutlassElement> {
    /// Device-resident element storage.
    pub data: DeviceSliceMut<'a, T>,
    /// Number of rows.
    pub rows: i32,
    /// Number of columns.
    pub cols: i32,
    /// Leading dimension in elements.
    pub ld: i64,
}

/// Read-only view of a device-resident vector.
#[derive(Debug)]
pub struct VectorRef<'a, T: CutlassElement> {
    /// Device-resident element storage.
    pub data: DeviceSlice<'a, T>,
    /// Number of elements.
    pub len: i32,
    /// Stride in elements.
    pub stride: i64,
}

/// Problem shape and configuration handed to [`GemmPlan::select`](crate::GemmPlan::select).
#[derive(Copy, Clone, Debug)]
pub struct GemmDescriptor {
    /// Output row count.
    pub m: i32,
    /// Output column count.
    pub n: i32,
    /// Reduction depth.
    pub k: i32,
    /// Layout SKU. v0: must be [`LayoutSku::Rcr`].
    pub layout: LayoutSku,
    /// Epilogue kind. v0 ships only [`EpilogueKind::Identity`]; the `Bias`
    /// variant was removed during the Fuel team design review and will
    /// return when its kernel instantiation lands.
    pub epilogue: EpilogueKind,
}

/// Per-launch arguments for a [`GemmPlan::run`](crate::GemmPlan::run) call.
///
/// `c` is optional: when `None`, `β` is ignored at the safe layer (treated
/// as `0`) and the kernel computes `D = α · A · B`. When `Some`, the
/// kernel computes `D = α · A · B + β · C` — including the
/// `c.data == d.data` case for in-place accumulation.
///
/// `bias` is required iff the descriptor's epilogue is
/// [`EpilogueKind::Bias`], in which case the kernel computes
/// `D = α · A · B + β · C + bias_broadcast(N)`.
#[derive(Debug)]
pub struct GemmArgs<'a, T: CutlassElement> {
    /// Left input. Row-major `[M, K]`.
    pub a: MatrixRef<'a, T>,
    /// Right input. Layout depends on the descriptor's [`LayoutSku`]:
    /// column-major `[K, N]` for [`LayoutSku::Rcr`], row-major `[K, N]`
    /// for [`LayoutSku::Rrr`].
    pub b: MatrixRef<'a, T>,
    /// Optional accumulation source. Row-major `[M, N]`.
    pub c: Option<MatrixRef<'a, T>>,
    /// Output. Row-major `[M, N]`.
    pub d: MatrixMut<'a, T>,
    /// Optional bias vector. Required (`Some`) when the descriptor's
    /// epilogue is [`EpilogueKind::Bias`]; must be `None` for
    /// [`EpilogueKind::Identity`]. Length-`N`, contiguous (stride 1)
    /// device memory; broadcast across rows of `D`.
    pub bias: Option<VectorRef<'a, T>>,
    /// Multiplier on the matrix-multiply accumulator.
    pub alpha: f32,
    /// Multiplier on `c`. Forced to `0.0` internally when `c` is `None`,
    /// so callers don't need to pre-zero it for the no-accumulate case.
    pub beta: f32,
}

/// Problem shape and configuration handed to
/// [`BatchedGemmPlan::select`](crate::BatchedGemmPlan::select).
///
/// All batches share the same `(M, N, K)` and per-batch operands are
/// addressed by adding `i * stride_*` (in elements) to the base
/// pointer — see [`BatchedGemmArgs`]. For variable-shape grouped
/// problems use [`GroupedGemmPlan`](crate::GroupedGemmPlan) instead.
#[derive(Copy, Clone, Debug)]
pub struct BatchedGemmDescriptor {
    /// Output row count (per batch).
    pub m: i32,
    /// Output column count (per batch).
    pub n: i32,
    /// Reduction depth (per batch).
    pub k: i32,
    /// Number of batches launched in a single kernel invocation.
    pub batch_count: i32,
    /// Layout SKU. v1 supports only [`LayoutSku::Rcr`].
    pub layout: LayoutSku,
    /// Epilogue kind. v1 supports only [`EpilogueKind::Identity`].
    pub epilogue: EpilogueKind,
}

/// Per-launch arguments for a
/// [`BatchedGemmPlan::run`](crate::BatchedGemmPlan::run) call.
///
/// `stride_*` fields are in **elements**, not bytes — matching CUTLASS's
/// `GemmBatched` API. Pass `0` for stride if the same matrix should be
/// reused across all batches (broadcast).
#[derive(Debug)]
pub struct BatchedGemmArgs<'a, T: CutlassElement> {
    /// Left input — base pointer for batch 0.
    pub a: MatrixRef<'a, T>,
    /// Element offset between consecutive A batches.
    pub stride_a: i64,
    /// Right input — base pointer for batch 0.
    pub b: MatrixRef<'a, T>,
    /// Element offset between consecutive B batches.
    pub stride_b: i64,
    /// Optional accumulation source.
    pub c: Option<MatrixRef<'a, T>>,
    /// Element offset between consecutive C batches. Ignored when `c` is `None`.
    pub stride_c: i64,
    /// Output — base pointer for batch 0.
    pub d: MatrixMut<'a, T>,
    /// Element offset between consecutive D batches.
    pub stride_d: i64,
    /// α multiplier (shared across batches).
    pub alpha: f32,
    /// β multiplier (shared across batches). Forced to `0.0` internally
    /// when `c` is `None`.
    pub beta: f32,
}

/// One per-group entry for a grouped GEMM launch.
///
/// Each group has its own shape and pointers; CUTLASS dispatches them in
/// a single kernel invocation. Used by
/// [`GroupedGemmPlan::run`](crate::GroupedGemmPlan::run).
#[derive(Debug)]
pub struct GroupedProblem<'a, T: CutlassElement> {
    /// Group `M`.
    pub m: i32,
    /// Group `N`.
    pub n: i32,
    /// Group `K`.
    pub k: i32,
    /// Left input.
    pub a: MatrixRef<'a, T>,
    /// Right input.
    pub b: MatrixRef<'a, T>,
    /// Optional accumulation source.
    pub c: Option<MatrixRef<'a, T>>,
    /// Output.
    pub d: MatrixMut<'a, T>,
    /// α for this group.
    pub alpha: f32,
    /// β for this group. Forced to `0.0` internally when `c` is `None`.
    pub beta: f32,
}

/// Hints that influence kernel selection inside [`GemmPlan::select`](crate::GemmPlan::select).
#[derive(Copy, Clone, Debug)]
pub struct PlanPreference {
    /// Maximum workspace the caller is willing to provide. The selector
    /// only considers kernels whose `workspace_size` for the descriptor
    /// fits in this budget. Use `usize::MAX` to disable the constraint.
    pub max_workspace_bytes: usize,
    /// Allow Hopper-specialized (`sm_90a`) kernels in selection. Has no
    /// effect when the `sm90a` feature is off (no such kernels exist in
    /// the build).
    pub allow_sm90a: bool,
}

impl Default for PlanPreference {
    fn default() -> Self {
        Self {
            max_workspace_bytes: usize::MAX,
            allow_sm90a: true,
        }
    }
}

/// How CUTLASS schedules tiles across the grouped problem set.
///
/// v0 ships only [`GroupedScheduleMode::DeviceOnly`]. CUTLASS also offers
/// a `HostPrecompute` mode that pre-walks the schedule on the host and
/// uploads it; we'll add it later if profiling justifies the API surface.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum GroupedScheduleMode {
    /// All schedule decisions made on-device by the kernel itself.
    #[default]
    DeviceOnly,
}

/// Hints for [`GroupedGemmPlan::select`](crate::GroupedGemmPlan::select).
///
/// Wraps a [`PlanPreference`] for the underlying GEMM tile selection,
/// plus grouped-specific knobs.
#[derive(Copy, Clone, Debug, Default)]
pub struct GroupedPlanPreference {
    /// Tile-selection preferences (forwarded to the underlying GEMM picker).
    pub base: PlanPreference,
    /// CUTLASS schedule mode (v0: only [`GroupedScheduleMode::DeviceOnly`]).
    pub schedule: GroupedScheduleMode,
}

/// Identity of the kernel a plan picked.
///
/// Useful for caching plan selections in higher layers and for telemetry
/// (e.g., logging which SKU the autotuner picked).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct GemmSku {
    /// Architecture the kernel was compiled for.
    pub arch: ArchSku,
    /// Layout the kernel implements.
    pub layout: LayoutSku,
    /// Epilogue the kernel implements.
    pub epilogue: EpilogueKind,
    /// Element type the kernel operates on.
    pub element: ElementKind,
}

impl PrecisionGuarantee {
    /// Numerical guarantees for the kernel identified by `sku`.
    ///
    /// Pure host-side lookup; returns the same value for the same SKU
    /// across calls. The mapping is part of the public contract: a
    /// stable SKU implies a stable precision guarantee.
    pub fn for_sku(sku: GemmSku) -> Self {
        // All shipped kernels accumulate into F32 and use tensor cores.
        // None use cross-block atomics, so all are deterministic.
        // Tensor-core warp-reduction order isn't pinned by the spec, so
        // cross-driver bit-stability is not guaranteed.
        let math_precision = match sku.element {
            ElementKind::F16 => MathPrecision::F16,
            ElementKind::Bf16 => MathPrecision::Bf16,
            // Reserved for the f32-input kernel: routes through TF32
            // tensor cores under the hood (not full F32 SIMT).
            ElementKind::F32 => MathPrecision::Tf32,
        };
        Self {
            math_precision,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: false,
            deterministic: true,
        }
    }
}
