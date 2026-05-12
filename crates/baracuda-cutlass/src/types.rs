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

mod scalar_sealed {
    pub trait Sealed {}
}

/// Sealed marker for the alpha/beta scalar type a [`CutlassElement`] uses.
///
/// `f32` for f16/bf16/f32/[`F32Strict`] kernels (epilogue compute runs at
/// f32). `f64` for f64 kernels. Sealed to keep the kernel-side dispatch
/// closed — adding a new scalar type requires shipping new C ABI
/// signatures in `baracuda-cutlass-kernels-sys`.
pub trait ScalarType: scalar_sealed::Sealed + Copy + Default + PartialEq + 'static {
    /// Discriminant used by the plan layer to dispatch to f32-scalar vs
    /// f64-scalar FFI entry points.
    const IS_F64: bool;

    /// Convert to `f32`. Used by the plan layer to feed the f32-scalar
    /// FFI dispatchers when `IS_F64` is `false` (round-trip is lossless
    /// because the underlying type IS `f32` in that branch). When called
    /// on the `f64` impl this is a narrowing cast — only callers that
    /// gate on `IS_F64 == false` should reach it.
    #[doc(hidden)]
    fn to_f32(self) -> f32;

    /// Convert to `f64`. Used by the plan layer to feed the f64-scalar
    /// FFI dispatchers when `IS_F64` is `true`. Lossless from both
    /// underlying types.
    #[doc(hidden)]
    fn to_f64(self) -> f64;
}

impl scalar_sealed::Sealed for f32 {}
impl scalar_sealed::Sealed for f64 {}

impl ScalarType for f32 {
    const IS_F64: bool = false;
    #[inline] fn to_f32(self) -> f32 { self }
    #[inline] fn to_f64(self) -> f64 { self as f64 }
}
impl ScalarType for f64 {
    const IS_F64: bool = true;
    #[inline] fn to_f32(self) -> f32 { self as f32 }
    #[inline] fn to_f64(self) -> f64 { self }
}

/// Element types supported by the CUTLASS kernel set.
///
/// Sealed to prevent downstream `impl`s — adding a new dtype requires
/// shipping a new kernel instantiation in `baracuda-cutlass-kernels-sys`.
///
/// `f32` and [`F32Strict`] are both implementations of this trait and
/// differ only in the math precision used inside the kernel:
/// - `f32` → TF32 tensor cores (10-bit mantissa, ~tensor-core-throughput)
/// - [`F32Strict`] → SIMT CUDA cores (full IEEE 754 binary32, bit-stable)
///
/// Both store inputs as IEEE 754 binary32; the choice of math op is
/// driven by which Rust type the caller picks.
pub trait CutlassElement: DeviceRepr + sealed::Sealed + Copy + 'static {
    /// Runtime tag for this element type.
    const KIND: ElementKind;
    /// Scalar type used for the kernel's alpha / beta parameters (and
    /// the epilogue compute type). `f32` for f16/bf16/f32/[`F32Strict`]
    /// — the epilogue runs at f32 to match the F32 accumulator. `f64`
    /// for [`prim@f64`] — the DGEMM path uses an F64 accumulator and
    /// f64 alpha/beta.
    type Scalar: ScalarType;
}

impl sealed::Sealed for f16 {}
impl sealed::Sealed for bf16 {}
impl sealed::Sealed for f32 {}
impl sealed::Sealed for F32Strict {}
impl sealed::Sealed for f64 {}

impl CutlassElement for f16 {
    const KIND: ElementKind = ElementKind::F16;
    type Scalar = f32;
}

impl CutlassElement for bf16 {
    const KIND: ElementKind = ElementKind::Bf16;
    type Scalar = f32;
}

/// `f32` GEMM routes through TF32 tensor cores — see
/// [`PrecisionGuarantee::math_precision`] (returns
/// [`MathPrecision::Tf32`]). Inputs are full F32; the math instruction
/// reduces to TF32 (10-bit mantissa) and accumulates into F32. Use
/// [`F32Strict`] instead when bit-stable, full-precision IEEE 754
/// binary32 math is required.
impl CutlassElement for f32 {
    const KIND: ElementKind = ElementKind::F32;
    type Scalar = f32;
}

/// `f64` GEMM via Ampere FP64 tensor cores (DGEMM). Full IEEE 754
/// binary64 inputs, accumulator, and scalars. Analogous to cuBLAS's
/// `CUBLAS_COMPUTE_64F`.
impl CutlassElement for f64 {
    const KIND: ElementKind = ElementKind::F64;
    type Scalar = f64;
}

/// Strict-precision f32 element marker.
///
/// `#[repr(transparent)]` wrapper around `f32`. Identical memory layout
/// to a plain `f32` device buffer — a `DeviceBuffer<f32>` can be
/// reinterpreted as a `DeviceBuffer<F32Strict>` via `view_as` without
/// copying. The wrapper exists purely to drive kernel selection at the
/// Rust type level: choosing `GemmPlan::<F32Strict>` routes the launch
/// through the SIMT (CUDA-cores) GEMM kernels, while
/// `GemmPlan::<f32>` routes through the TF32 tensor-core kernels.
///
/// Numerical contract: full IEEE 754 binary32 multiply-add throughout
/// (no tensor-core warp-reduction nondeterminism). See
/// [`PrecisionGuarantee`] returned by [`crate::GemmPlan::precision_guarantee`].
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct F32Strict(pub f32);

// SAFETY: F32Strict is #[repr(transparent)] around f32, which is itself
// DeviceRepr. Same ABI, same Copy + 'static bounds.
unsafe impl DeviceRepr for F32Strict {}

impl CutlassElement for F32Strict {
    const KIND: ElementKind = ElementKind::F32Strict;
    type Scalar = f32;
}

/// Runtime tag for a [`CutlassElement`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ElementKind {
    /// IEEE 754 binary16.
    F16,
    /// Brain-float 16.
    Bf16,
    /// IEEE 754 binary32 inputs reduced through TF32 tensor cores
    /// (10-bit mantissa). Maps to the `f32` Rust type.
    F32,
    /// IEEE 754 binary32 inputs reduced through SIMT CUDA cores at full
    /// f32 precision. Maps to the [`F32Strict`] wrapper type. Bit-stable
    /// on the same hardware.
    F32Strict,
    /// IEEE 754 binary64. Maps to the [`prim@f64`] Rust type.
    F64,
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
    /// IEEE 754 binary64 multiply-add via Ampere FP64 tensor cores
    /// (DGEMM).
    F64,
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
    /// across rows. Shipped for `{Rcr, Rrr} × {F16, Bf16, F32 (TF32),
    /// F32Strict (SIMT), F64 (DGEMM)}` on sm_80; the same coverage
    /// applies to every other `Bias*` variant.
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
    /// Multiplier on the matrix-multiply accumulator. Scalar type
    /// matches `T::Scalar` — `f32` for f16/bf16/f32/[`F32Strict`], `f64`
    /// for [`prim@f64`].
    pub alpha: T::Scalar,
    /// Multiplier on `c`. Forced to `0` internally when `c` is `None`,
    /// so callers don't need to pre-zero it for the no-accumulate case.
    pub beta: T::Scalar,
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
    /// α multiplier (shared across batches). Scalar type matches
    /// `T::Scalar` — `f32` for f16/bf16/f32/[`F32Strict`], `f64` for
    /// [`prim@f64`].
    pub alpha: T::Scalar,
    /// β multiplier (shared across batches). Forced to `0` internally
    /// when `c` is `None`.
    pub beta: T::Scalar,
}

/// One per-group entry for a grouped GEMM launch.
///
/// Each group has its own shape and pointers; CUTLASS dispatches them in
/// a single kernel invocation. Passed as a slice to
/// [`GroupedGemmPlan::prepare`](crate::GroupedGemmPlan::prepare), which
/// returns a [`PreparedGroupedGemm`](crate::PreparedGroupedGemm) whose
/// [`run`](crate::PreparedGroupedGemm::run) method performs the launch.
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
    /// α for this group. Scalar type matches `T::Scalar` — `f32` for
    /// f16/bf16/f32/[`F32Strict`], `f64` for [`prim@f64`].
    pub alpha: T::Scalar,
    /// β for this group. Forced to `0` internally when `c` is `None`.
    pub beta: T::Scalar,
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
            ElementKind::F32 => MathPrecision::Tf32,
            ElementKind::F32Strict => MathPrecision::F32,
            ElementKind::F64 => MathPrecision::F64,
        };
        // F32Strict uses SIMT CUDA cores — no warp-reduction nondeterminism,
        // so results are bit-identical across runs on the same hardware.
        // Tensor-core kernels (F16/Bf16/Tf32/F64) don't pin the warp-level
        // reduction order in the spec so they can differ in the last bit.
        let bit_stable_on_same_hardware = matches!(sku.element, ElementKind::F32Strict);
        // F64 GEMM accumulates into F64; all others accumulate into F32.
        let accumulator = match sku.element {
            ElementKind::F64 => ElementKind::F64,
            _ => ElementKind::F32,
        };
        Self {
            math_precision,
            accumulator,
            bit_stable_on_same_hardware,
            deterministic: true,
        }
    }
}
