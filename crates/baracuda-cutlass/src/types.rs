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

impl CutlassElement for f16 {
    const KIND: ElementKind = ElementKind::F16;
}

impl CutlassElement for bf16 {
    const KIND: ElementKind = ElementKind::Bf16;
}

/// Runtime tag for a [`CutlassElement`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ElementKind {
    /// IEEE 754 binary16.
    F16,
    /// Brain-float 16.
    Bf16,
}

/// Layout SKU. Describes the row/column orientation of A, B, C, and D.
///
/// v0 supports only [`LayoutSku::Rcr`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum LayoutSku {
    /// `A` row-major `[M, K]`, `B` column-major `[K, N]`, `C/D` row-major `[M, N]`.
    ///
    /// This is the most useful ML layout: a row-major weight tensor stored
    /// as `[N, K]` can be passed as logical column-major `B` without a
    /// transpose copy.
    Rcr,
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
/// v0 ships only [`Identity`](EpilogueKind::Identity). The `Bias` variant
/// was removed during the Fuel team's design review because the safe API
/// would silently drop bias values until the corresponding kernel
/// instantiation lands. Once a `LinearCombinationBias` kernel ships in a
/// follow-up sub-phase, this enum will gain a `Bias` variant — and the
/// `bias` field will return to [`GemmArgs`] / [`GroupedProblem`] — at the
/// same time.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum EpilogueKind {
    /// `D = α · (A · B) + β · C` (no activation).
    Identity,
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
#[derive(Debug)]
pub struct GemmArgs<'a, T: CutlassElement> {
    /// Left input. Row-major `[M, K]`.
    pub a: MatrixRef<'a, T>,
    /// Right input. Column-major `[K, N]`.
    pub b: MatrixRef<'a, T>,
    /// Optional accumulation source. Row-major `[M, N]`.
    pub c: Option<MatrixRef<'a, T>>,
    /// Output. Row-major `[M, N]`.
    pub d: MatrixMut<'a, T>,
    /// Multiplier on the matrix-multiply accumulator.
    pub alpha: f32,
    /// Multiplier on `c`. Forced to `0.0` internally when `c` is `None`,
    /// so callers don't need to pre-zero it for the no-accumulate case.
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
