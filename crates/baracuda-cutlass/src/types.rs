//! Value types for the CUTLASS plan-based API.
//!
//! All types here are pure data — no hidden device allocations, no
//! handles, no driver state. Plans cache *selection metadata* on top of
//! these descriptors but never own device memory.
//!
//! The shared type vocabulary (element / layout / epilogue / matrix-view
//! / plan-preference / precision-guarantee / workspace) lives in
//! [`baracuda_kernels_types`]. This module re-exports it for back-compat
//! and additionally hosts the CUTLASS-specific descriptors (GEMM /
//! batched GEMM / grouped GEMM / int-GEMM problem + args structs and
//! the [`GemmSku`] tag) that aren't shared with the wider facade.
//!
//! The trait formerly known as `CutlassElement` is now
//! [`baracuda_kernels_types::Element`]; the old name is preserved as a
//! type alias below (see [`CutlassElement`]).

// Re-export the shared vocabulary so downstream callers that import
// from `baracuda_cutlass::types::*` (and any code inside this crate
// referencing these types via `crate::types::Foo`) continues to work
// unchanged.
pub use baracuda_kernels_types::{
    ActivationKind, ArchSku, BiasElement, BiasElementKind, Element, ElementKind, EpilogueKind,
    F32Strict, IntElement, LayoutSku, MathPrecision, MatrixMut, MatrixRef, PlanPreference,
    PrecisionGuarantee, S8, ScalarType, U8, VectorRef, Workspace,
};

/// Back-compat alias for [`Element`].
///
/// Originally the float-family element trait was named `CutlassElement`;
/// it was renamed to `Element` in workspace alpha.16 when the shared
/// type vocabulary moved into [`baracuda_kernels_types`]. The old name
/// is preserved here so existing downstream imports keep working.
///
/// Prefer importing `Element` from `baracuda_kernels_types` (or, via
/// re-export, from `baracuda_cutlass` / `baracuda_kernels`) in new code.
pub use baracuda_kernels_types::Element as CutlassElement;

/// Problem shape and configuration handed to [`GemmPlan::select`](crate::GemmPlan::select).
#[derive(Copy, Clone, Debug)]
pub struct GemmDescriptor {
    /// Output row count.
    pub m: i32,
    /// Output column count.
    pub n: i32,
    /// Reduction depth.
    pub k: i32,
    /// Layout SKU.
    pub layout: LayoutSku,
    /// Epilogue kind.
    pub epilogue: EpilogueKind,
}

/// Per-launch arguments for a [`GemmPlan::run`](crate::GemmPlan::run) call.
///
/// `c` is optional: when `None`, `β` is ignored at the safe layer (treated
/// as `0`) and the kernel computes `D = α · A · B`. When `Some`, the
/// kernel computes `D = α · A · B + β · C` — including the
/// `c.data == d.data` case for in-place accumulation.
///
/// `bias` is required iff the descriptor's epilogue is one of the
/// `Bias*` variants, in which case the kernel computes
/// `D = activation(α · A · B + β · C + bias_broadcast(N))`.
#[derive(Debug)]
pub struct GemmArgs<'a, T: Element> {
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
    /// epilogue is any `Bias*` variant; must be `None` for
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
pub struct BatchedGemmArgs<'a, T: Element> {
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
pub struct GroupedProblem<'a, T: Element> {
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

/// Problem shape and configuration handed to
/// [`IntGemmPlan::select`](crate::IntGemmPlan::select).
///
/// Parallel to [`GemmDescriptor`] for the integer GEMM family.
/// `LayoutSku` and [`EpilogueKind`] are shared with the float family,
/// but coverage on int8 is limited to [`LayoutSku::Rcr`] in this
/// release — selecting [`LayoutSku::Rrr`] returns
/// [`Error::Unsupported`](crate::Error::Unsupported). The
/// `RowMajor × RowMajor` integer SKU lives in the bespoke
/// `baracuda-kernels-sys` kernel family (lands in workspace alpha.16).
#[derive(Copy, Clone, Debug)]
pub struct IntGemmDescriptor {
    /// Output row count.
    pub m: i32,
    /// Output column count.
    pub n: i32,
    /// Reduction depth.
    pub k: i32,
    /// Layout SKU. Today's int8 CUTLASS SKUs require [`LayoutSku::Rcr`].
    pub layout: LayoutSku,
    /// Epilogue kind. All five variants are supported on int8 RCR.
    pub epilogue: EpilogueKind,
}

/// Per-launch arguments for an
/// [`IntGemmPlan::run`](crate::IntGemmPlan::run) call.
///
/// Parallel to [`GemmArgs`] for the integer GEMM family. The matrix
/// operands carry the kernel element type `T: IntElement`
/// (today: [`S8`] or [`U8`]); the optional `bias` carries the
/// independent bias element type `BT: BiasElement` (today: `f32` or
/// `i32`). Scalar `alpha` / `beta` are always `f32` regardless of `T`
/// or `BT` — CUTLASS's `LinearCombinationClamp` /
/// `LinearCombinationBiasElementwise` epilogues do the entire
/// alpha/beta/bias/activation chain in float (after int32→float
/// dequant of the accumulator) and saturating-cast back to the int
/// output range on store.
#[derive(Debug)]
pub struct IntGemmArgs<'a, T: IntElement, BT: BiasElement = f32> {
    /// Left input. Row-major `[M, K]`.
    pub a: MatrixRef<'a, T>,
    /// Right input. Column-major `[K, N]` (RCR).
    pub b: MatrixRef<'a, T>,
    /// Optional accumulation source. Row-major `[M, N]`.
    pub c: Option<MatrixRef<'a, T>>,
    /// Output. Row-major `[M, N]`.
    pub d: MatrixMut<'a, T>,
    /// Optional bias vector. Required when the descriptor's epilogue
    /// is any `Bias*` variant; must be `None` for
    /// [`EpilogueKind::Identity`]. Length-`N`, contiguous (stride 1)
    /// device memory; broadcast across rows of `D`.
    pub bias: Option<VectorRef<'a, BT>>,
    /// Multiplier on the matrix-multiply accumulator. Always `f32`
    /// for int GEMM — CUTLASS does the entire epilogue compute in
    /// float space.
    pub alpha: f32,
    /// Multiplier on `c`. Forced to `0` internally when `c` is `None`.
    pub beta: f32,
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
///
/// `bias_element` distinguishes int-GEMM bias kernels at the SKU level:
/// the same `(arch, layout, epilogue=Bias, element=S8)` tuple maps to
/// two distinct kernels depending on whether the bias broadcast is `f32`
/// or `i32`. Float-GEMM bias kernels and Identity kernels leave this
/// field `None` because the bias element (when present) is implied by
/// `element`.
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
    /// Bias broadcast element type. `Some` only for int-GEMM bias
    /// kernels (which can have either `f32` or `i32` bias);
    /// `None` for Identity kernels and for float-GEMM bias kernels
    /// (where the bias element is implied to match `element`).
    pub bias_element: Option<BiasElementKind>,
}

impl GemmSku {
    /// Numerical guarantees for the kernel identified by this SKU.
    ///
    /// Pure host-side lookup; returns the same value for the same SKU
    /// across calls. The mapping is part of the public contract: a
    /// stable SKU implies a stable precision guarantee.
    pub fn precision_guarantee(self) -> PrecisionGuarantee {
        // All shipped kernels accumulate into F32 (floats) or int32
        // (int8). None use cross-block atomics, so all are deterministic.
        // Tensor-core warp-reduction order isn't pinned by the spec for
        // float MMA, so float tensor-core kernels are not bit-stable
        // cross-driver. Integer MMA reductions are deterministic — the
        // int32 accumulator has no rounding nondeterminism — so int8
        // kernels ARE bit-stable on the same hardware.
        let math_precision = match self.element {
            ElementKind::F16 => MathPrecision::F16,
            ElementKind::Bf16 => MathPrecision::Bf16,
            ElementKind::F32 => MathPrecision::Tf32,
            ElementKind::F32Strict => MathPrecision::F32,
            ElementKind::F64 => MathPrecision::F64,
            ElementKind::S8 | ElementKind::U8 => MathPrecision::Int8,
            // `I32` is an accumulator-only kind, never a kernel input
            // element. A `GemmSku` constructed with `element = I32` is
            // a programming error; report Int8 math precision (the
            // only int kernel family that produces an int32 accum) as
            // a defensive fallback.
            ElementKind::I32 => MathPrecision::Int8,
            // FP8 kernels live in baracuda-kernels-sys, not baracuda-cutlass.
            // No CUTLASS SKU produces these element kinds; defensive arm.
            ElementKind::Fp8E4M3 => MathPrecision::Fp8E4M3,
            ElementKind::Fp8E5M2 => MathPrecision::Fp8E5M2,
            // Int4 kernels (S4 / U4) live in baracuda-kernels-sys, not
            // baracuda-cutlass. Defensive arm.
            ElementKind::S4 | ElementKind::U4 => MathPrecision::Int4,
            // Binary (Bin) GEMM lives in baracuda-kernels-sys. Defensive arm.
            ElementKind::Bin => MathPrecision::Binary,
        };
        // F32Strict (SIMT CUDA cores) and int8 (integer tensor cores)
        // are bit-stable on the same hardware. Float tensor-core
        // kernels (F16 / Bf16 / Tf32 / F64) don't pin the warp-level
        // reduction order so they can differ in the last bit.
        let bit_stable_on_same_hardware = matches!(
            self.element,
            ElementKind::F32Strict
                | ElementKind::S8
                | ElementKind::U8
                | ElementKind::S4
                | ElementKind::U4
                | ElementKind::Bin,
        );
        let accumulator = match self.element {
            ElementKind::F64 => ElementKind::F64,
            ElementKind::S8
            | ElementKind::U8
            | ElementKind::S4
            | ElementKind::U4
            | ElementKind::Bin => ElementKind::I32,
            _ => ElementKind::F32,
        };
        PrecisionGuarantee {
            math_precision,
            accumulator,
            bit_stable_on_same_hardware,
            deterministic: true,
        }
    }
}

