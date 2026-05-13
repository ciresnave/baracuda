//! Element types and trait hierarchy shared across baracuda kernel
//! wrappers.
//!
//! [`Element`] is the base sealed trait every dtype implements; today
//! that's the floating-point family (`f16`, `bf16`, `f32`, [`F32Strict`],
//! `f64`) via [`Element`] itself with a `type Scalar: ScalarType` projection
//! (the alpha/beta scalar matching the kernel's accumulator), plus the
//! integer family via the sibling [`IntElement`] trait. [`BiasElement`]
//! marks the bias broadcast element types accepted by integer GEMM
//! epilogues.
//!
//! `Element` was originally named `CutlassElement` in the
//! `baracuda-cutlass` crate. The rename here unifies the vocabulary
//! across the wider kernel facade — `baracuda-cutlass` keeps the
//! `CutlassElement` name available as a re-export for back-compat.

use baracuda_types::DeviceRepr;
use half::{bf16, f16};

mod sealed {
    pub trait Sealed {}
}

mod scalar_sealed {
    pub trait Sealed {}
}

mod int_sealed {
    pub trait Sealed {}
}

mod bias_sealed {
    pub trait Sealed {}
}

/// Sealed marker for the alpha/beta scalar type an [`Element`] uses.
///
/// `f32` for f16/bf16/f32/[`F32Strict`] kernels (epilogue compute runs at
/// f32). `f64` for f64 kernels. Sealed to keep the kernel-side dispatch
/// closed — adding a new scalar type requires shipping new C ABI
/// signatures in the underlying `*-kernels-sys` crate.
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

/// Floating-point element types supported by the kernel facade.
///
/// Sealed to prevent downstream `impl`s — adding a new dtype requires
/// shipping a new kernel instantiation in the corresponding `*-kernels-sys`
/// crate.
///
/// `f32` and [`F32Strict`] are both implementations of this trait and
/// differ only in the math precision used inside the kernel:
/// - `f32` → TF32 tensor cores (10-bit mantissa, ~tensor-core-throughput)
/// - [`F32Strict`] → SIMT CUDA cores (full IEEE 754 binary32, bit-stable)
///
/// Both store inputs as IEEE 754 binary32; the choice of math op is
/// driven by which Rust type the caller picks.
pub trait Element: DeviceRepr + sealed::Sealed + Copy + 'static {
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

impl Element for f16 {
    const KIND: ElementKind = ElementKind::F16;
    type Scalar = f32;
}

impl Element for bf16 {
    const KIND: ElementKind = ElementKind::Bf16;
    type Scalar = f32;
}

/// `f32` GEMM routes through TF32 tensor cores — see
/// [`crate::PrecisionGuarantee::math_precision`] (returns
/// [`MathPrecision::Tf32`]). Inputs are full F32; the math instruction
/// reduces to TF32 (10-bit mantissa) and accumulates into F32. Use
/// [`F32Strict`] instead when bit-stable, full-precision IEEE 754
/// binary32 math is required.
impl Element for f32 {
    const KIND: ElementKind = ElementKind::F32;
    type Scalar = f32;
}

/// `f64` GEMM via Ampere FP64 tensor cores (DGEMM). Full IEEE 754
/// binary64 inputs, accumulator, and scalars. Analogous to cuBLAS's
/// `CUBLAS_COMPUTE_64F`.
impl Element for f64 {
    const KIND: ElementKind = ElementKind::F64;
    type Scalar = f64;
}

// ============================================================================
// Integer element family — sibling to Element
// ============================================================================

/// Signed 8-bit integer element marker. `#[repr(transparent)]` around
/// `i8`.
///
/// Identical memory layout to `i8`, so a `DeviceBuffer<i8>` (or any byte
/// substrate the caller has) can be reinterpreted as a `DeviceBuffer<S8>`
/// via `view_as` without copying. The wrapper exists to drive kernel
/// selection at the Rust type level: integer GEMM plans parameterized on
/// `S8` route the launch through the signed int8 tensor-core kernels.
///
/// Numerical contract: int8 inputs, int32 accumulator, float alpha/beta
/// scaling, saturating round-to-nearest cast back to int8 on store.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct S8(pub i8);

/// Unsigned 8-bit integer element marker. `#[repr(transparent)]` around
/// `u8`.
///
/// Identical memory layout to `u8`, so a `DeviceBuffer<u8>` (byte
/// substrate) can be reinterpreted as a `DeviceBuffer<U8>` (quantized
/// GEMM operand) via `view_as` without copying. The wrapper exists to
/// disambiguate "byte buffer" from "quantized operand" at the Rust type
/// level — a `DeviceBuffer<U8>` is unambiguously a GEMM operand,
/// `DeviceBuffer<u8>` stays a byte-storage abstraction.
///
/// Numerical contract: same as [`S8`] except the multiply operands are
/// unsigned. The accumulator is still int32 and alpha/beta are still
/// float; saturating cast at store clamps to `[0, 255]`.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct U8(pub u8);

// SAFETY: S8 / U8 are #[repr(transparent)] around i8 / u8, which are
// both DeviceRepr. Same ABI, same Copy + 'static bounds.
unsafe impl DeviceRepr for S8 {}
unsafe impl DeviceRepr for U8 {}

/// Integer element types supported by the int-GEMM kernel set.
///
/// Sibling trait to [`Element`] (the float family) — kept separate
/// because the kernel-level dispatch, accumulator type (int32 vs f32),
/// and epilogue family differ enough that mixing them through a single
/// trait would smear the type signatures of integer plans.
///
/// Sealed to prevent downstream `impl`s — adding a new int dtype
/// requires shipping new kernel instantiations.
pub trait IntElement: DeviceRepr + int_sealed::Sealed + Copy + 'static {
    /// Runtime tag for this element type. Always one of the integer
    /// variants of [`ElementKind`].
    const KIND: ElementKind;
}

impl int_sealed::Sealed for S8 {}
impl int_sealed::Sealed for U8 {}

impl IntElement for S8 {
    const KIND: ElementKind = ElementKind::S8;
}

impl IntElement for U8 {
    const KIND: ElementKind = ElementKind::U8;
}

/// Bias element types accepted by the int-GEMM bias epilogue family.
///
/// Integer GEMM kernels can broadcast either a per-channel `f32` bias
/// (matching the float bias convention used elsewhere) or a per-channel
/// `i32` bias (matching TensorRT's int8 inference convention). The
/// choice is a compile-time generic on integer plans — `<T, f32>` and
/// `<T, i32>` resolve to distinct kernel SKUs.
///
/// Sealed because the bias-element kernel variants are baked into the
/// `*-kernels-sys` crates at build time.
pub trait BiasElement: DeviceRepr + bias_sealed::Sealed + Copy + 'static {
    /// Runtime tag for this bias element type.
    const KIND: BiasElementKind;
}

impl bias_sealed::Sealed for f32 {}
impl bias_sealed::Sealed for i32 {}

impl BiasElement for f32 {
    const KIND: BiasElementKind = BiasElementKind::F32;
}
impl BiasElement for i32 {
    const KIND: BiasElementKind = BiasElementKind::I32;
}

/// Runtime tag for a [`BiasElement`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum BiasElementKind {
    /// IEEE 754 binary32 bias broadcast. The conservative default —
    /// matches the float-GEMM bias convention.
    F32,
    /// Signed 32-bit integer bias broadcast. Matches the convention
    /// TensorRT uses for int8 inference (per-channel int32 bias).
    I32,
}

/// Strict-precision f32 element marker.
///
/// `#[repr(transparent)]` wrapper around `f32`. Identical memory layout
/// to a plain `f32` device buffer — a `DeviceBuffer<f32>` can be
/// reinterpreted as a `DeviceBuffer<F32Strict>` via `view_as` without
/// copying. The wrapper exists purely to drive kernel selection at the
/// Rust type level: choosing the `F32Strict` element routes the launch
/// through the SIMT (CUDA-cores) GEMM kernels, while the plain `f32`
/// element routes through the TF32 tensor-core kernels.
///
/// Numerical contract: full IEEE 754 binary32 multiply-add throughout
/// (no tensor-core warp-reduction nondeterminism).
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct F32Strict(pub f32);

// SAFETY: F32Strict is #[repr(transparent)] around f32, which is itself
// DeviceRepr. Same ABI, same Copy + 'static bounds.
unsafe impl DeviceRepr for F32Strict {}

impl Element for F32Strict {
    const KIND: ElementKind = ElementKind::F32Strict;
    type Scalar = f32;
}

/// Runtime tag for an [`Element`] or [`IntElement`].
///
/// Unified across the float and integer kernel families so that a single
/// kernel-SKU descriptor can describe any baracuda kernel.
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
    /// Signed 8-bit integer. Maps to the [`S8`] wrapper type. Routed
    /// through Ampere int8 tensor cores (`mma.sync m16n8k32` integer
    /// variant) with int32 accumulation; float `alpha` / `beta` let
    /// the kernel act as a dequantize-in-epilogue.
    S8,
    /// Unsigned 8-bit integer. Maps to the [`U8`] wrapper type. Same
    /// kernel family as [`S8`] with unsigned operands.
    U8,
    /// Signed 32-bit integer. Surfaced only as the accumulator type of
    /// integer GEMM SKUs in [`crate::PrecisionGuarantee::accumulator`];
    /// not a valid kernel *input* element type and has no corresponding
    /// [`IntElement`] impl.
    I32,
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
    /// 8-bit integer multiply-add (`mma.sync m16n8k32` integer variant)
    /// with int32 accumulation. Used by both signed (s8) and unsigned
    /// (u8) integer GEMM SKUs; the multiply operands are 8-bit, the
    /// accumulator is 32-bit, and the multiply-add uses the
    /// `OpMultiplyAddSaturate` operator (clamps the accumulator on
    /// overflow rather than wrapping).
    Int8,
}
