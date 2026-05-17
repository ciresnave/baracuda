//! Element types and trait hierarchy shared across baracuda kernel
//! wrappers.
//!
//! [`Element`] is the base sealed trait every dtype implements; today
//! that's the floating-point family (`f16`, `bf16`, `f32`, [`F32Strict`],
//! `f64`) via [`Element`] itself with a `type Scalar: ScalarType` projection
//! (the alpha/beta scalar matching the kernel's accumulator), plus the
//! integer family via the sibling [`IntElement`] trait and the 8-bit
//! floating-point family via the sibling [`FpElement`] trait.
//! [`BiasElement`] marks the bias broadcast element types accepted by
//! integer GEMM epilogues.
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

mod fp_sealed {
    pub trait Sealed {}
}

mod bin_sealed {
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

/// Element types supported by the kernel facade.
///
/// Sealed to prevent downstream `impl`s — adding a new dtype requires
/// shipping a new kernel instantiation in the corresponding `*-kernels-sys`
/// crate.
///
/// The trait spans three families that share the `<T: Element>`-
/// parameterized plan shape but route through distinct kernel SKUs:
///
/// - **Floating-point**: `f16`, `bf16`, `f32`, [`F32Strict`], `f64`.
///   `f32` reduces through TF32 tensor cores (10-bit mantissa);
///   [`F32Strict`] uses SIMT CUDA cores at full IEEE 754 binary32 with
///   bit-stable results. The `Scalar` projection is `f32` for the
///   16-bit / 32-bit float members and `f64` for `f64`.
/// - **Integer**: `i32`, `i64`. Used for elementwise integer arithmetic
///   (bitwise ops, integer comparison). The `Scalar` projection is
///   `f32` — these types don't participate in α/β-scaled epilogues, so
///   the projection is nominal. Note: [`S8`] / [`U8`] / [`S4`] / [`U4`]
///   are GEMM-only operand types and live on the separate [`IntElement`]
///   trait — they don't implement [`Element`].
/// - **Boolean**: [`Bool`] (1-byte storage, 0/non-zero truthiness).
///   Used for logical ops and as the output type of comparison ops.
///   The `Scalar` projection is `f32` (also nominal).
///
/// Sibling traits [`IntElement`], [`FpElement`], [`BinElement`], and
/// [`BiasElement`] cover GEMM-only / FP8 / packed-bit / bias-broadcast
/// types respectively; those have their own kernel families and don't
/// route through `<T: Element>`-parameterized elementwise plans.
pub trait Element: DeviceRepr + sealed::Sealed + Copy + 'static {
    /// Runtime tag for this element type.
    const KIND: ElementKind;
    /// Scalar type used for the kernel's alpha / beta parameters (and
    /// the epilogue compute type). `f32` for f16/bf16/f32/[`F32Strict`]
    /// — the epilogue runs at f32 to match the F32 accumulator. `f64`
    /// for [`prim@f64`] — the DGEMM path uses an F64 accumulator and
    /// f64 alpha/beta. For integer / [`Bool`] elements the projection
    /// is nominally `f32` (no α/β-scaled epilogue applies).
    type Scalar: ScalarType;
}

impl sealed::Sealed for f16 {}
impl sealed::Sealed for bf16 {}
impl sealed::Sealed for f32 {}
impl sealed::Sealed for F32Strict {}
impl sealed::Sealed for f64 {}
impl sealed::Sealed for i32 {}
impl sealed::Sealed for i64 {}
impl sealed::Sealed for Bool {}

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

/// `i32` as an elementwise kernel input element. Used by the integer
/// arithmetic kernels (bitwise and / or / xor / shift, integer
/// comparison, integer scans). Distinct from [`ElementKind::I32`]'s
/// historical use as an accumulator-only marker for integer GEMMs —
/// here `i32` is a first-class kernel *input* type with an `Element`
/// impl, so the same `BinaryPlan<T, N>` / `UnaryPlan<T, N>` shapes
/// extend to integer arithmetic.
///
/// The `Scalar` projection is `f32` (nominal — integer kernels don't
/// use α/β-scaled epilogues today).
impl Element for i32 {
    const KIND: ElementKind = ElementKind::I32;
    type Scalar = f32;
}

/// `i64` as an elementwise kernel input element. Sibling of the `i32`
/// impl above for 64-bit integer arithmetic (PyTorch's default integer
/// tensor dtype). Same kernel families, twice the storage width.
impl Element for i64 {
    const KIND: ElementKind = ElementKind::I64;
    type Scalar = f32;
}

/// Boolean as an elementwise kernel input element. Used by the logical
/// op family (`logical_and` / `logical_or` / `logical_xor`) and as the
/// output type of comparison ops. Storage is 1 byte per element via the
/// [`Bool`] wrapper.
///
/// The `Scalar` projection is `f32` (nominal).
impl Element for Bool {
    const KIND: ElementKind = ElementKind::Bool;
    type Scalar = f32;
}

impl sealed::Sealed for Complex32 {}
impl sealed::Sealed for Complex64 {}

/// Single-precision complex (interleaved real/imag pair of `f32`) as an
/// elementwise kernel input element. Used by the FFT family (`fft`,
/// `ifft`, `rfft` output / `irfft` input, etc.) for spectrum-domain
/// tensors. The `Scalar` projection is `f32` (matches the real width).
impl Element for Complex32 {
    const KIND: ElementKind = ElementKind::Complex32;
    type Scalar = f32;
}

/// Double-precision complex (interleaved real/imag pair of `f64`) as an
/// elementwise kernel input element. Sibling to [`Complex32`]; the
/// `Scalar` projection is `f64`.
impl Element for Complex64 {
    const KIND: ElementKind = ElementKind::Complex64;
    type Scalar = f64;
}

// ============================================================================
// Boolean element type — implements Element directly
// ============================================================================

/// Boolean element marker. `#[repr(transparent)]` wrapper around `u8`
/// (1-byte storage).
///
/// Truthiness convention follows PyTorch / NumPy: `0` is false; **any**
/// non-zero byte is true. Kernels that consume `Bool` operands normalize
/// the input to `0` or `1` before applying the logical op so the result
/// is always strictly `0` or `1`. The wrapper is `#[repr(transparent)]`
/// over `u8`, so a `DeviceBuffer<u8>` (byte substrate) can be
/// reinterpreted as a `DeviceBuffer<Bool>` via `view_as` without
/// copying.
///
/// Used as the element type of comparison-op output tensors (`eq`, `gt`,
/// …) and as the input element type for the logical-op family
/// (`logical_and`, `logical_or`, `logical_xor`). Implements [`Element`]
/// so the same `BinaryPlan<T, N>` / `UnaryPlan<T, N>` shapes extend to
/// boolean tensors.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Bool(pub u8);

impl Bool {
    /// Build a [`Bool`] from a Rust `bool`. `true` becomes `1`, `false`
    /// becomes `0`.
    #[inline]
    pub const fn new(b: bool) -> Self {
        Self(b as u8)
    }

    /// Convert to a Rust `bool` using the PyTorch convention: any
    /// non-zero byte is true.
    #[inline]
    pub const fn to_bool(self) -> bool {
        self.0 != 0
    }
}

// SAFETY: Bool is #[repr(transparent)] around u8, which is DeviceRepr.
// Same ABI, same Copy + 'static bounds.
unsafe impl DeviceRepr for Bool {}

// ============================================================================
// Complex element types — implements Element directly
// ============================================================================

/// Single-precision complex element. `#[repr(C)]` struct of two `f32`
/// fields (real, imag) — ABI-compatible with cuFFT's `cufftComplex`
/// (which is itself an alias for CUDA's `float2`), with NumPy's
/// `complex64`, and with PyTorch's `torch.complex64`.
///
/// Used by the FFT op family (Milestone 6.4) as the element type for
/// spectrum-domain tensors. Complex arithmetic is not a kernel concern
/// at this layer — Rust callers build / inspect complex values via the
/// `re` / `im` fields and pass `DeviceBuffer<Complex32>` directly to
/// the FFT plans, which reinterpret them as `cufftComplex` over the
/// FFI boundary.
///
/// Layout invariant: `Complex32 { re, im }` and `cufftComplex { x, y }`
/// share identical byte storage on every platform CUDA supports
/// (`(f32, f32)` is 8-byte aligned, naturally padded). A
/// `DeviceBuffer<Complex32>` can be reinterpreted as a
/// `DeviceBuffer<cufftComplex>` via `view_as` without copying.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Complex32 {
    /// Real component.
    pub re: f32,
    /// Imaginary component.
    pub im: f32,
}

impl Complex32 {
    /// Build a `Complex32` from real and imaginary `f32` parts.
    #[inline]
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }
}

/// Double-precision complex element. `#[repr(C)]` struct of two `f64`
/// fields — ABI-compatible with cuFFT's `cufftDoubleComplex`, NumPy's
/// `complex128`, and PyTorch's `torch.complex128`. Sibling to
/// [`Complex32`].
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Complex64 {
    /// Real component.
    pub re: f64,
    /// Imaginary component.
    pub im: f64,
}

impl Complex64 {
    /// Build a `Complex64` from real and imaginary `f64` parts.
    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
}

// SAFETY: Complex32 / Complex64 are #[repr(C)] structs of two FP fields
// each, with no padding (8-byte and 16-byte natural alignment), so they
// satisfy DeviceRepr's invariants (no uninitialized bytes, no host-side
// resource handles, byte-for-byte transferable between host and device).
unsafe impl DeviceRepr for Complex32 {}
unsafe impl DeviceRepr for Complex64 {}

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

/// Signed 4-bit integer element marker — **packed-pair storage**.
///
/// `#[repr(transparent)]` around `u8`. One [`S4`] *storage slot* is one
/// byte and holds **two** packed s4 elements: the low nibble is the
/// element at even logical index, the high nibble is the element at
/// odd logical index (along the K axis for A/B operands, along the
/// N axis for D output). Sign-extended to s32 on the GPU side via
/// `((s8)(nibble << 4)) >> 4`.
///
/// A `DeviceBuffer<u8>` of `(M*K)/2` bytes can be reinterpreted as a
/// `DeviceBuffer<S4>` of `(M*K)/2` storage slots via `view_as` without
/// copying — `S4` is byte-storage at the buffer layer, and *element
/// count* lives at the plan-layer descriptor (M / N / K).
///
/// Numerical range per element: `[-8, +7]`. The plan layer
/// (`Int4GemmPlan` in `baracuda-kernels`) takes `M`, `N`, `K` in
/// **element** counts and leading dimensions in **storage-slot
/// (= byte)** counts — `MatrixRef<S4>::ld` therefore equals `K / 2` for
/// row-major A with no padding. `K` must be even (packing is byte-
/// aligned). Routes through Ada Lovelace int4 tensor cores
/// (`mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.s4.s32`) with
/// S32 accumulation and float `alpha` / `beta` scaling. First landed in
/// baracuda-kernels Phase 2.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct S4(pub u8);

/// Unsigned 4-bit integer element marker — **packed-pair storage**.
///
/// `#[repr(transparent)]` around `u8`. Packing convention is identical
/// to [`S4`] (low nibble = even index, high nibble = odd index); the
/// only difference is zero-extension to s32 on the GPU side
/// (`nibble & 0xF`).
///
/// Numerical range per element: `[0, 15]`. Plan-layer conventions
/// (M/N/K in elements, LDs in storage slots, K even) match [`S4`].
/// Routes through Ada Lovelace int4 tensor cores
/// (`mma.sync.aligned.m16n8k64.row.col.satfinite.s32.u4.u4.s32`) with
/// the same S32 accumulator and `float` α/β family as [`S4`].
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct U4(pub u8);

// SAFETY: S4 / U4 are #[repr(transparent)] around u8, which is DeviceRepr.
// Same ABI, same Copy + 'static bounds.
unsafe impl DeviceRepr for S4 {}
unsafe impl DeviceRepr for U4 {}

impl S4 {
    /// Pack two s4 values `[lo, hi]` (each in `[-8, +7]`) into one
    /// storage slot. Values outside the range are masked to their low 4
    /// bits (no saturation — pre-clamp on the caller side if needed).
    #[inline]
    pub fn pack(lo: i8, hi: i8) -> Self {
        Self(((lo as u8) & 0x0F) | (((hi as u8) & 0x0F) << 4))
    }

    /// Unpack into `[low_nibble_as_s4, high_nibble_as_s4]`. Each
    /// returned value is sign-extended from the 4-bit nibble.
    #[inline]
    pub fn unpack(self) -> [i8; 2] {
        let lo = ((self.0 & 0x0F) << 4) as i8 >> 4;
        let hi = (self.0 & 0xF0) as i8 >> 4;
        [lo, hi]
    }
}

impl U4 {
    /// Pack two u4 values `[lo, hi]` (each in `[0, 15]`) into one
    /// storage slot. Values outside the range are masked to their low 4
    /// bits.
    #[inline]
    pub fn pack(lo: u8, hi: u8) -> Self {
        Self((lo & 0x0F) | ((hi & 0x0F) << 4))
    }

    /// Unpack into `[low_nibble, high_nibble]`. Each returned value is
    /// in `[0, 15]`.
    #[inline]
    pub fn unpack(self) -> [u8; 2] {
        [self.0 & 0x0F, (self.0 >> 4) & 0x0F]
    }
}

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
impl int_sealed::Sealed for S4 {}
impl int_sealed::Sealed for U4 {}

impl IntElement for S8 {
    const KIND: ElementKind = ElementKind::S8;
}

impl IntElement for U8 {
    const KIND: ElementKind = ElementKind::U8;
}

impl IntElement for S4 {
    const KIND: ElementKind = ElementKind::S4;
}

impl IntElement for U4 {
    const KIND: ElementKind = ElementKind::U4;
}

// ============================================================================
// 8-bit floating-point element family — sibling to Element / IntElement
// ============================================================================

/// 8-bit floating-point, E4M3 encoding (1 sign + 4 exponent + 3 mantissa,
/// exponent bias 7).
///
/// `#[repr(transparent)]` around `u8` storage — bit-compatible with
/// `__nv_fp8_storage_t` on the CUDA side and with `float8::F8E4M3` on the
/// host side. A `DeviceBuffer<u8>` (byte substrate) can be reinterpreted
/// as `DeviceBuffer<Fp8E4M3>` via `view_as` without copying.
///
/// Numerical range: ±448 (max finite). One NaN encoding only
/// (`S.1111.111`); E4M3 has **no infinities**. The conversion path
/// matches NVIDIA's `__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3)`:
/// round-half-to-even, saturating-to-max-finite on overflow.
///
/// Routes through Ada Lovelace FP8 tensor cores
/// (`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`) with F32
/// accumulation and float alpha / beta scaling. First landed in
/// baracuda-kernels Phase 2.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Fp8E4M3(pub u8);

impl Fp8E4M3 {
    /// Convert from `f32` using NVIDIA's `SATFINITE` semantics
    /// (round-half-to-even, clamp `|x|` to the E4M3 max-finite `448.0`).
    #[inline]
    pub fn from_f32(x: f32) -> Self {
        Self(float8::F8E4M3::from_f32(x).to_bits())
    }

    /// Convert to `f32`. The E4M3 grid is sparse — the result is one of
    /// 254 finite values (or NaN) on the E4M3 lattice.
    #[inline]
    pub fn to_f32(self) -> f32 {
        float8::F8E4M3::from_bits(self.0).to_f32()
    }
}

// SAFETY: Fp8E4M3 is #[repr(transparent)] over u8, which is DeviceRepr.
// Same ABI, same Copy + 'static bounds.
unsafe impl DeviceRepr for Fp8E4M3 {}

/// 8-bit floating-point, E5M2 encoding (1 sign + 5 exponent + 2 mantissa,
/// exponent bias 15).
///
/// `#[repr(transparent)]` around `u8` storage — bit-compatible with
/// `__nv_fp8_storage_t` on the CUDA side and with `float8::F8E5M2` on the
/// host side. A `DeviceBuffer<u8>` (byte substrate) can be reinterpreted
/// as `DeviceBuffer<Fp8E5M2>` via `view_as` without copying.
///
/// Numerical range: ±57344 (max finite). IEEE-style infinity and NaN
/// encodings (unlike [`Fp8E4M3`], which has neither). The conversion
/// path matches NVIDIA's
/// `__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E5M2)`:
/// round-half-to-even, saturating-to-max-finite on overflow.
///
/// Routes through Ada Lovelace FP8 tensor cores
/// (`mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32`) with F32
/// accumulation and float alpha / beta scaling.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Fp8E5M2(pub u8);

impl Fp8E5M2 {
    /// Convert from `f32` using NVIDIA's `SATFINITE` semantics
    /// (round-half-to-even, clamp `|x|` to the E5M2 max-finite `57344.0`).
    #[inline]
    pub fn from_f32(x: f32) -> Self {
        Self(float8::F8E5M2::from_f32(x).to_bits())
    }

    /// Convert to `f32`. The E5M2 grid is sparse — the result is one of
    /// the finite values (or inf / NaN) on the E5M2 lattice.
    #[inline]
    pub fn to_f32(self) -> f32 {
        float8::F8E5M2::from_bits(self.0).to_f32()
    }
}

// SAFETY: Fp8E5M2 is #[repr(transparent)] over u8, which is DeviceRepr.
// Same ABI, same Copy + 'static bounds.
unsafe impl DeviceRepr for Fp8E5M2 {}

/// 8-bit floating-point element types supported by the kernel facade.
///
/// Sibling trait to [`Element`] (which covers f16 / bf16 / f32 /
/// [`F32Strict`] / f64) and to [`IntElement`] (which covers S8 / U8) —
/// kept separate because the FP8 kernel family has its own MMA
/// instruction set (`mma.sync ... .f32.e4m3.e4m3.f32`), arch requirement
/// (sm_89+), and conversion semantics (saturating-to-max-finite vs the
/// int family's saturating-to-INT_MAX).
///
/// Sealed because adding a new FP8 variant requires shipping new kernel
/// instantiations in `baracuda-kernels-sys`.
pub trait FpElement: DeviceRepr + fp_sealed::Sealed + Copy + 'static {
    /// Runtime tag for this element type. Always one of the FP8 variants
    /// of [`ElementKind`].
    const KIND: ElementKind;
}

impl fp_sealed::Sealed for Fp8E4M3 {}
impl fp_sealed::Sealed for Fp8E5M2 {}

impl FpElement for Fp8E4M3 {
    const KIND: ElementKind = ElementKind::Fp8E4M3;
}

impl FpElement for Fp8E5M2 {
    const KIND: ElementKind = ElementKind::Fp8E5M2;
}

// ============================================================================
// Binary element family — sibling to Element / IntElement / FpElement
// ============================================================================

/// 1-bit binary element marker — **packed-byte storage**.
///
/// `#[repr(transparent)]` around `u8`. One [`Bin`] *storage slot* is one
/// byte and holds **eight** packed b1 elements: bit `i` of the byte
/// (LSB = bit 0) is the element at K offset `8 * byte_idx + i`. Packing
/// is along the K axis for A/B operands.
///
/// A `DeviceBuffer<u8>` of `(M*K)/8` bytes can be reinterpreted as a
/// `DeviceBuffer<Bin>` of `(M*K)/8` storage slots via `view_as` without
/// copying.
///
/// Routes through Ampere+ binary tensor cores
/// (`mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc`) with
/// an **S32 output accumulator**. Unlike the int4 / int8 / FP8
/// families, bin GEMM does **not** quantize its output back to the
/// input element type — the result is the raw popcount accumulator
/// (`popcount(xor(A_row, B_col))` summed over K bytes), surfaced as
/// `i32`. No α / β / bias / activation chain (the popcount programming
/// model doesn't have a meaningful place for them).
///
/// The plan layer ([`Bin` is consumed by `BinGemmPlan` in
/// `baracuda-kernels`) takes `M`, `N`, `K` in **element** counts and
/// leading dimensions in **storage-slot (= byte)** counts —
/// `MatrixRef<Bin>::ld` therefore equals `K / 8` for row-major A with
/// no padding. `K` must be divisible by 8 (packing is byte-aligned).
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Bin(pub u8);

// SAFETY: Bin is #[repr(transparent)] around u8, which is DeviceRepr.
unsafe impl DeviceRepr for Bin {}

impl Bin {
    /// Pack eight `bool` values `bits[0..8]` into one storage byte.
    /// `bits[i]` becomes bit `i` of the result (LSB-first).
    #[inline]
    pub fn pack(bits: [bool; 8]) -> Self {
        let mut b = 0u8;
        let mut i = 0;
        while i < 8 {
            if bits[i] {
                b |= 1 << i;
            }
            i += 1;
        }
        Self(b)
    }

    /// Unpack one storage byte into eight `bool` values along K (LSB-first).
    #[inline]
    pub fn unpack(self) -> [bool; 8] {
        let b = self.0;
        [
            (b >> 0) & 1 != 0,
            (b >> 1) & 1 != 0,
            (b >> 2) & 1 != 0,
            (b >> 3) & 1 != 0,
            (b >> 4) & 1 != 0,
            (b >> 5) & 1 != 0,
            (b >> 6) & 1 != 0,
            (b >> 7) & 1 != 0,
        ]
    }
}

/// Binary (1-bit) element types supported by the kernel facade.
///
/// Sibling trait to [`Element`] / [`IntElement`] / [`FpElement`] —
/// kept separate because the bin kernel family has a distinct
/// programming model (popcount-based, `D = popcount(xor(A, B))`, no
/// α/β/bias/activation chain) and a non-matching output type (raw
/// `i32` accumulator rather than re-quantized to the input type).
pub trait BinElement: DeviceRepr + bin_sealed::Sealed + Copy + 'static {
    /// Runtime tag for this element type. Always [`ElementKind::Bin`]
    /// today (the bin family currently has a single member).
    const KIND: ElementKind;
}

impl bin_sealed::Sealed for Bin {}

impl BinElement for Bin {
    const KIND: ElementKind = ElementKind::Bin;
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
    /// Signed 32-bit integer. Maps to the `i32` Rust type via the
    /// [`Element`] impl. Two roles:
    /// 1. **Accumulator marker** for integer GEMM SKUs (reported by
    ///    [`crate::PrecisionGuarantee::accumulator`]).
    /// 2. **Input element** for elementwise integer arithmetic
    ///    (bitwise / comparison / scan ops). The same plan shapes used
    ///    for floating-point inputs extend to `i32` via the [`Element`]
    ///    impl.
    I32,
    /// Signed 64-bit integer. Maps to the `i64` Rust type via the
    /// [`Element`] impl. Used as an input element for the elementwise
    /// integer arithmetic family (bitwise / comparison / scan ops).
    /// PyTorch's default integer tensor dtype.
    I64,
    /// Boolean (1-byte storage). Maps to the [`Bool`] wrapper type via
    /// the [`Element`] impl. Used as the input element for the logical-
    /// op family (`logical_and` / `logical_or` / `logical_xor`) and as
    /// the output element for the comparison-op family
    /// (`eq` / `ne` / `gt` / `ge` / `lt` / `le`). Truthiness convention
    /// follows PyTorch: 0 = false, any non-zero byte = true.
    Bool,
    /// 8-bit floating-point, E4M3 encoding (1 sign + 4 exponent + 3
    /// mantissa, bias 7, max-finite 448, no infinities). Maps to the
    /// [`Fp8E4M3`] wrapper type. Routed through Ada / Hopper FP8 tensor
    /// cores (`mma.sync m16n8k32` FP8 variant) with F32 accumulation.
    Fp8E4M3,
    /// 8-bit floating-point, E5M2 encoding (1 sign + 5 exponent + 2
    /// mantissa, bias 15, IEEE-754-compatible inf / NaN). Maps to the
    /// [`Fp8E5M2`] wrapper type. Same FP8 tensor-core path as
    /// [`Fp8E4M3`] with the alternate operand tag
    /// (`.e5m2.e5m2.f32`).
    Fp8E5M2,
    /// Signed 4-bit integer — packed-pair storage. Maps to the [`S4`]
    /// wrapper type. Routed through Ada Lovelace int4 tensor cores
    /// (`mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.s4.s32`)
    /// with int32 accumulation; float `alpha` / `beta` let the kernel
    /// act as a dequantize-in-epilogue (same convention as the int8
    /// family).
    S4,
    /// Unsigned 4-bit integer — packed-pair storage. Maps to the [`U4`]
    /// wrapper type. Same kernel family as [`S4`] with the alternate
    /// operand tag (`.u4.u4.s32`).
    U4,
    /// 1-bit binary — packed-byte storage (8 bits per byte, LSB =
    /// lowest K index). Maps to the [`Bin`] wrapper type. Routed
    /// through Ampere+ binary tensor cores
    /// (`mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc`).
    /// Distinct programming model: the output is the raw popcount
    /// accumulator (s32), not a re-quantized b1.
    Bin,
    /// Single-precision complex — interleaved real/imag pair of `f32`
    /// (`#[repr(C)]`). Maps to the [`Complex32`] wrapper type. Used by
    /// the FFT op family (Milestone 6.4) for spectrum-domain tensors.
    /// ABI-compatible with cuFFT's `cufftComplex`, NumPy's `complex64`,
    /// and PyTorch's `torch.complex64`.
    Complex32,
    /// Double-precision complex — interleaved real/imag pair of `f64`.
    /// Maps to the [`Complex64`] wrapper type. ABI-compatible with
    /// cuFFT's `cufftDoubleComplex`, NumPy's `complex128`, and
    /// PyTorch's `torch.complex128`.
    Complex64,
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
    /// FP8 E4M3 multiply-add (`mma.sync m16n8k32` FP8 variant) with F32
    /// accumulation. Inputs are E4M3 (8-bit), the accumulator is F32,
    /// and the epilogue cast saturates to the E4M3 max-finite (±448).
    Fp8E4M3,
    /// FP8 E5M2 multiply-add. Same instruction family as
    /// [`Fp8E4M3`](Self::Fp8E4M3) but with the E5M2 encoding (wider
    /// exponent, narrower mantissa).
    Fp8E5M2,
    /// 4-bit integer multiply-add (`mma.sync m16n8k64` int4 variant)
    /// with int32 accumulation. Used by both signed (s4) and unsigned
    /// (u4) integer GEMM SKUs; the multiply operands are 4-bit
    /// (packed-pair storage in memory), the accumulator is 32-bit, and
    /// the multiply-add uses the `satfinite` operator (clamps the
    /// accumulator on overflow rather than wrapping). sm_89+.
    Int4,
    /// 1-bit binary `xor.popc` multiply-add
    /// (`mma.sync m16n8k256` b1 variant) with int32 accumulation. The
    /// "multiply" is per-bit XOR and the "add" is popcount. Used by
    /// the binary GEMM SKU; operands are 1-bit (packed 8-per-byte in
    /// memory), the accumulator is 32-bit, and the output is the
    /// **raw** popcount accumulator — no re-quantization back to b1.
    /// sm_80+.
    Binary,
}
