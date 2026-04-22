//! Numeric types used across the CUDA stack.
//!
//! These are thin, `#[repr(transparent)]` / `#[repr(C)]` wrappers chosen to
//! match the layout NVIDIA's headers use for `__half`, `__nv_bfloat16`,
//! `cuFloatComplex`, and `cuDoubleComplex`. All conversion methods return
//! the same bit patterns the CUDA runtime itself would produce for typical
//! inputs; exact agreement with NVIDIA's rounding on edge cases is tested
//! in the integration suite against `half` and CUDA itself.
//!
//! If you already depend on `half` / `num-complex`, enable the `half-crate`
//! / `num-complex-crate` features for zero-cost `From`/`Into` adapters.

use core::cmp::Ordering;
use core::fmt;

/// IEEE 754 binary16 ("half-precision", `__half` in CUDA).
///
/// Layout: 1 sign bit, 5 exponent bits (bias 15), 10 mantissa bits.
#[derive(Copy, Clone, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Half(pub u16);

impl Half {
    pub const ZERO: Self = Self(0x0000);
    pub const NEG_ZERO: Self = Self(0x8000);
    pub const ONE: Self = Self(0x3C00);
    pub const NEG_ONE: Self = Self(0xBC00);
    pub const INFINITY: Self = Self(0x7C00);
    pub const NEG_INFINITY: Self = Self(0xFC00);
    pub const NAN: Self = Self(0x7E00);
    pub const MIN_POSITIVE: Self = Self(0x0400); // smallest normal
    pub const MAX: Self = Self(0x7BFF);
    pub const MIN: Self = Self(0xFBFF);
    pub const EPSILON: Self = Self(0x1400); // 2^-10

    #[inline]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    #[inline]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    #[inline]
    pub const fn is_nan(self) -> bool {
        (self.0 & 0x7FFF) > 0x7C00
    }

    #[inline]
    pub const fn is_infinite(self) -> bool {
        (self.0 & 0x7FFF) == 0x7C00
    }

    #[inline]
    pub const fn is_finite(self) -> bool {
        (self.0 & 0x7C00) != 0x7C00
    }

    #[inline]
    pub const fn is_sign_negative(self) -> bool {
        (self.0 & 0x8000) != 0
    }

    /// Round-to-nearest-even conversion from `f32`.
    pub fn from_f32(f: f32) -> Self {
        let bits = f.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp_raw = ((bits >> 23) & 0xFF) as i32;
        let mant = bits & 0x007F_FFFF;

        // NaN or Inf
        if exp_raw == 0xFF {
            if mant != 0 {
                // NaN: preserve signaling bit position as "quiet" and carry a
                // subset of the payload.
                return Self(sign | 0x7E00 | ((mant >> 13) as u16));
            }
            return Self(sign | 0x7C00);
        }

        let e_unbiased = exp_raw - 127; // true exponent
        let e_half = e_unbiased + 15;

        if e_half >= 0x1F {
            // Overflow -> inf (any value with true exp >= 16).
            return Self(sign | 0x7C00);
        }

        if e_half >= 1 {
            // Normal result.
            let trunc = (mant >> 13) as u16;
            let guard = (mant >> 12) & 1;
            let sticky = mant & 0x0FFF;
            let lsb = trunc & 1;
            let round_up = guard == 1 && (sticky != 0 || lsb == 1);
            let base = sign | ((e_half as u16) << 10) | trunc;
            let half = base.wrapping_add(round_up as u16);
            return Self(half);
        }

        // Subnormal or underflow.
        if e_unbiased < -24 {
            // Fully underflows to signed zero.
            return Self(sign);
        }

        // Subnormal range: true exponent in [-24, -15].
        // The implicit leading 1 comes back into play.
        let mant_full = mant | 0x0080_0000; // 24-bit mantissa with leading 1
        let shift = (-14 - e_unbiased) as u32 + 13; // right-shift amount to get 10-bit result
        let trunc = (mant_full >> shift) as u16;
        let guard = (mant_full >> (shift - 1)) & 1;
        let sticky_mask = (1u32 << (shift - 1)) - 1;
        let sticky = mant_full & sticky_mask;
        let lsb = trunc & 1;
        let round_up = guard == 1 && (sticky != 0 || lsb == 1);
        let half = sign | trunc.wrapping_add(round_up as u16);
        Self(half)
    }

    /// Exact conversion to `f32` (every finite `Half` is representable as `f32`).
    pub fn to_f32(self) -> f32 {
        let h = self.0 as u32;
        let sign = (h & 0x8000) << 16;
        let exp = (h >> 10) & 0x1F;
        let mant = h & 0x03FF;

        let bits = if exp == 0 {
            if mant == 0 {
                sign
            } else {
                // Subnormal: normalize.
                let mut m = mant;
                let mut e: i32 = 1;
                while (m & 0x0400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x03FF;
                let exp_f32 = (e + 127 - 15) as u32;
                sign | (exp_f32 << 23) | (m << 13)
            }
        } else if exp == 0x1F {
            sign | 0x7F80_0000 | (mant << 13)
        } else {
            let exp_f32 = exp + 127 - 15;
            sign | (exp_f32 << 23) | (mant << 13)
        };

        f32::from_bits(bits)
    }
}

impl fmt::Debug for Half {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Half({})", self.to_f32())
    }
}

impl fmt::Display for Half {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.to_f32(), f)
    }
}

impl PartialOrd for Half {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.to_f32().partial_cmp(&other.to_f32())
    }
}

impl From<Half> for f32 {
    #[inline]
    fn from(h: Half) -> f32 {
        h.to_f32()
    }
}

impl From<Half> for f64 {
    #[inline]
    fn from(h: Half) -> f64 {
        h.to_f32() as f64
    }
}

impl From<f32> for Half {
    #[inline]
    fn from(f: f32) -> Self {
        Self::from_f32(f)
    }
}

/// Brain Floating Point 16 (`__nv_bfloat16` in CUDA). The top 16 bits of an IEEE 754 `f32`.
#[derive(Copy, Clone, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct BFloat16(pub u16);

impl BFloat16 {
    pub const ZERO: Self = Self(0x0000);
    pub const NEG_ZERO: Self = Self(0x8000);
    pub const ONE: Self = Self(0x3F80);
    pub const NEG_ONE: Self = Self(0xBF80);
    pub const INFINITY: Self = Self(0x7F80);
    pub const NEG_INFINITY: Self = Self(0xFF80);
    pub const NAN: Self = Self(0x7FC0);
    pub const MIN_POSITIVE: Self = Self(0x0080);
    pub const MAX: Self = Self(0x7F7F);
    pub const MIN: Self = Self(0xFF7F);
    pub const EPSILON: Self = Self(0x3C00);

    #[inline]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    #[inline]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    #[inline]
    pub const fn is_nan(self) -> bool {
        (self.0 & 0x7FFF) > 0x7F80
    }

    #[inline]
    pub const fn is_infinite(self) -> bool {
        (self.0 & 0x7FFF) == 0x7F80
    }

    #[inline]
    pub const fn is_sign_negative(self) -> bool {
        (self.0 & 0x8000) != 0
    }

    /// Round-to-nearest-even conversion from `f32` (matches NVIDIA's bfloat16 truncation + rounding).
    pub fn from_f32(f: f32) -> Self {
        if f.is_nan() {
            return Self(0x7FC0);
        }
        let bits = f.to_bits();
        let lsb = (bits >> 16) & 1;
        // Round-half-to-even: add 0x7FFF + lsb to upper half, then truncate.
        let rounding_bias = 0x7FFF + lsb;
        let rounded = bits.wrapping_add(rounding_bias);
        Self((rounded >> 16) as u16)
    }

    #[inline]
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }
}

impl fmt::Debug for BFloat16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BFloat16({})", self.to_f32())
    }
}

impl fmt::Display for BFloat16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.to_f32(), f)
    }
}

impl PartialOrd for BFloat16 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.to_f32().partial_cmp(&other.to_f32())
    }
}

impl From<BFloat16> for f32 {
    #[inline]
    fn from(b: BFloat16) -> f32 {
        b.to_f32()
    }
}

impl From<f32> for BFloat16 {
    #[inline]
    fn from(f: f32) -> Self {
        Self::from_f32(f)
    }
}

/// Single-precision complex number (`cuFloatComplex`, layout-compatible with `float2`).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Complex32 {
    pub re: f32,
    pub im: f32,
}

impl Complex32 {
    /// The complex number `0 + 0i`.
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
    /// The complex number `1 + 0i`.
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };
    /// The imaginary unit `0 + 1i`.
    pub const I: Self = Self { re: 0.0, im: 1.0 };

    /// Construct a complex number from its real and imaginary parts.
    #[inline]
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    /// Squared magnitude: `re² + im²`.
    #[inline]
    pub fn norm_sqr(self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    /// Complex conjugate: `re - i·im`.
    #[inline]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
}

/// Double-precision complex number (`cuDoubleComplex`, layout-compatible with `double2`).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    /// The complex number `0 + 0i`.
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
    /// The complex number `1 + 0i`.
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };
    /// The imaginary unit `0 + 1i`.
    pub const I: Self = Self { re: 0.0, im: 1.0 };

    /// Construct a complex number from its real and imaginary parts.
    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline]
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    #[inline]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
}

#[cfg(feature = "half-crate")]
mod half_adapters {
    use super::{BFloat16, Half};

    impl From<half::f16> for Half {
        #[inline]
        fn from(v: half::f16) -> Self {
            Self(v.to_bits())
        }
    }

    impl From<Half> for half::f16 {
        #[inline]
        fn from(v: Half) -> Self {
            half::f16::from_bits(v.0)
        }
    }

    impl From<half::bf16> for BFloat16 {
        #[inline]
        fn from(v: half::bf16) -> Self {
            Self(v.to_bits())
        }
    }

    impl From<BFloat16> for half::bf16 {
        #[inline]
        fn from(v: BFloat16) -> Self {
            half::bf16::from_bits(v.0)
        }
    }
}

#[cfg(feature = "num-complex-crate")]
mod num_complex_adapters {
    use super::{Complex32, Complex64};

    impl From<num_complex::Complex<f32>> for Complex32 {
        #[inline]
        fn from(v: num_complex::Complex<f32>) -> Self {
            Self { re: v.re, im: v.im }
        }
    }

    impl From<Complex32> for num_complex::Complex<f32> {
        #[inline]
        fn from(v: Complex32) -> Self {
            Self::new(v.re, v.im)
        }
    }

    impl From<num_complex::Complex<f64>> for Complex64 {
        #[inline]
        fn from(v: num_complex::Complex<f64>) -> Self {
            Self { re: v.re, im: v.im }
        }
    }

    impl From<Complex64> for num_complex::Complex<f64> {
        #[inline]
        fn from(v: Complex64) -> Self {
            Self::new(v.re, v.im)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_constants_roundtrip() {
        assert_eq!(Half::ZERO.to_f32(), 0.0);
        assert_eq!(Half::ONE.to_f32(), 1.0);
        assert_eq!(Half::NEG_ONE.to_f32(), -1.0);
        assert!(Half::INFINITY.to_f32().is_infinite());
        assert!(Half::NEG_INFINITY.to_f32().is_infinite());
        assert!(Half::NAN.to_f32().is_nan());
    }

    #[test]
    fn half_roundtrip_exact_values() {
        for v in [0.0f32, 1.0, -1.0, 0.5, -0.5, 2.0, 65504.0, -65504.0, 1e-4] {
            let h = Half::from_f32(v);
            let back = h.to_f32();
            assert!(
                (back - v).abs() < (v.abs() * 1e-3 + 1e-7),
                "{v} -> {back} (half bits = {:#06x})",
                h.to_bits()
            );
        }
    }

    #[test]
    fn half_overflow_to_infinity() {
        assert_eq!(Half::from_f32(1e30).to_bits(), Half::INFINITY.to_bits());
        assert_eq!(
            Half::from_f32(-1e30).to_bits(),
            Half::NEG_INFINITY.to_bits()
        );
    }

    #[test]
    fn half_underflow_to_zero() {
        assert_eq!(Half::from_f32(1e-30).to_bits(), 0);
        assert_eq!(Half::from_f32(-1e-30).to_bits(), 0x8000);
    }

    #[test]
    fn half_nan_stays_nan() {
        assert!(Half::from_f32(f32::NAN).is_nan());
    }

    #[test]
    fn bfloat_constants_roundtrip() {
        assert_eq!(BFloat16::ZERO.to_f32(), 0.0);
        assert_eq!(BFloat16::ONE.to_f32(), 1.0);
        assert_eq!(BFloat16::NEG_ONE.to_f32(), -1.0);
        assert!(BFloat16::INFINITY.to_f32().is_infinite());
        assert!(BFloat16::NAN.to_f32().is_nan());
    }

    #[test]
    fn bfloat_truncates_top_16_bits() {
        // A value whose low 16 f32 bits are zero round-trips exactly.
        let v: f32 = 1.5; // 0x3FC0_0000
        let b = BFloat16::from_f32(v);
        assert_eq!(b.to_bits(), 0x3FC0);
        assert_eq!(b.to_f32(), 1.5);
    }

    #[test]
    fn bfloat_nan_stays_nan() {
        assert!(BFloat16::from_f32(f32::NAN).is_nan());
    }

    #[test]
    fn complex_layout_is_two_floats() {
        use core::mem::{align_of, size_of};
        assert_eq!(size_of::<Complex32>(), 8);
        assert_eq!(size_of::<Complex64>(), 16);
        assert!(align_of::<Complex32>() >= align_of::<f32>());
        assert!(align_of::<Complex64>() >= align_of::<f64>());
    }
}
