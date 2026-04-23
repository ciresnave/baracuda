//! Third-party type integrations.
//!
//! When a user enables a `*-crate` feature, baracuda automatically teaches
//! the external crate's numeric types how to speak baracuda's trait
//! vocabulary. The impls here are intentionally one-way: we implement
//! baracuda traits *for* the external types, never the other way around.
//!
//! # `half-crate`
//!
//! Implements [`DeviceRepr`] and [`ValidAsZeroBits`] for `half::f16` and
//! `half::bf16`. Both types are `#[repr(transparent)] over u16` in the
//! `half` crate, so zero bytes represent `0.0`.
//!
//! # `f8-crate`
//!
//! Implements [`DeviceRepr`] and [`ValidAsZeroBits`] for `float8::F8E4M3`
//! and `float8::F8E5M2`. The `float8` crate currently ships these two
//! variants only; Fuel's wider F4/F6/F8E8M0 coverage will require either
//! a richer upstream crate or baracuda growing its own newtypes in
//! [`crate::numeric`].
//!
//! # KernelArg
//!
//! The [`crate::KernelArg`] blanket impl covers `&T` / `&mut T` for any
//! `T: DeviceRepr`, so adding `DeviceRepr` here is enough to make these
//! types usable as kernel arguments. Example:
//!
//! ```ignore
//! # #[cfg(feature = "half-crate")]
//! # fn demo() {
//! use baracuda_types::KernelArg;
//! let h: half::f16 = half::f16::from_f32(1.0);
//! let _arg: *mut core::ffi::c_void = (&h).as_kernel_arg_ptr();
//! # }
//! ```

#[cfg(feature = "half-crate")]
mod half_impls {
    use crate::{DeviceRepr, ValidAsZeroBits};

    // SAFETY: half::f16 is #[repr(transparent)] over u16. All-zero bits
    // are the valid IEEE 754 half-precision representation of +0.0.
    unsafe impl DeviceRepr for half::f16 {}
    unsafe impl ValidAsZeroBits for half::f16 {}

    // SAFETY: half::bf16 is #[repr(transparent)] over u16. All-zero bits
    // are the valid brain-float-16 representation of +0.0.
    unsafe impl DeviceRepr for half::bf16 {}
    unsafe impl ValidAsZeroBits for half::bf16 {}
}

#[cfg(feature = "f8-crate")]
mod f8_impls {
    use crate::{DeviceRepr, ValidAsZeroBits};

    // SAFETY: float8::F8E4M3 is #[repr(transparent)] over u8. All-zero
    // bits are a valid F8E4M3 value (positive zero).
    unsafe impl DeviceRepr for float8::F8E4M3 {}
    unsafe impl ValidAsZeroBits for float8::F8E4M3 {}

    // SAFETY: float8::F8E5M2 is #[repr(transparent)] over u8. All-zero
    // bits are a valid F8E5M2 value (positive zero).
    unsafe impl DeviceRepr for float8::F8E5M2 {}
    unsafe impl ValidAsZeroBits for float8::F8E5M2 {}
}

#[cfg(all(test, feature = "half-crate"))]
mod half_tests {
    use crate::{DeviceRepr, ValidAsZeroBits};

    fn assert_device_repr<T: DeviceRepr>() {}
    fn assert_valid_as_zero<T: ValidAsZeroBits>() {}

    #[test]
    fn half_types_implement_the_trio() {
        assert_device_repr::<half::f16>();
        assert_device_repr::<half::bf16>();
        assert_valid_as_zero::<half::f16>();
        assert_valid_as_zero::<half::bf16>();
    }

    #[test]
    fn half_zero_bits_round_trip() {
        // ValidAsZeroBits means all-zero bytes decode to a valid T.
        let h: half::f16 = unsafe { core::mem::zeroed() };
        assert_eq!(h.to_f32(), 0.0);
        let b: half::bf16 = unsafe { core::mem::zeroed() };
        assert_eq!(b.to_f32(), 0.0);
    }
}

#[cfg(all(test, feature = "f8-crate"))]
mod f8_tests {
    use crate::{DeviceRepr, ValidAsZeroBits};

    fn assert_device_repr<T: DeviceRepr>() {}
    fn assert_valid_as_zero<T: ValidAsZeroBits>() {}

    #[test]
    fn f8_types_implement_the_trio() {
        assert_device_repr::<float8::F8E4M3>();
        assert_device_repr::<float8::F8E5M2>();
        assert_valid_as_zero::<float8::F8E4M3>();
        assert_valid_as_zero::<float8::F8E5M2>();
    }

    #[test]
    fn f8_zero_bits_round_trip() {
        let a: float8::F8E4M3 = unsafe { core::mem::zeroed() };
        assert_eq!(a.to_f32(), 0.0);
        let b: float8::F8E5M2 = unsafe { core::mem::zeroed() };
        assert_eq!(b.to_f32(), 0.0);
    }
}
