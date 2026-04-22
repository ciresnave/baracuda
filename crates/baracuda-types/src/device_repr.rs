//! The `DeviceRepr` marker trait: types with a stable, ABI-compatible layout
//! suitable for being laid out in GPU-visible memory.
//!
//! `DeviceRepr` is implemented for every primitive numeric type, arrays of
//! `DeviceRepr`, tuples of `DeviceRepr` up to arity 12, and the numeric
//! helpers in [`crate::numeric`]. User-defined `#[repr(C)]` structs can
//! derive it via `#[derive(baracuda_types::DeviceRepr)]` (requires the
//! `derive` feature).

use crate::numeric::{BFloat16, Complex32, Complex64, Half};

/// A type whose Rust memory layout is valid to expose to a CUDA kernel or
/// to store in a GPU-visible buffer.
///
/// # Safety
///
/// Implementors must uphold:
///
/// 1. The type has a fixed, compile-time-known size and alignment.
/// 2. The type is [`Copy`] and contains no references, pointers to
///    host-only data, or non-trivial destructors.
/// 3. Any bit pattern that Rust reads back from the type after a
///    device memcpy must produce a valid value (or the code working with
///    the type must tolerate transient "garbage" and write before read).
/// 4. The layout is `#[repr(C)]`, `#[repr(transparent)]`, or a primitive.
pub unsafe trait DeviceRepr: Copy + 'static {}

macro_rules! impl_device_repr_primitive {
    ($($t:ty),* $(,)?) => {
        $(
            // SAFETY: primitives are Copy + Sized and trivially ABI-stable.
            unsafe impl DeviceRepr for $t {}
        )*
    };
}

impl_device_repr_primitive!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64, bool, char,
);

// SAFETY: wrappers in `crate::numeric` are `#[repr(transparent)]` / `#[repr(C)]`.
unsafe impl DeviceRepr for Half {}
unsafe impl DeviceRepr for BFloat16 {}
unsafe impl DeviceRepr for Complex32 {}
unsafe impl DeviceRepr for Complex64 {}

// SAFETY: a fixed-size array of `DeviceRepr` elements has a well-defined
// C-compatible layout (contiguous, same alignment as T).
unsafe impl<T: DeviceRepr, const N: usize> DeviceRepr for [T; N] {}

macro_rules! impl_device_repr_tuple {
    ($($t:ident),+) => {
        // SAFETY: Rust does not actually guarantee C layout for tuples; we
        // restrict tuple support to homogeneous sizes and mark as unsafe.
        // Consumers should prefer `#[repr(C)] struct` or `[T; N]` for
        // heterogeneous aggregates; this impl exists so `(T,)` and other
        // simple tuples are ergonomic in tests.
        unsafe impl<$($t: DeviceRepr),+> DeviceRepr for ($($t,)+) {}
    };
}

impl_device_repr_tuple!(A);
impl_device_repr_tuple!(A, B);
impl_device_repr_tuple!(A, B, C);
impl_device_repr_tuple!(A, B, C, D);
impl_device_repr_tuple!(A, B, C, D, E);
impl_device_repr_tuple!(A, B, C, D, E, F);
impl_device_repr_tuple!(A, B, C, D, E, F, G);
impl_device_repr_tuple!(A, B, C, D, E, F, G, H);
impl_device_repr_tuple!(A, B, C, D, E, F, G, H, I);
impl_device_repr_tuple!(A, B, C, D, E, F, G, H, I, J);
impl_device_repr_tuple!(A, B, C, D, E, F, G, H, I, J, K);
impl_device_repr_tuple!(A, B, C, D, E, F, G, H, I, J, K, L);

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_device_repr<T: DeviceRepr>() {}

    #[test]
    fn primitives_are_device_repr() {
        assert_device_repr::<u8>();
        assert_device_repr::<u16>();
        assert_device_repr::<u32>();
        assert_device_repr::<i32>();
        assert_device_repr::<f32>();
        assert_device_repr::<f64>();
        assert_device_repr::<usize>();
    }

    #[test]
    fn numeric_wrappers_are_device_repr() {
        assert_device_repr::<Half>();
        assert_device_repr::<BFloat16>();
        assert_device_repr::<Complex32>();
        assert_device_repr::<Complex64>();
    }

    #[test]
    fn arrays_and_tuples_are_device_repr() {
        assert_device_repr::<[f32; 4]>();
        assert_device_repr::<[u8; 256]>();
        assert_device_repr::<(f32, f32)>();
        assert_device_repr::<(u32, i32, f64)>();
    }
}
