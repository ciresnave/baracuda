//! [`ValidAsZeroBits`] — safety marker for zero-initializable types.

use crate::{BFloat16, Complex32, Complex64, Half};

/// Marker trait for types whose all-zero bit pattern is a valid value.
///
/// Crates that allocate GPU memory often use `cudaMemset(0)` / `cuMemsetD8`
/// to initialize buffers to zero and then treat the allocation as a
/// populated `[T]`. That's only sound for types where all-zero bytes are a
/// valid `T` — e.g. integer primitives and IEEE-754 floating-point types
/// (where `0x00000000` is the representation of `+0.0`).
///
/// Implementing `ValidAsZeroBits` is a promise that:
///
/// 1. Reading a zero-initialized `T` is not undefined behavior.
/// 2. The zero-initialized value is semantically sensible (`0` / `0.0` /
///    `(0, 0, ..., 0)` — *not* a niche-optimized enum where zero means
///    something else).
///
/// # Safety
///
/// This trait is `unsafe` to implement: a wrong impl will lead to UB the
/// first time downstream code zero-initializes a buffer and reads it back.
/// Common counter-examples include:
///
/// - `&T`, `&mut T` (null references are UB).
/// - `NonZero*` from `std::num`.
/// - Enums with a non-zero discriminant for their zero-value variant.
/// - `Box<T>`, `Arc<T>`, `String`, `Vec<T>` (heap-owned data).
///
/// # Supplied impls
///
/// - Every unsigned integer (`u8`..`u128`, `usize`).
/// - Every signed integer (`i8`..`i128`, `isize`).
/// - `f32`, `f64` (zero = `+0.0`).
/// - [`Half`], [`BFloat16`], [`Complex32`], [`Complex64`].
/// - `()` — the unit type.
/// - `bool` — zero = `false`.
/// - Tuples of arity 1–12 where every element is `ValidAsZeroBits`.
/// - Fixed-size arrays `[T; N]` where `T: ValidAsZeroBits`.
///
/// # Example
///
/// ```
/// use baracuda_types::ValidAsZeroBits;
///
/// fn zeroed_vec<T: ValidAsZeroBits>(n: usize) -> Vec<T> {
///     // Safe because T: ValidAsZeroBits guarantees all-zero bytes are a valid T.
///     let mut v = Vec::with_capacity(n);
///     unsafe {
///         core::ptr::write_bytes(v.as_mut_ptr(), 0, n);
///         v.set_len(n);
///     }
///     v
/// }
///
/// let xs: Vec<f32> = zeroed_vec(4);
/// assert_eq!(xs, [0.0, 0.0, 0.0, 0.0]);
/// ```
pub unsafe trait ValidAsZeroBits: Copy + 'static {}

// ---- Primitives ---------------------------------------------------------

macro_rules! impl_zero_bits {
    ($($t:ty),* $(,)?) => {
        $(unsafe impl ValidAsZeroBits for $t {})*
    };
}

impl_zero_bits!(
    u8, u16, u32, u64, u128, usize,
    i8, i16, i32, i64, i128, isize,
    f32, f64,
    bool, (),
    Half, BFloat16, Complex32, Complex64,
);

// ---- Fixed-size arrays --------------------------------------------------

unsafe impl<T: ValidAsZeroBits, const N: usize> ValidAsZeroBits for [T; N] {}

// ---- Tuples -------------------------------------------------------------
//
// Up to arity 12, matching the `DeviceRepr` / `KernelArg` coverage.

macro_rules! impl_zero_bits_tuple {
    ($($t:ident),+) => {
        unsafe impl<$($t: ValidAsZeroBits),+> ValidAsZeroBits for ($($t,)+) {}
    };
}

impl_zero_bits_tuple!(A);
impl_zero_bits_tuple!(A, B);
impl_zero_bits_tuple!(A, B, C);
impl_zero_bits_tuple!(A, B, C, D);
impl_zero_bits_tuple!(A, B, C, D, E);
impl_zero_bits_tuple!(A, B, C, D, E, F);
impl_zero_bits_tuple!(A, B, C, D, E, F, G);
impl_zero_bits_tuple!(A, B, C, D, E, F, G, H);
impl_zero_bits_tuple!(A, B, C, D, E, F, G, H, I);
impl_zero_bits_tuple!(A, B, C, D, E, F, G, H, I, J);
impl_zero_bits_tuple!(A, B, C, D, E, F, G, H, I, J, K);
impl_zero_bits_tuple!(A, B, C, D, E, F, G, H, I, J, K, L);
