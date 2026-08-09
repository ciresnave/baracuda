//! Re-export of the `DeviceRepr` marker trait plus impls for this crate's own
//! numeric wrapper types.
//!
//! The trait itself — together with the primitive, `[T; N]`, tuple, and foreign
//! scalar (`half` / `float8`) impls — is canonically owned by the driver-free,
//! neutral [`unpopped_vocab`] crate. `baracuda-types` re-exports it so
//! existing consumers keep writing `baracuda_types::DeviceRepr`, and adds the
//! impls for its own [`crate::numeric`] wrapper types here (a local type with a
//! foreign trait — orphan-rule legal). User `#[repr(C)]` structs derive it via
//! `#[derive(baracuda_types::DeviceRepr)]` (the `derive` feature).

use crate::numeric::{BFloat16, Complex32, Complex64, Half};

pub use unpopped_vocab::DeviceRepr;

// SAFETY: the wrappers in `crate::numeric` are `#[repr(transparent)]` /
// `#[repr(C)]` over primitives that are themselves `DeviceRepr`.
unsafe impl DeviceRepr for Half {}
unsafe impl DeviceRepr for BFloat16 {}
unsafe impl DeviceRepr for Complex32 {}
unsafe impl DeviceRepr for Complex64 {}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_device_repr<T: DeviceRepr>() {}

    #[test]
    fn numeric_wrappers_are_device_repr() {
        assert_device_repr::<Half>();
        assert_device_repr::<BFloat16>();
        assert_device_repr::<Complex32>();
        assert_device_repr::<Complex64>();
    }

    #[test]
    fn reexported_trait_covers_primitives_and_aggregates() {
        // Confirms the re-export from baracuda-kernel-vocab resolves and that
        // its primitive / array / tuple impls are visible through this crate.
        assert_device_repr::<f32>();
        assert_device_repr::<[u8; 16]>();
        assert_device_repr::<(f32, f32)>();
    }
}
