//! Shared type vocabulary for the baracuda CUDA stack.
//!
//! This crate contains only pure-data types — no I/O, no `libloading`, no
//! runtime machinery. It's intended to be a cheap dependency for any crate
//! (including user-authored CUDA wrappers) that wants to speak baracuda's
//! type vocabulary without pulling in the loader infrastructure.
//!
//! # Modules
//!
//! - [`numeric`] — [`Half`], [`BFloat16`], [`Complex32`], [`Complex64`].
//! - [`device_repr`] — the [`DeviceRepr`] trait marking ABI-stable types.
//! - [`kernel_arg`] — the [`KernelArg`] trait for marshalling kernel arguments.
//! - [`version`] — [`CudaVersion`] and the [`Feature`] enum with [`supports`].
//! - [`status`] — the [`CudaStatus`] trait every library's status enum implements.
//! - [`stream_mode`] — the [`StreamMode`] enum (Legacy vs PerThread default streams).
//!
//! # Examples
//!
//! Round-trip a value through [`Half`] (IEEE 754 binary16):
//!
//! ```
//! use baracuda_types::Half;
//!
//! let h = Half::from_f32(1.5);
//! assert_eq!(h.to_bits(), 0x3E00);
//! assert_eq!(h.to_f32(), 1.5);
//! ```
//!
//! Round-trip a value through [`BFloat16`] (top 16 bits of an `f32`):
//!
//! ```
//! use baracuda_types::BFloat16;
//!
//! let b = BFloat16::from_f32(1.5);
//! // 1.5 = 0x3FC0_0000 in f32; bf16 takes the top 16 bits = 0x3FC0.
//! assert_eq!(b.to_bits(), 0x3FC0);
//! assert_eq!(b.to_f32(), 1.5);
//! ```
//!
//! Build and conjugate a [`Complex32`]:
//!
//! ```
//! use baracuda_types::Complex32;
//!
//! let z = Complex32::new(3.0, 4.0);
//! assert_eq!(z.norm_sqr(), 25.0); // 3² + 4²
//! let zc = z.conj();
//! assert_eq!(zc.re, 3.0);
//! assert_eq!(zc.im, -4.0);
//! ```
//!
//! The [`DeviceRepr`] trait is a compile-time marker for types that are
//! safe to expose to a CUDA kernel. It is implemented for primitives, the
//! numeric helpers above, fixed-size arrays, and small tuples:
//!
//! ```
//! use baracuda_types::{BFloat16, Complex32, DeviceRepr, Half};
//!
//! fn assert_kernel_safe<T: DeviceRepr>() {}
//!
//! assert_kernel_safe::<f32>();
//! assert_kernel_safe::<Half>();
//! assert_kernel_safe::<BFloat16>();
//! assert_kernel_safe::<Complex32>();
//! assert_kernel_safe::<[u8; 16]>();
//! ```

#![no_std]
#![warn(missing_debug_implementations)]

pub mod device_repr;
pub mod external_impls;
pub mod host_slice;
pub mod kernel_arg;
pub mod numeric;
pub mod status;
pub mod stream_mode;
pub mod version;
pub mod zero_bits;

pub use device_repr::DeviceRepr;
pub use host_slice::HostSlice;
pub use kernel_arg::KernelArg;
pub use numeric::{BFloat16, Complex32, Complex64, Half};
pub use status::CudaStatus;
pub use stream_mode::StreamMode;
pub use version::{supports, CudaVersion, Feature};
pub use zero_bits::ValidAsZeroBits;

/// `#[derive(DeviceRepr)]` — attribute macro re-exported from
/// `baracuda-types-derive` when the `derive` feature is enabled.
#[cfg(feature = "derive")]
pub use baracuda_types_derive::DeviceRepr;
