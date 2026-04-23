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
