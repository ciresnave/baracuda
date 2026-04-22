//! Safe Rust wrappers for the CUDA Runtime API.
//!
//! The Runtime API is "higher level" than the Driver API: contexts are
//! implicit (each device has a primary context the runtime uses
//! automatically), kernels are typically linked at build time by `nvcc`,
//! and most operations dispatch to the current thread's current device.
//! baracuda-runtime mirrors the Driver-side types where it makes sense
//! ([`Device`], [`Stream`], [`Event`], [`DeviceBuffer`]) and uses the
//! CUDA 12.0+ library API ([`Library`], [`Kernel`]) for loading PTX at
//! runtime — the Driver-API equivalent of `Module::load_ptx` +
//! `Module::get_function`.
//!
//! # Driver ↔ Runtime interop
//!
//! `CUstream` and `cudaStream_t` are the same C type. With the
//! `driver-interop` feature, `Stream::as_raw_driver()` and
//! `Event::as_raw_driver()` return views usable by `baracuda-driver`
//! APIs. See [`interop`].

#![warn(missing_debug_implementations)]

pub mod array;
pub mod device;
pub mod driver_entry;
pub mod error;
pub mod event;
pub mod external;
pub mod graph;
pub mod graphics;
pub mod green;
pub mod init;
pub mod ipc;
pub mod launch;
pub mod launch_attr;
pub mod memcpy2d;
pub mod memcpy3d;
pub mod memory;
pub mod mempool;
pub mod module;
pub mod multicast;
pub mod profiler;
pub mod query;
pub mod stream;
pub mod user_object;
pub mod vmm;

#[cfg(feature = "driver-interop")]
pub mod interop;

pub use device::Device;
pub use error::{Error, Result};
pub use event::Event;
pub use graph::{CaptureMode, Graph, GraphExec, GraphNode, UpdateResult};
pub use init::{
    device_synchronize, driver_version, get_device_flags, last_error, peek_last_error,
    runtime_version, set_device_flags,
};
pub use launch::{Dim3, LaunchBuilder};
pub use memory::DeviceBuffer;
pub use module::{Kernel, Library};
pub use stream::Stream;
