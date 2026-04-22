//! Safe Rust wrappers for the CUDA Driver API.
//!
//! This crate takes the raw FFI in [`baracuda_cuda_sys`] and dresses it up
//! with RAII handles, typed memory, lifetime-checked slices, and a kernel
//! launch builder. It deliberately does not hide the Driver-API model:
//! contexts are explicit, modules are explicit, streams are explicit.
//!
//! # Quickstart
//!
//! ```no_run
//! use baracuda_driver::{Context, Device, DeviceBuffer, Module, Stream};
//!
//! # fn demo() -> baracuda_driver::Result<()> {
//! let device = Device::get(0)?;
//! let ctx = Context::new(&device)?;
//! let stream = Stream::new(&ctx)?;
//! let host_data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
//! let device_data = DeviceBuffer::from_slice(&ctx, &host_data)?;
//! let mut back = vec![0.0f32; host_data.len()];
//! device_data.copy_to_host(&mut back)?;
//! stream.synchronize()?;
//! assert_eq!(host_data, back);
//! # Ok(())
//! # }
//! ```
//!
//! # Modules
//!
//! - [`device`] — [`Device`] enumeration and attributes.
//! - [`context`] — [`Context`] (explicit CUDA contexts + primary-context reuse).
//! - [`stream`] — [`Stream`], ordered async work queues.
//! - [`event`] — [`Event`], synchronization and timing.
//! - [`memory`] — [`DeviceBuffer<T>`], [`DeviceSlice<'_, T>`], [`DeviceSliceMut<'_, T>`].
//! - [`module`] — [`Module`], [`Function`] (PTX/CUBIN loading).
//! - [`launch`] — [`launch::LaunchBuilder`] for `cuLaunchKernel`.
//! - [`init()`] — `init()` helper and driver version queries.

#![warn(missing_debug_implementations)]

pub mod array;
pub mod context;
pub mod coredump;
pub mod device;
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
pub mod library;
pub mod memcpy2d;
pub mod memcpy3d;
pub mod memory;
pub mod mempool;
pub mod module;
pub mod multicast;
pub mod occupancy;
pub mod pinned;
pub mod pointer;
pub mod profiler;
pub mod stream;
pub mod tensor_map;
pub mod user_object;
pub mod vmm;

pub use array::{
    Array, ArrayFormat, SurfaceObject, TextureAddressMode, TextureDesc, TextureFilterMode,
    TextureObject,
};
pub use context::Context;
pub use device::Device;
pub use error::{Error, Result};
pub use event::Event;
pub use graph::{instantiate_flags, CaptureMode, Graph, GraphExec, GraphNode};
pub use init::{init, version};
pub use launch::{Dim3, LaunchBuilder};
pub use memory::{
    mem_get_info, DeviceBuffer, DeviceSlice, DeviceSliceMut, ManagedAttach, ManagedBuffer,
    MemAdvise,
};
pub use module::{Function, Module};
pub use stream::Stream;
