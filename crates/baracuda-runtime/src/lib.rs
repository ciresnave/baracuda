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
//!
//! # Examples
//!
//! **Device query** — discover the visible GPUs and inspect compute
//! capability + SM count.
//!
//! ```no_run
//! use baracuda_runtime::Device;
//!
//! # fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! let count = Device::count()?;
//! for d in Device::all()? {
//!     let (major, minor) = d.compute_capability()?;
//!     println!("device {}: cc {major}.{minor}, {} SMs", d.ordinal(),
//!         d.multiprocessor_count()?);
//! }
//! # let _ = count; Ok(()) }
//! ```
//!
//! **Async memory copy** — overlap H2D upload with later kernel launches
//! by issuing on a non-blocking [`Stream`].
//!
//! ```no_run
//! use baracuda_runtime::{Device, DeviceBuffer, Stream};
//!
//! # fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! Device::from_ordinal(0).set_current()?;
//! let stream = Stream::non_blocking()?;
//!
//! let host: Vec<f32> = (0..4096).map(|i| i as f32).collect();
//! let device: DeviceBuffer<f32> = DeviceBuffer::new(host.len())?;
//! device.copy_from_host_async(&host, &stream)?;
//!
//! let mut back = vec![0.0f32; host.len()];
//! device.copy_to_host_async(&mut back, &stream)?;
//! stream.synchronize()?;
//! assert_eq!(host, back);
//! # Ok(()) }
//! ```
//!
//! **Event timing** — measure the elapsed device time between two
//! [`Event::record`] calls.
//!
//! ```no_run
//! use baracuda_runtime::{Device, DeviceBuffer, Event, Stream};
//!
//! # fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! Device::from_ordinal(0).set_current()?;
//! let stream = Stream::new()?;
//! let start = Event::new()?;
//! let end   = Event::new()?;
//!
//! // Record START -> issue some work -> record END.
//! start.record(&stream)?;
//! let buf: DeviceBuffer<f32> = DeviceBuffer::zeros(1 << 20)?;
//! end.record(&stream)?;
//! end.synchronize()?;
//!
//! let ms = Event::elapsed_time_ms(&start, &end)?;
//! println!("device-side elapsed: {ms} ms");
//! # let _ = buf; Ok(()) }
//! ```

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
