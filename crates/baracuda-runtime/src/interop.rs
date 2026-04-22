//! Zero-cost conversions between `baracuda-driver` and `baracuda-runtime`
//! handles.
//!
//! `CUstream == cudaStream_t` and `CUevent == cudaEvent_t` at the C level,
//! so the conversions are essentially pointer reinterpretation.
//!
//! Enable with the `driver-interop` feature on `baracuda-runtime` (pulled
//! in automatically when the umbrella `baracuda` crate has both `driver`
//! and `runtime` features on).

#![cfg(feature = "driver-interop")]

use baracuda_cuda_sys::runtime::{cudaEvent_t, cudaStream_t};
use baracuda_cuda_sys::{CUevent, CUstream};

/// Extension trait: view a driver-side [`baracuda_driver::Stream`] as a
/// runtime-side `cudaStream_t`. Non-owning — the driver Stream still owns
/// the underlying CUDA resource.
pub trait DriverStreamExt {
    fn as_raw_runtime(&self) -> cudaStream_t;
}

/// Extension trait: view a driver-side [`baracuda_driver::Event`] as a
/// runtime-side `cudaEvent_t`.
pub trait DriverEventExt {
    fn as_raw_runtime(&self) -> cudaEvent_t;
}

impl DriverStreamExt for baracuda_driver::Stream {
    #[inline]
    fn as_raw_runtime(&self) -> cudaStream_t {
        self.as_raw() as cudaStream_t
    }
}

impl DriverEventExt for baracuda_driver::Event {
    #[inline]
    fn as_raw_runtime(&self) -> cudaEvent_t {
        self.as_raw() as cudaEvent_t
    }
}

impl crate::Stream {
    /// View this runtime stream as a raw driver `CUstream`. Non-owning.
    #[inline]
    pub fn as_raw_driver(&self) -> CUstream {
        self.as_raw() as CUstream
    }
}

impl crate::Event {
    /// View this runtime event as a raw driver `CUevent`. Non-owning.
    #[inline]
    pub fn as_raw_driver(&self) -> CUevent {
        self.as_raw() as CUevent
    }
}
