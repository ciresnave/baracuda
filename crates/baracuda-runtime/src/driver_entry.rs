//! Runtime-to-Driver entry-point bridge — `cudaGetDriverEntryPoint`.
//!
//! Most code using this crate goes through the typed Driver loader in
//! `baracuda-driver`. The runtime's `cudaGetDriverEntryPoint` is useful
//! for one narrow case: asking the installed runtime which driver
//! symbol name / fptr it would resolve for a given API, without
//! touching `libcuda` directly. Handy for diagnostic tools and for
//! picking up versioned symbol variants (`_ptsz`, `_v2`, …).

use core::ffi::{c_int, c_void};
use std::ffi::CString;

use baracuda_cuda_sys::runtime::runtime;

use crate::error::{check, Error, Result};

/// Typed outcome of [`driver_entry_point`]. `status` mirrors the
/// `cudaDriverEntryPointQueryResult` enum reported by the runtime:
/// 0 = Success, 1 = SymbolNotFound, 2 = VersionNotSufficient.
#[derive(Copy, Clone, Debug)]
pub struct DriverEntryPoint {
    pub fn_ptr: *mut c_void,
    pub status: i32,
}

impl DriverEntryPoint {
    #[inline]
    pub fn is_success(&self) -> bool {
        self.status == 0 && !self.fn_ptr.is_null()
    }
}

/// Resolve a Driver-API symbol by name through the Runtime API
/// (`cudaGetDriverEntryPoint`). `flags = 0` = default; bit 0 = legacy
/// stream, bit 1 = per-thread stream (mirrors `cuGetProcAddress`).
pub fn driver_entry_point(symbol: &str, flags: u64) -> Result<DriverEntryPoint> {
    let c_sym = CString::new(symbol).map_err(|_| {
        Error::Loader(baracuda_core::LoaderError::SymbolNotFound {
            library: "cuda-runtime",
            symbol: "cudaGetDriverEntryPoint(symbol contained a NUL byte)",
        })
    })?;
    let r = runtime()?;
    let cu = r.cuda_get_driver_entry_point()?;
    let mut fn_ptr: *mut c_void = core::ptr::null_mut();
    let mut driver_status: c_int = 0;
    check(unsafe { cu(c_sym.as_ptr(), &mut fn_ptr, flags, &mut driver_status) })?;
    Ok(DriverEntryPoint {
        fn_ptr,
        status: driver_status,
    })
}
