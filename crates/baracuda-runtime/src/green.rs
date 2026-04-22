//! Green contexts (Runtime API, CUDA 13.1+) — lightweight sub-contexts
//! that share the primary context's memory but carve out an SM subset.
//!
//! Returns [`crate::Error::FeatureNotSupported`] on older drivers.
//!
//! For the Driver-side API see `baracuda_driver::green`.

use baracuda_cuda_sys::runtime::runtime;
use baracuda_cuda_sys::runtime::types::{cudaGreenCtx_t, cudaStream_t};
use baracuda_types::CudaVersion;

use crate::error::{check, Error, Result};
use crate::event::Event;
use crate::stream::Stream;

fn require_green_ctx() -> Result<()> {
    let installed = crate::init::driver_version()?;
    // The runtime green-context API lands in CUDA 13.1.
    if installed.at_least(13, 1) {
        Ok(())
    } else {
        Err(Error::FeatureNotSupported {
            api: "cudaGreenCtx*",
            since: CudaVersion::from_major_minor(13, 1),
        })
    }
}

/// A green context. Drop destroys it.
pub struct GreenContext {
    handle: cudaGreenCtx_t,
}

unsafe impl Send for GreenContext {}
unsafe impl Sync for GreenContext {}

impl core::fmt::Debug for GreenContext {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GreenContext")
            .field("handle", &self.handle)
            .finish()
    }
}

impl GreenContext {
    /// Create a green context from an opaque `cudaDevResourceDesc`
    /// pointer (built via the driver-side resource-desc APIs).
    ///
    /// # Safety
    ///
    /// `desc` must be a valid `cudaDevResourceDesc` descriptor.
    pub unsafe fn from_resource_desc(desc: *const core::ffi::c_void, flags: u32) -> Result<Self> {
        require_green_ctx()?;
        let r = runtime()?;
        let cu = r.cuda_device_create_green_ctx()?;
        let mut h: cudaGreenCtx_t = core::ptr::null_mut();
        check(cu(&mut h, desc, flags))?;
        Ok(Self { handle: h })
    }

    /// Wrap an already-created handle.
    ///
    /// # Safety
    ///
    /// `handle` must be live. Drop destroys it.
    pub unsafe fn from_raw(handle: cudaGreenCtx_t) -> Self {
        Self { handle }
    }

    #[inline]
    pub fn as_raw(&self) -> cudaGreenCtx_t {
        self.handle
    }

    /// Record `event` on this green context.
    pub fn record_event(&self, event: &Event) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_green_ctx_record_event()?;
        check(unsafe { cu(self.handle, event.as_raw()) })
    }

    /// Wait for `event` on this green context.
    pub fn wait_event(&self, event: &Event) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_green_ctx_wait_event()?;
        check(unsafe { cu(self.handle, event.as_raw()) })
    }

    /// Create a stream on this green context.
    pub fn create_stream(&self, flags: u32, priority: i32) -> Result<Stream> {
        let r = runtime()?;
        let cu = r.cuda_green_ctx_stream_create()?;
        let mut s: cudaStream_t = core::ptr::null_mut();
        check(unsafe { cu(&mut s, self.handle, flags, priority) })?;
        // SAFETY: driver returned a live stream; Stream::from_raw adopts
        // ownership (drops with cudaStreamDestroy).
        unsafe { Ok(Stream::from_raw(s)) }
    }
}

impl Drop for GreenContext {
    fn drop(&mut self) {
        if let Ok(r) = runtime() {
            if let Ok(cu) = r.cuda_green_ctx_destroy() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}
