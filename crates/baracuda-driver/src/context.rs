//! CUDA contexts — both primary (shared with the Runtime API) and explicit.
//!
//! A [`Context`] owns the handle returned by `cuCtxCreate`. Contexts are
//! reference-counted via `Arc` so multiple streams/events/modules can
//! share ownership; the underlying `cuCtxDestroy` runs when the last clone
//! drops.

use std::sync::Arc;

use baracuda_cuda_sys::types::CUcontext_flags;
use baracuda_cuda_sys::{driver, CUcontext};

use crate::device::Device;
use crate::error::{check, Result};
use crate::init::init;

/// A CUDA context created by `cuCtxCreate`.
///
/// Multiple [`Context`] clones refer to the same underlying driver context.
#[derive(Clone, Debug)]
pub struct Context {
    inner: Arc<ContextInner>,
}

struct ContextInner {
    handle: CUcontext,
    device: Device,
}

// SAFETY: CUcontext is a raw pointer, but NVIDIA documents that a context
// object may be shared between threads so long as each thread calls
// `cuCtxSetCurrent` before issuing work. Concurrent kernel submission on
// different streams is explicitly supported.
unsafe impl Send for ContextInner {}
unsafe impl Sync for ContextInner {}

impl core::fmt::Debug for ContextInner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Context")
            .field("handle", &self.handle)
            .field("device", &self.device)
            .finish()
    }
}

impl Context {
    /// Create a new context on `device` with default scheduling flags.
    pub fn new(device: &Device) -> Result<Self> {
        Self::with_flags(device, CUcontext_flags::SCHED_AUTO)
    }

    /// Create a new context on `device`, passing `flags` verbatim to
    /// `cuCtxCreate`. See [`baracuda_cuda_sys::types::CUcontext_flags`] for
    /// the permitted values.
    pub fn with_flags(device: &Device, flags: u32) -> Result<Self> {
        init()?;
        let d = driver()?;
        let cu = d.cu_ctx_create()?;
        let mut ctx: CUcontext = core::ptr::null_mut();
        // SAFETY: `ctx` is a writable pointer; `device.0` is a live CUdevice.
        check(unsafe { cu(&mut ctx, flags, device.0) })?;
        Ok(Self {
            inner: Arc::new(ContextInner {
                handle: ctx,
                device: *device,
            }),
        })
    }

    /// Retrieve the thread's currently-current context, if any. Returns
    /// `Ok(None)` when no context is current.
    ///
    /// **Note:** the returned `Context` is a _non-owning_ view — its `Drop`
    /// will not call `cuCtxDestroy` on the handle. Use this only for
    /// interop inspection, not lifecycle management.
    pub fn current() -> Result<Option<CUcontext>> {
        init()?;
        let d = driver()?;
        let cu = d.cu_ctx_get_current()?;
        let mut ctx: CUcontext = core::ptr::null_mut();
        check(unsafe { cu(&mut ctx) })?;
        if ctx.is_null() {
            Ok(None)
        } else {
            Ok(Some(ctx))
        }
    }

    /// Make this context current on the calling thread.
    pub fn set_current(&self) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_ctx_set_current()?;
        // SAFETY: `self.inner.handle` is alive for at least the duration of
        // this call (held by Arc).
        check(unsafe { cu(self.inner.handle) })
    }

    /// Push this context onto the thread's context stack.
    pub fn push(&self) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_ctx_push_current()?;
        check(unsafe { cu(self.inner.handle) })
    }

    /// Pop the top context off the thread's context stack.
    pub fn pop() -> Result<CUcontext> {
        init()?;
        let d = driver()?;
        let cu = d.cu_ctx_pop_current()?;
        let mut ctx: CUcontext = core::ptr::null_mut();
        check(unsafe { cu(&mut ctx) })?;
        Ok(ctx)
    }

    /// Block the calling thread until all work previously submitted to
    /// streams in this context has completed.
    pub fn synchronize(&self) -> Result<()> {
        self.set_current()?;
        let d = driver()?;
        let cu = d.cu_ctx_synchronize()?;
        check(unsafe { cu() })
    }

    /// API version this context was created with (major*1000 + minor*10, e.g. 12060).
    pub fn api_version(&self) -> Result<u32> {
        let d = driver()?;
        let cu = d.cu_ctx_get_api_version()?;
        let mut v: core::ffi::c_uint = 0;
        check(unsafe { cu(self.inner.handle, &mut v) })?;
        Ok(v)
    }

    /// Device ordinal of the thread's currently-current context.
    /// Fails with `CUDA_ERROR_INVALID_CONTEXT` if no context is current.
    pub fn current_device() -> Result<Device> {
        let d = driver()?;
        let cu = d.cu_ctx_get_device()?;
        let mut dev = baracuda_cuda_sys::CUdevice::default();
        check(unsafe { cu(&mut dev) })?;
        Ok(Device(dev))
    }

    /// Flags the current context was created with (`SCHED_*`, `MAP_HOST`, etc.).
    ///
    /// Operates on the thread's current context, so make sure you've made
    /// this one current first.
    pub fn current_flags() -> Result<u32> {
        let d = driver()?;
        let cu = d.cu_ctx_get_flags()?;
        let mut f: core::ffi::c_uint = 0;
        check(unsafe { cu(&mut f) })?;
        Ok(f)
    }

    /// Query a resource limit of the current context. `limit` is one of
    /// [`baracuda_cuda_sys::types::CUlimit`].
    pub fn get_limit(limit: u32) -> Result<usize> {
        let d = driver()?;
        let cu = d.cu_ctx_get_limit()?;
        let mut v: usize = 0;
        check(unsafe { cu(&mut v, limit) })?;
        Ok(v)
    }

    /// Set a resource limit of the current context. `limit` is one of
    /// [`baracuda_cuda_sys::types::CUlimit`]. Not all limits are
    /// writable on every device.
    pub fn set_limit(limit: u32, value: usize) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_ctx_set_limit()?;
        check(unsafe { cu(limit, value) })
    }

    /// Current context's L1/shared-memory preference. Values are from
    /// [`baracuda_cuda_sys::types::CUfunc_cache`].
    pub fn cache_config() -> Result<u32> {
        let d = driver()?;
        let cu = d.cu_ctx_get_cache_config()?;
        let mut c: core::ffi::c_uint = 0;
        check(unsafe { cu(&mut c) })?;
        Ok(c)
    }

    /// Set the current context's L1/shared-memory preference.
    pub fn set_cache_config(config: u32) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_ctx_set_cache_config()?;
        check(unsafe { cu(config) })
    }

    /// Hardware-supported stream priority range `(least_priority, greatest_priority)`.
    /// On most GPUs that's `(0, -1)` — lower numbers = higher priority.
    pub fn stream_priority_range() -> Result<(i32, i32)> {
        let d = driver()?;
        let cu = d.cu_ctx_get_stream_priority_range()?;
        let mut least: core::ffi::c_int = 0;
        let mut greatest: core::ffi::c_int = 0;
        check(unsafe { cu(&mut least, &mut greatest) })?;
        Ok((least, greatest))
    }

    /// Enable peer access from the current context to `peer`'s context.
    /// After this call, kernels in the current context can read/write
    /// allocations owned by `peer`.
    pub fn enable_peer_access(peer: &Context) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_ctx_enable_peer_access()?;
        check(unsafe { cu(peer.inner.handle, 0) })
    }

    /// Revert [`enable_peer_access`](Self::enable_peer_access).
    pub fn disable_peer_access(peer: &Context) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_ctx_disable_peer_access()?;
        check(unsafe { cu(peer.inner.handle) })
    }

    /// The [`Device`] this context was created on.
    #[inline]
    pub fn device(&self) -> Device {
        self.inner.device
    }

    /// Raw `CUcontext`. Use with care.
    #[inline]
    pub fn as_raw(&self) -> CUcontext {
        self.inner.handle
    }

    /// Driver-assigned 64-bit context ID. Useful for correlating
    /// CUPTI / Nsight traces against this `Context`.
    pub fn id(&self) -> Result<u64> {
        let d = driver()?;
        let cu = d.cu_ctx_get_id()?;
        let mut out: u64 = 0;
        check(unsafe { cu(self.inner.handle, &mut out) })?;
        Ok(out)
    }

    /// Record `event` on this context (rather than tying it to a specific
    /// stream). CUDA 12+.
    pub fn record_event(&self, event: &crate::Event) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_ctx_record_event()?;
        check(unsafe { cu(self.inner.handle, event.as_raw()) })
    }

    /// Make this context wait on `event`. CUDA 12+.
    pub fn wait_event(&self, event: &crate::Event) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_ctx_wait_event()?;
        check(unsafe { cu(self.inner.handle, event.as_raw()) })
    }
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        if let Ok(d) = driver() {
            if let Ok(cu) = d.cu_ctx_destroy() {
                // SAFETY: `self.handle` was produced by cuCtxCreate and has
                // not been destroyed elsewhere (we're dropping the last Arc).
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}
