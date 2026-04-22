//! Green contexts (CUDA 12.4+) — partition a GPU's SMs into isolated
//! "green" subsets that each run their own stream/kernel pipeline without
//! contending for SMs with other green contexts on the same device.
//!
//! Availability: driver-reported CUDA 12.4 or newer. Falls through with
//! `LoaderError::SymbolNotFound` on older drivers.

use std::sync::Arc;

use baracuda_cuda_sys::types::{CUdevResource, CUdevResourceType, CUdevSmResource};
use baracuda_cuda_sys::{driver, CUcontext, CUdevResourceDesc, CUgreenCtx, CUstream};

use crate::device::Device;
use crate::error::{check, Result};

/// Fetch the device's full SM resource — the starting point for
/// `split_by_count`.
pub fn device_sm_resource(device: &Device) -> Result<CUdevResource> {
    let d = driver()?;
    let cu = d.cu_device_get_dev_resource()?;
    let mut r = CUdevResource::default();
    check(unsafe { cu(device.as_raw(), &mut r, CUdevResourceType::SM) })?;
    Ok(r)
}

/// Split an SM resource into groups of `min_count` SMs each. Returns the
/// vector of new resources plus the leftover remainder resource.
pub fn sm_resource_split_by_count(
    input: &CUdevResource,
    min_count: u32,
) -> Result<(Vec<CUdevResource>, CUdevResource)> {
    let d = driver()?;
    let cu = d.cu_dev_sm_resource_split_by_count()?;
    // Probe: pass nb_groups=0 to query how many groups would result.
    let mut nb: core::ffi::c_uint = 0;
    let mut remaining = CUdevResource::default();
    check(unsafe {
        cu(
            core::ptr::null_mut(),
            &mut nb,
            input,
            &mut remaining,
            0,
            min_count,
        )
    })?;
    let mut result = vec![CUdevResource::default(); nb as usize];
    if nb > 0 {
        check(unsafe {
            cu(
                result.as_mut_ptr(),
                &mut nb,
                input,
                &mut remaining,
                0,
                min_count,
            )
        })?;
    }
    Ok((result, remaining))
}

/// A green context — an SM-partitioned CUDA context. Drops destroy the
/// green context (does not affect the parent device context).
#[derive(Clone)]
pub struct GreenContext {
    inner: Arc<GreenContextInner>,
}

struct GreenContextInner {
    handle: CUgreenCtx,
}

unsafe impl Send for GreenContextInner {}
unsafe impl Sync for GreenContextInner {}

impl core::fmt::Debug for GreenContextInner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GreenContext")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

impl core::fmt::Debug for GreenContext {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.inner.fmt(f)
    }
}

impl GreenContext {
    /// Create a green context on `device` from a single resource (e.g. an
    /// SM partition produced by [`sm_resource_split_by_count`]).
    pub fn from_resource(device: &Device, mut resource: CUdevResource) -> Result<Self> {
        let d = driver()?;
        // 1) Pack the resource into a CUdevResourceDesc.
        let gen = d.cu_dev_resource_generate_desc()?;
        let mut desc: CUdevResourceDesc = core::ptr::null_mut();
        check(unsafe { gen(&mut desc, &mut resource, 1) })?;
        // 2) Actually create the green context.
        let create = d.cu_green_ctx_create()?;
        let mut handle: CUgreenCtx = core::ptr::null_mut();
        check(unsafe { create(&mut handle, desc, device.as_raw(), 0) })?;
        Ok(Self {
            inner: Arc::new(GreenContextInner { handle }),
        })
    }

    /// Convert this green context into a regular `CUcontext` handle that
    /// can be set as current or passed to other APIs.
    pub fn as_ctx_raw(&self) -> Result<CUcontext> {
        let d = driver()?;
        let cu = d.cu_ctx_from_green_ctx()?;
        let mut out: CUcontext = core::ptr::null_mut();
        check(unsafe { cu(&mut out, self.inner.handle) })?;
        Ok(out)
    }

    /// Fetch the SM resource the green context was created with.
    pub fn sm_resource(&self) -> Result<CUdevSmResource> {
        let d = driver()?;
        let cu = d.cu_green_ctx_get_dev_resource()?;
        let mut r = CUdevResource::default();
        check(unsafe { cu(self.inner.handle, &mut r, CUdevResourceType::SM) })?;
        Ok(r.as_sm())
    }

    /// Create a stream that only runs work on this green context's SMs.
    /// Returns the raw `CUstream` — baracuda's [`crate::Stream`] requires
    /// a full `Context`, so we return the raw handle for now.
    pub fn create_stream_raw(&self, flags: u32, priority: i32) -> Result<CUstream> {
        let d = driver()?;
        let cu = d.cu_green_ctx_stream_create()?;
        let mut s: CUstream = core::ptr::null_mut();
        check(unsafe { cu(&mut s, self.inner.handle, flags, priority) })?;
        Ok(s)
    }

    /// Wrap a raw green-context handle — see [`GreenContext::from_resource`]
    /// for the normal constructor.
    ///
    /// # Safety
    ///
    /// The caller guarantees `handle` is a valid `CUgreenCtx` that baracuda
    /// may take ownership of (destroyed on drop).
    pub unsafe fn from_raw(handle: CUgreenCtx) -> Self {
        Self {
            inner: Arc::new(GreenContextInner { handle }),
        }
    }

    /// Consume the green context and return its raw handle without
    /// destroying it. The caller takes over the destroy responsibility.
    pub fn into_raw(self) -> CUgreenCtx {
        match Arc::try_unwrap(self.inner) {
            Ok(mut inner) => {
                let h = inner.handle;
                inner.handle = core::ptr::null_mut();
                h
            }
            Err(arc) => arc.handle,
        }
    }

    #[inline]
    pub fn as_raw(&self) -> CUgreenCtx {
        self.inner.handle
    }
}

impl Drop for GreenContextInner {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        if let Ok(d) = driver() {
            if let Ok(cu) = d.cu_green_ctx_destroy() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}
