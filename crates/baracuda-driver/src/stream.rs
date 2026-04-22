//! CUDA streams — ordered queues of work on a device.

use std::sync::Arc;

use baracuda_cuda_sys::types::CUstream_flags;
use baracuda_cuda_sys::{driver, CUstream};

use crate::context::Context;
use crate::error::{check, Result};

/// An asynchronous work queue on a CUDA device.
///
/// Work submitted to the same stream executes in order; work on different
/// streams may run concurrently, subject to device scheduling. Streams are
/// `Send + Sync` — CUDA explicitly permits concurrent submission from
/// multiple host threads.
#[derive(Clone)]
pub struct Stream {
    inner: Arc<StreamInner>,
}

struct StreamInner {
    handle: CUstream,
    // Hold the owning context so it outlives the stream.
    context: Context,
}

// SAFETY: NVIDIA documents that a CUstream may be used from any thread.
unsafe impl Send for StreamInner {}
unsafe impl Sync for StreamInner {}

impl core::fmt::Debug for StreamInner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Stream")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

impl core::fmt::Debug for Stream {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.inner.fmt(f)
    }
}

impl Stream {
    /// Create a new stream on `context` with default flags (blocking wrt the
    /// legacy default stream).
    pub fn new(context: &Context) -> Result<Self> {
        Self::with_flags(context, CUstream_flags::DEFAULT)
    }

    /// Create a non-blocking stream — work on this stream does not
    /// synchronize with the legacy null stream.
    pub fn non_blocking(context: &Context) -> Result<Self> {
        Self::with_flags(context, CUstream_flags::NON_BLOCKING)
    }

    /// Create a stream with a raw flag bitmask (see [`CUstream_flags`]).
    pub fn with_flags(context: &Context, flags: u32) -> Result<Self> {
        context.set_current()?;
        let d = driver()?;
        let cu = d.cu_stream_create()?;
        let mut stream: CUstream = core::ptr::null_mut();
        // SAFETY: writable pointer; flags are from a known module.
        check(unsafe { cu(&mut stream, flags) })?;
        Ok(Self {
            inner: Arc::new(StreamInner {
                handle: stream,
                context: context.clone(),
            }),
        })
    }

    /// Create a stream with a specific priority. Use
    /// [`Context::stream_priority_range`] to discover the legal range on
    /// this device (lower = higher priority).
    pub fn with_priority(context: &Context, flags: u32, priority: i32) -> Result<Self> {
        context.set_current()?;
        let d = driver()?;
        let cu = d.cu_stream_create_with_priority()?;
        let mut stream: CUstream = core::ptr::null_mut();
        check(unsafe { cu(&mut stream, flags, priority) })?;
        Ok(Self {
            inner: Arc::new(StreamInner {
                handle: stream,
                context: context.clone(),
            }),
        })
    }

    /// This stream's scheduling priority.
    pub fn priority(&self) -> Result<i32> {
        let d = driver()?;
        let cu = d.cu_stream_get_priority()?;
        let mut p: core::ffi::c_int = 0;
        check(unsafe { cu(self.inner.handle, &mut p) })?;
        Ok(p)
    }

    /// This stream's flags bitmask.
    pub fn flags(&self) -> Result<u32> {
        let d = driver()?;
        let cu = d.cu_stream_get_flags()?;
        let mut f: core::ffi::c_uint = 0;
        check(unsafe { cu(self.inner.handle, &mut f) })?;
        Ok(f)
    }

    /// Enqueue a host-side callback on this stream. The callback runs on
    /// a driver-owned thread after all prior stream work completes.
    ///
    /// The closure is boxed and freed after it runs; a panic inside will
    /// abort the process (there's no way to propagate it through the C
    /// callback). Keep the closure simple.
    pub fn launch_host_func<F>(&self, f: F) -> Result<()>
    where
        F: FnOnce() + Send + 'static,
    {
        use core::ffi::c_void;

        // Box up the closure and hand the raw pointer to CUDA.
        let boxed: Box<Box<dyn FnOnce() + Send>> = Box::new(Box::new(f));
        let raw = Box::into_raw(boxed) as *mut c_void;

        unsafe extern "C" fn trampoline(user_data: *mut c_void) {
            // SAFETY: user_data was `Box::into_raw`'d just above.
            let f: Box<Box<dyn FnOnce() + Send>> =
                unsafe { Box::from_raw(user_data as *mut Box<dyn FnOnce() + Send>) };
            (*f)();
        }

        let d = driver()?;
        let cu = d.cu_launch_host_func()?;
        // SAFETY: trampoline owns and frees the boxed closure; stream handle is live.
        let rc = unsafe { cu(self.inner.handle, Some(trampoline), raw) };
        if rc != baracuda_cuda_sys::CUresult::SUCCESS {
            // Reclaim the box so we don't leak on submission failure.
            // SAFETY: cuLaunchHostFunc didn't take ownership on error.
            drop(unsafe { Box::from_raw(raw as *mut Box<dyn FnOnce() + Send>) });
            return Err(crate::error::Error::Status { status: rc });
        }
        Ok(())
    }

    /// Block the calling thread until all work previously enqueued on this
    /// stream has completed.
    pub fn synchronize(&self) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_stream_synchronize()?;
        check(unsafe { cu(self.inner.handle) })
    }

    /// `Ok(true)` if the stream has completed all queued work, `Ok(false)`
    /// if work is still outstanding.
    pub fn is_complete(&self) -> Result<bool> {
        use baracuda_cuda_sys::CUresult;
        let d = driver()?;
        let cu = d.cu_stream_query()?;
        let res = unsafe { cu(self.inner.handle) };
        match res {
            CUresult::SUCCESS => Ok(true),
            CUresult::ERROR_NOT_READY => Ok(false),
            other => Err(crate::error::Error::Status { status: other }),
        }
    }

    /// The [`Context`] this stream lives in.
    #[inline]
    pub fn context(&self) -> &Context {
        &self.inner.context
    }

    /// Raw `CUstream` handle. Use with care.
    #[inline]
    pub fn as_raw(&self) -> CUstream {
        self.inner.handle
    }

    /// Return the driver-assigned 64-bit ID for this stream. Useful for
    /// correlating CUPTI traces against baracuda streams.
    pub fn id(&self) -> Result<u64> {
        let d = driver()?;
        let cu = d.cu_stream_get_id()?;
        let mut out: u64 = 0;
        check(unsafe { cu(self.inner.handle, &mut out) })?;
        Ok(out)
    }

    /// Copy all CUDA-managed attributes (access policy window, sync
    /// policy) from `src` onto `self`. Does not copy priority or flags
    /// (those are set at stream creation time).
    pub fn copy_attributes_from(&self, src: &Stream) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_stream_copy_attributes()?;
        check(unsafe { cu(self.inner.handle, src.inner.handle) })
    }

    /// Associate a managed-memory region with this stream. Pass
    /// `flags = 0` for the default ("one thread").
    pub fn attach_mem_async(
        &self,
        dptr: baracuda_cuda_sys::CUdeviceptr,
        length: usize,
        flags: u32,
    ) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_stream_attach_mem_async()?;
        check(unsafe { cu(self.inner.handle, dptr, length, flags) })
    }

    /// Enqueue a 32-bit write of `value` to device memory `addr` on this
    /// stream, ordered like any other stream op.
    ///
    /// `flags` is a bitmask of
    /// [`baracuda_cuda_sys::types::CUstreamWriteValue_flags`].
    pub fn write_value_32(
        &self,
        addr: baracuda_cuda_sys::CUdeviceptr,
        value: u32,
        flags: u32,
    ) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_stream_write_value_32()?;
        check(unsafe { cu(self.inner.handle, addr, value, flags) })
    }

    pub fn write_value_64(
        &self,
        addr: baracuda_cuda_sys::CUdeviceptr,
        value: u64,
        flags: u32,
    ) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_stream_write_value_64()?;
        check(unsafe { cu(self.inner.handle, addr, value, flags) })
    }

    /// Block the stream until the device memory at `addr` satisfies the
    /// condition specified by `flags` (see
    /// [`baracuda_cuda_sys::types::CUstreamWaitValue_flags`] —
    /// GEQ / EQ / AND / NOR, optionally OR'd with FLUSH).
    pub fn wait_value_32(
        &self,
        addr: baracuda_cuda_sys::CUdeviceptr,
        value: u32,
        flags: u32,
    ) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_stream_wait_value_32()?;
        check(unsafe { cu(self.inner.handle, addr, value, flags) })
    }

    pub fn wait_value_64(
        &self,
        addr: baracuda_cuda_sys::CUdeviceptr,
        value: u64,
        flags: u32,
    ) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_stream_wait_value_64()?;
        check(unsafe { cu(self.inner.handle, addr, value, flags) })
    }

    /// Submit a batch of wait/write value ops atomically on this stream.
    /// `ops` is typically a small array built via
    /// [`baracuda_cuda_sys::types::CUstreamBatchMemOpParams::wait_value_32`]
    /// etc.
    pub fn batch_mem_op(
        &self,
        ops: &mut [baracuda_cuda_sys::types::CUstreamBatchMemOpParams],
        flags: u32,
    ) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_stream_batch_mem_op()?;
        check(unsafe {
            cu(
                self.inner.handle,
                ops.len() as core::ffi::c_uint,
                ops.as_mut_ptr(),
                flags,
            )
        })
    }

    /// Query stream-capture state. Returns `(active, capture_id, graph_handle)`
    /// where `active` is `true` if the stream is currently capturing. The
    /// graph handle is only meaningful while capturing.
    pub fn capture_info(&self) -> Result<(bool, u64, baracuda_cuda_sys::CUgraph)> {
        let d = driver()?;
        let cu = d.cu_stream_get_capture_info()?;
        let mut status: core::ffi::c_int = 0;
        let mut id: u64 = 0;
        let mut graph: baracuda_cuda_sys::CUgraph = core::ptr::null_mut();
        let mut deps_ptr: *const baracuda_cuda_sys::CUgraphNode = core::ptr::null();
        let mut num_deps: usize = 0;
        check(unsafe {
            cu(
                self.inner.handle,
                &mut status,
                &mut id,
                &mut graph,
                &mut deps_ptr,
                &mut num_deps,
            )
        })?;
        // CUstreamCaptureStatus: NONE=0, ACTIVE=1, INVALIDATED=2.
        Ok((status == 1, id, graph))
    }
}

impl Drop for StreamInner {
    fn drop(&mut self) {
        if let Ok(d) = driver() {
            if let Ok(cu) = d.cu_stream_destroy() {
                // SAFETY: last Arc drop; handle is unique.
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}
