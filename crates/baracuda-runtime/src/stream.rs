//! Runtime-API streams.

use std::sync::Arc;

use baracuda_cuda_sys::runtime::{cudaStream_t, runtime, types::cudaStreamFlags};

use crate::device::Device;
use crate::error::{check, Result};

/// An asynchronous work queue on the current CUDA device.
#[derive(Clone)]
pub struct Stream {
    inner: Arc<StreamInner>,
}

struct StreamInner {
    handle: cudaStream_t,
    device: Device,
}

unsafe impl Send for StreamInner {}
unsafe impl Sync for StreamInner {}

impl core::fmt::Debug for StreamInner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Stream")
            .field("handle", &self.handle)
            .field("device", &self.device)
            .finish()
    }
}

impl core::fmt::Debug for Stream {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.inner.fmt(f)
    }
}

impl Stream {
    /// Create a stream with default (legacy-default-stream-synchronizing) flags
    /// on the current device.
    pub fn new() -> Result<Self> {
        Self::with_flags(cudaStreamFlags::DEFAULT)
    }

    /// Create a non-blocking stream — does not synchronize with the legacy
    /// default stream.
    pub fn non_blocking() -> Result<Self> {
        Self::with_flags(cudaStreamFlags::NON_BLOCKING)
    }

    /// Adopt a raw `cudaStream_t` handle. The wrapper will call
    /// `cudaStreamDestroy` on drop.
    ///
    /// # Safety
    ///
    /// `handle` must be a live stream on the current device. Do not
    /// destroy it externally.
    pub unsafe fn from_raw(handle: cudaStream_t) -> Self {
        let device = Device::current().unwrap_or(Device::from_ordinal(0));
        Self {
            inner: Arc::new(StreamInner { handle, device }),
        }
    }

    /// Create a stream with raw flags (see [`cudaStreamFlags`]).
    pub fn with_flags(flags: u32) -> Result<Self> {
        let r = runtime()?;
        let cu = r.cuda_stream_create_with_flags()?;
        let mut stream: cudaStream_t = core::ptr::null_mut();
        check(unsafe { cu(&mut stream, flags) })?;
        let device = Device::current()?;
        Ok(Self {
            inner: Arc::new(StreamInner {
                handle: stream,
                device,
            }),
        })
    }

    /// Block the calling thread until all prior work on this stream is complete.
    pub fn synchronize(&self) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_stream_synchronize()?;
        check(unsafe { cu(self.inner.handle) })
    }

    /// `Ok(true)` if all queued work has finished, `Ok(false)` if work remains.
    pub fn is_complete(&self) -> Result<bool> {
        use baracuda_cuda_sys::runtime::cudaError_t;
        let r = runtime()?;
        let cu = r.cuda_stream_query()?;
        match unsafe { cu(self.inner.handle) } {
            cudaError_t::Success => Ok(true),
            cudaError_t::NotReady => Ok(false),
            other => Err(crate::error::Error::Status { status: other }),
        }
    }

    /// The device this stream belongs to.
    #[inline]
    pub fn device(&self) -> Device {
        self.inner.device
    }

    /// Raw `cudaStream_t` handle. Use with care.
    #[inline]
    pub fn as_raw(&self) -> cudaStream_t {
        self.inner.handle
    }

    /// Create a stream with a specific scheduling priority (lower = higher
    /// priority). Use [`stream_priority_range`] to discover the legal
    /// range on the current device.
    pub fn with_priority(flags: u32, priority: i32) -> Result<Self> {
        let r = runtime()?;
        let cu = r.cuda_stream_create_with_priority()?;
        let mut stream: cudaStream_t = core::ptr::null_mut();
        check(unsafe { cu(&mut stream, flags, priority) })?;
        let device = Device::current()?;
        Ok(Self {
            inner: Arc::new(StreamInner {
                handle: stream,
                device,
            }),
        })
    }

    /// This stream's scheduling priority.
    pub fn priority(&self) -> Result<i32> {
        let r = runtime()?;
        let cu = r.cuda_stream_get_priority()?;
        let mut p: core::ffi::c_int = 0;
        check(unsafe { cu(self.inner.handle, &mut p) })?;
        Ok(p)
    }

    /// This stream's flags bitmask.
    pub fn flags(&self) -> Result<u32> {
        let r = runtime()?;
        let cu = r.cuda_stream_get_flags()?;
        let mut f: core::ffi::c_uint = 0;
        check(unsafe { cu(self.inner.handle, &mut f) })?;
        Ok(f)
    }

    /// Wait for `event` on this stream — blocks future work on `self`
    /// until the event has completed.
    pub fn wait_event(&self, event: &crate::Event, flags: u32) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_stream_wait_event()?;
        check(unsafe { cu(self.inner.handle, event.as_raw(), flags) })
    }
}

/// Return `(least_priority, greatest_priority)` supported on the current
/// device. Lower numbers are higher priority.
pub fn stream_priority_range() -> Result<(i32, i32)> {
    let r = runtime()?;
    let cu = r.cuda_device_get_stream_priority_range()?;
    let mut low: core::ffi::c_int = 0;
    let mut high: core::ffi::c_int = 0;
    check(unsafe { cu(&mut low, &mut high) })?;
    Ok((low, high))
}

impl Stream {
    /// Enqueue a host-side callback on this stream. Runs on a
    /// driver-owned thread after prior stream work completes.
    ///
    /// The closure is boxed and freed after it runs; a panic inside
    /// aborts the process.
    pub fn launch_host_func<F>(&self, f: F) -> Result<()>
    where
        F: FnOnce() + Send + 'static,
    {
        use core::ffi::c_void;

        let boxed: Box<Box<dyn FnOnce() + Send>> = Box::new(Box::new(f));
        let raw = Box::into_raw(boxed) as *mut c_void;

        unsafe extern "C" fn trampoline(user_data: *mut c_void) {
            let f: Box<Box<dyn FnOnce() + Send>> =
                unsafe { Box::from_raw(user_data as *mut Box<dyn FnOnce() + Send>) };
            (*f)();
        }

        let r = runtime()?;
        let cu = r.cuda_launch_host_func()?;
        let rc = unsafe { cu(self.inner.handle, Some(trampoline), raw) };
        if rc != baracuda_cuda_sys::runtime::cudaError_t::Success {
            // Reclaim the box — cudaLaunchHostFunc didn't take ownership on error.
            drop(unsafe { Box::from_raw(raw as *mut Box<dyn FnOnce() + Send>) });
            return Err(crate::error::Error::Status { status: rc });
        }
        Ok(())
    }

    /// Enqueue a 32-bit write of `value` to device memory `addr`.
    ///
    /// # Safety
    ///
    /// `addr` must be a live device-addressable pointer.
    pub unsafe fn write_value_32(
        &self,
        addr: *mut core::ffi::c_void,
        value: u32,
        flags: u32,
    ) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_stream_write_value_32()?;
        check(cu(self.inner.handle, addr, value, flags))
    }

    /// # Safety
    ///
    /// Same as [`write_value_32`].
    pub unsafe fn write_value_64(
        &self,
        addr: *mut core::ffi::c_void,
        value: u64,
        flags: u32,
    ) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_stream_write_value_64()?;
        check(cu(self.inner.handle, addr, value, flags))
    }

    /// Block the stream until the 32-bit device memory at `addr` satisfies
    /// the condition selected by `flags` (GEQ / EQ / AND / NOR, optionally
    /// OR'd with FLUSH).
    ///
    /// # Safety
    ///
    /// `addr` must be a live device-addressable pointer.
    pub unsafe fn wait_value_32(
        &self,
        addr: *mut core::ffi::c_void,
        value: u32,
        flags: u32,
    ) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_stream_wait_value_32()?;
        check(cu(self.inner.handle, addr, value, flags))
    }

    /// # Safety
    ///
    /// Same as [`wait_value_32`].
    pub unsafe fn wait_value_64(
        &self,
        addr: *mut core::ffi::c_void,
        value: u64,
        flags: u32,
    ) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_stream_wait_value_64()?;
        check(cu(self.inner.handle, addr, value, flags))
    }

    /// Associate a managed-memory region with this stream
    /// (`cudaStreamAttachMemAsync`). Pass `flags = 0` for the default.
    ///
    /// # Safety
    ///
    /// `dev_ptr` must be a managed-memory allocation.
    pub unsafe fn attach_mem_async(
        &self,
        dev_ptr: *mut core::ffi::c_void,
        length: usize,
        flags: u32,
    ) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_stream_attach_mem_async()?;
        check(cu(self.inner.handle, dev_ptr, length, flags))
    }

    /// Copy CUDA-managed attributes (access-policy window, sync policy)
    /// from `src` onto `self`.
    pub fn copy_attributes_from(&self, src: &Stream) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_stream_copy_attributes()?;
        check(unsafe { cu(self.inner.handle, src.inner.handle) })
    }

    /// Enqueue a batch of stream mem-ops (`WAIT_VALUE_32/64`,
    /// `WRITE_VALUE_32/64`) atomically. Much cheaper than issuing the
    /// ops one at a time.
    ///
    /// Build entries with [`baracuda_cuda_sys::types::CUstreamBatchMemOpParams::write_value_32`]
    /// etc. Pass `flags = 0` for the default.
    ///
    /// # Safety
    ///
    /// Every entry's `address` must be a live device-addressable pointer.
    pub unsafe fn batch_mem_op(
        &self,
        params: &mut [baracuda_cuda_sys::types::CUstreamBatchMemOpParams],
        flags: u32,
    ) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_stream_batch_mem_op()?;
        check(cu(
            self.inner.handle,
            params.len() as core::ffi::c_uint,
            params.as_mut_ptr(),
            flags,
        ))
    }
}

impl Drop for StreamInner {
    fn drop(&mut self) {
        if let Ok(r) = runtime() {
            if let Ok(cu) = r.cuda_stream_destroy() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}
