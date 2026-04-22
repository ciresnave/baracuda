//! Safe Rust wrappers for NVIDIA cuFile (GPUDirect Storage).
//!
//! cuFile is **Linux-only** and requires a GDS-capable filesystem (ext4
//! / XFS on NVMe with the NVIDIA GDS kernel driver). On Windows and
//! macOS every API returns
//! [`baracuda_core::LoaderError::UnsupportedPlatform`].
//!
//! # Workflow
//!
//! 1. [`Driver::open`] — initialize the driver (do this once per process).
//! 2. [`FileHandle::register`] an open file descriptor.
//! 3. [`BufRegistration::register`] a CUDA device buffer (optional but
//!    strongly recommended for performance).
//! 4. [`FileHandle::read`] / [`FileHandle::write`] directly between the
//!    file and the device buffer — no bounce through host memory.

#![warn(missing_debug_implementations)]

use core::ffi::c_void;

use baracuda_cufile_sys::{cufile, CUfileDescr_t, CUfileError_t, CUfileHandle_t, CUfileOpError};

/// Error type for cuFile operations.
pub type Error = baracuda_core::Error<CUfileOpError>;
/// Result alias.
pub type Result<T, E = Error> = core::result::Result<T, E>;

fn check(rc: CUfileError_t) -> Result<()> {
    if rc.err.0 == 0 {
        Ok(())
    } else {
        Err(Error::Status { status: rc.err })
    }
}

/// Verify cuFile is loadable on this host. Fails on non-Linux platforms.
pub fn probe() -> Result<()> {
    cufile()?;
    Ok(())
}

/// cuFile runtime version as reported by `cuFileGetVersion`.
pub fn version() -> Result<i32> {
    let c = cufile()?;
    let cu = c.cu_file_get_version()?;
    let mut v: core::ffi::c_int = 0;
    check(unsafe { cu(&mut v) })?;
    Ok(v as i32)
}

/// RAII handle for the cuFile driver lifecycle.
///
/// `cuFileDriverOpen` is idempotent inside the library, but baracuda
/// matches every [`Driver::open`] with a `cuFileDriverClose` on drop for
/// symmetry.
#[derive(Debug)]
pub struct Driver {
    _nonsend: core::marker::PhantomData<*const ()>,
}

impl Driver {
    /// Initialize the cuFile driver. Keep the returned handle alive for
    /// the lifetime of any I/O you do.
    pub fn open() -> Result<Self> {
        let c = cufile()?;
        let cu = c.cu_file_driver_open()?;
        check(unsafe { cu() })?;
        Ok(Self {
            _nonsend: core::marker::PhantomData,
        })
    }

    /// Toggle polling vs interrupt-driven I/O; `poll_threshold_size` is
    /// the smallest I/O below which polling is used.
    pub fn set_poll_mode(&self, poll: bool, poll_threshold_size: usize) -> Result<()> {
        let c = cufile()?;
        let cu = c.cu_file_driver_set_poll_mode()?;
        check(unsafe { cu(poll, poll_threshold_size) })
    }

    /// Maximum direct-I/O chunk size in KiB (default 16 MiB).
    pub fn set_max_direct_io_size_kb(&self, size_kb: usize) -> Result<()> {
        let c = cufile()?;
        let cu = c.cu_file_driver_set_max_direct_io_size()?;
        check(unsafe { cu(size_kb) })
    }

    /// Maximum page-cache size cuFile can use, in KiB.
    pub fn set_max_cache_size_kb(&self, size_kb: usize) -> Result<()> {
        let c = cufile()?;
        let cu = c.cu_file_driver_set_max_cache_size()?;
        check(unsafe { cu(size_kb) })
    }

    /// Maximum pinned-host-memory budget cuFile can allocate, in KiB.
    pub fn set_max_pinned_mem_size_kb(&self, size_kb: usize) -> Result<()> {
        let c = cufile()?;
        let cu = c.cu_file_driver_set_max_pinned_mem_size()?;
        check(unsafe { cu(size_kb) })
    }

    /// Fill `props` with the current cuFile driver properties. The struct
    /// layout follows `CUfileDrvProps_t` in the cuFile headers; callers
    /// typically allocate the struct as `std::mem::zeroed::<[u8; 64]>()`
    /// first and then reinterpret the bytes.
    ///
    /// # Safety
    /// `props` must point to at least `sizeof(CUfileDrvProps_t)` bytes.
    pub unsafe fn properties(&self, props: *mut core::ffi::c_void) -> Result<()> {
        let c = cufile()?;
        let cu = c.cu_file_driver_get_properties()?;
        check(cu(props))
    }
}

/// Human-readable string describing a [`CUfileOpError`] code.
pub fn op_status_error_string(status: CUfileOpError) -> Result<String> {
    let c = cufile()?;
    let cu = c.cu_file_op_status_error()?;
    let ptr = unsafe { cu(status) };
    if ptr.is_null() {
        return Ok(String::new());
    }
    let cstr = unsafe { core::ffi::CStr::from_ptr(ptr) };
    Ok(cstr.to_string_lossy().into_owned())
}

impl Drop for Driver {
    fn drop(&mut self) {
        if let Ok(c) = cufile() {
            if let Ok(cu) = c.cu_file_driver_close() {
                let _ = unsafe { cu() };
            }
        }
    }
}

/// A registered file descriptor.
#[derive(Debug)]
pub struct FileHandle {
    handle: CUfileHandle_t,
}

impl FileHandle {
    /// Register an open file descriptor with cuFile. `fd` is typically
    /// obtained from `std::fs::File::as_raw_fd()` on Linux.
    ///
    /// # Safety
    ///
    /// `fd` must stay open for the lifetime of the returned handle.
    pub unsafe fn register(fd: i32) -> Result<Self> {
        let c = cufile()?;
        let cu = c.cu_file_handle_register()?;
        let mut descr = CUfileDescr_t {
            handle_fd: fd,
            ..Default::default()
        };
        let mut h: CUfileHandle_t = core::ptr::null_mut();
        check(cu(&mut h, &mut descr))?;
        Ok(Self { handle: h })
    }

    #[inline]
    pub fn as_raw(&self) -> CUfileHandle_t {
        self.handle
    }

    /// Read `size` bytes from `file_offset` into `dev_buf + buf_offset`.
    /// Returns the number of bytes actually read (negative on failure;
    /// we map negatives to `Err`).
    ///
    /// # Safety
    ///
    /// `dev_buf` must be a device pointer with at least `size` bytes
    /// live starting at `buf_offset`.
    pub unsafe fn read(
        &self,
        dev_buf: *mut c_void,
        size: usize,
        file_offset: i64,
        buf_offset: i64,
    ) -> Result<usize> {
        let c = cufile()?;
        let cu = c.cu_file_read()?;
        let n = cu(self.handle, dev_buf, size, file_offset, buf_offset);
        if n < 0 {
            Err(Error::Status {
                status: CUfileOpError(n as i32),
            })
        } else {
            Ok(n as usize)
        }
    }

    /// Write `size` bytes from `dev_buf + buf_offset` into `file_offset`.
    ///
    /// # Safety
    ///
    /// Same as [`Self::read`].
    pub unsafe fn write(
        &self,
        dev_buf: *const c_void,
        size: usize,
        file_offset: i64,
        buf_offset: i64,
    ) -> Result<usize> {
        let c = cufile()?;
        let cu = c.cu_file_write()?;
        let n = cu(self.handle, dev_buf, size, file_offset, buf_offset);
        if n < 0 {
            Err(Error::Status {
                status: CUfileOpError(n as i32),
            })
        } else {
            Ok(n as usize)
        }
    }
}

impl Drop for FileHandle {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        if let Ok(c) = cufile() {
            if let Ok(cu) = c.cu_file_handle_deregister() {
                unsafe { cu(self.handle) };
            }
        }
    }
}

/// A registered device-buffer region. Registration is optional — cuFile
/// works without it — but it unlocks the fastest DMA path.
#[derive(Debug)]
pub struct BufRegistration {
    ptr: *mut c_void,
    _marker: core::marker::PhantomData<*const ()>,
}

impl BufRegistration {
    /// Register `length` bytes starting at `dev_ptr`. `flags = 0` for
    /// default.
    ///
    /// # Safety
    ///
    /// `dev_ptr` must be a device-memory pointer with `length` live
    /// bytes. Keep it alive for the full registration lifetime.
    pub unsafe fn register(dev_ptr: *mut c_void, length: usize, flags: i32) -> Result<Self> {
        let c = cufile()?;
        let cu = c.cu_file_buf_register()?;
        check(cu(dev_ptr, length, flags))?;
        Ok(Self {
            ptr: dev_ptr,
            _marker: core::marker::PhantomData,
        })
    }
}

impl Drop for BufRegistration {
    fn drop(&mut self) {
        if let Ok(c) = cufile() {
            if let Ok(cu) = c.cu_file_buf_deregister() {
                let _ = unsafe { cu(self.ptr) };
            }
        }
    }
}

// =================== Async I/O (v1.6+) ===================

/// A cuFile-aware CUDA stream. Register a stream once to use it with
/// [`FileHandle::read_async`] / [`FileHandle::write_async`] — cuFile
/// will queue the I/O behind prior stream work and signal completion
/// as another stream op.
#[derive(Debug)]
pub struct StreamRegistration {
    stream: *mut c_void,
}

impl StreamRegistration {
    /// Register `stream` for async I/O. `flags = 0` = default.
    ///
    /// # Safety
    ///
    /// `stream` must be a live `cudaStream_t` on the current context.
    pub unsafe fn register(stream: *mut c_void, flags: u32) -> Result<Self> {
        let c = cufile()?;
        let cu = c.cu_file_stream_register()?;
        check(cu(stream, flags))?;
        Ok(Self { stream })
    }
}

impl Drop for StreamRegistration {
    fn drop(&mut self) {
        if let Ok(c) = cufile() {
            if let Ok(cu) = c.cu_file_stream_deregister() {
                let _ = unsafe { cu(self.stream) };
            }
        }
    }
}

impl FileHandle {
    /// Queue a stream-ordered read. All parameters point at device /
    /// pinned-host memory that cuFile reads *when the stream reaches
    /// this op* (not at call time). `bytes_read` is written when the
    /// op completes.
    ///
    /// # Safety
    ///
    /// All pointers must stay live until the stream reports completion.
    /// `stream` must be previously registered via [`StreamRegistration::register`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn read_async(
        &self,
        dev_buf: *mut c_void,
        size_p: *mut usize,
        file_offset_p: *mut i64,
        buf_offset_p: *mut i64,
        bytes_read: *mut isize,
        stream: *mut c_void,
    ) -> Result<()> {
        let c = cufile()?;
        let cu = c.cu_file_read_async()?;
        check(cu(
            self.handle,
            dev_buf,
            size_p,
            file_offset_p,
            buf_offset_p,
            bytes_read,
            stream,
        ))
    }

    /// Queue a stream-ordered write.
    ///
    /// # Safety
    ///
    /// Same as [`Self::read_async`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn write_async(
        &self,
        dev_buf: *const c_void,
        size_p: *mut usize,
        file_offset_p: *mut i64,
        buf_offset_p: *mut i64,
        bytes_written: *mut isize,
        stream: *mut c_void,
    ) -> Result<()> {
        let c = cufile()?;
        let cu = c.cu_file_write_async()?;
        check(cu(
            self.handle,
            dev_buf,
            size_p,
            file_offset_p,
            buf_offset_p,
            bytes_written,
            stream,
        ))
    }

    /// cuFile's per-handle ref-count (non-zero = handle in use by
    /// outstanding I/O).
    pub fn use_count(&self) -> Result<i32> {
        let c = cufile()?;
        let cu = c.cu_file_use_count()?;
        Ok(unsafe { cu(self.handle) })
    }
}

// =================== Batched I/O (v1.6+) ===================

pub use baracuda_cufile_sys::{
    CUfileBatchHandle_t, CUfileIOEvents_t, CUfileIOParams_t, CUfileOpcode,
};

/// RAII handle for a cuFile batch-I/O request group. Supports up to
/// `capacity` entries per submission cycle.
#[derive(Debug)]
pub struct BatchIO {
    handle: CUfileBatchHandle_t,
}

impl BatchIO {
    /// Create a new batch handle with room for `capacity` concurrent
    /// entries. Typical value: 64–256.
    pub fn new(capacity: u32) -> Result<Self> {
        let c = cufile()?;
        let cu = c.cu_file_batch_io_set_up()?;
        let mut h: CUfileBatchHandle_t = core::ptr::null_mut();
        check(unsafe { cu(&mut h, capacity) })?;
        Ok(Self { handle: h })
    }

    /// Submit `params.len()` entries for execution. Returns before
    /// completion; call [`Self::poll`] to reap.
    ///
    /// # Safety
    ///
    /// Every entry's pointers must stay live until reaped.
    pub unsafe fn submit(&self, params: &mut [CUfileIOParams_t], flags: u32) -> Result<()> {
        let c = cufile()?;
        let cu = c.cu_file_batch_io_submit()?;
        check(cu(
            self.handle,
            params.len() as u32,
            params.as_mut_ptr(),
            flags,
        ))
    }

    /// Wait for at least `min_nr` entries to complete. Fills
    /// `events[..nr]` with outcomes. `timeout_ns = None` blocks
    /// indefinitely.
    ///
    /// # Safety
    ///
    /// `events` is written up to its capacity.
    pub unsafe fn poll(&self, min_nr: u32, events: &mut [CUfileIOEvents_t]) -> Result<u32> {
        let c = cufile()?;
        let cu = c.cu_file_batch_io_get_status()?;
        let mut nr: u32 = events.len() as u32;
        check(cu(
            self.handle,
            min_nr,
            &mut nr,
            events.as_mut_ptr(),
            core::ptr::null_mut(),
        ))?;
        Ok(nr)
    }

    /// Cancel pending entries.
    pub fn cancel(&self) -> Result<()> {
        let c = cufile()?;
        let cu = c.cu_file_batch_io_cancel()?;
        check(unsafe { cu(self.handle) })
    }

    #[inline]
    pub fn as_raw(&self) -> CUfileBatchHandle_t {
        self.handle
    }
}

impl Drop for BatchIO {
    fn drop(&mut self) {
        if let Ok(c) = cufile() {
            if let Ok(cu) = c.cu_file_batch_io_destroy() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}
