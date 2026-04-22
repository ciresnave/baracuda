//! Runtime-API device memory.

use core::ffi::c_void;
use core::marker::PhantomData;
use core::mem::size_of;

use baracuda_cuda_sys::runtime::{cudaMemcpyKind, runtime};
use baracuda_types::DeviceRepr;

use crate::error::{check, Result};
use crate::stream::Stream;

/// Owned, typed allocation of device memory (Runtime API).
pub struct DeviceBuffer<T: DeviceRepr> {
    ptr: *mut c_void,
    len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: DeviceRepr + Send> Send for DeviceBuffer<T> {}

impl<T: DeviceRepr> core::fmt::Debug for DeviceBuffer<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .field("type", &core::any::type_name::<T>())
            .finish()
    }
}

impl<T: DeviceRepr> DeviceBuffer<T> {
    /// Allocate an uninitialized buffer of `len` elements on the current device.
    pub fn new(len: usize) -> Result<Self> {
        let r = runtime()?;
        let cu = r.cuda_malloc()?;
        let bytes = len
            .checked_mul(size_of::<T>())
            .expect("overflow computing allocation size");
        let mut ptr: *mut c_void = core::ptr::null_mut();
        check(unsafe { cu(&mut ptr, bytes) })?;
        Ok(Self {
            ptr,
            len,
            _marker: PhantomData,
        })
    }

    /// Allocate and zero-fill.
    pub fn zeros(len: usize) -> Result<Self> {
        let buf = Self::new(len)?;
        let r = runtime()?;
        let cu = r.cuda_memset()?;
        let bytes = len * size_of::<T>();
        check(unsafe { cu(buf.ptr, 0, bytes) })?;
        Ok(buf)
    }

    /// Allocate and synchronously copy `src` from host memory.
    pub fn from_slice(src: &[T]) -> Result<Self> {
        let buf = Self::new(src.len())?;
        buf.copy_from_host(src)?;
        Ok(buf)
    }

    /// Synchronous H2D copy.
    pub fn copy_from_host(&self, src: &[T]) -> Result<()> {
        assert_eq!(src.len(), self.len);
        let r = runtime()?;
        let cu = r.cuda_memcpy()?;
        let bytes = self.len * size_of::<T>();
        check(unsafe {
            cu(
                self.ptr,
                src.as_ptr() as *const c_void,
                bytes,
                cudaMemcpyKind::HostToDevice,
            )
        })
    }

    /// Synchronous D2H copy.
    pub fn copy_to_host(&self, dst: &mut [T]) -> Result<()> {
        assert_eq!(dst.len(), self.len);
        let r = runtime()?;
        let cu = r.cuda_memcpy()?;
        let bytes = self.len * size_of::<T>();
        check(unsafe {
            cu(
                dst.as_mut_ptr() as *mut c_void,
                self.ptr,
                bytes,
                cudaMemcpyKind::DeviceToHost,
            )
        })
    }

    /// Asynchronous H2D copy on `stream`.
    pub fn copy_from_host_async(&self, src: &[T], stream: &Stream) -> Result<()> {
        assert_eq!(src.len(), self.len);
        let r = runtime()?;
        let cu = r.cuda_memcpy_async()?;
        let bytes = self.len * size_of::<T>();
        check(unsafe {
            cu(
                self.ptr,
                src.as_ptr() as *const c_void,
                bytes,
                cudaMemcpyKind::HostToDevice,
                stream.as_raw(),
            )
        })
    }

    /// Asynchronous D2H copy on `stream`.
    pub fn copy_to_host_async(&self, dst: &mut [T], stream: &Stream) -> Result<()> {
        assert_eq!(dst.len(), self.len);
        let r = runtime()?;
        let cu = r.cuda_memcpy_async()?;
        let bytes = self.len * size_of::<T>();
        check(unsafe {
            cu(
                dst.as_mut_ptr() as *mut c_void,
                self.ptr,
                bytes,
                cudaMemcpyKind::DeviceToHost,
                stream.as_raw(),
            )
        })
    }

    /// Number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Size in bytes.
    #[inline]
    pub fn byte_size(&self) -> usize {
        self.len * size_of::<T>()
    }

    /// `true` if zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw device pointer. Use with care.
    #[inline]
    pub fn as_raw(&self) -> *mut c_void {
        self.ptr
    }

    /// Raw device pointer as the u64 value kernels expect. Convenience
    /// wrapper around [`as_raw`](Self::as_raw).
    #[inline]
    pub fn as_device_ptr(&self) -> u64 {
        self.ptr as u64
    }
}

impl<T: DeviceRepr> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        if let Ok(r) = runtime() {
            if let Ok(cu) = r.cuda_free() {
                let _ = unsafe { cu(self.ptr) };
            }
        }
    }
}

// ---- Mem info / prefetch / advise ----------------------------------------

/// `cudaMemGetInfo` — `(free, total)` bytes on the current device.
pub fn mem_get_info() -> Result<(u64, u64)> {
    let r = runtime()?;
    let cu = r.cuda_mem_get_info()?;
    let mut free: usize = 0;
    let mut total: usize = 0;
    check(unsafe { cu(&mut free, &mut total) })?;
    Ok((free as u64, total as u64))
}

/// Target for [`mem_prefetch_async`] / [`mem_advise`]. The CUDA Runtime
/// API's v1 variants take an ordinal — pass `cudaCpuDeviceId` (-1) for host.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PrefetchTarget {
    /// Prefetch to a specific CUDA device (by ordinal).
    Device(i32),
    /// Prefetch to the host CPU.
    Host,
}

impl PrefetchTarget {
    #[inline]
    fn as_raw(self) -> i32 {
        match self {
            PrefetchTarget::Device(i) => i,
            PrefetchTarget::Host => -1, // cudaCpuDeviceId
        }
    }
}

/// Prefetch `count` bytes of unified memory at `dev_ptr` to `target`,
/// ordered on `stream`. `dev_ptr` must be a managed-memory allocation
/// (from [`ManagedBuffer`] or `cudaMallocManaged`).
///
/// # Safety
///
/// `dev_ptr..dev_ptr+count` must be a live managed allocation.
pub unsafe fn mem_prefetch_async(
    dev_ptr: *const core::ffi::c_void,
    count: usize,
    target: PrefetchTarget,
    stream: &Stream,
) -> Result<()> {
    let r = runtime()?;
    let cu = r.cuda_mem_prefetch_async()?;
    check(cu(dev_ptr, count, target.as_raw(), stream.as_raw()))
}

/// `cudaMemAdvise` — unified-memory placement hint. `advice` is a
/// constant from [`baracuda_cuda_sys::runtime::types::cudaMemoryAdvise`].
///
/// # Safety
///
/// `dev_ptr..dev_ptr+count` must be a live managed allocation.
pub unsafe fn mem_advise(
    dev_ptr: *const core::ffi::c_void,
    count: usize,
    advice: i32,
    target: PrefetchTarget,
) -> Result<()> {
    let r = runtime()?;
    let cu = r.cuda_mem_advise()?;
    check(cu(dev_ptr, count, advice, target.as_raw()))
}

// ---- Managed memory -------------------------------------------------------

/// Unified managed-memory buffer — allocated via `cudaMallocManaged`.
/// Accessible from both host and device without explicit copies.
pub struct ManagedBuffer<T: DeviceRepr> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: DeviceRepr + Send> Send for ManagedBuffer<T> {}
unsafe impl<T: DeviceRepr + Sync> Sync for ManagedBuffer<T> {}

impl<T: DeviceRepr> core::fmt::Debug for ManagedBuffer<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ManagedBuffer")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .field("type", &core::any::type_name::<T>())
            .finish()
    }
}

impl<T: DeviceRepr> ManagedBuffer<T> {
    /// Allocate `len` managed elements with the default attach (`GLOBAL`).
    pub fn new(len: usize) -> Result<Self> {
        use baracuda_cuda_sys::runtime::types::cudaMemAttach;
        Self::with_flags(len, cudaMemAttach::GLOBAL)
    }

    /// Allocate with explicit attach flags (see
    /// [`baracuda_cuda_sys::runtime::types::cudaMemAttach`]).
    pub fn with_flags(len: usize, flags: u32) -> Result<Self> {
        let r = runtime()?;
        let cu = r.cuda_malloc_managed()?;
        let bytes = len
            .checked_mul(size_of::<T>())
            .expect("overflow computing allocation size");
        let mut ptr: *mut c_void = core::ptr::null_mut();
        check(unsafe { cu(&mut ptr, bytes, flags) })?;
        Ok(Self {
            ptr: ptr as *mut T,
            len,
            _marker: PhantomData,
        })
    }

    /// Number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw pointer — usable from both host and device code.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Access as a host slice (synchronizes through device cache on access).
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: ptr is live for len elements; managed memory is
        // host-accessible on supported platforms.
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T: DeviceRepr> Drop for ManagedBuffer<T> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        if let Ok(r) = runtime() {
            if let Ok(cu) = r.cuda_free() {
                let _ = unsafe { cu(self.ptr as *mut c_void) };
            }
        }
    }
}

// ---- Pinned host memory --------------------------------------------------

/// Flags for `cudaHostAlloc`. See
/// [`baracuda_cuda_sys::runtime::types::cudaHostAllocFlags`] for raw values.
pub mod pinned_flags {
    pub use baracuda_cuda_sys::runtime::types::cudaHostAllocFlags::*;
}

/// Pinned (page-locked) host allocation — CUDA-owned memory that supports
/// real async H↔D copies without staging.
pub struct PinnedHostBuffer<T: DeviceRepr> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: DeviceRepr + Send> Send for PinnedHostBuffer<T> {}
unsafe impl<T: DeviceRepr + Sync> Sync for PinnedHostBuffer<T> {}

impl<T: DeviceRepr> core::fmt::Debug for PinnedHostBuffer<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PinnedHostBuffer")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .finish()
    }
}

impl<T: DeviceRepr> PinnedHostBuffer<T> {
    /// Allocate `len` pinned elements with default flags.
    pub fn new(len: usize) -> Result<Self> {
        Self::with_flags(len, 0)
    }

    /// Allocate with `cudaHostAllocFlags` bitmask.
    pub fn with_flags(len: usize, flags: u32) -> Result<Self> {
        let r = runtime()?;
        let cu = r.cuda_host_alloc()?;
        let bytes = len
            .checked_mul(size_of::<T>())
            .expect("overflow computing allocation size");
        let mut ptr: *mut c_void = core::ptr::null_mut();
        check(unsafe { cu(&mut ptr, bytes, flags) })?;
        Ok(Self {
            ptr: ptr as *mut T,
            len,
            _marker: PhantomData,
        })
    }

    /// Device-side pointer that aliases this pinned region (requires
    /// `MAPPED` flag at alloc time).
    pub fn device_ptr(&self) -> Result<*mut c_void> {
        let r = runtime()?;
        let cu = r.cuda_host_get_device_pointer()?;
        let mut dev: *mut c_void = core::ptr::null_mut();
        check(unsafe { cu(&mut dev, self.ptr as *mut c_void, 0) })?;
        Ok(dev)
    }

    /// Query the flags this buffer was created with.
    pub fn flags(&self) -> Result<u32> {
        let r = runtime()?;
        let cu = r.cuda_host_get_flags()?;
        let mut f: core::ffi::c_uint = 0;
        check(unsafe { cu(&mut f, self.ptr as *mut c_void) })?;
        Ok(f)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

impl<T: DeviceRepr> core::ops::Deref for PinnedHostBuffer<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T: DeviceRepr> core::ops::DerefMut for PinnedHostBuffer<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T: DeviceRepr> Drop for PinnedHostBuffer<T> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        if let Ok(r) = runtime() {
            if let Ok(cu) = r.cuda_free_host() {
                let _ = unsafe { cu(self.ptr as *mut c_void) };
            }
        }
    }
}

/// RAII guard for `cudaHostRegister` — pins an existing host slice and
/// unregisters on drop.
pub struct PinnedRegistration<'a, T: DeviceRepr> {
    ptr: *mut T,
    len: usize,
    _borrow: PhantomData<&'a mut [T]>,
}

unsafe impl<T: DeviceRepr + Send> Send for PinnedRegistration<'_, T> {}

impl<T: DeviceRepr> core::fmt::Debug for PinnedRegistration<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PinnedRegistration")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .finish()
    }
}

impl<'a, T: DeviceRepr> PinnedRegistration<'a, T> {
    /// Pin `slice` with `flags = 0` until the guard drops.
    pub fn register(slice: &'a mut [T]) -> Result<Self> {
        Self::register_with_flags(slice, 0)
    }

    pub fn register_with_flags(slice: &'a mut [T], flags: u32) -> Result<Self> {
        let r = runtime()?;
        let cu = r.cuda_host_register()?;
        check(unsafe {
            cu(
                slice.as_mut_ptr() as *mut c_void,
                core::mem::size_of_val(slice),
                flags,
            )
        })?;
        Ok(Self {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
            _borrow: PhantomData,
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T: DeviceRepr> Drop for PinnedRegistration<'_, T> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        if let Ok(r) = runtime() {
            if let Ok(cu) = r.cuda_host_unregister() {
                let _ = unsafe { cu(self.ptr as *mut c_void) };
            }
        }
    }
}

// ---- Async alloc / free --------------------------------------------------

impl<T: DeviceRepr> DeviceBuffer<T> {
    /// Asynchronously allocate `len` elements on `stream` from the device's
    /// default memory pool (CUDA 11.2+).
    pub fn new_async(len: usize, stream: &Stream) -> Result<Self> {
        let r = runtime()?;
        let cu = r.cuda_malloc_async()?;
        let bytes = len
            .checked_mul(size_of::<T>())
            .expect("overflow computing allocation size");
        let mut ptr: *mut c_void = core::ptr::null_mut();
        check(unsafe { cu(&mut ptr, bytes, stream.as_raw()) })?;
        Ok(Self {
            ptr,
            len,
            _marker: PhantomData,
        })
    }

    /// Free this buffer asynchronously on `stream`. Consumes `self` so
    /// the sync `Drop` does not also free.
    pub fn free_async(mut self, stream: &Stream) -> Result<()> {
        let ptr = core::mem::replace(&mut self.ptr, core::ptr::null_mut());
        if ptr.is_null() {
            return Ok(());
        }
        let r = runtime()?;
        let cu = r.cuda_free_async()?;
        check(unsafe { cu(ptr, stream.as_raw()) })
    }

    /// Asynchronous memset of `self` to byte value `value` on `stream`.
    pub fn memset_async(&self, value: u8, stream: &Stream) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_memset_async()?;
        let bytes = self.len * size_of::<T>();
        check(unsafe { cu(self.ptr, value as core::ffi::c_int, bytes, stream.as_raw()) })
    }
}

// ---- Peer memcpy ---------------------------------------------------------

/// Peer-to-peer device memory copy. Both buffers must be on enabled-peer
/// devices (see [`crate::Device::enable_peer_access`]).
pub fn memcpy_peer<T: DeviceRepr>(
    dst: &DeviceBuffer<T>,
    dst_device: &crate::Device,
    src: &DeviceBuffer<T>,
    src_device: &crate::Device,
) -> Result<()> {
    assert_eq!(dst.len(), src.len());
    let r = runtime()?;
    let cu = r.cuda_memcpy_peer()?;
    let bytes = src.len() * size_of::<T>();
    check(unsafe {
        cu(
            dst.as_raw(),
            dst_device.ordinal(),
            src.as_raw(),
            src_device.ordinal(),
            bytes,
        )
    })
}

/// Async peer-to-peer memcpy ordered on `stream`.
pub fn memcpy_peer_async<T: DeviceRepr>(
    dst: &DeviceBuffer<T>,
    dst_device: &crate::Device,
    src: &DeviceBuffer<T>,
    src_device: &crate::Device,
    stream: &Stream,
) -> Result<()> {
    assert_eq!(dst.len(), src.len());
    let r = runtime()?;
    let cu = r.cuda_memcpy_peer_async()?;
    let bytes = src.len() * size_of::<T>();
    check(unsafe {
        cu(
            dst.as_raw(),
            dst_device.ordinal(),
            src.as_raw(),
            src_device.ordinal(),
            bytes,
            stream.as_raw(),
        )
    })
}
