//! Device-memory types.
//!
//! - [`DeviceBuffer<T>`] — owned, typed GPU allocation.
//! - [`DeviceSlice<'_, T>`] / [`DeviceSliceMut<'_, T>`] — non-owning views
//!   into a buffer, with borrow-checker-tracked lifetimes.

use core::ffi::c_void;
use core::marker::PhantomData;
use core::mem::size_of;

use baracuda_cuda_sys::{driver, CUdeviceptr};
use baracuda_types::DeviceRepr;

use crate::context::Context;
use crate::error::{check, Result};
use crate::stream::Stream;

/// Owned, typed allocation of device memory.
///
/// The underlying bytes are freed when the buffer drops. Clone/copy is
/// deliberately *not* implemented — copying `len` bytes of device memory is
/// not free, so baracuda makes the user spell it out as an explicit
/// stream-ordered D2D memcpy.
pub struct DeviceBuffer<T: DeviceRepr> {
    ptr: CUdeviceptr,
    len: usize,
    context: Context,
    _marker: PhantomData<T>,
}

// SAFETY: a device pointer can be moved between threads but concurrent
// mutation requires external synchronization (streams).
unsafe impl<T: DeviceRepr + Send> Send for DeviceBuffer<T> {}

impl<T: DeviceRepr> core::fmt::Debug for DeviceBuffer<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("ptr", &format_args!("{:#x}", self.ptr.0))
            .field("len", &self.len)
            .field("type", &core::any::type_name::<T>())
            .finish()
    }
}

impl<T: DeviceRepr> DeviceBuffer<T> {
    /// Allocate an uninitialized buffer of `len` elements on the given context's device.
    pub fn new(context: &Context, len: usize) -> Result<Self> {
        context.set_current()?;
        let d = driver()?;
        let cu = d.cu_mem_alloc()?;
        let bytes = len
            .checked_mul(size_of::<T>())
            .expect("overflow computing allocation size");
        let mut ptr = CUdeviceptr(0);
        // SAFETY: `ptr` is writable; `bytes` is the requested allocation size.
        // CUDA rejects 0-byte allocations with CUDA_ERROR_INVALID_VALUE, which
        // we surface to the caller.
        check(unsafe { cu(&mut ptr, bytes) })?;
        Ok(Self {
            ptr,
            len,
            context: context.clone(),
            _marker: PhantomData,
        })
    }

    /// Allocate `len` elements **asynchronously** on `stream` using the
    /// device's default memory pool. Requires CUDA 11.2+.
    ///
    /// Unlike [`new`](Self::new), this call doesn't block — the
    /// allocation becomes usable for any subsequent operation on `stream`
    /// in stream order. Use [`free_async`](Self::free_async) to reclaim
    /// on the same stream, or let `Drop` reclaim synchronously.
    pub fn new_async(context: &Context, len: usize, stream: &Stream) -> Result<Self> {
        context.set_current()?;
        let d = driver()?;
        let cu = d.cu_mem_alloc_async()?;
        let bytes = len
            .checked_mul(size_of::<T>())
            .expect("overflow computing allocation size");
        let mut ptr = CUdeviceptr(0);
        // SAFETY: `ptr` writable; `stream` is live.
        check(unsafe { cu(&mut ptr, bytes, stream.as_raw()) })?;
        Ok(Self {
            ptr,
            len,
            context: context.clone(),
            _marker: PhantomData,
        })
    }

    /// Free `self` asynchronously on `stream`. The buffer becomes invalid
    /// stream-ordered-after this call completes on the device. Consumes
    /// `self` so `Drop` does not also try to free.
    ///
    /// Requires CUDA 11.2+.
    pub fn free_async(mut self, stream: &Stream) -> Result<()> {
        let ptr = core::mem::replace(&mut self.ptr, CUdeviceptr(0));
        if ptr.0 == 0 {
            return Ok(());
        }
        let d = driver()?;
        let cu = d.cu_mem_free_async()?;
        check(unsafe { cu(ptr, stream.as_raw()) })
    }

    /// Allocate and fill with zero bytes.
    pub fn zeros(context: &Context, len: usize) -> Result<Self> {
        let buf = Self::new(context, len)?;
        let d = driver()?;
        let cu = d.cu_memset_d8()?;
        let bytes = len * size_of::<T>();
        check(unsafe { cu(buf.ptr, 0, bytes) })?;
        Ok(buf)
    }

    /// Allocate and copy `src` synchronously from host memory.
    pub fn from_slice(context: &Context, src: &[T]) -> Result<Self> {
        let buf = Self::new(context, src.len())?;
        buf.copy_from_host(src)?;
        Ok(buf)
    }

    /// Synchronous H2D copy. `src.len()` must equal `self.len()`.
    pub fn copy_from_host(&self, src: &[T]) -> Result<()> {
        assert_eq!(
            src.len(),
            self.len,
            "copy_from_host: source length {} != buffer length {}",
            src.len(),
            self.len
        );
        let d = driver()?;
        let cu = d.cu_memcpy_htod()?;
        let bytes = self.len * size_of::<T>();
        // SAFETY: `self.ptr` is a valid device pointer for `bytes` bytes;
        // `src.as_ptr()` is valid for reads of `bytes` bytes.
        check(unsafe { cu(self.ptr, src.as_ptr() as *const c_void, bytes) })
    }

    /// Synchronous D2H copy. `dst.len()` must equal `self.len()`.
    pub fn copy_to_host(&self, dst: &mut [T]) -> Result<()> {
        assert_eq!(
            dst.len(),
            self.len,
            "copy_to_host: destination length {} != buffer length {}",
            dst.len(),
            self.len
        );
        let d = driver()?;
        let cu = d.cu_memcpy_dtoh()?;
        let bytes = self.len * size_of::<T>();
        // SAFETY: mirror of `copy_from_host`; `dst` is valid for writes.
        check(unsafe { cu(dst.as_mut_ptr() as *mut c_void, self.ptr, bytes) })
    }

    /// Asynchronous H2D copy on `stream`.
    pub fn copy_from_host_async(&self, src: &[T], stream: &Stream) -> Result<()> {
        assert_eq!(src.len(), self.len);
        let d = driver()?;
        let cu = d.cu_memcpy_htod_async()?;
        let bytes = self.len * size_of::<T>();
        check(unsafe {
            cu(
                self.ptr,
                src.as_ptr() as *const c_void,
                bytes,
                stream.as_raw(),
            )
        })
    }

    /// Asynchronous D2H copy on `stream`.
    pub fn copy_to_host_async(&self, dst: &mut [T], stream: &Stream) -> Result<()> {
        assert_eq!(dst.len(), self.len);
        let d = driver()?;
        let cu = d.cu_memcpy_dtoh_async()?;
        let bytes = self.len * size_of::<T>();
        check(unsafe {
            cu(
                dst.as_mut_ptr() as *mut c_void,
                self.ptr,
                bytes,
                stream.as_raw(),
            )
        })
    }

    /// Device-to-device copy into another buffer of the same length.
    pub fn copy_to_device(&self, dst: &DeviceBuffer<T>) -> Result<()> {
        assert_eq!(dst.len, self.len);
        let d = driver()?;
        let cu = d.cu_memcpy_dtod()?;
        let bytes = self.len * size_of::<T>();
        check(unsafe { cu(dst.ptr, self.ptr, bytes) })
    }

    /// Asynchronous device-to-device copy on `stream`.
    pub fn copy_to_device_async(&self, dst: &DeviceBuffer<T>, stream: &Stream) -> Result<()> {
        assert_eq!(dst.len, self.len);
        let d = driver()?;
        let cu = d.cu_memcpy_dtod_async()?;
        let bytes = self.len * size_of::<T>();
        check(unsafe { cu(dst.ptr, self.ptr, bytes, stream.as_raw()) })
    }

    /// Number of elements in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Size of the buffer in bytes.
    #[inline]
    pub fn byte_size(&self) -> usize {
        self.len * size_of::<T>()
    }

    /// `true` if the buffer has zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The [`Context`] this buffer was allocated in.
    #[inline]
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// Raw device pointer. Use with care — baracuda still owns the allocation.
    #[inline]
    pub fn as_raw(&self) -> CUdeviceptr {
        self.ptr
    }

    /// Borrow the whole buffer as a [`DeviceSlice<'_, T>`].
    #[inline]
    pub fn as_slice(&self) -> DeviceSlice<'_, T> {
        DeviceSlice {
            ptr: self.ptr,
            len: self.len,
            _marker: PhantomData,
        }
    }

    /// Borrow the whole buffer as a [`DeviceSliceMut<'_, T>`].
    #[inline]
    pub fn as_slice_mut(&mut self) -> DeviceSliceMut<'_, T> {
        DeviceSliceMut {
            ptr: self.ptr,
            len: self.len,
            _marker: PhantomData,
        }
    }
}

// ---- Unified (managed) memory --------------------------------------------

/// Attach-mode for [`ManagedBuffer::new_with_flags`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum ManagedAttach {
    /// Accessible from any stream on any device. **Default.**
    #[default]
    Global,
    /// Pinned to the host — accessible from the host, not from the GPU.
    Host,
    /// Accessible only on the stream it was later attached to.
    Single,
}

impl ManagedAttach {
    #[inline]
    fn raw(self) -> u32 {
        use baracuda_cuda_sys::types::CUmemAttach_flags as F;
        match self {
            ManagedAttach::Global => F::GLOBAL,
            ManagedAttach::Host => F::HOST,
            ManagedAttach::Single => F::SINGLE,
        }
    }
}

/// Memory-usage advice for `cuMemAdvise`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MemAdvise {
    SetReadMostly,
    UnsetReadMostly,
    SetPreferredLocation,
    UnsetPreferredLocation,
    SetAccessedBy,
    UnsetAccessedBy,
}

impl MemAdvise {
    #[inline]
    fn raw(self) -> i32 {
        use baracuda_cuda_sys::types::CUmem_advise as A;
        match self {
            MemAdvise::SetReadMostly => A::SET_READ_MOSTLY,
            MemAdvise::UnsetReadMostly => A::UNSET_READ_MOSTLY,
            MemAdvise::SetPreferredLocation => A::SET_PREFERRED_LOCATION,
            MemAdvise::UnsetPreferredLocation => A::UNSET_PREFERRED_LOCATION,
            MemAdvise::SetAccessedBy => A::SET_ACCESSED_BY,
            MemAdvise::UnsetAccessedBy => A::UNSET_ACCESSED_BY,
        }
    }
}

/// Owned allocation of **unified (managed) memory** — a single pointer that
/// is accessible from both the host and the GPU, with on-demand migration
/// handled by the driver. Compare with [`DeviceBuffer`], which is
/// device-only and requires explicit memcpys.
///
/// On a discrete GPU, accessing the buffer from host code after a kernel
/// finishes (and vice versa) requires a stream synchronize in the
/// unified-memory model; [`ManagedBuffer::as_host_slice`] assumes the
/// caller has done that.
pub struct ManagedBuffer<T: DeviceRepr> {
    ptr: CUdeviceptr,
    len: usize,
    context: Context,
    _marker: PhantomData<T>,
}

unsafe impl<T: DeviceRepr + Send> Send for ManagedBuffer<T> {}

impl<T: DeviceRepr> core::fmt::Debug for ManagedBuffer<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ManagedBuffer")
            .field("ptr", &format_args!("{:#x}", self.ptr.0))
            .field("len", &self.len)
            .field("type", &core::any::type_name::<T>())
            .finish()
    }
}

impl<T: DeviceRepr> ManagedBuffer<T> {
    /// Allocate `len` elements of unified memory with the default
    /// ([`ManagedAttach::Global`]) attach mode.
    pub fn new(context: &Context, len: usize) -> Result<Self> {
        Self::new_with_flags(context, len, ManagedAttach::Global)
    }

    /// Allocate with an explicit [`ManagedAttach`] mode.
    pub fn new_with_flags(context: &Context, len: usize, attach: ManagedAttach) -> Result<Self> {
        context.set_current()?;
        let d = driver()?;
        let cu = d.cu_mem_alloc_managed()?;
        let bytes = len
            .checked_mul(size_of::<T>())
            .expect("overflow computing allocation size");
        let mut ptr = CUdeviceptr(0);
        // SAFETY: writable out-pointer; positive byte count.
        check(unsafe { cu(&mut ptr, bytes, attach.raw()) })?;
        Ok(Self {
            ptr,
            len,
            context: context.clone(),
            _marker: PhantomData,
        })
    }

    /// Provide a hint to the Unified-Memory subsystem about how this range
    /// will be accessed. `device` is the ordinal this advice targets (e.g.
    /// the compute device for `SET_ACCESSED_BY`); pass the current device's
    /// ordinal when in doubt.
    pub fn advise(&self, advice: MemAdvise, device: &crate::Device) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_mem_advise()?;
        let bytes = self.len * size_of::<T>();
        check(unsafe { cu(self.ptr, bytes, advice.raw(), device.as_raw()) })
    }

    /// Asynchronously prefetch this range to `device` on `stream`.
    pub fn prefetch_async(&self, device: &crate::Device, stream: &Stream) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_mem_prefetch_async()?;
        let bytes = self.len * size_of::<T>();
        check(unsafe { cu(self.ptr, bytes, device.as_raw(), stream.as_raw()) })
    }

    /// Access the buffer as a host slice. Safe to call on integrated GPUs or
    /// after a synchronize on discrete GPUs; otherwise you'll see stale data.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// 1. No concurrent kernel is writing to this buffer.
    /// 2. On discrete GPUs, a relevant synchronize has been issued since
    ///    the last device-side write.
    pub unsafe fn as_host_slice(&self) -> &[T] {
        core::slice::from_raw_parts(self.ptr.0 as *const T, self.len)
    }

    /// Mutable host view. Same safety rules as [`as_host_slice`](Self::as_host_slice).
    ///
    /// # Safety
    ///
    /// The caller must ensure no concurrent device or host access.
    pub unsafe fn as_host_slice_mut(&mut self) -> &mut [T] {
        core::slice::from_raw_parts_mut(self.ptr.0 as *mut T, self.len)
    }

    /// Number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` if zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw device pointer — the same value as the host pointer under UVM.
    #[inline]
    pub fn as_raw(&self) -> CUdeviceptr {
        self.ptr
    }

    /// Owning context.
    #[inline]
    pub fn context(&self) -> &Context {
        &self.context
    }
}

impl<T: DeviceRepr> Drop for ManagedBuffer<T> {
    fn drop(&mut self) {
        if self.ptr.0 == 0 {
            return;
        }
        if let Ok(d) = driver() {
            if let Ok(cu) = d.cu_mem_free() {
                let _ = unsafe { cu(self.ptr) };
            }
        }
    }
}

/// Current device's free and total global memory, in bytes.
///
/// Requires a CUDA context to be current on the calling thread.
pub fn mem_get_info() -> Result<(u64, u64)> {
    let d = driver()?;
    let cu = d.cu_mem_get_info()?;
    let mut free: usize = 0;
    let mut total: usize = 0;
    check(unsafe { cu(&mut free, &mut total) })?;
    Ok((free as u64, total as u64))
}

/// Peer-to-peer device memory copy between two contexts — the pointers
/// must be valid device pointers in their respective contexts, and peer
/// access must be enabled (see [`Context::enable_peer_access`]).
pub fn memcpy_peer<T: DeviceRepr>(
    dst: &DeviceBuffer<T>,
    dst_ctx: &Context,
    src: &DeviceBuffer<T>,
    src_ctx: &Context,
) -> Result<()> {
    assert_eq!(dst.len(), src.len());
    let d = driver()?;
    let cu = d.cu_memcpy_peer()?;
    let bytes = src.len() * size_of::<T>();
    check(unsafe {
        cu(
            dst.as_raw(),
            dst_ctx.as_raw(),
            src.as_raw(),
            src_ctx.as_raw(),
            bytes,
        )
    })
}

/// Async peer-to-peer device memory copy ordered on `stream`.
pub fn memcpy_peer_async<T: DeviceRepr>(
    dst: &DeviceBuffer<T>,
    dst_ctx: &Context,
    src: &DeviceBuffer<T>,
    src_ctx: &Context,
    stream: &Stream,
) -> Result<()> {
    assert_eq!(dst.len(), src.len());
    let d = driver()?;
    let cu = d.cu_memcpy_peer_async()?;
    let bytes = src.len() * size_of::<T>();
    check(unsafe {
        cu(
            dst.as_raw(),
            dst_ctx.as_raw(),
            src.as_raw(),
            src_ctx.as_raw(),
            bytes,
            stream.as_raw(),
        )
    })
}

/// Fill `count` 16-bit elements at `dst` with `value` (synchronous).
pub fn memset_u16(dst: CUdeviceptr, value: u16, count: usize) -> Result<()> {
    let d = driver()?;
    let cu = d.cu_memset_d16()?;
    check(unsafe { cu(dst, value, count) })
}

/// Async variant of [`memset_u16`] ordered on `stream`.
pub fn memset_u16_async(dst: CUdeviceptr, value: u16, count: usize, stream: &Stream) -> Result<()> {
    let d = driver()?;
    let cu = d.cu_memset_d16_async()?;
    check(unsafe { cu(dst, value, count, stream.as_raw()) })
}

/// Async 8-bit memset on `stream`.
pub fn memset_u8_async(dst: CUdeviceptr, value: u8, count: usize, stream: &Stream) -> Result<()> {
    let d = driver()?;
    let cu = d.cu_memset_d8_async()?;
    check(unsafe { cu(dst, value, count, stream.as_raw()) })
}

/// Async 32-bit memset on `stream`.
pub fn memset_u32_async(dst: CUdeviceptr, value: u32, count: usize, stream: &Stream) -> Result<()> {
    let d = driver()?;
    let cu = d.cu_memset_d32_async()?;
    check(unsafe { cu(dst, value, count, stream.as_raw()) })
}

// ---- Wave 27: v2 advise/prefetch + VMM reverse lookups ------------------

/// Destination for [`mem_prefetch_v2`] / [`mem_advise_v2`]. Composes the
/// `CUmemLocation::{type_, id}` pair.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PrefetchTarget {
    /// Prefetch to a specific device.
    Device(i32),
    /// Prefetch to the host (unified memory).
    Host,
    /// Prefetch to a specific NUMA node on the host.
    HostNuma(i32),
    /// Prefetch to the current host thread's NUMA node.
    HostNumaCurrent,
}

impl PrefetchTarget {
    fn as_location(self) -> baracuda_cuda_sys::types::CUmemLocation {
        use baracuda_cuda_sys::types::CUmemLocationType;
        let (type_, id) = match self {
            PrefetchTarget::Device(i) => (CUmemLocationType::DEVICE, i),
            PrefetchTarget::Host => (CUmemLocationType::HOST, 0),
            PrefetchTarget::HostNuma(n) => (CUmemLocationType::HOST_NUMA, n),
            PrefetchTarget::HostNumaCurrent => (CUmemLocationType::HOST_NUMA_CURRENT, 0),
        };
        baracuda_cuda_sys::types::CUmemLocation { type_, id }
    }
}

/// `cuMemPrefetchAsync_v2` — prefetch `count` bytes starting at `dptr` to
/// the given [`PrefetchTarget`], ordered on `stream`.
pub fn mem_prefetch_v2(
    dptr: CUdeviceptr,
    count: usize,
    target: PrefetchTarget,
    stream: &Stream,
) -> Result<()> {
    let d = driver()?;
    let cu = d.cu_mem_prefetch_async_v2()?;
    check(unsafe { cu(dptr, count, target.as_location(), 0, stream.as_raw()) })
}

/// `cuMemAdvise_v2` — unified-memory hint at a specific location.
pub fn mem_advise_v2(
    dptr: CUdeviceptr,
    count: usize,
    advice: i32,
    target: PrefetchTarget,
) -> Result<()> {
    let d = driver()?;
    let cu = d.cu_mem_advise_v2()?;
    check(unsafe { cu(dptr, count, advice, target.as_location()) })
}

/// Reverse lookup: given a device pointer inside a VMM mapping, bump the
/// underlying allocation handle's refcount and return it. Pair with
/// `cuMemRelease` to drop the extra ref.
pub fn retain_allocation_handle(
    addr: CUdeviceptr,
) -> Result<baracuda_cuda_sys::CUmemGenericAllocationHandle> {
    let d = driver()?;
    let cu = d.cu_mem_retain_allocation_handle()?;
    let mut h: baracuda_cuda_sys::CUmemGenericAllocationHandle = 0;
    check(unsafe { cu(&mut h, addr.0 as *mut core::ffi::c_void) })?;
    Ok(h)
}

/// Query the creation props of an existing allocation handle.
pub fn allocation_properties_from_handle(
    handle: baracuda_cuda_sys::CUmemGenericAllocationHandle,
) -> Result<baracuda_cuda_sys::types::CUmemAllocationProp> {
    let d = driver()?;
    let cu = d.cu_mem_get_allocation_properties_from_handle()?;
    let mut prop = baracuda_cuda_sys::types::CUmemAllocationProp::default();
    check(unsafe { cu(&mut prop, handle) })?;
    Ok(prop)
}

/// Export an OS-level handle (e.g. a DMA-buf file descriptor on Linux)
/// for a `size`-byte VA range starting at `dptr`.
///
/// # Safety
///
/// `handle_out` must point to a buffer appropriate for the `handle_type`
/// (typically `*mut c_int` for `DMA_BUF_FD`). `dptr..dptr+size` must be a
/// fully mapped VMM region.
pub unsafe fn get_handle_for_address_range(
    handle_out: *mut core::ffi::c_void,
    dptr: CUdeviceptr,
    size: usize,
    handle_type: i32,
) -> Result<()> {
    let d = driver()?;
    let cu = d.cu_mem_get_handle_for_address_range()?;
    check(cu(handle_out, dptr, size, handle_type, 0))
}

impl<T: DeviceRepr> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if self.ptr.0 == 0 {
            return;
        }
        if let Ok(d) = driver() {
            if let Ok(cu) = d.cu_mem_free() {
                let _ = unsafe { cu(self.ptr) };
            }
        }
    }
}

/// Immutable view into a range of a [`DeviceBuffer`].
#[derive(Copy, Clone)]
pub struct DeviceSlice<'a, T: DeviceRepr> {
    pub(crate) ptr: CUdeviceptr,
    pub(crate) len: usize,
    pub(crate) _marker: PhantomData<&'a T>,
}

impl<'a, T: DeviceRepr> core::fmt::Debug for DeviceSlice<'a, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DeviceSlice")
            .field("ptr", &format_args!("{:#x}", self.ptr.0))
            .field("len", &self.len)
            .finish()
    }
}

impl<'a, T: DeviceRepr> DeviceSlice<'a, T> {
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    #[inline]
    pub fn as_raw(&self) -> CUdeviceptr {
        self.ptr
    }
}

/// Mutable view into a range of a [`DeviceBuffer`].
pub struct DeviceSliceMut<'a, T: DeviceRepr> {
    pub(crate) ptr: CUdeviceptr,
    pub(crate) len: usize,
    pub(crate) _marker: PhantomData<&'a mut T>,
}

impl<'a, T: DeviceRepr> core::fmt::Debug for DeviceSliceMut<'a, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DeviceSliceMut")
            .field("ptr", &format_args!("{:#x}", self.ptr.0))
            .field("len", &self.len)
            .finish()
    }
}

impl<'a, T: DeviceRepr> DeviceSliceMut<'a, T> {
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    #[inline]
    pub fn as_raw(&self) -> CUdeviceptr {
        self.ptr
    }
}
