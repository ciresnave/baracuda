//! Device-memory types.
//!
//! - [`DeviceBuffer<T>`] — owned, typed GPU allocation.
//! - [`DeviceSlice<'_, T>`] / [`DeviceSliceMut<'_, T>`] — non-owning views
//!   into a buffer, with borrow-checker-tracked lifetimes.

use core::ffi::c_void;
use core::marker::PhantomData;
use core::mem::size_of;
use core::ops::Range;

use baracuda_cuda_sys::{driver, CUdeviceptr};
use baracuda_types::{DeviceRepr, KernelArg};

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
    ///
    /// `len == 0` (or a zero-sized `T`) short-circuits: CUDA rejects 0-byte
    /// allocations with `CUDA_ERROR_INVALID_VALUE`, so we produce a sentinel
    /// null-pointer buffer. [`Drop`] knows to skip the free on such buffers,
    /// and every copy method below treats `len == 0` as a no-op.
    pub fn new(context: &Context, len: usize) -> Result<Self> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .expect("overflow computing allocation size");
        if bytes == 0 {
            return Ok(Self {
                ptr: CUdeviceptr(0),
                len,
                context: context.clone(),
                _marker: PhantomData,
            });
        }
        context.set_current()?;
        let d = driver()?;
        let cu = d.cu_mem_alloc()?;
        let mut ptr = CUdeviceptr(0);
        // SAFETY: `ptr` is writable; `bytes > 0` so cuMemAlloc is happy.
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
        let bytes = len
            .checked_mul(size_of::<T>())
            .expect("overflow computing allocation size");
        if bytes == 0 {
            return Ok(Self {
                ptr: CUdeviceptr(0),
                len,
                context: context.clone(),
                _marker: PhantomData,
            });
        }
        context.set_current()?;
        let d = driver()?;
        let cu = d.cu_mem_alloc_async()?;
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

    /// Allocate and fill with zero bytes. Zero-length allocations are a
    /// no-op (no `cuMemsetD8` call is issued).
    pub fn zeros(context: &Context, len: usize) -> Result<Self> {
        let buf = Self::new(context, len)?;
        let bytes = len * size_of::<T>();
        if bytes == 0 {
            return Ok(buf);
        }
        let d = driver()?;
        let cu = d.cu_memset_d8()?;
        check(unsafe { cu(buf.ptr, 0, bytes) })?;
        Ok(buf)
    }

    /// Allocate and copy `src` synchronously from host memory. Empty
    /// slices produce a sentinel zero-length buffer (no CUDA calls).
    pub fn from_slice(context: &Context, src: &[T]) -> Result<Self> {
        let buf = Self::new(context, src.len())?;
        buf.copy_from_host(src)?;
        Ok(buf)
    }

    /// Synchronous H2D copy. `src.len()` must equal `self.len()`.
    /// No-op when the buffer is empty — no `cuMemcpy` is issued.
    pub fn copy_from_host(&self, src: &[T]) -> Result<()> {
        assert_eq!(
            src.len(),
            self.len,
            "copy_from_host: source length {} != buffer length {}",
            src.len(),
            self.len
        );
        let bytes = self.len * size_of::<T>();
        if bytes == 0 {
            return Ok(());
        }
        let d = driver()?;
        let cu = d.cu_memcpy_htod()?;
        // SAFETY: `self.ptr` is a valid device pointer for `bytes` bytes;
        // `src.as_ptr()` is valid for reads of `bytes` bytes.
        check(unsafe { cu(self.ptr, src.as_ptr() as *const c_void, bytes) })
    }

    /// Synchronous D2H copy. `dst.len()` must equal `self.len()`.
    /// No-op on empty buffers.
    pub fn copy_to_host(&self, dst: &mut [T]) -> Result<()> {
        assert_eq!(
            dst.len(),
            self.len,
            "copy_to_host: destination length {} != buffer length {}",
            dst.len(),
            self.len
        );
        let bytes = self.len * size_of::<T>();
        if bytes == 0 {
            return Ok(());
        }
        let d = driver()?;
        let cu = d.cu_memcpy_dtoh()?;
        // SAFETY: mirror of `copy_from_host`; `dst` is valid for writes.
        check(unsafe { cu(dst.as_mut_ptr() as *mut c_void, self.ptr, bytes) })
    }

    /// Asynchronous H2D copy on `stream`. No-op on empty buffers.
    pub fn copy_from_host_async(&self, src: &[T], stream: &Stream) -> Result<()> {
        assert_eq!(src.len(), self.len);
        let bytes = self.len * size_of::<T>();
        if bytes == 0 {
            return Ok(());
        }
        let d = driver()?;
        let cu = d.cu_memcpy_htod_async()?;
        check(unsafe {
            cu(
                self.ptr,
                src.as_ptr() as *const c_void,
                bytes,
                stream.as_raw(),
            )
        })
    }

    /// Asynchronous D2H copy on `stream`. No-op on empty buffers.
    pub fn copy_to_host_async(&self, dst: &mut [T], stream: &Stream) -> Result<()> {
        assert_eq!(dst.len(), self.len);
        let bytes = self.len * size_of::<T>();
        if bytes == 0 {
            return Ok(());
        }
        let d = driver()?;
        let cu = d.cu_memcpy_dtoh_async()?;
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
    /// No-op on empty buffers.
    pub fn copy_to_device(&self, dst: &DeviceBuffer<T>) -> Result<()> {
        assert_eq!(dst.len, self.len);
        let bytes = self.len * size_of::<T>();
        if bytes == 0 {
            return Ok(());
        }
        let d = driver()?;
        let cu = d.cu_memcpy_dtod()?;
        check(unsafe { cu(dst.ptr, self.ptr, bytes) })
    }

    /// Asynchronous device-to-device copy on `stream`. No-op on empty buffers.
    pub fn copy_to_device_async(&self, dst: &DeviceBuffer<T>, stream: &Stream) -> Result<()> {
        assert_eq!(dst.len, self.len);
        let bytes = self.len * size_of::<T>();
        if bytes == 0 {
            return Ok(());
        }
        let d = driver()?;
        let cu = d.cu_memcpy_dtod_async()?;
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

    /// Borrow a sub-range of the buffer as an immutable [`DeviceSlice`].
    ///
    /// Panics if the range is out of bounds or inverted. Element indices
    /// are used — the byte offset is `range.start * size_of::<T>()`.
    ///
    /// ```no_run
    /// # use baracuda_driver::{Context, Device, DeviceBuffer};
    /// # fn demo() -> baracuda_driver::Result<()> {
    /// let ctx = Context::new(&Device::get(0)?)?;
    /// let buf: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1024)?;
    /// let first_half = buf.slice(0..512);
    /// let tail = buf.slice(512..1024);
    /// # let _ = (first_half, tail); Ok(()) }
    /// ```
    #[inline]
    pub fn slice(&self, range: Range<usize>) -> DeviceSlice<'_, T> {
        assert!(
            range.start <= range.end && range.end <= self.len,
            "DeviceBuffer::slice({}..{}) out of bounds for len {}",
            range.start,
            range.end,
            self.len,
        );
        DeviceSlice {
            ptr: CUdeviceptr(self.ptr.0 + (range.start * size_of::<T>()) as u64),
            len: range.end - range.start,
            _marker: PhantomData,
        }
    }

    /// Mutable counterpart of [`slice`](Self::slice).
    #[inline]
    pub fn slice_mut(&mut self, range: Range<usize>) -> DeviceSliceMut<'_, T> {
        assert!(
            range.start <= range.end && range.end <= self.len,
            "DeviceBuffer::slice_mut({}..{}) out of bounds for len {}",
            range.start,
            range.end,
            self.len,
        );
        DeviceSliceMut {
            ptr: CUdeviceptr(self.ptr.0 + (range.start * size_of::<T>()) as u64),
            len: range.end - range.start,
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

    /// Borrow a sub-range. Panics on out-of-bounds / inverted ranges.
    #[inline]
    pub fn slice(&self, range: Range<usize>) -> DeviceSlice<'_, T> {
        assert!(
            range.start <= range.end && range.end <= self.len,
            "DeviceSlice::slice({}..{}) out of bounds for len {}",
            range.start,
            range.end,
            self.len,
        );
        DeviceSlice {
            ptr: CUdeviceptr(self.ptr.0 + (range.start * size_of::<T>()) as u64),
            len: range.end - range.start,
            _marker: PhantomData,
        }
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

    /// Borrow a sub-range immutably. Panics on out-of-bounds / inverted.
    #[inline]
    pub fn slice(&self, range: Range<usize>) -> DeviceSlice<'_, T> {
        assert!(
            range.start <= range.end && range.end <= self.len,
            "DeviceSliceMut::slice({}..{}) out of bounds for len {}",
            range.start,
            range.end,
            self.len,
        );
        DeviceSlice {
            ptr: CUdeviceptr(self.ptr.0 + (range.start * size_of::<T>()) as u64),
            len: range.end - range.start,
            _marker: PhantomData,
        }
    }

    /// Borrow a sub-range mutably.
    #[inline]
    pub fn slice_mut(&mut self, range: Range<usize>) -> DeviceSliceMut<'_, T> {
        assert!(
            range.start <= range.end && range.end <= self.len,
            "DeviceSliceMut::slice_mut({}..{}) out of bounds for len {}",
            range.start,
            range.end,
            self.len,
        );
        DeviceSliceMut {
            ptr: CUdeviceptr(self.ptr.0 + (range.start * size_of::<T>()) as u64),
            len: range.end - range.start,
            _marker: PhantomData,
        }
    }
}

// ============================================================================
// DevicePtr / DevicePtrMut — generic device-pointer trait surface
// ============================================================================

/// Anything that can be read as a `[T]` on the device.
///
/// This is the abstraction over [`DeviceBuffer<T>`], [`DeviceSlice<'_, T>`],
/// and (via [`DevicePtrMut`]) [`DeviceSliceMut<'_, T>`] — letting generic code
/// accept any of them without fighting the type system.
///
/// Typical usage:
///
/// ```no_run
/// use baracuda_driver::{DevicePtr, DeviceBuffer, DeviceSlice};
/// use baracuda_types::DeviceRepr;
///
/// fn sum_elements<T: DeviceRepr, P: DevicePtr<T>>(buf: &P) -> usize {
///     // You get len() + device_ptr() for free; deeper ops go through
///     // the concrete type via `buf.device_ptr()`.
///     buf.len()
/// }
/// ```
///
/// # Safety
///
/// `device_ptr()` returns an opaque [`CUdeviceptr`]. Any dereference is
/// `unsafe` as always. Implementors must guarantee the pointer is live for
/// at least `len() * size_of::<T>()` bytes.
pub unsafe trait DevicePtr<T: DeviceRepr> {
    /// Raw device pointer to element 0.
    fn device_ptr(&self) -> CUdeviceptr;

    /// Number of `T` elements visible through this pointer.
    fn len(&self) -> usize;

    /// `true` if [`len`](Self::len) is 0.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Size in bytes (`len * size_of::<T>()`).
    #[inline]
    fn byte_size(&self) -> usize {
        self.len() * core::mem::size_of::<T>()
    }
}

/// A [`DevicePtr`] that supports writes.
///
/// Implementors must hold a unique reference to the underlying storage for
/// the pointer's lifetime — e.g. `&mut DeviceBuffer<T>` or
/// [`DeviceSliceMut<'_, T>`]. This gives the trait the same borrow-checker
/// properties as `&mut [T]`.
pub unsafe trait DevicePtrMut<T: DeviceRepr>: DevicePtr<T> {
    /// Raw mutable device pointer.
    fn device_ptr_mut(&mut self) -> CUdeviceptr;
}

// ---- Impls on the owned + borrowed device types -------------------------

unsafe impl<T: DeviceRepr> DevicePtr<T> for DeviceBuffer<T> {
    #[inline]
    fn device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }
    #[inline]
    fn len(&self) -> usize {
        self.len
    }
}

unsafe impl<T: DeviceRepr> DevicePtrMut<T> for DeviceBuffer<T> {
    #[inline]
    fn device_ptr_mut(&mut self) -> CUdeviceptr {
        self.ptr
    }
}

unsafe impl<'a, T: DeviceRepr> DevicePtr<T> for DeviceSlice<'a, T> {
    #[inline]
    fn device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }
    #[inline]
    fn len(&self) -> usize {
        self.len
    }
}

unsafe impl<'a, T: DeviceRepr> DevicePtr<T> for DeviceSliceMut<'a, T> {
    #[inline]
    fn device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }
    #[inline]
    fn len(&self) -> usize {
        self.len
    }
}

unsafe impl<'a, T: DeviceRepr> DevicePtrMut<T> for DeviceSliceMut<'a, T> {
    #[inline]
    fn device_ptr_mut(&mut self) -> CUdeviceptr {
        self.ptr
    }
}

// References delegate transparently.
unsafe impl<T: DeviceRepr, P: DevicePtr<T> + ?Sized> DevicePtr<T> for &P {
    #[inline]
    fn device_ptr(&self) -> CUdeviceptr {
        (**self).device_ptr()
    }
    #[inline]
    fn len(&self) -> usize {
        (**self).len()
    }
}

unsafe impl<T: DeviceRepr, P: DevicePtr<T> + ?Sized> DevicePtr<T> for &mut P {
    #[inline]
    fn device_ptr(&self) -> CUdeviceptr {
        (**self).device_ptr()
    }
    #[inline]
    fn len(&self) -> usize {
        (**self).len()
    }
}

unsafe impl<T: DeviceRepr, P: DevicePtrMut<T> + ?Sized> DevicePtrMut<T> for &mut P {
    #[inline]
    fn device_ptr_mut(&mut self) -> CUdeviceptr {
        (**self).device_ptr_mut()
    }
}

// ============================================================================
// KernelArg auto-marshalling for DeviceBuffer / DeviceSlice / DeviceSliceMut
// ============================================================================
//
// CUDA kernels receive device buffers as raw `T*` pointers, and
// `cuLaunchKernel` expects an array of `void**` — i.e. each argument slot
// must point to the pointer value. baracuda's DeviceBuffer/DeviceSlice
// already store a `CUdeviceptr` inline, so the safest thing is to return
// a pointer *into* the buffer/slice itself; the returned pointer stays
// valid as long as the `&DeviceBuffer` / `&DeviceSlice` reference does,
// which Rust's borrow checker already enforces for kernel launches.

// SAFETY: `&self.ptr` points to a live `CUdeviceptr` owned by the
// DeviceBuffer; it remains valid for as long as the `&self` borrow does,
// which spans the kernel launch. CUDA reads the pointer value during
// submission and never writes it back through this slot.
unsafe impl<T: DeviceRepr> KernelArg for &DeviceBuffer<T> {
    #[inline]
    fn as_kernel_arg_ptr(&self) -> *mut c_void {
        &self.ptr as *const CUdeviceptr as *mut c_void
    }
}

unsafe impl<T: DeviceRepr> KernelArg for &mut DeviceBuffer<T> {
    #[inline]
    fn as_kernel_arg_ptr(&self) -> *mut c_void {
        &self.ptr as *const CUdeviceptr as *mut c_void
    }
}

unsafe impl<'a, T: DeviceRepr> KernelArg for &DeviceSlice<'a, T> {
    #[inline]
    fn as_kernel_arg_ptr(&self) -> *mut c_void {
        &self.ptr as *const CUdeviceptr as *mut c_void
    }
}

unsafe impl<'a, T: DeviceRepr> KernelArg for &DeviceSliceMut<'a, T> {
    #[inline]
    fn as_kernel_arg_ptr(&self) -> *mut c_void {
        &self.ptr as *const CUdeviceptr as *mut c_void
    }
}

unsafe impl<'a, T: DeviceRepr> KernelArg for &mut DeviceSliceMut<'a, T> {
    #[inline]
    fn as_kernel_arg_ptr(&self) -> *mut c_void {
        &self.ptr as *const CUdeviceptr as *mut c_void
    }
}

#[cfg(test)]
mod slice_tests {
    //! Host-only: verify slice / slice_mut bounds math without a GPU.
    use super::*;

    fn fake_slice<T: DeviceRepr>(ptr: u64, len: usize) -> DeviceSlice<'static, T> {
        DeviceSlice {
            ptr: CUdeviceptr(ptr),
            len,
            _marker: PhantomData,
        }
    }

    #[test]
    fn slice_offsets_ptr_by_element_bytes() {
        let s: DeviceSlice<'_, f32> = fake_slice(0x1000, 16);
        let sub = s.slice(4..12);
        assert_eq!(sub.len(), 8);
        assert_eq!(sub.as_raw().0, 0x1000 + 4 * 4); // 4 elements * 4 bytes/f32
    }

    #[test]
    fn slice_of_slice_stays_correct() {
        let s: DeviceSlice<'_, f64> = fake_slice(0x2000, 100);
        let mid = s.slice(10..90);
        let inner = mid.slice(5..15);
        assert_eq!(inner.len(), 10);
        // 10 + 5 = 15 from start, each f64 = 8 bytes.
        assert_eq!(inner.as_raw().0, 0x2000 + 15 * 8);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn slice_end_past_len_panics() {
        let s: DeviceSlice<'_, u8> = fake_slice(0, 10);
        let _ = s.slice(0..11);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn slice_inverted_range_panics() {
        let s: DeviceSlice<'_, u8> = fake_slice(0, 10);
        let _ = s.slice(5..3);
    }
}

#[cfg(test)]
mod kernel_arg_tests {
    //! Host-only: verify the returned pointer points to the CUdeviceptr
    //! actually stored inside the buffer/slice, so kernels see the right
    //! device address. We don't need a GPU to check this — we fabricate
    //! a DeviceSlice with PhantomData and inspect its bytes.

    use super::*;
    use core::mem::size_of;

    #[test]
    fn slice_kernel_arg_points_at_ptr_field() {
        let slice: DeviceSlice<'_, f32> = DeviceSlice {
            ptr: CUdeviceptr(0xDEAD_BEEF_u64),
            len: 42,
            _marker: PhantomData,
        };
        let kernel_arg = (&slice).as_kernel_arg_ptr();
        // The returned pointer should point to a u64 = 0xDEADBEEF.
        unsafe {
            let as_u64 = *(kernel_arg as *const u64);
            assert_eq!(as_u64, 0xDEAD_BEEF);
        }
        // And the pointer must live inside the slice struct itself.
        let slice_start = &slice as *const _ as usize;
        let slice_end = slice_start + size_of::<DeviceSlice<'_, f32>>();
        let arg_addr = kernel_arg as usize;
        assert!((slice_start..slice_end).contains(&arg_addr));
    }
}
