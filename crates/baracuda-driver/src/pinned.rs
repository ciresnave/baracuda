//! Pinned (page-locked) host memory.
//!
//! Regular Rust allocations are pageable — the kernel can move or swap
//! them out, so a CUDA async memcpy has to stage through the driver's
//! private pinned pool. Allocating the host buffer pinned from the start
//! cuts that staging step, unlocking real HtoD/DtoH/compute overlap.
//!
//! Two flavors:
//!
//! - [`PinnedBuffer<T>`] — pinned allocation owned by CUDA
//!   (`cuMemHostAlloc` + `cuMemFreeHost`).
//! - [`PinnedRegistration`] — pin an existing Rust allocation
//!   (`cuMemHostRegister` + `cuMemHostUnregister`).

use core::ffi::c_void;
use core::mem::size_of;
use core::ops::{Deref, DerefMut};

use baracuda_cuda_sys::{driver, CUdeviceptr};
use baracuda_types::DeviceRepr;

use crate::context::Context;
use crate::error::{check, Result};

/// Flags for [`PinnedBuffer::with_flags`] / [`PinnedRegistration::register_with_flags`].
///
/// `PORTABLE` makes the pinned pages visible to every CUDA context in the
/// process. `MAPPED` maps the allocation into device space so it can be
/// used directly by kernels (zero-copy). `WRITECOMBINED` uses a write-only
/// caching mode that speeds up HtoD at the cost of slow host reads.
#[allow(non_snake_case)]
pub mod flags {
    pub const PORTABLE: u32 = 0x01;
    pub const DEVICEMAP: u32 = 0x02;
    pub const WRITECOMBINED: u32 = 0x04;
}

/// A pinned host allocation owned by CUDA. `Deref`s to `&[T]` / `&mut [T]`.
pub struct PinnedBuffer<T: DeviceRepr> {
    ptr: *mut T,
    len: usize,
    #[allow(dead_code)]
    context: Context,
    _marker: core::marker::PhantomData<T>,
}

unsafe impl<T: DeviceRepr + Send> Send for PinnedBuffer<T> {}
unsafe impl<T: DeviceRepr + Sync> Sync for PinnedBuffer<T> {}

impl<T: DeviceRepr> core::fmt::Debug for PinnedBuffer<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PinnedBuffer")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .field("type", &core::any::type_name::<T>())
            .finish()
    }
}

impl<T: DeviceRepr> PinnedBuffer<T> {
    /// Allocate `len` pinned elements with flags = 0.
    pub fn new(context: &Context, len: usize) -> Result<Self> {
        Self::with_flags(context, len, 0)
    }

    /// Allocate `len` pinned elements, passing `flags` to `cuMemHostAlloc`
    /// (see the [`flags`] module).
    ///
    /// Zero-length allocations (`len == 0` or a zero-sized `T`) short-circuit:
    /// CUDA rejects 0-byte `cuMemHostAlloc` with a null pointer, which would
    /// make [`Deref`]'s `slice::from_raw_parts(null, 0)` unsound. Instead we
    /// use a dangling-but-aligned pointer (the same trick `Vec::new()` uses
    /// internally), skip the CUDA call entirely, and [`Drop`] recognizes
    /// the sentinel and skips `cuMemFreeHost`.
    pub fn with_flags(context: &Context, len: usize, flags: u32) -> Result<Self> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .expect("overflow in PinnedBuffer byte count");
        if bytes == 0 {
            return Ok(Self {
                ptr: core::ptr::NonNull::<T>::dangling().as_ptr(),
                len,
                context: context.clone(),
                _marker: core::marker::PhantomData,
            });
        }
        context.set_current()?;
        let d = driver()?;
        let cu = d.cu_mem_host_alloc()?;
        let mut p: *mut c_void = core::ptr::null_mut();
        check(unsafe { cu(&mut p, bytes, flags) })?;
        Ok(Self {
            ptr: p as *mut T,
            len,
            context: context.clone(),
            _marker: core::marker::PhantomData,
        })
    }

    /// Retrieve the device-visible pointer corresponding to this host
    /// allocation. Requires the buffer was created with the `DEVICEMAP`
    /// flag (or the process has `CU_CTX_MAP_HOST` enabled).
    ///
    /// On an empty buffer, returns `CUdeviceptr(0)` — there's no real
    /// allocation to map. This mirrors [`crate::DeviceBuffer`]'s null
    /// sentinel for zero-length device buffers.
    pub fn device_ptr(&self) -> Result<CUdeviceptr> {
        if self.len == 0 {
            return Ok(CUdeviceptr(0));
        }
        let d = driver()?;
        let cu = d.cu_mem_host_get_device_pointer()?;
        let mut dptr = CUdeviceptr(0);
        check(unsafe { cu(&mut dptr, self.ptr as *mut c_void, 0) })?;
        Ok(dptr)
    }

    /// Query the flags the pinned allocation was created with. On an
    /// empty buffer, returns `0` since there's no real allocation to
    /// query.
    pub fn flags(&self) -> Result<u32> {
        if self.len == 0 {
            return Ok(0);
        }
        let d = driver()?;
        let cu = d.cu_mem_host_get_flags()?;
        let mut flags: core::ffi::c_uint = 0;
        check(unsafe { cu(&mut flags, self.ptr as *mut c_void) })?;
        Ok(flags)
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

impl<T: DeviceRepr> Deref for PinnedBuffer<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: ptr is live for len elements until Drop.
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T: DeviceRepr> DerefMut for PinnedBuffer<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T: DeviceRepr> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        // Zero-length buffers use `NonNull::dangling()` as a sentinel —
        // there's nothing to free. We also guard against a null ptr just
        // in case some future constructor stores one.
        if self.len == 0 || self.ptr.is_null() {
            return;
        }
        if let Ok(d) = driver() {
            if let Ok(cu) = d.cu_mem_free_host() {
                let _ = unsafe { cu(self.ptr as *mut c_void) };
            }
        }
    }
}

/// Pin an existing Rust slice for the lifetime of this guard. Unpin on drop.
///
/// Use this when the host buffer already exists (e.g. from a `Vec<T>`) and
/// you want `cuMemcpy*Async` to fast-path on it.
pub struct PinnedRegistration<'a, T: DeviceRepr> {
    ptr: *mut T,
    len: usize,
    _borrow: core::marker::PhantomData<&'a mut [T]>,
}

unsafe impl<T: DeviceRepr + Send> Send for PinnedRegistration<'_, T> {}
unsafe impl<T: DeviceRepr + Sync> Sync for PinnedRegistration<'_, T> {}

impl<'a, T: DeviceRepr> PinnedRegistration<'a, T> {
    /// Pin `slice` with flags = 0.
    pub fn register(slice: &'a mut [T]) -> Result<Self> {
        Self::register_with_flags(slice, 0)
    }

    pub fn register_with_flags(slice: &'a mut [T], flags: u32) -> Result<Self> {
        let d = driver()?;
        let cu = d.cu_mem_host_register()?;
        let bytes = core::mem::size_of_val(slice);
        check(unsafe { cu(slice.as_mut_ptr() as *mut c_void, bytes, flags) })?;
        Ok(Self {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
            _borrow: core::marker::PhantomData,
        })
    }

    /// Device-side pointer aliasing this pinned region (requires `DEVICEMAP`).
    pub fn device_ptr(&self) -> Result<CUdeviceptr> {
        let d = driver()?;
        let cu = d.cu_mem_host_get_device_pointer()?;
        let mut dptr = CUdeviceptr(0);
        check(unsafe { cu(&mut dptr, self.ptr as *mut c_void, 0) })?;
        Ok(dptr)
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

impl<T: DeviceRepr> core::fmt::Debug for PinnedRegistration<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PinnedRegistration")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .finish()
    }
}

impl<T: DeviceRepr> Drop for PinnedRegistration<'_, T> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        if let Ok(d) = driver() {
            if let Ok(cu) = d.cu_mem_host_unregister() {
                let _ = unsafe { cu(self.ptr as *mut c_void) };
            }
        }
    }
}
