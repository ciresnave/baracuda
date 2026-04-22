//! Runtime-API 2-D memory copies + pitched device allocations.
//!
//! Mirrors [`baracuda_driver::memcpy2d`]. `PitchedBuffer<T>` owns a
//! 2-D device allocation with driver-chosen row stride; free functions
//! handle host ↔ pitched-device copies.

use core::ffi::c_void;
use core::mem::size_of;

use baracuda_cuda_sys::runtime::{cudaMemcpyKind, runtime};
use baracuda_types::DeviceRepr;

use crate::error::{check, Result};
use crate::stream::Stream;

/// A pitched device allocation — `height × width_elems` grid of `T`s
/// with driver-chosen `pitch_bytes ≥ width_elems * size_of::<T>()`.
pub struct PitchedBuffer<T: DeviceRepr> {
    ptr: *mut c_void,
    pitch_bytes: usize,
    width_elems: usize,
    height: usize,
    _marker: core::marker::PhantomData<T>,
}

unsafe impl<T: DeviceRepr + Send> Send for PitchedBuffer<T> {}

impl<T: DeviceRepr> core::fmt::Debug for PitchedBuffer<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PitchedBuffer")
            .field("ptr", &self.ptr)
            .field("width_elems", &self.width_elems)
            .field("height", &self.height)
            .field("pitch_bytes", &self.pitch_bytes)
            .field("type", &core::any::type_name::<T>())
            .finish()
    }
}

impl<T: DeviceRepr> PitchedBuffer<T> {
    /// Allocate a `height × width_elems` grid with driver-chosen pitch.
    pub fn new(width_elems: usize, height: usize) -> Result<Self> {
        let r = runtime()?;
        let cu = r.cuda_malloc_pitch()?;
        let width_bytes = width_elems
            .checked_mul(size_of::<T>())
            .expect("overflow in 2D allocation width");
        let mut ptr: *mut c_void = core::ptr::null_mut();
        let mut pitch: usize = 0;
        check(unsafe { cu(&mut ptr, &mut pitch, width_bytes, height) })?;
        Ok(Self {
            ptr,
            pitch_bytes: pitch,
            width_elems,
            height,
            _marker: core::marker::PhantomData,
        })
    }

    #[inline]
    pub fn width_elems(&self) -> usize {
        self.width_elems
    }
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }
    /// Row stride in bytes as chosen by the runtime.
    #[inline]
    pub fn pitch_bytes(&self) -> usize {
        self.pitch_bytes
    }
    #[inline]
    pub fn as_raw(&self) -> *mut c_void {
        self.ptr
    }
}

impl<T: DeviceRepr> Drop for PitchedBuffer<T> {
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

/// Synchronous host → pitched-device 2-D copy. `src` holds `height`
/// rows of `width_elems` `T`s starting `src_host_pitch_bytes` apart.
pub fn copy_h_to_d_2d<T: DeviceRepr>(
    src: &[T],
    src_host_pitch_bytes: usize,
    dst: &PitchedBuffer<T>,
    width_elems: usize,
    height: usize,
) -> Result<()> {
    assert!(width_elems <= dst.width_elems);
    assert!(height <= dst.height);
    let r = runtime()?;
    let cu = r.cuda_memcpy_2d()?;
    check(unsafe {
        cu(
            dst.ptr,
            dst.pitch_bytes,
            src.as_ptr() as *const c_void,
            src_host_pitch_bytes,
            width_elems * size_of::<T>(),
            height,
            cudaMemcpyKind::HostToDevice,
        )
    })
}

/// Synchronous pitched-device → host 2-D copy.
pub fn copy_d_to_h_2d<T: DeviceRepr>(
    src: &PitchedBuffer<T>,
    dst: &mut [T],
    dst_host_pitch_bytes: usize,
    width_elems: usize,
    height: usize,
) -> Result<()> {
    assert!(width_elems <= src.width_elems);
    assert!(height <= src.height);
    let r = runtime()?;
    let cu = r.cuda_memcpy_2d()?;
    check(unsafe {
        cu(
            dst.as_mut_ptr() as *mut c_void,
            dst_host_pitch_bytes,
            src.ptr,
            src.pitch_bytes,
            width_elems * size_of::<T>(),
            height,
            cudaMemcpyKind::DeviceToHost,
        )
    })
}

/// Async variant of [`copy_h_to_d_2d`].
pub fn copy_h_to_d_2d_async<T: DeviceRepr>(
    src: &[T],
    src_host_pitch_bytes: usize,
    dst: &PitchedBuffer<T>,
    width_elems: usize,
    height: usize,
    stream: &Stream,
) -> Result<()> {
    assert!(width_elems <= dst.width_elems);
    assert!(height <= dst.height);
    let r = runtime()?;
    let cu = r.cuda_memcpy_2d_async()?;
    check(unsafe {
        cu(
            dst.ptr,
            dst.pitch_bytes,
            src.as_ptr() as *const c_void,
            src_host_pitch_bytes,
            width_elems * size_of::<T>(),
            height,
            cudaMemcpyKind::HostToDevice,
            stream.as_raw(),
        )
    })
}

/// 2-D memset: fill a pitched region with byte `value`.
pub fn memset_2d<T: DeviceRepr>(
    dst: &PitchedBuffer<T>,
    value: u8,
    width_elems: usize,
    height: usize,
) -> Result<()> {
    let r = runtime()?;
    let cu = r.cuda_memset_2d()?;
    check(unsafe {
        cu(
            dst.ptr,
            dst.pitch_bytes,
            value as core::ffi::c_int,
            width_elems * size_of::<T>(),
            height,
        )
    })
}

/// Async 2-D memset on `stream`.
pub fn memset_2d_async<T: DeviceRepr>(
    dst: &PitchedBuffer<T>,
    value: u8,
    width_elems: usize,
    height: usize,
    stream: &Stream,
) -> Result<()> {
    let r = runtime()?;
    let cu = r.cuda_memset_2d_async()?;
    check(unsafe {
        cu(
            dst.ptr,
            dst.pitch_bytes,
            value as core::ffi::c_int,
            width_elems * size_of::<T>(),
            height,
            stream.as_raw(),
        )
    })
}
