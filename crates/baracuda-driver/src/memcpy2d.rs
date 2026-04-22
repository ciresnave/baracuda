//! Strided (2-D) memory copies via `cuMemcpy2D`, and pitched device
//! allocations via `cuMemAllocPitch`.
//!
//! 2-D memcpy is CUDA's tool for copying rectangular sub-regions of
//! images / matrices / tensors where one or both sides have a **pitch**
//! (row stride) different from the logical row width.

use core::ffi::c_void;
use core::mem::size_of;

use baracuda_cuda_sys::driver;
use baracuda_cuda_sys::types::{CUmemorytype, CUDA_MEMCPY2D};
use baracuda_cuda_sys::CUdeviceptr;
use baracuda_types::DeviceRepr;

use crate::context::Context;
use crate::error::{check, Result};
use crate::stream::Stream;

/// A pitched device allocation — a 2-D `height × width_in_bytes` block
/// where each row is stored at `pitch` bytes apart (`pitch >= width_in_bytes`).
/// Pitch is chosen by the driver to satisfy hardware alignment requirements.
pub struct PitchedBuffer<T: DeviceRepr> {
    ptr: CUdeviceptr,
    pitch_bytes: usize,
    width_elems: usize,
    height: usize,
    context: Context,
    _marker: core::marker::PhantomData<T>,
}

unsafe impl<T: DeviceRepr + Send> Send for PitchedBuffer<T> {}

impl<T: DeviceRepr> core::fmt::Debug for PitchedBuffer<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PitchedBuffer")
            .field("ptr", &format_args!("{:#x}", self.ptr.0))
            .field("width_elems", &self.width_elems)
            .field("height", &self.height)
            .field("pitch_bytes", &self.pitch_bytes)
            .field("type", &core::any::type_name::<T>())
            .finish()
    }
}

impl<T: DeviceRepr> PitchedBuffer<T> {
    /// Allocate a `height × width_elems` grid of `T`s with driver-chosen
    /// pitch. The element size hint steers the alignment.
    pub fn new(context: &Context, width_elems: usize, height: usize) -> Result<Self> {
        context.set_current()?;
        let d = driver()?;
        let cu = d.cu_mem_alloc_pitch()?;
        let mut ptr = CUdeviceptr(0);
        let mut pitch: usize = 0;
        let width_bytes = width_elems
            .checked_mul(size_of::<T>())
            .expect("overflow in 2D allocation width");
        check(unsafe {
            cu(
                &mut ptr,
                &mut pitch,
                width_bytes,
                height,
                size_of::<T>() as core::ffi::c_uint,
            )
        })?;
        Ok(Self {
            ptr,
            pitch_bytes: pitch,
            width_elems,
            height,
            context: context.clone(),
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
    /// Row stride in bytes as chosen by the driver.
    #[inline]
    pub fn pitch_bytes(&self) -> usize {
        self.pitch_bytes
    }
    #[inline]
    pub fn as_raw(&self) -> CUdeviceptr {
        self.ptr
    }
    #[inline]
    pub fn context(&self) -> &Context {
        &self.context
    }
}

impl<T: DeviceRepr> Drop for PitchedBuffer<T> {
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

/// Synchronous `height`-row / `width_elems`-column 2-D copy from a host
/// slice (`src_host_pitch` bytes between row starts) into a pitched
/// device buffer.
///
/// `src` must hold at least `(height - 1) * src_host_pitch + width_elems * size_of::<T>()` bytes.
pub fn copy_h_to_d_2d<T: DeviceRepr>(
    src: &[T],
    src_host_pitch_bytes: usize,
    dst: &PitchedBuffer<T>,
    width_elems: usize,
    height: usize,
) -> Result<()> {
    assert!(width_elems <= dst.width_elems);
    assert!(height <= dst.height);
    let d = driver()?;
    let cu = d.cu_memcpy_2d()?;
    let p = CUDA_MEMCPY2D {
        src_memory_type: CUmemorytype::HOST,
        src_host: src.as_ptr() as *const c_void,
        src_pitch: src_host_pitch_bytes,
        dst_memory_type: CUmemorytype::DEVICE,
        dst_device: dst.ptr,
        dst_pitch: dst.pitch_bytes,
        width_in_bytes: width_elems * size_of::<T>(),
        height,
        ..Default::default()
    };
    check(unsafe { cu(&p) })
}

/// Synchronous 2-D copy from a pitched device buffer back into a host slice.
pub fn copy_d_to_h_2d<T: DeviceRepr>(
    src: &PitchedBuffer<T>,
    dst: &mut [T],
    dst_host_pitch_bytes: usize,
    width_elems: usize,
    height: usize,
) -> Result<()> {
    assert!(width_elems <= src.width_elems);
    assert!(height <= src.height);
    let d = driver()?;
    let cu = d.cu_memcpy_2d()?;
    let p = CUDA_MEMCPY2D {
        src_memory_type: CUmemorytype::DEVICE,
        src_device: src.ptr,
        src_pitch: src.pitch_bytes,
        dst_memory_type: CUmemorytype::HOST,
        dst_host: dst.as_mut_ptr() as *mut c_void,
        dst_pitch: dst_host_pitch_bytes,
        width_in_bytes: width_elems * size_of::<T>(),
        height,
        ..Default::default()
    };
    check(unsafe { cu(&p) })
}

/// Asynchronous variant of [`copy_h_to_d_2d`] — issues on the given stream.
pub fn copy_h_to_d_2d_async<T: DeviceRepr>(
    src: &[T],
    src_host_pitch_bytes: usize,
    dst: &PitchedBuffer<T>,
    width_elems: usize,
    height: usize,
    stream: &Stream,
) -> Result<()> {
    let d = driver()?;
    let cu = d.cu_memcpy_2d_async()?;
    let p = CUDA_MEMCPY2D {
        src_memory_type: CUmemorytype::HOST,
        src_host: src.as_ptr() as *const c_void,
        src_pitch: src_host_pitch_bytes,
        dst_memory_type: CUmemorytype::DEVICE,
        dst_device: dst.ptr,
        dst_pitch: dst.pitch_bytes,
        width_in_bytes: width_elems * size_of::<T>(),
        height,
        ..Default::default()
    };
    check(unsafe { cu(&p, stream.as_raw()) })
}
