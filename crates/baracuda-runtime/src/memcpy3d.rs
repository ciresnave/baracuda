//! 3D memcpy + `cudaMalloc3D` pitched 3D buffers.

use core::ffi::c_void;

use baracuda_cuda_sys::runtime::runtime;
use baracuda_cuda_sys::runtime::types::{cudaExtent, cudaMemcpy3DParms, cudaPitchedPtr};
use baracuda_types::DeviceRepr;

use crate::error::{check, Result};
use crate::stream::Stream;

/// A pitched 3D device allocation (from `cudaMalloc3D`). Freed on drop.
pub struct Pitched3dBuffer<T: DeviceRepr> {
    ptr: cudaPitchedPtr,
    extent: cudaExtent,
    _marker: core::marker::PhantomData<T>,
}

impl<T: DeviceRepr> core::fmt::Debug for Pitched3dBuffer<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Pitched3dBuffer")
            .field("ptr", &self.ptr.ptr)
            .field("pitch", &self.ptr.pitch)
            .field("extent", &self.extent)
            .finish()
    }
}

impl<T: DeviceRepr> Pitched3dBuffer<T> {
    /// Allocate a `width × height × depth` box, with `width` in elements
    /// of `T` (the runtime measures in bytes, so we multiply).
    pub fn new(width: usize, height: usize, depth: usize) -> Result<Self> {
        let r = runtime()?;
        let cu = r.cuda_malloc_3d()?;
        let extent = cudaExtent {
            width: width * core::mem::size_of::<T>(),
            height,
            depth,
        };
        let mut pitched = cudaPitchedPtr::default();
        check(unsafe {
            cu(
                &mut pitched as *mut cudaPitchedPtr as *mut c_void,
                &extent as *const cudaExtent as *const c_void,
            )
        })?;
        Ok(Self {
            ptr: pitched,
            extent,
            _marker: core::marker::PhantomData,
        })
    }

    #[inline]
    pub fn as_pitched_ptr(&self) -> cudaPitchedPtr {
        self.ptr
    }

    #[inline]
    pub fn extent(&self) -> cudaExtent {
        self.extent
    }

    /// Pitch in bytes.
    #[inline]
    pub fn pitch_bytes(&self) -> usize {
        self.ptr.pitch
    }
}

impl<T: DeviceRepr> Drop for Pitched3dBuffer<T> {
    fn drop(&mut self) {
        if self.ptr.ptr.is_null() {
            return;
        }
        if let Ok(r) = runtime() {
            if let Ok(cu) = r.cuda_free() {
                let _ = unsafe { cu(self.ptr.ptr) };
            }
        }
    }
}

/// Issue a `cudaMemcpy3D` with the given parameters.
///
/// # Safety
///
/// Every pointer inside `params` (both array handles and pitched ptrs)
/// must be valid for the copy region.
pub unsafe fn memcpy_3d(params: &cudaMemcpy3DParms) -> Result<()> {
    let r = runtime()?;
    let cu = r.cuda_memcpy_3d()?;
    check(cu(params as *const cudaMemcpy3DParms as *const c_void))
}

/// `cudaMemcpy3DAsync`.
///
/// # Safety
///
/// Same as [`memcpy_3d`]; caller owns synchronization.
pub unsafe fn memcpy_3d_async(params: &cudaMemcpy3DParms, stream: &Stream) -> Result<()> {
    let r = runtime()?;
    let cu = r.cuda_memcpy_3d_async()?;
    check(cu(
        params as *const cudaMemcpy3DParms as *const c_void,
        stream.as_raw(),
    ))
}

/// `cudaMemcpy3DPeer`.
///
/// # Safety
///
/// Same as [`memcpy_3d`]. `params` must include `srcDevice` / `dstDevice`.
pub unsafe fn memcpy_3d_peer(params: &cudaMemcpy3DParms) -> Result<()> {
    let r = runtime()?;
    let cu = r.cuda_memcpy_3d_peer()?;
    check(cu(params as *const cudaMemcpy3DParms as *const c_void))
}

/// `cudaMemcpy3DPeerAsync`.
///
/// # Safety
///
/// Same as [`memcpy_3d_peer`].
pub unsafe fn memcpy_3d_peer_async(params: &cudaMemcpy3DParms, stream: &Stream) -> Result<()> {
    let r = runtime()?;
    let cu = r.cuda_memcpy_3d_peer_async()?;
    check(cu(
        params as *const cudaMemcpy3DParms as *const c_void,
        stream.as_raw(),
    ))
}

/// `cudaMemset3D` — fill a 3D region with a byte value.
///
/// # Safety
///
/// `pitched` must cover the `extent` region.
pub unsafe fn memset_3d(pitched: cudaPitchedPtr, value: i32, extent: cudaExtent) -> Result<()> {
    let r = runtime()?;
    let cu = r.cuda_memset_3d()?;
    // Note: the C signature takes cudaPitchedPtr by value, but we pass a
    // pointer for portability (matches how the PFN is typed at sys layer).
    let mut p = pitched;
    check(cu(
        &mut p as *mut cudaPitchedPtr as *mut c_void,
        value,
        &extent as *const cudaExtent as *const c_void,
    ))
}
