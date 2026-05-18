//! Shared ND-pooling helpers (Phase 11.8).
//!
//! Wraps cuDNN's rank-agnostic Nd-pooling entry points
//! ([`cudnnSetPoolingNdDescriptor`], [`cudnnSetTensorNdDescriptor`]) for
//! 1-D and 3-D pooling plans (and the cuDNN approximation of the
//! adaptive-pool family). The two exec entry points themselves —
//! [`cudnnPoolingForward`] and [`cudnnPoolingBackward`] — are rank-
//! agnostic; only the descriptor setup differs from the 2-D path.
//!
//! Layout conventions:
//! - 1-D: `[N, C, L]` (NCL).
//! - 3-D: `[N, C, D, H, W]` (NCDHW).
//!
//! The 2-D plans continue to use the legacy `cudnnSetTensor4dDescriptor`
//! + `cudnnSetPooling2dDescriptor` path — this module is **only** for
//! the rank-3 and rank-5 plans.

use core::cell::Cell;
use core::ffi::c_void;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_sys::{
    cudnnCreate, cudnnCreatePoolingDescriptor, cudnnCreateTensorDescriptor, cudnnDestroy,
    cudnnDestroyPoolingDescriptor, cudnnDestroyTensorDescriptor, cudnnHandle_t,
    cudnnPoolingBackward, cudnnPoolingDescriptor_t, cudnnPoolingForward,
    cudnnSetPoolingNdDescriptor, cudnnSetStream, cudnnSetTensorNdDescriptor,
    cudnnTensorDescriptor_t, CUDNN_NOT_PROPAGATE_NAN,
};
use baracuda_kernels_types::{Element, ElementKind};

use super::max_pool2d::{cudnn_dtype, cudnn_pool_mode, is_double_compute, PoolMode};

/// Spatial-axes-only output-dim formula
/// `out = floor((in + 2·pad - window) / stride) + 1`. Used by 1-D / 3-D
/// pool plans for their per-axis output extents.
#[inline]
pub(crate) fn out_dim(in_dim: i32, pad: i32, window: i32, stride: i32) -> i32 {
    (in_dim + 2 * pad - window) / stride + 1
}

/// Create a cuDNN handle on first call. Idempotent — returns the cached
/// handle on subsequent calls.
pub(crate) fn ensure_handle(handle: &Cell<cudnnHandle_t>) -> Result<cudnnHandle_t> {
    let h = handle.get();
    if !h.is_null() {
        return Ok(h);
    }
    let mut new_h: cudnnHandle_t = core::ptr::null_mut();
    let status = unsafe { cudnnCreate(&mut new_h as *mut _) };
    if status != 0 {
        return Err(Error::CutlassInternal(-status));
    }
    handle.set(new_h);
    Ok(new_h)
}

/// Bind the handle to the caller's CUDA stream on every launch.
pub(crate) fn bind_stream(h: cudnnHandle_t, stream: &Stream) -> Result<()> {
    let status = unsafe { cudnnSetStream(h, stream.as_raw() as *mut c_void) };
    if status != 0 {
        return Err(Error::CutlassInternal(-status));
    }
    Ok(())
}

/// Allocate (once) one cuDNN Nd-tensor descriptor with caller-supplied
/// dims and contiguous-row-major strides.
fn create_tensor_nd<T: Element>(
    desc_cell: &Cell<cudnnTensorDescriptor_t>,
    dims: &[i32],
) -> Result<()> {
    if !desc_cell.get().is_null() {
        return Ok(());
    }
    let mut td: cudnnTensorDescriptor_t = core::ptr::null_mut();
    let status = unsafe { cudnnCreateTensorDescriptor(&mut td as *mut _) };
    if status != 0 {
        return Err(Error::CutlassInternal(-status));
    }
    // cuDNN requires nb_dims >= 4 for `cudnnSetTensorNdDescriptor`. Pad
    // the rank-3 (1-D pool) case to rank-4 by appending W=1.
    let mut padded: [i32; 5] = [1; 5];
    let nb_dims = if dims.len() < 4 { 4 } else { dims.len() };
    for (i, &d) in dims.iter().enumerate() {
        padded[i] = d;
    }
    // Compute contiguous row-major strides over `nb_dims`.
    let mut strides: [i32; 5] = [1; 5];
    let mut acc: i64 = 1;
    let mut i = nb_dims;
    while i > 0 {
        i -= 1;
        strides[i] = acc as i32;
        acc = acc.saturating_mul(padded[i] as i64);
    }
    let dt = cudnn_dtype::<T>();
    let status = unsafe {
        cudnnSetTensorNdDescriptor(
            td,
            dt,
            nb_dims as i32,
            padded.as_ptr(),
            strides.as_ptr(),
        )
    };
    if status != 0 {
        unsafe {
            let _ = cudnnDestroyTensorDescriptor(td);
        }
        return Err(Error::CutlassInternal(-status));
    }
    desc_cell.set(td);
    Ok(())
}

/// Allocate (once) one cuDNN Nd-pooling descriptor with caller-supplied
/// window / pad / stride arrays.
fn create_pool_nd(
    pool_desc: &Cell<cudnnPoolingDescriptor_t>,
    mode: PoolMode,
    window: &[i32],
    padding: &[i32],
    stride: &[i32],
) -> Result<()> {
    if !pool_desc.get().is_null() {
        return Ok(());
    }
    let mut pd: cudnnPoolingDescriptor_t = core::ptr::null_mut();
    let status = unsafe { cudnnCreatePoolingDescriptor(&mut pd as *mut _) };
    if status != 0 {
        return Err(Error::CutlassInternal(-status));
    }
    // cuDNN requires nb_dims >= 2 for `cudnnSetPoolingNdDescriptor`. Pad
    // the rank-1 (1-D pool) case by appending a W=1 / pad=0 / stride=1
    // axis.
    let mut win: [i32; 3] = [1; 3];
    let mut pad: [i32; 3] = [0; 3];
    let mut str_: [i32; 3] = [1; 3];
    for (i, (&w, (&p, &s))) in window
        .iter()
        .zip(padding.iter().zip(stride.iter()))
        .enumerate()
    {
        win[i] = w;
        pad[i] = p;
        str_[i] = s;
    }
    let nb_dims = if window.len() < 2 { 2 } else { window.len() };
    let status = unsafe {
        cudnnSetPoolingNdDescriptor(
            pd,
            cudnn_pool_mode(mode),
            CUDNN_NOT_PROPAGATE_NAN,
            nb_dims as i32,
            win.as_ptr(),
            pad.as_ptr(),
            str_.as_ptr(),
        )
    };
    if status != 0 {
        unsafe {
            let _ = cudnnDestroyPoolingDescriptor(pd);
        }
        return Err(Error::CutlassInternal(-status));
    }
    pool_desc.set(pd);
    Ok(())
}

/// Build the (x_desc, y_desc, pool_desc) trio for an Nd pool plan.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ensure_descriptors_nd<T: Element>(
    x_dims: &[i32],
    y_dims: &[i32],
    window: &[i32],
    padding: &[i32],
    stride: &[i32],
    mode: PoolMode,
    x_desc: &Cell<cudnnTensorDescriptor_t>,
    y_desc: &Cell<cudnnTensorDescriptor_t>,
    pool_desc: &Cell<cudnnPoolingDescriptor_t>,
) -> Result<()> {
    create_tensor_nd::<T>(x_desc, x_dims)?;
    create_tensor_nd::<T>(y_desc, y_dims)?;
    create_pool_nd(pool_desc, mode, window, padding, stride)
}

/// Mirror of [`super::max_pool2d::drop_pool_descriptors`] for the Nd
/// path. Same lifecycle.
pub(crate) fn drop_descriptors_nd(
    x_desc: &Cell<cudnnTensorDescriptor_t>,
    y_desc: &Cell<cudnnTensorDescriptor_t>,
    pool_desc: &Cell<cudnnPoolingDescriptor_t>,
    handle: &Cell<cudnnHandle_t>,
) {
    let pd = pool_desc.get();
    if !pd.is_null() {
        unsafe {
            let _ = cudnnDestroyPoolingDescriptor(pd);
        }
        pool_desc.set(core::ptr::null_mut());
    }
    let yd = y_desc.get();
    if !yd.is_null() {
        unsafe {
            let _ = cudnnDestroyTensorDescriptor(yd);
        }
        y_desc.set(core::ptr::null_mut());
    }
    let xd = x_desc.get();
    if !xd.is_null() {
        unsafe {
            let _ = cudnnDestroyTensorDescriptor(xd);
        }
        x_desc.set(core::ptr::null_mut());
    }
    let h = handle.get();
    if !h.is_null() {
        unsafe {
            let _ = cudnnDestroy(h);
        }
        handle.set(core::ptr::null_mut());
    }
}

/// Drive `cudnnPoolingForward` with appropriate alpha/beta scalar dtype.
/// Rank-agnostic: the pool descriptor carries the rank.
pub(crate) fn run_fw_nd<T: Element>(
    h: cudnnHandle_t,
    pool_desc: cudnnPoolingDescriptor_t,
    x_desc: cudnnTensorDescriptor_t,
    y_desc: cudnnTensorDescriptor_t,
    x_ptr: u64,
    y_ptr: u64,
) -> Result<()> {
    let status = if is_double_compute::<T>() {
        let alpha: f64 = 1.0;
        let beta: f64 = 0.0;
        unsafe {
            cudnnPoolingForward(
                h,
                pool_desc,
                &alpha as *const f64 as *const c_void,
                x_desc,
                x_ptr as *const c_void,
                &beta as *const f64 as *const c_void,
                y_desc,
                y_ptr as *mut c_void,
            )
        }
    } else {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        unsafe {
            cudnnPoolingForward(
                h,
                pool_desc,
                &alpha as *const f32 as *const c_void,
                x_desc,
                x_ptr as *const c_void,
                &beta as *const f32 as *const c_void,
                y_desc,
                y_ptr as *mut c_void,
            )
        }
    };
    if status != 0 {
        return Err(Error::CutlassInternal(-status));
    }
    Ok(())
}

/// Drive `cudnnPoolingBackward` with appropriate alpha/beta scalar dtype.
pub(crate) fn run_bw_nd<T: Element>(
    h: cudnnHandle_t,
    pool_desc: cudnnPoolingDescriptor_t,
    x_desc: cudnnTensorDescriptor_t,
    y_desc: cudnnTensorDescriptor_t,
    y_ptr: u64,
    dy_ptr: u64,
    x_ptr: u64,
    dx_ptr: u64,
) -> Result<()> {
    let status = if is_double_compute::<T>() {
        let alpha: f64 = 1.0;
        let beta: f64 = 0.0;
        unsafe {
            cudnnPoolingBackward(
                h,
                pool_desc,
                &alpha as *const f64 as *const c_void,
                y_desc,
                y_ptr as *const c_void,
                y_desc,
                dy_ptr as *const c_void,
                x_desc,
                x_ptr as *const c_void,
                &beta as *const f64 as *const c_void,
                x_desc,
                dx_ptr as *mut c_void,
            )
        }
    } else {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        unsafe {
            cudnnPoolingBackward(
                h,
                pool_desc,
                &alpha as *const f32 as *const c_void,
                y_desc,
                y_ptr as *const c_void,
                y_desc,
                dy_ptr as *const c_void,
                x_desc,
                x_ptr as *const c_void,
                &beta as *const f32 as *const c_void,
                x_desc,
                dx_ptr as *mut c_void,
            )
        }
    };
    if status != 0 {
        return Err(Error::CutlassInternal(-status));
    }
    Ok(())
}

/// Validate dtype is one of cuDNN's supported FP types for pooling.
pub(crate) fn validate_dtype<T: Element>() -> Result<()> {
    if !matches!(
        T::KIND,
        ElementKind::F32 | ElementKind::F64 | ElementKind::F16 | ElementKind::Bf16
    ) {
        return Err(Error::Unsupported(
            "baracuda-kernels::PoolNdPlan: cuDNN pooling supports f32 / f64 / f16 / bf16",
        ));
    }
    Ok(())
}

/// PyTorch's adaptive-pool kernel-size / stride derivation:
/// `kernel = ceil(in / out)`, `stride = floor(in / out)`, `pad = 0`.
///
/// **This is NOT bit-exact PyTorch.** PyTorch's adaptive pool uses
/// non-uniform per-output-cell kernel sizes when `in % out != 0`; cuDNN
/// only supports uniform kernels across all output cells. The uniform
/// approximation here matches the common case `in % out == 0` exactly
/// and degrades gracefully (within ±1 input cell of the true window) for
/// the non-divisible case.
#[inline]
pub(crate) fn adaptive_kernel_stride(in_dim: i32, out_dim: i32) -> (i32, i32) {
    debug_assert!(in_dim > 0 && out_dim > 0);
    let stride = in_dim / out_dim;
    let kernel = (in_dim + out_dim - 1) / out_dim; // ceil(in / out)
    (kernel, stride)
}
