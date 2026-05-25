//! cuDNN pool FFI facade (Phase 19.1).
//!
//! Exposes `extern "C"` wrappers for the non-adaptive `MaxPool{1,2,3}d` /
//! `AvgPool{1,2,3}d` cuDNN-backed plans. Closes the Phase 17 design
//! gap that "library-backed plans live as Rust-only" — every
//! library-backed plan must also ship a `baracuda-kernels-sys` FFI
//! symbol so downstream non-Rust callers (Fuel) can drive them without
//! going through the safe-plan layer.
//!
//! ## Implementation language
//!
//! These wrappers are pure-Rust `#[no_mangle] extern "C"` functions that
//! reuse the cuDNN FFI declared elsewhere in this crate. Going through
//! Rust (rather than building a parallel C++ launcher) avoids
//! duplicating the rank-1 / rank-3 / rank-N descriptor setup that the
//! Rust plan layer already encodes, and avoids a second cuDNN-handle
//! ownership story in C++ space.
//!
//! ## Handle lifecycle
//!
//! Each FFI call creates a cuDNN handle, sets the stream, builds the
//! descriptors, executes the pooling op, then tears everything down
//! before returning. There is no cross-call caching at the FFI layer —
//! callers that need to amortize handle / descriptor allocation across
//! many launches should drive the matching `baracuda-kernels` Rust plan
//! directly (it caches handle + descriptors for the lifetime of the
//! plan object). A future opaque-plan FFI (open `*mut PoolPlan` handle
//! returned by `*_create` and consumed by `*_run` / `*_destroy`) would
//! recover that amortization at the FFI layer; out of scope for 19.1.
//!
//! ## Indices argument
//!
//! cuDNN's legacy pooling API does **not** materialize a per-window
//! argmax indices tensor — for max-pool it recovers the argmax
//! internally from `(y, x)` during `cudnnPoolingBackward`. Therefore
//! these wrappers do not take an `indices` argument despite the prompt
//! template; instead the MaxPool BW signature requires both the saved
//! FW output `y` and saved FW input `x` (matching the Rust plan's
//! `Pool*dBwArgs::y` + `x` fields). Adding a no-op `indices` parameter
//! would be misleading. Callers needing an explicit i64 indices tensor
//! must use the bespoke `FractionalMaxPool` family (Phase 16.3) or the
//! `AdaptiveMaxPool` family (Phase 16.1), which expose their argmax
//! via the standard saved-indices contract.
//!
//! ## Status codes
//!
//! Same as the rest of `baracuda-kernels-sys`:
//! * `0` — success.
//! * `2` — invalid problem (cuDNN rejected the descriptor or shape).
//! * `5` — internal kernel error (cuDNN exec returned non-zero).
//!
//! Argument-validation failures map to `2`; cuDNN runtime failures map
//! to `5`. This mirrors the LpPool family status map (Phase 16.2).

#![cfg(feature = "cudnn")]

use core::ffi::c_void;
use core::ptr;

use super::{
    cudnnCreate, cudnnCreatePoolingDescriptor, cudnnCreateTensorDescriptor, cudnnDataType_t,
    cudnnDestroy, cudnnDestroyPoolingDescriptor, cudnnDestroyTensorDescriptor, cudnnHandle_t,
    cudnnPoolingBackward, cudnnPoolingDescriptor_t, cudnnPoolingForward,
    cudnnSetPooling2dDescriptor, cudnnSetPoolingNdDescriptor, cudnnSetStream,
    cudnnSetTensor4dDescriptor, cudnnSetTensorNdDescriptor, cudnnTensorDescriptor_t,
    CUDNN_DATA_BFLOAT16, CUDNN_DATA_DOUBLE, CUDNN_DATA_FLOAT, CUDNN_DATA_HALF,
    CUDNN_NOT_PROPAGATE_NAN, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, CUDNN_POOLING_MAX, CUDNN_TENSOR_NCHW,
};

// =============================================================================
// Internal helpers
// =============================================================================

/// Per-launch RAII handle for cuDNN resources — one cuDNN handle plus
/// the three descriptors the legacy pooling API needs. Cleaned up on
/// `Drop` even if the pool exec returns an error mid-launch.
struct PoolResources {
    handle: cudnnHandle_t,
    x_desc: cudnnTensorDescriptor_t,
    y_desc: cudnnTensorDescriptor_t,
    pool_desc: cudnnPoolingDescriptor_t,
}

impl PoolResources {
    fn new() -> Self {
        Self {
            handle: ptr::null_mut(),
            x_desc: ptr::null_mut(),
            y_desc: ptr::null_mut(),
            pool_desc: ptr::null_mut(),
        }
    }
}

impl Drop for PoolResources {
    fn drop(&mut self) {
        if !self.pool_desc.is_null() {
            unsafe {
                let _ = cudnnDestroyPoolingDescriptor(self.pool_desc);
            }
        }
        if !self.y_desc.is_null() {
            unsafe {
                let _ = cudnnDestroyTensorDescriptor(self.y_desc);
            }
        }
        if !self.x_desc.is_null() {
            unsafe {
                let _ = cudnnDestroyTensorDescriptor(self.x_desc);
            }
        }
        if !self.handle.is_null() {
            unsafe {
                let _ = cudnnDestroy(self.handle);
            }
        }
    }
}

#[inline]
fn map_status(code: i32) -> i32 {
    if code == 0 {
        0
    } else {
        5
    }
}

/// Compute contiguous row-major strides for a rank-`nb_dims` tensor
/// over the leading entries of `dims`. Used by the Nd path because
/// `cudnnSetTensorNdDescriptor` requires explicit strides.
#[inline]
fn fill_row_major_strides(dims: &[i32; 5], nb_dims: usize, strides: &mut [i32; 5]) {
    let mut acc: i64 = 1;
    let mut i = nb_dims;
    while i > 0 {
        i -= 1;
        strides[i] = acc as i32;
        acc = acc.saturating_mul(dims[i] as i64);
    }
}

#[inline]
fn validate_2d(
    batch: i32,
    channels: i32,
    h_in: i32,
    w_in: i32,
    kh: i32,
    kw: i32,
    sh: i32,
    sw: i32,
    ph: i32,
    pw: i32,
) -> i32 {
    if batch <= 0 || channels <= 0 || h_in <= 0 || w_in <= 0 {
        return 2;
    }
    if kh <= 0 || kw <= 0 || sh <= 0 || sw <= 0 {
        return 2;
    }
    if ph < 0 || pw < 0 {
        return 2;
    }
    if ph * 2 > kh || pw * 2 > kw {
        return 2;
    }
    0
}

#[inline]
fn validate_1d(
    batch: i32,
    channels: i32,
    l_in: i32,
    kl: i32,
    sl: i32,
    pl: i32,
) -> i32 {
    if batch <= 0 || channels <= 0 || l_in <= 0 {
        return 2;
    }
    if kl <= 0 || sl <= 0 {
        return 2;
    }
    if pl < 0 {
        return 2;
    }
    if pl * 2 > kl {
        return 2;
    }
    0
}

#[inline]
fn validate_3d(
    batch: i32,
    channels: i32,
    d_in: i32,
    h_in: i32,
    w_in: i32,
    kd: i32,
    kh: i32,
    kw: i32,
    sd: i32,
    sh: i32,
    sw: i32,
    pd: i32,
    ph: i32,
    pw: i32,
) -> i32 {
    if batch <= 0 || channels <= 0 || d_in <= 0 || h_in <= 0 || w_in <= 0 {
        return 2;
    }
    if kd <= 0 || kh <= 0 || kw <= 0 || sd <= 0 || sh <= 0 || sw <= 0 {
        return 2;
    }
    if pd < 0 || ph < 0 || pw < 0 {
        return 2;
    }
    if pd * 2 > kd || ph * 2 > kh || pw * 2 > kw {
        return 2;
    }
    0
}

/// Build the (handle, x_desc, y_desc, pool_desc) trio for a 2-D pool
/// launch. Returns a status code (0 = success).
#[allow(clippy::too_many_arguments)]
unsafe fn setup_2d(
    res: &mut PoolResources,
    dtype: cudnnDataType_t,
    mode: i32,
    stream: *mut c_void,
    batch: i32,
    channels: i32,
    h_in: i32,
    w_in: i32,
    h_out: i32,
    w_out: i32,
    kh: i32,
    kw: i32,
    sh: i32,
    sw: i32,
    ph: i32,
    pw: i32,
) -> i32 {
    let s = unsafe { cudnnCreate(&mut res.handle as *mut _) };
    if s != 0 {
        return 5;
    }
    let s = unsafe { cudnnSetStream(res.handle, stream) };
    if s != 0 {
        return 5;
    }
    let s = unsafe { cudnnCreateTensorDescriptor(&mut res.x_desc as *mut _) };
    if s != 0 {
        return 5;
    }
    let s = unsafe {
        cudnnSetTensor4dDescriptor(
            res.x_desc,
            CUDNN_TENSOR_NCHW,
            dtype,
            batch,
            channels,
            h_in,
            w_in,
        )
    };
    if s != 0 {
        return 5;
    }
    let s = unsafe { cudnnCreateTensorDescriptor(&mut res.y_desc as *mut _) };
    if s != 0 {
        return 5;
    }
    let s = unsafe {
        cudnnSetTensor4dDescriptor(
            res.y_desc,
            CUDNN_TENSOR_NCHW,
            dtype,
            batch,
            channels,
            h_out,
            w_out,
        )
    };
    if s != 0 {
        return 5;
    }
    let s = unsafe { cudnnCreatePoolingDescriptor(&mut res.pool_desc as *mut _) };
    if s != 0 {
        return 5;
    }
    let s = unsafe {
        cudnnSetPooling2dDescriptor(
            res.pool_desc,
            mode,
            CUDNN_NOT_PROPAGATE_NAN,
            kh,
            kw,
            ph,
            pw,
            sh,
            sw,
        )
    };
    if s != 0 {
        return 5;
    }
    0
}

/// Build the (handle, x_desc, y_desc, pool_desc) trio for an Nd pool
/// launch (rank 1 or 3). `x_dims` / `y_dims` are the rank-3 or rank-5
/// shape arrays; cuDNN's `nb_dims >= 4` floor forces rank-3 to be
/// padded internally with a trailing `W = 1`.
#[allow(clippy::too_many_arguments)]
unsafe fn setup_nd(
    res: &mut PoolResources,
    dtype: cudnnDataType_t,
    mode: i32,
    stream: *mut c_void,
    rank: usize, // 1 or 3 (spatial axes)
    x_dims: &[i32],
    y_dims: &[i32],
    window: &[i32],
    padding: &[i32],
    stride: &[i32],
) -> i32 {
    let s = unsafe { cudnnCreate(&mut res.handle as *mut _) };
    if s != 0 {
        return 5;
    }
    let s = unsafe { cudnnSetStream(res.handle, stream) };
    if s != 0 {
        return 5;
    }

    // Pad rank-3 tensor dims to rank-4 (W = 1) because cuDNN's
    // SetTensorNdDescriptor rejects nb_dims < 4.
    let nb_dims = if rank == 1 { 4 } else { 5 };

    let mut x_padded: [i32; 5] = [1; 5];
    let mut y_padded: [i32; 5] = [1; 5];
    for (i, &d) in x_dims.iter().enumerate() {
        x_padded[i] = d;
    }
    for (i, &d) in y_dims.iter().enumerate() {
        y_padded[i] = d;
    }
    let mut x_str: [i32; 5] = [1; 5];
    let mut y_str: [i32; 5] = [1; 5];
    fill_row_major_strides(&x_padded, nb_dims, &mut x_str);
    fill_row_major_strides(&y_padded, nb_dims, &mut y_str);

    let s = unsafe { cudnnCreateTensorDescriptor(&mut res.x_desc as *mut _) };
    if s != 0 {
        return 5;
    }
    let s = unsafe {
        cudnnSetTensorNdDescriptor(
            res.x_desc,
            dtype,
            nb_dims as i32,
            x_padded.as_ptr(),
            x_str.as_ptr(),
        )
    };
    if s != 0 {
        return 5;
    }

    let s = unsafe { cudnnCreateTensorDescriptor(&mut res.y_desc as *mut _) };
    if s != 0 {
        return 5;
    }
    let s = unsafe {
        cudnnSetTensorNdDescriptor(
            res.y_desc,
            dtype,
            nb_dims as i32,
            y_padded.as_ptr(),
            y_str.as_ptr(),
        )
    };
    if s != 0 {
        return 5;
    }

    // Pad rank-1 pool descriptor to rank-2 (cuDNN's SetPoolingNdDescriptor
    // requires nb_dims >= 2): degenerate trailing window=1 / pad=0 /
    // stride=1.
    let nb_pool_dims = if rank == 1 { 2 } else { 3 };
    let mut win: [i32; 3] = [1; 3];
    let mut pad: [i32; 3] = [0; 3];
    let mut str_: [i32; 3] = [1; 3];
    for (i, &w) in window.iter().enumerate() {
        win[i] = w;
    }
    for (i, &p) in padding.iter().enumerate() {
        pad[i] = p;
    }
    for (i, &s_) in stride.iter().enumerate() {
        str_[i] = s_;
    }

    let s = unsafe { cudnnCreatePoolingDescriptor(&mut res.pool_desc as *mut _) };
    if s != 0 {
        return 5;
    }
    let s = unsafe {
        cudnnSetPoolingNdDescriptor(
            res.pool_desc,
            mode,
            CUDNN_NOT_PROPAGATE_NAN,
            nb_pool_dims as i32,
            win.as_ptr(),
            pad.as_ptr(),
            str_.as_ptr(),
        )
    };
    if s != 0 {
        return 5;
    }
    0
}

/// Drive `cudnnPoolingForward` with the appropriate alpha/beta dtype
/// (f64 for CUDNN_DATA_DOUBLE, f32 otherwise).
unsafe fn exec_fw(
    res: &PoolResources,
    dtype: cudnnDataType_t,
    x: *const c_void,
    y: *mut c_void,
) -> i32 {
    let status = if dtype == CUDNN_DATA_DOUBLE {
        let alpha: f64 = 1.0;
        let beta: f64 = 0.0;
        unsafe {
            cudnnPoolingForward(
                res.handle,
                res.pool_desc,
                &alpha as *const f64 as *const c_void,
                res.x_desc,
                x,
                &beta as *const f64 as *const c_void,
                res.y_desc,
                y,
            )
        }
    } else {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        unsafe {
            cudnnPoolingForward(
                res.handle,
                res.pool_desc,
                &alpha as *const f32 as *const c_void,
                res.x_desc,
                x,
                &beta as *const f32 as *const c_void,
                res.y_desc,
                y,
            )
        }
    };
    map_status(status)
}

/// Drive `cudnnPoolingBackward` with the appropriate alpha/beta dtype.
unsafe fn exec_bw(
    res: &PoolResources,
    dtype: cudnnDataType_t,
    y: *const c_void,
    dy: *const c_void,
    x: *const c_void,
    dx: *mut c_void,
) -> i32 {
    let status = if dtype == CUDNN_DATA_DOUBLE {
        let alpha: f64 = 1.0;
        let beta: f64 = 0.0;
        unsafe {
            cudnnPoolingBackward(
                res.handle,
                res.pool_desc,
                &alpha as *const f64 as *const c_void,
                res.y_desc,
                y,
                res.y_desc,
                dy,
                res.x_desc,
                x,
                &beta as *const f64 as *const c_void,
                res.x_desc,
                dx,
            )
        }
    } else {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        unsafe {
            cudnnPoolingBackward(
                res.handle,
                res.pool_desc,
                &alpha as *const f32 as *const c_void,
                res.y_desc,
                y,
                res.y_desc,
                dy,
                res.x_desc,
                x,
                &beta as *const f32 as *const c_void,
                res.x_desc,
                dx,
            )
        }
    };
    map_status(status)
}

// =============================================================================
// Core launcher templates
// =============================================================================

#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn run_pool_2d_fw(
    mode: i32,
    dtype: cudnnDataType_t,
    batch: i32,
    channels: i32,
    h_in: i32,
    w_in: i32,
    h_out: i32,
    w_out: i32,
    kh: i32,
    kw: i32,
    sh: i32,
    sw: i32,
    ph: i32,
    pw: i32,
    x: *const c_void,
    y: *mut c_void,
    stream: *mut c_void,
) -> i32 {
    let v = validate_2d(batch, channels, h_in, w_in, kh, kw, sh, sw, ph, pw);
    if v != 0 {
        return v;
    }
    if h_out <= 0 || w_out <= 0 {
        return 2;
    }
    let mut res = PoolResources::new();
    let s = unsafe {
        setup_2d(
            &mut res, dtype, mode, stream, batch, channels, h_in, w_in, h_out, w_out, kh, kw, sh,
            sw, ph, pw,
        )
    };
    if s != 0 {
        return s;
    }
    unsafe { exec_fw(&res, dtype, x, y) }
}

#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn run_pool_2d_bw(
    mode: i32,
    dtype: cudnnDataType_t,
    batch: i32,
    channels: i32,
    h_in: i32,
    w_in: i32,
    h_out: i32,
    w_out: i32,
    kh: i32,
    kw: i32,
    sh: i32,
    sw: i32,
    ph: i32,
    pw: i32,
    y: *const c_void,
    dy: *const c_void,
    x: *const c_void,
    dx: *mut c_void,
    stream: *mut c_void,
) -> i32 {
    let v = validate_2d(batch, channels, h_in, w_in, kh, kw, sh, sw, ph, pw);
    if v != 0 {
        return v;
    }
    if h_out <= 0 || w_out <= 0 {
        return 2;
    }
    let mut res = PoolResources::new();
    let s = unsafe {
        setup_2d(
            &mut res, dtype, mode, stream, batch, channels, h_in, w_in, h_out, w_out, kh, kw, sh,
            sw, ph, pw,
        )
    };
    if s != 0 {
        return s;
    }
    unsafe { exec_bw(&res, dtype, y, dy, x, dx) }
}

#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn run_pool_1d_fw(
    mode: i32,
    dtype: cudnnDataType_t,
    batch: i32,
    channels: i32,
    l_in: i32,
    l_out: i32,
    kl: i32,
    sl: i32,
    pl: i32,
    x: *const c_void,
    y: *mut c_void,
    stream: *mut c_void,
) -> i32 {
    let v = validate_1d(batch, channels, l_in, kl, sl, pl);
    if v != 0 {
        return v;
    }
    if l_out <= 0 {
        return 2;
    }
    let x_dims = [batch, channels, l_in];
    let y_dims = [batch, channels, l_out];
    let window = [kl];
    let padding = [pl];
    let stride = [sl];
    let mut res = PoolResources::new();
    let s = unsafe {
        setup_nd(
            &mut res, dtype, mode, stream, 1, &x_dims, &y_dims, &window, &padding, &stride,
        )
    };
    if s != 0 {
        return s;
    }
    unsafe { exec_fw(&res, dtype, x, y) }
}

#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn run_pool_1d_bw(
    mode: i32,
    dtype: cudnnDataType_t,
    batch: i32,
    channels: i32,
    l_in: i32,
    l_out: i32,
    kl: i32,
    sl: i32,
    pl: i32,
    y: *const c_void,
    dy: *const c_void,
    x: *const c_void,
    dx: *mut c_void,
    stream: *mut c_void,
) -> i32 {
    let v = validate_1d(batch, channels, l_in, kl, sl, pl);
    if v != 0 {
        return v;
    }
    if l_out <= 0 {
        return 2;
    }
    let x_dims = [batch, channels, l_in];
    let y_dims = [batch, channels, l_out];
    let window = [kl];
    let padding = [pl];
    let stride = [sl];
    let mut res = PoolResources::new();
    let s = unsafe {
        setup_nd(
            &mut res, dtype, mode, stream, 1, &x_dims, &y_dims, &window, &padding, &stride,
        )
    };
    if s != 0 {
        return s;
    }
    unsafe { exec_bw(&res, dtype, y, dy, x, dx) }
}

#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn run_pool_3d_fw(
    mode: i32,
    dtype: cudnnDataType_t,
    batch: i32,
    channels: i32,
    d_in: i32,
    h_in: i32,
    w_in: i32,
    d_out: i32,
    h_out: i32,
    w_out: i32,
    kd: i32,
    kh: i32,
    kw: i32,
    sd: i32,
    sh: i32,
    sw: i32,
    pd: i32,
    ph: i32,
    pw: i32,
    x: *const c_void,
    y: *mut c_void,
    stream: *mut c_void,
) -> i32 {
    let v = validate_3d(
        batch, channels, d_in, h_in, w_in, kd, kh, kw, sd, sh, sw, pd, ph, pw,
    );
    if v != 0 {
        return v;
    }
    if d_out <= 0 || h_out <= 0 || w_out <= 0 {
        return 2;
    }
    let x_dims = [batch, channels, d_in, h_in, w_in];
    let y_dims = [batch, channels, d_out, h_out, w_out];
    let window = [kd, kh, kw];
    let padding = [pd, ph, pw];
    let stride = [sd, sh, sw];
    let mut res = PoolResources::new();
    let s = unsafe {
        setup_nd(
            &mut res, dtype, mode, stream, 3, &x_dims, &y_dims, &window, &padding, &stride,
        )
    };
    if s != 0 {
        return s;
    }
    unsafe { exec_fw(&res, dtype, x, y) }
}

#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn run_pool_3d_bw(
    mode: i32,
    dtype: cudnnDataType_t,
    batch: i32,
    channels: i32,
    d_in: i32,
    h_in: i32,
    w_in: i32,
    d_out: i32,
    h_out: i32,
    w_out: i32,
    kd: i32,
    kh: i32,
    kw: i32,
    sd: i32,
    sh: i32,
    sw: i32,
    pd: i32,
    ph: i32,
    pw: i32,
    y: *const c_void,
    dy: *const c_void,
    x: *const c_void,
    dx: *mut c_void,
    stream: *mut c_void,
) -> i32 {
    let v = validate_3d(
        batch, channels, d_in, h_in, w_in, kd, kh, kw, sd, sh, sw, pd, ph, pw,
    );
    if v != 0 {
        return v;
    }
    if d_out <= 0 || h_out <= 0 || w_out <= 0 {
        return 2;
    }
    let x_dims = [batch, channels, d_in, h_in, w_in];
    let y_dims = [batch, channels, d_out, h_out, w_out];
    let window = [kd, kh, kw];
    let padding = [pd, ph, pw];
    let stride = [sd, sh, sw];
    let mut res = PoolResources::new();
    let s = unsafe {
        setup_nd(
            &mut res, dtype, mode, stream, 3, &x_dims, &y_dims, &window, &padding, &stride,
        )
    };
    if s != 0 {
        return s;
    }
    unsafe { exec_bw(&res, dtype, y, dy, x, dx) }
}

// =============================================================================
// MaxPool 1D — FW + BW × 4 fp dtypes
// =============================================================================
//
// Signature (FW): `(batch, channels, l_in, l_out, kl, sl, pl, x, y, stream)`.
// Signature (BW): `(batch, channels, l_in, l_out, kl, sl, pl, y, dy, x, dx,
// stream)`. No `indices` parameter — cuDNN reconstructs the argmax from
// (y, x) internally; saved-FW input + output must both be passed in BW.

macro_rules! max_pool_1d_pair {
    ($fw:ident, $bw:ident, $dtype:expr) => {
        /// MaxPool1d FW (`y := max_pool(x)`) — cuDNN-backed.
        ///
        /// # Safety
        /// All pointer args must be device-resident and remain valid for
        /// the duration of the launch. `stream` must be a live CUDA
        /// stream in the current context.
        #[unsafe(no_mangle)]
        #[allow(clippy::too_many_arguments)]
        pub unsafe extern "C" fn $fw(
            batch: i32,
            channels: i32,
            l_in: i32,
            l_out: i32,
            kl: i32,
            sl: i32,
            pl: i32,
            x: *const c_void,
            y: *mut c_void,
            stream: *mut c_void,
        ) -> i32 {
            unsafe {
                run_pool_1d_fw(
                    CUDNN_POOLING_MAX,
                    $dtype,
                    batch,
                    channels,
                    l_in,
                    l_out,
                    kl,
                    sl,
                    pl,
                    x,
                    y,
                    stream,
                )
            }
        }

        /// MaxPool1d BW (`dx := max_pool_grad(y, dy, x)`) — cuDNN-backed.
        ///
        /// Both `y` (saved FW output) and `x` (saved FW input) are
        /// required; cuDNN's legacy pooling API recovers the per-window
        /// argmax from `(y, x)` rather than from a separate indices
        /// tensor.
        ///
        /// # Safety
        /// As for FW.
        #[unsafe(no_mangle)]
        #[allow(clippy::too_many_arguments)]
        pub unsafe extern "C" fn $bw(
            batch: i32,
            channels: i32,
            l_in: i32,
            l_out: i32,
            kl: i32,
            sl: i32,
            pl: i32,
            y: *const c_void,
            dy: *const c_void,
            x: *const c_void,
            dx: *mut c_void,
            stream: *mut c_void,
        ) -> i32 {
            unsafe {
                run_pool_1d_bw(
                    CUDNN_POOLING_MAX,
                    $dtype,
                    batch,
                    channels,
                    l_in,
                    l_out,
                    kl,
                    sl,
                    pl,
                    y,
                    dy,
                    x,
                    dx,
                    stream,
                )
            }
        }
    };
}

max_pool_1d_pair!(
    baracuda_kernels_max_pool_1d_fw_f32_run,
    baracuda_kernels_max_pool_1d_bw_f32_run,
    CUDNN_DATA_FLOAT
);
max_pool_1d_pair!(
    baracuda_kernels_max_pool_1d_fw_f64_run,
    baracuda_kernels_max_pool_1d_bw_f64_run,
    CUDNN_DATA_DOUBLE
);
max_pool_1d_pair!(
    baracuda_kernels_max_pool_1d_fw_f16_run,
    baracuda_kernels_max_pool_1d_bw_f16_run,
    CUDNN_DATA_HALF
);
max_pool_1d_pair!(
    baracuda_kernels_max_pool_1d_fw_bf16_run,
    baracuda_kernels_max_pool_1d_bw_bf16_run,
    CUDNN_DATA_BFLOAT16
);

// =============================================================================
// MaxPool 2D — FW + BW × 4 fp dtypes
// =============================================================================

macro_rules! max_pool_2d_pair {
    ($fw:ident, $bw:ident, $dtype:expr) => {
        /// MaxPool2d FW (`y := max_pool(x)`) — cuDNN-backed, NCHW.
        ///
        /// # Safety
        /// As for the 1D FW entry point.
        #[unsafe(no_mangle)]
        #[allow(clippy::too_many_arguments)]
        pub unsafe extern "C" fn $fw(
            batch: i32,
            channels: i32,
            h_in: i32,
            w_in: i32,
            h_out: i32,
            w_out: i32,
            kh: i32,
            kw: i32,
            sh: i32,
            sw: i32,
            ph: i32,
            pw: i32,
            x: *const c_void,
            y: *mut c_void,
            stream: *mut c_void,
        ) -> i32 {
            unsafe {
                run_pool_2d_fw(
                    CUDNN_POOLING_MAX,
                    $dtype,
                    batch,
                    channels,
                    h_in,
                    w_in,
                    h_out,
                    w_out,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    x,
                    y,
                    stream,
                )
            }
        }

        /// MaxPool2d BW (`dx := max_pool_grad(y, dy, x)`) — cuDNN-backed.
        ///
        /// Both `y` and `x` are required (cuDNN reconstructs the
        /// per-window argmax from them — no separate indices tensor).
        ///
        /// # Safety
        /// As for the 1D BW entry point.
        #[unsafe(no_mangle)]
        #[allow(clippy::too_many_arguments)]
        pub unsafe extern "C" fn $bw(
            batch: i32,
            channels: i32,
            h_in: i32,
            w_in: i32,
            h_out: i32,
            w_out: i32,
            kh: i32,
            kw: i32,
            sh: i32,
            sw: i32,
            ph: i32,
            pw: i32,
            y: *const c_void,
            dy: *const c_void,
            x: *const c_void,
            dx: *mut c_void,
            stream: *mut c_void,
        ) -> i32 {
            unsafe {
                run_pool_2d_bw(
                    CUDNN_POOLING_MAX,
                    $dtype,
                    batch,
                    channels,
                    h_in,
                    w_in,
                    h_out,
                    w_out,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    y,
                    dy,
                    x,
                    dx,
                    stream,
                )
            }
        }
    };
}

max_pool_2d_pair!(
    baracuda_kernels_max_pool_2d_fw_f32_run,
    baracuda_kernels_max_pool_2d_bw_f32_run,
    CUDNN_DATA_FLOAT
);
max_pool_2d_pair!(
    baracuda_kernels_max_pool_2d_fw_f64_run,
    baracuda_kernels_max_pool_2d_bw_f64_run,
    CUDNN_DATA_DOUBLE
);
max_pool_2d_pair!(
    baracuda_kernels_max_pool_2d_fw_f16_run,
    baracuda_kernels_max_pool_2d_bw_f16_run,
    CUDNN_DATA_HALF
);
max_pool_2d_pair!(
    baracuda_kernels_max_pool_2d_fw_bf16_run,
    baracuda_kernels_max_pool_2d_bw_bf16_run,
    CUDNN_DATA_BFLOAT16
);

// =============================================================================
// MaxPool 3D — FW + BW × 4 fp dtypes
// =============================================================================

macro_rules! max_pool_3d_pair {
    ($fw:ident, $bw:ident, $dtype:expr) => {
        /// MaxPool3d FW (`y := max_pool(x)`) — cuDNN-backed, NCDHW.
        ///
        /// # Safety
        /// As for the 1D FW entry point.
        #[unsafe(no_mangle)]
        #[allow(clippy::too_many_arguments)]
        pub unsafe extern "C" fn $fw(
            batch: i32,
            channels: i32,
            d_in: i32,
            h_in: i32,
            w_in: i32,
            d_out: i32,
            h_out: i32,
            w_out: i32,
            kd: i32,
            kh: i32,
            kw: i32,
            sd: i32,
            sh: i32,
            sw: i32,
            pd: i32,
            ph: i32,
            pw: i32,
            x: *const c_void,
            y: *mut c_void,
            stream: *mut c_void,
        ) -> i32 {
            unsafe {
                run_pool_3d_fw(
                    CUDNN_POOLING_MAX,
                    $dtype,
                    batch,
                    channels,
                    d_in,
                    h_in,
                    w_in,
                    d_out,
                    h_out,
                    w_out,
                    kd,
                    kh,
                    kw,
                    sd,
                    sh,
                    sw,
                    pd,
                    ph,
                    pw,
                    x,
                    y,
                    stream,
                )
            }
        }

        /// MaxPool3d BW (`dx := max_pool_grad(y, dy, x)`) — cuDNN-backed.
        ///
        /// # Safety
        /// As for the 1D BW entry point.
        #[unsafe(no_mangle)]
        #[allow(clippy::too_many_arguments)]
        pub unsafe extern "C" fn $bw(
            batch: i32,
            channels: i32,
            d_in: i32,
            h_in: i32,
            w_in: i32,
            d_out: i32,
            h_out: i32,
            w_out: i32,
            kd: i32,
            kh: i32,
            kw: i32,
            sd: i32,
            sh: i32,
            sw: i32,
            pd: i32,
            ph: i32,
            pw: i32,
            y: *const c_void,
            dy: *const c_void,
            x: *const c_void,
            dx: *mut c_void,
            stream: *mut c_void,
        ) -> i32 {
            unsafe {
                run_pool_3d_bw(
                    CUDNN_POOLING_MAX,
                    $dtype,
                    batch,
                    channels,
                    d_in,
                    h_in,
                    w_in,
                    d_out,
                    h_out,
                    w_out,
                    kd,
                    kh,
                    kw,
                    sd,
                    sh,
                    sw,
                    pd,
                    ph,
                    pw,
                    y,
                    dy,
                    x,
                    dx,
                    stream,
                )
            }
        }
    };
}

max_pool_3d_pair!(
    baracuda_kernels_max_pool_3d_fw_f32_run,
    baracuda_kernels_max_pool_3d_bw_f32_run,
    CUDNN_DATA_FLOAT
);
max_pool_3d_pair!(
    baracuda_kernels_max_pool_3d_fw_f64_run,
    baracuda_kernels_max_pool_3d_bw_f64_run,
    CUDNN_DATA_DOUBLE
);
max_pool_3d_pair!(
    baracuda_kernels_max_pool_3d_fw_f16_run,
    baracuda_kernels_max_pool_3d_bw_f16_run,
    CUDNN_DATA_HALF
);
max_pool_3d_pair!(
    baracuda_kernels_max_pool_3d_fw_bf16_run,
    baracuda_kernels_max_pool_3d_bw_bf16_run,
    CUDNN_DATA_BFLOAT16
);

// =============================================================================
// AvgPool 1D — FW + BW × 4 fp dtypes
// =============================================================================
//
// `count_include_pad`: `i32` truthy flag — non-zero selects
// `CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING` (TensorFlow default);
// zero selects `CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING` (PyTorch
// `nn.AvgPool*d` default, `count_include_pad=False`).

#[inline]
fn avg_mode(count_include_pad: i32) -> i32 {
    if count_include_pad != 0 {
        CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
    } else {
        CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
    }
}

macro_rules! avg_pool_1d_pair {
    ($fw:ident, $bw:ident, $dtype:expr) => {
        /// AvgPool1d FW — cuDNN-backed. `count_include_pad` non-zero =
        /// TensorFlow-style (divide by full window); zero = PyTorch
        /// default (divide by valid-cell count).
        ///
        /// # Safety
        /// As for `MaxPool1d` FW.
        #[unsafe(no_mangle)]
        #[allow(clippy::too_many_arguments)]
        pub unsafe extern "C" fn $fw(
            batch: i32,
            channels: i32,
            l_in: i32,
            l_out: i32,
            kl: i32,
            sl: i32,
            pl: i32,
            count_include_pad: i32,
            x: *const c_void,
            y: *mut c_void,
            stream: *mut c_void,
        ) -> i32 {
            unsafe {
                run_pool_1d_fw(
                    avg_mode(count_include_pad),
                    $dtype,
                    batch,
                    channels,
                    l_in,
                    l_out,
                    kl,
                    sl,
                    pl,
                    x,
                    y,
                    stream,
                )
            }
        }

        /// AvgPool1d BW — cuDNN-backed. cuDNN demands both `y` and `x`
        /// even though avg-pool's gradient depends only on `x`
        /// mathematically; pass the saved FW output and input.
        ///
        /// # Safety
        /// As for `MaxPool1d` BW.
        #[unsafe(no_mangle)]
        #[allow(clippy::too_many_arguments)]
        pub unsafe extern "C" fn $bw(
            batch: i32,
            channels: i32,
            l_in: i32,
            l_out: i32,
            kl: i32,
            sl: i32,
            pl: i32,
            count_include_pad: i32,
            y: *const c_void,
            dy: *const c_void,
            x: *const c_void,
            dx: *mut c_void,
            stream: *mut c_void,
        ) -> i32 {
            unsafe {
                run_pool_1d_bw(
                    avg_mode(count_include_pad),
                    $dtype,
                    batch,
                    channels,
                    l_in,
                    l_out,
                    kl,
                    sl,
                    pl,
                    y,
                    dy,
                    x,
                    dx,
                    stream,
                )
            }
        }
    };
}

avg_pool_1d_pair!(
    baracuda_kernels_avg_pool_1d_fw_f32_run,
    baracuda_kernels_avg_pool_1d_bw_f32_run,
    CUDNN_DATA_FLOAT
);
avg_pool_1d_pair!(
    baracuda_kernels_avg_pool_1d_fw_f64_run,
    baracuda_kernels_avg_pool_1d_bw_f64_run,
    CUDNN_DATA_DOUBLE
);
avg_pool_1d_pair!(
    baracuda_kernels_avg_pool_1d_fw_f16_run,
    baracuda_kernels_avg_pool_1d_bw_f16_run,
    CUDNN_DATA_HALF
);
avg_pool_1d_pair!(
    baracuda_kernels_avg_pool_1d_fw_bf16_run,
    baracuda_kernels_avg_pool_1d_bw_bf16_run,
    CUDNN_DATA_BFLOAT16
);

// =============================================================================
// AvgPool 2D — FW + BW × 4 fp dtypes
// =============================================================================

macro_rules! avg_pool_2d_pair {
    ($fw:ident, $bw:ident, $dtype:expr) => {
        /// AvgPool2d FW — cuDNN-backed, NCHW.
        ///
        /// # Safety
        /// As for `MaxPool2d` FW.
        #[unsafe(no_mangle)]
        #[allow(clippy::too_many_arguments)]
        pub unsafe extern "C" fn $fw(
            batch: i32,
            channels: i32,
            h_in: i32,
            w_in: i32,
            h_out: i32,
            w_out: i32,
            kh: i32,
            kw: i32,
            sh: i32,
            sw: i32,
            ph: i32,
            pw: i32,
            count_include_pad: i32,
            x: *const c_void,
            y: *mut c_void,
            stream: *mut c_void,
        ) -> i32 {
            unsafe {
                run_pool_2d_fw(
                    avg_mode(count_include_pad),
                    $dtype,
                    batch,
                    channels,
                    h_in,
                    w_in,
                    h_out,
                    w_out,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    x,
                    y,
                    stream,
                )
            }
        }

        /// AvgPool2d BW — cuDNN-backed.
        ///
        /// # Safety
        /// As for `MaxPool2d` BW.
        #[unsafe(no_mangle)]
        #[allow(clippy::too_many_arguments)]
        pub unsafe extern "C" fn $bw(
            batch: i32,
            channels: i32,
            h_in: i32,
            w_in: i32,
            h_out: i32,
            w_out: i32,
            kh: i32,
            kw: i32,
            sh: i32,
            sw: i32,
            ph: i32,
            pw: i32,
            count_include_pad: i32,
            y: *const c_void,
            dy: *const c_void,
            x: *const c_void,
            dx: *mut c_void,
            stream: *mut c_void,
        ) -> i32 {
            unsafe {
                run_pool_2d_bw(
                    avg_mode(count_include_pad),
                    $dtype,
                    batch,
                    channels,
                    h_in,
                    w_in,
                    h_out,
                    w_out,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    y,
                    dy,
                    x,
                    dx,
                    stream,
                )
            }
        }
    };
}

avg_pool_2d_pair!(
    baracuda_kernels_avg_pool_2d_fw_f32_run,
    baracuda_kernels_avg_pool_2d_bw_f32_run,
    CUDNN_DATA_FLOAT
);
avg_pool_2d_pair!(
    baracuda_kernels_avg_pool_2d_fw_f64_run,
    baracuda_kernels_avg_pool_2d_bw_f64_run,
    CUDNN_DATA_DOUBLE
);
avg_pool_2d_pair!(
    baracuda_kernels_avg_pool_2d_fw_f16_run,
    baracuda_kernels_avg_pool_2d_bw_f16_run,
    CUDNN_DATA_HALF
);
avg_pool_2d_pair!(
    baracuda_kernels_avg_pool_2d_fw_bf16_run,
    baracuda_kernels_avg_pool_2d_bw_bf16_run,
    CUDNN_DATA_BFLOAT16
);

// =============================================================================
// AvgPool 3D — FW + BW × 4 fp dtypes
// =============================================================================

macro_rules! avg_pool_3d_pair {
    ($fw:ident, $bw:ident, $dtype:expr) => {
        /// AvgPool3d FW — cuDNN-backed, NCDHW.
        ///
        /// # Safety
        /// As for `MaxPool3d` FW.
        #[unsafe(no_mangle)]
        #[allow(clippy::too_many_arguments)]
        pub unsafe extern "C" fn $fw(
            batch: i32,
            channels: i32,
            d_in: i32,
            h_in: i32,
            w_in: i32,
            d_out: i32,
            h_out: i32,
            w_out: i32,
            kd: i32,
            kh: i32,
            kw: i32,
            sd: i32,
            sh: i32,
            sw: i32,
            pd: i32,
            ph: i32,
            pw: i32,
            count_include_pad: i32,
            x: *const c_void,
            y: *mut c_void,
            stream: *mut c_void,
        ) -> i32 {
            unsafe {
                run_pool_3d_fw(
                    avg_mode(count_include_pad),
                    $dtype,
                    batch,
                    channels,
                    d_in,
                    h_in,
                    w_in,
                    d_out,
                    h_out,
                    w_out,
                    kd,
                    kh,
                    kw,
                    sd,
                    sh,
                    sw,
                    pd,
                    ph,
                    pw,
                    x,
                    y,
                    stream,
                )
            }
        }

        /// AvgPool3d BW — cuDNN-backed.
        ///
        /// # Safety
        /// As for `MaxPool3d` BW.
        #[unsafe(no_mangle)]
        #[allow(clippy::too_many_arguments)]
        pub unsafe extern "C" fn $bw(
            batch: i32,
            channels: i32,
            d_in: i32,
            h_in: i32,
            w_in: i32,
            d_out: i32,
            h_out: i32,
            w_out: i32,
            kd: i32,
            kh: i32,
            kw: i32,
            sd: i32,
            sh: i32,
            sw: i32,
            pd: i32,
            ph: i32,
            pw: i32,
            count_include_pad: i32,
            y: *const c_void,
            dy: *const c_void,
            x: *const c_void,
            dx: *mut c_void,
            stream: *mut c_void,
        ) -> i32 {
            unsafe {
                run_pool_3d_bw(
                    avg_mode(count_include_pad),
                    $dtype,
                    batch,
                    channels,
                    d_in,
                    h_in,
                    w_in,
                    d_out,
                    h_out,
                    w_out,
                    kd,
                    kh,
                    kw,
                    sd,
                    sh,
                    sw,
                    pd,
                    ph,
                    pw,
                    y,
                    dy,
                    x,
                    dx,
                    stream,
                )
            }
        }
    };
}

avg_pool_3d_pair!(
    baracuda_kernels_avg_pool_3d_fw_f32_run,
    baracuda_kernels_avg_pool_3d_bw_f32_run,
    CUDNN_DATA_FLOAT
);
avg_pool_3d_pair!(
    baracuda_kernels_avg_pool_3d_fw_f64_run,
    baracuda_kernels_avg_pool_3d_bw_f64_run,
    CUDNN_DATA_DOUBLE
);
avg_pool_3d_pair!(
    baracuda_kernels_avg_pool_3d_fw_f16_run,
    baracuda_kernels_avg_pool_3d_bw_f16_run,
    CUDNN_DATA_HALF
);
avg_pool_3d_pair!(
    baracuda_kernels_avg_pool_3d_fw_bf16_run,
    baracuda_kernels_avg_pool_3d_bw_bf16_run,
    CUDNN_DATA_BFLOAT16
);
