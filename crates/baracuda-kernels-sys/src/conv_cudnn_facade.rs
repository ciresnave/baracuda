//! Phase 19.2 — `baracuda-kernels-sys` C-ABI FFI wrappers for the
//! cuDNN-backed convolution family.
//!
//! Background: baracuda's design intent is "one unified CUDA-stack
//! facade" — downstream consumers get a single C-ABI surface that
//! covers both bespoke and library-backed ops. The cuDNN-backed
//! `Conv{1,2,3}dPlan` / `ConvTranspose{1,2,3}dPlan` plans that landed
//! in Phase 7 / Phase 11 were missing this layer (they only existed
//! as Rust plans). Phase 19.2 closes the gap by adding 72 thin
//! `extern "C"` wrappers — one per (plan × direction × fp dtype) — that
//! manage a transient cuDNN handle + descriptor set internally and
//! invoke the same cuDNN exec entry points the Rust plans use.
//!
//! Gated behind `feature = "cudnn"`. Each wrapper:
//!
//! 1. Validates the input pointers / extents.
//! 2. Creates a transient cuDNN handle + the four descriptors
//!    (input tensor, output tensor, filter, convolution).
//! 3. Binds the caller-supplied CUDA stream.
//! 4. Queries the per-direction workspace size, allocates from
//!    `cudaMalloc` if the caller-supplied workspace pointer is null
//!    AND a non-zero size is needed (otherwise uses the caller's
//!    buffer).
//! 5. Invokes the cuDNN exec entry point.
//! 6. Destroys the descriptors + handle.
//!
//! **Algorithm pinning**: FW = `IMPLICIT_GEMM` (algo 0); BW-data /
//! BW-filter = `ALGO_1`. Matches the Rust plans exactly.
//!
//! **Status code convention** (mirrors bespoke kernels):
//!   0 success
//!   2 invalid problem (null pointer, negative extent, etc.)
//!   5 internal cuDNN error (negated cuDNN status code passed up
//!     as `-status` in the lower bits, but pinned to `5` here for
//!     parity with the bespoke family)
//!
//! Workspace handling: when `workspace_bytes == 0` and a non-zero
//! workspace is required by cuDNN's algorithm selection, the wrapper
//! falls back to a transient `cudaMalloc` / `cudaFree` per launch.
//! This is a performance-hit-only fallback — callers performing
//! repeated launches should pre-size the workspace (query via the
//! Rust plan's `query_*_workspace_size` accessor on a sibling plan)
//! and pass it through.
//!
//! 1D conv pad-to-rank-4 internal detail: cuDNN's
//! `cudnnSetTensorNdDescriptor` / `cudnnSetFilterNdDescriptor` reject
//! `nb_dims < 4`. The 1D wrappers pad the rank-3 NCL shape to rank-4
//! NCLW with `W = 1` internally — the caller still passes rank-3
//! extents at the FFI boundary; the dummy axis is invisible.

#![cfg(feature = "cudnn")]

use core::ffi::c_void;
use core::ptr;

use crate::cudnn_ffi::{
    cudnnConvolutionBackwardData, cudnnConvolutionBackwardFilter, cudnnConvolutionDescriptor_t,
    cudnnConvolutionForward, cudnnCreate, cudnnCreateConvolutionDescriptor,
    cudnnCreateFilterDescriptor, cudnnCreateTensorDescriptor, cudnnDestroy,
    cudnnDestroyConvolutionDescriptor, cudnnDestroyFilterDescriptor, cudnnDestroyTensorDescriptor,
    cudnnFilterDescriptor_t, cudnnGetConvolutionBackwardDataWorkspaceSize,
    cudnnGetConvolutionBackwardFilterWorkspaceSize, cudnnGetConvolutionForwardWorkspaceSize,
    cudnnHandle_t, cudnnSetConvolution2dDescriptor, cudnnSetConvolutionGroupCount,
    cudnnSetConvolutionNdDescriptor, cudnnSetFilter4dDescriptor, cudnnSetFilterNdDescriptor,
    cudnnSetStream, cudnnSetTensor4dDescriptor, cudnnSetTensorNdDescriptor,
    cudnnTensorDescriptor_t, CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_BFLOAT16, CUDNN_DATA_DOUBLE, CUDNN_DATA_FLOAT,
    CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW,
};

// CUDA runtime — `cudaMalloc` / `cudaFree` for the workspace fallback.
unsafe extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(ptr: *mut c_void) -> i32;
}

/// Dtype tag for the cuDNN wrappers — used to fan a single
/// `_ <dtype>_run` symbol set out across `f32 / f64 / f16 / bf16`.
#[derive(Copy, Clone)]
enum DtypeTag {
    F32,
    F64,
    F16,
    Bf16,
}

impl DtypeTag {
    #[inline]
    fn cudnn_dtype(self) -> i32 {
        match self {
            DtypeTag::F32 => CUDNN_DATA_FLOAT,
            DtypeTag::F64 => CUDNN_DATA_DOUBLE,
            DtypeTag::F16 => CUDNN_DATA_HALF,
            DtypeTag::Bf16 => CUDNN_DATA_BFLOAT16,
        }
    }
    #[inline]
    fn is_double_compute(self) -> bool {
        matches!(self, DtypeTag::F64)
    }
}

/// Result codes returned by every wrapper.
const OK: i32 = 0;
const INVALID: i32 = 2;
const INTERNAL: i32 = 5;

// =============================================================================
// Direction tag — selects FW / BW-data / BW-filter dispatch inside
// the shared helpers. The descriptor + workspace queries follow the
// same path for FW / BW-data; only the exec call differs. BW-filter
// queries a separate workspace.
// =============================================================================

#[derive(Copy, Clone)]
enum Dir {
    Fw,
    BwData,
    BwFilter,
}

// =============================================================================
// Conv2d — 4d tensor descriptors via `cudnnSetTensor4dDescriptor`
// + `cudnnSetFilter4dDescriptor` + `cudnnSetConvolution2dDescriptor`.
// =============================================================================

#[inline]
fn compute_conv2d_out(d: &Conv2dParams) -> (i32, i32) {
    let h_eff = d.dilation_h * (d.h_filt - 1) + 1;
    let w_eff = d.dilation_w * (d.w_filt - 1) + 1;
    (
        (d.h_in + 2 * d.pad_h - h_eff) / d.stride_h + 1,
        (d.w_in + 2 * d.pad_w - w_eff) / d.stride_w + 1,
    )
}

#[derive(Copy, Clone)]
struct Conv2dParams {
    batch: i32,
    c_in: i32,
    c_out: i32,
    h_in: i32,
    w_in: i32,
    h_filt: i32,
    w_filt: i32,
    pad_h: i32,
    pad_w: i32,
    stride_h: i32,
    stride_w: i32,
    dilation_h: i32,
    dilation_w: i32,
    groups: i32,
}

/// Validate the Conv2d descriptor extents. Returns `INVALID` on any
/// rejection so the caller can short-circuit before touching cuDNN.
fn validate_conv2d_params(p: &Conv2dParams) -> i32 {
    if p.batch <= 0 || p.c_in <= 0 || p.h_in <= 0 || p.w_in <= 0 {
        return INVALID;
    }
    if p.c_out <= 0 || p.h_filt <= 0 || p.w_filt <= 0 {
        return INVALID;
    }
    if p.stride_h <= 0 || p.stride_w <= 0 || p.dilation_h <= 0 || p.dilation_w <= 0 {
        return INVALID;
    }
    if p.pad_h < 0 || p.pad_w < 0 || p.groups <= 0 {
        return INVALID;
    }
    if p.c_in % p.groups != 0 || p.c_out % p.groups != 0 {
        return INVALID;
    }
    let (h_out, w_out) = compute_conv2d_out(p);
    if h_out <= 0 || w_out <= 0 {
        return INVALID;
    }
    OK
}

/// RAII-style guard for the four cuDNN descriptors + handle. `drop`
/// destroys whatever was allocated (idempotent on nulls).
struct ConvDescGuard {
    handle: cudnnHandle_t,
    x_desc: cudnnTensorDescriptor_t,
    y_desc: cudnnTensorDescriptor_t,
    w_desc: cudnnFilterDescriptor_t,
    conv_desc: cudnnConvolutionDescriptor_t,
}

impl ConvDescGuard {
    fn new() -> Self {
        Self {
            handle: ptr::null_mut(),
            x_desc: ptr::null_mut(),
            y_desc: ptr::null_mut(),
            w_desc: ptr::null_mut(),
            conv_desc: ptr::null_mut(),
        }
    }
}

impl Drop for ConvDescGuard {
    fn drop(&mut self) {
        unsafe {
            if !self.conv_desc.is_null() {
                let _ = cudnnDestroyConvolutionDescriptor(self.conv_desc);
            }
            if !self.w_desc.is_null() {
                let _ = cudnnDestroyFilterDescriptor(self.w_desc);
            }
            if !self.y_desc.is_null() {
                let _ = cudnnDestroyTensorDescriptor(self.y_desc);
            }
            if !self.x_desc.is_null() {
                let _ = cudnnDestroyTensorDescriptor(self.x_desc);
            }
            if !self.handle.is_null() {
                let _ = cudnnDestroy(self.handle);
            }
        }
    }
}

/// Allocate handle + bind stream. Returns `INTERNAL` on cuDNN failure.
fn setup_handle(g: &mut ConvDescGuard, stream: *mut c_void) -> i32 {
    let s = unsafe { cudnnCreate(&mut g.handle as *mut _) };
    if s != 0 {
        return INTERNAL;
    }
    let s = unsafe { cudnnSetStream(g.handle, stream) };
    if s != 0 {
        return INTERNAL;
    }
    OK
}

/// Build the four Conv2d descriptors. Returns `INTERNAL` on cuDNN
/// failure (rare — usually means dim out of cuDNN's accepted range).
fn build_conv2d_descs(g: &mut ConvDescGuard, p: &Conv2dParams, dt: DtypeTag) -> i32 {
    let cudnn_dt = dt.cudnn_dtype();
    let compute_dt = if dt.is_double_compute() {
        CUDNN_DATA_DOUBLE
    } else {
        CUDNN_DATA_FLOAT
    };
    let (h_out, w_out) = compute_conv2d_out(p);
    let c_in_per_group = p.c_in / p.groups;

    // x
    let s = unsafe { cudnnCreateTensorDescriptor(&mut g.x_desc as *mut _) };
    if s != 0 {
        return INTERNAL;
    }
    let s = unsafe {
        cudnnSetTensor4dDescriptor(
            g.x_desc,
            CUDNN_TENSOR_NCHW,
            cudnn_dt,
            p.batch,
            p.c_in,
            p.h_in,
            p.w_in,
        )
    };
    if s != 0 {
        return INTERNAL;
    }

    // y
    let s = unsafe { cudnnCreateTensorDescriptor(&mut g.y_desc as *mut _) };
    if s != 0 {
        return INTERNAL;
    }
    let s = unsafe {
        cudnnSetTensor4dDescriptor(
            g.y_desc,
            CUDNN_TENSOR_NCHW,
            cudnn_dt,
            p.batch,
            p.c_out,
            h_out,
            w_out,
        )
    };
    if s != 0 {
        return INTERNAL;
    }

    // w
    let s = unsafe { cudnnCreateFilterDescriptor(&mut g.w_desc as *mut _) };
    if s != 0 {
        return INTERNAL;
    }
    let s = unsafe {
        cudnnSetFilter4dDescriptor(
            g.w_desc,
            cudnn_dt,
            CUDNN_TENSOR_NCHW,
            p.c_out,
            c_in_per_group,
            p.h_filt,
            p.w_filt,
        )
    };
    if s != 0 {
        return INTERNAL;
    }

    // conv
    let s = unsafe { cudnnCreateConvolutionDescriptor(&mut g.conv_desc as *mut _) };
    if s != 0 {
        return INTERNAL;
    }
    let s = unsafe {
        cudnnSetConvolution2dDescriptor(
            g.conv_desc,
            p.pad_h,
            p.pad_w,
            p.stride_h,
            p.stride_w,
            p.dilation_h,
            p.dilation_w,
            CUDNN_CROSS_CORRELATION,
            compute_dt,
        )
    };
    if s != 0 {
        return INTERNAL;
    }
    let s = unsafe { cudnnSetConvolutionGroupCount(g.conv_desc, p.groups) };
    if s != 0 {
        return INTERNAL;
    }
    OK
}

/// Query the per-direction workspace size for the four descriptors.
/// On failure returns `(INTERNAL, 0)`.
fn query_conv_ws(g: &ConvDescGuard, dir: Dir) -> (i32, usize) {
    let mut bytes: usize = 0;
    let s = match dir {
        Dir::Fw => unsafe {
            cudnnGetConvolutionForwardWorkspaceSize(
                g.handle,
                g.x_desc,
                g.w_desc,
                g.conv_desc,
                g.y_desc,
                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                &mut bytes as *mut usize,
            )
        },
        Dir::BwData => unsafe {
            cudnnGetConvolutionBackwardDataWorkspaceSize(
                g.handle,
                g.w_desc,
                g.y_desc,
                g.conv_desc,
                g.x_desc,
                CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
                &mut bytes as *mut usize,
            )
        },
        Dir::BwFilter => unsafe {
            cudnnGetConvolutionBackwardFilterWorkspaceSize(
                g.handle,
                g.x_desc,
                g.y_desc,
                g.conv_desc,
                g.w_desc,
                CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
                &mut bytes as *mut usize,
            )
        },
    };
    if s != 0 {
        (INTERNAL, 0)
    } else {
        (OK, bytes)
    }
}

/// Workspace fallback — if caller passed null + 0 bytes and cuDNN
/// needs a non-empty workspace, allocate one transiently via
/// `cudaMalloc`. The returned `WsHolder` frees on drop.
struct WsHolder {
    ptr: *mut c_void,
    owned: bool,
}

impl WsHolder {
    fn ensure(caller_ptr: *mut c_void, caller_bytes: usize, needed: usize) -> (Self, i32) {
        if needed == 0 {
            return (
                WsHolder {
                    ptr: ptr::null_mut(),
                    owned: false,
                },
                OK,
            );
        }
        if !caller_ptr.is_null() && caller_bytes >= needed {
            return (
                WsHolder {
                    ptr: caller_ptr,
                    owned: false,
                },
                OK,
            );
        }
        // Fall back to cudaMalloc.
        let mut p: *mut c_void = ptr::null_mut();
        let s = unsafe { cudaMalloc(&mut p as *mut _, needed) };
        if s != 0 || p.is_null() {
            return (
                WsHolder {
                    ptr: ptr::null_mut(),
                    owned: false,
                },
                INTERNAL,
            );
        }
        (WsHolder { ptr: p, owned: true }, OK)
    }
}

impl Drop for WsHolder {
    fn drop(&mut self) {
        if self.owned && !self.ptr.is_null() {
            unsafe {
                let _ = cudaFree(self.ptr);
            }
        }
    }
}

/// Dispatch a Conv2d FW. `x`, `w`, `y` are device pointers; ws is
/// optional. Returns OK / INVALID / INTERNAL.
#[allow(clippy::too_many_arguments)]
fn run_conv2d_fw(
    p: &Conv2dParams,
    dt: DtypeTag,
    x: *const c_void,
    w: *const c_void,
    y: *mut c_void,
    ws_ptr: *mut c_void,
    ws_bytes: usize,
    stream: *mut c_void,
) -> i32 {
    if x.is_null() || w.is_null() || y.is_null() {
        return INVALID;
    }
    let v = validate_conv2d_params(p);
    if v != OK {
        return v;
    }
    let mut g = ConvDescGuard::new();
    let s = setup_handle(&mut g, stream);
    if s != OK {
        return s;
    }
    let s = build_conv2d_descs(&mut g, p, dt);
    if s != OK {
        return s;
    }
    let (s, needed) = query_conv_ws(&g, Dir::Fw);
    if s != OK {
        return s;
    }
    let (ws, s) = WsHolder::ensure(ws_ptr, ws_bytes, needed);
    if s != OK {
        return s;
    }
    let status = if dt.is_double_compute() {
        let alpha: f64 = 1.0;
        let beta: f64 = 0.0;
        unsafe {
            cudnnConvolutionForward(
                g.handle,
                &alpha as *const f64 as *const c_void,
                g.x_desc,
                x,
                g.w_desc,
                w,
                g.conv_desc,
                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                ws.ptr,
                needed,
                &beta as *const f64 as *const c_void,
                g.y_desc,
                y,
            )
        }
    } else {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        unsafe {
            cudnnConvolutionForward(
                g.handle,
                &alpha as *const f32 as *const c_void,
                g.x_desc,
                x,
                g.w_desc,
                w,
                g.conv_desc,
                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                ws.ptr,
                needed,
                &beta as *const f32 as *const c_void,
                g.y_desc,
                y,
            )
        }
    };
    if status != 0 {
        return INTERNAL;
    }
    OK
}

#[allow(clippy::too_many_arguments)]
fn run_conv2d_bw_data(
    p: &Conv2dParams,
    dt: DtypeTag,
    w: *const c_void,
    dy: *const c_void,
    dx: *mut c_void,
    ws_ptr: *mut c_void,
    ws_bytes: usize,
    stream: *mut c_void,
) -> i32 {
    if w.is_null() || dy.is_null() || dx.is_null() {
        return INVALID;
    }
    let v = validate_conv2d_params(p);
    if v != OK {
        return v;
    }
    let mut g = ConvDescGuard::new();
    let s = setup_handle(&mut g, stream);
    if s != OK {
        return s;
    }
    let s = build_conv2d_descs(&mut g, p, dt);
    if s != OK {
        return s;
    }
    let (s, needed) = query_conv_ws(&g, Dir::BwData);
    if s != OK {
        return s;
    }
    let (ws, s) = WsHolder::ensure(ws_ptr, ws_bytes, needed);
    if s != OK {
        return s;
    }
    let status = if dt.is_double_compute() {
        let alpha: f64 = 1.0;
        let beta: f64 = 0.0;
        unsafe {
            cudnnConvolutionBackwardData(
                g.handle,
                &alpha as *const f64 as *const c_void,
                g.w_desc,
                w,
                g.y_desc,
                dy,
                g.conv_desc,
                CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
                ws.ptr,
                needed,
                &beta as *const f64 as *const c_void,
                g.x_desc,
                dx,
            )
        }
    } else {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        unsafe {
            cudnnConvolutionBackwardData(
                g.handle,
                &alpha as *const f32 as *const c_void,
                g.w_desc,
                w,
                g.y_desc,
                dy,
                g.conv_desc,
                CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
                ws.ptr,
                needed,
                &beta as *const f32 as *const c_void,
                g.x_desc,
                dx,
            )
        }
    };
    if status != 0 {
        return INTERNAL;
    }
    OK
}

#[allow(clippy::too_many_arguments)]
fn run_conv2d_bw_filter(
    p: &Conv2dParams,
    dt: DtypeTag,
    x: *const c_void,
    dy: *const c_void,
    dw: *mut c_void,
    ws_ptr: *mut c_void,
    ws_bytes: usize,
    stream: *mut c_void,
) -> i32 {
    if x.is_null() || dy.is_null() || dw.is_null() {
        return INVALID;
    }
    let v = validate_conv2d_params(p);
    if v != OK {
        return v;
    }
    let mut g = ConvDescGuard::new();
    let s = setup_handle(&mut g, stream);
    if s != OK {
        return s;
    }
    let s = build_conv2d_descs(&mut g, p, dt);
    if s != OK {
        return s;
    }
    let (s, needed) = query_conv_ws(&g, Dir::BwFilter);
    if s != OK {
        return s;
    }
    let (ws, s) = WsHolder::ensure(ws_ptr, ws_bytes, needed);
    if s != OK {
        return s;
    }
    let status = if dt.is_double_compute() {
        let alpha: f64 = 1.0;
        let beta: f64 = 0.0;
        unsafe {
            cudnnConvolutionBackwardFilter(
                g.handle,
                &alpha as *const f64 as *const c_void,
                g.x_desc,
                x,
                g.y_desc,
                dy,
                g.conv_desc,
                CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
                ws.ptr,
                needed,
                &beta as *const f64 as *const c_void,
                g.w_desc,
                dw,
            )
        }
    } else {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        unsafe {
            cudnnConvolutionBackwardFilter(
                g.handle,
                &alpha as *const f32 as *const c_void,
                g.x_desc,
                x,
                g.y_desc,
                dy,
                g.conv_desc,
                CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
                ws.ptr,
                needed,
                &beta as *const f32 as *const c_void,
                g.w_desc,
                dw,
            )
        }
    };
    if status != 0 {
        return INTERNAL;
    }
    OK
}

// =============================================================================
// Conv2d C-ABI surface (12 symbols: 3 dirs × 4 dtypes).
//
// Signature template per the Phase 19.2 plan:
//
//   baracuda_kernels_conv_2d_fw_<dtype>_run(
//       batch, c_in, c_out, h_in, w_in, h_out, w_out,
//       kh, kw, stride_h, stride_w, pad_h, pad_w,
//       dilation_h, dilation_w, groups,
//       input, filter, output, stream) -> i32
//
// Note: `h_out` / `w_out` are passed but currently unused (cuDNN
// recomputes from pad/stride/dilation). They're in the signature so
// downstream wrappers don't have to re-implement the formula.
// =============================================================================

/// Conv2d forward, f32. See module docs for argument semantics.
///
/// # Safety
/// All pointers must be live device memory; `stream` must be a valid
/// CUDA stream (or null for the default stream).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn baracuda_kernels_conv_2d_fw_f32_run(
    batch: i32, c_in: i32, c_out: i32,
    h_in: i32, w_in: i32, _h_out: i32, _w_out: i32,
    kh: i32, kw: i32,
    stride_h: i32, stride_w: i32,
    pad_h: i32, pad_w: i32,
    dilation_h: i32, dilation_w: i32,
    groups: i32,
    input: *const c_void,
    filter: *const c_void,
    output: *mut c_void,
    workspace: *mut c_void,
    workspace_bytes: usize,
    stream: *mut c_void,
) -> i32 {
    let p = Conv2dParams {
        batch, c_in, c_out, h_in, w_in,
        h_filt: kh, w_filt: kw,
        pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, groups,
    };
    run_conv2d_fw(&p, DtypeTag::F32, input, filter, output, workspace, workspace_bytes, stream)
}

// Macro factory for the remaining Conv2d FW dtype variants.
macro_rules! conv2d_fw_impl {
    ($name:ident, $dt:expr) => {
        #[doc = "Conv2d forward. See [`baracuda_kernels_conv_2d_fw_f32_run`]."]
        #[doc = ""]
        #[doc = "# Safety"]
        #[doc = "As [`baracuda_kernels_conv_2d_fw_f32_run`]."]
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            batch: i32, c_in: i32, c_out: i32,
            h_in: i32, w_in: i32, _h_out: i32, _w_out: i32,
            kh: i32, kw: i32,
            stride_h: i32, stride_w: i32,
            pad_h: i32, pad_w: i32,
            dilation_h: i32, dilation_w: i32,
            groups: i32,
            input: *const c_void,
            filter: *const c_void,
            output: *mut c_void,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32 {
            let p = Conv2dParams {
                batch, c_in, c_out, h_in, w_in,
                h_filt: kh, w_filt: kw,
                pad_h, pad_w, stride_h, stride_w,
                dilation_h, dilation_w, groups,
            };
            run_conv2d_fw(&p, $dt, input, filter, output, workspace, workspace_bytes, stream)
        }
    };
}

conv2d_fw_impl!(baracuda_kernels_conv_2d_fw_f64_run, DtypeTag::F64);
conv2d_fw_impl!(baracuda_kernels_conv_2d_fw_f16_run, DtypeTag::F16);
conv2d_fw_impl!(baracuda_kernels_conv_2d_fw_bf16_run, DtypeTag::Bf16);

macro_rules! conv2d_bw_data_impl {
    ($name:ident, $dt:expr) => {
        /// Conv2d backward-data. `w` and `dy` are inputs; `dx` is the
        /// output gradient (fully overwritten, beta = 0).
        ///
        /// # Safety
        /// As [`baracuda_kernels_conv_2d_fw_f32_run`].
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            batch: i32, c_in: i32, c_out: i32,
            h_in: i32, w_in: i32, _h_out: i32, _w_out: i32,
            kh: i32, kw: i32,
            stride_h: i32, stride_w: i32,
            pad_h: i32, pad_w: i32,
            dilation_h: i32, dilation_w: i32,
            groups: i32,
            filter: *const c_void,
            grad_output: *const c_void,
            grad_input: *mut c_void,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32 {
            let p = Conv2dParams {
                batch, c_in, c_out, h_in, w_in,
                h_filt: kh, w_filt: kw,
                pad_h, pad_w, stride_h, stride_w,
                dilation_h, dilation_w, groups,
            };
            run_conv2d_bw_data(&p, $dt, filter, grad_output, grad_input, workspace, workspace_bytes, stream)
        }
    };
}

conv2d_bw_data_impl!(baracuda_kernels_conv_2d_bw_data_f32_run, DtypeTag::F32);
conv2d_bw_data_impl!(baracuda_kernels_conv_2d_bw_data_f64_run, DtypeTag::F64);
conv2d_bw_data_impl!(baracuda_kernels_conv_2d_bw_data_f16_run, DtypeTag::F16);
conv2d_bw_data_impl!(baracuda_kernels_conv_2d_bw_data_bf16_run, DtypeTag::Bf16);

macro_rules! conv2d_bw_filter_impl {
    ($name:ident, $dt:expr) => {
        /// Conv2d backward-filter. `x` and `dy` are inputs; `dw` is
        /// the output gradient (fully overwritten, beta = 0).
        ///
        /// # Safety
        /// As [`baracuda_kernels_conv_2d_fw_f32_run`].
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            batch: i32, c_in: i32, c_out: i32,
            h_in: i32, w_in: i32, _h_out: i32, _w_out: i32,
            kh: i32, kw: i32,
            stride_h: i32, stride_w: i32,
            pad_h: i32, pad_w: i32,
            dilation_h: i32, dilation_w: i32,
            groups: i32,
            input: *const c_void,
            grad_output: *const c_void,
            grad_filter: *mut c_void,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32 {
            let p = Conv2dParams {
                batch, c_in, c_out, h_in, w_in,
                h_filt: kh, w_filt: kw,
                pad_h, pad_w, stride_h, stride_w,
                dilation_h, dilation_w, groups,
            };
            run_conv2d_bw_filter(&p, $dt, input, grad_output, grad_filter, workspace, workspace_bytes, stream)
        }
    };
}

conv2d_bw_filter_impl!(baracuda_kernels_conv_2d_bw_filter_f32_run, DtypeTag::F32);
conv2d_bw_filter_impl!(baracuda_kernels_conv_2d_bw_filter_f64_run, DtypeTag::F64);
conv2d_bw_filter_impl!(baracuda_kernels_conv_2d_bw_filter_f16_run, DtypeTag::F16);
conv2d_bw_filter_impl!(baracuda_kernels_conv_2d_bw_filter_bf16_run, DtypeTag::Bf16);

// =============================================================================
// Conv1d — rank-3 NCL extents at the FFI boundary, internally padded
// to rank-4 NCLW with W = 1 + array_length = 2 to satisfy cuDNN's
// `cudnnSetTensorNdDescriptor` / `cudnnSetFilterNdDescriptor`
// (which reject `nb_dims < 4`).
// =============================================================================

#[derive(Copy, Clone)]
struct Conv1dParams {
    batch: i32,
    c_in: i32,
    c_out: i32,
    l_in: i32,
    l_filt: i32,
    pad_l: i32,
    stride_l: i32,
    dilation_l: i32,
    groups: i32,
}

#[inline]
fn compute_conv1d_out(p: &Conv1dParams) -> i32 {
    let eff = p.dilation_l * (p.l_filt - 1) + 1;
    (p.l_in + 2 * p.pad_l - eff) / p.stride_l + 1
}

fn validate_conv1d_params(p: &Conv1dParams) -> i32 {
    if p.batch <= 0 || p.c_in <= 0 || p.l_in <= 0 {
        return INVALID;
    }
    if p.c_out <= 0 || p.l_filt <= 0 {
        return INVALID;
    }
    if p.stride_l <= 0 || p.dilation_l <= 0 || p.pad_l < 0 || p.groups <= 0 {
        return INVALID;
    }
    if p.c_in % p.groups != 0 || p.c_out % p.groups != 0 {
        return INVALID;
    }
    if compute_conv1d_out(p) <= 0 {
        return INVALID;
    }
    OK
}

/// Build the Conv1d descriptors as rank-4 NCLW with `W = 1`.
fn build_conv1d_descs(g: &mut ConvDescGuard, p: &Conv1dParams, dt: DtypeTag) -> i32 {
    let cudnn_dt = dt.cudnn_dtype();
    let compute_dt = if dt.is_double_compute() {
        CUDNN_DATA_DOUBLE
    } else {
        CUDNN_DATA_FLOAT
    };
    let l_out = compute_conv1d_out(p);
    let c_in_per_group = p.c_in / p.groups;

    // x — rank-4 [N, C_in, L_in, 1]
    let s = unsafe { cudnnCreateTensorDescriptor(&mut g.x_desc as *mut _) };
    if s != 0 {
        return INTERNAL;
    }
    let x_dims = [p.batch, p.c_in, p.l_in, 1];
    let x_strides = [p.c_in * p.l_in, p.l_in, 1, 1];
    let s = unsafe {
        cudnnSetTensorNdDescriptor(g.x_desc, cudnn_dt, 4, x_dims.as_ptr(), x_strides.as_ptr())
    };
    if s != 0 {
        return INTERNAL;
    }

    // y — rank-4 [N, C_out, L_out, 1]
    let s = unsafe { cudnnCreateTensorDescriptor(&mut g.y_desc as *mut _) };
    if s != 0 {
        return INTERNAL;
    }
    let y_dims = [p.batch, p.c_out, l_out, 1];
    let y_strides = [p.c_out * l_out, l_out, 1, 1];
    let s = unsafe {
        cudnnSetTensorNdDescriptor(g.y_desc, cudnn_dt, 4, y_dims.as_ptr(), y_strides.as_ptr())
    };
    if s != 0 {
        return INTERNAL;
    }

    // w — rank-4 [C_out, C_in/groups, L_filt, 1]
    let s = unsafe { cudnnCreateFilterDescriptor(&mut g.w_desc as *mut _) };
    if s != 0 {
        return INTERNAL;
    }
    let w_dims = [p.c_out, c_in_per_group, p.l_filt, 1];
    let s = unsafe {
        cudnnSetFilterNdDescriptor(g.w_desc, cudnn_dt, CUDNN_TENSOR_NCHW, 4, w_dims.as_ptr())
    };
    if s != 0 {
        return INTERNAL;
    }

    // conv — array_length = 2 (spatial rank including the dummy axis).
    let s = unsafe { cudnnCreateConvolutionDescriptor(&mut g.conv_desc as *mut _) };
    if s != 0 {
        return INTERNAL;
    }
    let pad_a = [p.pad_l, 0];
    let stride_a = [p.stride_l, 1];
    let dilation_a = [p.dilation_l, 1];
    let s = unsafe {
        cudnnSetConvolutionNdDescriptor(
            g.conv_desc,
            2,
            pad_a.as_ptr(),
            stride_a.as_ptr(),
            dilation_a.as_ptr(),
            CUDNN_CROSS_CORRELATION,
            compute_dt,
        )
    };
    if s != 0 {
        return INTERNAL;
    }
    let s = unsafe { cudnnSetConvolutionGroupCount(g.conv_desc, p.groups) };
    if s != 0 {
        return INTERNAL;
    }
    OK
}

#[allow(clippy::too_many_arguments)]
fn run_conv1d_dispatch(
    p: &Conv1dParams,
    dt: DtypeTag,
    dir: Dir,
    a: *const c_void,
    b: *const c_void,
    c: *mut c_void,
    ws_ptr: *mut c_void,
    ws_bytes: usize,
    stream: *mut c_void,
) -> i32 {
    if a.is_null() || b.is_null() || c.is_null() {
        return INVALID;
    }
    let v = validate_conv1d_params(p);
    if v != OK {
        return v;
    }
    let mut g = ConvDescGuard::new();
    let s = setup_handle(&mut g, stream);
    if s != OK {
        return s;
    }
    let s = build_conv1d_descs(&mut g, p, dt);
    if s != OK {
        return s;
    }
    let (s, needed) = query_conv_ws(&g, dir);
    if s != OK {
        return s;
    }
    let (ws, s) = WsHolder::ensure(ws_ptr, ws_bytes, needed);
    if s != OK {
        return s;
    }
    // alpha / beta scalars match the compute dtype.
    let status = if dt.is_double_compute() {
        let alpha: f64 = 1.0;
        let beta: f64 = 0.0;
        let ap = &alpha as *const f64 as *const c_void;
        let bp = &beta as *const f64 as *const c_void;
        match dir {
            Dir::Fw => unsafe {
                cudnnConvolutionForward(
                    g.handle, ap, g.x_desc, a, g.w_desc, b, g.conv_desc,
                    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, ws.ptr, needed, bp, g.y_desc, c,
                )
            },
            Dir::BwData => unsafe {
                cudnnConvolutionBackwardData(
                    g.handle, ap, g.w_desc, a, g.y_desc, b, g.conv_desc,
                    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, ws.ptr, needed, bp, g.x_desc, c,
                )
            },
            Dir::BwFilter => unsafe {
                cudnnConvolutionBackwardFilter(
                    g.handle, ap, g.x_desc, a, g.y_desc, b, g.conv_desc,
                    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, ws.ptr, needed, bp, g.w_desc, c,
                )
            },
        }
    } else {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let ap = &alpha as *const f32 as *const c_void;
        let bp = &beta as *const f32 as *const c_void;
        match dir {
            Dir::Fw => unsafe {
                cudnnConvolutionForward(
                    g.handle, ap, g.x_desc, a, g.w_desc, b, g.conv_desc,
                    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, ws.ptr, needed, bp, g.y_desc, c,
                )
            },
            Dir::BwData => unsafe {
                cudnnConvolutionBackwardData(
                    g.handle, ap, g.w_desc, a, g.y_desc, b, g.conv_desc,
                    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, ws.ptr, needed, bp, g.x_desc, c,
                )
            },
            Dir::BwFilter => unsafe {
                cudnnConvolutionBackwardFilter(
                    g.handle, ap, g.x_desc, a, g.y_desc, b, g.conv_desc,
                    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, ws.ptr, needed, bp, g.w_desc, c,
                )
            },
        }
    };
    if status != 0 {
        return INTERNAL;
    }
    OK
}

macro_rules! conv1d_fw_impl {
    ($name:ident, $dt:expr) => {
        /// Conv1d forward. Activation is rank-3 `[N, C_in, L_in]`;
        /// filter is rank-3 `[C_out, C_in/groups, L_filt]`. The
        /// wrapper internally pads to cuDNN's rank-4 NCLW with W=1.
        ///
        /// # Safety
        /// All pointers must be live device memory; `stream` valid.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            batch: i32, c_in: i32, c_out: i32,
            l_in: i32, _l_out: i32,
            l_filt: i32,
            stride_l: i32, pad_l: i32, dilation_l: i32,
            groups: i32,
            input: *const c_void,
            filter: *const c_void,
            output: *mut c_void,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32 {
            let p = Conv1dParams {
                batch, c_in, c_out, l_in, l_filt,
                pad_l, stride_l, dilation_l, groups,
            };
            run_conv1d_dispatch(
                &p, $dt, Dir::Fw,
                input, filter, output,
                workspace, workspace_bytes, stream,
            )
        }
    };
}

macro_rules! conv1d_bw_data_impl {
    ($name:ident, $dt:expr) => {
        /// Conv1d backward-data. See [`baracuda_kernels_conv_2d_fw_f32_run`].
        ///
        /// # Safety: as the Conv1d FW.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            batch: i32, c_in: i32, c_out: i32,
            l_in: i32, _l_out: i32,
            l_filt: i32,
            stride_l: i32, pad_l: i32, dilation_l: i32,
            groups: i32,
            filter: *const c_void,
            grad_output: *const c_void,
            grad_input: *mut c_void,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32 {
            let p = Conv1dParams {
                batch, c_in, c_out, l_in, l_filt,
                pad_l, stride_l, dilation_l, groups,
            };
            run_conv1d_dispatch(
                &p, $dt, Dir::BwData,
                filter, grad_output, grad_input,
                workspace, workspace_bytes, stream,
            )
        }
    };
}

macro_rules! conv1d_bw_filter_impl {
    ($name:ident, $dt:expr) => {
        /// Conv1d backward-filter. See [`baracuda_kernels_conv_2d_fw_f32_run`].
        ///
        /// # Safety: as the Conv1d FW.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            batch: i32, c_in: i32, c_out: i32,
            l_in: i32, _l_out: i32,
            l_filt: i32,
            stride_l: i32, pad_l: i32, dilation_l: i32,
            groups: i32,
            input: *const c_void,
            grad_output: *const c_void,
            grad_filter: *mut c_void,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32 {
            let p = Conv1dParams {
                batch, c_in, c_out, l_in, l_filt,
                pad_l, stride_l, dilation_l, groups,
            };
            run_conv1d_dispatch(
                &p, $dt, Dir::BwFilter,
                input, grad_output, grad_filter,
                workspace, workspace_bytes, stream,
            )
        }
    };
}

conv1d_fw_impl!(baracuda_kernels_conv_1d_fw_f32_run, DtypeTag::F32);
conv1d_fw_impl!(baracuda_kernels_conv_1d_fw_f64_run, DtypeTag::F64);
conv1d_fw_impl!(baracuda_kernels_conv_1d_fw_f16_run, DtypeTag::F16);
conv1d_fw_impl!(baracuda_kernels_conv_1d_fw_bf16_run, DtypeTag::Bf16);
conv1d_bw_data_impl!(baracuda_kernels_conv_1d_bw_data_f32_run, DtypeTag::F32);
conv1d_bw_data_impl!(baracuda_kernels_conv_1d_bw_data_f64_run, DtypeTag::F64);
conv1d_bw_data_impl!(baracuda_kernels_conv_1d_bw_data_f16_run, DtypeTag::F16);
conv1d_bw_data_impl!(baracuda_kernels_conv_1d_bw_data_bf16_run, DtypeTag::Bf16);
conv1d_bw_filter_impl!(baracuda_kernels_conv_1d_bw_filter_f32_run, DtypeTag::F32);
conv1d_bw_filter_impl!(baracuda_kernels_conv_1d_bw_filter_f64_run, DtypeTag::F64);
conv1d_bw_filter_impl!(baracuda_kernels_conv_1d_bw_filter_f16_run, DtypeTag::F16);
conv1d_bw_filter_impl!(baracuda_kernels_conv_1d_bw_filter_bf16_run, DtypeTag::Bf16);

// =============================================================================
// Conv3d — rank-5 NCDHW; cuDNN's NdDescriptor APIs handle 3D natively
// (`array_length = 3`).
// =============================================================================

#[derive(Copy, Clone)]
struct Conv3dParams {
    batch: i32,
    c_in: i32,
    c_out: i32,
    d_in: i32, h_in: i32, w_in: i32,
    d_filt: i32, h_filt: i32, w_filt: i32,
    pad_d: i32, pad_h: i32, pad_w: i32,
    stride_d: i32, stride_h: i32, stride_w: i32,
    dilation_d: i32, dilation_h: i32, dilation_w: i32,
    groups: i32,
}

#[inline]
fn compute_conv3d_out(p: &Conv3dParams) -> (i32, i32, i32) {
    let d_eff = p.dilation_d * (p.d_filt - 1) + 1;
    let h_eff = p.dilation_h * (p.h_filt - 1) + 1;
    let w_eff = p.dilation_w * (p.w_filt - 1) + 1;
    (
        (p.d_in + 2 * p.pad_d - d_eff) / p.stride_d + 1,
        (p.h_in + 2 * p.pad_h - h_eff) / p.stride_h + 1,
        (p.w_in + 2 * p.pad_w - w_eff) / p.stride_w + 1,
    )
}

fn validate_conv3d_params(p: &Conv3dParams) -> i32 {
    if p.batch <= 0 || p.c_in <= 0 || p.d_in <= 0 || p.h_in <= 0 || p.w_in <= 0 {
        return INVALID;
    }
    if p.c_out <= 0 || p.d_filt <= 0 || p.h_filt <= 0 || p.w_filt <= 0 {
        return INVALID;
    }
    if p.stride_d <= 0 || p.stride_h <= 0 || p.stride_w <= 0 {
        return INVALID;
    }
    if p.dilation_d <= 0 || p.dilation_h <= 0 || p.dilation_w <= 0 {
        return INVALID;
    }
    if p.pad_d < 0 || p.pad_h < 0 || p.pad_w < 0 || p.groups <= 0 {
        return INVALID;
    }
    if p.c_in % p.groups != 0 || p.c_out % p.groups != 0 {
        return INVALID;
    }
    let (d_out, h_out, w_out) = compute_conv3d_out(p);
    if d_out <= 0 || h_out <= 0 || w_out <= 0 {
        return INVALID;
    }
    OK
}

fn build_conv3d_descs(g: &mut ConvDescGuard, p: &Conv3dParams, dt: DtypeTag) -> i32 {
    let cudnn_dt = dt.cudnn_dtype();
    let compute_dt = if dt.is_double_compute() {
        CUDNN_DATA_DOUBLE
    } else {
        CUDNN_DATA_FLOAT
    };
    let (d_out, h_out, w_out) = compute_conv3d_out(p);
    let c_in_per_group = p.c_in / p.groups;

    // x — rank-5 NCDHW.
    let s = unsafe { cudnnCreateTensorDescriptor(&mut g.x_desc as *mut _) };
    if s != 0 {
        return INTERNAL;
    }
    let x_dims = [p.batch, p.c_in, p.d_in, p.h_in, p.w_in];
    let s_w = 1;
    let s_h = p.w_in;
    let s_d = p.h_in * p.w_in;
    let s_c = p.d_in * p.h_in * p.w_in;
    let s_n = p.c_in * s_c;
    let x_strides = [s_n, s_c, s_d, s_h, s_w];
    let s = unsafe {
        cudnnSetTensorNdDescriptor(g.x_desc, cudnn_dt, 5, x_dims.as_ptr(), x_strides.as_ptr())
    };
    if s != 0 {
        return INTERNAL;
    }

    // y — rank-5 NCDHW with computed output extents.
    let s = unsafe { cudnnCreateTensorDescriptor(&mut g.y_desc as *mut _) };
    if s != 0 {
        return INTERNAL;
    }
    let y_dims = [p.batch, p.c_out, d_out, h_out, w_out];
    let y_s_w = 1;
    let y_s_h = w_out;
    let y_s_d = h_out * w_out;
    let y_s_c = d_out * h_out * w_out;
    let y_s_n = p.c_out * y_s_c;
    let y_strides = [y_s_n, y_s_c, y_s_d, y_s_h, y_s_w];
    let s = unsafe {
        cudnnSetTensorNdDescriptor(g.y_desc, cudnn_dt, 5, y_dims.as_ptr(), y_strides.as_ptr())
    };
    if s != 0 {
        return INTERNAL;
    }

    // w — rank-5 [C_out, C_in/groups, D, H, W].
    let s = unsafe { cudnnCreateFilterDescriptor(&mut g.w_desc as *mut _) };
    if s != 0 {
        return INTERNAL;
    }
    let w_dims = [p.c_out, c_in_per_group, p.d_filt, p.h_filt, p.w_filt];
    let s = unsafe {
        cudnnSetFilterNdDescriptor(g.w_desc, cudnn_dt, CUDNN_TENSOR_NCHW, 5, w_dims.as_ptr())
    };
    if s != 0 {
        return INTERNAL;
    }

    // conv — array_length = 3.
    let s = unsafe { cudnnCreateConvolutionDescriptor(&mut g.conv_desc as *mut _) };
    if s != 0 {
        return INTERNAL;
    }
    let pad_a = [p.pad_d, p.pad_h, p.pad_w];
    let stride_a = [p.stride_d, p.stride_h, p.stride_w];
    let dil_a = [p.dilation_d, p.dilation_h, p.dilation_w];
    let s = unsafe {
        cudnnSetConvolutionNdDescriptor(
            g.conv_desc,
            3,
            pad_a.as_ptr(),
            stride_a.as_ptr(),
            dil_a.as_ptr(),
            CUDNN_CROSS_CORRELATION,
            compute_dt,
        )
    };
    if s != 0 {
        return INTERNAL;
    }
    let s = unsafe { cudnnSetConvolutionGroupCount(g.conv_desc, p.groups) };
    if s != 0 {
        return INTERNAL;
    }
    OK
}

#[allow(clippy::too_many_arguments)]
fn run_conv3d_dispatch(
    p: &Conv3dParams,
    dt: DtypeTag,
    dir: Dir,
    a: *const c_void,
    b: *const c_void,
    c: *mut c_void,
    ws_ptr: *mut c_void,
    ws_bytes: usize,
    stream: *mut c_void,
) -> i32 {
    if a.is_null() || b.is_null() || c.is_null() {
        return INVALID;
    }
    let v = validate_conv3d_params(p);
    if v != OK {
        return v;
    }
    let mut g = ConvDescGuard::new();
    let s = setup_handle(&mut g, stream);
    if s != OK {
        return s;
    }
    let s = build_conv3d_descs(&mut g, p, dt);
    if s != OK {
        return s;
    }
    let (s, needed) = query_conv_ws(&g, dir);
    if s != OK {
        return s;
    }
    let (ws, s) = WsHolder::ensure(ws_ptr, ws_bytes, needed);
    if s != OK {
        return s;
    }
    let status = if dt.is_double_compute() {
        let alpha: f64 = 1.0;
        let beta: f64 = 0.0;
        let ap = &alpha as *const f64 as *const c_void;
        let bp = &beta as *const f64 as *const c_void;
        match dir {
            Dir::Fw => unsafe {
                cudnnConvolutionForward(
                    g.handle, ap, g.x_desc, a, g.w_desc, b, g.conv_desc,
                    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, ws.ptr, needed, bp, g.y_desc, c,
                )
            },
            Dir::BwData => unsafe {
                cudnnConvolutionBackwardData(
                    g.handle, ap, g.w_desc, a, g.y_desc, b, g.conv_desc,
                    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, ws.ptr, needed, bp, g.x_desc, c,
                )
            },
            Dir::BwFilter => unsafe {
                cudnnConvolutionBackwardFilter(
                    g.handle, ap, g.x_desc, a, g.y_desc, b, g.conv_desc,
                    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, ws.ptr, needed, bp, g.w_desc, c,
                )
            },
        }
    } else {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let ap = &alpha as *const f32 as *const c_void;
        let bp = &beta as *const f32 as *const c_void;
        match dir {
            Dir::Fw => unsafe {
                cudnnConvolutionForward(
                    g.handle, ap, g.x_desc, a, g.w_desc, b, g.conv_desc,
                    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, ws.ptr, needed, bp, g.y_desc, c,
                )
            },
            Dir::BwData => unsafe {
                cudnnConvolutionBackwardData(
                    g.handle, ap, g.w_desc, a, g.y_desc, b, g.conv_desc,
                    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, ws.ptr, needed, bp, g.x_desc, c,
                )
            },
            Dir::BwFilter => unsafe {
                cudnnConvolutionBackwardFilter(
                    g.handle, ap, g.x_desc, a, g.y_desc, b, g.conv_desc,
                    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, ws.ptr, needed, bp, g.w_desc, c,
                )
            },
        }
    };
    if status != 0 {
        return INTERNAL;
    }
    OK
}

macro_rules! conv3d_dir_impl {
    ($name:ident, $dt:expr, $dir:expr, $a:ident, $b:ident, $c:ident) => {
        /// Conv3d direction.
        ///
        /// # Safety: all pointers must be live device memory; `stream` valid.
        #[unsafe(no_mangle)]
        #[allow(clippy::too_many_arguments)]
        pub unsafe extern "C" fn $name(
            batch: i32, c_in: i32, c_out: i32,
            d_in: i32, h_in: i32, w_in: i32,
            _d_out: i32, _h_out: i32, _w_out: i32,
            kd: i32, kh: i32, kw: i32,
            stride_d: i32, stride_h: i32, stride_w: i32,
            pad_d: i32, pad_h: i32, pad_w: i32,
            dilation_d: i32, dilation_h: i32, dilation_w: i32,
            groups: i32,
            $a: *const c_void,
            $b: *const c_void,
            $c: *mut c_void,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32 {
            let p = Conv3dParams {
                batch, c_in, c_out,
                d_in, h_in, w_in,
                d_filt: kd, h_filt: kh, w_filt: kw,
                pad_d, pad_h, pad_w,
                stride_d, stride_h, stride_w,
                dilation_d, dilation_h, dilation_w,
                groups,
            };
            run_conv3d_dispatch(
                &p, $dt, $dir,
                $a, $b, $c,
                workspace, workspace_bytes, stream,
            )
        }
    };
}

conv3d_dir_impl!(baracuda_kernels_conv_3d_fw_f32_run, DtypeTag::F32, Dir::Fw, input, filter, output);
conv3d_dir_impl!(baracuda_kernels_conv_3d_fw_f64_run, DtypeTag::F64, Dir::Fw, input, filter, output);
conv3d_dir_impl!(baracuda_kernels_conv_3d_fw_f16_run, DtypeTag::F16, Dir::Fw, input, filter, output);
conv3d_dir_impl!(baracuda_kernels_conv_3d_fw_bf16_run, DtypeTag::Bf16, Dir::Fw, input, filter, output);
conv3d_dir_impl!(baracuda_kernels_conv_3d_bw_data_f32_run, DtypeTag::F32, Dir::BwData, filter, grad_output, grad_input);
conv3d_dir_impl!(baracuda_kernels_conv_3d_bw_data_f64_run, DtypeTag::F64, Dir::BwData, filter, grad_output, grad_input);
conv3d_dir_impl!(baracuda_kernels_conv_3d_bw_data_f16_run, DtypeTag::F16, Dir::BwData, filter, grad_output, grad_input);
conv3d_dir_impl!(baracuda_kernels_conv_3d_bw_data_bf16_run, DtypeTag::Bf16, Dir::BwData, filter, grad_output, grad_input);
conv3d_dir_impl!(baracuda_kernels_conv_3d_bw_filter_f32_run, DtypeTag::F32, Dir::BwFilter, input, grad_output, grad_filter);
conv3d_dir_impl!(baracuda_kernels_conv_3d_bw_filter_f64_run, DtypeTag::F64, Dir::BwFilter, input, grad_output, grad_filter);
conv3d_dir_impl!(baracuda_kernels_conv_3d_bw_filter_f16_run, DtypeTag::F16, Dir::BwFilter, input, grad_output, grad_filter);
conv3d_dir_impl!(baracuda_kernels_conv_3d_bw_filter_bf16_run, DtypeTag::Bf16, Dir::BwFilter, input, grad_output, grad_filter);

// =============================================================================
// ConvTranspose family. cuDNN has no direct "transpose forward" entry
// point — we configure descriptors for a synthetic dense forward conv
// that maps the ConvTranspose's output to its input, then dispatch via
// role-swapped backward-data (FW) / forward (BW-data) / backward-filter
// (BW-filter, role-swapped). Matches the Rust plans' strategy exactly.
//
// Filter shape follows PyTorch: `[C_in, C_out/groups, ...]` — i.e.
// `K = C_in` in cuDNN's `[K, C/groups, ...]` filter convention.
// =============================================================================

#[derive(Copy, Clone)]
struct ConvT2dParams {
    batch: i32,
    c_in: i32,
    c_out: i32,
    h_in: i32, w_in: i32,
    h_filt: i32, w_filt: i32,
    pad_h: i32, pad_w: i32,
    stride_h: i32, stride_w: i32,
    dilation_h: i32, dilation_w: i32,
    output_pad_h: i32,
    output_pad_w: i32,
    groups: i32,
}

#[inline]
fn compute_convt2d_out(p: &ConvT2dParams) -> (i32, i32) {
    let h_out = (p.h_in - 1) * p.stride_h - 2 * p.pad_h
        + p.dilation_h * (p.h_filt - 1)
        + p.output_pad_h
        + 1;
    let w_out = (p.w_in - 1) * p.stride_w - 2 * p.pad_w
        + p.dilation_w * (p.w_filt - 1)
        + p.output_pad_w
        + 1;
    (h_out, w_out)
}

fn validate_convt2d_params(p: &ConvT2dParams) -> i32 {
    if p.batch <= 0 || p.c_in <= 0 || p.h_in <= 0 || p.w_in <= 0 {
        return INVALID;
    }
    if p.c_out <= 0 || p.h_filt <= 0 || p.w_filt <= 0 {
        return INVALID;
    }
    if p.stride_h <= 0 || p.stride_w <= 0 || p.dilation_h <= 0 || p.dilation_w <= 0 {
        return INVALID;
    }
    if p.pad_h < 0 || p.pad_w < 0 || p.output_pad_h < 0 || p.output_pad_w < 0 {
        return INVALID;
    }
    if p.groups <= 0 || p.c_in % p.groups != 0 || p.c_out % p.groups != 0 {
        return INVALID;
    }
    if p.output_pad_h >= p.stride_h.max(p.dilation_h)
        || p.output_pad_w >= p.stride_w.max(p.dilation_w)
    {
        return INVALID;
    }
    let (h_out, w_out) = compute_convt2d_out(p);
    if h_out <= 0 || w_out <= 0 {
        return INVALID;
    }
    OK
}

/// Build the synthetic-forward-conv descriptors: synth_x is the
/// ConvTranspose **output** ([N, C_out, H_out, W_out]); synth_y is the
/// ConvTranspose **input** ([N, C_in, H_in, W_in]); synth_w is the
/// filter `[C_in, C_out/groups, kH, kW]` (PyTorch order = cuDNN's
/// `[K, C/groups, ...]` with K = C_in).
fn build_convt2d_descs(g: &mut ConvDescGuard, p: &ConvT2dParams, dt: DtypeTag) -> i32 {
    let cudnn_dt = dt.cudnn_dtype();
    let compute_dt = if dt.is_double_compute() {
        CUDNN_DATA_DOUBLE
    } else {
        CUDNN_DATA_FLOAT
    };
    let (h_out, w_out) = compute_convt2d_out(p);
    let c_out_per_group = p.c_out / p.groups;

    // synth_x = [N, C_out, H_out, W_out].
    let s = unsafe { cudnnCreateTensorDescriptor(&mut g.x_desc as *mut _) };
    if s != 0 {
        return INTERNAL;
    }
    let s = unsafe {
        cudnnSetTensor4dDescriptor(g.x_desc, CUDNN_TENSOR_NCHW, cudnn_dt, p.batch, p.c_out, h_out, w_out)
    };
    if s != 0 {
        return INTERNAL;
    }

    // synth_y = [N, C_in, H_in, W_in].
    let s = unsafe { cudnnCreateTensorDescriptor(&mut g.y_desc as *mut _) };
    if s != 0 {
        return INTERNAL;
    }
    let s = unsafe {
        cudnnSetTensor4dDescriptor(
            g.y_desc, CUDNN_TENSOR_NCHW, cudnn_dt,
            p.batch, p.c_in, p.h_in, p.w_in,
        )
    };
    if s != 0 {
        return INTERNAL;
    }

    // synth_w = [C_in, C_out/groups, kH, kW] (PyTorch / cuDNN K=C_in).
    let s = unsafe { cudnnCreateFilterDescriptor(&mut g.w_desc as *mut _) };
    if s != 0 {
        return INTERNAL;
    }
    let s = unsafe {
        cudnnSetFilter4dDescriptor(
            g.w_desc, cudnn_dt, CUDNN_TENSOR_NCHW,
            p.c_in, c_out_per_group, p.h_filt, p.w_filt,
        )
    };
    if s != 0 {
        return INTERNAL;
    }

    // conv — same pad/stride/dilation as the ConvTranspose.
    let s = unsafe { cudnnCreateConvolutionDescriptor(&mut g.conv_desc as *mut _) };
    if s != 0 {
        return INTERNAL;
    }
    let s = unsafe {
        cudnnSetConvolution2dDescriptor(
            g.conv_desc, p.pad_h, p.pad_w, p.stride_h, p.stride_w,
            p.dilation_h, p.dilation_w, CUDNN_CROSS_CORRELATION, compute_dt,
        )
    };
    if s != 0 {
        return INTERNAL;
    }
    let s = unsafe { cudnnSetConvolutionGroupCount(g.conv_desc, p.groups) };
    if s != 0 {
        return INTERNAL;
    }
    OK
}

/// ConvTranspose2d FW = synthetic BackwardData. Args: x, w → y.
/// The synthetic-conv mapping treats `y` as synth_x (output of cuDNN
/// BackwardData), `x` as synth_y, and `w` as synth_w.
#[allow(clippy::too_many_arguments)]
fn run_convt2d_fw(
    p: &ConvT2dParams, dt: DtypeTag,
    x: *const c_void, w: *const c_void, y: *mut c_void,
    ws_ptr: *mut c_void, ws_bytes: usize, stream: *mut c_void,
) -> i32 {
    if x.is_null() || w.is_null() || y.is_null() {
        return INVALID;
    }
    let v = validate_convt2d_params(p);
    if v != OK {
        return v;
    }
    let mut g = ConvDescGuard::new();
    let s = setup_handle(&mut g, stream);
    if s != OK { return s; }
    let s = build_convt2d_descs(&mut g, p, dt);
    if s != OK { return s; }
    // FW workspace = BackwardData workspace of the synthetic conv.
    let mut needed: usize = 0;
    let s = unsafe {
        cudnnGetConvolutionBackwardDataWorkspaceSize(
            g.handle, g.w_desc, g.y_desc, g.conv_desc, g.x_desc,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, &mut needed as *mut usize,
        )
    };
    if s != 0 { return INTERNAL; }
    let (ws, s) = WsHolder::ensure(ws_ptr, ws_bytes, needed);
    if s != OK { return s; }

    let status = if dt.is_double_compute() {
        let alpha: f64 = 1.0; let beta: f64 = 0.0;
        unsafe {
            cudnnConvolutionBackwardData(
                g.handle, &alpha as *const f64 as *const c_void,
                g.w_desc, w, g.y_desc, x, g.conv_desc,
                CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, ws.ptr, needed,
                &beta as *const f64 as *const c_void, g.x_desc, y,
            )
        }
    } else {
        let alpha: f32 = 1.0; let beta: f32 = 0.0;
        unsafe {
            cudnnConvolutionBackwardData(
                g.handle, &alpha as *const f32 as *const c_void,
                g.w_desc, w, g.y_desc, x, g.conv_desc,
                CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, ws.ptr, needed,
                &beta as *const f32 as *const c_void, g.x_desc, y,
            )
        }
    };
    if status != 0 { return INTERNAL; }
    OK
}

/// ConvTranspose2d BW-data = synthetic Forward. Args: w, dy → dx.
#[allow(clippy::too_many_arguments)]
fn run_convt2d_bw_data(
    p: &ConvT2dParams, dt: DtypeTag,
    w: *const c_void, dy: *const c_void, dx: *mut c_void,
    ws_ptr: *mut c_void, ws_bytes: usize, stream: *mut c_void,
) -> i32 {
    if w.is_null() || dy.is_null() || dx.is_null() {
        return INVALID;
    }
    let v = validate_convt2d_params(p);
    if v != OK { return v; }
    let mut g = ConvDescGuard::new();
    let s = setup_handle(&mut g, stream);
    if s != OK { return s; }
    let s = build_convt2d_descs(&mut g, p, dt);
    if s != OK { return s; }
    let mut needed: usize = 0;
    let s = unsafe {
        cudnnGetConvolutionForwardWorkspaceSize(
            g.handle, g.x_desc, g.w_desc, g.conv_desc, g.y_desc,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, &mut needed as *mut usize,
        )
    };
    if s != 0 { return INTERNAL; }
    let (ws, s) = WsHolder::ensure(ws_ptr, ws_bytes, needed);
    if s != OK { return s; }

    let status = if dt.is_double_compute() {
        let alpha: f64 = 1.0; let beta: f64 = 0.0;
        unsafe {
            cudnnConvolutionForward(
                g.handle, &alpha as *const f64 as *const c_void,
                g.x_desc, dy, g.w_desc, w, g.conv_desc,
                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, ws.ptr, needed,
                &beta as *const f64 as *const c_void, g.y_desc, dx,
            )
        }
    } else {
        let alpha: f32 = 1.0; let beta: f32 = 0.0;
        unsafe {
            cudnnConvolutionForward(
                g.handle, &alpha as *const f32 as *const c_void,
                g.x_desc, dy, g.w_desc, w, g.conv_desc,
                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, ws.ptr, needed,
                &beta as *const f32 as *const c_void, g.y_desc, dx,
            )
        }
    };
    if status != 0 { return INTERNAL; }
    OK
}

/// ConvTranspose2d BW-filter = synthetic BackwardFilter with the FW
/// role swap. Args: x, dy → dw. The cuDNN call uses synth_x = dy and
/// synth_y = x (matches the Rust plan's `run_dw`).
#[allow(clippy::too_many_arguments)]
fn run_convt2d_bw_filter(
    p: &ConvT2dParams, dt: DtypeTag,
    x: *const c_void, dy: *const c_void, dw: *mut c_void,
    ws_ptr: *mut c_void, ws_bytes: usize, stream: *mut c_void,
) -> i32 {
    if x.is_null() || dy.is_null() || dw.is_null() {
        return INVALID;
    }
    let v = validate_convt2d_params(p);
    if v != OK { return v; }
    let mut g = ConvDescGuard::new();
    let s = setup_handle(&mut g, stream);
    if s != OK { return s; }
    let s = build_convt2d_descs(&mut g, p, dt);
    if s != OK { return s; }
    let mut needed: usize = 0;
    let s = unsafe {
        cudnnGetConvolutionBackwardFilterWorkspaceSize(
            g.handle, g.x_desc, g.y_desc, g.conv_desc, g.w_desc,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, &mut needed as *mut usize,
        )
    };
    if s != 0 { return INTERNAL; }
    let (ws, s) = WsHolder::ensure(ws_ptr, ws_bytes, needed);
    if s != OK { return s; }

    let status = if dt.is_double_compute() {
        let alpha: f64 = 1.0; let beta: f64 = 0.0;
        unsafe {
            cudnnConvolutionBackwardFilter(
                g.handle, &alpha as *const f64 as *const c_void,
                g.x_desc, dy, g.y_desc, x, g.conv_desc,
                CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, ws.ptr, needed,
                &beta as *const f64 as *const c_void, g.w_desc, dw,
            )
        }
    } else {
        let alpha: f32 = 1.0; let beta: f32 = 0.0;
        unsafe {
            cudnnConvolutionBackwardFilter(
                g.handle, &alpha as *const f32 as *const c_void,
                g.x_desc, dy, g.y_desc, x, g.conv_desc,
                CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, ws.ptr, needed,
                &beta as *const f32 as *const c_void, g.w_desc, dw,
            )
        }
    };
    if status != 0 { return INTERNAL; }
    OK
}

macro_rules! convt2d_dir_impl {
    ($name:ident, $dt:expr, $runner:ident, $a:ident, $b:ident, $c:ident) => {
        /// ConvTranspose2d direction.
        ///
        /// # Safety: all pointers must be live device memory; `stream` valid.
        #[unsafe(no_mangle)]
        #[allow(clippy::too_many_arguments)]
        pub unsafe extern "C" fn $name(
            batch: i32, c_in: i32, c_out: i32,
            h_in: i32, w_in: i32, _h_out: i32, _w_out: i32,
            kh: i32, kw: i32,
            stride_h: i32, stride_w: i32,
            pad_h: i32, pad_w: i32,
            dilation_h: i32, dilation_w: i32,
            output_pad_h: i32, output_pad_w: i32,
            groups: i32,
            $a: *const c_void,
            $b: *const c_void,
            $c: *mut c_void,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32 {
            let p = ConvT2dParams {
                batch, c_in, c_out,
                h_in, w_in,
                h_filt: kh, w_filt: kw,
                pad_h, pad_w, stride_h, stride_w,
                dilation_h, dilation_w,
                output_pad_h, output_pad_w,
                groups,
            };
            $runner(&p, $dt, $a, $b, $c, workspace, workspace_bytes, stream)
        }
    };
}

convt2d_dir_impl!(baracuda_kernels_conv_transpose_2d_fw_f32_run, DtypeTag::F32, run_convt2d_fw, input, filter, output);
convt2d_dir_impl!(baracuda_kernels_conv_transpose_2d_fw_f64_run, DtypeTag::F64, run_convt2d_fw, input, filter, output);
convt2d_dir_impl!(baracuda_kernels_conv_transpose_2d_fw_f16_run, DtypeTag::F16, run_convt2d_fw, input, filter, output);
convt2d_dir_impl!(baracuda_kernels_conv_transpose_2d_fw_bf16_run, DtypeTag::Bf16, run_convt2d_fw, input, filter, output);
convt2d_dir_impl!(baracuda_kernels_conv_transpose_2d_bw_data_f32_run, DtypeTag::F32, run_convt2d_bw_data, filter, grad_output, grad_input);
convt2d_dir_impl!(baracuda_kernels_conv_transpose_2d_bw_data_f64_run, DtypeTag::F64, run_convt2d_bw_data, filter, grad_output, grad_input);
convt2d_dir_impl!(baracuda_kernels_conv_transpose_2d_bw_data_f16_run, DtypeTag::F16, run_convt2d_bw_data, filter, grad_output, grad_input);
convt2d_dir_impl!(baracuda_kernels_conv_transpose_2d_bw_data_bf16_run, DtypeTag::Bf16, run_convt2d_bw_data, filter, grad_output, grad_input);
convt2d_dir_impl!(baracuda_kernels_conv_transpose_2d_bw_filter_f32_run, DtypeTag::F32, run_convt2d_bw_filter, input, grad_output, grad_filter);
convt2d_dir_impl!(baracuda_kernels_conv_transpose_2d_bw_filter_f64_run, DtypeTag::F64, run_convt2d_bw_filter, input, grad_output, grad_filter);
convt2d_dir_impl!(baracuda_kernels_conv_transpose_2d_bw_filter_f16_run, DtypeTag::F16, run_convt2d_bw_filter, input, grad_output, grad_filter);
convt2d_dir_impl!(baracuda_kernels_conv_transpose_2d_bw_filter_bf16_run, DtypeTag::Bf16, run_convt2d_bw_filter, input, grad_output, grad_filter);

// =============================================================================
// ConvTranspose1d — rank-3 NCL with W=1 padding (same as Conv1d).
// =============================================================================

#[derive(Copy, Clone)]
struct ConvT1dParams {
    batch: i32,
    c_in: i32,
    c_out: i32,
    l_in: i32,
    l_filt: i32,
    pad_l: i32,
    stride_l: i32,
    dilation_l: i32,
    output_pad_l: i32,
    groups: i32,
}

#[inline]
fn compute_convt1d_out(p: &ConvT1dParams) -> i32 {
    (p.l_in - 1) * p.stride_l - 2 * p.pad_l
        + p.dilation_l * (p.l_filt - 1)
        + p.output_pad_l
        + 1
}

fn validate_convt1d_params(p: &ConvT1dParams) -> i32 {
    if p.batch <= 0 || p.c_in <= 0 || p.l_in <= 0 { return INVALID; }
    if p.c_out <= 0 || p.l_filt <= 0 { return INVALID; }
    if p.stride_l <= 0 || p.dilation_l <= 0 || p.pad_l < 0 || p.output_pad_l < 0 {
        return INVALID;
    }
    if p.groups <= 0 || p.c_in % p.groups != 0 || p.c_out % p.groups != 0 {
        return INVALID;
    }
    if p.output_pad_l >= p.stride_l.max(p.dilation_l) {
        return INVALID;
    }
    if compute_convt1d_out(p) <= 0 { return INVALID; }
    OK
}

fn build_convt1d_descs(g: &mut ConvDescGuard, p: &ConvT1dParams, dt: DtypeTag) -> i32 {
    let cudnn_dt = dt.cudnn_dtype();
    let compute_dt = if dt.is_double_compute() {
        CUDNN_DATA_DOUBLE
    } else {
        CUDNN_DATA_FLOAT
    };
    let l_out = compute_convt1d_out(p);
    let c_out_per_group = p.c_out / p.groups;

    // synth_x = [N, C_out, L_out, 1].
    let s = unsafe { cudnnCreateTensorDescriptor(&mut g.x_desc as *mut _) };
    if s != 0 { return INTERNAL; }
    let x_dims = [p.batch, p.c_out, l_out, 1];
    let x_strides = [p.c_out * l_out, l_out, 1, 1];
    let s = unsafe {
        cudnnSetTensorNdDescriptor(g.x_desc, cudnn_dt, 4, x_dims.as_ptr(), x_strides.as_ptr())
    };
    if s != 0 { return INTERNAL; }

    // synth_y = [N, C_in, L_in, 1].
    let s = unsafe { cudnnCreateTensorDescriptor(&mut g.y_desc as *mut _) };
    if s != 0 { return INTERNAL; }
    let y_dims = [p.batch, p.c_in, p.l_in, 1];
    let y_strides = [p.c_in * p.l_in, p.l_in, 1, 1];
    let s = unsafe {
        cudnnSetTensorNdDescriptor(g.y_desc, cudnn_dt, 4, y_dims.as_ptr(), y_strides.as_ptr())
    };
    if s != 0 { return INTERNAL; }

    // synth_w = [C_in, C_out/groups, L_filt, 1].
    let s = unsafe { cudnnCreateFilterDescriptor(&mut g.w_desc as *mut _) };
    if s != 0 { return INTERNAL; }
    let w_dims = [p.c_in, c_out_per_group, p.l_filt, 1];
    let s = unsafe {
        cudnnSetFilterNdDescriptor(g.w_desc, cudnn_dt, CUDNN_TENSOR_NCHW, 4, w_dims.as_ptr())
    };
    if s != 0 { return INTERNAL; }

    // conv — array_length = 2 (W axis is dummy).
    let s = unsafe { cudnnCreateConvolutionDescriptor(&mut g.conv_desc as *mut _) };
    if s != 0 { return INTERNAL; }
    let pad_a = [p.pad_l, 0];
    let stride_a = [p.stride_l, 1];
    let dil_a = [p.dilation_l, 1];
    let s = unsafe {
        cudnnSetConvolutionNdDescriptor(
            g.conv_desc, 2,
            pad_a.as_ptr(), stride_a.as_ptr(), dil_a.as_ptr(),
            CUDNN_CROSS_CORRELATION, compute_dt,
        )
    };
    if s != 0 { return INTERNAL; }
    let s = unsafe { cudnnSetConvolutionGroupCount(g.conv_desc, p.groups) };
    if s != 0 { return INTERNAL; }
    OK
}

#[allow(clippy::too_many_arguments)]
fn run_convt1d_fw(
    p: &ConvT1dParams, dt: DtypeTag,
    x: *const c_void, w: *const c_void, y: *mut c_void,
    ws_ptr: *mut c_void, ws_bytes: usize, stream: *mut c_void,
) -> i32 {
    if x.is_null() || w.is_null() || y.is_null() { return INVALID; }
    let v = validate_convt1d_params(p); if v != OK { return v; }
    let mut g = ConvDescGuard::new();
    let s = setup_handle(&mut g, stream); if s != OK { return s; }
    let s = build_convt1d_descs(&mut g, p, dt); if s != OK { return s; }
    let mut needed: usize = 0;
    let s = unsafe {
        cudnnGetConvolutionBackwardDataWorkspaceSize(
            g.handle, g.w_desc, g.y_desc, g.conv_desc, g.x_desc,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, &mut needed as *mut usize,
        )
    };
    if s != 0 { return INTERNAL; }
    let (ws, s) = WsHolder::ensure(ws_ptr, ws_bytes, needed); if s != OK { return s; }
    let status = if dt.is_double_compute() {
        let alpha: f64 = 1.0; let beta: f64 = 0.0;
        unsafe {
            cudnnConvolutionBackwardData(
                g.handle, &alpha as *const f64 as *const c_void,
                g.w_desc, w, g.y_desc, x, g.conv_desc,
                CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, ws.ptr, needed,
                &beta as *const f64 as *const c_void, g.x_desc, y,
            )
        }
    } else {
        let alpha: f32 = 1.0; let beta: f32 = 0.0;
        unsafe {
            cudnnConvolutionBackwardData(
                g.handle, &alpha as *const f32 as *const c_void,
                g.w_desc, w, g.y_desc, x, g.conv_desc,
                CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, ws.ptr, needed,
                &beta as *const f32 as *const c_void, g.x_desc, y,
            )
        }
    };
    if status != 0 { return INTERNAL; }
    OK
}

#[allow(clippy::too_many_arguments)]
fn run_convt1d_bw_data(
    p: &ConvT1dParams, dt: DtypeTag,
    w: *const c_void, dy: *const c_void, dx: *mut c_void,
    ws_ptr: *mut c_void, ws_bytes: usize, stream: *mut c_void,
) -> i32 {
    if w.is_null() || dy.is_null() || dx.is_null() { return INVALID; }
    let v = validate_convt1d_params(p); if v != OK { return v; }
    let mut g = ConvDescGuard::new();
    let s = setup_handle(&mut g, stream); if s != OK { return s; }
    let s = build_convt1d_descs(&mut g, p, dt); if s != OK { return s; }
    let mut needed: usize = 0;
    let s = unsafe {
        cudnnGetConvolutionForwardWorkspaceSize(
            g.handle, g.x_desc, g.w_desc, g.conv_desc, g.y_desc,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, &mut needed as *mut usize,
        )
    };
    if s != 0 { return INTERNAL; }
    let (ws, s) = WsHolder::ensure(ws_ptr, ws_bytes, needed); if s != OK { return s; }
    let status = if dt.is_double_compute() {
        let alpha: f64 = 1.0; let beta: f64 = 0.0;
        unsafe {
            cudnnConvolutionForward(
                g.handle, &alpha as *const f64 as *const c_void,
                g.x_desc, dy, g.w_desc, w, g.conv_desc,
                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, ws.ptr, needed,
                &beta as *const f64 as *const c_void, g.y_desc, dx,
            )
        }
    } else {
        let alpha: f32 = 1.0; let beta: f32 = 0.0;
        unsafe {
            cudnnConvolutionForward(
                g.handle, &alpha as *const f32 as *const c_void,
                g.x_desc, dy, g.w_desc, w, g.conv_desc,
                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, ws.ptr, needed,
                &beta as *const f32 as *const c_void, g.y_desc, dx,
            )
        }
    };
    if status != 0 { return INTERNAL; }
    OK
}

#[allow(clippy::too_many_arguments)]
fn run_convt1d_bw_filter(
    p: &ConvT1dParams, dt: DtypeTag,
    x: *const c_void, dy: *const c_void, dw: *mut c_void,
    ws_ptr: *mut c_void, ws_bytes: usize, stream: *mut c_void,
) -> i32 {
    if x.is_null() || dy.is_null() || dw.is_null() { return INVALID; }
    let v = validate_convt1d_params(p); if v != OK { return v; }
    let mut g = ConvDescGuard::new();
    let s = setup_handle(&mut g, stream); if s != OK { return s; }
    let s = build_convt1d_descs(&mut g, p, dt); if s != OK { return s; }
    let mut needed: usize = 0;
    let s = unsafe {
        cudnnGetConvolutionBackwardFilterWorkspaceSize(
            g.handle, g.x_desc, g.y_desc, g.conv_desc, g.w_desc,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, &mut needed as *mut usize,
        )
    };
    if s != 0 { return INTERNAL; }
    let (ws, s) = WsHolder::ensure(ws_ptr, ws_bytes, needed); if s != OK { return s; }
    let status = if dt.is_double_compute() {
        let alpha: f64 = 1.0; let beta: f64 = 0.0;
        unsafe {
            cudnnConvolutionBackwardFilter(
                g.handle, &alpha as *const f64 as *const c_void,
                g.x_desc, dy, g.y_desc, x, g.conv_desc,
                CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, ws.ptr, needed,
                &beta as *const f64 as *const c_void, g.w_desc, dw,
            )
        }
    } else {
        let alpha: f32 = 1.0; let beta: f32 = 0.0;
        unsafe {
            cudnnConvolutionBackwardFilter(
                g.handle, &alpha as *const f32 as *const c_void,
                g.x_desc, dy, g.y_desc, x, g.conv_desc,
                CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, ws.ptr, needed,
                &beta as *const f32 as *const c_void, g.w_desc, dw,
            )
        }
    };
    if status != 0 { return INTERNAL; }
    OK
}

macro_rules! convt1d_dir_impl {
    ($name:ident, $dt:expr, $runner:ident, $a:ident, $b:ident, $c:ident) => {
        /// ConvTranspose1d direction.
        ///
        /// # Safety: all pointers must be live device memory; `stream` valid.
        #[unsafe(no_mangle)]
        #[allow(clippy::too_many_arguments)]
        pub unsafe extern "C" fn $name(
            batch: i32, c_in: i32, c_out: i32,
            l_in: i32, _l_out: i32,
            l_filt: i32,
            stride_l: i32, pad_l: i32, dilation_l: i32,
            output_pad_l: i32,
            groups: i32,
            $a: *const c_void,
            $b: *const c_void,
            $c: *mut c_void,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32 {
            let p = ConvT1dParams {
                batch, c_in, c_out, l_in, l_filt,
                pad_l, stride_l, dilation_l, output_pad_l, groups,
            };
            $runner(&p, $dt, $a, $b, $c, workspace, workspace_bytes, stream)
        }
    };
}

convt1d_dir_impl!(baracuda_kernels_conv_transpose_1d_fw_f32_run, DtypeTag::F32, run_convt1d_fw, input, filter, output);
convt1d_dir_impl!(baracuda_kernels_conv_transpose_1d_fw_f64_run, DtypeTag::F64, run_convt1d_fw, input, filter, output);
convt1d_dir_impl!(baracuda_kernels_conv_transpose_1d_fw_f16_run, DtypeTag::F16, run_convt1d_fw, input, filter, output);
convt1d_dir_impl!(baracuda_kernels_conv_transpose_1d_fw_bf16_run, DtypeTag::Bf16, run_convt1d_fw, input, filter, output);
convt1d_dir_impl!(baracuda_kernels_conv_transpose_1d_bw_data_f32_run, DtypeTag::F32, run_convt1d_bw_data, filter, grad_output, grad_input);
convt1d_dir_impl!(baracuda_kernels_conv_transpose_1d_bw_data_f64_run, DtypeTag::F64, run_convt1d_bw_data, filter, grad_output, grad_input);
convt1d_dir_impl!(baracuda_kernels_conv_transpose_1d_bw_data_f16_run, DtypeTag::F16, run_convt1d_bw_data, filter, grad_output, grad_input);
convt1d_dir_impl!(baracuda_kernels_conv_transpose_1d_bw_data_bf16_run, DtypeTag::Bf16, run_convt1d_bw_data, filter, grad_output, grad_input);
convt1d_dir_impl!(baracuda_kernels_conv_transpose_1d_bw_filter_f32_run, DtypeTag::F32, run_convt1d_bw_filter, input, grad_output, grad_filter);
convt1d_dir_impl!(baracuda_kernels_conv_transpose_1d_bw_filter_f64_run, DtypeTag::F64, run_convt1d_bw_filter, input, grad_output, grad_filter);
convt1d_dir_impl!(baracuda_kernels_conv_transpose_1d_bw_filter_f16_run, DtypeTag::F16, run_convt1d_bw_filter, input, grad_output, grad_filter);
convt1d_dir_impl!(baracuda_kernels_conv_transpose_1d_bw_filter_bf16_run, DtypeTag::Bf16, run_convt1d_bw_filter, input, grad_output, grad_filter);

// =============================================================================
// ConvTranspose3d — rank-5 NCDHW; cuDNN handles 3D natively.
// =============================================================================

#[derive(Copy, Clone)]
struct ConvT3dParams {
    batch: i32,
    c_in: i32,
    c_out: i32,
    d_in: i32, h_in: i32, w_in: i32,
    d_filt: i32, h_filt: i32, w_filt: i32,
    pad_d: i32, pad_h: i32, pad_w: i32,
    stride_d: i32, stride_h: i32, stride_w: i32,
    dilation_d: i32, dilation_h: i32, dilation_w: i32,
    output_pad_d: i32, output_pad_h: i32, output_pad_w: i32,
    groups: i32,
}

#[inline]
fn compute_convt3d_out(p: &ConvT3dParams) -> (i32, i32, i32) {
    let d_out = (p.d_in - 1) * p.stride_d - 2 * p.pad_d
        + p.dilation_d * (p.d_filt - 1) + p.output_pad_d + 1;
    let h_out = (p.h_in - 1) * p.stride_h - 2 * p.pad_h
        + p.dilation_h * (p.h_filt - 1) + p.output_pad_h + 1;
    let w_out = (p.w_in - 1) * p.stride_w - 2 * p.pad_w
        + p.dilation_w * (p.w_filt - 1) + p.output_pad_w + 1;
    (d_out, h_out, w_out)
}

fn validate_convt3d_params(p: &ConvT3dParams) -> i32 {
    if p.batch <= 0 || p.c_in <= 0 || p.d_in <= 0 || p.h_in <= 0 || p.w_in <= 0 { return INVALID; }
    if p.c_out <= 0 || p.d_filt <= 0 || p.h_filt <= 0 || p.w_filt <= 0 { return INVALID; }
    if p.stride_d <= 0 || p.stride_h <= 0 || p.stride_w <= 0 { return INVALID; }
    if p.dilation_d <= 0 || p.dilation_h <= 0 || p.dilation_w <= 0 { return INVALID; }
    if p.pad_d < 0 || p.pad_h < 0 || p.pad_w < 0 { return INVALID; }
    if p.output_pad_d < 0 || p.output_pad_h < 0 || p.output_pad_w < 0 { return INVALID; }
    if p.groups <= 0 || p.c_in % p.groups != 0 || p.c_out % p.groups != 0 { return INVALID; }
    if p.output_pad_d >= p.stride_d.max(p.dilation_d)
        || p.output_pad_h >= p.stride_h.max(p.dilation_h)
        || p.output_pad_w >= p.stride_w.max(p.dilation_w)
    {
        return INVALID;
    }
    let (d_out, h_out, w_out) = compute_convt3d_out(p);
    if d_out <= 0 || h_out <= 0 || w_out <= 0 { return INVALID; }
    OK
}

fn build_convt3d_descs(g: &mut ConvDescGuard, p: &ConvT3dParams, dt: DtypeTag) -> i32 {
    let cudnn_dt = dt.cudnn_dtype();
    let compute_dt = if dt.is_double_compute() { CUDNN_DATA_DOUBLE } else { CUDNN_DATA_FLOAT };
    let (d_out, h_out, w_out) = compute_convt3d_out(p);
    let c_out_per_group = p.c_out / p.groups;

    // synth_x = [N, C_out, D_out, H_out, W_out] (ConvTranspose output role).
    let s = unsafe { cudnnCreateTensorDescriptor(&mut g.x_desc as *mut _) };
    if s != 0 { return INTERNAL; }
    let x_dims = [p.batch, p.c_out, d_out, h_out, w_out];
    let s_w = 1;
    let s_h = w_out;
    let s_d = h_out * w_out;
    let s_c = d_out * h_out * w_out;
    let s_n = p.c_out * s_c;
    let x_strides = [s_n, s_c, s_d, s_h, s_w];
    let s = unsafe {
        cudnnSetTensorNdDescriptor(g.x_desc, cudnn_dt, 5, x_dims.as_ptr(), x_strides.as_ptr())
    };
    if s != 0 { return INTERNAL; }

    // synth_y = [N, C_in, D_in, H_in, W_in] (ConvTranspose input role).
    let s = unsafe { cudnnCreateTensorDescriptor(&mut g.y_desc as *mut _) };
    if s != 0 { return INTERNAL; }
    let y_dims = [p.batch, p.c_in, p.d_in, p.h_in, p.w_in];
    let y_s_w = 1;
    let y_s_h = p.w_in;
    let y_s_d = p.h_in * p.w_in;
    let y_s_c = p.d_in * p.h_in * p.w_in;
    let y_s_n = p.c_in * y_s_c;
    let y_strides = [y_s_n, y_s_c, y_s_d, y_s_h, y_s_w];
    let s = unsafe {
        cudnnSetTensorNdDescriptor(g.y_desc, cudnn_dt, 5, y_dims.as_ptr(), y_strides.as_ptr())
    };
    if s != 0 { return INTERNAL; }

    // synth_w = [C_in, C_out/groups, D_filt, H_filt, W_filt] (PyTorch).
    let s = unsafe { cudnnCreateFilterDescriptor(&mut g.w_desc as *mut _) };
    if s != 0 { return INTERNAL; }
    let w_dims = [p.c_in, c_out_per_group, p.d_filt, p.h_filt, p.w_filt];
    let s = unsafe {
        cudnnSetFilterNdDescriptor(g.w_desc, cudnn_dt, CUDNN_TENSOR_NCHW, 5, w_dims.as_ptr())
    };
    if s != 0 { return INTERNAL; }

    // conv — array_length = 3.
    let s = unsafe { cudnnCreateConvolutionDescriptor(&mut g.conv_desc as *mut _) };
    if s != 0 { return INTERNAL; }
    let pad_a = [p.pad_d, p.pad_h, p.pad_w];
    let stride_a = [p.stride_d, p.stride_h, p.stride_w];
    let dil_a = [p.dilation_d, p.dilation_h, p.dilation_w];
    let s = unsafe {
        cudnnSetConvolutionNdDescriptor(
            g.conv_desc, 3,
            pad_a.as_ptr(), stride_a.as_ptr(), dil_a.as_ptr(),
            CUDNN_CROSS_CORRELATION, compute_dt,
        )
    };
    if s != 0 { return INTERNAL; }
    let s = unsafe { cudnnSetConvolutionGroupCount(g.conv_desc, p.groups) };
    if s != 0 { return INTERNAL; }
    OK
}

#[allow(clippy::too_many_arguments)]
fn run_convt3d_fw(
    p: &ConvT3dParams, dt: DtypeTag,
    x: *const c_void, w: *const c_void, y: *mut c_void,
    ws_ptr: *mut c_void, ws_bytes: usize, stream: *mut c_void,
) -> i32 {
    if x.is_null() || w.is_null() || y.is_null() { return INVALID; }
    let v = validate_convt3d_params(p); if v != OK { return v; }
    let mut g = ConvDescGuard::new();
    let s = setup_handle(&mut g, stream); if s != OK { return s; }
    let s = build_convt3d_descs(&mut g, p, dt); if s != OK { return s; }
    let mut needed: usize = 0;
    let s = unsafe {
        cudnnGetConvolutionBackwardDataWorkspaceSize(
            g.handle, g.w_desc, g.y_desc, g.conv_desc, g.x_desc,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, &mut needed as *mut usize,
        )
    };
    if s != 0 { return INTERNAL; }
    let (ws, s) = WsHolder::ensure(ws_ptr, ws_bytes, needed); if s != OK { return s; }
    let status = if dt.is_double_compute() {
        let alpha: f64 = 1.0; let beta: f64 = 0.0;
        unsafe {
            cudnnConvolutionBackwardData(
                g.handle, &alpha as *const f64 as *const c_void,
                g.w_desc, w, g.y_desc, x, g.conv_desc,
                CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, ws.ptr, needed,
                &beta as *const f64 as *const c_void, g.x_desc, y,
            )
        }
    } else {
        let alpha: f32 = 1.0; let beta: f32 = 0.0;
        unsafe {
            cudnnConvolutionBackwardData(
                g.handle, &alpha as *const f32 as *const c_void,
                g.w_desc, w, g.y_desc, x, g.conv_desc,
                CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, ws.ptr, needed,
                &beta as *const f32 as *const c_void, g.x_desc, y,
            )
        }
    };
    if status != 0 { return INTERNAL; }
    OK
}

#[allow(clippy::too_many_arguments)]
fn run_convt3d_bw_data(
    p: &ConvT3dParams, dt: DtypeTag,
    w: *const c_void, dy: *const c_void, dx: *mut c_void,
    ws_ptr: *mut c_void, ws_bytes: usize, stream: *mut c_void,
) -> i32 {
    if w.is_null() || dy.is_null() || dx.is_null() { return INVALID; }
    let v = validate_convt3d_params(p); if v != OK { return v; }
    let mut g = ConvDescGuard::new();
    let s = setup_handle(&mut g, stream); if s != OK { return s; }
    let s = build_convt3d_descs(&mut g, p, dt); if s != OK { return s; }
    let mut needed: usize = 0;
    let s = unsafe {
        cudnnGetConvolutionForwardWorkspaceSize(
            g.handle, g.x_desc, g.w_desc, g.conv_desc, g.y_desc,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, &mut needed as *mut usize,
        )
    };
    if s != 0 { return INTERNAL; }
    let (ws, s) = WsHolder::ensure(ws_ptr, ws_bytes, needed); if s != OK { return s; }
    let status = if dt.is_double_compute() {
        let alpha: f64 = 1.0; let beta: f64 = 0.0;
        unsafe {
            cudnnConvolutionForward(
                g.handle, &alpha as *const f64 as *const c_void,
                g.x_desc, dy, g.w_desc, w, g.conv_desc,
                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, ws.ptr, needed,
                &beta as *const f64 as *const c_void, g.y_desc, dx,
            )
        }
    } else {
        let alpha: f32 = 1.0; let beta: f32 = 0.0;
        unsafe {
            cudnnConvolutionForward(
                g.handle, &alpha as *const f32 as *const c_void,
                g.x_desc, dy, g.w_desc, w, g.conv_desc,
                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, ws.ptr, needed,
                &beta as *const f32 as *const c_void, g.y_desc, dx,
            )
        }
    };
    if status != 0 { return INTERNAL; }
    OK
}

#[allow(clippy::too_many_arguments)]
fn run_convt3d_bw_filter(
    p: &ConvT3dParams, dt: DtypeTag,
    x: *const c_void, dy: *const c_void, dw: *mut c_void,
    ws_ptr: *mut c_void, ws_bytes: usize, stream: *mut c_void,
) -> i32 {
    if x.is_null() || dy.is_null() || dw.is_null() { return INVALID; }
    let v = validate_convt3d_params(p); if v != OK { return v; }
    let mut g = ConvDescGuard::new();
    let s = setup_handle(&mut g, stream); if s != OK { return s; }
    let s = build_convt3d_descs(&mut g, p, dt); if s != OK { return s; }
    let mut needed: usize = 0;
    let s = unsafe {
        cudnnGetConvolutionBackwardFilterWorkspaceSize(
            g.handle, g.x_desc, g.y_desc, g.conv_desc, g.w_desc,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, &mut needed as *mut usize,
        )
    };
    if s != 0 { return INTERNAL; }
    let (ws, s) = WsHolder::ensure(ws_ptr, ws_bytes, needed); if s != OK { return s; }
    let status = if dt.is_double_compute() {
        let alpha: f64 = 1.0; let beta: f64 = 0.0;
        unsafe {
            cudnnConvolutionBackwardFilter(
                g.handle, &alpha as *const f64 as *const c_void,
                g.x_desc, dy, g.y_desc, x, g.conv_desc,
                CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, ws.ptr, needed,
                &beta as *const f64 as *const c_void, g.w_desc, dw,
            )
        }
    } else {
        let alpha: f32 = 1.0; let beta: f32 = 0.0;
        unsafe {
            cudnnConvolutionBackwardFilter(
                g.handle, &alpha as *const f32 as *const c_void,
                g.x_desc, dy, g.y_desc, x, g.conv_desc,
                CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, ws.ptr, needed,
                &beta as *const f32 as *const c_void, g.w_desc, dw,
            )
        }
    };
    if status != 0 { return INTERNAL; }
    OK
}

macro_rules! convt3d_dir_impl {
    ($name:ident, $dt:expr, $runner:ident, $a:ident, $b:ident, $c:ident) => {
        /// ConvTranspose3d direction.
        ///
        /// # Safety: all pointers must be live device memory; `stream` valid.
        #[unsafe(no_mangle)]
        #[allow(clippy::too_many_arguments)]
        pub unsafe extern "C" fn $name(
            batch: i32, c_in: i32, c_out: i32,
            d_in: i32, h_in: i32, w_in: i32,
            _d_out: i32, _h_out: i32, _w_out: i32,
            kd: i32, kh: i32, kw: i32,
            stride_d: i32, stride_h: i32, stride_w: i32,
            pad_d: i32, pad_h: i32, pad_w: i32,
            dilation_d: i32, dilation_h: i32, dilation_w: i32,
            output_pad_d: i32, output_pad_h: i32, output_pad_w: i32,
            groups: i32,
            $a: *const c_void,
            $b: *const c_void,
            $c: *mut c_void,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32 {
            let p = ConvT3dParams {
                batch, c_in, c_out,
                d_in, h_in, w_in,
                d_filt: kd, h_filt: kh, w_filt: kw,
                pad_d, pad_h, pad_w,
                stride_d, stride_h, stride_w,
                dilation_d, dilation_h, dilation_w,
                output_pad_d, output_pad_h, output_pad_w,
                groups,
            };
            $runner(&p, $dt, $a, $b, $c, workspace, workspace_bytes, stream)
        }
    };
}

convt3d_dir_impl!(baracuda_kernels_conv_transpose_3d_fw_f32_run, DtypeTag::F32, run_convt3d_fw, input, filter, output);
convt3d_dir_impl!(baracuda_kernels_conv_transpose_3d_fw_f64_run, DtypeTag::F64, run_convt3d_fw, input, filter, output);
convt3d_dir_impl!(baracuda_kernels_conv_transpose_3d_fw_f16_run, DtypeTag::F16, run_convt3d_fw, input, filter, output);
convt3d_dir_impl!(baracuda_kernels_conv_transpose_3d_fw_bf16_run, DtypeTag::Bf16, run_convt3d_fw, input, filter, output);
convt3d_dir_impl!(baracuda_kernels_conv_transpose_3d_bw_data_f32_run, DtypeTag::F32, run_convt3d_bw_data, filter, grad_output, grad_input);
convt3d_dir_impl!(baracuda_kernels_conv_transpose_3d_bw_data_f64_run, DtypeTag::F64, run_convt3d_bw_data, filter, grad_output, grad_input);
convt3d_dir_impl!(baracuda_kernels_conv_transpose_3d_bw_data_f16_run, DtypeTag::F16, run_convt3d_bw_data, filter, grad_output, grad_input);
convt3d_dir_impl!(baracuda_kernels_conv_transpose_3d_bw_data_bf16_run, DtypeTag::Bf16, run_convt3d_bw_data, filter, grad_output, grad_input);
convt3d_dir_impl!(baracuda_kernels_conv_transpose_3d_bw_filter_f32_run, DtypeTag::F32, run_convt3d_bw_filter, input, grad_output, grad_filter);
convt3d_dir_impl!(baracuda_kernels_conv_transpose_3d_bw_filter_f64_run, DtypeTag::F64, run_convt3d_bw_filter, input, grad_output, grad_filter);
convt3d_dir_impl!(baracuda_kernels_conv_transpose_3d_bw_filter_f16_run, DtypeTag::F16, run_convt3d_bw_filter, input, grad_output, grad_filter);
convt3d_dir_impl!(baracuda_kernels_conv_transpose_3d_bw_filter_bf16_run, DtypeTag::Bf16, run_convt3d_bw_filter, input, grad_output, grad_filter);
