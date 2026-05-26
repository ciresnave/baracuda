//! Phase 23 — `baracuda-kernels-sys` C-ABI FFI wrappers for the
//! cuFFT-backed Fast Fourier Transform family.
//!
//! Background: continues the Phase 19 + Phase 22 design correction — every
//! Rust plan in `baracuda-kernels` that wraps an NVIDIA library MUST also
//! expose a flat C-ABI entry so non-Rust callers (Fuel) can drive it
//! without the safe-plan layer. Phase 22 covered cuSOLVER; Phase 23
//! covers cuFFT, cuRAND, and (no plans yet) cuSPARSE.
//!
//! ## Coverage
//!
//! 6 cuFFT-backed plan families, 24 FFI symbols total (12 `_run` + 12
//! `_workspace_size`):
//!
//! - `fft_1d` — C2C 1-D FFT, forward + inverse via direction flag.
//!   Wraps `cufftExecC2C` (`c32`) / `cufftExecZ2Z` (`c64`).
//! - `rfft_1d` — R2C 1-D forward FFT. Wraps `cufftExecR2C` (`f32`) /
//!   `cufftExecD2Z` (`f64`).
//! - `irfft_1d` — C2R 1-D inverse FFT. Wraps `cufftExecC2R` (`f32`) /
//!   `cufftExecZ2D` (`f64`). Inverse path applies `1/n` normalization.
//! - `fft_nd` — C2C ND FFT (rank `1..=3`), forward + inverse.
//! - `rfft_nd` — R2C ND forward FFT, Hermitian-half on the last axis.
//! - `irfft_nd` — C2R ND inverse FFT, with `1/N` normalization.
//!
//! ## Handle lifecycle
//!
//! Each FFI call creates a transient `cufftHandle` via `cufftPlan1d` /
//! `cufftPlanMany`, binds it to the caller's stream via `cufftSetStream`,
//! executes the transform, applies in-place normalization for inverse
//! variants, then destroys the handle. No cross-call caching at the FFI
//! layer — callers that repeat a launch should drive the matching
//! `baracuda-kernels` Rust plan directly (which caches the handle for
//! the lifetime of the plan object).
//!
//! ## Status codes
//!
//! Same convention as the rest of `baracuda-kernels-sys`:
//! - `0` — success.
//! - `2` — invalid problem (caller passed null pointers, non-positive
//!   extents, or out-of-range rank).
//! - `5` — internal cuFFT error (non-zero status from the library).
//!
//! ## Workspace contract
//!
//! cuFFT plans manage their own internal workspace via the plan struct —
//! no caller-supplied workspace is required for the basic 1-D / ND APIs.
//! The `*_workspace_size` queries always return `0`; the matching `*_run`
//! ignores the `workspace` / `workspace_bytes` pair (kept in the ABI for
//! symmetry with the rest of the FFI surface and to leave room for a
//! future external-workspace API switch).
//!
//! ## Normalization contract
//!
//! Inverse transforms (`fft_1d` with `inverse=1`, `irfft_*`, `fft_nd`
//! with `inverse=1`) apply `1/N` normalization after the cuFFT exec
//! via the bespoke `baracuda_kernels_scale_inplace_*` kernels —
//! matching PyTorch's `norm="backward"` convention. `N` is the
//! **signal length** (1-D: `n`; ND: `product(dims[..rank])`), not the
//! complex-cell count (which differs for IRFFT where Hermitian-half
//! input has fewer cells than real output).
//!
//! ## Hermitian-half layout
//!
//! `rfft_*` / `irfft_*` follow cuFFT's "Hermitian-half on the last
//! transformed axis" convention. For real-side `n` cells, the complex
//! side has `n/2 + 1` cells along the last axis (earlier transformed
//! axes carry their full length on both sides for ND).
//!
//! cuFFT cannot infer the real-side last-axis extent from the Hermitian-
//! half input alone — `irfft_*` requires the caller to pass `n` (1-D)
//! or `dims[rank-1]` (ND) explicitly, since both `2 * (n/2)` and
//! `2 * (n/2) + 1` produce inputs of identical complex-side length.

#![allow(non_camel_case_types)]
#![allow(clippy::too_many_arguments)]

use core::ffi::c_void;

use super::{
    baracuda_kernels_scale_inplace_c32_run, baracuda_kernels_scale_inplace_c64_run,
    baracuda_kernels_scale_inplace_real_f32_run, baracuda_kernels_scale_inplace_real_f64_run,
    cufftComplex, cufftDestroy, cufftDoubleComplex, cufftExecC2C, cufftExecC2R, cufftExecD2Z,
    cufftExecR2C, cufftExecZ2D, cufftExecZ2Z, cufftHandle, cufftPlan1d, cufftPlanMany,
    cufftSetStream, CUFFT_C2C, CUFFT_C2R, CUFFT_D2Z, CUFFT_FORWARD, CUFFT_INVERSE, CUFFT_R2C,
    CUFFT_Z2D, CUFFT_Z2Z,
};

// =============================================================================
// Status codes
// =============================================================================

const OK: i32 = 0;
const INVALID: i32 = 2;
const INTERNAL: i32 = 5;

#[inline]
fn map_cufft(status: i32) -> i32 {
    if status == 0 { OK } else { INTERNAL }
}

/// cuFFT handle sentinel value for "uninitialized". cuFFT handles are
/// non-negative integers when live; `-1` is the canonical out-of-band
/// marker (matches the safe-plan layer).
const HANDLE_UNINIT: cufftHandle = -1;

// =============================================================================
// Internal RAII helpers
// =============================================================================

/// RAII guard for a `cufftHandle`. Destroys on `Drop`, idempotent when
/// the sentinel `HANDLE_UNINIT` is still in place.
struct CufftPlan {
    h: cufftHandle,
}

impl CufftPlan {
    #[inline]
    fn new() -> Self {
        Self { h: HANDLE_UNINIT }
    }
}

impl Drop for CufftPlan {
    fn drop(&mut self) {
        if self.h != HANDLE_UNINIT {
            unsafe {
                let _ = cufftDestroy(self.h);
            }
        }
    }
}

/// Bind a freshly-created cuFFT plan to the caller's stream. Returns a
/// status code (`OK` / `INTERNAL`).
#[inline]
unsafe fn bind_stream(plan: cufftHandle, stream: *mut c_void) -> i32 {
    let s = unsafe { cufftSetStream(plan, stream) };
    if s != 0 { INTERNAL } else { OK }
}

// =============================================================================
// fft_1d — C2C 1-D forward + inverse × {c32, c64}
// =============================================================================
//
// Signature: (n, batch, inverse, x, y, workspace, workspace_bytes, stream).
// `inverse` is a flag (`0` = forward, non-zero = inverse). Both buffers
// are complex (`cufftComplex` / `cufftDoubleComplex`). For inverse,
// applies `1/n` normalization in-place via `scale_inplace_c{32,64}`.

macro_rules! fft_1d_pair {
    ($run:ident, $ws:ident, $cufft_type:expr, $exec:ident, $T:ty, $scale_inplace:ident, $cell:ty) => {
        /// 1-D C2C FFT workspace size in bytes. cuFFT manages its own
        /// internal workspace; this entry always writes `0`.
        ///
        /// # Safety
        /// `out_bytes` must point to a writable `usize`.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $ws(_n: i32, _batch: i32, out_bytes: *mut usize) -> i32 {
            if out_bytes.is_null() {
                return INVALID;
            }
            unsafe { *out_bytes = 0 };
            OK
        }

        /// 1-D C2C FFT (forward + inverse via flag). Wraps cuFFT's
        /// `cufftExecC2C` (`c32`) / `cufftExecZ2Z` (`c64`). For inverse,
        /// applies `1/n` normalization in-place after exec.
        ///
        /// `inverse`: 0 = forward, non-zero = inverse.
        ///
        /// # Safety
        /// All pointer args must be device-resident and remain valid
        /// for the duration of the launch. `stream` must be a live
        /// CUDA stream in the current context. Both `x` and `y` hold
        /// at least `batch * n` complex cells.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $run(
            n: i32,
            batch: i32,
            inverse: i32,
            x: *mut c_void,
            y: *mut c_void,
            _workspace: *mut c_void,
            _workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32 {
            if n <= 0 || batch <= 0 || x.is_null() || y.is_null() {
                return INVALID;
            }
            let mut plan = CufftPlan::new();
            let st =
                unsafe { cufftPlan1d(&mut plan.h as *mut _, n, $cufft_type, batch) };
            if st != 0 {
                return INTERNAL;
            }
            let s = unsafe { bind_stream(plan.h, stream) };
            if s != OK {
                return s;
            }
            let direction = if inverse != 0 { CUFFT_INVERSE } else { CUFFT_FORWARD };
            let st = unsafe {
                $exec(plan.h, x as *mut $cell, y as *mut $cell, direction)
            };
            if st != 0 {
                return INTERNAL;
            }
            if inverse != 0 {
                let numel = (batch as i64) * (n as i64);
                let scale = 1.0 as $T / (n as $T);
                let s = unsafe {
                    $scale_inplace(numel, scale, y, core::ptr::null_mut(), 0, stream)
                };
                if s != OK {
                    return s;
                }
            }
            OK
        }
    };
}

fft_1d_pair!(
    baracuda_kernels_fft_1d_c32_run,
    baracuda_kernels_fft_1d_c32_workspace_size,
    CUFFT_C2C,
    cufftExecC2C,
    f32,
    baracuda_kernels_scale_inplace_c32_run,
    cufftComplex
);
fft_1d_pair!(
    baracuda_kernels_fft_1d_c64_run,
    baracuda_kernels_fft_1d_c64_workspace_size,
    CUFFT_Z2Z,
    cufftExecZ2Z,
    f64,
    baracuda_kernels_scale_inplace_c64_run,
    cufftDoubleComplex
);

// =============================================================================
// rfft_1d — R2C 1-D forward × {f32, f64}
// =============================================================================
//
// Signature: (n, batch, x, y, workspace, workspace_bytes, stream).
// `x` is real `[batch, n]`; `y` is complex `[batch, n/2 + 1]`. Forward
// only — unnormalized.

macro_rules! rfft_1d {
    ($run:ident, $ws:ident, $cufft_type:expr, $exec:ident, $T:ty, $cell:ty) => {
        /// 1-D R2C FFT workspace size in bytes — always `0`.
        ///
        /// # Safety
        /// `out_bytes` must point to a writable `usize`.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $ws(_n: i32, _batch: i32, out_bytes: *mut usize) -> i32 {
            if out_bytes.is_null() {
                return INVALID;
            }
            unsafe { *out_bytes = 0 };
            OK
        }

        /// 1-D R2C FFT (real → Hermitian-half complex). Unnormalized
        /// (matches PyTorch's `norm="backward"`).
        ///
        /// # Safety
        /// `x` is `batch * n` real cells; `y` is `batch * (n/2 + 1)`
        /// complex cells. `stream` must be live.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $run(
            n: i32,
            batch: i32,
            x: *mut c_void,
            y: *mut c_void,
            _workspace: *mut c_void,
            _workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32 {
            if n <= 0 || batch <= 0 || x.is_null() || y.is_null() {
                return INVALID;
            }
            let mut plan = CufftPlan::new();
            let st =
                unsafe { cufftPlan1d(&mut plan.h as *mut _, n, $cufft_type, batch) };
            if st != 0 {
                return INTERNAL;
            }
            let s = unsafe { bind_stream(plan.h, stream) };
            if s != OK {
                return s;
            }
            let st = unsafe { $exec(plan.h, x as *mut $T, y as *mut $cell) };
            map_cufft(st)
        }
    };
}

rfft_1d!(
    baracuda_kernels_rfft_1d_f32_run,
    baracuda_kernels_rfft_1d_f32_workspace_size,
    CUFFT_R2C,
    cufftExecR2C,
    f32,
    cufftComplex
);
rfft_1d!(
    baracuda_kernels_rfft_1d_f64_run,
    baracuda_kernels_rfft_1d_f64_workspace_size,
    CUFFT_D2Z,
    cufftExecD2Z,
    f64,
    cufftDoubleComplex
);

// =============================================================================
// irfft_1d — C2R 1-D inverse × {f32, f64}
// =============================================================================
//
// Signature: (n, batch, x, y, workspace, workspace_bytes, stream).
// `x` is complex `[batch, n/2 + 1]`; `y` is real `[batch, n]`. The
// caller passes the **real-side** `n` — cuFFT can't infer it from the
// Hermitian-half input alone. Applies `1/n` normalization in-place.

macro_rules! irfft_1d {
    ($run:ident, $ws:ident, $cufft_type:expr, $exec:ident, $T:ty, $cell:ty, $scale_inplace:ident) => {
        /// 1-D C2R FFT workspace size in bytes — always `0`.
        ///
        /// # Safety
        /// `out_bytes` must point to a writable `usize`.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $ws(_n: i32, _batch: i32, out_bytes: *mut usize) -> i32 {
            if out_bytes.is_null() {
                return INVALID;
            }
            unsafe { *out_bytes = 0 };
            OK
        }

        /// 1-D C2R FFT (Hermitian-half complex → real). Applies `1/n`
        /// normalization in-place (PyTorch `norm="backward"`). `n` is
        /// the real-side output length; complex input shape is
        /// `[batch, n/2 + 1]`.
        ///
        /// # Safety
        /// `x` is `batch * (n/2 + 1)` complex cells; `y` is `batch * n`
        /// real cells. `stream` must be live.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $run(
            n: i32,
            batch: i32,
            x: *mut c_void,
            y: *mut c_void,
            _workspace: *mut c_void,
            _workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32 {
            if n <= 0 || batch <= 0 || x.is_null() || y.is_null() {
                return INVALID;
            }
            let mut plan = CufftPlan::new();
            let st =
                unsafe { cufftPlan1d(&mut plan.h as *mut _, n, $cufft_type, batch) };
            if st != 0 {
                return INTERNAL;
            }
            let s = unsafe { bind_stream(plan.h, stream) };
            if s != OK {
                return s;
            }
            let st = unsafe { $exec(plan.h, x as *mut $cell, y as *mut $T) };
            if st != 0 {
                return INTERNAL;
            }
            let numel = (batch as i64) * (n as i64);
            let scale = 1.0 as $T / (n as $T);
            let s = unsafe {
                $scale_inplace(numel, scale, y, core::ptr::null_mut(), 0, stream)
            };
            if s != OK {
                return s;
            }
            OK
        }
    };
}

irfft_1d!(
    baracuda_kernels_irfft_1d_f32_run,
    baracuda_kernels_irfft_1d_f32_workspace_size,
    CUFFT_C2R,
    cufftExecC2R,
    f32,
    cufftComplex,
    baracuda_kernels_scale_inplace_real_f32_run
);
irfft_1d!(
    baracuda_kernels_irfft_1d_f64_run,
    baracuda_kernels_irfft_1d_f64_workspace_size,
    CUFFT_Z2D,
    cufftExecZ2D,
    f64,
    cufftDoubleComplex,
    baracuda_kernels_scale_inplace_real_f64_run
);

// =============================================================================
// fft_nd — C2C ND forward + inverse × {c32, c64}
// =============================================================================
//
// Signature: (rank, dims, batch, inverse, x, y, workspace,
//             workspace_bytes, stream).
// `rank` is `1..=3` (matches the Rust plan trailblazer). `dims` is a
// host-side `i32[rank]` array of per-axis extents slowest-first.
// `batch` is the cuFFT batch (product of leading non-transformed axes).
// For inverse, applies `1/product(dims[..rank])` normalization in-place.
//
// **dims residency**: HOST pointer — read synchronously inside
// `cufftPlanMany` before it returns; safe to free after the FFI call.
// Matches NVIDIA's `cufftPlanMany` contract.

const FFT_ND_MAX_RANK: i32 = 3;

macro_rules! fft_nd_pair {
    ($run:ident, $ws:ident, $cufft_type:expr, $exec:ident, $T:ty, $scale_inplace:ident, $cell:ty) => {
        /// ND C2C FFT workspace size in bytes — always `0`.
        ///
        /// # Safety
        /// `out_bytes` must point to a writable `usize`.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $ws(
            _rank: i32,
            _dims: *const i32,
            _batch: i32,
            out_bytes: *mut usize,
        ) -> i32 {
            if out_bytes.is_null() {
                return INVALID;
            }
            unsafe { *out_bytes = 0 };
            OK
        }

        /// ND C2C FFT (forward + inverse via flag).
        ///
        /// `rank ∈ {1, 2, 3}`. `dims` is a host-side `i32[rank]` array
        /// (read synchronously inside `cufftPlanMany`). `inverse`: 0 =
        /// forward, non-zero = inverse with `1/product(dims[..rank])`
        /// normalization.
        ///
        /// # Safety
        /// `dims` is a HOST pointer (not device) to `rank` valid `i32`
        /// cells. `x` / `y` are device pointers to at least
        /// `batch * product(dims[..rank])` complex cells each. `stream`
        /// must be live.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $run(
            rank: i32,
            dims: *const i32,
            batch: i32,
            inverse: i32,
            x: *mut c_void,
            y: *mut c_void,
            _workspace: *mut c_void,
            _workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32 {
            if !(1..=FFT_ND_MAX_RANK).contains(&rank)
                || batch <= 0
                || dims.is_null()
                || x.is_null()
                || y.is_null()
            {
                return INVALID;
            }
            // Copy dims onto the stack — cufftPlanMany needs `*mut i32`
            // but we don't want to assume the caller's buffer is mutable
            // beyond the FFI call window.
            let mut n_arr = [0i32; FFT_ND_MAX_RANK as usize];
            let mut total: i64 = 1;
            for i in 0..rank as usize {
                let d = unsafe { *dims.add(i) };
                if d <= 0 {
                    return INVALID;
                }
                n_arr[i] = d;
                total = total.saturating_mul(d as i64);
            }
            let dist = total as i32;
            let mut plan = CufftPlan::new();
            let st = unsafe {
                cufftPlanMany(
                    &mut plan.h as *mut _,
                    rank,
                    n_arr.as_mut_ptr(),
                    core::ptr::null_mut(),
                    1,
                    dist,
                    core::ptr::null_mut(),
                    1,
                    dist,
                    $cufft_type,
                    batch,
                )
            };
            if st != 0 {
                return INTERNAL;
            }
            let s = unsafe { bind_stream(plan.h, stream) };
            if s != OK {
                return s;
            }
            let direction = if inverse != 0 { CUFFT_INVERSE } else { CUFFT_FORWARD };
            let st = unsafe {
                $exec(plan.h, x as *mut $cell, y as *mut $cell, direction)
            };
            if st != 0 {
                return INTERNAL;
            }
            if inverse != 0 {
                let total_with_batch = total.saturating_mul(batch as i64);
                let scale = 1.0 as $T / (total as $T);
                let s = unsafe {
                    $scale_inplace(total_with_batch, scale, y, core::ptr::null_mut(), 0, stream)
                };
                if s != OK {
                    return s;
                }
            }
            OK
        }
    };
}

fft_nd_pair!(
    baracuda_kernels_fft_nd_c32_run,
    baracuda_kernels_fft_nd_c32_workspace_size,
    CUFFT_C2C,
    cufftExecC2C,
    f32,
    baracuda_kernels_scale_inplace_c32_run,
    cufftComplex
);
fft_nd_pair!(
    baracuda_kernels_fft_nd_c64_run,
    baracuda_kernels_fft_nd_c64_workspace_size,
    CUFFT_Z2Z,
    cufftExecZ2Z,
    f64,
    baracuda_kernels_scale_inplace_c64_run,
    cufftDoubleComplex
);

// =============================================================================
// rfft_nd — R2C ND forward × {f32, f64}
// =============================================================================
//
// Signature: (rank, dims, batch, x, y, workspace, workspace_bytes, stream).
// `dims[..rank]` are the **real-side** per-axis extents. Complex output
// has `dims[rank-1] / 2 + 1` on the last axis (Hermitian-half), full
// `dims[i]` on earlier transformed axes.
//
// Real-side numel:    `product(dims[..rank])`
// Complex-side numel: `product(dims[..rank-1]) * (dims[rank-1]/2 + 1)`

macro_rules! rfft_nd {
    ($run:ident, $ws:ident, $cufft_type:expr, $exec:ident, $T:ty, $cell:ty) => {
        /// ND R2C FFT workspace size in bytes — always `0`.
        ///
        /// # Safety
        /// `out_bytes` must point to a writable `usize`.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $ws(
            _rank: i32,
            _dims: *const i32,
            _batch: i32,
            out_bytes: *mut usize,
        ) -> i32 {
            if out_bytes.is_null() {
                return INVALID;
            }
            unsafe { *out_bytes = 0 };
            OK
        }

        /// ND R2C FFT (real → Hermitian-half complex). Unnormalized.
        /// `dims[..rank]` are real-side extents; complex output has
        /// `dims[rank-1] / 2 + 1` on the last transformed axis.
        ///
        /// # Safety
        /// `dims` is a HOST pointer to `rank` valid `i32` cells. `x` is
        /// `batch * product(dims[..rank])` real cells; `y` is
        /// `batch * product(dims[..rank-1]) * (dims[rank-1]/2 + 1)`
        /// complex cells. `stream` must be live.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $run(
            rank: i32,
            dims: *const i32,
            batch: i32,
            x: *mut c_void,
            y: *mut c_void,
            _workspace: *mut c_void,
            _workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32 {
            if !(1..=FFT_ND_MAX_RANK).contains(&rank)
                || batch <= 0
                || dims.is_null()
                || x.is_null()
                || y.is_null()
            {
                return INVALID;
            }
            let mut n_arr = [0i32; FFT_ND_MAX_RANK as usize];
            let mut real_numel: i64 = 1;
            let mut complex_numel: i64 = 1;
            for i in 0..rank as usize {
                let d = unsafe { *dims.add(i) };
                if d <= 0 {
                    return INVALID;
                }
                n_arr[i] = d;
                real_numel = real_numel.saturating_mul(d as i64);
                if i + 1 < rank as usize {
                    complex_numel = complex_numel.saturating_mul(d as i64);
                } else {
                    complex_numel = complex_numel.saturating_mul((d / 2 + 1) as i64);
                }
            }
            let real_dist = real_numel as i32;
            let complex_dist = complex_numel as i32;
            let mut plan = CufftPlan::new();
            let st = unsafe {
                cufftPlanMany(
                    &mut plan.h as *mut _,
                    rank,
                    n_arr.as_mut_ptr(),
                    core::ptr::null_mut(),
                    1,
                    real_dist,
                    core::ptr::null_mut(),
                    1,
                    complex_dist,
                    $cufft_type,
                    batch,
                )
            };
            if st != 0 {
                return INTERNAL;
            }
            let s = unsafe { bind_stream(plan.h, stream) };
            if s != OK {
                return s;
            }
            let st = unsafe { $exec(plan.h, x as *mut $T, y as *mut $cell) };
            map_cufft(st)
        }
    };
}

rfft_nd!(
    baracuda_kernels_rfft_nd_f32_run,
    baracuda_kernels_rfft_nd_f32_workspace_size,
    CUFFT_R2C,
    cufftExecR2C,
    f32,
    cufftComplex
);
rfft_nd!(
    baracuda_kernels_rfft_nd_f64_run,
    baracuda_kernels_rfft_nd_f64_workspace_size,
    CUFFT_D2Z,
    cufftExecD2Z,
    f64,
    cufftDoubleComplex
);

// =============================================================================
// irfft_nd — C2R ND inverse × {f32, f64}
// =============================================================================
//
// Signature: (rank, dims, batch, x, y, workspace, workspace_bytes, stream).
// `dims[..rank]` are the **real-side** per-axis extents (caller-supplied
// because cuFFT can't infer last-axis real extent from the Hermitian
// half). Applies `1/product(dims[..rank])` normalization to the real
// output (matches PyTorch's `norm="backward"`).

macro_rules! irfft_nd {
    ($run:ident, $ws:ident, $cufft_type:expr, $exec:ident, $T:ty, $cell:ty, $scale_inplace:ident) => {
        /// ND C2R FFT workspace size in bytes — always `0`.
        ///
        /// # Safety
        /// `out_bytes` must point to a writable `usize`.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $ws(
            _rank: i32,
            _dims: *const i32,
            _batch: i32,
            out_bytes: *mut usize,
        ) -> i32 {
            if out_bytes.is_null() {
                return INVALID;
            }
            unsafe { *out_bytes = 0 };
            OK
        }

        /// ND C2R FFT (Hermitian-half complex → real). Applies
        /// `1/product(dims[..rank])` normalization in-place. `dims`
        /// carries the **real-side** extents.
        ///
        /// # Safety
        /// `dims` is a HOST pointer to `rank` valid `i32` cells. `x` is
        /// `batch * product(dims[..rank-1]) * (dims[rank-1]/2 + 1)`
        /// complex cells; `y` is `batch * product(dims[..rank])` real
        /// cells. `stream` must be live.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $run(
            rank: i32,
            dims: *const i32,
            batch: i32,
            x: *mut c_void,
            y: *mut c_void,
            _workspace: *mut c_void,
            _workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32 {
            if !(1..=FFT_ND_MAX_RANK).contains(&rank)
                || batch <= 0
                || dims.is_null()
                || x.is_null()
                || y.is_null()
            {
                return INVALID;
            }
            let mut n_arr = [0i32; FFT_ND_MAX_RANK as usize];
            let mut real_numel: i64 = 1;
            let mut complex_numel: i64 = 1;
            for i in 0..rank as usize {
                let d = unsafe { *dims.add(i) };
                if d <= 0 {
                    return INVALID;
                }
                n_arr[i] = d;
                real_numel = real_numel.saturating_mul(d as i64);
                if i + 1 < rank as usize {
                    complex_numel = complex_numel.saturating_mul(d as i64);
                } else {
                    complex_numel = complex_numel.saturating_mul((d / 2 + 1) as i64);
                }
            }
            let real_dist = real_numel as i32;
            let complex_dist = complex_numel as i32;
            let mut plan = CufftPlan::new();
            let st = unsafe {
                cufftPlanMany(
                    &mut plan.h as *mut _,
                    rank,
                    n_arr.as_mut_ptr(),
                    core::ptr::null_mut(),
                    1,
                    complex_dist,
                    core::ptr::null_mut(),
                    1,
                    real_dist,
                    $cufft_type,
                    batch,
                )
            };
            if st != 0 {
                return INTERNAL;
            }
            let s = unsafe { bind_stream(plan.h, stream) };
            if s != OK {
                return s;
            }
            let st = unsafe { $exec(plan.h, x as *mut $cell, y as *mut $T) };
            if st != 0 {
                return INTERNAL;
            }
            let total_with_batch = real_numel.saturating_mul(batch as i64);
            let scale = 1.0 as $T / (real_numel as $T);
            let s = unsafe {
                $scale_inplace(total_with_batch, scale, y, core::ptr::null_mut(), 0, stream)
            };
            if s != OK {
                return s;
            }
            OK
        }
    };
}

irfft_nd!(
    baracuda_kernels_irfft_nd_f32_run,
    baracuda_kernels_irfft_nd_f32_workspace_size,
    CUFFT_C2R,
    cufftExecC2R,
    f32,
    cufftComplex,
    baracuda_kernels_scale_inplace_real_f32_run
);
irfft_nd!(
    baracuda_kernels_irfft_nd_f64_run,
    baracuda_kernels_irfft_nd_f64_workspace_size,
    CUFFT_Z2D,
    cufftExecZ2D,
    f64,
    cufftDoubleComplex,
    baracuda_kernels_scale_inplace_real_f64_run
);
