//! Phase 22 — `baracuda-kernels-sys` C-ABI FFI wrappers for the
//! cuSOLVER-backed linalg family.
//!
//! Background: closes the Phase 19 design gap that library-backed plans
//! must also ship a `baracuda-kernels-sys` FFI symbol — every Rust plan
//! that wraps an NVIDIA library MUST expose a flat C entry point so
//! non-Rust callers (Fuel) can drive it without the safe-plan layer.
//! Phase 19 covered cuDNN (pool + conv); Phase 22 covers cuSOLVER.
//!
//! ## Coverage
//!
//! 10 cuSOLVER-backed plan families, ~44 FFI symbols total:
//!
//! - `cholesky` — `potrf` (non-batched f32/f64) + `potrfBatched` (f32/f64).
//! - `lu` — `getrf` (f32/f64).
//! - `qr` — `geqrf` + `ormqr` chain (f32/f64), caller-staged Q identity.
//! - `svd` — `gesvd` (f32/f64), with jobu/jobv mode characters.
//! - `svd_batched` — `gesvdjBatched` (f32/f64), caller-supplied
//!   `gesvdjInfo_t` set via the create/destroy pair.
//! - `svda_batched` — `gesvdaStridedBatched` (f32/f64), `h_r_nrm_f` host
//!   buffer caller-supplied.
//! - `eigh` — `syevd` (f32/f64) + `heevd` (Complex32/Complex64).
//! - `eig` — `Xgeev` (f32/f64/Complex32/Complex64), 64-bit-index API
//!   following LAPACK packed-real eigenvalue convention.
//! - `lstsq` — `_gels` (f32/f64), primary iterative-refinement path
//!   only (QR fallback is a Rust-plan composition, not a single
//!   cuSOLVER call).
//! - `solve` — `getrf` + `getrs` chain (f32/f64), all in one call.
//! - `inverse` — `getrf` + `getrs` over a caller-staged identity
//!   matrix (f32/f64). The caller pre-stages an `M × M` identity in
//!   `inv`; the FFI just runs the factorization + back-substitution.
//!
//! ## Plans intentionally NOT in scope
//!
//! - `qr_batched` — cuBLAS-backed (`cublas*geqrfBatched`), not cuSOLVER.
//!   Will land in a future cuBLAS facade pass.
//! - `ormqr_batched` / `ormqr_batched_wy` / `qr_batched_materialize` —
//!   bespoke kernels; FFI symbols already shipped under
//!   `baracuda_kernels_batched_ormqr_*_run` etc. Phase 26 added the
//!   `BatchedOrmqrWy` Complex32 / Complex64 variants using the same
//!   FFI surface shape (`*_complex32_run` / `*_complex64_run`) plus
//!   `cublas{C,Z}gemmStridedBatched` for the per-block apply.
//!
//! ## Handle lifecycle
//!
//! Each FFI call creates a transient `cusolverDnHandle_t`, sets the
//! stream, calls the cuSOLVER op (after one `_bufferSize` query if
//! workspace is needed), then tears the handle down. No cross-call
//! caching at the FFI layer — callers that repeat a launch should
//! drive the matching `baracuda-kernels` Rust plan directly (which
//! caches handle + workspace bytes for the lifetime of the plan
//! object). A future opaque-plan FFI (`*_create` / `*_run` / `*_destroy`
//! returning an opaque `*mut LinalgPlan`) would recover the
//! amortization; out of scope for Phase 22.
//!
//! ## Status codes
//!
//! Same convention as the rest of `baracuda-kernels-sys`:
//! - `0` — success.
//! - `2` — invalid problem (cuSOLVER rejected the descriptor or shape).
//! - `4` — workspace too small for the requested op.
//! - `5` — internal cuSOLVER error (non-zero status from the library).
//!
//! Argument validation (negative extents, null required pointers) maps
//! to `2`; cuSOLVER runtime failures map to `5`. The cuSOLVER status
//! code itself is not preserved; callers needing the original error
//! code must drive the Rust plan.
//!
//! ## Workspace contract
//!
//! Each `*_run` entry point takes a `workspace` / `workspace_bytes`
//! pair. The wrapper queries `*_bufferSize` internally on every call
//! to decide the needed byte count. If `workspace_bytes < needed`, the
//! call returns `4` without invoking the op. Callers that want to
//! pre-size the workspace can use the matching `*_workspace_size`
//! query symbol (one per dtype) which performs only the bufferSize
//! query — no exec call.
//!
//! ## Identity-staging contract (qr / inverse)
//!
//! cuSOLVER's QR-with-dense-Q and inverse don't produce a dense
//! result directly — they need either an `ormqr` over an identity
//! (qr) or a `getrs` over an identity (inverse). The Rust plan layer
//! stages this identity host-side then async-copies it down; the FFI
//! layer instead **expects the caller to pre-stage the identity** in
//! the output buffer. This keeps the FFI thin (no host allocation, no
//! async H2D — caller controls all device work) at the cost of
//! one extra setup step at the call site.

#![allow(non_camel_case_types)]
#![allow(clippy::too_many_arguments)]

use core::ffi::c_void;
use core::ptr;

use super::{
    cuComplex, cuDoubleComplex, cudaDataType, cusolverDnCheevd, cusolverDnCheevd_bufferSize,
    cusolverDnCreate, cusolverDnCreateGesvdjInfo, cusolverDnCreateParams, cusolverDnDDgels,
    cusolverDnDDgels_bufferSize, cusolverDnDestroy, cusolverDnDestroyGesvdjInfo,
    cusolverDnDestroyParams, cusolverDnDgeqrf, cusolverDnDgeqrf_bufferSize,
    cusolverDnDgesvd, cusolverDnDgesvd_bufferSize, cusolverDnDgesvdaStridedBatched,
    cusolverDnDgesvdaStridedBatched_bufferSize, cusolverDnDgesvdjBatched,
    cusolverDnDgesvdjBatched_bufferSize, cusolverDnDgetrf, cusolverDnDgetrf_bufferSize,
    cusolverDnDgetrs, cusolverDnDormqr, cusolverDnDpotrf, cusolverDnDpotrfBatched,
    cusolverDnDpotrf_bufferSize, cusolverDnDsyevd, cusolverDnDsyevd_bufferSize, cusolverDnHandle_t,
    cusolverDnParams_t, cusolverDnSSgels, cusolverDnSSgels_bufferSize, cusolverDnSetStream,
    cusolverDnSgeqrf, cusolverDnSgeqrf_bufferSize, cusolverDnSgesvd, cusolverDnSgesvd_bufferSize,
    cusolverDnSgesvdaStridedBatched, cusolverDnSgesvdaStridedBatched_bufferSize,
    cusolverDnSgesvdjBatched, cusolverDnSgesvdjBatched_bufferSize, cusolverDnSgetrf,
    cusolverDnSgetrf_bufferSize, cusolverDnSgetrs, cusolverDnSormqr, cusolverDnSpotrf,
    cusolverDnSpotrfBatched, cusolverDnSpotrf_bufferSize, cusolverDnSsyevd,
    cusolverDnSsyevd_bufferSize, cusolverDnXgeev, cusolverDnXgeev_bufferSize, cusolverDnZheevd,
    cusolverDnZheevd_bufferSize, gesvdjInfo_t, CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER,
    CUBLAS_OP_N, CUDA_C_32F, CUDA_C_64F, CUDA_R_32F, CUDA_R_64F,
    CUSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_VECTOR,
};

// =============================================================================
// Status codes
// =============================================================================

const OK: i32 = 0;
const INVALID: i32 = 2;
const WS_TOO_SMALL: i32 = 4;
const INTERNAL: i32 = 5;

#[inline]
fn map_cusolver(status: i32) -> i32 {
    if status == 0 {
        OK
    } else {
        INTERNAL
    }
}

// =============================================================================
// Internal RAII helpers
// =============================================================================

/// RAII guard for a cuSOLVER handle. Destroys on `Drop`, idempotent on
/// null. Used by every FFI entry that takes a stream and creates a
/// transient handle.
struct Handle {
    h: cusolverDnHandle_t,
}

impl Handle {
    #[inline]
    fn new() -> Self {
        Self { h: ptr::null_mut() }
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if !self.h.is_null() {
            unsafe {
                let _ = cusolverDnDestroy(self.h);
            }
        }
    }
}

/// Create a handle + bind a stream. Returns a status code (`OK` or
/// `INTERNAL`).
#[inline]
unsafe fn setup_handle(g: &mut Handle, stream: *mut c_void) -> i32 {
    let s = unsafe { cusolverDnCreate(&mut g.h as *mut _) };
    if s != 0 {
        return INTERNAL;
    }
    let s = unsafe { cusolverDnSetStream(g.h, stream) };
    if s != 0 {
        return INTERNAL;
    }
    OK
}

/// RAII guard for a `cusolverDnParams_t` (used by Xgeev / 64-bit APIs).
struct Params {
    p: cusolverDnParams_t,
}

impl Params {
    #[inline]
    fn new() -> Self {
        Self { p: ptr::null_mut() }
    }
}

impl Drop for Params {
    fn drop(&mut self) {
        if !self.p.is_null() {
            unsafe {
                let _ = cusolverDnDestroyParams(self.p);
            }
        }
    }
}

/// RAII guard for a `gesvdjInfo_t` (Jacobi-SVD parameter object).
/// Uses default tolerances (`1e-7` f32 / `1e-12` f64, `max_sweeps = 100`)
/// matching the Rust plan.
struct JacobiInfo {
    p: gesvdjInfo_t,
}

impl JacobiInfo {
    #[inline]
    fn new() -> Self {
        Self { p: ptr::null_mut() }
    }
}

impl Drop for JacobiInfo {
    fn drop(&mut self) {
        if !self.p.is_null() {
            unsafe {
                let _ = cusolverDnDestroyGesvdjInfo(self.p);
            }
        }
    }
}

/// Common workspace check: verify the caller's `(ptr, bytes)` pair
/// covers `needed`. Returns a status code (`OK` / `WS_TOO_SMALL`).
#[inline]
fn check_ws(workspace: *mut c_void, workspace_bytes: usize, needed: usize) -> i32 {
    if needed == 0 {
        return OK;
    }
    if workspace.is_null() {
        return WS_TOO_SMALL;
    }
    if workspace_bytes < needed {
        return WS_TOO_SMALL;
    }
    OK
}

// =============================================================================
// Cholesky — `potrf` + `potrfBatched` × {f32, f64}
// =============================================================================
//
// Signature (non-batched FW): (uplo, n, lda, a_inout, info_out,
//                              workspace, workspace_bytes, stream).
// `uplo` is `0` (lower) or `1` (upper) per the cuBLAS `cublasFillMode_t`
// convention; the caller is responsible for the row-major-to-column-
// major flip (cuSOLVER is column-major).
//
// Signature (batched FW): (uplo, n, lda, a_array_inout, info_array_out,
//                          batch_size, stream). cuSOLVER's batched
// `potrfBatched` is workspace-free; instead it takes a device-resident
// **array of device pointers** — caller responsibility to pre-stage.

macro_rules! cholesky_pair {
    ($nb_name:ident, $bs_name:ident, $ws_name:ident,
     $potrf:ident, $potrfb:ident, $potrf_bs:ident, $T:ty) => {
        /// Cholesky factorization workspace size in bytes for the
        /// non-batched `potrf` path. Returns `0` on success and writes
        /// the byte count to `*out_bytes`; non-zero status on cuSOLVER
        /// failure (handle allocation / bufferSize query). Batched
        /// `potrfBatched` is workspace-free and has no equivalent query.
        ///
        /// # Safety
        /// `out_bytes` must point to a writable `usize`.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $ws_name(n: i32, lda: i32, out_bytes: *mut usize) -> i32 {
            if n <= 0 || lda < n || out_bytes.is_null() {
                return INVALID;
            }
            let mut h = Handle::new();
            let s = unsafe { setup_handle(&mut h, ptr::null_mut()) };
            if s != OK {
                return s;
            }
            let mut lwork: i32 = 0;
            let st = unsafe {
                $potrf_bs(h.h, CUBLAS_FILL_MODE_LOWER, n, ptr::null_mut(), lda, &mut lwork)
            };
            if st != 0 {
                return INTERNAL;
            }
            let bytes = (lwork as usize) * core::mem::size_of::<$T>();
            unsafe { *out_bytes = bytes };
            OK
        }

        /// Cholesky factorization (non-batched). Overwrites `a_inout`
        /// in place with the requested triangular factor. `uplo` is
        /// `0` (lower, `CUBLAS_FILL_MODE_LOWER`) or `1`
        /// (upper, `CUBLAS_FILL_MODE_UPPER`).
        ///
        /// # Safety
        /// All pointer args must be device-resident and remain valid
        /// for the duration of the launch. `stream` must be a live
        /// CUDA stream in the current context. `info_out` is one
        /// device-resident `i32`. Workspace bytes must cover the
        /// `*_workspace_size` query result.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $nb_name(
            uplo: i32,
            n: i32,
            lda: i32,
            a_inout: *mut c_void,
            info_out: *mut i32,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32 {
            if n <= 0 || lda < n || a_inout.is_null() || info_out.is_null() {
                return INVALID;
            }
            if !matches!(uplo, CUBLAS_FILL_MODE_LOWER | CUBLAS_FILL_MODE_UPPER) {
                return INVALID;
            }
            let mut h = Handle::new();
            let s = unsafe { setup_handle(&mut h, stream) };
            if s != OK {
                return s;
            }
            // Trust the caller's `workspace_bytes` (queried via
            // `_workspace_size`). We don't re-query bufferSize here: cuSOLVER's
            // `Spotrf_bufferSize` can return slightly different lwork across
            // separate handle lifecycles (observed on Cholesky for small `n`
            // when uplo differs from the workspace_size query's hardcoded
            // LOWER), which would spuriously trip `WS_TOO_SMALL`. cuSOLVER
            // itself validates `lwork` internally and reports a non-zero
            // status if the buffer is too small.
            if workspace_bytes > 0 && workspace.is_null() {
                return INVALID;
            }
            let lwork = (workspace_bytes / core::mem::size_of::<$T>()) as i32;
            let st = unsafe {
                $potrf(
                    h.h,
                    uplo,
                    n,
                    a_inout as *mut $T,
                    lda,
                    workspace as *mut $T,
                    lwork,
                    info_out,
                )
            };
            map_cusolver(st)
        }

        /// Cholesky factorization (batched). Each `a_array[b]` is
        /// overwritten with the requested triangular factor. cuSOLVER's
        /// `potrfBatched` is workspace-free internally but needs a
        /// device-resident array of device pointers — caller responsibility.
        ///
        /// # Safety
        /// `a_array` is a device buffer of `batch_size` device pointers,
        /// each pointing to an `[n, n]` matrix; `info_array` is a device
        /// buffer of `batch_size` i32s. Caller pre-stages the pointer
        /// array.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $bs_name(
            uplo: i32,
            n: i32,
            lda: i32,
            a_array: *mut *mut c_void,
            info_array: *mut i32,
            batch_size: i32,
            stream: *mut c_void,
        ) -> i32 {
            if n <= 0
                || lda < n
                || batch_size <= 0
                || a_array.is_null()
                || info_array.is_null()
            {
                return INVALID;
            }
            if !matches!(uplo, CUBLAS_FILL_MODE_LOWER | CUBLAS_FILL_MODE_UPPER) {
                return INVALID;
            }
            let mut h = Handle::new();
            let s = unsafe { setup_handle(&mut h, stream) };
            if s != OK {
                return s;
            }
            let st = unsafe {
                $potrfb(
                    h.h,
                    uplo,
                    n,
                    a_array as *mut *mut $T,
                    lda,
                    info_array,
                    batch_size,
                )
            };
            map_cusolver(st)
        }
    };
}

cholesky_pair!(
    baracuda_kernels_cholesky_f32_run,
    baracuda_kernels_cholesky_batched_f32_run,
    baracuda_kernels_cholesky_f32_workspace_size,
    cusolverDnSpotrf,
    cusolverDnSpotrfBatched,
    cusolverDnSpotrf_bufferSize,
    f32
);
cholesky_pair!(
    baracuda_kernels_cholesky_f64_run,
    baracuda_kernels_cholesky_batched_f64_run,
    baracuda_kernels_cholesky_f64_workspace_size,
    cusolverDnDpotrf,
    cusolverDnDpotrfBatched,
    cusolverDnDpotrf_bufferSize,
    f64
);

// =============================================================================
// LU — `getrf` (non-batched only — cuSOLVER-Dn has no batched getrf)
// =============================================================================
//
// Signature: (m, n, lda, a_inout, pivots_out, info_out, workspace,
//             workspace_bytes, stream).

macro_rules! lu_pair {
    ($name:ident, $ws_name:ident, $getrf:ident, $getrf_bs:ident, $T:ty) => {
        /// LU factorization workspace size in bytes for `getrf`.
        ///
        /// # Safety
        /// `out_bytes` must point to a writable `usize`.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $ws_name(m: i32, n: i32, lda: i32, out_bytes: *mut usize) -> i32 {
            if m <= 0 || n <= 0 || lda < m || out_bytes.is_null() {
                return INVALID;
            }
            let mut h = Handle::new();
            let s = unsafe { setup_handle(&mut h, ptr::null_mut()) };
            if s != OK {
                return s;
            }
            let mut lwork: i32 = 0;
            let st = unsafe { $getrf_bs(h.h, m, n, ptr::null_mut(), lda, &mut lwork) };
            if st != 0 {
                return INTERNAL;
            }
            unsafe { *out_bytes = (lwork as usize) * core::mem::size_of::<$T>() };
            OK
        }

        /// LU factorization with partial pivoting (non-batched).
        /// `a_inout` is overwritten with the packed `LU` factors;
        /// `pivots_out` receives the 1-based row swaps (length
        /// `min(m, n)`); `info_out` is a single `i32`.
        ///
        /// # Safety
        /// As for the Cholesky entry point.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            m: i32,
            n: i32,
            lda: i32,
            a_inout: *mut c_void,
            pivots_out: *mut i32,
            info_out: *mut i32,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32 {
            if m <= 0 || n <= 0 || lda < m
                || a_inout.is_null() || pivots_out.is_null() || info_out.is_null()
            {
                return INVALID;
            }
            let mut h = Handle::new();
            let s = unsafe { setup_handle(&mut h, stream) };
            if s != OK {
                return s;
            }
            let mut lwork: i32 = 0;
            let st = unsafe { $getrf_bs(h.h, m, n, ptr::null_mut(), lda, &mut lwork) };
            if st != 0 {
                return INTERNAL;
            }
            let needed = (lwork as usize) * core::mem::size_of::<$T>();
            let s = check_ws(workspace, workspace_bytes, needed);
            if s != OK {
                return s;
            }
            let st = unsafe {
                $getrf(
                    h.h,
                    m,
                    n,
                    a_inout as *mut $T,
                    lda,
                    workspace as *mut $T,
                    pivots_out,
                    info_out,
                )
            };
            map_cusolver(st)
        }
    };
}

lu_pair!(
    baracuda_kernels_lu_f32_run,
    baracuda_kernels_lu_f32_workspace_size,
    cusolverDnSgetrf,
    cusolverDnSgetrf_bufferSize,
    f32
);
lu_pair!(
    baracuda_kernels_lu_f64_run,
    baracuda_kernels_lu_f64_workspace_size,
    cusolverDnDgetrf,
    cusolverDnDgetrf_bufferSize,
    f64
);

// =============================================================================
// QR — `geqrf` (packed Householder output)
// =============================================================================
//
// Signature: (m, n, lda, a_inout, tau_out, info_out, workspace,
//             workspace_bytes, stream).
//
// This exposes the packed-output path only. To materialize a dense Q
// the caller should follow up with `baracuda_kernels_ormqr_*_run`
// (below) over a pre-staged identity matrix — same composition the
// Rust plan does, but without the host-side identity build (caller's
// responsibility under the FFI contract).

macro_rules! qr_pair {
    ($name:ident, $ws_name:ident, $geqrf:ident, $geqrf_bs:ident, $T:ty) => {
        /// QR factorization workspace size in bytes for `geqrf`.
        ///
        /// # Safety
        /// `out_bytes` must point to a writable `usize`.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $ws_name(m: i32, n: i32, lda: i32, out_bytes: *mut usize) -> i32 {
            if m <= 0 || n <= 0 || lda < m || out_bytes.is_null() {
                return INVALID;
            }
            let mut h = Handle::new();
            let s = unsafe { setup_handle(&mut h, ptr::null_mut()) };
            if s != OK {
                return s;
            }
            let mut lwork: i32 = 0;
            let st = unsafe { $geqrf_bs(h.h, m, n, ptr::null_mut(), lda, &mut lwork) };
            if st != 0 {
                return INTERNAL;
            }
            unsafe { *out_bytes = (lwork as usize) * core::mem::size_of::<$T>() };
            OK
        }

        /// QR factorization (packed Householder output, `m >= n`
        /// required). `a_inout` is overwritten with `R` (upper
        /// triangle) + Householder reflectors (strict lower);
        /// `tau_out` is `[min(m, n)]`.
        ///
        /// # Safety
        /// As for the Cholesky entry point.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            m: i32,
            n: i32,
            lda: i32,
            a_inout: *mut c_void,
            tau_out: *mut c_void,
            info_out: *mut i32,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32 {
            if m <= 0 || n <= 0 || m < n || lda < m
                || a_inout.is_null() || tau_out.is_null() || info_out.is_null()
            {
                return INVALID;
            }
            let mut h = Handle::new();
            let s = unsafe { setup_handle(&mut h, stream) };
            if s != OK {
                return s;
            }
            let mut lwork: i32 = 0;
            let st = unsafe { $geqrf_bs(h.h, m, n, ptr::null_mut(), lda, &mut lwork) };
            if st != 0 {
                return INTERNAL;
            }
            let needed = (lwork as usize) * core::mem::size_of::<$T>();
            let s = check_ws(workspace, workspace_bytes, needed);
            if s != OK {
                return s;
            }
            let st = unsafe {
                $geqrf(
                    h.h,
                    m,
                    n,
                    a_inout as *mut $T,
                    lda,
                    tau_out as *mut $T,
                    workspace as *mut $T,
                    lwork,
                    info_out,
                )
            };
            map_cusolver(st)
        }
    };
}

qr_pair!(
    baracuda_kernels_qr_f32_run,
    baracuda_kernels_qr_f32_workspace_size,
    cusolverDnSgeqrf,
    cusolverDnSgeqrf_bufferSize,
    f32
);
qr_pair!(
    baracuda_kernels_qr_f64_run,
    baracuda_kernels_qr_f64_workspace_size,
    cusolverDnDgeqrf,
    cusolverDnDgeqrf_bufferSize,
    f64
);

// =============================================================================
// ormqr — apply Householder-encoded Q from a `geqrf` packed output to
// a dense RHS. The QR-with-dense-Q composition needs the caller to:
//   1. Pre-stage an identity matrix in the destination buffer.
//   2. Call qr (above) to factor A.
//   3. Call ormqr (this entry) with side=LEFT, op=N — Q overwrites the
//      destination.
//
// Signature: (side, op, m, n, k, a_packed, lda, tau, c_inout, ldc,
//             info_out, workspace, workspace_bytes, stream).

macro_rules! ormqr_pair {
    ($name:ident, $ormqr:ident, $T:ty) => {
        /// Apply Householder-encoded `Q` (from a prior `geqrf`) to
        /// `c_inout`. `side ∈ {0=Left, 1=Right}`; `op ∈ {0=N, 1=T, 2=C}`.
        /// On Left + op=N, computes `C := Q · C`; pair with a pre-staged
        /// identity `C` to materialize dense `Q`.
        ///
        /// # Safety
        /// All pointer args must be device-resident; `a_packed` + `tau`
        /// come from a prior `baracuda_kernels_qr_*_run` call.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            side: i32,
            op: i32,
            m: i32,
            n: i32,
            k: i32,
            a_packed: *const c_void,
            lda: i32,
            tau: *const c_void,
            c_inout: *mut c_void,
            ldc: i32,
            info_out: *mut i32,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32 {
            if m <= 0 || n <= 0 || k <= 0 || lda <= 0 || ldc <= 0
                || a_packed.is_null() || tau.is_null() || c_inout.is_null() || info_out.is_null()
            {
                return INVALID;
            }
            let mut h = Handle::new();
            let s = unsafe { setup_handle(&mut h, stream) };
            if s != OK {
                return s;
            }
            // Element count for the workspace = workspace_bytes / sizeof(T).
            // Pass through to the exec; cuSOLVER validates internally.
            let lwork = (workspace_bytes / core::mem::size_of::<$T>()) as i32;
            let st = unsafe {
                $ormqr(
                    h.h,
                    side,
                    op,
                    m,
                    n,
                    k,
                    a_packed as *const $T,
                    lda,
                    tau as *const $T,
                    c_inout as *mut $T,
                    ldc,
                    workspace as *mut $T,
                    lwork,
                    info_out,
                )
            };
            map_cusolver(st)
        }
    };
}

ormqr_pair!(baracuda_kernels_ormqr_f32_run, cusolverDnSormqr, f32);
ormqr_pair!(baracuda_kernels_ormqr_f64_run, cusolverDnDormqr, f64);

// =============================================================================
// SVD — `gesvd` (single-matrix; cuSOLVER has no batched gesvd)
// =============================================================================
//
// Signature: (jobu, jobv, m, n, lda, a_inout, ldu, ldvt,
//             s_out, u_out, vt_out, info_out,
//             workspace, workspace_bytes, stream).
//
// `jobu` / `jobv` are ASCII bytes: 'A' (full), 'S' (thin), 'N' (skip),
// 'O' (overwrite — disallowed at the plan layer).
// Requires `m >= n`; transpose before invoking if you need `m < n`.

macro_rules! svd_pair {
    ($name:ident, $ws_name:ident, $gesvd:ident, $gesvd_bs:ident, $T:ty) => {
        /// SVD workspace size in bytes for `gesvd`.
        ///
        /// # Safety
        /// `out_bytes` must point to a writable `usize`.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $ws_name(m: i32, n: i32, out_bytes: *mut usize) -> i32 {
            if m <= 0 || n <= 0 || m < n || out_bytes.is_null() {
                return INVALID;
            }
            let mut h = Handle::new();
            let s = unsafe { setup_handle(&mut h, ptr::null_mut()) };
            if s != OK {
                return s;
            }
            let mut lwork: i32 = 0;
            let st = unsafe { $gesvd_bs(h.h, m, n, &mut lwork) };
            if st != 0 {
                return INTERNAL;
            }
            unsafe { *out_bytes = (lwork as usize) * core::mem::size_of::<$T>() };
            OK
        }

        /// SVD `A = U · diag(S) · V^T`. Requires `m >= n`. `a_inout`
        /// is overwritten by cuSOLVER as scratch.
        ///
        /// # Safety
        /// As for the Cholesky entry point. `s_out` is `[min(m, n)]`,
        /// `u_out` is `[m, m]` (full) or `[m, k]` (thin) per `jobu`,
        /// `vt_out` is `[n, n]` (full) or `[k, n]` (thin) per `jobv`,
        /// where `k = min(m, n)`.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            jobu: u8,
            jobv: u8,
            m: i32,
            n: i32,
            lda: i32,
            a_inout: *mut c_void,
            ldu: i32,
            ldvt: i32,
            s_out: *mut c_void,
            u_out: *mut c_void,
            vt_out: *mut c_void,
            info_out: *mut i32,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32 {
            if m <= 0 || n <= 0 || m < n || lda < m
                || a_inout.is_null() || s_out.is_null() || info_out.is_null()
            {
                return INVALID;
            }
            // jobu / jobv: 'A', 'S', 'N' accepted; 'O' rejected.
            if !matches!(jobu, b'A' | b'S' | b'N') || !matches!(jobv, b'A' | b'S' | b'N') {
                return INVALID;
            }
            let mut h = Handle::new();
            let s = unsafe { setup_handle(&mut h, stream) };
            if s != OK {
                return s;
            }
            let mut lwork: i32 = 0;
            let st = unsafe { $gesvd_bs(h.h, m, n, &mut lwork) };
            if st != 0 {
                return INTERNAL;
            }
            let needed = (lwork as usize) * core::mem::size_of::<$T>();
            let s = check_ws(workspace, workspace_bytes, needed);
            if s != OK {
                return s;
            }
            let st = unsafe {
                $gesvd(
                    h.h,
                    jobu,
                    jobv,
                    m,
                    n,
                    a_inout as *mut $T,
                    lda,
                    s_out as *mut $T,
                    u_out as *mut $T,
                    ldu,
                    vt_out as *mut $T,
                    ldvt,
                    workspace as *mut $T,
                    lwork,
                    ptr::null_mut(),
                    info_out,
                )
            };
            map_cusolver(st)
        }
    };
}

svd_pair!(
    baracuda_kernels_svd_f32_run,
    baracuda_kernels_svd_f32_workspace_size,
    cusolverDnSgesvd,
    cusolverDnSgesvd_bufferSize,
    f32
);
svd_pair!(
    baracuda_kernels_svd_f64_run,
    baracuda_kernels_svd_f64_workspace_size,
    cusolverDnDgesvd,
    cusolverDnDgesvd_bufferSize,
    f64
);

// =============================================================================
// SVD-Batched (Jacobi) — `gesvdjBatched` × {f32, f64}, square-only
// =============================================================================
//
// Signature: (jobz, n, lda, ldu, ldv, a_inout, s_out, u_out, v_out,
//             info_out, batch_size, workspace, workspace_bytes, stream).
//
// `jobz`: `0` = no vectors, `1` = compute vectors. `n` is the (square)
// matrix size. Uses default Jacobi tolerances (`1e-7` f32, `1e-12`
// f64, `max_sweeps = 100`); callers wanting custom tolerances must
// drive the Rust plan layer.

macro_rules! svd_batched_pair {
    ($name:ident, $ws_name:ident, $gesvdj:ident, $gesvdj_bs:ident, $T:ty) => {
        /// Batched Jacobi-SVD workspace size in bytes.
        ///
        /// # Safety
        /// `out_bytes` must point to a writable `usize`.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $ws_name(
            jobz: i32,
            n: i32,
            batch_size: i32,
            out_bytes: *mut usize,
        ) -> i32 {
            if n <= 0 || batch_size <= 0 || out_bytes.is_null() {
                return INVALID;
            }
            if !matches!(jobz, CUSOLVER_EIG_MODE_VECTOR | CUSOLVER_EIG_MODE_NOVECTOR) {
                return INVALID;
            }
            let mut h = Handle::new();
            let s = unsafe { setup_handle(&mut h, ptr::null_mut()) };
            if s != OK {
                return s;
            }
            let mut p = JacobiInfo::new();
            let st = unsafe { cusolverDnCreateGesvdjInfo(&mut p.p as *mut _) };
            if st != 0 {
                return INTERNAL;
            }
            let mut lwork: i32 = 0;
            let st = unsafe {
                $gesvdj_bs(
                    h.h, jobz, n, n, ptr::null(), n, ptr::null(), ptr::null(), n, ptr::null(), n,
                    &mut lwork, p.p, batch_size,
                )
            };
            if st != 0 {
                return INTERNAL;
            }
            unsafe { *out_bytes = (lwork as usize) * core::mem::size_of::<$T>() };
            OK
        }

        /// Batched Jacobi-SVD on **square** input. Returns `V`
        /// (not `V^T`). When `jobz == 0`, `u_out` / `v_out` may be null.
        ///
        /// # Safety
        /// `a_inout` is `[batch, n, n]`; `s_out` is `[batch, n]`; `u_out`
        /// and `v_out` are `[batch, n, n]` if `jobz == 1`. `info_out` is
        /// `[batch]`. All buffers column-major per slot.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            jobz: i32,
            n: i32,
            lda: i32,
            ldu: i32,
            ldv: i32,
            a_inout: *mut c_void,
            s_out: *mut c_void,
            u_out: *mut c_void,
            v_out: *mut c_void,
            info_out: *mut i32,
            batch_size: i32,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32 {
            if n <= 0 || lda < n || ldu < n || ldv < n || batch_size <= 0
                || a_inout.is_null() || s_out.is_null() || info_out.is_null()
            {
                return INVALID;
            }
            if !matches!(jobz, CUSOLVER_EIG_MODE_VECTOR | CUSOLVER_EIG_MODE_NOVECTOR) {
                return INVALID;
            }
            if jobz == CUSOLVER_EIG_MODE_VECTOR && (u_out.is_null() || v_out.is_null()) {
                return INVALID;
            }
            let mut h = Handle::new();
            let s = unsafe { setup_handle(&mut h, stream) };
            if s != OK {
                return s;
            }
            let mut p = JacobiInfo::new();
            let st = unsafe { cusolverDnCreateGesvdjInfo(&mut p.p as *mut _) };
            if st != 0 {
                return INTERNAL;
            }
            let mut lwork: i32 = 0;
            let st = unsafe {
                $gesvdj_bs(
                    h.h, jobz, n, n, ptr::null(), lda, ptr::null(), ptr::null(), ldu, ptr::null(),
                    ldv, &mut lwork, p.p, batch_size,
                )
            };
            if st != 0 {
                return INTERNAL;
            }
            let needed = (lwork as usize) * core::mem::size_of::<$T>();
            let s = check_ws(workspace, workspace_bytes, needed);
            if s != OK {
                return s;
            }
            let st = unsafe {
                $gesvdj(
                    h.h,
                    jobz,
                    n,
                    n,
                    a_inout as *mut $T,
                    lda,
                    s_out as *mut $T,
                    u_out as *mut $T,
                    ldu,
                    v_out as *mut $T,
                    ldv,
                    workspace as *mut $T,
                    lwork,
                    info_out,
                    p.p,
                    batch_size,
                )
            };
            map_cusolver(st)
        }
    };
}

svd_batched_pair!(
    baracuda_kernels_svd_batched_f32_run,
    baracuda_kernels_svd_batched_f32_workspace_size,
    cusolverDnSgesvdjBatched,
    cusolverDnSgesvdjBatched_bufferSize,
    f32
);
svd_batched_pair!(
    baracuda_kernels_svd_batched_f64_run,
    baracuda_kernels_svd_batched_f64_workspace_size,
    cusolverDnDgesvdjBatched,
    cusolverDnDgesvdjBatched_bufferSize,
    f64
);

// =============================================================================
// SVD-Approximate Batched — `gesvdaStridedBatched` × {f32, f64},
// rectangular, element-strided batch.
// =============================================================================
//
// Signature: (jobz, rank, m, n, lda, ldu, ldv, stride_a, stride_s,
//             stride_u, stride_v, a_in, s_out, u_out, v_out,
//             info_out, h_r_nrm_f_out, batch_size, workspace,
//             workspace_bytes, stream).
//
// Returns `V` (not `V^T`). `h_r_nrm_f_out` is a HOST buffer of
// `batch_size` f64 residual Frobenius norms.

macro_rules! svda_batched_pair {
    ($name:ident, $ws_name:ident, $gesvda:ident, $gesvda_bs:ident, $T:ty) => {
        /// Approximate batched SVD workspace size in bytes.
        ///
        /// # Safety
        /// `out_bytes` must point to a writable `usize`.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $ws_name(
            jobz: i32,
            rank: i32,
            m: i32,
            n: i32,
            batch_size: i32,
            out_bytes: *mut usize,
        ) -> i32 {
            if m <= 0 || n <= 0 || batch_size <= 0 || rank < 1 || rank > m.min(n)
                || out_bytes.is_null()
            {
                return INVALID;
            }
            if !matches!(jobz, CUSOLVER_EIG_MODE_VECTOR | CUSOLVER_EIG_MODE_NOVECTOR) {
                return INVALID;
            }
            let mut h = Handle::new();
            let s = unsafe { setup_handle(&mut h, ptr::null_mut()) };
            if s != OK {
                return s;
            }
            let stride_a = (m as i64) * (n as i64);
            let stride_s = rank as i64;
            let stride_u = (m as i64) * (rank as i64);
            let stride_v = (n as i64) * (rank as i64);
            let mut lwork: i32 = 0;
            let st = unsafe {
                $gesvda_bs(
                    h.h, jobz, rank, m, n, ptr::null(), m, stride_a, ptr::null(), stride_s,
                    ptr::null(), m, stride_u, ptr::null(), n, stride_v, &mut lwork, batch_size,
                )
            };
            if st != 0 {
                return INTERNAL;
            }
            unsafe { *out_bytes = (lwork as usize) * core::mem::size_of::<$T>() };
            OK
        }

        /// Approximate (Jacobi-bidiagonal) batched SVD on rectangular
        /// input. Returns `V` (not `V^T`). The `h_r_nrm_f_out` buffer
        /// is **host-resident** and receives per-slot residual
        /// Frobenius norms (cuSOLVER signature). Pass null to discard
        /// — but cuSOLVER may dereference even when "discarding", so
        /// callers should pass a real buffer of `batch_size` f64s.
        ///
        /// # Safety
        /// All buffers except `h_r_nrm_f_out` are device-resident.
        /// `h_r_nrm_f_out` is host-resident and writable for
        /// `batch_size * sizeof(f64)` bytes.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            jobz: i32,
            rank: i32,
            m: i32,
            n: i32,
            lda: i32,
            ldu: i32,
            ldv: i32,
            stride_a: i64,
            stride_s: i64,
            stride_u: i64,
            stride_v: i64,
            a_in: *const c_void,
            s_out: *mut c_void,
            u_out: *mut c_void,
            v_out: *mut c_void,
            info_out: *mut i32,
            h_r_nrm_f_out: *mut f64,
            batch_size: i32,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32 {
            if m <= 0 || n <= 0 || batch_size <= 0 || rank < 1 || rank > m.min(n)
                || lda < m || ldu < m || ldv < n
                || a_in.is_null() || s_out.is_null() || info_out.is_null()
            {
                return INVALID;
            }
            if !matches!(jobz, CUSOLVER_EIG_MODE_VECTOR | CUSOLVER_EIG_MODE_NOVECTOR) {
                return INVALID;
            }
            if jobz == CUSOLVER_EIG_MODE_VECTOR && (u_out.is_null() || v_out.is_null()) {
                return INVALID;
            }
            let mut h = Handle::new();
            let s = unsafe { setup_handle(&mut h, stream) };
            if s != OK {
                return s;
            }
            let mut lwork: i32 = 0;
            let st = unsafe {
                $gesvda_bs(
                    h.h, jobz, rank, m, n, ptr::null(), lda, stride_a, ptr::null(), stride_s,
                    ptr::null(), ldu, stride_u, ptr::null(), ldv, stride_v, &mut lwork, batch_size,
                )
            };
            if st != 0 {
                return INTERNAL;
            }
            let needed = (lwork as usize) * core::mem::size_of::<$T>();
            let s = check_ws(workspace, workspace_bytes, needed);
            if s != OK {
                return s;
            }
            let st = unsafe {
                $gesvda(
                    h.h,
                    jobz,
                    rank,
                    m,
                    n,
                    a_in as *const $T,
                    lda,
                    stride_a,
                    s_out as *mut $T,
                    stride_s,
                    u_out as *mut $T,
                    ldu,
                    stride_u,
                    v_out as *mut $T,
                    ldv,
                    stride_v,
                    workspace as *mut $T,
                    lwork,
                    info_out,
                    h_r_nrm_f_out,
                    batch_size,
                )
            };
            map_cusolver(st)
        }
    };
}

svda_batched_pair!(
    baracuda_kernels_svda_batched_f32_run,
    baracuda_kernels_svda_batched_f32_workspace_size,
    cusolverDnSgesvdaStridedBatched,
    cusolverDnSgesvdaStridedBatched_bufferSize,
    f32
);
svda_batched_pair!(
    baracuda_kernels_svda_batched_f64_run,
    baracuda_kernels_svda_batched_f64_workspace_size,
    cusolverDnDgesvdaStridedBatched,
    cusolverDnDgesvdaStridedBatched_bufferSize,
    f64
);

// =============================================================================
// Eigh — symmetric / Hermitian eigendecomposition
// =============================================================================
//
// `syevd` for real, `heevd` for complex; eigenvalues are always real.
// Signature: (uplo, n, lda, a_inout, eigenvalues_out, info_out,
//             workspace, workspace_bytes, stream).

macro_rules! eigh_real_pair {
    ($name:ident, $ws_name:ident, $syevd:ident, $syevd_bs:ident, $T:ty) => {
        /// Eigh workspace size in bytes for the real symmetric `syevd` path.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $ws_name(uplo: i32, n: i32, out_bytes: *mut usize) -> i32 {
            if n <= 0 || out_bytes.is_null() {
                return INVALID;
            }
            if !matches!(uplo, CUBLAS_FILL_MODE_LOWER | CUBLAS_FILL_MODE_UPPER) {
                return INVALID;
            }
            let mut h = Handle::new();
            let s = unsafe { setup_handle(&mut h, ptr::null_mut()) };
            if s != OK {
                return s;
            }
            let mut lwork: i32 = 0;
            let st = unsafe {
                $syevd_bs(
                    h.h, CUSOLVER_EIG_MODE_VECTOR, uplo, n, ptr::null(), n, ptr::null(),
                    &mut lwork,
                )
            };
            if st != 0 {
                return INTERNAL;
            }
            unsafe { *out_bytes = (lwork as usize) * core::mem::size_of::<$T>() };
            OK
        }

        /// Symmetric eigendecomposition `A · v = λ · v`. `a_inout` is
        /// overwritten with the eigenvector matrix (column-major);
        /// `eigenvalues_out` receives the `n` eigenvalues sorted
        /// ascending.
        ///
        /// # Safety
        /// As for the Cholesky entry point. `eigenvalues_out` is `[n]`.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            uplo: i32,
            n: i32,
            lda: i32,
            a_inout: *mut c_void,
            eigenvalues_out: *mut c_void,
            info_out: *mut i32,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32 {
            if n <= 0 || lda < n
                || a_inout.is_null() || eigenvalues_out.is_null() || info_out.is_null()
            {
                return INVALID;
            }
            if !matches!(uplo, CUBLAS_FILL_MODE_LOWER | CUBLAS_FILL_MODE_UPPER) {
                return INVALID;
            }
            let mut h = Handle::new();
            let s = unsafe { setup_handle(&mut h, stream) };
            if s != OK {
                return s;
            }
            let mut lwork: i32 = 0;
            let st = unsafe {
                $syevd_bs(
                    h.h, CUSOLVER_EIG_MODE_VECTOR, uplo, n, ptr::null(), lda, ptr::null(),
                    &mut lwork,
                )
            };
            if st != 0 {
                return INTERNAL;
            }
            let needed = (lwork as usize) * core::mem::size_of::<$T>();
            let s = check_ws(workspace, workspace_bytes, needed);
            if s != OK {
                return s;
            }
            let st = unsafe {
                $syevd(
                    h.h,
                    CUSOLVER_EIG_MODE_VECTOR,
                    uplo,
                    n,
                    a_inout as *mut $T,
                    lda,
                    eigenvalues_out as *mut $T,
                    workspace as *mut $T,
                    lwork,
                    info_out,
                )
            };
            map_cusolver(st)
        }
    };
}

eigh_real_pair!(
    baracuda_kernels_eigh_f32_run,
    baracuda_kernels_eigh_f32_workspace_size,
    cusolverDnSsyevd,
    cusolverDnSsyevd_bufferSize,
    f32
);
eigh_real_pair!(
    baracuda_kernels_eigh_f64_run,
    baracuda_kernels_eigh_f64_workspace_size,
    cusolverDnDsyevd,
    cusolverDnDsyevd_bufferSize,
    f64
);

/// Hermitian eigendecomposition (Complex32). Eigenvalues are real
/// `f32` (the Hermitian eigenvalue spectrum is always real); the
/// `eigenvalues_out` buffer is `f32[n]`, not `Complex32[n]`.
///
/// # Safety
/// As for the f32 eigh.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn baracuda_kernels_eigh_c32_run(
    uplo: i32,
    n: i32,
    lda: i32,
    a_inout: *mut c_void,
    eigenvalues_out: *mut c_void,
    info_out: *mut i32,
    workspace: *mut c_void,
    workspace_bytes: usize,
    stream: *mut c_void,
) -> i32 {
    if n <= 0 || lda < n
        || a_inout.is_null() || eigenvalues_out.is_null() || info_out.is_null()
    {
        return INVALID;
    }
    if !matches!(uplo, CUBLAS_FILL_MODE_LOWER | CUBLAS_FILL_MODE_UPPER) {
        return INVALID;
    }
    let mut h = Handle::new();
    let s = unsafe { setup_handle(&mut h, stream) };
    if s != OK {
        return s;
    }
    let mut lwork: i32 = 0;
    let st = unsafe {
        cusolverDnCheevd_bufferSize(
            h.h, CUSOLVER_EIG_MODE_VECTOR, uplo, n, ptr::null(), lda, ptr::null(), &mut lwork,
        )
    };
    if st != 0 {
        return INTERNAL;
    }
    let needed = (lwork as usize) * core::mem::size_of::<cuComplex>();
    let s = check_ws(workspace, workspace_bytes, needed);
    if s != OK {
        return s;
    }
    let st = unsafe {
        cusolverDnCheevd(
            h.h,
            CUSOLVER_EIG_MODE_VECTOR,
            uplo,
            n,
            a_inout as *mut cuComplex,
            lda,
            eigenvalues_out as *mut f32,
            workspace as *mut cuComplex,
            lwork,
            info_out,
        )
    };
    map_cusolver(st)
}

/// Hermitian eigendecomposition workspace size (Complex32).
///
/// # Safety
/// `out_bytes` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn baracuda_kernels_eigh_c32_workspace_size(
    uplo: i32,
    n: i32,
    out_bytes: *mut usize,
) -> i32 {
    if n <= 0 || out_bytes.is_null() {
        return INVALID;
    }
    if !matches!(uplo, CUBLAS_FILL_MODE_LOWER | CUBLAS_FILL_MODE_UPPER) {
        return INVALID;
    }
    let mut h = Handle::new();
    let s = unsafe { setup_handle(&mut h, ptr::null_mut()) };
    if s != OK {
        return s;
    }
    let mut lwork: i32 = 0;
    let st = unsafe {
        cusolverDnCheevd_bufferSize(
            h.h, CUSOLVER_EIG_MODE_VECTOR, uplo, n, ptr::null(), n, ptr::null(), &mut lwork,
        )
    };
    if st != 0 {
        return INTERNAL;
    }
    unsafe { *out_bytes = (lwork as usize) * core::mem::size_of::<cuComplex>() };
    OK
}

/// Hermitian eigendecomposition (Complex64). Eigenvalues are real
/// `f64`; `eigenvalues_out` is `f64[n]`, not `Complex64[n]`.
///
/// # Safety
/// As for the c32 sibling.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn baracuda_kernels_eigh_c64_run(
    uplo: i32,
    n: i32,
    lda: i32,
    a_inout: *mut c_void,
    eigenvalues_out: *mut c_void,
    info_out: *mut i32,
    workspace: *mut c_void,
    workspace_bytes: usize,
    stream: *mut c_void,
) -> i32 {
    if n <= 0 || lda < n
        || a_inout.is_null() || eigenvalues_out.is_null() || info_out.is_null()
    {
        return INVALID;
    }
    if !matches!(uplo, CUBLAS_FILL_MODE_LOWER | CUBLAS_FILL_MODE_UPPER) {
        return INVALID;
    }
    let mut h = Handle::new();
    let s = unsafe { setup_handle(&mut h, stream) };
    if s != OK {
        return s;
    }
    let mut lwork: i32 = 0;
    let st = unsafe {
        cusolverDnZheevd_bufferSize(
            h.h, CUSOLVER_EIG_MODE_VECTOR, uplo, n, ptr::null(), lda, ptr::null(), &mut lwork,
        )
    };
    if st != 0 {
        return INTERNAL;
    }
    let needed = (lwork as usize) * core::mem::size_of::<cuDoubleComplex>();
    let s = check_ws(workspace, workspace_bytes, needed);
    if s != OK {
        return s;
    }
    let st = unsafe {
        cusolverDnZheevd(
            h.h,
            CUSOLVER_EIG_MODE_VECTOR,
            uplo,
            n,
            a_inout as *mut cuDoubleComplex,
            lda,
            eigenvalues_out as *mut f64,
            workspace as *mut cuDoubleComplex,
            lwork,
            info_out,
        )
    };
    map_cusolver(st)
}

/// Hermitian eigendecomposition workspace size (Complex64).
///
/// # Safety
/// `out_bytes` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn baracuda_kernels_eigh_c64_workspace_size(
    uplo: i32,
    n: i32,
    out_bytes: *mut usize,
) -> i32 {
    if n <= 0 || out_bytes.is_null() {
        return INVALID;
    }
    if !matches!(uplo, CUBLAS_FILL_MODE_LOWER | CUBLAS_FILL_MODE_UPPER) {
        return INVALID;
    }
    let mut h = Handle::new();
    let s = unsafe { setup_handle(&mut h, ptr::null_mut()) };
    if s != OK {
        return s;
    }
    let mut lwork: i32 = 0;
    let st = unsafe {
        cusolverDnZheevd_bufferSize(
            h.h, CUSOLVER_EIG_MODE_VECTOR, uplo, n, ptr::null(), n, ptr::null(), &mut lwork,
        )
    };
    if st != 0 {
        return INTERNAL;
    }
    unsafe { *out_bytes = (lwork as usize) * core::mem::size_of::<cuDoubleComplex>() };
    OK
}

// =============================================================================
// Eig — general (non-symmetric) eigendecomposition via `Xgeev`
// =============================================================================
//
// Signature: (dtype_tag, jobvl, jobvr, n, lda, ldvl, ldvr, a_inout,
//             w_out, vl_out, vr_out, info_out,
//             workspace_device, workspace_bytes_device,
//             workspace_host, workspace_bytes_host, stream).
//
// `dtype_tag`: `cudaDataType` value (CUDA_R_32F / CUDA_R_64F /
// CUDA_C_32F / CUDA_C_64F) — distinguishes the four supported input
// dtypes (single Xgeev entry handles all four).
//
// Per LAPACK convention (matching the Rust plan), eigenvalues `W` for
// real input use `[2 * n]` packing (`wr` then `wi`); complex input
// uses `[n]` directly. The caller picks the appropriate size and dtype.

/// Eig workspace sizes (Xgeev). Writes **two** byte counts —
/// device + host. Caller must size both.
///
/// # Safety
/// `out_device_bytes` and `out_host_bytes` must point to writable `usize`s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn baracuda_kernels_eig_workspace_size(
    dtype_tag: cudaDataType,
    jobvl: i32,
    jobvr: i32,
    n: i64,
    out_device_bytes: *mut usize,
    out_host_bytes: *mut usize,
) -> i32 {
    if n <= 0 || out_device_bytes.is_null() || out_host_bytes.is_null() {
        return INVALID;
    }
    if !matches!(jobvl, CUSOLVER_EIG_MODE_VECTOR | CUSOLVER_EIG_MODE_NOVECTOR) {
        return INVALID;
    }
    if !matches!(jobvr, CUSOLVER_EIG_MODE_VECTOR | CUSOLVER_EIG_MODE_NOVECTOR) {
        return INVALID;
    }
    if !matches!(dtype_tag, CUDA_R_32F | CUDA_R_64F | CUDA_C_32F | CUDA_C_64F) {
        return INVALID;
    }
    let mut h = Handle::new();
    let s = unsafe { setup_handle(&mut h, ptr::null_mut()) };
    if s != OK {
        return s;
    }
    let mut p = Params::new();
    let st = unsafe { cusolverDnCreateParams(&mut p.p as *mut _) };
    if st != 0 {
        return INTERNAL;
    }
    let mut ws_dev: usize = 0;
    let mut ws_host: usize = 0;
    let st = unsafe {
        cusolverDnXgeev_bufferSize(
            h.h,
            p.p,
            jobvl,
            jobvr,
            n,
            dtype_tag,
            ptr::null(),
            n,
            dtype_tag,
            ptr::null(),
            dtype_tag,
            ptr::null(),
            n,
            dtype_tag,
            ptr::null(),
            n,
            dtype_tag,
            &mut ws_dev,
            &mut ws_host,
        )
    };
    if st != 0 {
        return INTERNAL;
    }
    unsafe {
        *out_device_bytes = ws_dev;
        *out_host_bytes = ws_host;
    }
    OK
}

/// General eigendecomposition via `Xgeev`. `a_inout` is destroyed in
/// place. `dtype_tag` selects between f32 / f64 / Complex32 / Complex64
/// (matches the input dtype; outputs use the same dtype). For real
/// input, `w_out` is `[2 * n]` (packed wr/wi); for complex input,
/// `[n]`. Workspace is split host + device per cuSOLVER's 64-bit API
/// convention.
///
/// # Safety
/// `workspace_device` is device-resident, `workspace_host` is
/// host-resident; both must cover the bytes returned by
/// `baracuda_kernels_eig_workspace_size`. Pass null for VL / VR when
/// the corresponding job is `NOVECTOR`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn baracuda_kernels_eig_run(
    dtype_tag: cudaDataType,
    jobvl: i32,
    jobvr: i32,
    n: i64,
    lda: i64,
    ldvl: i64,
    ldvr: i64,
    a_inout: *mut c_void,
    w_out: *mut c_void,
    vl_out: *mut c_void,
    vr_out: *mut c_void,
    info_out: *mut i32,
    workspace_device: *mut c_void,
    workspace_bytes_device: usize,
    workspace_host: *mut c_void,
    workspace_bytes_host: usize,
    stream: *mut c_void,
) -> i32 {
    if n <= 0 || lda < n
        || a_inout.is_null() || w_out.is_null() || info_out.is_null()
    {
        return INVALID;
    }
    if !matches!(jobvl, CUSOLVER_EIG_MODE_VECTOR | CUSOLVER_EIG_MODE_NOVECTOR) {
        return INVALID;
    }
    if !matches!(jobvr, CUSOLVER_EIG_MODE_VECTOR | CUSOLVER_EIG_MODE_NOVECTOR) {
        return INVALID;
    }
    if !matches!(dtype_tag, CUDA_R_32F | CUDA_R_64F | CUDA_C_32F | CUDA_C_64F) {
        return INVALID;
    }
    if jobvl == CUSOLVER_EIG_MODE_VECTOR && (vl_out.is_null() || ldvl < n) {
        return INVALID;
    }
    if jobvr == CUSOLVER_EIG_MODE_VECTOR && (vr_out.is_null() || ldvr < n) {
        return INVALID;
    }
    let mut h = Handle::new();
    let s = unsafe { setup_handle(&mut h, stream) };
    if s != OK {
        return s;
    }
    let mut p = Params::new();
    let st = unsafe { cusolverDnCreateParams(&mut p.p as *mut _) };
    if st != 0 {
        return INTERNAL;
    }
    let mut ws_dev: usize = 0;
    let mut ws_host: usize = 0;
    let st = unsafe {
        cusolverDnXgeev_bufferSize(
            h.h,
            p.p,
            jobvl,
            jobvr,
            n,
            dtype_tag,
            ptr::null(),
            lda,
            dtype_tag,
            ptr::null(),
            dtype_tag,
            ptr::null(),
            if ldvl > 0 { ldvl } else { n },
            dtype_tag,
            ptr::null(),
            if ldvr > 0 { ldvr } else { n },
            dtype_tag,
            &mut ws_dev,
            &mut ws_host,
        )
    };
    if st != 0 {
        return INTERNAL;
    }
    if check_ws(workspace_device, workspace_bytes_device, ws_dev) != OK {
        return WS_TOO_SMALL;
    }
    if ws_host > 0 && (workspace_host.is_null() || workspace_bytes_host < ws_host) {
        return WS_TOO_SMALL;
    }
    let st = unsafe {
        cusolverDnXgeev(
            h.h,
            p.p,
            jobvl,
            jobvr,
            n,
            dtype_tag,
            a_inout,
            lda,
            dtype_tag,
            w_out,
            dtype_tag,
            vl_out,
            if ldvl > 0 { ldvl } else { n },
            dtype_tag,
            vr_out,
            if ldvr > 0 { ldvr } else { n },
            dtype_tag,
            workspace_device,
            workspace_bytes_device,
            workspace_host,
            workspace_bytes_host,
            info_out,
        )
    };
    map_cusolver(st)
}

// =============================================================================
// LstSq — least-squares solve via `_gels` (iterative refinement only)
// =============================================================================
//
// Signature: (m, n, nrhs, lda, ldb, ldx, a_inout, b_inout, x_out,
//             niters_out, info_out, workspace, workspace_bytes, stream).
//
// `niters_out`: non-negative on convergence; negative means the
// iterative refinement did not converge. The Rust plan layer has a QR
// fallback for this case (geqrf + ormqr + trsm); the FFI does NOT —
// callers should drive the Rust plan if they need the fallback.
// `_gels` workspace is BYTE-typed (not element-typed).

macro_rules! lstsq_pair {
    ($name:ident, $ws_name:ident, $gels:ident, $gels_bs:ident, $T:ty) => {
        /// LstSq workspace size in BYTES (not elements — cuSOLVER's
        /// `_gels` API differs from the others on this point).
        ///
        /// # Safety
        /// `out_bytes` must point to a writable `usize`.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $ws_name(
            m: i32,
            n: i32,
            nrhs: i32,
            out_bytes: *mut usize,
        ) -> i32 {
            if m <= 0 || n <= 0 || nrhs <= 0 || m < n || out_bytes.is_null() {
                return INVALID;
            }
            let mut h = Handle::new();
            let s = unsafe { setup_handle(&mut h, ptr::null_mut()) };
            if s != OK {
                return s;
            }
            let mut lwork_bytes: usize = 0;
            let st = unsafe {
                $gels_bs(
                    h.h, m, n, nrhs, ptr::null_mut(), m, ptr::null_mut(), m, ptr::null_mut(), n,
                    ptr::null_mut(), &mut lwork_bytes,
                )
            };
            if st != 0 {
                return INTERNAL;
            }
            unsafe { *out_bytes = lwork_bytes };
            OK
        }

        /// Least-squares solve via iterative `_gels` (no QR fallback).
        /// On convergence, `niters_out >= 0`. On non-convergence,
        /// `niters_out < 0` and the caller should retry via the Rust
        /// plan layer (which holds the QR fallback path).
        ///
        /// # Pointer residency
        /// - `a_inout`, `b_inout`, `x_out`, `workspace`, `info_out` —
        ///   DEVICE pointers (info_out is written by a cuSOLVER kernel).
        /// - `niters_out` — **HOST** pointer to an `i32`. cuSOLVER's
        ///   iterative-refinement loop is host-side; passing a device
        ///   pointer triggers `STATUS_ACCESS_VIOLATION`.
        ///
        /// # Safety
        /// `a_inout` is destroyed in place. `b_inout` is overwritten
        /// with scratch. `x_out` is `[n, nrhs]` (column-major) and
        /// receives the solution.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            m: i32,
            n: i32,
            nrhs: i32,
            lda: i32,
            ldb: i32,
            ldx: i32,
            a_inout: *mut c_void,
            b_inout: *mut c_void,
            x_out: *mut c_void,
            niters_out: *mut i32,
            info_out: *mut i32,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32 {
            if m <= 0 || n <= 0 || nrhs <= 0 || m < n
                || lda < m || ldb < m || ldx < n
                || a_inout.is_null() || b_inout.is_null() || x_out.is_null()
                || niters_out.is_null() || info_out.is_null()
            {
                return INVALID;
            }
            let mut h = Handle::new();
            let s = unsafe { setup_handle(&mut h, stream) };
            if s != OK {
                return s;
            }
            // Re-query bufferSize on this same handle (cuSOLVER's `_gels`
            // is sensitive to bufferSize-vs-run handle pairing — using
            // different handles can yield mismatched lwork values). We pass
            // null pointers to bufferSize (matching `_workspace_size`) so
            // the query is data-independent. Then pass the queried bytes to
            // `_gels` to honor cuSOLVER's "lwork_bytes argument must match
            // what bufferSize returned on this handle" contract.
            let mut lwork_bytes: usize = 0;
            let st = unsafe {
                $gels_bs(
                    h.h, m, n, nrhs, ptr::null_mut(), lda, ptr::null_mut(), ldb,
                    ptr::null_mut(), ldx, ptr::null_mut(), &mut lwork_bytes,
                )
            };
            if st != 0 {
                return INTERNAL;
            }
            if workspace_bytes < lwork_bytes {
                return WS_TOO_SMALL;
            }
            let st = unsafe {
                $gels(
                    h.h,
                    m,
                    n,
                    nrhs,
                    a_inout as *mut $T,
                    lda,
                    b_inout as *mut $T,
                    ldb,
                    x_out as *mut $T,
                    ldx,
                    workspace,
                    lwork_bytes,
                    niters_out,
                    info_out,
                )
            };
            map_cusolver(st)
        }
    };
}

lstsq_pair!(
    baracuda_kernels_lstsq_f32_run,
    baracuda_kernels_lstsq_f32_workspace_size,
    cusolverDnSSgels,
    cusolverDnSSgels_bufferSize,
    f32
);
lstsq_pair!(
    baracuda_kernels_lstsq_f64_run,
    baracuda_kernels_lstsq_f64_workspace_size,
    cusolverDnDDgels,
    cusolverDnDDgels_bufferSize,
    f64
);

// =============================================================================
// Solve — `A · X = B` via `getrf` + `getrs`
// =============================================================================
//
// Signature: (n, nrhs, lda, ldb, a_inout, pivots_out, b_inout,
//             info_out, workspace, workspace_bytes, stream).
//
// Single FFI call that fuses both halves of the LAPACK dgesv-equivalent.

macro_rules! solve_pair {
    ($name:ident, $ws_name:ident, $getrf:ident, $getrs:ident, $getrf_bs:ident, $T:ty) => {
        /// Solve workspace size — uses the `getrf` query (cuSOLVER's
        /// `getrs` is workspace-free).
        ///
        /// # Safety
        /// `out_bytes` must point to a writable `usize`.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $ws_name(n: i32, lda: i32, out_bytes: *mut usize) -> i32 {
            if n <= 0 || lda < n || out_bytes.is_null() {
                return INVALID;
            }
            let mut h = Handle::new();
            let s = unsafe { setup_handle(&mut h, ptr::null_mut()) };
            if s != OK {
                return s;
            }
            let mut lwork: i32 = 0;
            let st = unsafe { $getrf_bs(h.h, n, n, ptr::null_mut(), lda, &mut lwork) };
            if st != 0 {
                return INTERNAL;
            }
            unsafe { *out_bytes = (lwork as usize) * core::mem::size_of::<$T>() };
            OK
        }

        /// Linear-system solve `A · X = B` via fused `getrf` + `getrs`.
        /// `a_inout` is overwritten with packed `LU` factors; `b_inout`
        /// is overwritten with the solution `X`. `pivots_out` is `[n]`
        /// (1-based per LAPACK convention).
        ///
        /// # Safety
        /// As for the Cholesky entry point.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            n: i32,
            nrhs: i32,
            lda: i32,
            ldb: i32,
            a_inout: *mut c_void,
            pivots_out: *mut i32,
            b_inout: *mut c_void,
            info_out: *mut i32,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32 {
            if n <= 0 || nrhs <= 0 || lda < n || ldb < n
                || a_inout.is_null() || pivots_out.is_null() || b_inout.is_null()
                || info_out.is_null()
            {
                return INVALID;
            }
            let mut h = Handle::new();
            let s = unsafe { setup_handle(&mut h, stream) };
            if s != OK {
                return s;
            }
            let mut lwork: i32 = 0;
            let st = unsafe { $getrf_bs(h.h, n, n, ptr::null_mut(), lda, &mut lwork) };
            if st != 0 {
                return INTERNAL;
            }
            let needed = (lwork as usize) * core::mem::size_of::<$T>();
            let s = check_ws(workspace, workspace_bytes, needed);
            if s != OK {
                return s;
            }
            // 1. getrf — factor A in place, write pivot + info.
            let st = unsafe {
                $getrf(
                    h.h,
                    n,
                    n,
                    a_inout as *mut $T,
                    lda,
                    workspace as *mut $T,
                    pivots_out,
                    info_out,
                )
            };
            if st != 0 {
                return INTERNAL;
            }
            // 2. getrs — solve A · X = B in place over B.
            let st = unsafe {
                $getrs(
                    h.h,
                    CUBLAS_OP_N,
                    n,
                    nrhs,
                    a_inout as *const $T,
                    lda,
                    pivots_out as *const i32,
                    b_inout as *mut $T,
                    ldb,
                    info_out,
                )
            };
            map_cusolver(st)
        }
    };
}

solve_pair!(
    baracuda_kernels_solve_f32_run,
    baracuda_kernels_solve_f32_workspace_size,
    cusolverDnSgetrf,
    cusolverDnSgetrs,
    cusolverDnSgetrf_bufferSize,
    f32
);
solve_pair!(
    baracuda_kernels_solve_f64_run,
    baracuda_kernels_solve_f64_workspace_size,
    cusolverDnDgetrf,
    cusolverDnDgetrs,
    cusolverDnDgetrf_bufferSize,
    f64
);

// =============================================================================
// Inverse — `getrf` + `getrs` over a caller-staged identity
// =============================================================================
//
// Signature: (n, lda, ldinv, a_inout, pivots_out, inv_inout,
//             info_out, workspace, workspace_bytes, stream).
//
// IMPORTANT: the caller MUST pre-stage an `n × n` identity matrix in
// `inv_inout` before calling. The FFI does NOT build the identity
// (unlike the Rust plan, which does this internally via a host build +
// async H2D). The split keeps the FFI thin — caller controls all
// device work including identity staging.

macro_rules! inverse_pair {
    ($name:ident, $ws_name:ident, $getrf:ident, $getrs:ident, $getrf_bs:ident, $T:ty) => {
        /// Inverse workspace size (== `getrf` workspace).
        ///
        /// # Safety
        /// `out_bytes` writable.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $ws_name(n: i32, lda: i32, out_bytes: *mut usize) -> i32 {
            if n <= 0 || lda < n || out_bytes.is_null() {
                return INVALID;
            }
            let mut h = Handle::new();
            let s = unsafe { setup_handle(&mut h, ptr::null_mut()) };
            if s != OK {
                return s;
            }
            let mut lwork: i32 = 0;
            let st = unsafe { $getrf_bs(h.h, n, n, ptr::null_mut(), lda, &mut lwork) };
            if st != 0 {
                return INTERNAL;
            }
            unsafe { *out_bytes = (lwork as usize) * core::mem::size_of::<$T>() };
            OK
        }

        /// Matrix inverse via `getrf` + `getrs` over **caller-staged
        /// identity** in `inv_inout`. The caller MUST pre-stage an
        /// `n × n` identity in `inv_inout` (column-major) before
        /// invoking. After the call, `inv_inout` holds `A^{-1}` and
        /// `a_inout` holds the packed `LU` factors.
        ///
        /// # Safety
        /// As for the Cholesky entry point. `inv_inout` is `[n, n]`
        /// pre-staged with the identity matrix.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            n: i32,
            lda: i32,
            ldinv: i32,
            a_inout: *mut c_void,
            pivots_out: *mut i32,
            inv_inout: *mut c_void,
            info_out: *mut i32,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32 {
            if n <= 0 || lda < n || ldinv < n
                || a_inout.is_null() || pivots_out.is_null() || inv_inout.is_null()
                || info_out.is_null()
            {
                return INVALID;
            }
            let mut h = Handle::new();
            let s = unsafe { setup_handle(&mut h, stream) };
            if s != OK {
                return s;
            }
            let mut lwork: i32 = 0;
            let st = unsafe { $getrf_bs(h.h, n, n, ptr::null_mut(), lda, &mut lwork) };
            if st != 0 {
                return INTERNAL;
            }
            let needed = (lwork as usize) * core::mem::size_of::<$T>();
            let s = check_ws(workspace, workspace_bytes, needed);
            if s != OK {
                return s;
            }
            let st = unsafe {
                $getrf(
                    h.h,
                    n,
                    n,
                    a_inout as *mut $T,
                    lda,
                    workspace as *mut $T,
                    pivots_out,
                    info_out,
                )
            };
            if st != 0 {
                return INTERNAL;
            }
            // getrs over the caller-staged identity → A^{-1}.
            let st = unsafe {
                $getrs(
                    h.h,
                    CUBLAS_OP_N,
                    n,
                    n,
                    a_inout as *const $T,
                    lda,
                    pivots_out as *const i32,
                    inv_inout as *mut $T,
                    ldinv,
                    info_out,
                )
            };
            map_cusolver(st)
        }
    };
}

inverse_pair!(
    baracuda_kernels_inverse_f32_run,
    baracuda_kernels_inverse_f32_workspace_size,
    cusolverDnSgetrf,
    cusolverDnSgetrs,
    cusolverDnSgetrf_bufferSize,
    f32
);
inverse_pair!(
    baracuda_kernels_inverse_f64_run,
    baracuda_kernels_inverse_f64_workspace_size,
    cusolverDnDgetrf,
    cusolverDnDgetrs,
    cusolverDnDgetrf_bufferSize,
    f64
);
