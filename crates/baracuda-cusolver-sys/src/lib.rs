//! Raw FFI + dynamic loader for NVIDIA cuSOLVER (Dense + Sparse + Refactor).
//!
//! `baracuda-cusolver` wraps this with a safe, typed API. Use this
//! crate directly only if you need a function that the safe layer
//! hasn't wrapped yet (in which case please file a bug).

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
#![warn(missing_debug_implementations)]

use core::ffi::{c_int, c_void};
use std::sync::OnceLock;

use baracuda_core::{platform, Library, LoaderError};
use baracuda_cuda_sys::runtime::cudaStream_t;
use baracuda_types::CudaStatus;

// ---- handles --------------------------------------------------------------

/// Opaque dense cuSOLVER handle.
pub type cusolverDnHandle_t = *mut c_void;
/// Opaque sparse cuSOLVER handle.
pub type cusolverSpHandle_t = *mut c_void;
/// Opaque sparse-refactor cuSOLVER handle.
pub type cusolverRfHandle_t = *mut c_void;
/// Opaque generic-API parameter object for `cusolverDnX*` routines.
pub type cusolverDnParams_t = *mut c_void;
/// Opaque parameter object for iterative-refinement solvers.
pub type cusolverDnIRSParams_t = *mut c_void;
/// Opaque info object for iterative-refinement solvers.
pub type cusolverDnIRSInfos_t = *mut c_void;
/// Opaque control object for Jacobi-based symmetric eigendecomposition.
pub type syevjInfo_t = *mut c_void;
/// Opaque control object for Jacobi-based SVD.
pub type gesvdjInfo_t = *mut c_void;

// ---- enums ----------------------------------------------------------------

/// Transpose selector — same values as `cublasOperation_t` (N=0, T=1, C=2).
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cublasOperation_t {
    /// No transpose.
    N = 0,
    /// Transpose.
    T = 1,
    /// Conjugate transpose.
    C = 2,
}

/// Triangular fill mode (matches the cuBLAS enum).
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cublasFillMode_t {
    /// Lower-triangular.
    Lower = 0,
    /// Upper-triangular.
    Upper = 1,
    /// Full / dense matrix.
    Full = 2,
}

/// Side selector for one-sided operators (apply from left or right).
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cublasSideMode_t {
    /// Apply from the left.
    Left = 0,
    /// Apply from the right.
    Right = 1,
}

/// Diagonal-unit selector (matches the cuBLAS enum).
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cublasDiagType_t {
    /// Non-unit diagonal.
    NonUnit = 0,
    /// Unit diagonal.
    Unit = 1,
}

/// Generalized eigenproblem variant for `sygv*` / `hegv*` routines.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusolverEigType_t {
    /// Type-1 generalized eigenproblem: `A*x = lambda*B*x`.
    Type1 = 1,
    /// Type-2 generalized eigenproblem: `A*B*x = lambda*x`.
    Type2 = 2,
    /// Type-3 generalized eigenproblem: `B*A*x = lambda*x`.
    Type3 = 3,
}

/// Eigenvalue-only vs eigenvalue-and-vector selector for eigensolvers.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusolverEigMode_t {
    /// Compute eigenvalues only.
    NoVector = 0,
    /// Compute eigenvalues and eigenvectors.
    Vector = 1,
}

/// Subset-of-spectrum selector for partial eigensolvers.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusolverEigRange_t {
    /// Compute all eigenvalues.
    All = 1001,
    /// Compute eigenvalues with indices in `[il, iu]`.
    I = 1002,
    /// Compute eigenvalues in the half-open interval `(vl, vu]`.
    V = 1003,
}

/// Element-dtype selector used by the generic-API (`cusolverDnX*`) routines.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudaDataType {
    /// 32-bit real (`f32`).
    R_32F = 0,
    /// 64-bit real (`f64`).
    R_64F = 1,
    /// 16-bit real (IEEE half / `f16`).
    R_16F = 2,
    /// 32-bit complex (`Complex<f32>`).
    C_32F = 4,
    /// 64-bit complex (`Complex<f64>`).
    C_64F = 5,
    /// 16-bit real bfloat16.
    R_16BF = 14,
}

// ---- status ---------------------------------------------------------------

/// Status / error code returned by cuSOLVER FFI calls.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct cusolverStatus_t(pub i32);

impl cusolverStatus_t {
    /// Status: success.
    pub const SUCCESS: Self = Self(0);
    /// Status: not initialized.
    pub const NOT_INITIALIZED: Self = Self(1);
    /// Status: alloc failed.
    pub const ALLOC_FAILED: Self = Self(2);
    /// Status: invalid value.
    pub const INVALID_VALUE: Self = Self(3);
    /// Status: arch mismatch.
    pub const ARCH_MISMATCH: Self = Self(4);
    /// Status: execution failed.
    pub const EXECUTION_FAILED: Self = Self(6);
    /// Status: internal error.
    pub const INTERNAL_ERROR: Self = Self(7);
    /// Status: not supported.
    pub const NOT_SUPPORTED: Self = Self(9);
    /// Status: zero pivot.
    pub const ZERO_PIVOT: Self = Self(10);

    /// Returns `true` when this is the success status code.
    pub const fn is_success(self) -> bool {
        self.0 == 0
    }
}

impl CudaStatus for cusolverStatus_t {
    fn code(self) -> i32 {
        self.0
    }
    fn name(self) -> &'static str {
        match self.0 {
            0 => "CUSOLVER_STATUS_SUCCESS",
            1 => "CUSOLVER_STATUS_NOT_INITIALIZED",
            2 => "CUSOLVER_STATUS_ALLOC_FAILED",
            3 => "CUSOLVER_STATUS_INVALID_VALUE",
            6 => "CUSOLVER_STATUS_EXECUTION_FAILED",
            7 => "CUSOLVER_STATUS_INTERNAL_ERROR",
            9 => "CUSOLVER_STATUS_NOT_SUPPORTED",
            10 => "CUSOLVER_STATUS_ZERO_PIVOT",
            _ => "CUSOLVER_STATUS_UNRECOGNIZED",
        }
    }
    fn description(self) -> &'static str {
        match self.0 {
            0 => "success",
            1 => "cuSOLVER not initialized",
            6 => "execution failed on device",
            10 => "factorization produced a zero pivot",
            _ => "unrecognized cuSOLVER status code",
        }
    }
    fn is_success(self) -> bool {
        cusolverStatus_t::is_success(self)
    }
    fn library(self) -> &'static str {
        "cusolver"
    }
}

// ---- complex types (alias to plain 2-element arrays) ---------------------

/// Single-precision complex number, ABI-compatible with cuBLAS / cuSOLVER `cuComplex`.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct cuComplex {
    /// Real component (`f32`).
    pub x: f32,
    /// Imaginary component (`f32`).
    pub y: f32,
}

/// Double-precision complex number, ABI-compatible with cuBLAS / cuSOLVER `cuDoubleComplex`.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct cuDoubleComplex {
    /// Real component (`f64`).
    pub x: f64,
    /// Imaginary component (`f64`).
    pub y: f64,
}

// ---- PFN type declaration macros -----------------------------------------

/// `getrf_bufferSize(handle, m, n, a, lda, lwork) -> status`
macro_rules! dn_getrf_bufsize {
    ($(#[$attr:meta])* $name:ident, $t:ty) => {
        $(#[$attr])*
        pub type $name = unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: c_int,
            n: c_int,
            a: *mut $t,
            lda: c_int,
            lwork: *mut c_int,
        ) -> cusolverStatus_t;
    };
}

macro_rules! dn_getrf {
    ($(#[$attr:meta])* $name:ident, $t:ty) => {
        $(#[$attr])*
        pub type $name = unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: c_int,
            n: c_int,
            a: *mut $t,
            lda: c_int,
            workspace: *mut $t,
            ipiv: *mut c_int,
            info: *mut c_int,
        ) -> cusolverStatus_t;
    };
}

macro_rules! dn_getrs {
    ($(#[$attr:meta])* $name:ident, $t:ty) => {
        $(#[$attr])*
        pub type $name = unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            trans: cublasOperation_t,
            n: c_int,
            nrhs: c_int,
            a: *const $t,
            lda: c_int,
            ipiv: *const c_int,
            b: *mut $t,
            ldb: c_int,
            info: *mut c_int,
        ) -> cusolverStatus_t;
    };
}

macro_rules! dn_geqrf_bufsize {
    ($(#[$attr:meta])* $name:ident, $t:ty) => {
        $(#[$attr])*
        pub type $name = unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: c_int,
            n: c_int,
            a: *mut $t,
            lda: c_int,
            lwork: *mut c_int,
        ) -> cusolverStatus_t;
    };
}

macro_rules! dn_geqrf {
    ($(#[$attr:meta])* $name:ident, $t:ty) => {
        $(#[$attr])*
        pub type $name = unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: c_int,
            n: c_int,
            a: *mut $t,
            lda: c_int,
            tau: *mut $t,
            workspace: *mut $t,
            lwork: c_int,
            info: *mut c_int,
        ) -> cusolverStatus_t;
    };
}

macro_rules! dn_potrf_bufsize {
    ($(#[$attr:meta])* $name:ident, $t:ty) => {
        $(#[$attr])*
        pub type $name = unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: c_int,
            a: *mut $t,
            lda: c_int,
            lwork: *mut c_int,
        ) -> cusolverStatus_t;
    };
}

macro_rules! dn_potrf {
    ($(#[$attr:meta])* $name:ident, $t:ty) => {
        $(#[$attr])*
        pub type $name = unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: c_int,
            a: *mut $t,
            lda: c_int,
            workspace: *mut $t,
            lwork: c_int,
            info: *mut c_int,
        ) -> cusolverStatus_t;
    };
}

macro_rules! dn_potrs {
    ($(#[$attr:meta])* $name:ident, $t:ty) => {
        $(#[$attr])*
        pub type $name = unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: c_int,
            nrhs: c_int,
            a: *const $t,
            lda: c_int,
            b: *mut $t,
            ldb: c_int,
            info: *mut c_int,
        ) -> cusolverStatus_t;
    };
}

macro_rules! dn_gesvd_bufsize {
    ($(#[$attr:meta])* $name:ident) => {
        $(#[$attr])*
        pub type $name = unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: c_int,
            n: c_int,
            lwork: *mut c_int,
        ) -> cusolverStatus_t;
    };
}

macro_rules! dn_gesvd_real {
    ($(#[$attr:meta])* $name:ident, $t:ty) => {
        $(#[$attr])*
        pub type $name = unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobu: u8,
            jobvt: u8,
            m: c_int,
            n: c_int,
            a: *mut $t,
            lda: c_int,
            s: *mut $t,
            u: *mut $t,
            ldu: c_int,
            vt: *mut $t,
            ldvt: c_int,
            work: *mut $t,
            lwork: c_int,
            rwork: *mut $t,
            info: *mut c_int,
        ) -> cusolverStatus_t;
    };
}

macro_rules! dn_gesvd_complex {
    ($(#[$attr:meta])* $name:ident, $t:ty, $real:ty) => {
        $(#[$attr])*
        pub type $name = unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobu: u8,
            jobvt: u8,
            m: c_int,
            n: c_int,
            a: *mut $t,
            lda: c_int,
            s: *mut $real,
            u: *mut $t,
            ldu: c_int,
            vt: *mut $t,
            ldvt: c_int,
            work: *mut $t,
            lwork: c_int,
            rwork: *mut $real,
            info: *mut c_int,
        ) -> cusolverStatus_t;
    };
}

macro_rules! dn_syevd_bufsize {
    ($(#[$attr:meta])* $name:ident, $t:ty, $real:ty) => {
        $(#[$attr])*
        pub type $name = unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: c_int,
            a: *const $t,
            lda: c_int,
            w: *const $real,
            lwork: *mut c_int,
        ) -> cusolverStatus_t;
    };
}

macro_rules! dn_syevd {
    ($(#[$attr:meta])* $name:ident, $t:ty, $real:ty) => {
        $(#[$attr])*
        pub type $name = unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: c_int,
            a: *mut $t,
            lda: c_int,
            w: *mut $real,
            work: *mut $t,
            lwork: c_int,
            info: *mut c_int,
        ) -> cusolverStatus_t;
    };
}

// ---- core Dn handle ------------------------------------------------------

/// cuSOLVER: create a dense cuSOLVER handle. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverDnCreate =
    unsafe extern "C" fn(handle: *mut cusolverDnHandle_t) -> cusolverStatus_t;
/// cuSOLVER: destroy a dense cuSOLVER handle. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverDnDestroy =
    unsafe extern "C" fn(handle: cusolverDnHandle_t) -> cusolverStatus_t;
/// cuSOLVER: bind a CUDA stream to a dense cuSOLVER handle. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverDnSetStream =
    unsafe extern "C" fn(handle: cusolverDnHandle_t, stream: cudaStream_t) -> cusolverStatus_t;
/// cuSOLVER: query the CUDA stream bound to a dense cuSOLVER handle. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverDnGetStream =
    unsafe extern "C" fn(handle: cusolverDnHandle_t, stream: *mut cudaStream_t) -> cusolverStatus_t;

/// cuSOLVER: return the cuSOLVER library version. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverGetVersion = unsafe extern "C" fn(version: *mut c_int) -> cusolverStatus_t;

// ---- LU factorization (getrf / getrs) — S/D/C/Z --------------------------

dn_getrf_bufsize!(#[doc = "cuSOLVER: single-precision workspace-size query for LU factorization with partial pivoting. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnSgetrf_bufferSize, f32);
dn_getrf_bufsize!(#[doc = "cuSOLVER: double-precision workspace-size query for LU factorization with partial pivoting. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnDgetrf_bufferSize, f64);
dn_getrf_bufsize!(#[doc = "cuSOLVER: single-precision complex workspace-size query for LU factorization with partial pivoting. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnCgetrf_bufferSize, cuComplex);
dn_getrf_bufsize!(#[doc = "cuSOLVER: double-precision complex workspace-size query for LU factorization with partial pivoting. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnZgetrf_bufferSize, cuDoubleComplex);

dn_getrf!(#[doc = "cuSOLVER: single-precision LU factorization with partial pivoting. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnSgetrf, f32);
dn_getrf!(#[doc = "cuSOLVER: double-precision LU factorization with partial pivoting. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnDgetrf, f64);
dn_getrf!(#[doc = "cuSOLVER: single-precision complex LU factorization with partial pivoting. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnCgetrf, cuComplex);
dn_getrf!(#[doc = "cuSOLVER: double-precision complex LU factorization with partial pivoting. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnZgetrf, cuDoubleComplex);

dn_getrs!(#[doc = "cuSOLVER: single-precision solve linear system using LU factors. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnSgetrs, f32);
dn_getrs!(#[doc = "cuSOLVER: double-precision solve linear system using LU factors. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnDgetrs, f64);
dn_getrs!(#[doc = "cuSOLVER: single-precision complex solve linear system using LU factors. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnCgetrs, cuComplex);
dn_getrs!(#[doc = "cuSOLVER: double-precision complex solve linear system using LU factors. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnZgetrs, cuDoubleComplex);

// ---- QR factorization (geqrf) — S/D/C/Z ----------------------------------

dn_geqrf_bufsize!(#[doc = "cuSOLVER: single-precision workspace-size query for QR factorization (Householder). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnSgeqrf_bufferSize, f32);
dn_geqrf_bufsize!(#[doc = "cuSOLVER: double-precision workspace-size query for QR factorization (Householder). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnDgeqrf_bufferSize, f64);
dn_geqrf_bufsize!(#[doc = "cuSOLVER: single-precision complex workspace-size query for QR factorization (Householder). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnCgeqrf_bufferSize, cuComplex);
dn_geqrf_bufsize!(#[doc = "cuSOLVER: double-precision complex workspace-size query for QR factorization (Householder). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnZgeqrf_bufferSize, cuDoubleComplex);

dn_geqrf!(#[doc = "cuSOLVER: single-precision QR factorization (Householder). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnSgeqrf, f32);
dn_geqrf!(#[doc = "cuSOLVER: double-precision QR factorization (Householder). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnDgeqrf, f64);
dn_geqrf!(#[doc = "cuSOLVER: single-precision complex QR factorization (Householder). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnCgeqrf, cuComplex);
dn_geqrf!(#[doc = "cuSOLVER: double-precision complex QR factorization (Householder). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnZgeqrf, cuDoubleComplex);

// ---- Cholesky (potrf / potrs) — S/D/C/Z ----------------------------------

dn_potrf_bufsize!(#[doc = "cuSOLVER: single-precision workspace-size query for Cholesky factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnSpotrf_bufferSize, f32);
dn_potrf_bufsize!(#[doc = "cuSOLVER: double-precision workspace-size query for Cholesky factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnDpotrf_bufferSize, f64);
dn_potrf_bufsize!(#[doc = "cuSOLVER: single-precision complex workspace-size query for Cholesky factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnCpotrf_bufferSize, cuComplex);
dn_potrf_bufsize!(#[doc = "cuSOLVER: double-precision complex workspace-size query for Cholesky factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnZpotrf_bufferSize, cuDoubleComplex);

dn_potrf!(#[doc = "cuSOLVER: single-precision Cholesky factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnSpotrf, f32);
dn_potrf!(#[doc = "cuSOLVER: double-precision Cholesky factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnDpotrf, f64);
dn_potrf!(#[doc = "cuSOLVER: single-precision complex Cholesky factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnCpotrf, cuComplex);
dn_potrf!(#[doc = "cuSOLVER: double-precision complex Cholesky factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnZpotrf, cuDoubleComplex);

dn_potrs!(#[doc = "cuSOLVER: single-precision solve linear system using Cholesky factors. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnSpotrs, f32);
dn_potrs!(#[doc = "cuSOLVER: double-precision solve linear system using Cholesky factors. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnDpotrs, f64);
dn_potrs!(#[doc = "cuSOLVER: single-precision complex solve linear system using Cholesky factors. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnCpotrs, cuComplex);
dn_potrs!(#[doc = "cuSOLVER: double-precision complex solve linear system using Cholesky factors. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnZpotrs, cuDoubleComplex);

// ---- SVD — S/D/C/Z -------------------------------------------------------

dn_gesvd_bufsize!(#[doc = "cuSOLVER: single-precision workspace-size query for singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnSgesvd_bufferSize);
dn_gesvd_bufsize!(#[doc = "cuSOLVER: double-precision workspace-size query for singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnDgesvd_bufferSize);
dn_gesvd_bufsize!(#[doc = "cuSOLVER: single-precision complex workspace-size query for singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnCgesvd_bufferSize);
dn_gesvd_bufsize!(#[doc = "cuSOLVER: double-precision complex workspace-size query for singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnZgesvd_bufferSize);

dn_gesvd_real!(#[doc = "cuSOLVER: single-precision singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnSgesvd, f32);
dn_gesvd_real!(#[doc = "cuSOLVER: double-precision singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnDgesvd, f64);
dn_gesvd_complex!(#[doc = "cuSOLVER: single-precision complex singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnCgesvd, cuComplex, f32);
dn_gesvd_complex!(#[doc = "cuSOLVER: double-precision complex singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnZgesvd, cuDoubleComplex, f64);

// ---- Symmetric/Hermitian eigendecomposition (syevd/heevd) --------------

dn_syevd_bufsize!(#[doc = "cuSOLVER: single-precision workspace-size query for symmetric eigendecomposition (divide-and-conquer). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnSsyevd_bufferSize, f32, f32);
dn_syevd_bufsize!(#[doc = "cuSOLVER: double-precision workspace-size query for symmetric eigendecomposition (divide-and-conquer). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnDsyevd_bufferSize, f64, f64);
dn_syevd_bufsize!(#[doc = "cuSOLVER: single-precision complex workspace-size query for Hermitian eigendecomposition (divide-and-conquer). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnCheevd_bufferSize, cuComplex, f32);
dn_syevd_bufsize!(#[doc = "cuSOLVER: double-precision complex workspace-size query for Hermitian eigendecomposition (divide-and-conquer). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnZheevd_bufferSize, cuDoubleComplex, f64);

dn_syevd!(#[doc = "cuSOLVER: single-precision symmetric eigendecomposition (divide-and-conquer). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnSsyevd, f32, f32);
dn_syevd!(#[doc = "cuSOLVER: double-precision symmetric eigendecomposition (divide-and-conquer). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnDsyevd, f64, f64);
dn_syevd!(#[doc = "cuSOLVER: single-precision complex Hermitian eigendecomposition (divide-and-conquer). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnCheevd, cuComplex, f32);
dn_syevd!(#[doc = "cuSOLVER: double-precision complex Hermitian eigendecomposition (divide-and-conquer). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnZheevd, cuDoubleComplex, f64);

// ---- Generic 64-bit / mixed-precision API (cusolverDnX…) ----------------

/// cuSOLVER: create a generic-API parameter object. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverDnCreateParams =
    unsafe extern "C" fn(params: *mut cusolverDnParams_t) -> cusolverStatus_t;
/// cuSOLVER: destroy a generic-API parameter object. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverDnDestroyParams =
    unsafe extern "C" fn(params: cusolverDnParams_t) -> cusolverStatus_t;

/// cuSOLVER: generic-API workspace-size query for LU factorization with partial pivoting (dtype configurable via `cudaDataType`). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverDnXgetrf_bufferSize = unsafe extern "C" fn(
    handle: cusolverDnHandle_t,
    params: cusolverDnParams_t,
    m: i64,
    n: i64,
    data_type_a: cudaDataType,
    a: *const c_void,
    lda: i64,
    compute_type: cudaDataType,
    workspace_in_bytes_on_device: *mut usize,
    workspace_in_bytes_on_host: *mut usize,
) -> cusolverStatus_t;

/// cuSOLVER: generic-API LU factorization with partial pivoting (dtype configurable via `cudaDataType`). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverDnXgetrf = unsafe extern "C" fn(
    handle: cusolverDnHandle_t,
    params: cusolverDnParams_t,
    m: i64,
    n: i64,
    data_type_a: cudaDataType,
    a: *mut c_void,
    lda: i64,
    ipiv: *mut i64,
    compute_type: cudaDataType,
    bufferondevice: *mut c_void,
    workspace_in_bytes_on_device: usize,
    bufferonhost: *mut c_void,
    workspace_in_bytes_on_host: usize,
    info: *mut c_int,
) -> cusolverStatus_t;

/// cuSOLVER: generic-API solve linear system using LU factors (dtype configurable via `cudaDataType`). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverDnXgetrs = unsafe extern "C" fn(
    handle: cusolverDnHandle_t,
    params: cusolverDnParams_t,
    trans: cublasOperation_t,
    n: i64,
    nrhs: i64,
    data_type_a: cudaDataType,
    a: *const c_void,
    lda: i64,
    ipiv: *const i64,
    data_type_b: cudaDataType,
    b: *mut c_void,
    ldb: i64,
    info: *mut c_int,
) -> cusolverStatus_t;

/// cuSOLVER: generic-API workspace-size query for QR factorization (Householder) (dtype configurable via `cudaDataType`). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverDnXgeqrf_bufferSize = unsafe extern "C" fn(
    handle: cusolverDnHandle_t,
    params: cusolverDnParams_t,
    m: i64,
    n: i64,
    data_type_a: cudaDataType,
    a: *const c_void,
    lda: i64,
    data_type_tau: cudaDataType,
    tau: *const c_void,
    compute_type: cudaDataType,
    workspace_in_bytes_on_device: *mut usize,
    workspace_in_bytes_on_host: *mut usize,
) -> cusolverStatus_t;

/// cuSOLVER: generic-API QR factorization (Householder) (dtype configurable via `cudaDataType`). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverDnXgeqrf = unsafe extern "C" fn(
    handle: cusolverDnHandle_t,
    params: cusolverDnParams_t,
    m: i64,
    n: i64,
    data_type_a: cudaDataType,
    a: *mut c_void,
    lda: i64,
    data_type_tau: cudaDataType,
    tau: *mut c_void,
    compute_type: cudaDataType,
    bufferondevice: *mut c_void,
    workspace_in_bytes_on_device: usize,
    bufferonhost: *mut c_void,
    workspace_in_bytes_on_host: usize,
    info: *mut c_int,
) -> cusolverStatus_t;

/// cuSOLVER: generic-API workspace-size query for Cholesky factorization (dtype configurable via `cudaDataType`). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverDnXpotrf_bufferSize = unsafe extern "C" fn(
    handle: cusolverDnHandle_t,
    params: cusolverDnParams_t,
    uplo: cublasFillMode_t,
    n: i64,
    data_type_a: cudaDataType,
    a: *const c_void,
    lda: i64,
    compute_type: cudaDataType,
    workspace_in_bytes_on_device: *mut usize,
    workspace_in_bytes_on_host: *mut usize,
) -> cusolverStatus_t;

/// cuSOLVER: generic-API Cholesky factorization (dtype configurable via `cudaDataType`). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverDnXpotrf = unsafe extern "C" fn(
    handle: cusolverDnHandle_t,
    params: cusolverDnParams_t,
    uplo: cublasFillMode_t,
    n: i64,
    data_type_a: cudaDataType,
    a: *mut c_void,
    lda: i64,
    compute_type: cudaDataType,
    bufferondevice: *mut c_void,
    workspace_in_bytes_on_device: usize,
    bufferonhost: *mut c_void,
    workspace_in_bytes_on_host: usize,
    info: *mut c_int,
) -> cusolverStatus_t;

/// cuSOLVER: generic-API solve linear system using Cholesky factors (dtype configurable via `cudaDataType`). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverDnXpotrs = unsafe extern "C" fn(
    handle: cusolverDnHandle_t,
    params: cusolverDnParams_t,
    uplo: cublasFillMode_t,
    n: i64,
    nrhs: i64,
    data_type_a: cudaDataType,
    a: *const c_void,
    lda: i64,
    data_type_b: cudaDataType,
    b: *mut c_void,
    ldb: i64,
    info: *mut c_int,
) -> cusolverStatus_t;

/// cuSOLVER: generic-API workspace-size query for symmetric eigendecomposition (divide-and-conquer) (dtype configurable via `cudaDataType`). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverDnXsyevd_bufferSize = unsafe extern "C" fn(
    handle: cusolverDnHandle_t,
    params: cusolverDnParams_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: i64,
    data_type_a: cudaDataType,
    a: *const c_void,
    lda: i64,
    data_type_w: cudaDataType,
    w: *const c_void,
    compute_type: cudaDataType,
    device_bytes: *mut usize,
    host_bytes: *mut usize,
) -> cusolverStatus_t;

/// cuSOLVER: generic-API symmetric eigendecomposition (divide-and-conquer) (dtype configurable via `cudaDataType`). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverDnXsyevd = unsafe extern "C" fn(
    handle: cusolverDnHandle_t,
    params: cusolverDnParams_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: i64,
    data_type_a: cudaDataType,
    a: *mut c_void,
    lda: i64,
    data_type_w: cudaDataType,
    w: *mut c_void,
    compute_type: cudaDataType,
    bufferondevice: *mut c_void,
    device_bytes: usize,
    bufferonhost: *mut c_void,
    host_bytes: usize,
    info: *mut c_int,
) -> cusolverStatus_t;

// ==========================================================================
// Jacobi-based eigendecompositions + SVD
// ==========================================================================

/// cuSOLVER: create a Jacobi-eigendecomposition control object. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverDnCreateSyevjInfo =
    unsafe extern "C" fn(info: *mut syevjInfo_t) -> cusolverStatus_t;
/// cuSOLVER: destroy a Jacobi-eigendecomposition control object. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverDnDestroySyevjInfo =
    unsafe extern "C" fn(info: syevjInfo_t) -> cusolverStatus_t;
/// cuSOLVER: set convergence tolerance on a Jacobi-eigendecomposition control object. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverDnXsyevjSetTolerance =
    unsafe extern "C" fn(info: syevjInfo_t, tolerance: f64) -> cusolverStatus_t;
/// cuSOLVER: set the maximum number of sweeps on a Jacobi-eigendecomposition control object. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverDnXsyevjSetMaxSweeps =
    unsafe extern "C" fn(info: syevjInfo_t, max_sweeps: c_int) -> cusolverStatus_t;

macro_rules! dn_syevj_bufsize {
    ($(#[$attr:meta])* $name:ident, $t:ty, $real:ty) => {
        $(#[$attr])*
        pub type $name = unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: c_int,
            a: *const $t,
            lda: c_int,
            w: *const $real,
            lwork: *mut c_int,
            params: syevjInfo_t,
        ) -> cusolverStatus_t;
    };
}
dn_syevj_bufsize!(#[doc = "cuSOLVER: single-precision workspace-size query for symmetric eigendecomposition (Jacobi). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnSsyevj_bufferSize, f32, f32);
dn_syevj_bufsize!(#[doc = "cuSOLVER: double-precision workspace-size query for symmetric eigendecomposition (Jacobi). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnDsyevj_bufferSize, f64, f64);
dn_syevj_bufsize!(#[doc = "cuSOLVER: single-precision complex workspace-size query for Hermitian eigendecomposition (Jacobi). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnCheevj_bufferSize, cuComplex, f32);
dn_syevj_bufsize!(#[doc = "cuSOLVER: double-precision complex workspace-size query for Hermitian eigendecomposition (Jacobi). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnZheevj_bufferSize, cuDoubleComplex, f64);

macro_rules! dn_syevj {
    ($(#[$attr:meta])* $name:ident, $t:ty, $real:ty) => {
        $(#[$attr])*
        pub type $name = unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: c_int,
            a: *mut $t,
            lda: c_int,
            w: *mut $real,
            work: *mut $t,
            lwork: c_int,
            info: *mut c_int,
            params: syevjInfo_t,
        ) -> cusolverStatus_t;
    };
}
dn_syevj!(#[doc = "cuSOLVER: single-precision symmetric eigendecomposition (Jacobi). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnSsyevj, f32, f32);
dn_syevj!(#[doc = "cuSOLVER: double-precision symmetric eigendecomposition (Jacobi). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnDsyevj, f64, f64);
dn_syevj!(#[doc = "cuSOLVER: single-precision complex Hermitian eigendecomposition (Jacobi). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnCheevj, cuComplex, f32);
dn_syevj!(#[doc = "cuSOLVER: double-precision complex Hermitian eigendecomposition (Jacobi). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnZheevj, cuDoubleComplex, f64);

/// cuSOLVER: create a Jacobi-SVD control object. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverDnCreateGesvdjInfo =
    unsafe extern "C" fn(info: *mut gesvdjInfo_t) -> cusolverStatus_t;
/// cuSOLVER: destroy a Jacobi-SVD control object. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverDnDestroyGesvdjInfo =
    unsafe extern "C" fn(info: gesvdjInfo_t) -> cusolverStatus_t;

macro_rules! dn_gesvdj_bufsize {
    ($(#[$attr:meta])* $name:ident, $t:ty, $real:ty) => {
        $(#[$attr])*
        pub type $name = unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            econ: c_int,
            m: c_int,
            n: c_int,
            a: *const $t,
            lda: c_int,
            s: *const $real,
            u: *const $t,
            ldu: c_int,
            v: *const $t,
            ldv: c_int,
            lwork: *mut c_int,
            params: gesvdjInfo_t,
        ) -> cusolverStatus_t;
    };
}
dn_gesvdj_bufsize!(#[doc = "cuSOLVER: single-precision workspace-size query for Jacobi-method singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnSgesvdj_bufferSize, f32, f32);
dn_gesvdj_bufsize!(#[doc = "cuSOLVER: double-precision workspace-size query for Jacobi-method singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnDgesvdj_bufferSize, f64, f64);
dn_gesvdj_bufsize!(#[doc = "cuSOLVER: single-precision complex workspace-size query for Jacobi-method singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnCgesvdj_bufferSize, cuComplex, f32);
dn_gesvdj_bufsize!(#[doc = "cuSOLVER: double-precision complex workspace-size query for Jacobi-method singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnZgesvdj_bufferSize, cuDoubleComplex, f64);

macro_rules! dn_gesvdj {
    ($(#[$attr:meta])* $name:ident, $t:ty, $real:ty) => {
        $(#[$attr])*
        pub type $name = unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            econ: c_int,
            m: c_int,
            n: c_int,
            a: *mut $t,
            lda: c_int,
            s: *mut $real,
            u: *mut $t,
            ldu: c_int,
            v: *mut $t,
            ldv: c_int,
            work: *mut $t,
            lwork: c_int,
            info: *mut c_int,
            params: gesvdjInfo_t,
        ) -> cusolverStatus_t;
    };
}
dn_gesvdj!(#[doc = "cuSOLVER: single-precision Jacobi-method singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnSgesvdj, f32, f32);
dn_gesvdj!(#[doc = "cuSOLVER: double-precision Jacobi-method singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnDgesvdj, f64, f64);
dn_gesvdj!(#[doc = "cuSOLVER: single-precision complex Jacobi-method singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnCgesvdj, cuComplex, f32);
dn_gesvdj!(#[doc = "cuSOLVER: double-precision complex Jacobi-method singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnZgesvdj, cuDoubleComplex, f64);

// ==========================================================================
// Apply Q from QR (orgqr / ormqr)
// ==========================================================================

macro_rules! dn_orgqr_bufsize {
    ($(#[$attr:meta])* $name:ident, $t:ty) => {
        $(#[$attr])*
        pub type $name = unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: c_int,
            n: c_int,
            k: c_int,
            a: *const $t,
            lda: c_int,
            tau: *const $t,
            lwork: *mut c_int,
        ) -> cusolverStatus_t;
    };
}
dn_orgqr_bufsize!(#[doc = "cuSOLVER: single-precision workspace-size query for generate the explicit Q from a QR factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnSorgqr_bufferSize, f32);
dn_orgqr_bufsize!(#[doc = "cuSOLVER: double-precision workspace-size query for generate the explicit Q from a QR factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnDorgqr_bufferSize, f64);
dn_orgqr_bufsize!(#[doc = "cuSOLVER: single-precision complex workspace-size query for generate the explicit unitary Q from a QR factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnCungqr_bufferSize, cuComplex);
dn_orgqr_bufsize!(#[doc = "cuSOLVER: double-precision complex workspace-size query for generate the explicit unitary Q from a QR factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnZungqr_bufferSize, cuDoubleComplex);

macro_rules! dn_orgqr {
    ($(#[$attr:meta])* $name:ident, $t:ty) => {
        $(#[$attr])*
        pub type $name = unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: c_int,
            n: c_int,
            k: c_int,
            a: *mut $t,
            lda: c_int,
            tau: *const $t,
            work: *mut $t,
            lwork: c_int,
            info: *mut c_int,
        ) -> cusolverStatus_t;
    };
}
dn_orgqr!(#[doc = "cuSOLVER: single-precision generate the explicit Q from a QR factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnSorgqr, f32);
dn_orgqr!(#[doc = "cuSOLVER: double-precision generate the explicit Q from a QR factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnDorgqr, f64);
dn_orgqr!(#[doc = "cuSOLVER: single-precision complex generate the explicit unitary Q from a QR factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnCungqr, cuComplex);
dn_orgqr!(#[doc = "cuSOLVER: double-precision complex generate the explicit unitary Q from a QR factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnZungqr, cuDoubleComplex);

macro_rules! dn_ormqr_bufsize {
    ($(#[$attr:meta])* $name:ident, $t:ty) => {
        $(#[$attr])*
        pub type $name = unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            side: c_int,
            trans: cublasOperation_t,
            m: c_int,
            n: c_int,
            k: c_int,
            a: *const $t,
            lda: c_int,
            tau: *const $t,
            c: *const $t,
            ldc: c_int,
            lwork: *mut c_int,
        ) -> cusolverStatus_t;
    };
}
dn_ormqr_bufsize!(#[doc = "cuSOLVER: single-precision workspace-size query for apply Q (or Q^T) from a QR factorization to a matrix. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnSormqr_bufferSize, f32);
dn_ormqr_bufsize!(#[doc = "cuSOLVER: double-precision workspace-size query for apply Q (or Q^T) from a QR factorization to a matrix. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnDormqr_bufferSize, f64);
dn_ormqr_bufsize!(#[doc = "cuSOLVER: single-precision complex workspace-size query for apply Q (or Q^H) from a QR factorization to a matrix. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnCunmqr_bufferSize, cuComplex);
dn_ormqr_bufsize!(#[doc = "cuSOLVER: double-precision complex workspace-size query for apply Q (or Q^H) from a QR factorization to a matrix. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnZunmqr_bufferSize, cuDoubleComplex);

macro_rules! dn_ormqr {
    ($(#[$attr:meta])* $name:ident, $t:ty) => {
        $(#[$attr])*
        pub type $name = unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            side: c_int,
            trans: cublasOperation_t,
            m: c_int,
            n: c_int,
            k: c_int,
            a: *const $t,
            lda: c_int,
            tau: *const $t,
            c: *mut $t,
            ldc: c_int,
            work: *mut $t,
            lwork: c_int,
            info: *mut c_int,
        ) -> cusolverStatus_t;
    };
}
dn_ormqr!(#[doc = "cuSOLVER: single-precision apply Q (or Q^T) from a QR factorization to a matrix. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnSormqr, f32);
dn_ormqr!(#[doc = "cuSOLVER: double-precision apply Q (or Q^T) from a QR factorization to a matrix. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnDormqr, f64);
dn_ormqr!(#[doc = "cuSOLVER: single-precision complex apply Q (or Q^H) from a QR factorization to a matrix. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnCunmqr, cuComplex);
dn_ormqr!(#[doc = "cuSOLVER: double-precision complex apply Q (or Q^H) from a QR factorization to a matrix. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnZunmqr, cuDoubleComplex);

// ---- Sparse cuSOLVER -----------------------------------------------------

/// cuSOLVER: create a sparse cuSOLVER handle. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverSpCreate =
    unsafe extern "C" fn(handle: *mut cusolverSpHandle_t) -> cusolverStatus_t;
/// cuSOLVER: destroy a sparse cuSOLVER handle. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverSpDestroy =
    unsafe extern "C" fn(handle: cusolverSpHandle_t) -> cusolverStatus_t;
/// cuSOLVER: bind a CUDA stream to a sparse cuSOLVER handle. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverSpSetStream =
    unsafe extern "C" fn(handle: cusolverSpHandle_t, stream: cudaStream_t) -> cusolverStatus_t;

/// cuSOLVER: single-precision sparse linear solve via Cholesky on a CSR matrix. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverSpScsrlsvchol = unsafe extern "C" fn(
    handle: cusolverSpHandle_t,
    m: c_int,
    nnz: c_int,
    descr_a: *mut c_void,
    csr_val: *const f32,
    csr_row_ptr: *const c_int,
    csr_col_ind: *const c_int,
    b: *const f32,
    tol: f32,
    reorder: c_int,
    x: *mut f32,
    singularity: *mut c_int,
) -> cusolverStatus_t;

/// cuSOLVER: double-precision sparse linear solve via Cholesky on a CSR matrix. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverSpDcsrlsvchol = unsafe extern "C" fn(
    handle: cusolverSpHandle_t,
    m: c_int,
    nnz: c_int,
    descr_a: *mut c_void,
    csr_val: *const f64,
    csr_row_ptr: *const c_int,
    csr_col_ind: *const c_int,
    b: *const f64,
    tol: f64,
    reorder: c_int,
    x: *mut f64,
    singularity: *mut c_int,
) -> cusolverStatus_t;

/// cuSOLVER: single-precision sparse linear solve via QR on a CSR matrix. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverSpScsrlsvqr = unsafe extern "C" fn(
    handle: cusolverSpHandle_t,
    m: c_int,
    nnz: c_int,
    descr_a: *mut c_void,
    csr_val: *const f32,
    csr_row_ptr: *const c_int,
    csr_col_ind: *const c_int,
    b: *const f32,
    tol: f32,
    reorder: c_int,
    x: *mut f32,
    singularity: *mut c_int,
) -> cusolverStatus_t;

/// cuSOLVER: double-precision sparse linear solve via QR on a CSR matrix. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverSpDcsrlsvqr = unsafe extern "C" fn(
    handle: cusolverSpHandle_t,
    m: c_int,
    nnz: c_int,
    descr_a: *mut c_void,
    csr_val: *const f64,
    csr_row_ptr: *const c_int,
    csr_col_ind: *const c_int,
    b: *const f64,
    tol: f64,
    reorder: c_int,
    x: *mut f64,
    singularity: *mut c_int,
) -> cusolverStatus_t;

// ---- Refactor (Rf) sparse LU refactorization -----------------------------

/// cuSOLVER: create a sparse-refactor cuSOLVER handle. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverRfCreate =
    unsafe extern "C" fn(handle: *mut cusolverRfHandle_t) -> cusolverStatus_t;
/// cuSOLVER: destroy a sparse-refactor cuSOLVER handle. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverRfDestroy =
    unsafe extern "C" fn(handle: cusolverRfHandle_t) -> cusolverStatus_t;
/// cuSOLVER: supply sparse triangular factors and pivot vectors to the refactor handle. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverRfSetupDevice = unsafe extern "C" fn(
    n: c_int,
    nnz_a: c_int,
    h_csr_row_ptr_a: *mut c_int,
    h_csr_col_ind_a: *mut c_int,
    h_csr_val_a: *mut f64,
    nnz_l: c_int,
    h_csr_row_ptr_l: *mut c_int,
    h_csr_col_ind_l: *mut c_int,
    h_csr_val_l: *mut f64,
    nnz_u: c_int,
    h_csr_row_ptr_u: *mut c_int,
    h_csr_col_ind_u: *mut c_int,
    h_csr_val_u: *mut f64,
    p: *mut c_int,
    q: *mut c_int,
    handle: cusolverRfHandle_t,
) -> cusolverStatus_t;
/// cuSOLVER: analyze the sparsity pattern of the refactor handle. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverRfAnalyze =
    unsafe extern "C" fn(handle: cusolverRfHandle_t) -> cusolverStatus_t;
/// cuSOLVER: refactor with new numerical values reusing the analyzed pattern. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverRfRefactor =
    unsafe extern "C" fn(handle: cusolverRfHandle_t) -> cusolverStatus_t;
/// cuSOLVER: solve linear systems using the refactor handle. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverRfSolve = unsafe extern "C" fn(
    handle: cusolverRfHandle_t,
    p: *mut c_int,
    q: *mut c_int,
    nrhs: c_int,
    temp: *mut f64,
    ld_temp: c_int,
    xf: *mut f64,
    ld_xf: c_int,
) -> cusolverStatus_t;

// ==========================================================================
// Least-squares (gels): A*X = B → X
// ==========================================================================

macro_rules! dn_gels_bufsize {
    ($(#[$attr:meta])* $name:ident, $t:ty) => {
        $(#[$attr])*
        pub type $name = unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: c_int,
            n: c_int,
            nrhs: c_int,
            d_a: *mut $t,
            lda: c_int,
            d_b: *mut $t,
            ldb: c_int,
            d_x: *mut $t,
            ldx: c_int,
            d_work: *mut c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t;
    };
}
dn_gels_bufsize!(#[doc = "cuSOLVER: single-precision workspace-size query for least-squares solver (A*X = B). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnSSgels_bufferSize, f32);
dn_gels_bufsize!(#[doc = "cuSOLVER: double-precision workspace-size query for least-squares solver (A*X = B). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnDDgels_bufferSize, f64);
dn_gels_bufsize!(#[doc = "cuSOLVER: single-precision complex workspace-size query for least-squares solver (A*X = B). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnCCgels_bufferSize, cuComplex);
dn_gels_bufsize!(#[doc = "cuSOLVER: double-precision complex workspace-size query for least-squares solver (A*X = B). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnZZgels_bufferSize, cuDoubleComplex);

macro_rules! dn_gels {
    ($(#[$attr:meta])* $name:ident, $t:ty) => {
        $(#[$attr])*
        pub type $name = unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: c_int,
            n: c_int,
            nrhs: c_int,
            d_a: *mut $t,
            lda: c_int,
            d_b: *mut $t,
            ldb: c_int,
            d_x: *mut $t,
            ldx: c_int,
            d_work: *mut c_void,
            lwork_bytes: usize,
            iter: *mut c_int,
            d_info: *mut c_int,
        ) -> cusolverStatus_t;
    };
}
dn_gels!(#[doc = "cuSOLVER: single-precision least-squares solver (A*X = B). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnSSgels, f32);
dn_gels!(#[doc = "cuSOLVER: double-precision least-squares solver (A*X = B). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnDDgels, f64);
dn_gels!(#[doc = "cuSOLVER: single-precision complex least-squares solver (A*X = B). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnCCgels, cuComplex);
dn_gels!(#[doc = "cuSOLVER: double-precision complex least-squares solver (A*X = B). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnZZgels, cuDoubleComplex);

// ==========================================================================
// Inverse from Cholesky (potri)
// ==========================================================================

macro_rules! dn_potri_bufsize {
    ($(#[$attr:meta])* $name:ident, $t:ty) => {
        $(#[$attr])*
        pub type $name = unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: c_int,
            a: *mut $t,
            lda: c_int,
            lwork: *mut c_int,
        ) -> cusolverStatus_t;
    };
}
dn_potri_bufsize!(#[doc = "cuSOLVER: single-precision workspace-size query for matrix inverse from Cholesky factors. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnSpotri_bufferSize, f32);
dn_potri_bufsize!(#[doc = "cuSOLVER: double-precision workspace-size query for matrix inverse from Cholesky factors. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnDpotri_bufferSize, f64);
dn_potri_bufsize!(#[doc = "cuSOLVER: single-precision complex workspace-size query for matrix inverse from Cholesky factors. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnCpotri_bufferSize, cuComplex);
dn_potri_bufsize!(#[doc = "cuSOLVER: double-precision complex workspace-size query for matrix inverse from Cholesky factors. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnZpotri_bufferSize, cuDoubleComplex);

macro_rules! dn_potri {
    ($(#[$attr:meta])* $name:ident, $t:ty) => {
        $(#[$attr])*
        pub type $name = unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: c_int,
            a: *mut $t,
            lda: c_int,
            work: *mut $t,
            lwork: c_int,
            info: *mut c_int,
        ) -> cusolverStatus_t;
    };
}
dn_potri!(#[doc = "cuSOLVER: single-precision matrix inverse from Cholesky factors. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnSpotri, f32);
dn_potri!(#[doc = "cuSOLVER: double-precision matrix inverse from Cholesky factors. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnDpotri, f64);
dn_potri!(#[doc = "cuSOLVER: single-precision complex matrix inverse from Cholesky factors. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnCpotri, cuComplex);
dn_potri!(#[doc = "cuSOLVER: double-precision complex matrix inverse from Cholesky factors. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnZpotri, cuDoubleComplex);

// ==========================================================================
// Batched Jacobi eigen / SVD
// ==========================================================================

macro_rules! dn_syevj_batched_bufsize {
    ($(#[$attr:meta])* $name:ident, $t:ty, $real:ty) => {
        $(#[$attr])*
        pub type $name = unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: c_int,
            a: *const $t,
            lda: c_int,
            w: *const $real,
            lwork: *mut c_int,
            params: syevjInfo_t,
            batch_size: c_int,
        ) -> cusolverStatus_t;
    };
}
dn_syevj_batched_bufsize!(#[doc = "cuSOLVER: single-precision workspace-size query for batched symmetric eigendecomposition (Jacobi). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnSsyevjBatched_bufferSize, f32, f32);
dn_syevj_batched_bufsize!(#[doc = "cuSOLVER: double-precision workspace-size query for batched symmetric eigendecomposition (Jacobi). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnDsyevjBatched_bufferSize, f64, f64);
dn_syevj_batched_bufsize!(#[doc = "cuSOLVER: single-precision complex workspace-size query for batched Hermitian eigendecomposition (Jacobi). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnCheevjBatched_bufferSize, cuComplex, f32);
dn_syevj_batched_bufsize!(#[doc = "cuSOLVER: double-precision complex workspace-size query for batched Hermitian eigendecomposition (Jacobi). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnZheevjBatched_bufferSize, cuDoubleComplex, f64);

macro_rules! dn_syevj_batched {
    ($(#[$attr:meta])* $name:ident, $t:ty, $real:ty) => {
        $(#[$attr])*
        pub type $name = unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: c_int,
            a: *mut $t,
            lda: c_int,
            w: *mut $real,
            work: *mut $t,
            lwork: c_int,
            info: *mut c_int,
            params: syevjInfo_t,
            batch_size: c_int,
        ) -> cusolverStatus_t;
    };
}
dn_syevj_batched!(#[doc = "cuSOLVER: single-precision batched symmetric eigendecomposition (Jacobi). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnSsyevjBatched, f32, f32);
dn_syevj_batched!(#[doc = "cuSOLVER: double-precision batched symmetric eigendecomposition (Jacobi). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnDsyevjBatched, f64, f64);
dn_syevj_batched!(#[doc = "cuSOLVER: single-precision complex batched Hermitian eigendecomposition (Jacobi). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnCheevjBatched, cuComplex, f32);
dn_syevj_batched!(#[doc = "cuSOLVER: double-precision complex batched Hermitian eigendecomposition (Jacobi). See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnZheevjBatched, cuDoubleComplex, f64);

macro_rules! dn_gesvdj_batched_bufsize {
    ($(#[$attr:meta])* $name:ident, $t:ty, $real:ty) => {
        $(#[$attr])*
        pub type $name = unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            m: c_int,
            n: c_int,
            a: *const $t,
            lda: c_int,
            s: *const $real,
            u: *const $t,
            ldu: c_int,
            v: *const $t,
            ldv: c_int,
            lwork: *mut c_int,
            params: gesvdjInfo_t,
            batch_size: c_int,
        ) -> cusolverStatus_t;
    };
}
dn_gesvdj_batched_bufsize!(#[doc = "cuSOLVER: single-precision workspace-size query for batched Jacobi-method singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnSgesvdjBatched_bufferSize, f32, f32);
dn_gesvdj_batched_bufsize!(#[doc = "cuSOLVER: double-precision workspace-size query for batched Jacobi-method singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnDgesvdjBatched_bufferSize, f64, f64);
dn_gesvdj_batched_bufsize!(#[doc = "cuSOLVER: single-precision complex workspace-size query for batched Jacobi-method singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnCgesvdjBatched_bufferSize, cuComplex, f32);
dn_gesvdj_batched_bufsize!(#[doc = "cuSOLVER: double-precision complex workspace-size query for batched Jacobi-method singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnZgesvdjBatched_bufferSize, cuDoubleComplex, f64);

macro_rules! dn_gesvdj_batched {
    ($(#[$attr:meta])* $name:ident, $t:ty, $real:ty) => {
        $(#[$attr])*
        pub type $name = unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            m: c_int,
            n: c_int,
            a: *mut $t,
            lda: c_int,
            s: *mut $real,
            u: *mut $t,
            ldu: c_int,
            v: *mut $t,
            ldv: c_int,
            work: *mut $t,
            lwork: c_int,
            info: *mut c_int,
            params: gesvdjInfo_t,
            batch_size: c_int,
        ) -> cusolverStatus_t;
    };
}
dn_gesvdj_batched!(#[doc = "cuSOLVER: single-precision batched Jacobi-method singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnSgesvdjBatched, f32, f32);
dn_gesvdj_batched!(#[doc = "cuSOLVER: double-precision batched Jacobi-method singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnDgesvdjBatched, f64, f64);
dn_gesvdj_batched!(#[doc = "cuSOLVER: single-precision complex batched Jacobi-method singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnCgesvdjBatched, cuComplex, f32);
dn_gesvdj_batched!(#[doc = "cuSOLVER: double-precision complex batched Jacobi-method singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>."] PFN_cusolverDnZgesvdjBatched, cuDoubleComplex, f64);

// ==========================================================================
// cuSOLVERMg — multi-GPU dense solvers (separate library libcusolverMg)
// ==========================================================================

/// Opaque multi-GPU dense cuSOLVERMg handle.
pub type cusolverMgHandle_t = *mut c_void;
/// Opaque multi-GPU matrix descriptor.
pub type cudaLibMgMatrixDesc_t = *mut c_void;
/// Opaque multi-GPU device grid.
pub type cudaLibMgGrid_t = *mut c_void;

/// cuSOLVER: create a multi-GPU dense cuSOLVERMg handle. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverMgCreate =
    unsafe extern "C" fn(handle: *mut cusolverMgHandle_t) -> cusolverStatus_t;
/// cuSOLVER: destroy a multi-GPU dense cuSOLVERMg handle. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverMgDestroy =
    unsafe extern "C" fn(handle: cusolverMgHandle_t) -> cusolverStatus_t;
/// cuSOLVER: select GPU devices for a cuSOLVERMg handle. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverMgDeviceSelect = unsafe extern "C" fn(
    handle: cusolverMgHandle_t,
    n_devices: c_int,
    device_id: *const c_int,
) -> cusolverStatus_t;

/// cuSOLVER: create a multi-GPU device grid. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverMgCreateDeviceGrid = unsafe extern "C" fn(
    grid: *mut cudaLibMgGrid_t,
    num_row_devices: i32,
    num_col_devices: i32,
    device_id: *const i32,
    mapping: i32,
) -> cusolverStatus_t;

/// cuSOLVER: destroy a multi-GPU device grid. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverMgDestroyGrid =
    unsafe extern "C" fn(grid: cudaLibMgGrid_t) -> cusolverStatus_t;

/// cuSOLVER: create a multi-GPU matrix descriptor. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverMgCreateMatrixDesc = unsafe extern "C" fn(
    desc: *mut cudaLibMgMatrixDesc_t,
    num_rows: i64,
    num_cols: i64,
    row_block_size: i64,
    col_block_size: i64,
    data_type: cudaDataType,
    grid: cudaLibMgGrid_t,
) -> cusolverStatus_t;

/// cuSOLVER: destroy a multi-GPU matrix descriptor. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverMgDestroyMatrixDesc =
    unsafe extern "C" fn(desc: cudaLibMgMatrixDesc_t) -> cusolverStatus_t;

/// cuSOLVER: multi-GPU workspace-size query for LU factorization with partial pivoting. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverMgGetrf_bufferSize = unsafe extern "C" fn(
    handle: cusolverMgHandle_t,
    m: c_int,
    n: c_int,
    array_d_a: *mut *mut c_void,
    ia: c_int,
    ja: c_int,
    desc_a: cudaLibMgMatrixDesc_t,
    array_d_ipiv: *mut *mut c_int,
    compute_type: cudaDataType,
    lwork: *mut i64,
) -> cusolverStatus_t;

/// cuSOLVER: multi-GPU LU factorization with partial pivoting. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverMgGetrf = unsafe extern "C" fn(
    handle: cusolverMgHandle_t,
    m: c_int,
    n: c_int,
    array_d_a: *mut *mut c_void,
    ia: c_int,
    ja: c_int,
    desc_a: cudaLibMgMatrixDesc_t,
    array_d_ipiv: *mut *mut c_int,
    compute_type: cudaDataType,
    array_d_work: *mut *mut c_void,
    lwork: i64,
    info: *mut c_int,
) -> cusolverStatus_t;

/// cuSOLVER: multi-GPU workspace-size query for Cholesky factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverMgPotrf_bufferSize = unsafe extern "C" fn(
    handle: cusolverMgHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    array_d_a: *mut *mut c_void,
    ia: c_int,
    ja: c_int,
    desc_a: cudaLibMgMatrixDesc_t,
    compute_type: cudaDataType,
    lwork: *mut i64,
) -> cusolverStatus_t;

/// cuSOLVER: multi-GPU Cholesky factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverMgPotrf = unsafe extern "C" fn(
    handle: cusolverMgHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    array_d_a: *mut *mut c_void,
    ia: c_int,
    ja: c_int,
    desc_a: cudaLibMgMatrixDesc_t,
    compute_type: cudaDataType,
    array_d_work: *mut *mut c_void,
    lwork: i64,
    info: *mut c_int,
) -> cusolverStatus_t;

/// cuSOLVER: multi-GPU workspace-size query for symmetric eigendecomposition (divide-and-conquer). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverMgSyevd_bufferSize = unsafe extern "C" fn(
    handle: cusolverMgHandle_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: c_int,
    array_d_a: *mut *mut c_void,
    ia: c_int,
    ja: c_int,
    desc_a: cudaLibMgMatrixDesc_t,
    w: *mut c_void,
    data_type_w: cudaDataType,
    compute_type: cudaDataType,
    lwork: *mut i64,
) -> cusolverStatus_t;

/// cuSOLVER: multi-GPU symmetric eigendecomposition (divide-and-conquer). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
pub type PFN_cusolverMgSyevd = unsafe extern "C" fn(
    handle: cusolverMgHandle_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: c_int,
    array_d_a: *mut *mut c_void,
    ia: c_int,
    ja: c_int,
    desc_a: cudaLibMgMatrixDesc_t,
    w: *mut c_void,
    data_type_w: cudaDataType,
    compute_type: cudaDataType,
    array_d_work: *mut *mut c_void,
    lwork: i64,
    info: *mut c_int,
) -> cusolverStatus_t;

// ---- loader --------------------------------------------------------------

fn cusolver_candidates() -> Vec<String> {
    platform::versioned_library_candidates("cusolver", &["13", "12", "11"])
}

macro_rules! cusolver_fns {
    ($($(#[$m:meta])* $name:ident as $sym:literal : $pfn:ty);* $(;)?) => {
        /// Loaded cuSOLVER shared library plus a per-symbol `OnceLock` of function pointers.
        pub struct Cusolver {
            lib: Library,
            $($name: OnceLock<$pfn>,)*
        }
        impl core::fmt::Debug for Cusolver {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct("Cusolver").field("lib", &self.lib).finish_non_exhaustive()
            }
        }
        impl Cusolver {
            $(
                $(#[$m])*
                pub fn $name(&self) -> Result<$pfn, LoaderError> {
                    if let Some(&p) = self.$name.get() { return Ok(p); }
                    let raw: *mut () = unsafe { self.lib.raw_symbol($sym)? };
                    let p: $pfn = unsafe { core::mem::transmute_copy::<*mut (), $pfn>(&raw) };
                    let _ = self.$name.set(p);
                    Ok(p)
                }
            )*
            fn empty(lib: Library) -> Self {
                Self { lib, $($name: OnceLock::new(),)* }
            }
        }
    };
}

cusolver_fns! {
    // Dn handle
    /// cuSOLVER: create a dense cuSOLVER handle. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_create as "cusolverDnCreate": PFN_cusolverDnCreate;
    /// cuSOLVER: destroy a dense cuSOLVER handle. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_destroy as "cusolverDnDestroy": PFN_cusolverDnDestroy;
    /// cuSOLVER: bind a CUDA stream to a dense cuSOLVER handle. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_set_stream as "cusolverDnSetStream": PFN_cusolverDnSetStream;
    /// cuSOLVER: query the CUDA stream bound to a dense cuSOLVER handle. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_get_stream as "cusolverDnGetStream": PFN_cusolverDnGetStream;
    /// cuSOLVER: return the cuSOLVER library version. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_get_version as "cusolverGetVersion": PFN_cusolverGetVersion;
    // LU (getrf/getrs) S/D/C/Z
    /// cuSOLVER: single-precision workspace-size query for LU factorization with partial pivoting. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_sgetrf_buffer_size as "cusolverDnSgetrf_bufferSize": PFN_cusolverDnSgetrf_bufferSize;
    /// cuSOLVER: double-precision workspace-size query for LU factorization with partial pivoting. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_dgetrf_buffer_size as "cusolverDnDgetrf_bufferSize": PFN_cusolverDnDgetrf_bufferSize;
    /// cuSOLVER: single-precision complex workspace-size query for LU factorization with partial pivoting. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_cgetrf_buffer_size as "cusolverDnCgetrf_bufferSize": PFN_cusolverDnCgetrf_bufferSize;
    /// cuSOLVER: double-precision complex workspace-size query for LU factorization with partial pivoting. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_zgetrf_buffer_size as "cusolverDnZgetrf_bufferSize": PFN_cusolverDnZgetrf_bufferSize;
    /// cuSOLVER: single-precision LU factorization with partial pivoting. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_sgetrf as "cusolverDnSgetrf": PFN_cusolverDnSgetrf;
    /// cuSOLVER: double-precision LU factorization with partial pivoting. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_dgetrf as "cusolverDnDgetrf": PFN_cusolverDnDgetrf;
    /// cuSOLVER: single-precision complex LU factorization with partial pivoting. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_cgetrf as "cusolverDnCgetrf": PFN_cusolverDnCgetrf;
    /// cuSOLVER: double-precision complex LU factorization with partial pivoting. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_zgetrf as "cusolverDnZgetrf": PFN_cusolverDnZgetrf;
    /// cuSOLVER: single-precision solve linear system using LU factors. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_sgetrs as "cusolverDnSgetrs": PFN_cusolverDnSgetrs;
    /// cuSOLVER: double-precision solve linear system using LU factors. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_dgetrs as "cusolverDnDgetrs": PFN_cusolverDnDgetrs;
    /// cuSOLVER: single-precision complex solve linear system using LU factors. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_cgetrs as "cusolverDnCgetrs": PFN_cusolverDnCgetrs;
    /// cuSOLVER: double-precision complex solve linear system using LU factors. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_zgetrs as "cusolverDnZgetrs": PFN_cusolverDnZgetrs;
    // QR (geqrf) S/D/C/Z
    /// cuSOLVER: single-precision workspace-size query for QR factorization (Householder). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_sgeqrf_buffer_size as "cusolverDnSgeqrf_bufferSize": PFN_cusolverDnSgeqrf_bufferSize;
    /// cuSOLVER: double-precision workspace-size query for QR factorization (Householder). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_dgeqrf_buffer_size as "cusolverDnDgeqrf_bufferSize": PFN_cusolverDnDgeqrf_bufferSize;
    /// cuSOLVER: single-precision complex workspace-size query for QR factorization (Householder). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_cgeqrf_buffer_size as "cusolverDnCgeqrf_bufferSize": PFN_cusolverDnCgeqrf_bufferSize;
    /// cuSOLVER: double-precision complex workspace-size query for QR factorization (Householder). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_zgeqrf_buffer_size as "cusolverDnZgeqrf_bufferSize": PFN_cusolverDnZgeqrf_bufferSize;
    /// cuSOLVER: single-precision QR factorization (Householder). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_sgeqrf as "cusolverDnSgeqrf": PFN_cusolverDnSgeqrf;
    /// cuSOLVER: double-precision QR factorization (Householder). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_dgeqrf as "cusolverDnDgeqrf": PFN_cusolverDnDgeqrf;
    /// cuSOLVER: single-precision complex QR factorization (Householder). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_cgeqrf as "cusolverDnCgeqrf": PFN_cusolverDnCgeqrf;
    /// cuSOLVER: double-precision complex QR factorization (Householder). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_zgeqrf as "cusolverDnZgeqrf": PFN_cusolverDnZgeqrf;
    // Cholesky (potrf/potrs) S/D/C/Z
    /// cuSOLVER: single-precision workspace-size query for Cholesky factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_spotrf_buffer_size as "cusolverDnSpotrf_bufferSize": PFN_cusolverDnSpotrf_bufferSize;
    /// cuSOLVER: double-precision workspace-size query for Cholesky factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_dpotrf_buffer_size as "cusolverDnDpotrf_bufferSize": PFN_cusolverDnDpotrf_bufferSize;
    /// cuSOLVER: single-precision complex workspace-size query for Cholesky factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_cpotrf_buffer_size as "cusolverDnCpotrf_bufferSize": PFN_cusolverDnCpotrf_bufferSize;
    /// cuSOLVER: double-precision complex workspace-size query for Cholesky factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_zpotrf_buffer_size as "cusolverDnZpotrf_bufferSize": PFN_cusolverDnZpotrf_bufferSize;
    /// cuSOLVER: single-precision Cholesky factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_spotrf as "cusolverDnSpotrf": PFN_cusolverDnSpotrf;
    /// cuSOLVER: double-precision Cholesky factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_dpotrf as "cusolverDnDpotrf": PFN_cusolverDnDpotrf;
    /// cuSOLVER: single-precision complex Cholesky factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_cpotrf as "cusolverDnCpotrf": PFN_cusolverDnCpotrf;
    /// cuSOLVER: double-precision complex Cholesky factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_zpotrf as "cusolverDnZpotrf": PFN_cusolverDnZpotrf;
    /// cuSOLVER: single-precision solve linear system using Cholesky factors. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_spotrs as "cusolverDnSpotrs": PFN_cusolverDnSpotrs;
    /// cuSOLVER: double-precision solve linear system using Cholesky factors. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_dpotrs as "cusolverDnDpotrs": PFN_cusolverDnDpotrs;
    /// cuSOLVER: single-precision complex solve linear system using Cholesky factors. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_cpotrs as "cusolverDnCpotrs": PFN_cusolverDnCpotrs;
    /// cuSOLVER: double-precision complex solve linear system using Cholesky factors. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_zpotrs as "cusolverDnZpotrs": PFN_cusolverDnZpotrs;
    // SVD S/D/C/Z
    /// cuSOLVER: single-precision workspace-size query for singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_sgesvd_buffer_size as "cusolverDnSgesvd_bufferSize": PFN_cusolverDnSgesvd_bufferSize;
    /// cuSOLVER: double-precision workspace-size query for singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_dgesvd_buffer_size as "cusolverDnDgesvd_bufferSize": PFN_cusolverDnDgesvd_bufferSize;
    /// cuSOLVER: single-precision complex workspace-size query for singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_cgesvd_buffer_size as "cusolverDnCgesvd_bufferSize": PFN_cusolverDnCgesvd_bufferSize;
    /// cuSOLVER: double-precision complex workspace-size query for singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_zgesvd_buffer_size as "cusolverDnZgesvd_bufferSize": PFN_cusolverDnZgesvd_bufferSize;
    /// cuSOLVER: single-precision singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_sgesvd as "cusolverDnSgesvd": PFN_cusolverDnSgesvd;
    /// cuSOLVER: double-precision singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_dgesvd as "cusolverDnDgesvd": PFN_cusolverDnDgesvd;
    /// cuSOLVER: single-precision complex singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_cgesvd as "cusolverDnCgesvd": PFN_cusolverDnCgesvd;
    /// cuSOLVER: double-precision complex singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_zgesvd as "cusolverDnZgesvd": PFN_cusolverDnZgesvd;
    // syevd / heevd
    /// cuSOLVER: single-precision workspace-size query for symmetric eigendecomposition (divide-and-conquer). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_ssyevd_buffer_size as "cusolverDnSsyevd_bufferSize": PFN_cusolverDnSsyevd_bufferSize;
    /// cuSOLVER: double-precision workspace-size query for symmetric eigendecomposition (divide-and-conquer). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_dsyevd_buffer_size as "cusolverDnDsyevd_bufferSize": PFN_cusolverDnDsyevd_bufferSize;
    /// cuSOLVER: single-precision complex workspace-size query for Hermitian eigendecomposition (divide-and-conquer). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_cheevd_buffer_size as "cusolverDnCheevd_bufferSize": PFN_cusolverDnCheevd_bufferSize;
    /// cuSOLVER: double-precision complex workspace-size query for Hermitian eigendecomposition (divide-and-conquer). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_zheevd_buffer_size as "cusolverDnZheevd_bufferSize": PFN_cusolverDnZheevd_bufferSize;
    /// cuSOLVER: single-precision symmetric eigendecomposition (divide-and-conquer). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_ssyevd as "cusolverDnSsyevd": PFN_cusolverDnSsyevd;
    /// cuSOLVER: double-precision symmetric eigendecomposition (divide-and-conquer). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_dsyevd as "cusolverDnDsyevd": PFN_cusolverDnDsyevd;
    /// cuSOLVER: single-precision complex Hermitian eigendecomposition (divide-and-conquer). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_cheevd as "cusolverDnCheevd": PFN_cusolverDnCheevd;
    /// cuSOLVER: double-precision complex Hermitian eigendecomposition (divide-and-conquer). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_zheevd as "cusolverDnZheevd": PFN_cusolverDnZheevd;
    // Generic 64-bit X… API
    /// cuSOLVER: create a generic-API parameter object. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_create_params as "cusolverDnCreateParams": PFN_cusolverDnCreateParams;
    /// cuSOLVER: destroy a generic-API parameter object. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_destroy_params as "cusolverDnDestroyParams": PFN_cusolverDnDestroyParams;
    /// cuSOLVER: generic-API workspace-size query for LU factorization with partial pivoting (dtype configurable via `cudaDataType`). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_xgetrf_buffer_size as "cusolverDnXgetrf_bufferSize": PFN_cusolverDnXgetrf_bufferSize;
    /// cuSOLVER: generic-API LU factorization with partial pivoting (dtype configurable via `cudaDataType`). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_xgetrf as "cusolverDnXgetrf": PFN_cusolverDnXgetrf;
    /// cuSOLVER: generic-API solve linear system using LU factors (dtype configurable via `cudaDataType`). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_xgetrs as "cusolverDnXgetrs": PFN_cusolverDnXgetrs;
    /// cuSOLVER: generic-API workspace-size query for QR factorization (Householder) (dtype configurable via `cudaDataType`). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_xgeqrf_buffer_size as "cusolverDnXgeqrf_bufferSize": PFN_cusolverDnXgeqrf_bufferSize;
    /// cuSOLVER: generic-API QR factorization (Householder) (dtype configurable via `cudaDataType`). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_xgeqrf as "cusolverDnXgeqrf": PFN_cusolverDnXgeqrf;
    /// cuSOLVER: generic-API workspace-size query for Cholesky factorization (dtype configurable via `cudaDataType`). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_xpotrf_buffer_size as "cusolverDnXpotrf_bufferSize": PFN_cusolverDnXpotrf_bufferSize;
    /// cuSOLVER: generic-API Cholesky factorization (dtype configurable via `cudaDataType`). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_xpotrf as "cusolverDnXpotrf": PFN_cusolverDnXpotrf;
    /// cuSOLVER: generic-API solve linear system using Cholesky factors (dtype configurable via `cudaDataType`). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_xpotrs as "cusolverDnXpotrs": PFN_cusolverDnXpotrs;
    /// cuSOLVER: generic-API workspace-size query for symmetric eigendecomposition (divide-and-conquer) (dtype configurable via `cudaDataType`). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_xsyevd_buffer_size as "cusolverDnXsyevd_bufferSize": PFN_cusolverDnXsyevd_bufferSize;
    /// cuSOLVER: generic-API symmetric eigendecomposition (divide-and-conquer) (dtype configurable via `cudaDataType`). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_xsyevd as "cusolverDnXsyevd": PFN_cusolverDnXsyevd;
    // Jacobi eigen
    /// cuSOLVER: create a Jacobi-eigendecomposition control object. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_create_syevj_info as "cusolverDnCreateSyevjInfo": PFN_cusolverDnCreateSyevjInfo;
    /// cuSOLVER: destroy a Jacobi-eigendecomposition control object. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_destroy_syevj_info as "cusolverDnDestroySyevjInfo": PFN_cusolverDnDestroySyevjInfo;
    /// cuSOLVER: set convergence tolerance on a Jacobi-eigendecomposition control object. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_xsyevj_set_tolerance as "cusolverDnXsyevjSetTolerance": PFN_cusolverDnXsyevjSetTolerance;
    /// cuSOLVER: set the maximum number of sweeps on a Jacobi-eigendecomposition control object. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_xsyevj_set_max_sweeps as "cusolverDnXsyevjSetMaxSweeps": PFN_cusolverDnXsyevjSetMaxSweeps;
    /// cuSOLVER: single-precision workspace-size query for symmetric eigendecomposition (Jacobi). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_ssyevj_buffer_size as "cusolverDnSsyevj_bufferSize": PFN_cusolverDnSsyevj_bufferSize;
    /// cuSOLVER: double-precision workspace-size query for symmetric eigendecomposition (Jacobi). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_dsyevj_buffer_size as "cusolverDnDsyevj_bufferSize": PFN_cusolverDnDsyevj_bufferSize;
    /// cuSOLVER: single-precision complex workspace-size query for Hermitian eigendecomposition (Jacobi). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_cheevj_buffer_size as "cusolverDnCheevj_bufferSize": PFN_cusolverDnCheevj_bufferSize;
    /// cuSOLVER: double-precision complex workspace-size query for Hermitian eigendecomposition (Jacobi). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_zheevj_buffer_size as "cusolverDnZheevj_bufferSize": PFN_cusolverDnZheevj_bufferSize;
    /// cuSOLVER: single-precision symmetric eigendecomposition (Jacobi). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_ssyevj as "cusolverDnSsyevj": PFN_cusolverDnSsyevj;
    /// cuSOLVER: double-precision symmetric eigendecomposition (Jacobi). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_dsyevj as "cusolverDnDsyevj": PFN_cusolverDnDsyevj;
    /// cuSOLVER: single-precision complex Hermitian eigendecomposition (Jacobi). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_cheevj as "cusolverDnCheevj": PFN_cusolverDnCheevj;
    /// cuSOLVER: double-precision complex Hermitian eigendecomposition (Jacobi). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_zheevj as "cusolverDnZheevj": PFN_cusolverDnZheevj;
    // Jacobi SVD
    /// cuSOLVER: create a Jacobi-SVD control object. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_create_gesvdj_info as "cusolverDnCreateGesvdjInfo": PFN_cusolverDnCreateGesvdjInfo;
    /// cuSOLVER: destroy a Jacobi-SVD control object. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_destroy_gesvdj_info as "cusolverDnDestroyGesvdjInfo": PFN_cusolverDnDestroyGesvdjInfo;
    /// cuSOLVER: single-precision workspace-size query for Jacobi-method singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_sgesvdj_buffer_size as "cusolverDnSgesvdj_bufferSize": PFN_cusolverDnSgesvdj_bufferSize;
    /// cuSOLVER: double-precision workspace-size query for Jacobi-method singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_dgesvdj_buffer_size as "cusolverDnDgesvdj_bufferSize": PFN_cusolverDnDgesvdj_bufferSize;
    /// cuSOLVER: single-precision complex workspace-size query for Jacobi-method singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_cgesvdj_buffer_size as "cusolverDnCgesvdj_bufferSize": PFN_cusolverDnCgesvdj_bufferSize;
    /// cuSOLVER: double-precision complex workspace-size query for Jacobi-method singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_zgesvdj_buffer_size as "cusolverDnZgesvdj_bufferSize": PFN_cusolverDnZgesvdj_bufferSize;
    /// cuSOLVER: single-precision Jacobi-method singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_sgesvdj as "cusolverDnSgesvdj": PFN_cusolverDnSgesvdj;
    /// cuSOLVER: double-precision Jacobi-method singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_dgesvdj as "cusolverDnDgesvdj": PFN_cusolverDnDgesvdj;
    /// cuSOLVER: single-precision complex Jacobi-method singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_cgesvdj as "cusolverDnCgesvdj": PFN_cusolverDnCgesvdj;
    /// cuSOLVER: double-precision complex Jacobi-method singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_zgesvdj as "cusolverDnZgesvdj": PFN_cusolverDnZgesvdj;
    // orgqr / ormqr (apply/generate Q from QR)
    /// cuSOLVER: single-precision workspace-size query for generate the explicit Q from a QR factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_sorgqr_buffer_size as "cusolverDnSorgqr_bufferSize": PFN_cusolverDnSorgqr_bufferSize;
    /// cuSOLVER: double-precision workspace-size query for generate the explicit Q from a QR factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_dorgqr_buffer_size as "cusolverDnDorgqr_bufferSize": PFN_cusolverDnDorgqr_bufferSize;
    /// cuSOLVER: single-precision complex workspace-size query for generate the explicit unitary Q from a QR factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_cungqr_buffer_size as "cusolverDnCungqr_bufferSize": PFN_cusolverDnCungqr_bufferSize;
    /// cuSOLVER: double-precision complex workspace-size query for generate the explicit unitary Q from a QR factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_zungqr_buffer_size as "cusolverDnZungqr_bufferSize": PFN_cusolverDnZungqr_bufferSize;
    /// cuSOLVER: single-precision generate the explicit Q from a QR factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_sorgqr as "cusolverDnSorgqr": PFN_cusolverDnSorgqr;
    /// cuSOLVER: double-precision generate the explicit Q from a QR factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_dorgqr as "cusolverDnDorgqr": PFN_cusolverDnDorgqr;
    /// cuSOLVER: single-precision complex generate the explicit unitary Q from a QR factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_cungqr as "cusolverDnCungqr": PFN_cusolverDnCungqr;
    /// cuSOLVER: double-precision complex generate the explicit unitary Q from a QR factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_zungqr as "cusolverDnZungqr": PFN_cusolverDnZungqr;
    /// cuSOLVER: single-precision workspace-size query for apply Q (or Q^T) from a QR factorization to a matrix. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_sormqr_buffer_size as "cusolverDnSormqr_bufferSize": PFN_cusolverDnSormqr_bufferSize;
    /// cuSOLVER: double-precision workspace-size query for apply Q (or Q^T) from a QR factorization to a matrix. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_dormqr_buffer_size as "cusolverDnDormqr_bufferSize": PFN_cusolverDnDormqr_bufferSize;
    /// cuSOLVER: single-precision complex workspace-size query for apply Q (or Q^H) from a QR factorization to a matrix. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_cunmqr_buffer_size as "cusolverDnCunmqr_bufferSize": PFN_cusolverDnCunmqr_bufferSize;
    /// cuSOLVER: double-precision complex workspace-size query for apply Q (or Q^H) from a QR factorization to a matrix. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_zunmqr_buffer_size as "cusolverDnZunmqr_bufferSize": PFN_cusolverDnZunmqr_bufferSize;
    /// cuSOLVER: single-precision apply Q (or Q^T) from a QR factorization to a matrix. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_sormqr as "cusolverDnSormqr": PFN_cusolverDnSormqr;
    /// cuSOLVER: double-precision apply Q (or Q^T) from a QR factorization to a matrix. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_dormqr as "cusolverDnDormqr": PFN_cusolverDnDormqr;
    /// cuSOLVER: single-precision complex apply Q (or Q^H) from a QR factorization to a matrix. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_cunmqr as "cusolverDnCunmqr": PFN_cusolverDnCunmqr;
    /// cuSOLVER: double-precision complex apply Q (or Q^H) from a QR factorization to a matrix. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_zunmqr as "cusolverDnZunmqr": PFN_cusolverDnZunmqr;
    // Sparse
    /// cuSOLVER: create a sparse cuSOLVER handle. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_sp_create as "cusolverSpCreate": PFN_cusolverSpCreate;
    /// cuSOLVER: destroy a sparse cuSOLVER handle. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_sp_destroy as "cusolverSpDestroy": PFN_cusolverSpDestroy;
    /// cuSOLVER: bind a CUDA stream to a sparse cuSOLVER handle. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_sp_set_stream as "cusolverSpSetStream": PFN_cusolverSpSetStream;
    /// cuSOLVER: single-precision sparse linear solve via Cholesky on a CSR matrix. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_sp_scsrlsvchol as "cusolverSpScsrlsvchol": PFN_cusolverSpScsrlsvchol;
    /// cuSOLVER: double-precision sparse linear solve via Cholesky on a CSR matrix. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_sp_dcsrlsvchol as "cusolverSpDcsrlsvchol": PFN_cusolverSpDcsrlsvchol;
    /// cuSOLVER: single-precision sparse linear solve via QR on a CSR matrix. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_sp_scsrlsvqr as "cusolverSpScsrlsvqr": PFN_cusolverSpScsrlsvqr;
    /// cuSOLVER: double-precision sparse linear solve via QR on a CSR matrix. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_sp_dcsrlsvqr as "cusolverSpDcsrlsvqr": PFN_cusolverSpDcsrlsvqr;
    // Refactor
    /// cuSOLVER: create a sparse-refactor cuSOLVER handle. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_rf_create as "cusolverRfCreate": PFN_cusolverRfCreate;
    /// cuSOLVER: destroy a sparse-refactor cuSOLVER handle. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_rf_destroy as "cusolverRfDestroy": PFN_cusolverRfDestroy;
    /// cuSOLVER: supply sparse triangular factors and pivot vectors to the refactor handle. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_rf_setup_device as "cusolverRfSetupDevice": PFN_cusolverRfSetupDevice;
    /// cuSOLVER: analyze the sparsity pattern of the refactor handle. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_rf_analyze as "cusolverRfAnalyze": PFN_cusolverRfAnalyze;
    /// cuSOLVER: refactor with new numerical values reusing the analyzed pattern. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_rf_refactor as "cusolverRfRefactor": PFN_cusolverRfRefactor;
    /// cuSOLVER: solve linear systems using the refactor handle. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_rf_solve as "cusolverRfSolve": PFN_cusolverRfSolve;
    // Least squares (gels) S/D/C/Z
    /// cuSOLVER: single-precision workspace-size query for least-squares solver (A*X = B). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_ssgels_buffer_size as "cusolverDnSSgels_bufferSize": PFN_cusolverDnSSgels_bufferSize;
    /// cuSOLVER: double-precision workspace-size query for least-squares solver (A*X = B). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_ddgels_buffer_size as "cusolverDnDDgels_bufferSize": PFN_cusolverDnDDgels_bufferSize;
    /// cuSOLVER: single-precision complex workspace-size query for least-squares solver (A*X = B). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_ccgels_buffer_size as "cusolverDnCCgels_bufferSize": PFN_cusolverDnCCgels_bufferSize;
    /// cuSOLVER: double-precision complex workspace-size query for least-squares solver (A*X = B). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_zzgels_buffer_size as "cusolverDnZZgels_bufferSize": PFN_cusolverDnZZgels_bufferSize;
    /// cuSOLVER: single-precision least-squares solver (A*X = B). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_ssgels as "cusolverDnSSgels": PFN_cusolverDnSSgels;
    /// cuSOLVER: double-precision least-squares solver (A*X = B). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_ddgels as "cusolverDnDDgels": PFN_cusolverDnDDgels;
    /// cuSOLVER: single-precision complex least-squares solver (A*X = B). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_ccgels as "cusolverDnCCgels": PFN_cusolverDnCCgels;
    /// cuSOLVER: double-precision complex least-squares solver (A*X = B). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_zzgels as "cusolverDnZZgels": PFN_cusolverDnZZgels;
    // potri (inverse from Cholesky) S/D/C/Z
    /// cuSOLVER: single-precision workspace-size query for matrix inverse from Cholesky factors. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_spotri_buffer_size as "cusolverDnSpotri_bufferSize": PFN_cusolverDnSpotri_bufferSize;
    /// cuSOLVER: double-precision workspace-size query for matrix inverse from Cholesky factors. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_dpotri_buffer_size as "cusolverDnDpotri_bufferSize": PFN_cusolverDnDpotri_bufferSize;
    /// cuSOLVER: single-precision complex workspace-size query for matrix inverse from Cholesky factors. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_cpotri_buffer_size as "cusolverDnCpotri_bufferSize": PFN_cusolverDnCpotri_bufferSize;
    /// cuSOLVER: double-precision complex workspace-size query for matrix inverse from Cholesky factors. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_zpotri_buffer_size as "cusolverDnZpotri_bufferSize": PFN_cusolverDnZpotri_bufferSize;
    /// cuSOLVER: single-precision matrix inverse from Cholesky factors. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_spotri as "cusolverDnSpotri": PFN_cusolverDnSpotri;
    /// cuSOLVER: double-precision matrix inverse from Cholesky factors. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_dpotri as "cusolverDnDpotri": PFN_cusolverDnDpotri;
    /// cuSOLVER: single-precision complex matrix inverse from Cholesky factors. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_cpotri as "cusolverDnCpotri": PFN_cusolverDnCpotri;
    /// cuSOLVER: double-precision complex matrix inverse from Cholesky factors. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_zpotri as "cusolverDnZpotri": PFN_cusolverDnZpotri;
    // Batched Jacobi eigen
    /// cuSOLVER: single-precision workspace-size query for batched symmetric eigendecomposition (Jacobi). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_ssyevj_batched_buffer_size as "cusolverDnSsyevjBatched_bufferSize": PFN_cusolverDnSsyevjBatched_bufferSize;
    /// cuSOLVER: double-precision workspace-size query for batched symmetric eigendecomposition (Jacobi). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_dsyevj_batched_buffer_size as "cusolverDnDsyevjBatched_bufferSize": PFN_cusolverDnDsyevjBatched_bufferSize;
    /// cuSOLVER: single-precision complex workspace-size query for batched Hermitian eigendecomposition (Jacobi). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_cheevj_batched_buffer_size as "cusolverDnCheevjBatched_bufferSize": PFN_cusolverDnCheevjBatched_bufferSize;
    /// cuSOLVER: double-precision complex workspace-size query for batched Hermitian eigendecomposition (Jacobi). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_zheevj_batched_buffer_size as "cusolverDnZheevjBatched_bufferSize": PFN_cusolverDnZheevjBatched_bufferSize;
    /// cuSOLVER: single-precision batched symmetric eigendecomposition (Jacobi). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_ssyevj_batched as "cusolverDnSsyevjBatched": PFN_cusolverDnSsyevjBatched;
    /// cuSOLVER: double-precision batched symmetric eigendecomposition (Jacobi). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_dsyevj_batched as "cusolverDnDsyevjBatched": PFN_cusolverDnDsyevjBatched;
    /// cuSOLVER: single-precision complex batched Hermitian eigendecomposition (Jacobi). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_cheevj_batched as "cusolverDnCheevjBatched": PFN_cusolverDnCheevjBatched;
    /// cuSOLVER: double-precision complex batched Hermitian eigendecomposition (Jacobi). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_zheevj_batched as "cusolverDnZheevjBatched": PFN_cusolverDnZheevjBatched;
    // Batched Jacobi SVD
    /// cuSOLVER: single-precision workspace-size query for batched Jacobi-method singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_sgesvdj_batched_buffer_size as "cusolverDnSgesvdjBatched_bufferSize": PFN_cusolverDnSgesvdjBatched_bufferSize;
    /// cuSOLVER: double-precision workspace-size query for batched Jacobi-method singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_dgesvdj_batched_buffer_size as "cusolverDnDgesvdjBatched_bufferSize": PFN_cusolverDnDgesvdjBatched_bufferSize;
    /// cuSOLVER: single-precision complex workspace-size query for batched Jacobi-method singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_cgesvdj_batched_buffer_size as "cusolverDnCgesvdjBatched_bufferSize": PFN_cusolverDnCgesvdjBatched_bufferSize;
    /// cuSOLVER: double-precision complex workspace-size query for batched Jacobi-method singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_zgesvdj_batched_buffer_size as "cusolverDnZgesvdjBatched_bufferSize": PFN_cusolverDnZgesvdjBatched_bufferSize;
    /// cuSOLVER: single-precision batched Jacobi-method singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_sgesvdj_batched as "cusolverDnSgesvdjBatched": PFN_cusolverDnSgesvdjBatched;
    /// cuSOLVER: double-precision batched Jacobi-method singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_dgesvdj_batched as "cusolverDnDgesvdjBatched": PFN_cusolverDnDgesvdjBatched;
    /// cuSOLVER: single-precision complex batched Jacobi-method singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_cgesvdj_batched as "cusolverDnCgesvdjBatched": PFN_cusolverDnCgesvdjBatched;
    /// cuSOLVER: double-precision complex batched Jacobi-method singular value decomposition. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_dn_zgesvdj_batched as "cusolverDnZgesvdjBatched": PFN_cusolverDnZgesvdjBatched;
}

/// Lazy-load the cuSOLVER dense / sparse / refactor shared library and return its function-pointer table.
pub fn cusolver() -> Result<&'static Cusolver, LoaderError> {
    static CUSOLVER: OnceLock<Cusolver> = OnceLock::new();
    if let Some(c) = CUSOLVER.get() {
        return Ok(c);
    }
    let candidates: Vec<&'static str> = cusolver_candidates()
        .into_iter()
        .map(|s| Box::leak(s.into_boxed_str()) as &'static str)
        .collect();
    let candidates_leaked: &'static [&'static str] = Box::leak(candidates.into_boxed_slice());
    let lib = Library::open("cusolver", candidates_leaked)?;
    let c = Cusolver::empty(lib);
    let _ = CUSOLVER.set(c);
    Ok(CUSOLVER.get().expect("OnceLock set or lost race"))
}

// ==========================================================================
// cuSOLVERMg — multi-GPU solver, ships in libcusolverMg (separate library)
// ==========================================================================

fn cusolver_mg_candidates() -> Vec<String> {
    platform::versioned_library_candidates("cusolverMg", &["13", "12", "11"])
}

macro_rules! cusolver_mg_fns {
    ($($(#[$m:meta])* $name:ident as $sym:literal : $pfn:ty);* $(;)?) => {
        /// Loaded cuSOLVERMg shared library plus a per-symbol `OnceLock` of function pointers.
        pub struct CusolverMg {
            lib: Library,
            $($name: OnceLock<$pfn>,)*
        }
        impl core::fmt::Debug for CusolverMg {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct("CusolverMg").field("lib", &self.lib).finish_non_exhaustive()
            }
        }
        impl CusolverMg {
            $(
                $(#[$m])*
                pub fn $name(&self) -> Result<$pfn, LoaderError> {
                    if let Some(&p) = self.$name.get() { return Ok(p); }
                    let raw: *mut () = unsafe { self.lib.raw_symbol($sym)? };
                    let p: $pfn = unsafe { core::mem::transmute_copy::<*mut (), $pfn>(&raw) };
                    let _ = self.$name.set(p);
                    Ok(p)
                }
            )*
            fn empty(lib: Library) -> Self {
                Self { lib, $($name: OnceLock::new(),)* }
            }
        }
    };
}

cusolver_mg_fns! {
    /// cuSOLVER: create a multi-GPU dense cuSOLVERMg handle. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_mg_create as "cusolverMgCreate": PFN_cusolverMgCreate;
    /// cuSOLVER: destroy a multi-GPU dense cuSOLVERMg handle. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_mg_destroy as "cusolverMgDestroy": PFN_cusolverMgDestroy;
    /// cuSOLVER: select GPU devices for a cuSOLVERMg handle. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_mg_device_select as "cusolverMgDeviceSelect": PFN_cusolverMgDeviceSelect;
    /// cuSOLVER: create a multi-GPU device grid. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_mg_create_device_grid as "cusolverMgCreateDeviceGrid": PFN_cusolverMgCreateDeviceGrid;
    /// cuSOLVER: destroy a multi-GPU device grid. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_mg_destroy_grid as "cusolverMgDestroyGrid": PFN_cusolverMgDestroyGrid;
    /// cuSOLVER: create a multi-GPU matrix descriptor. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_mg_create_matrix_desc as "cusolverMgCreateMatrixDesc": PFN_cusolverMgCreateMatrixDesc;
    /// cuSOLVER: destroy a multi-GPU matrix descriptor. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_mg_destroy_matrix_desc as "cusolverMgDestroyMatrixDesc": PFN_cusolverMgDestroyMatrixDesc;
    /// cuSOLVER: multi-GPU workspace-size query for LU factorization with partial pivoting. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_mg_getrf_buffer_size as "cusolverMgGetrf_bufferSize": PFN_cusolverMgGetrf_bufferSize;
    /// cuSOLVER: multi-GPU LU factorization with partial pivoting. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_mg_getrf as "cusolverMgGetrf": PFN_cusolverMgGetrf;
    /// cuSOLVER: multi-GPU workspace-size query for Cholesky factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_mg_potrf_buffer_size as "cusolverMgPotrf_bufferSize": PFN_cusolverMgPotrf_bufferSize;
    /// cuSOLVER: multi-GPU Cholesky factorization. See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_mg_potrf as "cusolverMgPotrf": PFN_cusolverMgPotrf;
    /// cuSOLVER: multi-GPU workspace-size query for symmetric eigendecomposition (divide-and-conquer). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_mg_syevd_buffer_size as "cusolverMgSyevd_bufferSize": PFN_cusolverMgSyevd_bufferSize;
    /// cuSOLVER: multi-GPU symmetric eigendecomposition (divide-and-conquer). See <https://docs.nvidia.com/cuda/cusolver/index.html>.
    cusolver_mg_syevd as "cusolverMgSyevd": PFN_cusolverMgSyevd;
}

/// Lazy-load the cuSOLVERMg multi-GPU shared library and return its function-pointer table.
pub fn cusolver_mg() -> Result<&'static CusolverMg, LoaderError> {
    static MG: OnceLock<CusolverMg> = OnceLock::new();
    if let Some(c) = MG.get() {
        return Ok(c);
    }
    let candidates: Vec<&'static str> = cusolver_mg_candidates()
        .into_iter()
        .map(|s| Box::leak(s.into_boxed_str()) as &'static str)
        .collect();
    let leaked: &'static [&'static str] = Box::leak(candidates.into_boxed_slice());
    let lib = Library::open("cusolverMg", leaked)?;
    let _ = MG.set(CusolverMg::empty(lib));
    Ok(MG.get().expect("OnceLock set or lost race"))
}
