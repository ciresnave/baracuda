//! Raw FFI + dynamic loader for NVIDIA cuSOLVER (Dense + Sparse + Refactor).

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
#![warn(missing_debug_implementations)]

use core::ffi::{c_int, c_void};
use std::sync::OnceLock;

use baracuda_core::{platform, Library, LoaderError};
use baracuda_cuda_sys::runtime::cudaStream_t;
use baracuda_types::CudaStatus;

// ---- handles --------------------------------------------------------------

pub type cusolverDnHandle_t = *mut c_void;
pub type cusolverSpHandle_t = *mut c_void;
pub type cusolverRfHandle_t = *mut c_void;
pub type cusolverDnParams_t = *mut c_void;
pub type cusolverDnIRSParams_t = *mut c_void;
pub type cusolverDnIRSInfos_t = *mut c_void;
pub type syevjInfo_t = *mut c_void;
pub type gesvdjInfo_t = *mut c_void;

// ---- enums ----------------------------------------------------------------

/// Transpose selector — same values as `cublasOperation_t` (N=0, T=1, C=2).
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cublasOperation_t {
    N = 0,
    T = 1,
    C = 2,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cublasFillMode_t {
    Lower = 0,
    Upper = 1,
    Full = 2,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cublasSideMode_t {
    Left = 0,
    Right = 1,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cublasDiagType_t {
    NonUnit = 0,
    Unit = 1,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusolverEigType_t {
    Type1 = 1,
    Type2 = 2,
    Type3 = 3,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusolverEigMode_t {
    NoVector = 0,
    Vector = 1,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusolverEigRange_t {
    All = 1001,
    I = 1002,
    V = 1003,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudaDataType {
    R_32F = 0,
    R_64F = 1,
    R_16F = 2,
    C_32F = 4,
    C_64F = 5,
    R_16BF = 14,
}

// ---- status ---------------------------------------------------------------

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct cusolverStatus_t(pub i32);

impl cusolverStatus_t {
    pub const SUCCESS: Self = Self(0);
    pub const NOT_INITIALIZED: Self = Self(1);
    pub const ALLOC_FAILED: Self = Self(2);
    pub const INVALID_VALUE: Self = Self(3);
    pub const ARCH_MISMATCH: Self = Self(4);
    pub const EXECUTION_FAILED: Self = Self(6);
    pub const INTERNAL_ERROR: Self = Self(7);
    pub const NOT_SUPPORTED: Self = Self(9);
    pub const ZERO_PIVOT: Self = Self(10);

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

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct cuComplex {
    pub x: f32,
    pub y: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct cuDoubleComplex {
    pub x: f64,
    pub y: f64,
}

// ---- PFN type declaration macros -----------------------------------------

/// `getrf_bufferSize(handle, m, n, a, lda, lwork) -> status`
macro_rules! dn_getrf_bufsize {
    ($name:ident, $t:ty) => {
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
    ($name:ident, $t:ty) => {
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
    ($name:ident, $t:ty) => {
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
    ($name:ident, $t:ty) => {
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
    ($name:ident, $t:ty) => {
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
    ($name:ident, $t:ty) => {
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
    ($name:ident, $t:ty) => {
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
    ($name:ident, $t:ty) => {
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
    ($name:ident) => {
        pub type $name = unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: c_int,
            n: c_int,
            lwork: *mut c_int,
        ) -> cusolverStatus_t;
    };
}

macro_rules! dn_gesvd_real {
    ($name:ident, $t:ty) => {
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
    ($name:ident, $t:ty, $real:ty) => {
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
    ($name:ident, $t:ty, $real:ty) => {
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
    ($name:ident, $t:ty, $real:ty) => {
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

pub type PFN_cusolverDnCreate =
    unsafe extern "C" fn(handle: *mut cusolverDnHandle_t) -> cusolverStatus_t;
pub type PFN_cusolverDnDestroy =
    unsafe extern "C" fn(handle: cusolverDnHandle_t) -> cusolverStatus_t;
pub type PFN_cusolverDnSetStream =
    unsafe extern "C" fn(handle: cusolverDnHandle_t, stream: cudaStream_t) -> cusolverStatus_t;
pub type PFN_cusolverDnGetStream =
    unsafe extern "C" fn(handle: cusolverDnHandle_t, stream: *mut cudaStream_t) -> cusolverStatus_t;

pub type PFN_cusolverGetVersion = unsafe extern "C" fn(version: *mut c_int) -> cusolverStatus_t;

// ---- LU factorization (getrf / getrs) — S/D/C/Z --------------------------

dn_getrf_bufsize!(PFN_cusolverDnSgetrf_bufferSize, f32);
dn_getrf_bufsize!(PFN_cusolverDnDgetrf_bufferSize, f64);
dn_getrf_bufsize!(PFN_cusolverDnCgetrf_bufferSize, cuComplex);
dn_getrf_bufsize!(PFN_cusolverDnZgetrf_bufferSize, cuDoubleComplex);

dn_getrf!(PFN_cusolverDnSgetrf, f32);
dn_getrf!(PFN_cusolverDnDgetrf, f64);
dn_getrf!(PFN_cusolverDnCgetrf, cuComplex);
dn_getrf!(PFN_cusolverDnZgetrf, cuDoubleComplex);

dn_getrs!(PFN_cusolverDnSgetrs, f32);
dn_getrs!(PFN_cusolverDnDgetrs, f64);
dn_getrs!(PFN_cusolverDnCgetrs, cuComplex);
dn_getrs!(PFN_cusolverDnZgetrs, cuDoubleComplex);

// ---- QR factorization (geqrf) — S/D/C/Z ----------------------------------

dn_geqrf_bufsize!(PFN_cusolverDnSgeqrf_bufferSize, f32);
dn_geqrf_bufsize!(PFN_cusolverDnDgeqrf_bufferSize, f64);
dn_geqrf_bufsize!(PFN_cusolverDnCgeqrf_bufferSize, cuComplex);
dn_geqrf_bufsize!(PFN_cusolverDnZgeqrf_bufferSize, cuDoubleComplex);

dn_geqrf!(PFN_cusolverDnSgeqrf, f32);
dn_geqrf!(PFN_cusolverDnDgeqrf, f64);
dn_geqrf!(PFN_cusolverDnCgeqrf, cuComplex);
dn_geqrf!(PFN_cusolverDnZgeqrf, cuDoubleComplex);

// ---- Cholesky (potrf / potrs) — S/D/C/Z ----------------------------------

dn_potrf_bufsize!(PFN_cusolverDnSpotrf_bufferSize, f32);
dn_potrf_bufsize!(PFN_cusolverDnDpotrf_bufferSize, f64);
dn_potrf_bufsize!(PFN_cusolverDnCpotrf_bufferSize, cuComplex);
dn_potrf_bufsize!(PFN_cusolverDnZpotrf_bufferSize, cuDoubleComplex);

dn_potrf!(PFN_cusolverDnSpotrf, f32);
dn_potrf!(PFN_cusolverDnDpotrf, f64);
dn_potrf!(PFN_cusolverDnCpotrf, cuComplex);
dn_potrf!(PFN_cusolverDnZpotrf, cuDoubleComplex);

dn_potrs!(PFN_cusolverDnSpotrs, f32);
dn_potrs!(PFN_cusolverDnDpotrs, f64);
dn_potrs!(PFN_cusolverDnCpotrs, cuComplex);
dn_potrs!(PFN_cusolverDnZpotrs, cuDoubleComplex);

// ---- SVD — S/D/C/Z -------------------------------------------------------

dn_gesvd_bufsize!(PFN_cusolverDnSgesvd_bufferSize);
dn_gesvd_bufsize!(PFN_cusolverDnDgesvd_bufferSize);
dn_gesvd_bufsize!(PFN_cusolverDnCgesvd_bufferSize);
dn_gesvd_bufsize!(PFN_cusolverDnZgesvd_bufferSize);

dn_gesvd_real!(PFN_cusolverDnSgesvd, f32);
dn_gesvd_real!(PFN_cusolverDnDgesvd, f64);
dn_gesvd_complex!(PFN_cusolverDnCgesvd, cuComplex, f32);
dn_gesvd_complex!(PFN_cusolverDnZgesvd, cuDoubleComplex, f64);

// ---- Symmetric/Hermitian eigendecomposition (syevd/heevd) --------------

dn_syevd_bufsize!(PFN_cusolverDnSsyevd_bufferSize, f32, f32);
dn_syevd_bufsize!(PFN_cusolverDnDsyevd_bufferSize, f64, f64);
dn_syevd_bufsize!(PFN_cusolverDnCheevd_bufferSize, cuComplex, f32);
dn_syevd_bufsize!(PFN_cusolverDnZheevd_bufferSize, cuDoubleComplex, f64);

dn_syevd!(PFN_cusolverDnSsyevd, f32, f32);
dn_syevd!(PFN_cusolverDnDsyevd, f64, f64);
dn_syevd!(PFN_cusolverDnCheevd, cuComplex, f32);
dn_syevd!(PFN_cusolverDnZheevd, cuDoubleComplex, f64);

// ---- Generic 64-bit / mixed-precision API (cusolverDnX…) ----------------

pub type PFN_cusolverDnCreateParams =
    unsafe extern "C" fn(params: *mut cusolverDnParams_t) -> cusolverStatus_t;
pub type PFN_cusolverDnDestroyParams =
    unsafe extern "C" fn(params: cusolverDnParams_t) -> cusolverStatus_t;

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

pub type PFN_cusolverDnCreateSyevjInfo =
    unsafe extern "C" fn(info: *mut syevjInfo_t) -> cusolverStatus_t;
pub type PFN_cusolverDnDestroySyevjInfo =
    unsafe extern "C" fn(info: syevjInfo_t) -> cusolverStatus_t;
pub type PFN_cusolverDnXsyevjSetTolerance =
    unsafe extern "C" fn(info: syevjInfo_t, tolerance: f64) -> cusolverStatus_t;
pub type PFN_cusolverDnXsyevjSetMaxSweeps =
    unsafe extern "C" fn(info: syevjInfo_t, max_sweeps: c_int) -> cusolverStatus_t;

macro_rules! dn_syevj_bufsize {
    ($name:ident, $t:ty, $real:ty) => {
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
dn_syevj_bufsize!(PFN_cusolverDnSsyevj_bufferSize, f32, f32);
dn_syevj_bufsize!(PFN_cusolverDnDsyevj_bufferSize, f64, f64);
dn_syevj_bufsize!(PFN_cusolverDnCheevj_bufferSize, cuComplex, f32);
dn_syevj_bufsize!(PFN_cusolverDnZheevj_bufferSize, cuDoubleComplex, f64);

macro_rules! dn_syevj {
    ($name:ident, $t:ty, $real:ty) => {
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
dn_syevj!(PFN_cusolverDnSsyevj, f32, f32);
dn_syevj!(PFN_cusolverDnDsyevj, f64, f64);
dn_syevj!(PFN_cusolverDnCheevj, cuComplex, f32);
dn_syevj!(PFN_cusolverDnZheevj, cuDoubleComplex, f64);

pub type PFN_cusolverDnCreateGesvdjInfo =
    unsafe extern "C" fn(info: *mut gesvdjInfo_t) -> cusolverStatus_t;
pub type PFN_cusolverDnDestroyGesvdjInfo =
    unsafe extern "C" fn(info: gesvdjInfo_t) -> cusolverStatus_t;

macro_rules! dn_gesvdj_bufsize {
    ($name:ident, $t:ty, $real:ty) => {
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
dn_gesvdj_bufsize!(PFN_cusolverDnSgesvdj_bufferSize, f32, f32);
dn_gesvdj_bufsize!(PFN_cusolverDnDgesvdj_bufferSize, f64, f64);
dn_gesvdj_bufsize!(PFN_cusolverDnCgesvdj_bufferSize, cuComplex, f32);
dn_gesvdj_bufsize!(PFN_cusolverDnZgesvdj_bufferSize, cuDoubleComplex, f64);

macro_rules! dn_gesvdj {
    ($name:ident, $t:ty, $real:ty) => {
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
dn_gesvdj!(PFN_cusolverDnSgesvdj, f32, f32);
dn_gesvdj!(PFN_cusolverDnDgesvdj, f64, f64);
dn_gesvdj!(PFN_cusolverDnCgesvdj, cuComplex, f32);
dn_gesvdj!(PFN_cusolverDnZgesvdj, cuDoubleComplex, f64);

// ==========================================================================
// Apply Q from QR (orgqr / ormqr)
// ==========================================================================

macro_rules! dn_orgqr_bufsize {
    ($name:ident, $t:ty) => {
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
dn_orgqr_bufsize!(PFN_cusolverDnSorgqr_bufferSize, f32);
dn_orgqr_bufsize!(PFN_cusolverDnDorgqr_bufferSize, f64);
dn_orgqr_bufsize!(PFN_cusolverDnCungqr_bufferSize, cuComplex);
dn_orgqr_bufsize!(PFN_cusolverDnZungqr_bufferSize, cuDoubleComplex);

macro_rules! dn_orgqr {
    ($name:ident, $t:ty) => {
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
dn_orgqr!(PFN_cusolverDnSorgqr, f32);
dn_orgqr!(PFN_cusolverDnDorgqr, f64);
dn_orgqr!(PFN_cusolverDnCungqr, cuComplex);
dn_orgqr!(PFN_cusolverDnZungqr, cuDoubleComplex);

macro_rules! dn_ormqr_bufsize {
    ($name:ident, $t:ty) => {
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
dn_ormqr_bufsize!(PFN_cusolverDnSormqr_bufferSize, f32);
dn_ormqr_bufsize!(PFN_cusolverDnDormqr_bufferSize, f64);
dn_ormqr_bufsize!(PFN_cusolverDnCunmqr_bufferSize, cuComplex);
dn_ormqr_bufsize!(PFN_cusolverDnZunmqr_bufferSize, cuDoubleComplex);

macro_rules! dn_ormqr {
    ($name:ident, $t:ty) => {
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
dn_ormqr!(PFN_cusolverDnSormqr, f32);
dn_ormqr!(PFN_cusolverDnDormqr, f64);
dn_ormqr!(PFN_cusolverDnCunmqr, cuComplex);
dn_ormqr!(PFN_cusolverDnZunmqr, cuDoubleComplex);

// ---- Sparse cuSOLVER -----------------------------------------------------

pub type PFN_cusolverSpCreate =
    unsafe extern "C" fn(handle: *mut cusolverSpHandle_t) -> cusolverStatus_t;
pub type PFN_cusolverSpDestroy =
    unsafe extern "C" fn(handle: cusolverSpHandle_t) -> cusolverStatus_t;
pub type PFN_cusolverSpSetStream =
    unsafe extern "C" fn(handle: cusolverSpHandle_t, stream: cudaStream_t) -> cusolverStatus_t;

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

pub type PFN_cusolverRfCreate =
    unsafe extern "C" fn(handle: *mut cusolverRfHandle_t) -> cusolverStatus_t;
pub type PFN_cusolverRfDestroy =
    unsafe extern "C" fn(handle: cusolverRfHandle_t) -> cusolverStatus_t;
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
pub type PFN_cusolverRfAnalyze =
    unsafe extern "C" fn(handle: cusolverRfHandle_t) -> cusolverStatus_t;
pub type PFN_cusolverRfRefactor =
    unsafe extern "C" fn(handle: cusolverRfHandle_t) -> cusolverStatus_t;
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
    ($name:ident, $t:ty) => {
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
dn_gels_bufsize!(PFN_cusolverDnSSgels_bufferSize, f32);
dn_gels_bufsize!(PFN_cusolverDnDDgels_bufferSize, f64);
dn_gels_bufsize!(PFN_cusolverDnCCgels_bufferSize, cuComplex);
dn_gels_bufsize!(PFN_cusolverDnZZgels_bufferSize, cuDoubleComplex);

macro_rules! dn_gels {
    ($name:ident, $t:ty) => {
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
dn_gels!(PFN_cusolverDnSSgels, f32);
dn_gels!(PFN_cusolverDnDDgels, f64);
dn_gels!(PFN_cusolverDnCCgels, cuComplex);
dn_gels!(PFN_cusolverDnZZgels, cuDoubleComplex);

// ==========================================================================
// Inverse from Cholesky (potri)
// ==========================================================================

macro_rules! dn_potri_bufsize {
    ($name:ident, $t:ty) => {
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
dn_potri_bufsize!(PFN_cusolverDnSpotri_bufferSize, f32);
dn_potri_bufsize!(PFN_cusolverDnDpotri_bufferSize, f64);
dn_potri_bufsize!(PFN_cusolverDnCpotri_bufferSize, cuComplex);
dn_potri_bufsize!(PFN_cusolverDnZpotri_bufferSize, cuDoubleComplex);

macro_rules! dn_potri {
    ($name:ident, $t:ty) => {
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
dn_potri!(PFN_cusolverDnSpotri, f32);
dn_potri!(PFN_cusolverDnDpotri, f64);
dn_potri!(PFN_cusolverDnCpotri, cuComplex);
dn_potri!(PFN_cusolverDnZpotri, cuDoubleComplex);

// ==========================================================================
// Batched Jacobi eigen / SVD
// ==========================================================================

macro_rules! dn_syevj_batched_bufsize {
    ($name:ident, $t:ty, $real:ty) => {
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
dn_syevj_batched_bufsize!(PFN_cusolverDnSsyevjBatched_bufferSize, f32, f32);
dn_syevj_batched_bufsize!(PFN_cusolverDnDsyevjBatched_bufferSize, f64, f64);
dn_syevj_batched_bufsize!(PFN_cusolverDnCheevjBatched_bufferSize, cuComplex, f32);
dn_syevj_batched_bufsize!(PFN_cusolverDnZheevjBatched_bufferSize, cuDoubleComplex, f64);

macro_rules! dn_syevj_batched {
    ($name:ident, $t:ty, $real:ty) => {
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
dn_syevj_batched!(PFN_cusolverDnSsyevjBatched, f32, f32);
dn_syevj_batched!(PFN_cusolverDnDsyevjBatched, f64, f64);
dn_syevj_batched!(PFN_cusolverDnCheevjBatched, cuComplex, f32);
dn_syevj_batched!(PFN_cusolverDnZheevjBatched, cuDoubleComplex, f64);

macro_rules! dn_gesvdj_batched_bufsize {
    ($name:ident, $t:ty, $real:ty) => {
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
dn_gesvdj_batched_bufsize!(PFN_cusolverDnSgesvdjBatched_bufferSize, f32, f32);
dn_gesvdj_batched_bufsize!(PFN_cusolverDnDgesvdjBatched_bufferSize, f64, f64);
dn_gesvdj_batched_bufsize!(PFN_cusolverDnCgesvdjBatched_bufferSize, cuComplex, f32);
dn_gesvdj_batched_bufsize!(PFN_cusolverDnZgesvdjBatched_bufferSize, cuDoubleComplex, f64);

macro_rules! dn_gesvdj_batched {
    ($name:ident, $t:ty, $real:ty) => {
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
dn_gesvdj_batched!(PFN_cusolverDnSgesvdjBatched, f32, f32);
dn_gesvdj_batched!(PFN_cusolverDnDgesvdjBatched, f64, f64);
dn_gesvdj_batched!(PFN_cusolverDnCgesvdjBatched, cuComplex, f32);
dn_gesvdj_batched!(PFN_cusolverDnZgesvdjBatched, cuDoubleComplex, f64);

// ==========================================================================
// cuSOLVERMg — multi-GPU dense solvers (separate library libcusolverMg)
// ==========================================================================

pub type cusolverMgHandle_t = *mut c_void;
pub type cudaLibMgMatrixDesc_t = *mut c_void;
pub type cudaLibMgGrid_t = *mut c_void;

pub type PFN_cusolverMgCreate =
    unsafe extern "C" fn(handle: *mut cusolverMgHandle_t) -> cusolverStatus_t;
pub type PFN_cusolverMgDestroy =
    unsafe extern "C" fn(handle: cusolverMgHandle_t) -> cusolverStatus_t;
pub type PFN_cusolverMgDeviceSelect = unsafe extern "C" fn(
    handle: cusolverMgHandle_t,
    n_devices: c_int,
    device_id: *const c_int,
) -> cusolverStatus_t;

pub type PFN_cusolverMgCreateDeviceGrid = unsafe extern "C" fn(
    grid: *mut cudaLibMgGrid_t,
    num_row_devices: i32,
    num_col_devices: i32,
    device_id: *const i32,
    mapping: i32,
) -> cusolverStatus_t;

pub type PFN_cusolverMgDestroyGrid =
    unsafe extern "C" fn(grid: cudaLibMgGrid_t) -> cusolverStatus_t;

pub type PFN_cusolverMgCreateMatrixDesc = unsafe extern "C" fn(
    desc: *mut cudaLibMgMatrixDesc_t,
    num_rows: i64,
    num_cols: i64,
    row_block_size: i64,
    col_block_size: i64,
    data_type: cudaDataType,
    grid: cudaLibMgGrid_t,
) -> cusolverStatus_t;

pub type PFN_cusolverMgDestroyMatrixDesc =
    unsafe extern "C" fn(desc: cudaLibMgMatrixDesc_t) -> cusolverStatus_t;

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
    ($($name:ident as $sym:literal : $pfn:ty);* $(;)?) => {
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
    cusolver_dn_create as "cusolverDnCreate": PFN_cusolverDnCreate;
    cusolver_dn_destroy as "cusolverDnDestroy": PFN_cusolverDnDestroy;
    cusolver_dn_set_stream as "cusolverDnSetStream": PFN_cusolverDnSetStream;
    cusolver_dn_get_stream as "cusolverDnGetStream": PFN_cusolverDnGetStream;
    cusolver_get_version as "cusolverGetVersion": PFN_cusolverGetVersion;
    // LU (getrf/getrs) S/D/C/Z
    cusolver_dn_sgetrf_buffer_size as "cusolverDnSgetrf_bufferSize": PFN_cusolverDnSgetrf_bufferSize;
    cusolver_dn_dgetrf_buffer_size as "cusolverDnDgetrf_bufferSize": PFN_cusolverDnDgetrf_bufferSize;
    cusolver_dn_cgetrf_buffer_size as "cusolverDnCgetrf_bufferSize": PFN_cusolverDnCgetrf_bufferSize;
    cusolver_dn_zgetrf_buffer_size as "cusolverDnZgetrf_bufferSize": PFN_cusolverDnZgetrf_bufferSize;
    cusolver_dn_sgetrf as "cusolverDnSgetrf": PFN_cusolverDnSgetrf;
    cusolver_dn_dgetrf as "cusolverDnDgetrf": PFN_cusolverDnDgetrf;
    cusolver_dn_cgetrf as "cusolverDnCgetrf": PFN_cusolverDnCgetrf;
    cusolver_dn_zgetrf as "cusolverDnZgetrf": PFN_cusolverDnZgetrf;
    cusolver_dn_sgetrs as "cusolverDnSgetrs": PFN_cusolverDnSgetrs;
    cusolver_dn_dgetrs as "cusolverDnDgetrs": PFN_cusolverDnDgetrs;
    cusolver_dn_cgetrs as "cusolverDnCgetrs": PFN_cusolverDnCgetrs;
    cusolver_dn_zgetrs as "cusolverDnZgetrs": PFN_cusolverDnZgetrs;
    // QR (geqrf) S/D/C/Z
    cusolver_dn_sgeqrf_buffer_size as "cusolverDnSgeqrf_bufferSize": PFN_cusolverDnSgeqrf_bufferSize;
    cusolver_dn_dgeqrf_buffer_size as "cusolverDnDgeqrf_bufferSize": PFN_cusolverDnDgeqrf_bufferSize;
    cusolver_dn_cgeqrf_buffer_size as "cusolverDnCgeqrf_bufferSize": PFN_cusolverDnCgeqrf_bufferSize;
    cusolver_dn_zgeqrf_buffer_size as "cusolverDnZgeqrf_bufferSize": PFN_cusolverDnZgeqrf_bufferSize;
    cusolver_dn_sgeqrf as "cusolverDnSgeqrf": PFN_cusolverDnSgeqrf;
    cusolver_dn_dgeqrf as "cusolverDnDgeqrf": PFN_cusolverDnDgeqrf;
    cusolver_dn_cgeqrf as "cusolverDnCgeqrf": PFN_cusolverDnCgeqrf;
    cusolver_dn_zgeqrf as "cusolverDnZgeqrf": PFN_cusolverDnZgeqrf;
    // Cholesky (potrf/potrs) S/D/C/Z
    cusolver_dn_spotrf_buffer_size as "cusolverDnSpotrf_bufferSize": PFN_cusolverDnSpotrf_bufferSize;
    cusolver_dn_dpotrf_buffer_size as "cusolverDnDpotrf_bufferSize": PFN_cusolverDnDpotrf_bufferSize;
    cusolver_dn_cpotrf_buffer_size as "cusolverDnCpotrf_bufferSize": PFN_cusolverDnCpotrf_bufferSize;
    cusolver_dn_zpotrf_buffer_size as "cusolverDnZpotrf_bufferSize": PFN_cusolverDnZpotrf_bufferSize;
    cusolver_dn_spotrf as "cusolverDnSpotrf": PFN_cusolverDnSpotrf;
    cusolver_dn_dpotrf as "cusolverDnDpotrf": PFN_cusolverDnDpotrf;
    cusolver_dn_cpotrf as "cusolverDnCpotrf": PFN_cusolverDnCpotrf;
    cusolver_dn_zpotrf as "cusolverDnZpotrf": PFN_cusolverDnZpotrf;
    cusolver_dn_spotrs as "cusolverDnSpotrs": PFN_cusolverDnSpotrs;
    cusolver_dn_dpotrs as "cusolverDnDpotrs": PFN_cusolverDnDpotrs;
    cusolver_dn_cpotrs as "cusolverDnCpotrs": PFN_cusolverDnCpotrs;
    cusolver_dn_zpotrs as "cusolverDnZpotrs": PFN_cusolverDnZpotrs;
    // SVD S/D/C/Z
    cusolver_dn_sgesvd_buffer_size as "cusolverDnSgesvd_bufferSize": PFN_cusolverDnSgesvd_bufferSize;
    cusolver_dn_dgesvd_buffer_size as "cusolverDnDgesvd_bufferSize": PFN_cusolverDnDgesvd_bufferSize;
    cusolver_dn_cgesvd_buffer_size as "cusolverDnCgesvd_bufferSize": PFN_cusolverDnCgesvd_bufferSize;
    cusolver_dn_zgesvd_buffer_size as "cusolverDnZgesvd_bufferSize": PFN_cusolverDnZgesvd_bufferSize;
    cusolver_dn_sgesvd as "cusolverDnSgesvd": PFN_cusolverDnSgesvd;
    cusolver_dn_dgesvd as "cusolverDnDgesvd": PFN_cusolverDnDgesvd;
    cusolver_dn_cgesvd as "cusolverDnCgesvd": PFN_cusolverDnCgesvd;
    cusolver_dn_zgesvd as "cusolverDnZgesvd": PFN_cusolverDnZgesvd;
    // syevd / heevd
    cusolver_dn_ssyevd_buffer_size as "cusolverDnSsyevd_bufferSize": PFN_cusolverDnSsyevd_bufferSize;
    cusolver_dn_dsyevd_buffer_size as "cusolverDnDsyevd_bufferSize": PFN_cusolverDnDsyevd_bufferSize;
    cusolver_dn_cheevd_buffer_size as "cusolverDnCheevd_bufferSize": PFN_cusolverDnCheevd_bufferSize;
    cusolver_dn_zheevd_buffer_size as "cusolverDnZheevd_bufferSize": PFN_cusolverDnZheevd_bufferSize;
    cusolver_dn_ssyevd as "cusolverDnSsyevd": PFN_cusolverDnSsyevd;
    cusolver_dn_dsyevd as "cusolverDnDsyevd": PFN_cusolverDnDsyevd;
    cusolver_dn_cheevd as "cusolverDnCheevd": PFN_cusolverDnCheevd;
    cusolver_dn_zheevd as "cusolverDnZheevd": PFN_cusolverDnZheevd;
    // Generic 64-bit X… API
    cusolver_dn_create_params as "cusolverDnCreateParams": PFN_cusolverDnCreateParams;
    cusolver_dn_destroy_params as "cusolverDnDestroyParams": PFN_cusolverDnDestroyParams;
    cusolver_dn_xgetrf_buffer_size as "cusolverDnXgetrf_bufferSize": PFN_cusolverDnXgetrf_bufferSize;
    cusolver_dn_xgetrf as "cusolverDnXgetrf": PFN_cusolverDnXgetrf;
    cusolver_dn_xgetrs as "cusolverDnXgetrs": PFN_cusolverDnXgetrs;
    cusolver_dn_xgeqrf_buffer_size as "cusolverDnXgeqrf_bufferSize": PFN_cusolverDnXgeqrf_bufferSize;
    cusolver_dn_xgeqrf as "cusolverDnXgeqrf": PFN_cusolverDnXgeqrf;
    cusolver_dn_xpotrf_buffer_size as "cusolverDnXpotrf_bufferSize": PFN_cusolverDnXpotrf_bufferSize;
    cusolver_dn_xpotrf as "cusolverDnXpotrf": PFN_cusolverDnXpotrf;
    cusolver_dn_xpotrs as "cusolverDnXpotrs": PFN_cusolverDnXpotrs;
    cusolver_dn_xsyevd_buffer_size as "cusolverDnXsyevd_bufferSize": PFN_cusolverDnXsyevd_bufferSize;
    cusolver_dn_xsyevd as "cusolverDnXsyevd": PFN_cusolverDnXsyevd;
    // Jacobi eigen
    cusolver_dn_create_syevj_info as "cusolverDnCreateSyevjInfo": PFN_cusolverDnCreateSyevjInfo;
    cusolver_dn_destroy_syevj_info as "cusolverDnDestroySyevjInfo": PFN_cusolverDnDestroySyevjInfo;
    cusolver_dn_xsyevj_set_tolerance as "cusolverDnXsyevjSetTolerance": PFN_cusolverDnXsyevjSetTolerance;
    cusolver_dn_xsyevj_set_max_sweeps as "cusolverDnXsyevjSetMaxSweeps": PFN_cusolverDnXsyevjSetMaxSweeps;
    cusolver_dn_ssyevj_buffer_size as "cusolverDnSsyevj_bufferSize": PFN_cusolverDnSsyevj_bufferSize;
    cusolver_dn_dsyevj_buffer_size as "cusolverDnDsyevj_bufferSize": PFN_cusolverDnDsyevj_bufferSize;
    cusolver_dn_cheevj_buffer_size as "cusolverDnCheevj_bufferSize": PFN_cusolverDnCheevj_bufferSize;
    cusolver_dn_zheevj_buffer_size as "cusolverDnZheevj_bufferSize": PFN_cusolverDnZheevj_bufferSize;
    cusolver_dn_ssyevj as "cusolverDnSsyevj": PFN_cusolverDnSsyevj;
    cusolver_dn_dsyevj as "cusolverDnDsyevj": PFN_cusolverDnDsyevj;
    cusolver_dn_cheevj as "cusolverDnCheevj": PFN_cusolverDnCheevj;
    cusolver_dn_zheevj as "cusolverDnZheevj": PFN_cusolverDnZheevj;
    // Jacobi SVD
    cusolver_dn_create_gesvdj_info as "cusolverDnCreateGesvdjInfo": PFN_cusolverDnCreateGesvdjInfo;
    cusolver_dn_destroy_gesvdj_info as "cusolverDnDestroyGesvdjInfo": PFN_cusolverDnDestroyGesvdjInfo;
    cusolver_dn_sgesvdj_buffer_size as "cusolverDnSgesvdj_bufferSize": PFN_cusolverDnSgesvdj_bufferSize;
    cusolver_dn_dgesvdj_buffer_size as "cusolverDnDgesvdj_bufferSize": PFN_cusolverDnDgesvdj_bufferSize;
    cusolver_dn_cgesvdj_buffer_size as "cusolverDnCgesvdj_bufferSize": PFN_cusolverDnCgesvdj_bufferSize;
    cusolver_dn_zgesvdj_buffer_size as "cusolverDnZgesvdj_bufferSize": PFN_cusolverDnZgesvdj_bufferSize;
    cusolver_dn_sgesvdj as "cusolverDnSgesvdj": PFN_cusolverDnSgesvdj;
    cusolver_dn_dgesvdj as "cusolverDnDgesvdj": PFN_cusolverDnDgesvdj;
    cusolver_dn_cgesvdj as "cusolverDnCgesvdj": PFN_cusolverDnCgesvdj;
    cusolver_dn_zgesvdj as "cusolverDnZgesvdj": PFN_cusolverDnZgesvdj;
    // orgqr / ormqr (apply/generate Q from QR)
    cusolver_dn_sorgqr_buffer_size as "cusolverDnSorgqr_bufferSize": PFN_cusolverDnSorgqr_bufferSize;
    cusolver_dn_dorgqr_buffer_size as "cusolverDnDorgqr_bufferSize": PFN_cusolverDnDorgqr_bufferSize;
    cusolver_dn_cungqr_buffer_size as "cusolverDnCungqr_bufferSize": PFN_cusolverDnCungqr_bufferSize;
    cusolver_dn_zungqr_buffer_size as "cusolverDnZungqr_bufferSize": PFN_cusolverDnZungqr_bufferSize;
    cusolver_dn_sorgqr as "cusolverDnSorgqr": PFN_cusolverDnSorgqr;
    cusolver_dn_dorgqr as "cusolverDnDorgqr": PFN_cusolverDnDorgqr;
    cusolver_dn_cungqr as "cusolverDnCungqr": PFN_cusolverDnCungqr;
    cusolver_dn_zungqr as "cusolverDnZungqr": PFN_cusolverDnZungqr;
    cusolver_dn_sormqr_buffer_size as "cusolverDnSormqr_bufferSize": PFN_cusolverDnSormqr_bufferSize;
    cusolver_dn_dormqr_buffer_size as "cusolverDnDormqr_bufferSize": PFN_cusolverDnDormqr_bufferSize;
    cusolver_dn_cunmqr_buffer_size as "cusolverDnCunmqr_bufferSize": PFN_cusolverDnCunmqr_bufferSize;
    cusolver_dn_zunmqr_buffer_size as "cusolverDnZunmqr_bufferSize": PFN_cusolverDnZunmqr_bufferSize;
    cusolver_dn_sormqr as "cusolverDnSormqr": PFN_cusolverDnSormqr;
    cusolver_dn_dormqr as "cusolverDnDormqr": PFN_cusolverDnDormqr;
    cusolver_dn_cunmqr as "cusolverDnCunmqr": PFN_cusolverDnCunmqr;
    cusolver_dn_zunmqr as "cusolverDnZunmqr": PFN_cusolverDnZunmqr;
    // Sparse
    cusolver_sp_create as "cusolverSpCreate": PFN_cusolverSpCreate;
    cusolver_sp_destroy as "cusolverSpDestroy": PFN_cusolverSpDestroy;
    cusolver_sp_set_stream as "cusolverSpSetStream": PFN_cusolverSpSetStream;
    cusolver_sp_scsrlsvchol as "cusolverSpScsrlsvchol": PFN_cusolverSpScsrlsvchol;
    cusolver_sp_dcsrlsvchol as "cusolverSpDcsrlsvchol": PFN_cusolverSpDcsrlsvchol;
    cusolver_sp_scsrlsvqr as "cusolverSpScsrlsvqr": PFN_cusolverSpScsrlsvqr;
    cusolver_sp_dcsrlsvqr as "cusolverSpDcsrlsvqr": PFN_cusolverSpDcsrlsvqr;
    // Refactor
    cusolver_rf_create as "cusolverRfCreate": PFN_cusolverRfCreate;
    cusolver_rf_destroy as "cusolverRfDestroy": PFN_cusolverRfDestroy;
    cusolver_rf_setup_device as "cusolverRfSetupDevice": PFN_cusolverRfSetupDevice;
    cusolver_rf_analyze as "cusolverRfAnalyze": PFN_cusolverRfAnalyze;
    cusolver_rf_refactor as "cusolverRfRefactor": PFN_cusolverRfRefactor;
    cusolver_rf_solve as "cusolverRfSolve": PFN_cusolverRfSolve;
    // Least squares (gels) S/D/C/Z
    cusolver_dn_ssgels_buffer_size as "cusolverDnSSgels_bufferSize": PFN_cusolverDnSSgels_bufferSize;
    cusolver_dn_ddgels_buffer_size as "cusolverDnDDgels_bufferSize": PFN_cusolverDnDDgels_bufferSize;
    cusolver_dn_ccgels_buffer_size as "cusolverDnCCgels_bufferSize": PFN_cusolverDnCCgels_bufferSize;
    cusolver_dn_zzgels_buffer_size as "cusolverDnZZgels_bufferSize": PFN_cusolverDnZZgels_bufferSize;
    cusolver_dn_ssgels as "cusolverDnSSgels": PFN_cusolverDnSSgels;
    cusolver_dn_ddgels as "cusolverDnDDgels": PFN_cusolverDnDDgels;
    cusolver_dn_ccgels as "cusolverDnCCgels": PFN_cusolverDnCCgels;
    cusolver_dn_zzgels as "cusolverDnZZgels": PFN_cusolverDnZZgels;
    // potri (inverse from Cholesky) S/D/C/Z
    cusolver_dn_spotri_buffer_size as "cusolverDnSpotri_bufferSize": PFN_cusolverDnSpotri_bufferSize;
    cusolver_dn_dpotri_buffer_size as "cusolverDnDpotri_bufferSize": PFN_cusolverDnDpotri_bufferSize;
    cusolver_dn_cpotri_buffer_size as "cusolverDnCpotri_bufferSize": PFN_cusolverDnCpotri_bufferSize;
    cusolver_dn_zpotri_buffer_size as "cusolverDnZpotri_bufferSize": PFN_cusolverDnZpotri_bufferSize;
    cusolver_dn_spotri as "cusolverDnSpotri": PFN_cusolverDnSpotri;
    cusolver_dn_dpotri as "cusolverDnDpotri": PFN_cusolverDnDpotri;
    cusolver_dn_cpotri as "cusolverDnCpotri": PFN_cusolverDnCpotri;
    cusolver_dn_zpotri as "cusolverDnZpotri": PFN_cusolverDnZpotri;
    // Batched Jacobi eigen
    cusolver_dn_ssyevj_batched_buffer_size as "cusolverDnSsyevjBatched_bufferSize": PFN_cusolverDnSsyevjBatched_bufferSize;
    cusolver_dn_dsyevj_batched_buffer_size as "cusolverDnDsyevjBatched_bufferSize": PFN_cusolverDnDsyevjBatched_bufferSize;
    cusolver_dn_cheevj_batched_buffer_size as "cusolverDnCheevjBatched_bufferSize": PFN_cusolverDnCheevjBatched_bufferSize;
    cusolver_dn_zheevj_batched_buffer_size as "cusolverDnZheevjBatched_bufferSize": PFN_cusolverDnZheevjBatched_bufferSize;
    cusolver_dn_ssyevj_batched as "cusolverDnSsyevjBatched": PFN_cusolverDnSsyevjBatched;
    cusolver_dn_dsyevj_batched as "cusolverDnDsyevjBatched": PFN_cusolverDnDsyevjBatched;
    cusolver_dn_cheevj_batched as "cusolverDnCheevjBatched": PFN_cusolverDnCheevjBatched;
    cusolver_dn_zheevj_batched as "cusolverDnZheevjBatched": PFN_cusolverDnZheevjBatched;
    // Batched Jacobi SVD
    cusolver_dn_sgesvdj_batched_buffer_size as "cusolverDnSgesvdjBatched_bufferSize": PFN_cusolverDnSgesvdjBatched_bufferSize;
    cusolver_dn_dgesvdj_batched_buffer_size as "cusolverDnDgesvdjBatched_bufferSize": PFN_cusolverDnDgesvdjBatched_bufferSize;
    cusolver_dn_cgesvdj_batched_buffer_size as "cusolverDnCgesvdjBatched_bufferSize": PFN_cusolverDnCgesvdjBatched_bufferSize;
    cusolver_dn_zgesvdj_batched_buffer_size as "cusolverDnZgesvdjBatched_bufferSize": PFN_cusolverDnZgesvdjBatched_bufferSize;
    cusolver_dn_sgesvdj_batched as "cusolverDnSgesvdjBatched": PFN_cusolverDnSgesvdjBatched;
    cusolver_dn_dgesvdj_batched as "cusolverDnDgesvdjBatched": PFN_cusolverDnDgesvdjBatched;
    cusolver_dn_cgesvdj_batched as "cusolverDnCgesvdjBatched": PFN_cusolverDnCgesvdjBatched;
    cusolver_dn_zgesvdj_batched as "cusolverDnZgesvdjBatched": PFN_cusolverDnZgesvdjBatched;
}

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
    ($($name:ident as $sym:literal : $pfn:ty);* $(;)?) => {
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
    cusolver_mg_create as "cusolverMgCreate": PFN_cusolverMgCreate;
    cusolver_mg_destroy as "cusolverMgDestroy": PFN_cusolverMgDestroy;
    cusolver_mg_device_select as "cusolverMgDeviceSelect": PFN_cusolverMgDeviceSelect;
    cusolver_mg_create_device_grid as "cusolverMgCreateDeviceGrid": PFN_cusolverMgCreateDeviceGrid;
    cusolver_mg_destroy_grid as "cusolverMgDestroyGrid": PFN_cusolverMgDestroyGrid;
    cusolver_mg_create_matrix_desc as "cusolverMgCreateMatrixDesc": PFN_cusolverMgCreateMatrixDesc;
    cusolver_mg_destroy_matrix_desc as "cusolverMgDestroyMatrixDesc": PFN_cusolverMgDestroyMatrixDesc;
    cusolver_mg_getrf_buffer_size as "cusolverMgGetrf_bufferSize": PFN_cusolverMgGetrf_bufferSize;
    cusolver_mg_getrf as "cusolverMgGetrf": PFN_cusolverMgGetrf;
    cusolver_mg_potrf_buffer_size as "cusolverMgPotrf_bufferSize": PFN_cusolverMgPotrf_bufferSize;
    cusolver_mg_potrf as "cusolverMgPotrf": PFN_cusolverMgPotrf;
    cusolver_mg_syevd_buffer_size as "cusolverMgSyevd_bufferSize": PFN_cusolverMgSyevd_bufferSize;
    cusolver_mg_syevd as "cusolverMgSyevd": PFN_cusolverMgSyevd;
}

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
