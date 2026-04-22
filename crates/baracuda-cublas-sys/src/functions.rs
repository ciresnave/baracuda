//! cuBLAS function-pointer types.

#![allow(non_camel_case_types)]

use core::ffi::{c_int, c_void};

use baracuda_cuda_sys::runtime::cudaStream_t;

use super::status::cublasStatus_t;
use super::types::{cublasHandle_t, cublasMath_t, cublasOperation_t, cublasPointerMode_t};

// ---- context + version -----------------------------------------------------

pub type PFN_cublasCreate = unsafe extern "C" fn(handle: *mut cublasHandle_t) -> cublasStatus_t;
pub type PFN_cublasDestroy = unsafe extern "C" fn(handle: cublasHandle_t) -> cublasStatus_t;
pub type PFN_cublasGetVersion =
    unsafe extern "C" fn(handle: cublasHandle_t, version: *mut c_int) -> cublasStatus_t;
pub type PFN_cublasSetStream =
    unsafe extern "C" fn(handle: cublasHandle_t, stream: cudaStream_t) -> cublasStatus_t;
pub type PFN_cublasGetStream =
    unsafe extern "C" fn(handle: cublasHandle_t, stream: *mut cudaStream_t) -> cublasStatus_t;
pub type PFN_cublasSetPointerMode =
    unsafe extern "C" fn(handle: cublasHandle_t, mode: cublasPointerMode_t) -> cublasStatus_t;
pub type PFN_cublasSetMathMode =
    unsafe extern "C" fn(handle: cublasHandle_t, mode: cublasMath_t) -> cublasStatus_t;

// ---- BLAS-3 GEMM (single + double) -----------------------------------------

pub type PFN_cublasSgemm = unsafe extern "C" fn(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f32,
    a: *const f32,
    lda: c_int,
    b: *const f32,
    ldb: c_int,
    beta: *const f32,
    c: *mut f32,
    ldc: c_int,
) -> cublasStatus_t;

pub type PFN_cublasDgemm = unsafe extern "C" fn(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f64,
    a: *const f64,
    lda: c_int,
    b: *const f64,
    ldb: c_int,
    beta: *const f64,
    c: *mut f64,
    ldc: c_int,
) -> cublasStatus_t;

// ---- BLAS-3 strided-batched GEMM -------------------------------------------

pub type PFN_cublasSgemmStridedBatched = unsafe extern "C" fn(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f32,
    a: *const f32,
    lda: c_int,
    stride_a: i64,
    b: *const f32,
    ldb: c_int,
    stride_b: i64,
    beta: *const f32,
    c: *mut f32,
    ldc: c_int,
    stride_c: i64,
    batch_count: c_int,
) -> cublasStatus_t;

pub type PFN_cublasDgemmStridedBatched = unsafe extern "C" fn(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f64,
    a: *const f64,
    lda: c_int,
    stride_a: i64,
    b: *const f64,
    ldb: c_int,
    stride_b: i64,
    beta: *const f64,
    c: *mut f64,
    ldc: c_int,
    stride_c: i64,
    batch_count: c_int,
) -> cublasStatus_t;

// ---- BLAS-1 ax + y ---------------------------------------------------------

pub type PFN_cublasSaxpy = unsafe extern "C" fn(
    handle: cublasHandle_t,
    n: c_int,
    alpha: *const f32,
    x: *const f32,
    incx: c_int,
    y: *mut f32,
    incy: c_int,
) -> cublasStatus_t;

pub type PFN_cublasDaxpy = unsafe extern "C" fn(
    handle: cublasHandle_t,
    n: c_int,
    alpha: *const f64,
    x: *const f64,
    incx: c_int,
    y: *mut f64,
    incy: c_int,
) -> cublasStatus_t;

// ---- BLAS-1 dot ------------------------------------------------------------

pub type PFN_cublasSdot = unsafe extern "C" fn(
    handle: cublasHandle_t,
    n: c_int,
    x: *const f32,
    incx: c_int,
    y: *const f32,
    incy: c_int,
    result: *mut f32,
) -> cublasStatus_t;

pub type PFN_cublasDdot = unsafe extern "C" fn(
    handle: cublasHandle_t,
    n: c_int,
    x: *const f64,
    incx: c_int,
    y: *const f64,
    incy: c_int,
    result: *mut f64,
) -> cublasStatus_t;

// ---- L1 scalar ops --------------------------------------------------------

pub type PFN_cublasSscal = unsafe extern "C" fn(
    handle: cublasHandle_t,
    n: c_int,
    alpha: *const f32,
    x: *mut f32,
    incx: c_int,
) -> cublasStatus_t;

pub type PFN_cublasDscal = unsafe extern "C" fn(
    handle: cublasHandle_t,
    n: c_int,
    alpha: *const f64,
    x: *mut f64,
    incx: c_int,
) -> cublasStatus_t;

pub type PFN_cublasSnrm2 = unsafe extern "C" fn(
    handle: cublasHandle_t,
    n: c_int,
    x: *const f32,
    incx: c_int,
    result: *mut f32,
) -> cublasStatus_t;

pub type PFN_cublasDnrm2 = unsafe extern "C" fn(
    handle: cublasHandle_t,
    n: c_int,
    x: *const f64,
    incx: c_int,
    result: *mut f64,
) -> cublasStatus_t;

pub type PFN_cublasSasum = unsafe extern "C" fn(
    handle: cublasHandle_t,
    n: c_int,
    x: *const f32,
    incx: c_int,
    result: *mut f32,
) -> cublasStatus_t;

pub type PFN_cublasDasum = unsafe extern "C" fn(
    handle: cublasHandle_t,
    n: c_int,
    x: *const f64,
    incx: c_int,
    result: *mut f64,
) -> cublasStatus_t;

// Note: cuBLAS `i{s,d}ama{x,n}` return 1-based indices.
pub type PFN_cublasIsamax = unsafe extern "C" fn(
    handle: cublasHandle_t,
    n: c_int,
    x: *const f32,
    incx: c_int,
    result: *mut c_int,
) -> cublasStatus_t;

pub type PFN_cublasIdamax = unsafe extern "C" fn(
    handle: cublasHandle_t,
    n: c_int,
    x: *const f64,
    incx: c_int,
    result: *mut c_int,
) -> cublasStatus_t;

pub type PFN_cublasIsamin = unsafe extern "C" fn(
    handle: cublasHandle_t,
    n: c_int,
    x: *const f32,
    incx: c_int,
    result: *mut c_int,
) -> cublasStatus_t;

pub type PFN_cublasIdamin = unsafe extern "C" fn(
    handle: cublasHandle_t,
    n: c_int,
    x: *const f64,
    incx: c_int,
    result: *mut c_int,
) -> cublasStatus_t;

pub type PFN_cublasScopy = unsafe extern "C" fn(
    handle: cublasHandle_t,
    n: c_int,
    x: *const f32,
    incx: c_int,
    y: *mut f32,
    incy: c_int,
) -> cublasStatus_t;

pub type PFN_cublasDcopy = unsafe extern "C" fn(
    handle: cublasHandle_t,
    n: c_int,
    x: *const f64,
    incx: c_int,
    y: *mut f64,
    incy: c_int,
) -> cublasStatus_t;

// ---- L2 GEMV --------------------------------------------------------------

pub type PFN_cublasSgemv = unsafe extern "C" fn(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: c_int,
    n: c_int,
    alpha: *const f32,
    a: *const f32,
    lda: c_int,
    x: *const f32,
    incx: c_int,
    beta: *const f32,
    y: *mut f32,
    incy: c_int,
) -> cublasStatus_t;

pub type PFN_cublasDgemv = unsafe extern "C" fn(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: c_int,
    n: c_int,
    alpha: *const f64,
    a: *const f64,
    lda: c_int,
    x: *const f64,
    incx: c_int,
    beta: *const f64,
    y: *mut f64,
    incy: c_int,
) -> cublasStatus_t;

// Placeholder for library-specific opaque arguments passed as *void.
#[allow(dead_code)]
pub(crate) fn _placeholder_void() -> *mut c_void {
    core::ptr::null_mut()
}

// ==========================================================================
// Complex types (cuComplex / cuDoubleComplex)
// ==========================================================================

/// `cuComplex` — 2 × f32, ABI-compatible with `baracuda_types::Complex32`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct cuComplex {
    pub x: f32,
    pub y: f32,
}

/// `cuDoubleComplex` — 2 × f64.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct cuDoubleComplex {
    pub x: f64,
    pub y: f64,
}

/// Fill-mode / diagonal / side selectors used by symmetric/triangular ops.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cublasFillMode_t {
    Lower = 0,
    Upper = 1,
    Full = 2,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cublasDiagType_t {
    NonUnit = 0,
    Unit = 1,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cublasSideMode_t {
    Left = 0,
    Right = 1,
}

/// cuBLAS compute type (for Ex / generic GEMM).
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cublasComputeType_t {
    Compute16F = 64,
    Compute16FPedantic = 65,
    Compute32F = 68,
    Compute32FPedantic = 69,
    Compute32FFast16F = 74,
    Compute32FFast16BF = 75,
    Compute32FFastTF32 = 77,
    Compute64F = 70,
    Compute64FPedantic = 71,
    Compute32I = 72,
    Compute32IPedantic = 73,
}

/// Matches `cudaDataType_t` selector values needed by cuBLAS Ex APIs.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudaDataType_t {
    R_16F = 2,
    R_16BF = 14,
    R_32F = 0,
    R_64F = 1,
    R_8I = 3,
    R_8U = 8,
    R_32I = 10,
    R_32U = 12,
    C_16F = 6,
    C_16BF = 15,
    C_32F = 4,
    C_64F = 5,
    C_8I = 7,
    C_8U = 9,
    C_32I = 11,
    C_32U = 13,
}

// ==========================================================================
// L1 — full BLAS-level-1 for S/D/C/Z (complement the existing S/D entries)
// ==========================================================================

macro_rules! l1_axpy_like {
    ($name:ident, $scalar:ty) => {
        pub type $name = unsafe extern "C" fn(
            handle: cublasHandle_t,
            n: c_int,
            alpha: *const $scalar,
            x: *const $scalar,
            incx: c_int,
            y: *mut $scalar,
            incy: c_int,
        ) -> cublasStatus_t;
    };
}

macro_rules! l1_scal_like {
    ($name:ident, $scalar:ty) => {
        pub type $name = unsafe extern "C" fn(
            handle: cublasHandle_t,
            n: c_int,
            alpha: *const $scalar,
            x: *mut $scalar,
            incx: c_int,
        ) -> cublasStatus_t;
    };
}

macro_rules! l1_reduce_real {
    ($name:ident, $scalar:ty, $result:ty) => {
        pub type $name = unsafe extern "C" fn(
            handle: cublasHandle_t,
            n: c_int,
            x: *const $scalar,
            incx: c_int,
            result: *mut $result,
        ) -> cublasStatus_t;
    };
}

macro_rules! l1_dot_like {
    ($name:ident, $scalar:ty) => {
        pub type $name = unsafe extern "C" fn(
            handle: cublasHandle_t,
            n: c_int,
            x: *const $scalar,
            incx: c_int,
            y: *const $scalar,
            incy: c_int,
            result: *mut $scalar,
        ) -> cublasStatus_t;
    };
}

macro_rules! l1_swap_like {
    ($name:ident, $scalar:ty) => {
        pub type $name = unsafe extern "C" fn(
            handle: cublasHandle_t,
            n: c_int,
            x: *mut $scalar,
            incx: c_int,
            y: *mut $scalar,
            incy: c_int,
        ) -> cublasStatus_t;
    };
}

// AXPY: y = α·x + y — add C / Z variants.
l1_axpy_like!(PFN_cublasCaxpy, cuComplex);
l1_axpy_like!(PFN_cublasZaxpy, cuDoubleComplex);

// SCAL + CSSCAL/ZDSCAL (mixed-type scale).
l1_scal_like!(PFN_cublasCscal, cuComplex);
l1_scal_like!(PFN_cublasZscal, cuDoubleComplex);

pub type PFN_cublasCsscal = unsafe extern "C" fn(
    handle: cublasHandle_t,
    n: c_int,
    alpha: *const f32,
    x: *mut cuComplex,
    incx: c_int,
) -> cublasStatus_t;

pub type PFN_cublasZdscal = unsafe extern "C" fn(
    handle: cublasHandle_t,
    n: c_int,
    alpha: *const f64,
    x: *mut cuDoubleComplex,
    incx: c_int,
) -> cublasStatus_t;

// NRM2 / ASUM — reduce to a real.
l1_reduce_real!(PFN_cublasScnrm2, cuComplex, f32);
l1_reduce_real!(PFN_cublasDznrm2, cuDoubleComplex, f64);
l1_reduce_real!(PFN_cublasScasum, cuComplex, f32);
l1_reduce_real!(PFN_cublasDzasum, cuDoubleComplex, f64);

// DOT / DOTU / DOTC
l1_dot_like!(PFN_cublasCdotu, cuComplex);
l1_dot_like!(PFN_cublasCdotc, cuComplex);
l1_dot_like!(PFN_cublasZdotu, cuDoubleComplex);
l1_dot_like!(PFN_cublasZdotc, cuDoubleComplex);

// COPY / SWAP — C / Z
pub type PFN_cublasCcopy = unsafe extern "C" fn(
    handle: cublasHandle_t,
    n: c_int,
    x: *const cuComplex,
    incx: c_int,
    y: *mut cuComplex,
    incy: c_int,
) -> cublasStatus_t;

pub type PFN_cublasZcopy = unsafe extern "C" fn(
    handle: cublasHandle_t,
    n: c_int,
    x: *const cuDoubleComplex,
    incx: c_int,
    y: *mut cuDoubleComplex,
    incy: c_int,
) -> cublasStatus_t;

l1_swap_like!(PFN_cublasSswap, f32);
l1_swap_like!(PFN_cublasDswap, f64);
l1_swap_like!(PFN_cublasCswap, cuComplex);
l1_swap_like!(PFN_cublasZswap, cuDoubleComplex);

// IAMAX / IAMIN — add C / Z
pub type PFN_cublasIcamax = unsafe extern "C" fn(
    handle: cublasHandle_t,
    n: c_int,
    x: *const cuComplex,
    incx: c_int,
    result: *mut c_int,
) -> cublasStatus_t;

pub type PFN_cublasIzamax = unsafe extern "C" fn(
    handle: cublasHandle_t,
    n: c_int,
    x: *const cuDoubleComplex,
    incx: c_int,
    result: *mut c_int,
) -> cublasStatus_t;

pub type PFN_cublasIcamin = unsafe extern "C" fn(
    handle: cublasHandle_t,
    n: c_int,
    x: *const cuComplex,
    incx: c_int,
    result: *mut c_int,
) -> cublasStatus_t;

pub type PFN_cublasIzamin = unsafe extern "C" fn(
    handle: cublasHandle_t,
    n: c_int,
    x: *const cuDoubleComplex,
    incx: c_int,
    result: *mut c_int,
) -> cublasStatus_t;

// ROT / ROTG / ROTM
pub type PFN_cublasSrot = unsafe extern "C" fn(
    handle: cublasHandle_t,
    n: c_int,
    x: *mut f32,
    incx: c_int,
    y: *mut f32,
    incy: c_int,
    c: *const f32,
    s: *const f32,
) -> cublasStatus_t;

pub type PFN_cublasDrot = unsafe extern "C" fn(
    handle: cublasHandle_t,
    n: c_int,
    x: *mut f64,
    incx: c_int,
    y: *mut f64,
    incy: c_int,
    c: *const f64,
    s: *const f64,
) -> cublasStatus_t;

// ==========================================================================
// L2 — full GEMV/GBMV/SBMV/HBMV/SPMV/HPMV/TRMV/TRSV/SYR/HER + their variants
// (We cover GEMV/SYMV/TRMV/TRSV/GER/SYR × S/D/C/Z as the most-used core.)
// ==========================================================================

pub type PFN_cublasCgemv = unsafe extern "C" fn(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: c_int,
    n: c_int,
    alpha: *const cuComplex,
    a: *const cuComplex,
    lda: c_int,
    x: *const cuComplex,
    incx: c_int,
    beta: *const cuComplex,
    y: *mut cuComplex,
    incy: c_int,
) -> cublasStatus_t;

pub type PFN_cublasZgemv = unsafe extern "C" fn(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: c_int,
    n: c_int,
    alpha: *const cuDoubleComplex,
    a: *const cuDoubleComplex,
    lda: c_int,
    x: *const cuDoubleComplex,
    incx: c_int,
    beta: *const cuDoubleComplex,
    y: *mut cuDoubleComplex,
    incy: c_int,
) -> cublasStatus_t;

// Symmetric matrix-vector (SYMV) — S / D; complex uses HEMV (below).
pub type PFN_cublasSsymv = unsafe extern "C" fn(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    alpha: *const f32,
    a: *const f32,
    lda: c_int,
    x: *const f32,
    incx: c_int,
    beta: *const f32,
    y: *mut f32,
    incy: c_int,
) -> cublasStatus_t;

pub type PFN_cublasDsymv = unsafe extern "C" fn(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    alpha: *const f64,
    a: *const f64,
    lda: c_int,
    x: *const f64,
    incx: c_int,
    beta: *const f64,
    y: *mut f64,
    incy: c_int,
) -> cublasStatus_t;

// Triangular matrix-vector (TRMV / TRSV) — S / D
pub type PFN_cublasStrmv = unsafe extern "C" fn(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: c_int,
    a: *const f32,
    lda: c_int,
    x: *mut f32,
    incx: c_int,
) -> cublasStatus_t;

pub type PFN_cublasDtrmv = unsafe extern "C" fn(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: c_int,
    a: *const f64,
    lda: c_int,
    x: *mut f64,
    incx: c_int,
) -> cublasStatus_t;

pub type PFN_cublasStrsv = unsafe extern "C" fn(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: c_int,
    a: *const f32,
    lda: c_int,
    x: *mut f32,
    incx: c_int,
) -> cublasStatus_t;

pub type PFN_cublasDtrsv = unsafe extern "C" fn(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: c_int,
    a: *const f64,
    lda: c_int,
    x: *mut f64,
    incx: c_int,
) -> cublasStatus_t;

// GER — rank-1 update: A += α·x·yᵀ
pub type PFN_cublasSger = unsafe extern "C" fn(
    handle: cublasHandle_t,
    m: c_int,
    n: c_int,
    alpha: *const f32,
    x: *const f32,
    incx: c_int,
    y: *const f32,
    incy: c_int,
    a: *mut f32,
    lda: c_int,
) -> cublasStatus_t;

pub type PFN_cublasDger = unsafe extern "C" fn(
    handle: cublasHandle_t,
    m: c_int,
    n: c_int,
    alpha: *const f64,
    x: *const f64,
    incx: c_int,
    y: *const f64,
    incy: c_int,
    a: *mut f64,
    lda: c_int,
) -> cublasStatus_t;

// SYR — symmetric rank-1 update
pub type PFN_cublasSsyr = unsafe extern "C" fn(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    alpha: *const f32,
    x: *const f32,
    incx: c_int,
    a: *mut f32,
    lda: c_int,
) -> cublasStatus_t;

pub type PFN_cublasDsyr = unsafe extern "C" fn(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    alpha: *const f64,
    x: *const f64,
    incx: c_int,
    a: *mut f64,
    lda: c_int,
) -> cublasStatus_t;

// ==========================================================================
// L3 — GEMM/SYMM/HEMM/SYRK/HERK/SYR2K/HER2K/TRMM/TRSM × S/D/C/Z
// ==========================================================================

// GEMM C / Z (S / D already exist in the crate)
pub type PFN_cublasCgemm = unsafe extern "C" fn(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const cuComplex,
    a: *const cuComplex,
    lda: c_int,
    b: *const cuComplex,
    ldb: c_int,
    beta: *const cuComplex,
    c: *mut cuComplex,
    ldc: c_int,
) -> cublasStatus_t;

pub type PFN_cublasZgemm = unsafe extern "C" fn(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const cuDoubleComplex,
    a: *const cuDoubleComplex,
    lda: c_int,
    b: *const cuDoubleComplex,
    ldb: c_int,
    beta: *const cuDoubleComplex,
    c: *mut cuDoubleComplex,
    ldc: c_int,
) -> cublasStatus_t;

// cublasGemmEx — the generic mixed-precision path.
pub type PFN_cublasGemmEx = unsafe extern "C" fn(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_void,
    a: *const c_void,
    atype: cudaDataType_t,
    lda: c_int,
    b: *const c_void,
    btype: cudaDataType_t,
    ldb: c_int,
    beta: *const c_void,
    c: *mut c_void,
    ctype: cudaDataType_t,
    ldc: c_int,
    compute_type: cublasComputeType_t,
    algo: c_int,
) -> cublasStatus_t;

pub type PFN_cublasGemmStridedBatchedEx = unsafe extern "C" fn(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_void,
    a: *const c_void,
    atype: cudaDataType_t,
    lda: c_int,
    stride_a: i64,
    b: *const c_void,
    btype: cudaDataType_t,
    ldb: c_int,
    stride_b: i64,
    beta: *const c_void,
    c: *mut c_void,
    ctype: cudaDataType_t,
    ldc: c_int,
    stride_c: i64,
    batch_count: c_int,
    compute_type: cublasComputeType_t,
    algo: c_int,
) -> cublasStatus_t;

// SYMM / HEMM / SYRK / HERK / SYR2K / HER2K — single- and double-precision real + Hermitian complex
macro_rules! l3_triangular {
    ($name:ident, $scalar:ty) => {
        pub type $name = unsafe extern "C" fn(
            handle: cublasHandle_t,
            side: cublasSideMode_t,
            uplo: cublasFillMode_t,
            m: c_int,
            n: c_int,
            alpha: *const $scalar,
            a: *const $scalar,
            lda: c_int,
            b: *const $scalar,
            ldb: c_int,
            beta: *const $scalar,
            c: *mut $scalar,
            ldc: c_int,
        ) -> cublasStatus_t;
    };
}

l3_triangular!(PFN_cublasSsymm, f32);
l3_triangular!(PFN_cublasDsymm, f64);
l3_triangular!(PFN_cublasCsymm, cuComplex);
l3_triangular!(PFN_cublasZsymm, cuDoubleComplex);
l3_triangular!(PFN_cublasChemm, cuComplex);
l3_triangular!(PFN_cublasZhemm, cuDoubleComplex);

macro_rules! l3_syrk {
    ($name:ident, $scalar:ty) => {
        pub type $name = unsafe extern "C" fn(
            handle: cublasHandle_t,
            uplo: cublasFillMode_t,
            trans: cublasOperation_t,
            n: c_int,
            k: c_int,
            alpha: *const $scalar,
            a: *const $scalar,
            lda: c_int,
            beta: *const $scalar,
            c: *mut $scalar,
            ldc: c_int,
        ) -> cublasStatus_t;
    };
}

l3_syrk!(PFN_cublasSsyrk, f32);
l3_syrk!(PFN_cublasDsyrk, f64);
l3_syrk!(PFN_cublasCsyrk, cuComplex);
l3_syrk!(PFN_cublasZsyrk, cuDoubleComplex);

// HERK scalars are real even though the matrix is complex.
pub type PFN_cublasCherk = unsafe extern "C" fn(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: c_int,
    k: c_int,
    alpha: *const f32,
    a: *const cuComplex,
    lda: c_int,
    beta: *const f32,
    c: *mut cuComplex,
    ldc: c_int,
) -> cublasStatus_t;

pub type PFN_cublasZherk = unsafe extern "C" fn(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: c_int,
    k: c_int,
    alpha: *const f64,
    a: *const cuDoubleComplex,
    lda: c_int,
    beta: *const f64,
    c: *mut cuDoubleComplex,
    ldc: c_int,
) -> cublasStatus_t;

macro_rules! l3_trmm_trsm {
    ($name:ident, $scalar:ty) => {
        pub type $name = unsafe extern "C" fn(
            handle: cublasHandle_t,
            side: cublasSideMode_t,
            uplo: cublasFillMode_t,
            trans: cublasOperation_t,
            diag: cublasDiagType_t,
            m: c_int,
            n: c_int,
            alpha: *const $scalar,
            a: *const $scalar,
            lda: c_int,
            b: *mut $scalar,
            ldb: c_int,
        ) -> cublasStatus_t;
    };
}

l3_trmm_trsm!(PFN_cublasStrsm, f32);
l3_trmm_trsm!(PFN_cublasDtrsm, f64);
l3_trmm_trsm!(PFN_cublasCtrsm, cuComplex);
l3_trmm_trsm!(PFN_cublasZtrsm, cuDoubleComplex);

// TRMM_v2: separate input/output operands — A triangular, B read, C written.
macro_rules! l3_trmm_v2 {
    ($name:ident, $scalar:ty) => {
        pub type $name = unsafe extern "C" fn(
            handle: cublasHandle_t,
            side: cublasSideMode_t,
            uplo: cublasFillMode_t,
            trans: cublasOperation_t,
            diag: cublasDiagType_t,
            m: c_int,
            n: c_int,
            alpha: *const $scalar,
            a: *const $scalar,
            lda: c_int,
            b: *const $scalar,
            ldb: c_int,
            c: *mut $scalar,
            ldc: c_int,
        ) -> cublasStatus_t;
    };
}

l3_trmm_v2!(PFN_cublasStrmm, f32);
l3_trmm_v2!(PFN_cublasDtrmm, f64);
l3_trmm_v2!(PFN_cublasCtrmm, cuComplex);
l3_trmm_v2!(PFN_cublasZtrmm, cuDoubleComplex);

// ==========================================================================
// Batched GEMM
// ==========================================================================

pub type PFN_cublasSgemmBatched = unsafe extern "C" fn(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f32,
    a_array: *const *const f32,
    lda: c_int,
    b_array: *const *const f32,
    ldb: c_int,
    beta: *const f32,
    c_array: *mut *mut f32,
    ldc: c_int,
    batch_count: c_int,
) -> cublasStatus_t;

pub type PFN_cublasDgemmBatched = unsafe extern "C" fn(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f64,
    a_array: *const *const f64,
    lda: c_int,
    b_array: *const *const f64,
    ldb: c_int,
    beta: *const f64,
    c_array: *mut *mut f64,
    ldc: c_int,
    batch_count: c_int,
) -> cublasStatus_t;

pub type PFN_cublasCgemmBatched = unsafe extern "C" fn(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const cuComplex,
    a_array: *const *const cuComplex,
    lda: c_int,
    b_array: *const *const cuComplex,
    ldb: c_int,
    beta: *const cuComplex,
    c_array: *mut *mut cuComplex,
    ldc: c_int,
    batch_count: c_int,
) -> cublasStatus_t;

pub type PFN_cublasZgemmBatched = unsafe extern "C" fn(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const cuDoubleComplex,
    a_array: *const *const cuDoubleComplex,
    lda: c_int,
    b_array: *const *const cuDoubleComplex,
    ldb: c_int,
    beta: *const cuDoubleComplex,
    c_array: *mut *mut cuDoubleComplex,
    ldc: c_int,
    batch_count: c_int,
) -> cublasStatus_t;

pub type PFN_cublasCgemmStridedBatched = unsafe extern "C" fn(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const cuComplex,
    a: *const cuComplex,
    lda: c_int,
    stride_a: i64,
    b: *const cuComplex,
    ldb: c_int,
    stride_b: i64,
    beta: *const cuComplex,
    c: *mut cuComplex,
    ldc: c_int,
    stride_c: i64,
    batch_count: c_int,
) -> cublasStatus_t;

pub type PFN_cublasZgemmStridedBatched = unsafe extern "C" fn(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const cuDoubleComplex,
    a: *const cuDoubleComplex,
    lda: c_int,
    stride_a: i64,
    b: *const cuDoubleComplex,
    ldb: c_int,
    stride_b: i64,
    beta: *const cuDoubleComplex,
    c: *mut cuDoubleComplex,
    ldc: c_int,
    stride_c: i64,
    batch_count: c_int,
) -> cublasStatus_t;

// ==========================================================================
// Mixed-precision Ex variants for BLAS-1
// ==========================================================================

pub type PFN_cublasAxpyEx = unsafe extern "C" fn(
    handle: cublasHandle_t,
    n: c_int,
    alpha: *const c_void,
    alpha_type: cudaDataType_t,
    x: *const c_void,
    x_type: cudaDataType_t,
    incx: c_int,
    y: *mut c_void,
    y_type: cudaDataType_t,
    incy: c_int,
    execution_type: cudaDataType_t,
) -> cublasStatus_t;

pub type PFN_cublasDotEx = unsafe extern "C" fn(
    handle: cublasHandle_t,
    n: c_int,
    x: *const c_void,
    x_type: cudaDataType_t,
    incx: c_int,
    y: *const c_void,
    y_type: cudaDataType_t,
    incy: c_int,
    result: *mut c_void,
    result_type: cudaDataType_t,
    execution_type: cudaDataType_t,
) -> cublasStatus_t;

pub type PFN_cublasDotcEx = PFN_cublasDotEx;

pub type PFN_cublasNrm2Ex = unsafe extern "C" fn(
    handle: cublasHandle_t,
    n: c_int,
    x: *const c_void,
    x_type: cudaDataType_t,
    incx: c_int,
    result: *mut c_void,
    result_type: cudaDataType_t,
    execution_type: cudaDataType_t,
) -> cublasStatus_t;

pub type PFN_cublasScalEx = unsafe extern "C" fn(
    handle: cublasHandle_t,
    n: c_int,
    alpha: *const c_void,
    alpha_type: cudaDataType_t,
    x: *mut c_void,
    x_type: cudaDataType_t,
    incx: c_int,
    execution_type: cudaDataType_t,
) -> cublasStatus_t;

pub type PFN_cublasRotEx = unsafe extern "C" fn(
    handle: cublasHandle_t,
    n: c_int,
    x: *mut c_void,
    x_type: cudaDataType_t,
    incx: c_int,
    y: *mut c_void,
    y_type: cudaDataType_t,
    incy: c_int,
    c: *const c_void,
    s: *const c_void,
    cs_type: cudaDataType_t,
    execution_type: cudaDataType_t,
) -> cublasStatus_t;

// ==========================================================================
// Batched direct solvers (getrf/getri/matinv)
// ==========================================================================

macro_rules! batched_getrf {
    ($name:ident, $scalar:ty) => {
        pub type $name = unsafe extern "C" fn(
            handle: cublasHandle_t,
            n: c_int,
            a_array: *const *mut $scalar,
            lda: c_int,
            pivot_array: *mut c_int,
            info_array: *mut c_int,
            batch_size: c_int,
        ) -> cublasStatus_t;
    };
}
batched_getrf!(PFN_cublasSgetrfBatched, f32);
batched_getrf!(PFN_cublasDgetrfBatched, f64);
batched_getrf!(PFN_cublasCgetrfBatched, cuComplex);
batched_getrf!(PFN_cublasZgetrfBatched, cuDoubleComplex);

macro_rules! batched_getri {
    ($name:ident, $scalar:ty) => {
        pub type $name = unsafe extern "C" fn(
            handle: cublasHandle_t,
            n: c_int,
            a_array: *const *const $scalar,
            lda: c_int,
            pivot_array: *const c_int,
            c_array: *const *mut $scalar,
            ldc: c_int,
            info_array: *mut c_int,
            batch_size: c_int,
        ) -> cublasStatus_t;
    };
}
batched_getri!(PFN_cublasSgetriBatched, f32);
batched_getri!(PFN_cublasDgetriBatched, f64);
batched_getri!(PFN_cublasCgetriBatched, cuComplex);
batched_getri!(PFN_cublasZgetriBatched, cuDoubleComplex);

macro_rules! batched_matinv {
    ($name:ident, $scalar:ty) => {
        pub type $name = unsafe extern "C" fn(
            handle: cublasHandle_t,
            n: c_int,
            a_array: *const *const $scalar,
            lda: c_int,
            a_inv_array: *const *mut $scalar,
            lda_inv: c_int,
            info_array: *mut c_int,
            batch_size: c_int,
        ) -> cublasStatus_t;
    };
}
batched_matinv!(PFN_cublasSmatinvBatched, f32);
batched_matinv!(PFN_cublasDmatinvBatched, f64);
batched_matinv!(PFN_cublasCmatinvBatched, cuComplex);
batched_matinv!(PFN_cublasZmatinvBatched, cuDoubleComplex);

/// Batched getrs (solve after batched getrf).
macro_rules! batched_getrs {
    ($name:ident, $scalar:ty) => {
        pub type $name = unsafe extern "C" fn(
            handle: cublasHandle_t,
            trans: cublasOperation_t,
            n: c_int,
            nrhs: c_int,
            a_array: *const *const $scalar,
            lda: c_int,
            pivot_array: *const c_int,
            b_array: *const *mut $scalar,
            ldb: c_int,
            info_array: *mut c_int,
            batch_size: c_int,
        ) -> cublasStatus_t;
    };
}
batched_getrs!(PFN_cublasSgetrsBatched, f32);
batched_getrs!(PFN_cublasDgetrsBatched, f64);
batched_getrs!(PFN_cublasCgetrsBatched, cuComplex);
batched_getrs!(PFN_cublasZgetrsBatched, cuDoubleComplex);

// ==========================================================================
// cuBLASXt — multi-GPU GEMM-family routines
// ==========================================================================

pub type cublasXtHandle_t = *mut c_void;

pub type PFN_cublasXtCreate =
    unsafe extern "C" fn(handle: *mut cublasXtHandle_t) -> cublasStatus_t;
pub type PFN_cublasXtDestroy =
    unsafe extern "C" fn(handle: cublasXtHandle_t) -> cublasStatus_t;
pub type PFN_cublasXtDeviceSelect = unsafe extern "C" fn(
    handle: cublasXtHandle_t,
    n_devices: c_int,
    device_id: *const c_int,
) -> cublasStatus_t;
pub type PFN_cublasXtSetBlockDim = unsafe extern "C" fn(
    handle: cublasXtHandle_t,
    block_dim: c_int,
) -> cublasStatus_t;
pub type PFN_cublasXtGetBlockDim = unsafe extern "C" fn(
    handle: cublasXtHandle_t,
    block_dim: *mut c_int,
) -> cublasStatus_t;

macro_rules! xt_gemm {
    ($name:ident, $scalar:ty) => {
        pub type $name = unsafe extern "C" fn(
            handle: cublasXtHandle_t,
            transa: cublasOperation_t,
            transb: cublasOperation_t,
            m: usize,
            n: usize,
            k: usize,
            alpha: *const $scalar,
            a: *const $scalar,
            lda: usize,
            b: *const $scalar,
            ldb: usize,
            beta: *const $scalar,
            c: *mut $scalar,
            ldc: usize,
        ) -> cublasStatus_t;
    };
}
xt_gemm!(PFN_cublasXtSgemm, f32);
xt_gemm!(PFN_cublasXtDgemm, f64);
xt_gemm!(PFN_cublasXtCgemm, cuComplex);
xt_gemm!(PFN_cublasXtZgemm, cuDoubleComplex);

// ==========================================================================
// cuBLASLt — modern matmul with explicit layout/preference descriptors
// ==========================================================================

pub type cublasLtHandle_t = *mut c_void;
pub type cublasLtMatmulDesc_t = *mut c_void;
pub type cublasLtMatrixLayout_t = *mut c_void;
pub type cublasLtMatmulPreference_t = *mut c_void;

/// Opaque algorithm descriptor. 8 u64 words of cuBLASLt-internal state.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct cublasLtMatmulAlgo_t {
    pub data: [u64; 8],
}

/// One row of [`PFN_cublasLtMatmulAlgoGetHeuristic`]'s output table.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct cublasLtMatmulHeuristicResult_t {
    pub algo: cublasLtMatmulAlgo_t,
    pub workspace_size: usize,
    pub state: cublasStatus_t,
    pub waves_count: f32,
    pub reserved: [c_int; 4],
}

/// Attribute IDs on a [`cublasLtMatmulDesc_t`]. This is a common subset
/// useful for everyday matmul work; the full enum has ~30 entries.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cublasLtMatmulDescAttributes_t {
    ComputeType = 0,
    ScaleType = 1,
    PointerMode = 2,
    Transa = 3,
    Transb = 4,
    TransC = 5,
    FillMode = 6,
    Epilogue = 7,
    BiasPointer = 8,
    BiasDataType = 9,
    EpilogueAuxPointer = 10,
    EpilogueAuxLd = 11,
    EpilogueAuxBatchStride = 12,
    AlphaVectorBatchStride = 13,
    SmCountTarget = 14,
    AMax_Output_Pointer = 15,
}

/// Attribute IDs on a [`cublasLtMatrixLayout_t`].
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cublasLtMatrixLayoutAttribute_t {
    Type = 0,
    Order = 1,
    Rows = 2,
    Cols = 3,
    Ld = 4,
    BatchCount = 5,
    StridedBatchOffset = 6,
    PlaneOffset = 7,
}

/// Attribute IDs on a [`cublasLtMatmulPreference_t`].
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cublasLtMatmulPreferenceAttributes_t {
    SearchMode = 0,
    MaxWorkspaceBytes = 1,
    ReductionScheme = 2,
    MinAlignmentABytes = 3,
    MinAlignmentBBytes = 4,
    MinAlignmentCBytes = 5,
    MinAlignmentDBytes = 6,
    MaxWavesCount = 7,
    ImplMask = 8,
    PointerModeMask = 9,
    EpilogueMask = 10,
    SmCountTarget = 11,
}

pub type PFN_cublasLtCreate =
    unsafe extern "C" fn(handle: *mut cublasLtHandle_t) -> cublasStatus_t;

pub type PFN_cublasLtDestroy = unsafe extern "C" fn(handle: cublasLtHandle_t) -> cublasStatus_t;

pub type PFN_cublasLtMatmulDescCreate = unsafe extern "C" fn(
    desc_out: *mut cublasLtMatmulDesc_t,
    compute_type: cublasComputeType_t,
    scale_type: cudaDataType_t,
) -> cublasStatus_t;

pub type PFN_cublasLtMatmulDescDestroy =
    unsafe extern "C" fn(desc: cublasLtMatmulDesc_t) -> cublasStatus_t;

pub type PFN_cublasLtMatmulDescSetAttribute = unsafe extern "C" fn(
    desc: cublasLtMatmulDesc_t,
    attr: cublasLtMatmulDescAttributes_t,
    buf: *const c_void,
    size: usize,
) -> cublasStatus_t;

pub type PFN_cublasLtMatmulDescGetAttribute = unsafe extern "C" fn(
    desc: cublasLtMatmulDesc_t,
    attr: cublasLtMatmulDescAttributes_t,
    buf: *mut c_void,
    size_in: usize,
    size_out: *mut usize,
) -> cublasStatus_t;

pub type PFN_cublasLtMatrixLayoutCreate = unsafe extern "C" fn(
    layout_out: *mut cublasLtMatrixLayout_t,
    ty: cudaDataType_t,
    rows: u64,
    cols: u64,
    ld: i64,
) -> cublasStatus_t;

pub type PFN_cublasLtMatrixLayoutDestroy =
    unsafe extern "C" fn(layout: cublasLtMatrixLayout_t) -> cublasStatus_t;

pub type PFN_cublasLtMatrixLayoutSetAttribute = unsafe extern "C" fn(
    layout: cublasLtMatrixLayout_t,
    attr: cublasLtMatrixLayoutAttribute_t,
    buf: *const c_void,
    size: usize,
) -> cublasStatus_t;

pub type PFN_cublasLtMatmulPreferenceCreate =
    unsafe extern "C" fn(pref_out: *mut cublasLtMatmulPreference_t) -> cublasStatus_t;

pub type PFN_cublasLtMatmulPreferenceDestroy =
    unsafe extern "C" fn(pref: cublasLtMatmulPreference_t) -> cublasStatus_t;

pub type PFN_cublasLtMatmulPreferenceSetAttribute = unsafe extern "C" fn(
    pref: cublasLtMatmulPreference_t,
    attr: cublasLtMatmulPreferenceAttributes_t,
    buf: *const c_void,
    size: usize,
) -> cublasStatus_t;

pub type PFN_cublasLtMatmulAlgoGetHeuristic = unsafe extern "C" fn(
    lthandle: cublasLtHandle_t,
    op_desc: cublasLtMatmulDesc_t,
    a_desc: cublasLtMatrixLayout_t,
    b_desc: cublasLtMatrixLayout_t,
    c_desc: cublasLtMatrixLayout_t,
    d_desc: cublasLtMatrixLayout_t,
    pref: cublasLtMatmulPreference_t,
    requested_algo_count: c_int,
    heuristic_results: *mut cublasLtMatmulHeuristicResult_t,
    returned_algo_count: *mut c_int,
) -> cublasStatus_t;

pub type PFN_cublasLtMatmul = unsafe extern "C" fn(
    lthandle: cublasLtHandle_t,
    op_desc: cublasLtMatmulDesc_t,
    alpha: *const c_void,
    a: *const c_void,
    a_desc: cublasLtMatrixLayout_t,
    b: *const c_void,
    b_desc: cublasLtMatrixLayout_t,
    beta: *const c_void,
    c: *const c_void,
    c_desc: cublasLtMatrixLayout_t,
    d: *mut c_void,
    d_desc: cublasLtMatrixLayout_t,
    algo: *const cublasLtMatmulAlgo_t,
    workspace: *mut c_void,
    workspace_size: usize,
    stream: cudaStream_t,
) -> cublasStatus_t;

pub type PFN_cublasLtGetVersion = unsafe extern "C" fn() -> usize;
pub type PFN_cublasLtGetCudartVersion = unsafe extern "C" fn() -> usize;
