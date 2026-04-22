//! Batched and mixed-precision GEMM.
//!
//! [`gemm_batched`] takes a host-side slice of *device pointers*, one per
//! matrix, and schedules a single batched call. [`gemm_ex`] is the
//! mixed-precision general GEMM: you pick element/compute/scale types
//! explicitly and cuBLAS dispatches to the best kernel for the hardware.

use core::ffi::c_void;

use baracuda_cublas_sys::functions::{
    cuComplex, cuDoubleComplex, cublasComputeType_t, cudaDataType_t,
};
use baracuda_cublas_sys::{cublas, cublasHandle_t, cublasOperation_t, cublasStatus_t};
use baracuda_types::{Complex32, Complex64, DeviceRepr};

use crate::blas_scalar::Op;
use crate::error::{check, Result};

/// Dispatch trait for `gemm_batched` (fixed, non-strided pointer arrays).
pub trait BatchedGemmScalar: DeviceRepr + batched_sealed::Sealed {
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm_batched_raw(
        h: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        a: *const *const Self,
        lda: i32,
        b: *const *const Self,
        ldb: i32,
        beta: &Self,
        c: *const *mut Self,
        ldc: i32,
        batch_count: i32,
    ) -> cublasStatus_t;
}

impl BatchedGemmScalar for f32 {
    unsafe fn gemm_batched_raw(
        h: cublasHandle_t,
        ta: cublasOperation_t,
        tb: cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: &f32,
        a: *const *const f32,
        lda: i32,
        b: *const *const f32,
        ldb: i32,
        beta: &f32,
        c: *const *mut f32,
        ldc: i32,
        batch: i32,
    ) -> cublasStatus_t {
        match cublas().and_then(|c| c.cublas_sgemm_batched()) {
            Ok(f) => f(
                h,
                ta,
                tb,
                m,
                n,
                k,
                alpha,
                a,
                lda,
                b,
                ldb,
                beta,
                c as *mut *mut f32,
                ldc,
                batch,
            ),
            Err(_) => cublasStatus_t::NOT_INITIALIZED,
        }
    }
}

impl BatchedGemmScalar for f64 {
    unsafe fn gemm_batched_raw(
        h: cublasHandle_t,
        ta: cublasOperation_t,
        tb: cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: &f64,
        a: *const *const f64,
        lda: i32,
        b: *const *const f64,
        ldb: i32,
        beta: &f64,
        c: *const *mut f64,
        ldc: i32,
        batch: i32,
    ) -> cublasStatus_t {
        match cublas().and_then(|c| c.cublas_dgemm_batched()) {
            Ok(f) => f(
                h,
                ta,
                tb,
                m,
                n,
                k,
                alpha,
                a,
                lda,
                b,
                ldb,
                beta,
                c as *mut *mut f64,
                ldc,
                batch,
            ),
            Err(_) => cublasStatus_t::NOT_INITIALIZED,
        }
    }
}

impl BatchedGemmScalar for Complex32 {
    unsafe fn gemm_batched_raw(
        h: cublasHandle_t,
        ta: cublasOperation_t,
        tb: cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Complex32,
        a: *const *const Complex32,
        lda: i32,
        b: *const *const Complex32,
        ldb: i32,
        beta: &Complex32,
        c: *const *mut Complex32,
        ldc: i32,
        batch: i32,
    ) -> cublasStatus_t {
        match cublas().and_then(|c| c.cublas_cgemm_batched()) {
            Ok(f) => f(
                h,
                ta,
                tb,
                m,
                n,
                k,
                alpha as *const _ as *const cuComplex,
                a as *const *const cuComplex,
                lda,
                b as *const *const cuComplex,
                ldb,
                beta as *const _ as *const cuComplex,
                c as *mut *mut cuComplex,
                ldc,
                batch,
            ),
            Err(_) => cublasStatus_t::NOT_INITIALIZED,
        }
    }
}

impl BatchedGemmScalar for Complex64 {
    unsafe fn gemm_batched_raw(
        h: cublasHandle_t,
        ta: cublasOperation_t,
        tb: cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Complex64,
        a: *const *const Complex64,
        lda: i32,
        b: *const *const Complex64,
        ldb: i32,
        beta: &Complex64,
        c: *const *mut Complex64,
        ldc: i32,
        batch: i32,
    ) -> cublasStatus_t {
        match cublas().and_then(|c| c.cublas_zgemm_batched()) {
            Ok(f) => f(
                h,
                ta,
                tb,
                m,
                n,
                k,
                alpha as *const _ as *const cuDoubleComplex,
                a as *const *const cuDoubleComplex,
                lda,
                b as *const *const cuDoubleComplex,
                ldb,
                beta as *const _ as *const cuDoubleComplex,
                c as *mut *mut cuDoubleComplex,
                ldc,
                batch,
            ),
            Err(_) => cublasStatus_t::NOT_INITIALIZED,
        }
    }
}

mod batched_sealed {
    use baracuda_types::{Complex32, Complex64};
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for Complex32 {}
    impl Sealed for Complex64 {}
}

/// Batched GEMM over arrays of device pointers. `a_ptrs` / `b_ptrs` /
/// `c_ptrs` must be device-resident arrays of pointers to the per-matrix
/// data (not host pointers).
///
/// # Safety
/// All three pointer arrays must be on-device arrays of at least
/// `batch_count` valid device pointers; each pointed-to matrix must have the
/// expected shape.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_batched<T: BatchedGemmScalar>(
    handle: &crate::Handle,
    transa: Op,
    transb: Op,
    m: i32,
    n: i32,
    k: i32,
    alpha: T,
    a_ptrs: *const *const T,
    lda: i32,
    b_ptrs: *const *const T,
    ldb: i32,
    beta: T,
    c_ptrs: *const *mut T,
    ldc: i32,
    batch_count: i32,
) -> Result<()> {
    let status = T::gemm_batched_raw(
        handle.as_raw(),
        transa.raw(),
        transb.raw(),
        m,
        n,
        k,
        &alpha,
        a_ptrs,
        lda,
        b_ptrs,
        ldb,
        &beta,
        c_ptrs,
        ldc,
        batch_count,
    );
    check(status)
}

/// Mixed-precision, type-erased GEMM (`cublasGemmEx`).
///
/// `a_type`, `b_type`, `c_type` are the per-operand [`cudaDataType_t`];
/// `compute_type` is the accumulator / arithmetic precision.
///
/// # Safety
/// `alpha`/`beta`/`a`/`b`/`c` must point to valid, correctly-typed buffers
/// (host for scalars in the default pointer mode, device for matrix data).
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_ex(
    handle: &crate::Handle,
    transa: Op,
    transb: Op,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const c_void,
    a: *const c_void,
    a_type: cudaDataType_t,
    lda: i32,
    b: *const c_void,
    b_type: cudaDataType_t,
    ldb: i32,
    beta: *const c_void,
    c: *mut c_void,
    c_type: cudaDataType_t,
    ldc: i32,
    compute_type: cublasComputeType_t,
    algo: i32,
) -> Result<()> {
    let c_api = cublas()?;
    let f = c_api.cublas_gemm_ex()?;
    check(f(
        handle.as_raw(),
        transa.raw(),
        transb.raw(),
        m,
        n,
        k,
        alpha,
        a,
        a_type,
        lda,
        b,
        b_type,
        ldb,
        beta,
        c,
        c_type,
        ldc,
        compute_type,
        algo,
    ))
}

/// Strided-batched mixed-precision GEMM (`cublasGemmStridedBatchedEx`).
///
/// # Safety
/// Same requirements as [`gemm_ex`] plus valid strides for all batch
/// dimensions.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_strided_batched_ex(
    handle: &crate::Handle,
    transa: Op,
    transb: Op,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const c_void,
    a: *const c_void,
    a_type: cudaDataType_t,
    lda: i32,
    stride_a: i64,
    b: *const c_void,
    b_type: cudaDataType_t,
    ldb: i32,
    stride_b: i64,
    beta: *const c_void,
    c: *mut c_void,
    c_type: cudaDataType_t,
    ldc: i32,
    stride_c: i64,
    batch_count: i32,
    compute_type: cublasComputeType_t,
    algo: i32,
) -> Result<()> {
    let c_api = cublas()?;
    let f = c_api.cublas_gemm_strided_batched_ex()?;
    check(f(
        handle.as_raw(),
        transa.raw(),
        transb.raw(),
        m,
        n,
        k,
        alpha,
        a,
        a_type,
        lda,
        stride_a,
        b,
        b_type,
        ldb,
        stride_b,
        beta,
        c,
        c_type,
        ldc,
        stride_c,
        batch_count,
        compute_type,
        algo,
    ))
}
