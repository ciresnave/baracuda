//! `BlasScalar` — the element-type trait that parametrises the generic GEMM /
//! AXPY / DOT wrappers over `f32`, `f64`, `Complex32`, `Complex64`.
//!
//! Half-precision (`f16` / `bf16`) GEMM is reachable via [`crate::gemm_ex`].

use baracuda_cublas_sys::functions::{cuComplex, cuDoubleComplex};
use baracuda_cublas_sys::{cublas, cublasHandle_t, cublasOperation_t, cublasStatus_t};
use baracuda_driver::{DeviceBuffer, DeviceSlice};
use baracuda_types::{Complex32, Complex64, DeviceRepr};

use crate::error::{check, Result};

/// A scalar type supported by this crate's generic BLAS wrappers.
///
/// Implemented for `f32` and `f64`. The trait is sealed at v0.1 to prevent
/// downstream impls while we work out the f16/bf16/complex stories.
pub trait BlasScalar: DeviceRepr + sealed::Sealed {
    /// Dispatch `cublas?gemm` for this element type.
    ///
    /// # Safety
    ///
    /// The pointers `a`/`b`/`c` must point to at least `m*k`, `k*n`, `m*n`
    /// elements of `Self` respectively, using column-major storage with
    /// leading dimensions `lda`/`ldb`/`ldc`.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm_raw(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        a: *const Self,
        lda: i32,
        b: *const Self,
        ldb: i32,
        beta: &Self,
        c: *mut Self,
        ldc: i32,
    ) -> cublasStatus_t;

    /// Dispatch `cublas?axpy` for this element type.
    #[doc(hidden)]
    unsafe fn axpy_raw(
        handle: cublasHandle_t,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        y: *mut Self,
        incy: i32,
    ) -> cublasStatus_t;

    /// Dispatch `cublas?gemmStridedBatched` for this element type.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm_strided_batched_raw(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        a: *const Self,
        lda: i32,
        stride_a: i64,
        b: *const Self,
        ldb: i32,
        stride_b: i64,
        beta: &Self,
        c: *mut Self,
        ldc: i32,
        stride_c: i64,
        batch_count: i32,
    ) -> cublasStatus_t;
}

impl BlasScalar for f32 {
    unsafe fn gemm_raw(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: &f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: &f32,
        c: *mut f32,
        ldc: i32,
    ) -> cublasStatus_t {
        let cu = match cublas().and_then(|c| c.cublas_sgemm()) {
            Ok(f) => f,
            Err(_) => return cublasStatus_t::NOT_INITIALIZED,
        };
        cu(
            handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        )
    }

    unsafe fn axpy_raw(
        handle: cublasHandle_t,
        n: i32,
        alpha: &f32,
        x: *const f32,
        incx: i32,
        y: *mut f32,
        incy: i32,
    ) -> cublasStatus_t {
        let cu = match cublas().and_then(|c| c.cublas_saxpy()) {
            Ok(f) => f,
            Err(_) => return cublasStatus_t::NOT_INITIALIZED,
        };
        cu(handle, n, alpha, x, incx, y, incy)
    }

    unsafe fn gemm_strided_batched_raw(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: &f32,
        a: *const f32,
        lda: i32,
        stride_a: i64,
        b: *const f32,
        ldb: i32,
        stride_b: i64,
        beta: &f32,
        c: *mut f32,
        ldc: i32,
        stride_c: i64,
        batch_count: i32,
    ) -> cublasStatus_t {
        let cu = match cublas().and_then(|c| c.cublas_sgemm_strided_batched()) {
            Ok(f) => f,
            Err(_) => return cublasStatus_t::NOT_INITIALIZED,
        };
        cu(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            alpha,
            a,
            lda,
            stride_a,
            b,
            ldb,
            stride_b,
            beta,
            c,
            ldc,
            stride_c,
            batch_count,
        )
    }
}

impl BlasScalar for f64 {
    unsafe fn gemm_raw(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: &f64,
        a: *const f64,
        lda: i32,
        b: *const f64,
        ldb: i32,
        beta: &f64,
        c: *mut f64,
        ldc: i32,
    ) -> cublasStatus_t {
        let cu = match cublas().and_then(|c| c.cublas_dgemm()) {
            Ok(f) => f,
            Err(_) => return cublasStatus_t::NOT_INITIALIZED,
        };
        cu(
            handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        )
    }

    unsafe fn axpy_raw(
        handle: cublasHandle_t,
        n: i32,
        alpha: &f64,
        x: *const f64,
        incx: i32,
        y: *mut f64,
        incy: i32,
    ) -> cublasStatus_t {
        let cu = match cublas().and_then(|c| c.cublas_daxpy()) {
            Ok(f) => f,
            Err(_) => return cublasStatus_t::NOT_INITIALIZED,
        };
        cu(handle, n, alpha, x, incx, y, incy)
    }

    unsafe fn gemm_strided_batched_raw(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: &f64,
        a: *const f64,
        lda: i32,
        stride_a: i64,
        b: *const f64,
        ldb: i32,
        stride_b: i64,
        beta: &f64,
        c: *mut f64,
        ldc: i32,
        stride_c: i64,
        batch_count: i32,
    ) -> cublasStatus_t {
        let cu = match cublas().and_then(|c| c.cublas_dgemm_strided_batched()) {
            Ok(f) => f,
            Err(_) => return cublasStatus_t::NOT_INITIALIZED,
        };
        cu(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            alpha,
            a,
            lda,
            stride_a,
            b,
            ldb,
            stride_b,
            beta,
            c,
            ldc,
            stride_c,
            batch_count,
        )
    }
}

impl BlasScalar for Complex32 {
    unsafe fn gemm_raw(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Complex32,
        a: *const Complex32,
        lda: i32,
        b: *const Complex32,
        ldb: i32,
        beta: &Complex32,
        c: *mut Complex32,
        ldc: i32,
    ) -> cublasStatus_t {
        let cu = match cublas().and_then(|c| c.cublas_cgemm()) {
            Ok(f) => f,
            Err(_) => return cublasStatus_t::NOT_INITIALIZED,
        };
        cu(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            alpha as *const _ as *const cuComplex,
            a as *const cuComplex,
            lda,
            b as *const cuComplex,
            ldb,
            beta as *const _ as *const cuComplex,
            c as *mut cuComplex,
            ldc,
        )
    }

    unsafe fn axpy_raw(
        handle: cublasHandle_t,
        n: i32,
        alpha: &Complex32,
        x: *const Complex32,
        incx: i32,
        y: *mut Complex32,
        incy: i32,
    ) -> cublasStatus_t {
        let cu = match cublas().and_then(|c| c.cublas_caxpy()) {
            Ok(f) => f,
            Err(_) => return cublasStatus_t::NOT_INITIALIZED,
        };
        cu(
            handle,
            n,
            alpha as *const _ as *const cuComplex,
            x as *const cuComplex,
            incx,
            y as *mut cuComplex,
            incy,
        )
    }

    unsafe fn gemm_strided_batched_raw(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Complex32,
        a: *const Complex32,
        lda: i32,
        stride_a: i64,
        b: *const Complex32,
        ldb: i32,
        stride_b: i64,
        beta: &Complex32,
        c: *mut Complex32,
        ldc: i32,
        stride_c: i64,
        batch_count: i32,
    ) -> cublasStatus_t {
        let cu = match cublas().and_then(|c| c.cublas_cgemm_strided_batched()) {
            Ok(f) => f,
            Err(_) => return cublasStatus_t::NOT_INITIALIZED,
        };
        cu(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            alpha as *const _ as *const cuComplex,
            a as *const cuComplex,
            lda,
            stride_a,
            b as *const cuComplex,
            ldb,
            stride_b,
            beta as *const _ as *const cuComplex,
            c as *mut cuComplex,
            ldc,
            stride_c,
            batch_count,
        )
    }
}

impl BlasScalar for Complex64 {
    unsafe fn gemm_raw(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Complex64,
        a: *const Complex64,
        lda: i32,
        b: *const Complex64,
        ldb: i32,
        beta: &Complex64,
        c: *mut Complex64,
        ldc: i32,
    ) -> cublasStatus_t {
        let cu = match cublas().and_then(|c| c.cublas_zgemm()) {
            Ok(f) => f,
            Err(_) => return cublasStatus_t::NOT_INITIALIZED,
        };
        cu(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            alpha as *const _ as *const cuDoubleComplex,
            a as *const cuDoubleComplex,
            lda,
            b as *const cuDoubleComplex,
            ldb,
            beta as *const _ as *const cuDoubleComplex,
            c as *mut cuDoubleComplex,
            ldc,
        )
    }

    unsafe fn axpy_raw(
        handle: cublasHandle_t,
        n: i32,
        alpha: &Complex64,
        x: *const Complex64,
        incx: i32,
        y: *mut Complex64,
        incy: i32,
    ) -> cublasStatus_t {
        let cu = match cublas().and_then(|c| c.cublas_zaxpy()) {
            Ok(f) => f,
            Err(_) => return cublasStatus_t::NOT_INITIALIZED,
        };
        cu(
            handle,
            n,
            alpha as *const _ as *const cuDoubleComplex,
            x as *const cuDoubleComplex,
            incx,
            y as *mut cuDoubleComplex,
            incy,
        )
    }

    unsafe fn gemm_strided_batched_raw(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Complex64,
        a: *const Complex64,
        lda: i32,
        stride_a: i64,
        b: *const Complex64,
        ldb: i32,
        stride_b: i64,
        beta: &Complex64,
        c: *mut Complex64,
        ldc: i32,
        stride_c: i64,
        batch_count: i32,
    ) -> cublasStatus_t {
        let cu = match cublas().and_then(|c| c.cublas_zgemm_strided_batched()) {
            Ok(f) => f,
            Err(_) => return cublasStatus_t::NOT_INITIALIZED,
        };
        cu(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            alpha as *const _ as *const cuDoubleComplex,
            a as *const cuDoubleComplex,
            lda,
            stride_a,
            b as *const cuDoubleComplex,
            ldb,
            stride_b,
            beta as *const _ as *const cuDoubleComplex,
            c as *mut cuDoubleComplex,
            ldc,
            stride_c,
            batch_count,
        )
    }
}

mod sealed {
    use baracuda_types::{Complex32, Complex64};
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for Complex32 {}
    impl Sealed for Complex64 {}
}

/// Which matrix operand to transpose.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum Op {
    /// No transpose.
    #[default]
    N,
    /// Transpose.
    T,
    /// Conjugate transpose (real types: same as `T`).
    C,
}

impl Op {
    #[inline]
    pub(crate) fn raw(self) -> cublasOperation_t {
        match self {
            Op::N => cublasOperation_t::N,
            Op::T => cublasOperation_t::T,
            Op::C => cublasOperation_t::C,
        }
    }
}

/// Compute `C = alpha * op(A) * op(B) + beta * C`, where `A`/`B`/`C` are
/// **column-major** matrices. Buffers must reside in device memory.
///
/// - `m`, `n`, `k` are the matrix shapes with transposes applied.
/// - `lda`, `ldb`, `ldc` are leading dimensions (column strides) in elements.
///
/// Column-major: for a tightly packed `m×k` matrix with no transpose,
/// `lda = m`.
#[allow(clippy::too_many_arguments)]
pub fn gemm<T: BlasScalar>(
    handle: &crate::Handle,
    transa: Op,
    transb: Op,
    m: i32,
    n: i32,
    k: i32,
    alpha: T,
    a: &DeviceBuffer<T>,
    lda: i32,
    b: &DeviceBuffer<T>,
    ldb: i32,
    beta: T,
    c: &mut DeviceBuffer<T>,
    ldc: i32,
) -> Result<()> {
    // SAFETY: the raw pointers come from live DeviceBuffers; sizes are the
    // caller's responsibility (we'd need cuBLAS to sanity-check).
    let status = unsafe {
        T::gemm_raw(
            handle.as_raw(),
            transa.raw(),
            transb.raw(),
            m,
            n,
            k,
            &alpha,
            a.as_raw().0 as *const T,
            lda,
            b.as_raw().0 as *const T,
            ldb,
            &beta,
            c.as_raw().0 as *mut T,
            ldc,
        )
    };
    check(status)
}

/// Compute `y = alpha * x + y` element-wise (BLAS-1 AXPY).
///
/// `x` and `y` are 1-D device vectors; `incx` / `incy` are element strides
/// (usually `1`).
pub fn axpy<T: BlasScalar>(
    handle: &crate::Handle,
    n: i32,
    alpha: T,
    x: &DeviceSlice<'_, T>,
    incx: i32,
    y: &mut DeviceBuffer<T>,
    incy: i32,
) -> Result<()> {
    // SAFETY: buffers live; sizes are caller's responsibility.
    let status = unsafe {
        T::axpy_raw(
            handle.as_raw(),
            n,
            &alpha,
            x.as_raw().0 as *const T,
            incx,
            y.as_raw().0 as *mut T,
            incy,
        )
    };
    check(status)
}

/// Batched `C_i = alpha * op(A_i) * op(B_i) + beta * C_i` for `i` in
/// `[0, batch_count)`, with per-matrix element strides `stride_a` / `stride_b`
/// / `stride_c`. Column-major; same `lda`/`ldb`/`ldc` interpretation as
/// [`gemm`].
///
/// This is the core primitive behind transformer multi-head attention
/// (Q·Kᵀ and AttnScores·V are both strided-batched GEMMs).
#[allow(clippy::too_many_arguments)]
pub fn gemm_strided_batched<T: BlasScalar>(
    handle: &crate::Handle,
    transa: Op,
    transb: Op,
    m: i32,
    n: i32,
    k: i32,
    alpha: T,
    a: &DeviceBuffer<T>,
    lda: i32,
    stride_a: i64,
    b: &DeviceBuffer<T>,
    ldb: i32,
    stride_b: i64,
    beta: T,
    c: &mut DeviceBuffer<T>,
    ldc: i32,
    stride_c: i64,
    batch_count: i32,
) -> Result<()> {
    // SAFETY: buffers live; caller asserts shape.
    let status = unsafe {
        T::gemm_strided_batched_raw(
            handle.as_raw(),
            transa.raw(),
            transb.raw(),
            m,
            n,
            k,
            &alpha,
            a.as_raw().0 as *const T,
            lda,
            stride_a,
            b.as_raw().0 as *const T,
            ldb,
            stride_b,
            &beta,
            c.as_raw().0 as *mut T,
            ldc,
            stride_c,
            batch_count,
        )
    };
    check(status)
}
