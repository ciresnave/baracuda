//! BLAS-2 routines: `gemv`, `symv`, `trmv`, `trsv`, `ger`, `syr`.
//!
//! Column-major storage. Generic over `f32` / `f64` / [`Complex32`] /
//! [`Complex64`] where the underlying BLAS defines each op.

use baracuda_cublas_sys::functions::{
    cuComplex, cuDoubleComplex, cublasDiagType_t, cublasFillMode_t,
};
use baracuda_cublas_sys::{cublas, cublasHandle_t, cublasOperation_t, cublasStatus_t};
use baracuda_driver::{DeviceBuffer, DeviceSlice};
use baracuda_types::{Complex32, Complex64, DeviceRepr};

use crate::blas_scalar::Op;
use crate::error::{check, Result};
use crate::level3::{Diag, Fill};

/// Private dispatch trait for L2 ops.
pub trait L2Scalar: DeviceRepr + l2_sealed::Sealed {
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemv_raw(
        handle: cublasHandle_t,
        trans: cublasOperation_t,
        m: i32,
        n: i32,
        alpha: &Self,
        a: *const Self,
        lda: i32,
        x: *const Self,
        incx: i32,
        beta: &Self,
        y: *mut Self,
        incy: i32,
    ) -> cublasStatus_t;
}

impl L2Scalar for f32 {
    unsafe fn gemv_raw(
        handle: cublasHandle_t,
        trans: cublasOperation_t,
        m: i32,
        n: i32,
        alpha: &f32,
        a: *const f32,
        lda: i32,
        x: *const f32,
        incx: i32,
        beta: &f32,
        y: *mut f32,
        incy: i32,
    ) -> cublasStatus_t {
        match cublas().and_then(|c| c.cublas_sgemv()) {
            Ok(f) => f(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy),
            Err(_) => cublasStatus_t::NOT_INITIALIZED,
        }
    }
}

impl L2Scalar for f64 {
    unsafe fn gemv_raw(
        handle: cublasHandle_t,
        trans: cublasOperation_t,
        m: i32,
        n: i32,
        alpha: &f64,
        a: *const f64,
        lda: i32,
        x: *const f64,
        incx: i32,
        beta: &f64,
        y: *mut f64,
        incy: i32,
    ) -> cublasStatus_t {
        match cublas().and_then(|c| c.cublas_dgemv()) {
            Ok(f) => f(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy),
            Err(_) => cublasStatus_t::NOT_INITIALIZED,
        }
    }
}

impl L2Scalar for Complex32 {
    unsafe fn gemv_raw(
        handle: cublasHandle_t,
        trans: cublasOperation_t,
        m: i32,
        n: i32,
        alpha: &Complex32,
        a: *const Complex32,
        lda: i32,
        x: *const Complex32,
        incx: i32,
        beta: &Complex32,
        y: *mut Complex32,
        incy: i32,
    ) -> cublasStatus_t {
        match cublas().and_then(|c| c.cublas_cgemv()) {
            Ok(f) => f(
                handle,
                trans,
                m,
                n,
                alpha as *const _ as *const cuComplex,
                a as *const cuComplex,
                lda,
                x as *const cuComplex,
                incx,
                beta as *const _ as *const cuComplex,
                y as *mut cuComplex,
                incy,
            ),
            Err(_) => cublasStatus_t::NOT_INITIALIZED,
        }
    }
}

impl L2Scalar for Complex64 {
    unsafe fn gemv_raw(
        handle: cublasHandle_t,
        trans: cublasOperation_t,
        m: i32,
        n: i32,
        alpha: &Complex64,
        a: *const Complex64,
        lda: i32,
        x: *const Complex64,
        incx: i32,
        beta: &Complex64,
        y: *mut Complex64,
        incy: i32,
    ) -> cublasStatus_t {
        match cublas().and_then(|c| c.cublas_zgemv()) {
            Ok(f) => f(
                handle,
                trans,
                m,
                n,
                alpha as *const _ as *const cuDoubleComplex,
                a as *const cuDoubleComplex,
                lda,
                x as *const cuDoubleComplex,
                incx,
                beta as *const _ as *const cuDoubleComplex,
                y as *mut cuDoubleComplex,
                incy,
            ),
            Err(_) => cublasStatus_t::NOT_INITIALIZED,
        }
    }
}

mod l2_sealed {
    use baracuda_types::{Complex32, Complex64};
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for Complex32 {}
    impl Sealed for Complex64 {}

    pub trait SealedReal: Sealed {}
    impl SealedReal for f32 {}
    impl SealedReal for f64 {}
}

/// Compute `y = alpha * op(A) * x + beta * y`.
///
/// `A` is column-major `m × n` (before transpose). `lda` is the column
/// stride of `A` in elements.
#[allow(clippy::too_many_arguments)]
pub fn gemv<T: L2Scalar>(
    handle: &crate::Handle,
    trans: Op,
    m: i32,
    n: i32,
    alpha: T,
    a: &DeviceBuffer<T>,
    lda: i32,
    x: &DeviceSlice<'_, T>,
    incx: i32,
    beta: T,
    y: &mut DeviceBuffer<T>,
    incy: i32,
) -> Result<()> {
    let status = unsafe {
        T::gemv_raw(
            handle.as_raw(),
            trans.raw(),
            m,
            n,
            &alpha,
            a.as_raw().0 as *const T,
            lda,
            x.as_raw().0 as *const T,
            incx,
            &beta,
            y.as_raw().0 as *mut T,
            incy,
        )
    };
    check(status)
}

// ---- Real-only BLAS-2 routines ------------------------------------------

/// Private dispatch trait for the real-only BLAS-2 ops
/// (symv / trmv / trsv / ger / syr) which exist only for f32 and f64 in
/// cuBLAS's public API.
pub trait L2Real: L2Scalar + l2_sealed::SealedReal {
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn symv_raw(
        h: cublasHandle_t,
        uplo: cublasFillMode_t,
        n: i32,
        alpha: &Self,
        a: *const Self,
        lda: i32,
        x: *const Self,
        incx: i32,
        beta: &Self,
        y: *mut Self,
        incy: i32,
    ) -> cublasStatus_t;

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn trmv_raw(
        h: cublasHandle_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        diag: cublasDiagType_t,
        n: i32,
        a: *const Self,
        lda: i32,
        x: *mut Self,
        incx: i32,
    ) -> cublasStatus_t;

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn trsv_raw(
        h: cublasHandle_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        diag: cublasDiagType_t,
        n: i32,
        a: *const Self,
        lda: i32,
        x: *mut Self,
        incx: i32,
    ) -> cublasStatus_t;

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn ger_raw(
        h: cublasHandle_t,
        m: i32,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        a: *mut Self,
        lda: i32,
    ) -> cublasStatus_t;

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn syr_raw(
        h: cublasHandle_t,
        uplo: cublasFillMode_t,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        a: *mut Self,
        lda: i32,
    ) -> cublasStatus_t;
}

macro_rules! l2_real_impl {
    ($t:ty, $symv:ident, $trmv:ident, $trsv:ident, $ger:ident, $syr:ident) => {
        impl L2Real for $t {
            unsafe fn symv_raw(
                h: cublasHandle_t,
                uplo: cublasFillMode_t,
                n: i32,
                alpha: &$t,
                a: *const $t,
                lda: i32,
                x: *const $t,
                incx: i32,
                beta: &$t,
                y: *mut $t,
                incy: i32,
            ) -> cublasStatus_t {
                match cublas().and_then(|c| c.$symv()) {
                    Ok(f) => f(h, uplo, n, alpha, a, lda, x, incx, beta, y, incy),
                    Err(_) => cublasStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn trmv_raw(
                h: cublasHandle_t,
                uplo: cublasFillMode_t,
                trans: cublasOperation_t,
                diag: cublasDiagType_t,
                n: i32,
                a: *const $t,
                lda: i32,
                x: *mut $t,
                incx: i32,
            ) -> cublasStatus_t {
                match cublas().and_then(|c| c.$trmv()) {
                    Ok(f) => f(h, uplo, trans, diag, n, a, lda, x, incx),
                    Err(_) => cublasStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn trsv_raw(
                h: cublasHandle_t,
                uplo: cublasFillMode_t,
                trans: cublasOperation_t,
                diag: cublasDiagType_t,
                n: i32,
                a: *const $t,
                lda: i32,
                x: *mut $t,
                incx: i32,
            ) -> cublasStatus_t {
                match cublas().and_then(|c| c.$trsv()) {
                    Ok(f) => f(h, uplo, trans, diag, n, a, lda, x, incx),
                    Err(_) => cublasStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn ger_raw(
                h: cublasHandle_t,
                m: i32,
                n: i32,
                alpha: &$t,
                x: *const $t,
                incx: i32,
                y: *const $t,
                incy: i32,
                a: *mut $t,
                lda: i32,
            ) -> cublasStatus_t {
                match cublas().and_then(|c| c.$ger()) {
                    Ok(f) => f(h, m, n, alpha, x, incx, y, incy, a, lda),
                    Err(_) => cublasStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn syr_raw(
                h: cublasHandle_t,
                uplo: cublasFillMode_t,
                n: i32,
                alpha: &$t,
                x: *const $t,
                incx: i32,
                a: *mut $t,
                lda: i32,
            ) -> cublasStatus_t {
                match cublas().and_then(|c| c.$syr()) {
                    Ok(f) => f(h, uplo, n, alpha, x, incx, a, lda),
                    Err(_) => cublasStatus_t::NOT_INITIALIZED,
                }
            }
        }
    };
}

l2_real_impl!(
    f32,
    cublas_ssymv,
    cublas_strmv,
    cublas_strsv,
    cublas_sger,
    cublas_ssyr
);
l2_real_impl!(
    f64,
    cublas_dsymv,
    cublas_dtrmv,
    cublas_dtrsv,
    cublas_dger,
    cublas_dsyr
);

/// Symmetric matrix-vector multiply (real types).
#[allow(clippy::too_many_arguments)]
pub fn symv<T: L2Real>(
    handle: &crate::Handle,
    uplo: Fill,
    n: i32,
    alpha: T,
    a: &DeviceBuffer<T>,
    lda: i32,
    x: &DeviceSlice<'_, T>,
    incx: i32,
    beta: T,
    y: &mut DeviceBuffer<T>,
    incy: i32,
) -> Result<()> {
    let status = unsafe {
        T::symv_raw(
            handle.as_raw(),
            uplo.raw(),
            n,
            &alpha,
            a.as_raw().0 as *const T,
            lda,
            x.as_raw().0 as *const T,
            incx,
            &beta,
            y.as_raw().0 as *mut T,
            incy,
        )
    };
    check(status)
}

/// Triangular matrix-vector multiply, in place on `x`.
#[allow(clippy::too_many_arguments)]
pub fn trmv<T: L2Real>(
    handle: &crate::Handle,
    uplo: Fill,
    trans: Op,
    diag: Diag,
    n: i32,
    a: &DeviceBuffer<T>,
    lda: i32,
    x: &mut DeviceBuffer<T>,
    incx: i32,
) -> Result<()> {
    let status = unsafe {
        T::trmv_raw(
            handle.as_raw(),
            uplo.raw(),
            trans.raw(),
            diag.raw(),
            n,
            a.as_raw().0 as *const T,
            lda,
            x.as_raw().0 as *mut T,
            incx,
        )
    };
    check(status)
}

/// Solve `A * x = b` (triangular `A`), result in `x` (overwrites input).
#[allow(clippy::too_many_arguments)]
pub fn trsv<T: L2Real>(
    handle: &crate::Handle,
    uplo: Fill,
    trans: Op,
    diag: Diag,
    n: i32,
    a: &DeviceBuffer<T>,
    lda: i32,
    x: &mut DeviceBuffer<T>,
    incx: i32,
) -> Result<()> {
    let status = unsafe {
        T::trsv_raw(
            handle.as_raw(),
            uplo.raw(),
            trans.raw(),
            diag.raw(),
            n,
            a.as_raw().0 as *const T,
            lda,
            x.as_raw().0 as *mut T,
            incx,
        )
    };
    check(status)
}

/// Rank-1 update: `A = alpha * x * yᵀ + A`.
#[allow(clippy::too_many_arguments)]
pub fn ger<T: L2Real>(
    handle: &crate::Handle,
    m: i32,
    n: i32,
    alpha: T,
    x: &DeviceSlice<'_, T>,
    incx: i32,
    y: &DeviceSlice<'_, T>,
    incy: i32,
    a: &mut DeviceBuffer<T>,
    lda: i32,
) -> Result<()> {
    let status = unsafe {
        T::ger_raw(
            handle.as_raw(),
            m,
            n,
            &alpha,
            x.as_raw().0 as *const T,
            incx,
            y.as_raw().0 as *const T,
            incy,
            a.as_raw().0 as *mut T,
            lda,
        )
    };
    check(status)
}

/// Symmetric rank-1 update: `A = alpha * x * xᵀ + A`.
#[allow(clippy::too_many_arguments)]
pub fn syr<T: L2Real>(
    handle: &crate::Handle,
    uplo: Fill,
    n: i32,
    alpha: T,
    x: &DeviceSlice<'_, T>,
    incx: i32,
    a: &mut DeviceBuffer<T>,
    lda: i32,
) -> Result<()> {
    let status = unsafe {
        T::syr_raw(
            handle.as_raw(),
            uplo.raw(),
            n,
            &alpha,
            x.as_raw().0 as *const T,
            incx,
            a.as_raw().0 as *mut T,
            lda,
        )
    };
    check(status)
}
