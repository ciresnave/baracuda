//! BLAS-3 routines: `symm`, `hemm`, `syrk`, `herk`, `trmm`, `trsm`.
//!
//! Generic over `f32` / `f64` / [`Complex32`] / [`Complex64`] where the real
//! BLAS defines the operation. `hemm`/`herk` only make sense for complex
//! types (seal keeps real types out at compile time). Column-major storage,
//! all dimensions `i32`.

use baracuda_cublas_sys::functions::{
    cuComplex, cuDoubleComplex, cublasDiagType_t, cublasFillMode_t, cublasSideMode_t,
};
use baracuda_cublas_sys::{cublas, cublasHandle_t, cublasOperation_t, cublasStatus_t};
use baracuda_driver::DeviceBuffer;
use baracuda_types::{Complex32, Complex64, DeviceRepr};

use crate::blas_scalar::Op;
use crate::error::{check, Result};

/// Triangular storage mode.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Fill {
    Lower,
    Upper,
}

impl Fill {
    pub(crate) fn raw(self) -> cublasFillMode_t {
        match self {
            Fill::Lower => cublasFillMode_t::Lower,
            Fill::Upper => cublasFillMode_t::Upper,
        }
    }
}

/// Whether the structured operand is on the left or right of the product.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Side {
    Left,
    Right,
}

impl Side {
    pub(crate) fn raw(self) -> cublasSideMode_t {
        match self {
            Side::Left => cublasSideMode_t::Left,
            Side::Right => cublasSideMode_t::Right,
        }
    }
}

/// Triangular unit-diagonal flag.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Diag {
    NonUnit,
    Unit,
}

impl Diag {
    pub(crate) fn raw(self) -> cublasDiagType_t {
        match self {
            Diag::NonUnit => cublasDiagType_t::NonUnit,
            Diag::Unit => cublasDiagType_t::Unit,
        }
    }
}

/// Private dispatch trait for the BLAS-3 operations that exist for all four
/// scalar kinds (symm / syrk / trmm / trsm).
pub trait L3Scalar: DeviceRepr + l3_sealed::Sealed {
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn symm_raw(
        h: cublasHandle_t,
        side: cublasSideMode_t,
        uplo: cublasFillMode_t,
        m: i32,
        n: i32,
        alpha: &Self,
        a: *const Self,
        lda: i32,
        b: *const Self,
        ldb: i32,
        beta: &Self,
        c: *mut Self,
        ldc: i32,
    ) -> cublasStatus_t;

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn syrk_raw(
        h: cublasHandle_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        n: i32,
        k: i32,
        alpha: &Self,
        a: *const Self,
        lda: i32,
        beta: &Self,
        c: *mut Self,
        ldc: i32,
    ) -> cublasStatus_t;

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn trmm_raw(
        h: cublasHandle_t,
        side: cublasSideMode_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        diag: cublasDiagType_t,
        m: i32,
        n: i32,
        alpha: &Self,
        a: *const Self,
        lda: i32,
        b: *const Self,
        ldb: i32,
        c: *mut Self,
        ldc: i32,
    ) -> cublasStatus_t;

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn trsm_raw(
        h: cublasHandle_t,
        side: cublasSideMode_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        diag: cublasDiagType_t,
        m: i32,
        n: i32,
        alpha: &Self,
        a: *const Self,
        lda: i32,
        b: *mut Self,
        ldb: i32,
    ) -> cublasStatus_t;
}

macro_rules! l3_real_impl {
    ($t:ty, $symm:ident, $syrk:ident, $trmm:ident, $trsm:ident) => {
        impl L3Scalar for $t {
            unsafe fn symm_raw(
                h: cublasHandle_t,
                side: cublasSideMode_t,
                uplo: cublasFillMode_t,
                m: i32,
                n: i32,
                alpha: &$t,
                a: *const $t,
                lda: i32,
                b: *const $t,
                ldb: i32,
                beta: &$t,
                c: *mut $t,
                ldc: i32,
            ) -> cublasStatus_t {
                match cublas().and_then(|c| c.$symm()) {
                    Ok(f) => f(h, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc),
                    Err(_) => cublasStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn syrk_raw(
                h: cublasHandle_t,
                uplo: cublasFillMode_t,
                trans: cublasOperation_t,
                n: i32,
                k: i32,
                alpha: &$t,
                a: *const $t,
                lda: i32,
                beta: &$t,
                c: *mut $t,
                ldc: i32,
            ) -> cublasStatus_t {
                match cublas().and_then(|c| c.$syrk()) {
                    Ok(f) => f(h, uplo, trans, n, k, alpha, a, lda, beta, c, ldc),
                    Err(_) => cublasStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn trmm_raw(
                h: cublasHandle_t,
                side: cublasSideMode_t,
                uplo: cublasFillMode_t,
                trans: cublasOperation_t,
                diag: cublasDiagType_t,
                m: i32,
                n: i32,
                alpha: &$t,
                a: *const $t,
                lda: i32,
                b: *const $t,
                ldb: i32,
                c: *mut $t,
                ldc: i32,
            ) -> cublasStatus_t {
                match cublas().and_then(|c| c.$trmm()) {
                    Ok(f) => f(
                        h, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, c, ldc,
                    ),
                    Err(_) => cublasStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn trsm_raw(
                h: cublasHandle_t,
                side: cublasSideMode_t,
                uplo: cublasFillMode_t,
                trans: cublasOperation_t,
                diag: cublasDiagType_t,
                m: i32,
                n: i32,
                alpha: &$t,
                a: *const $t,
                lda: i32,
                b: *mut $t,
                ldb: i32,
            ) -> cublasStatus_t {
                match cublas().and_then(|c| c.$trsm()) {
                    Ok(f) => f(h, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb),
                    Err(_) => cublasStatus_t::NOT_INITIALIZED,
                }
            }
        }
    };
}

macro_rules! l3_complex_impl {
    ($t:ty, $raw:ty, $symm:ident, $syrk:ident, $trmm:ident, $trsm:ident) => {
        impl L3Scalar for $t {
            unsafe fn symm_raw(
                h: cublasHandle_t,
                side: cublasSideMode_t,
                uplo: cublasFillMode_t,
                m: i32,
                n: i32,
                alpha: &$t,
                a: *const $t,
                lda: i32,
                b: *const $t,
                ldb: i32,
                beta: &$t,
                c: *mut $t,
                ldc: i32,
            ) -> cublasStatus_t {
                match cublas().and_then(|c| c.$symm()) {
                    Ok(f) => f(
                        h,
                        side,
                        uplo,
                        m,
                        n,
                        alpha as *const _ as *const $raw,
                        a as *const $raw,
                        lda,
                        b as *const $raw,
                        ldb,
                        beta as *const _ as *const $raw,
                        c as *mut $raw,
                        ldc,
                    ),
                    Err(_) => cublasStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn syrk_raw(
                h: cublasHandle_t,
                uplo: cublasFillMode_t,
                trans: cublasOperation_t,
                n: i32,
                k: i32,
                alpha: &$t,
                a: *const $t,
                lda: i32,
                beta: &$t,
                c: *mut $t,
                ldc: i32,
            ) -> cublasStatus_t {
                match cublas().and_then(|c| c.$syrk()) {
                    Ok(f) => f(
                        h,
                        uplo,
                        trans,
                        n,
                        k,
                        alpha as *const _ as *const $raw,
                        a as *const $raw,
                        lda,
                        beta as *const _ as *const $raw,
                        c as *mut $raw,
                        ldc,
                    ),
                    Err(_) => cublasStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn trmm_raw(
                h: cublasHandle_t,
                side: cublasSideMode_t,
                uplo: cublasFillMode_t,
                trans: cublasOperation_t,
                diag: cublasDiagType_t,
                m: i32,
                n: i32,
                alpha: &$t,
                a: *const $t,
                lda: i32,
                b: *const $t,
                ldb: i32,
                c: *mut $t,
                ldc: i32,
            ) -> cublasStatus_t {
                match cublas().and_then(|c| c.$trmm()) {
                    Ok(f) => f(
                        h,
                        side,
                        uplo,
                        trans,
                        diag,
                        m,
                        n,
                        alpha as *const _ as *const $raw,
                        a as *const $raw,
                        lda,
                        b as *const $raw,
                        ldb,
                        c as *mut $raw,
                        ldc,
                    ),
                    Err(_) => cublasStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn trsm_raw(
                h: cublasHandle_t,
                side: cublasSideMode_t,
                uplo: cublasFillMode_t,
                trans: cublasOperation_t,
                diag: cublasDiagType_t,
                m: i32,
                n: i32,
                alpha: &$t,
                a: *const $t,
                lda: i32,
                b: *mut $t,
                ldb: i32,
            ) -> cublasStatus_t {
                match cublas().and_then(|c| c.$trsm()) {
                    Ok(f) => f(
                        h,
                        side,
                        uplo,
                        trans,
                        diag,
                        m,
                        n,
                        alpha as *const _ as *const $raw,
                        a as *const $raw,
                        lda,
                        b as *mut $raw,
                        ldb,
                    ),
                    Err(_) => cublasStatus_t::NOT_INITIALIZED,
                }
            }
        }
    };
}

l3_real_impl!(f32, cublas_ssymm, cublas_ssyrk, cublas_strmm, cublas_strsm);
l3_real_impl!(f64, cublas_dsymm, cublas_dsyrk, cublas_dtrmm, cublas_dtrsm);
l3_complex_impl!(
    Complex32,
    cuComplex,
    cublas_csymm,
    cublas_csyrk,
    cublas_ctrmm,
    cublas_ctrsm
);
l3_complex_impl!(
    Complex64,
    cuDoubleComplex,
    cublas_zsymm,
    cublas_zsyrk,
    cublas_ztrmm,
    cublas_ztrsm
);

/// `hemm` / `herk` — Hermitian-specific operations, complex only.
pub trait HermitianScalar: L3Scalar + l3_sealed::SealedComplex {
    /// Real-valued alpha type for `herk` (f32 for Complex32, f64 for Complex64).
    type Real: DeviceRepr;

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn hemm_raw(
        h: cublasHandle_t,
        side: cublasSideMode_t,
        uplo: cublasFillMode_t,
        m: i32,
        n: i32,
        alpha: &Self,
        a: *const Self,
        lda: i32,
        b: *const Self,
        ldb: i32,
        beta: &Self,
        c: *mut Self,
        ldc: i32,
    ) -> cublasStatus_t;

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn herk_raw(
        h: cublasHandle_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        n: i32,
        k: i32,
        alpha: &Self::Real,
        a: *const Self,
        lda: i32,
        beta: &Self::Real,
        c: *mut Self,
        ldc: i32,
    ) -> cublasStatus_t;
}

impl HermitianScalar for Complex32 {
    type Real = f32;

    unsafe fn hemm_raw(
        h: cublasHandle_t,
        side: cublasSideMode_t,
        uplo: cublasFillMode_t,
        m: i32,
        n: i32,
        alpha: &Complex32,
        a: *const Complex32,
        lda: i32,
        b: *const Complex32,
        ldb: i32,
        beta: &Complex32,
        c: *mut Complex32,
        ldc: i32,
    ) -> cublasStatus_t {
        match cublas().and_then(|c| c.cublas_chemm()) {
            Ok(f) => f(
                h,
                side,
                uplo,
                m,
                n,
                alpha as *const _ as *const cuComplex,
                a as *const cuComplex,
                lda,
                b as *const cuComplex,
                ldb,
                beta as *const _ as *const cuComplex,
                c as *mut cuComplex,
                ldc,
            ),
            Err(_) => cublasStatus_t::NOT_INITIALIZED,
        }
    }

    unsafe fn herk_raw(
        h: cublasHandle_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        n: i32,
        k: i32,
        alpha: &f32,
        a: *const Complex32,
        lda: i32,
        beta: &f32,
        c: *mut Complex32,
        ldc: i32,
    ) -> cublasStatus_t {
        match cublas().and_then(|c| c.cublas_cherk()) {
            Ok(f) => f(
                h,
                uplo,
                trans,
                n,
                k,
                alpha,
                a as *const cuComplex,
                lda,
                beta,
                c as *mut cuComplex,
                ldc,
            ),
            Err(_) => cublasStatus_t::NOT_INITIALIZED,
        }
    }
}

impl HermitianScalar for Complex64 {
    type Real = f64;

    unsafe fn hemm_raw(
        h: cublasHandle_t,
        side: cublasSideMode_t,
        uplo: cublasFillMode_t,
        m: i32,
        n: i32,
        alpha: &Complex64,
        a: *const Complex64,
        lda: i32,
        b: *const Complex64,
        ldb: i32,
        beta: &Complex64,
        c: *mut Complex64,
        ldc: i32,
    ) -> cublasStatus_t {
        match cublas().and_then(|c| c.cublas_zhemm()) {
            Ok(f) => f(
                h,
                side,
                uplo,
                m,
                n,
                alpha as *const _ as *const cuDoubleComplex,
                a as *const cuDoubleComplex,
                lda,
                b as *const cuDoubleComplex,
                ldb,
                beta as *const _ as *const cuDoubleComplex,
                c as *mut cuDoubleComplex,
                ldc,
            ),
            Err(_) => cublasStatus_t::NOT_INITIALIZED,
        }
    }

    unsafe fn herk_raw(
        h: cublasHandle_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        n: i32,
        k: i32,
        alpha: &f64,
        a: *const Complex64,
        lda: i32,
        beta: &f64,
        c: *mut Complex64,
        ldc: i32,
    ) -> cublasStatus_t {
        match cublas().and_then(|c| c.cublas_zherk()) {
            Ok(f) => f(
                h,
                uplo,
                trans,
                n,
                k,
                alpha,
                a as *const cuDoubleComplex,
                lda,
                beta,
                c as *mut cuDoubleComplex,
                ldc,
            ),
            Err(_) => cublasStatus_t::NOT_INITIALIZED,
        }
    }
}

mod l3_sealed {
    use baracuda_types::{Complex32, Complex64};

    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for Complex32 {}
    impl Sealed for Complex64 {}

    pub trait SealedComplex: Sealed {}
    impl SealedComplex for Complex32 {}
    impl SealedComplex for Complex64 {}
}

// ---- public API -----------------------------------------------------------

/// Symmetric matrix × general matrix: `C = alpha * A * B + beta * C` (Left)
/// or `C = alpha * B * A + beta * C` (Right), where `A` is symmetric and only
/// the triangle indicated by `uplo` is read.
#[allow(clippy::too_many_arguments)]
pub fn symm<T: L3Scalar>(
    handle: &crate::Handle,
    side: Side,
    uplo: Fill,
    m: i32,
    n: i32,
    alpha: T,
    a: &DeviceBuffer<T>,
    lda: i32,
    b: &DeviceBuffer<T>,
    ldb: i32,
    beta: T,
    c: &mut DeviceBuffer<T>,
    ldc: i32,
) -> Result<()> {
    let status = unsafe {
        T::symm_raw(
            handle.as_raw(),
            side.raw(),
            uplo.raw(),
            m,
            n,
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

/// Symmetric rank-k update: `C = alpha * A * Aᵀ + beta * C` (or with trans).
#[allow(clippy::too_many_arguments)]
pub fn syrk<T: L3Scalar>(
    handle: &crate::Handle,
    uplo: Fill,
    trans: Op,
    n: i32,
    k: i32,
    alpha: T,
    a: &DeviceBuffer<T>,
    lda: i32,
    beta: T,
    c: &mut DeviceBuffer<T>,
    ldc: i32,
) -> Result<()> {
    let status = unsafe {
        T::syrk_raw(
            handle.as_raw(),
            uplo.raw(),
            trans.raw(),
            n,
            k,
            &alpha,
            a.as_raw().0 as *const T,
            lda,
            &beta,
            c.as_raw().0 as *mut T,
            ldc,
        )
    };
    check(status)
}

/// Triangular matrix × general matrix: `C = alpha * op(A) * B` or
/// `C = alpha * B * op(A)`, writing to `C` (which may alias `B`).
#[allow(clippy::too_many_arguments)]
pub fn trmm<T: L3Scalar>(
    handle: &crate::Handle,
    side: Side,
    uplo: Fill,
    trans: Op,
    diag: Diag,
    m: i32,
    n: i32,
    alpha: T,
    a: &DeviceBuffer<T>,
    lda: i32,
    b: &DeviceBuffer<T>,
    ldb: i32,
    c: &mut DeviceBuffer<T>,
    ldc: i32,
) -> Result<()> {
    let status = unsafe {
        T::trmm_raw(
            handle.as_raw(),
            side.raw(),
            uplo.raw(),
            trans.raw(),
            diag.raw(),
            m,
            n,
            &alpha,
            a.as_raw().0 as *const T,
            lda,
            b.as_raw().0 as *const T,
            ldb,
            c.as_raw().0 as *mut T,
            ldc,
        )
    };
    check(status)
}

/// Solve `op(A) * X = alpha * B` for `X`, in place on `B`. `A` is triangular.
#[allow(clippy::too_many_arguments)]
pub fn trsm<T: L3Scalar>(
    handle: &crate::Handle,
    side: Side,
    uplo: Fill,
    trans: Op,
    diag: Diag,
    m: i32,
    n: i32,
    alpha: T,
    a: &DeviceBuffer<T>,
    lda: i32,
    b: &mut DeviceBuffer<T>,
    ldb: i32,
) -> Result<()> {
    let status = unsafe {
        T::trsm_raw(
            handle.as_raw(),
            side.raw(),
            uplo.raw(),
            trans.raw(),
            diag.raw(),
            m,
            n,
            &alpha,
            a.as_raw().0 as *const T,
            lda,
            b.as_raw().0 as *mut T,
            ldb,
        )
    };
    check(status)
}

/// Hermitian matrix × general matrix (complex only).
#[allow(clippy::too_many_arguments)]
pub fn hemm<T: HermitianScalar>(
    handle: &crate::Handle,
    side: Side,
    uplo: Fill,
    m: i32,
    n: i32,
    alpha: T,
    a: &DeviceBuffer<T>,
    lda: i32,
    b: &DeviceBuffer<T>,
    ldb: i32,
    beta: T,
    c: &mut DeviceBuffer<T>,
    ldc: i32,
) -> Result<()> {
    let status = unsafe {
        T::hemm_raw(
            handle.as_raw(),
            side.raw(),
            uplo.raw(),
            m,
            n,
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

/// Hermitian rank-k update (complex only). Note alpha/beta are real-typed.
#[allow(clippy::too_many_arguments)]
pub fn herk<T: HermitianScalar>(
    handle: &crate::Handle,
    uplo: Fill,
    trans: Op,
    n: i32,
    k: i32,
    alpha: T::Real,
    a: &DeviceBuffer<T>,
    lda: i32,
    beta: T::Real,
    c: &mut DeviceBuffer<T>,
    ldc: i32,
) -> Result<()> {
    let status = unsafe {
        T::herk_raw(
            handle.as_raw(),
            uplo.raw(),
            trans.raw(),
            n,
            k,
            &alpha,
            a.as_raw().0 as *const T,
            lda,
            &beta,
            c.as_raw().0 as *mut T,
            ldc,
        )
    };
    check(status)
}

