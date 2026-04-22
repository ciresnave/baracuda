//! Batched direct dense solvers: `getrf_batched`, `getrs_batched`,
//! `getri_batched`, `matinv_batched`.
//!
//! These let you LU-factorize or invert many independent small matrices in a
//! single cuBLAS call — a common pattern in physics simulation, Kalman
//! filtering, and small-system optimization.
//!
//! All four scalar kinds (`f32`, `f64`, [`Complex32`], [`Complex64`]) are
//! supported via the [`BatchedDirectScalar`] trait.

use baracuda_cublas_sys::functions::{cuComplex, cuDoubleComplex};
use baracuda_cublas_sys::{cublas, cublasHandle_t, cublasOperation_t, cublasStatus_t};
use baracuda_types::{Complex32, Complex64, DeviceRepr};

use crate::blas_scalar::Op;
use crate::error::{check, Result};

/// Scalars supported by the batched direct solvers.
pub trait BatchedDirectScalar: DeviceRepr + direct_sealed::Sealed {
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn getrf_batched_raw(
        h: cublasHandle_t,
        n: i32,
        a_array: *const *mut Self,
        lda: i32,
        pivots: *mut i32,
        info: *mut i32,
        batch: i32,
    ) -> cublasStatus_t;

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn getri_batched_raw(
        h: cublasHandle_t,
        n: i32,
        a_array: *const *const Self,
        lda: i32,
        pivots: *const i32,
        c_array: *const *mut Self,
        ldc: i32,
        info: *mut i32,
        batch: i32,
    ) -> cublasStatus_t;

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn getrs_batched_raw(
        h: cublasHandle_t,
        trans: cublasOperation_t,
        n: i32,
        nrhs: i32,
        a_array: *const *const Self,
        lda: i32,
        pivots: *const i32,
        b_array: *const *mut Self,
        ldb: i32,
        info: *mut i32,
        batch: i32,
    ) -> cublasStatus_t;

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn matinv_batched_raw(
        h: cublasHandle_t,
        n: i32,
        a_array: *const *const Self,
        lda: i32,
        a_inv_array: *const *mut Self,
        lda_inv: i32,
        info: *mut i32,
        batch: i32,
    ) -> cublasStatus_t;
}

macro_rules! real_impl {
    ($t:ty, $getrf:ident, $getri:ident, $getrs:ident, $matinv:ident) => {
        impl BatchedDirectScalar for $t {
            unsafe fn getrf_batched_raw(
                h: cublasHandle_t,
                n: i32,
                a: *const *mut $t,
                lda: i32,
                piv: *mut i32,
                info: *mut i32,
                batch: i32,
            ) -> cublasStatus_t {
                match cublas().and_then(|c| c.$getrf()) {
                    Ok(f) => f(h, n, a, lda, piv, info, batch),
                    Err(_) => cublasStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn getri_batched_raw(
                h: cublasHandle_t,
                n: i32,
                a: *const *const $t,
                lda: i32,
                piv: *const i32,
                c_arr: *const *mut $t,
                ldc: i32,
                info: *mut i32,
                batch: i32,
            ) -> cublasStatus_t {
                match cublas().and_then(|c| c.$getri()) {
                    Ok(f) => f(h, n, a, lda, piv, c_arr, ldc, info, batch),
                    Err(_) => cublasStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn getrs_batched_raw(
                h: cublasHandle_t,
                trans: cublasOperation_t,
                n: i32,
                nrhs: i32,
                a: *const *const $t,
                lda: i32,
                piv: *const i32,
                b: *const *mut $t,
                ldb: i32,
                info: *mut i32,
                batch: i32,
            ) -> cublasStatus_t {
                match cublas().and_then(|c| c.$getrs()) {
                    Ok(f) => f(h, trans, n, nrhs, a, lda, piv, b, ldb, info, batch),
                    Err(_) => cublasStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn matinv_batched_raw(
                h: cublasHandle_t,
                n: i32,
                a: *const *const $t,
                lda: i32,
                a_inv: *const *mut $t,
                lda_inv: i32,
                info: *mut i32,
                batch: i32,
            ) -> cublasStatus_t {
                match cublas().and_then(|c| c.$matinv()) {
                    Ok(f) => f(h, n, a, lda, a_inv, lda_inv, info, batch),
                    Err(_) => cublasStatus_t::NOT_INITIALIZED,
                }
            }
        }
    };
}

macro_rules! complex_impl {
    ($t:ty, $raw:ty, $getrf:ident, $getri:ident, $getrs:ident, $matinv:ident) => {
        impl BatchedDirectScalar for $t {
            unsafe fn getrf_batched_raw(
                h: cublasHandle_t,
                n: i32,
                a: *const *mut $t,
                lda: i32,
                piv: *mut i32,
                info: *mut i32,
                batch: i32,
            ) -> cublasStatus_t {
                match cublas().and_then(|c| c.$getrf()) {
                    Ok(f) => f(h, n, a as *const *mut $raw, lda, piv, info, batch),
                    Err(_) => cublasStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn getri_batched_raw(
                h: cublasHandle_t,
                n: i32,
                a: *const *const $t,
                lda: i32,
                piv: *const i32,
                c_arr: *const *mut $t,
                ldc: i32,
                info: *mut i32,
                batch: i32,
            ) -> cublasStatus_t {
                match cublas().and_then(|c| c.$getri()) {
                    Ok(f) => f(
                        h,
                        n,
                        a as *const *const $raw,
                        lda,
                        piv,
                        c_arr as *const *mut $raw,
                        ldc,
                        info,
                        batch,
                    ),
                    Err(_) => cublasStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn getrs_batched_raw(
                h: cublasHandle_t,
                trans: cublasOperation_t,
                n: i32,
                nrhs: i32,
                a: *const *const $t,
                lda: i32,
                piv: *const i32,
                b: *const *mut $t,
                ldb: i32,
                info: *mut i32,
                batch: i32,
            ) -> cublasStatus_t {
                match cublas().and_then(|c| c.$getrs()) {
                    Ok(f) => f(
                        h,
                        trans,
                        n,
                        nrhs,
                        a as *const *const $raw,
                        lda,
                        piv,
                        b as *const *mut $raw,
                        ldb,
                        info,
                        batch,
                    ),
                    Err(_) => cublasStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn matinv_batched_raw(
                h: cublasHandle_t,
                n: i32,
                a: *const *const $t,
                lda: i32,
                a_inv: *const *mut $t,
                lda_inv: i32,
                info: *mut i32,
                batch: i32,
            ) -> cublasStatus_t {
                match cublas().and_then(|c| c.$matinv()) {
                    Ok(f) => f(
                        h,
                        n,
                        a as *const *const $raw,
                        lda,
                        a_inv as *const *mut $raw,
                        lda_inv,
                        info,
                        batch,
                    ),
                    Err(_) => cublasStatus_t::NOT_INITIALIZED,
                }
            }
        }
    };
}

real_impl!(
    f32,
    cublas_sgetrf_batched,
    cublas_sgetri_batched,
    cublas_sgetrs_batched,
    cublas_smatinv_batched
);
real_impl!(
    f64,
    cublas_dgetrf_batched,
    cublas_dgetri_batched,
    cublas_dgetrs_batched,
    cublas_dmatinv_batched
);
complex_impl!(
    Complex32,
    cuComplex,
    cublas_cgetrf_batched,
    cublas_cgetri_batched,
    cublas_cgetrs_batched,
    cublas_cmatinv_batched
);
complex_impl!(
    Complex64,
    cuDoubleComplex,
    cublas_zgetrf_batched,
    cublas_zgetri_batched,
    cublas_zgetrs_batched,
    cublas_zmatinv_batched
);

mod direct_sealed {
    use baracuda_types::{Complex32, Complex64};
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for Complex32 {}
    impl Sealed for Complex64 {}
}

// ---- public API ---------------------------------------------------------

/// Batched LU factorization over `batch` matrices `A[k]` of size `n × n`.
///
/// `a_ptrs` is a device-side array of `batch` device pointers. `pivots` is
/// a contiguous device array of `n * batch` ints.
///
/// # Safety
/// Pointer arrays must be device-resident with at least `batch` valid
/// entries; each matrix must be `n × n` column-major.
#[allow(clippy::too_many_arguments)]
pub unsafe fn getrf<T: BatchedDirectScalar>(
    handle: &crate::Handle,
    n: i32,
    a_ptrs: *const *mut T,
    lda: i32,
    pivots: &mut baracuda_driver::DeviceBuffer<i32>,
    info: &mut baracuda_driver::DeviceBuffer<i32>,
    batch: i32,
) -> Result<()> {
    check(T::getrf_batched_raw(
        handle.as_raw(),
        n,
        a_ptrs,
        lda,
        pivots.as_raw().0 as *mut i32,
        info.as_raw().0 as *mut i32,
        batch,
    ))
}

/// Batched LU solve using pivots from [`getrf`].
///
/// # Safety
/// See [`getrf`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn getrs<T: BatchedDirectScalar>(
    handle: &crate::Handle,
    trans: Op,
    n: i32,
    nrhs: i32,
    a_ptrs: *const *const T,
    lda: i32,
    pivots: &baracuda_driver::DeviceBuffer<i32>,
    b_ptrs: *const *mut T,
    ldb: i32,
    info: &mut baracuda_driver::DeviceBuffer<i32>,
    batch: i32,
) -> Result<()> {
    check(T::getrs_batched_raw(
        handle.as_raw(),
        trans.raw(),
        n,
        nrhs,
        a_ptrs,
        lda,
        pivots.as_raw().0 as *const i32,
        b_ptrs,
        ldb,
        info.as_raw().0 as *mut i32,
        batch,
    ))
}

/// Batched inverse from an already-LU-factored matrix (companion to [`getrf`]).
///
/// # Safety
/// See [`getrf`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn getri<T: BatchedDirectScalar>(
    handle: &crate::Handle,
    n: i32,
    a_ptrs: *const *const T,
    lda: i32,
    pivots: &baracuda_driver::DeviceBuffer<i32>,
    c_ptrs: *const *mut T,
    ldc: i32,
    info: &mut baracuda_driver::DeviceBuffer<i32>,
    batch: i32,
) -> Result<()> {
    check(T::getri_batched_raw(
        handle.as_raw(),
        n,
        a_ptrs,
        lda,
        pivots.as_raw().0 as *const i32,
        c_ptrs,
        ldc,
        info.as_raw().0 as *mut i32,
        batch,
    ))
}

/// Direct batched matrix inverse — LU + inverse in a single call, for
/// matrices up to 32×32 (cuBLAS's documented cap).
///
/// # Safety
/// See [`getrf`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn matinv<T: BatchedDirectScalar>(
    handle: &crate::Handle,
    n: i32,
    a_ptrs: *const *const T,
    lda: i32,
    a_inv_ptrs: *const *mut T,
    lda_inv: i32,
    info: &mut baracuda_driver::DeviceBuffer<i32>,
    batch: i32,
) -> Result<()> {
    check(T::matinv_batched_raw(
        handle.as_raw(),
        n,
        a_ptrs,
        lda,
        a_inv_ptrs,
        lda_inv,
        info.as_raw().0 as *mut i32,
        batch,
    ))
}
