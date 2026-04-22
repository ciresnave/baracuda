//! cuBLASXt — multi-GPU GEMM.
//!
//! cuBLASXt shares the cuBLAS shared library. Create an [`XtHandle`],
//! tell it which CUDA devices to stripe across with
//! [`XtHandle::device_select`], then call [`gemm`] with **host-pointer**
//! operands — XT handles the chunking across the selected devices.

use baracuda_cublas_sys::functions::{cuComplex, cuDoubleComplex, cublasXtHandle_t};
use baracuda_cublas_sys::{cublas, cublasStatus_t};
use baracuda_types::{Complex32, Complex64, DeviceRepr};

use crate::blas_scalar::Op;
use crate::error::{check, Result};

/// Owned cuBLASXt handle.
#[derive(Debug)]
pub struct XtHandle {
    raw: cublasXtHandle_t,
}

unsafe impl Send for XtHandle {}

impl XtHandle {
    pub fn new() -> Result<Self> {
        let c = cublas()?;
        let f = c.cublas_xt_create()?;
        let mut h: cublasXtHandle_t = core::ptr::null_mut();
        check(unsafe { f(&mut h) })?;
        Ok(Self { raw: h })
    }

    pub fn device_select(&self, devices: &[i32]) -> Result<()> {
        let c = cublas()?;
        let f = c.cublas_xt_device_select()?;
        check(unsafe { f(self.raw, devices.len() as i32, devices.as_ptr()) })
    }

    pub fn set_block_dim(&self, block_dim: i32) -> Result<()> {
        let c = cublas()?;
        let f = c.cublas_xt_set_block_dim()?;
        check(unsafe { f(self.raw, block_dim) })
    }

    pub fn block_dim(&self) -> Result<i32> {
        let c = cublas()?;
        let f = c.cublas_xt_get_block_dim()?;
        let mut v = 0i32;
        check(unsafe { f(self.raw, &mut v) })?;
        Ok(v)
    }

    pub fn as_raw(&self) -> cublasXtHandle_t {
        self.raw
    }
}

impl Drop for XtHandle {
    fn drop(&mut self) {
        if let Ok(c) = cublas() {
            if let Ok(f) = c.cublas_xt_destroy() {
                let _ = unsafe { f(self.raw) };
            }
        }
    }
}

/// Scalars supported by cuBLASXt GEMM.
pub trait XtScalar: DeviceRepr + xt_sealed::Sealed {
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn xt_gemm_raw(
        h: cublasXtHandle_t,
        transa: Op,
        transb: Op,
        m: usize,
        n: usize,
        k: usize,
        alpha: &Self,
        a: *const Self,
        lda: usize,
        b: *const Self,
        ldb: usize,
        beta: &Self,
        c: *mut Self,
        ldc: usize,
    ) -> cublasStatus_t;
}

macro_rules! real_impl {
    ($t:ty, $xt:ident) => {
        impl XtScalar for $t {
            unsafe fn xt_gemm_raw(
                h: cublasXtHandle_t,
                transa: Op,
                transb: Op,
                m: usize,
                n: usize,
                k: usize,
                alpha: &$t,
                a: *const $t,
                lda: usize,
                b: *const $t,
                ldb: usize,
                beta: &$t,
                c: *mut $t,
                ldc: usize,
            ) -> cublasStatus_t {
                match cublas().and_then(|c| c.$xt()) {
                    Ok(f) => f(
                        h,
                        transa.raw(),
                        transb.raw(),
                        m,
                        n,
                        k,
                        alpha,
                        a,
                        lda,
                        b,
                        ldb,
                        beta,
                        c,
                        ldc,
                    ),
                    Err(_) => cublasStatus_t::NOT_INITIALIZED,
                }
            }
        }
    };
}

macro_rules! complex_impl {
    ($t:ty, $raw:ty, $xt:ident) => {
        impl XtScalar for $t {
            unsafe fn xt_gemm_raw(
                h: cublasXtHandle_t,
                transa: Op,
                transb: Op,
                m: usize,
                n: usize,
                k: usize,
                alpha: &$t,
                a: *const $t,
                lda: usize,
                b: *const $t,
                ldb: usize,
                beta: &$t,
                c: *mut $t,
                ldc: usize,
            ) -> cublasStatus_t {
                match cublas().and_then(|c| c.$xt()) {
                    Ok(f) => f(
                        h,
                        transa.raw(),
                        transb.raw(),
                        m,
                        n,
                        k,
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
        }
    };
}

real_impl!(f32, cublas_xt_sgemm);
real_impl!(f64, cublas_xt_dgemm);
complex_impl!(Complex32, cuComplex, cublas_xt_cgemm);
complex_impl!(Complex64, cuDoubleComplex, cublas_xt_zgemm);

mod xt_sealed {
    use baracuda_types::{Complex32, Complex64};
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for Complex32 {}
    impl Sealed for Complex64 {}
}

/// Multi-GPU GEMM. `a`, `b`, `c` are **host pointers**; cuBLASXt handles
/// device-side tiling internally.
///
/// # Safety
/// `a`/`b`/`c` must point to live host memory of the expected sizes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm<T: XtScalar>(
    handle: &XtHandle,
    transa: Op,
    transb: Op,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: *const T,
    lda: usize,
    b: *const T,
    ldb: usize,
    beta: T,
    c: *mut T,
    ldc: usize,
) -> Result<()> {
    let status = T::xt_gemm_raw(
        handle.as_raw(),
        transa,
        transb,
        m,
        n,
        k,
        &alpha,
        a,
        lda,
        b,
        ldb,
        &beta,
        c,
        ldc,
    );
    check(status)
}

