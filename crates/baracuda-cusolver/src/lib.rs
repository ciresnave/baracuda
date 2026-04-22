//! Safe Rust wrappers for NVIDIA cuSOLVER.
//!
//! Covers the dense API (`Dn`) for all four BLAS scalar types:
//! - LU factorization: `getrf` + `getrs`
//! - QR factorization: `geqrf`
//! - Cholesky: `potrf` + `potrs`
//! - SVD: `gesvd`
//! - Symmetric / Hermitian eigendecomposition: `syevd` / `heevd`
//!
//! The generic 64-bit X… API (`xgetrf`, `xgeqrf`, `xpotrf`) gives
//! type-erased data pointers and is exposed under [`xapi`]. The sparse API
//! (`cusolverSp*`) is under [`sparse`]. The refactor API (`cusolverRf*`) is
//! under [`refactor`].

#![warn(missing_debug_implementations)]

use core::ffi::{c_int, c_void};
use std::marker::PhantomData;

use baracuda_cusolver_sys::{
    cublasFillMode_t, cublasOperation_t, cuComplex, cuDoubleComplex, cusolver,
    cusolverDnHandle_t, cusolverEigMode_t, cusolverStatus_t,
};
use baracuda_driver::{DeviceBuffer, Stream};
use baracuda_types::{Complex32, Complex64, DeviceRepr};

pub use baracuda_cusolver_sys::{
    cublasFillMode_t as Fill, cusolverEigMode_t as EigMode,
};

/// Error type for cuSOLVER operations.
pub type Error = baracuda_core::Error<cusolverStatus_t>;
/// Result alias.
pub type Result<T, E = Error> = core::result::Result<T, E>;

#[inline]
fn check(status: cusolverStatus_t) -> Result<()> {
    Error::check(status)
}

/// Convert a driver allocation failure into a cuSOLVER ALLOC_FAILED.
fn alloc_fail<E>(_e: E) -> Error {
    Error::Status {
        status: cusolverStatus_t::ALLOC_FAILED,
    }
}

// ---- Handle -------------------------------------------------------------

/// Dense cuSOLVER handle.
pub struct DnHandle {
    handle: cusolverDnHandle_t,
}

unsafe impl Send for DnHandle {}

impl core::fmt::Debug for DnHandle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("cusolver::DnHandle")
            .field("handle", &self.handle)
            .finish()
    }
}

impl DnHandle {
    pub fn new() -> Result<Self> {
        let c = cusolver()?;
        let cu = c.cusolver_dn_create()?;
        let mut h: cusolverDnHandle_t = core::ptr::null_mut();
        check(unsafe { cu(&mut h) })?;
        Ok(Self { handle: h })
    }

    pub fn set_stream(&self, stream: &Stream) -> Result<()> {
        let c = cusolver()?;
        let cu = c.cusolver_dn_set_stream()?;
        check(unsafe { cu(self.handle, stream.as_raw() as _) })
    }

    pub fn version() -> Result<i32> {
        let c = cusolver()?;
        let cu = c.cusolver_get_version()?;
        let mut v: c_int = 0;
        check(unsafe { cu(&mut v) })?;
        Ok(v)
    }

    #[inline]
    pub fn as_raw(&self) -> cusolverDnHandle_t {
        self.handle
    }
}

impl Drop for DnHandle {
    fn drop(&mut self) {
        if let Ok(c) = cusolver() {
            if let Ok(cu) = c.cusolver_dn_destroy() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

/// Transposition selector for solve-step calls.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum Op {
    #[default]
    N,
    T,
    C,
}

impl Op {
    fn raw(self) -> cublasOperation_t {
        match self {
            Op::N => cublasOperation_t::N,
            Op::T => cublasOperation_t::T,
            Op::C => cublasOperation_t::C,
        }
    }
}

// ---- Trait framework ----------------------------------------------------

/// Scalars supported by cuSOLVER's Dn S/D/C/Z API.
pub trait SolverScalar: DeviceRepr + Copy + 'static + sealed::Sealed {
    /// Real-valued associate type for ops that mix scalar and norm
    /// (f32 → f32, f64 → f64, Complex32 → f32, Complex64 → f64).
    type Real: DeviceRepr + Copy + 'static;

    /// LU buffer size.
    #[doc(hidden)]
    unsafe fn getrf_buf(
        h: cusolverDnHandle_t,
        m: c_int,
        n: c_int,
        a: *mut Self,
        lda: c_int,
        lwork: *mut c_int,
    ) -> cusolverStatus_t;

    /// LU factorization.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn getrf(
        h: cusolverDnHandle_t,
        m: c_int,
        n: c_int,
        a: *mut Self,
        lda: c_int,
        workspace: *mut Self,
        ipiv: *mut c_int,
        info: *mut c_int,
    ) -> cusolverStatus_t;

    /// LU solve.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn getrs(
        h: cusolverDnHandle_t,
        trans: cublasOperation_t,
        n: c_int,
        nrhs: c_int,
        a: *const Self,
        lda: c_int,
        ipiv: *const c_int,
        b: *mut Self,
        ldb: c_int,
        info: *mut c_int,
    ) -> cusolverStatus_t;

    /// QR factorization buffer size.
    #[doc(hidden)]
    unsafe fn geqrf_buf(
        h: cusolverDnHandle_t,
        m: c_int,
        n: c_int,
        a: *mut Self,
        lda: c_int,
        lwork: *mut c_int,
    ) -> cusolverStatus_t;

    /// QR factorization.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn geqrf(
        h: cusolverDnHandle_t,
        m: c_int,
        n: c_int,
        a: *mut Self,
        lda: c_int,
        tau: *mut Self,
        workspace: *mut Self,
        lwork: c_int,
        info: *mut c_int,
    ) -> cusolverStatus_t;

    /// Cholesky buffer size.
    #[doc(hidden)]
    unsafe fn potrf_buf(
        h: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: c_int,
        a: *mut Self,
        lda: c_int,
        lwork: *mut c_int,
    ) -> cusolverStatus_t;

    /// Cholesky factorization.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn potrf(
        h: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: c_int,
        a: *mut Self,
        lda: c_int,
        workspace: *mut Self,
        lwork: c_int,
        info: *mut c_int,
    ) -> cusolverStatus_t;

    /// Cholesky solve.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn potrs(
        h: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: c_int,
        nrhs: c_int,
        a: *const Self,
        lda: c_int,
        b: *mut Self,
        ldb: c_int,
        info: *mut c_int,
    ) -> cusolverStatus_t;

    /// SVD buffer size.
    #[doc(hidden)]
    unsafe fn gesvd_buf(
        h: cusolverDnHandle_t,
        m: c_int,
        n: c_int,
        lwork: *mut c_int,
    ) -> cusolverStatus_t;

    /// SVD (generic, taking real-valued S + rwork for complex variants).
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn gesvd(
        h: cusolverDnHandle_t,
        jobu: u8,
        jobvt: u8,
        m: c_int,
        n: c_int,
        a: *mut Self,
        lda: c_int,
        s: *mut Self::Real,
        u: *mut Self,
        ldu: c_int,
        vt: *mut Self,
        ldvt: c_int,
        work: *mut Self,
        lwork: c_int,
        rwork: *mut Self::Real,
        info: *mut c_int,
    ) -> cusolverStatus_t;

    /// syevd / heevd buffer size.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn syevd_buf(
        h: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: c_int,
        a: *const Self,
        lda: c_int,
        w: *const Self::Real,
        lwork: *mut c_int,
    ) -> cusolverStatus_t;

    /// syevd / heevd.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn syevd(
        h: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: c_int,
        a: *mut Self,
        lda: c_int,
        w: *mut Self::Real,
        work: *mut Self,
        lwork: c_int,
        info: *mut c_int,
    ) -> cusolverStatus_t;
}

mod sealed {
    use baracuda_types::{Complex32, Complex64};
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for Complex32 {}
    impl Sealed for Complex64 {}
}

macro_rules! real_impl {
    ($t:ty, $getrf_buf:ident, $getrf:ident, $getrs:ident,
           $geqrf_buf:ident, $geqrf:ident,
           $potrf_buf:ident, $potrf:ident, $potrs:ident,
           $gesvd_buf:ident, $gesvd:ident,
           $syevd_buf:ident, $syevd:ident) => {
        impl SolverScalar for $t {
            type Real = $t;

            unsafe fn getrf_buf(
                h: cusolverDnHandle_t,
                m: c_int,
                n: c_int,
                a: *mut $t,
                lda: c_int,
                lwork: *mut c_int,
            ) -> cusolverStatus_t {
                match cusolver().and_then(|c| c.$getrf_buf()) {
                    Ok(f) => f(h, m, n, a, lda, lwork),
                    Err(_) => cusolverStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn getrf(
                h: cusolverDnHandle_t,
                m: c_int,
                n: c_int,
                a: *mut $t,
                lda: c_int,
                work: *mut $t,
                ipiv: *mut c_int,
                info: *mut c_int,
            ) -> cusolverStatus_t {
                match cusolver().and_then(|c| c.$getrf()) {
                    Ok(f) => f(h, m, n, a, lda, work, ipiv, info),
                    Err(_) => cusolverStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn getrs(
                h: cusolverDnHandle_t,
                trans: cublasOperation_t,
                n: c_int,
                nrhs: c_int,
                a: *const $t,
                lda: c_int,
                ipiv: *const c_int,
                b: *mut $t,
                ldb: c_int,
                info: *mut c_int,
            ) -> cusolverStatus_t {
                match cusolver().and_then(|c| c.$getrs()) {
                    Ok(f) => f(h, trans, n, nrhs, a, lda, ipiv, b, ldb, info),
                    Err(_) => cusolverStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn geqrf_buf(
                h: cusolverDnHandle_t,
                m: c_int,
                n: c_int,
                a: *mut $t,
                lda: c_int,
                lwork: *mut c_int,
            ) -> cusolverStatus_t {
                match cusolver().and_then(|c| c.$geqrf_buf()) {
                    Ok(f) => f(h, m, n, a, lda, lwork),
                    Err(_) => cusolverStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn geqrf(
                h: cusolverDnHandle_t,
                m: c_int,
                n: c_int,
                a: *mut $t,
                lda: c_int,
                tau: *mut $t,
                work: *mut $t,
                lwork: c_int,
                info: *mut c_int,
            ) -> cusolverStatus_t {
                match cusolver().and_then(|c| c.$geqrf()) {
                    Ok(f) => f(h, m, n, a, lda, tau, work, lwork, info),
                    Err(_) => cusolverStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn potrf_buf(
                h: cusolverDnHandle_t,
                uplo: cublasFillMode_t,
                n: c_int,
                a: *mut $t,
                lda: c_int,
                lwork: *mut c_int,
            ) -> cusolverStatus_t {
                match cusolver().and_then(|c| c.$potrf_buf()) {
                    Ok(f) => f(h, uplo, n, a, lda, lwork),
                    Err(_) => cusolverStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn potrf(
                h: cusolverDnHandle_t,
                uplo: cublasFillMode_t,
                n: c_int,
                a: *mut $t,
                lda: c_int,
                work: *mut $t,
                lwork: c_int,
                info: *mut c_int,
            ) -> cusolverStatus_t {
                match cusolver().and_then(|c| c.$potrf()) {
                    Ok(f) => f(h, uplo, n, a, lda, work, lwork, info),
                    Err(_) => cusolverStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn potrs(
                h: cusolverDnHandle_t,
                uplo: cublasFillMode_t,
                n: c_int,
                nrhs: c_int,
                a: *const $t,
                lda: c_int,
                b: *mut $t,
                ldb: c_int,
                info: *mut c_int,
            ) -> cusolverStatus_t {
                match cusolver().and_then(|c| c.$potrs()) {
                    Ok(f) => f(h, uplo, n, nrhs, a, lda, b, ldb, info),
                    Err(_) => cusolverStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn gesvd_buf(
                h: cusolverDnHandle_t,
                m: c_int,
                n: c_int,
                lwork: *mut c_int,
            ) -> cusolverStatus_t {
                match cusolver().and_then(|c| c.$gesvd_buf()) {
                    Ok(f) => f(h, m, n, lwork),
                    Err(_) => cusolverStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn gesvd(
                h: cusolverDnHandle_t,
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
            ) -> cusolverStatus_t {
                match cusolver().and_then(|c| c.$gesvd()) {
                    Ok(f) => f(
                        h, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info,
                    ),
                    Err(_) => cusolverStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn syevd_buf(
                h: cusolverDnHandle_t,
                jobz: cusolverEigMode_t,
                uplo: cublasFillMode_t,
                n: c_int,
                a: *const $t,
                lda: c_int,
                w: *const $t,
                lwork: *mut c_int,
            ) -> cusolverStatus_t {
                match cusolver().and_then(|c| c.$syevd_buf()) {
                    Ok(f) => f(h, jobz, uplo, n, a, lda, w, lwork),
                    Err(_) => cusolverStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn syevd(
                h: cusolverDnHandle_t,
                jobz: cusolverEigMode_t,
                uplo: cublasFillMode_t,
                n: c_int,
                a: *mut $t,
                lda: c_int,
                w: *mut $t,
                work: *mut $t,
                lwork: c_int,
                info: *mut c_int,
            ) -> cusolverStatus_t {
                match cusolver().and_then(|c| c.$syevd()) {
                    Ok(f) => f(h, jobz, uplo, n, a, lda, w, work, lwork, info),
                    Err(_) => cusolverStatus_t::NOT_INITIALIZED,
                }
            }
        }
    };
}

macro_rules! complex_impl {
    ($t:ty, $real:ty, $raw:ty,
     $getrf_buf:ident, $getrf:ident, $getrs:ident,
     $geqrf_buf:ident, $geqrf:ident,
     $potrf_buf:ident, $potrf:ident, $potrs:ident,
     $gesvd_buf:ident, $gesvd:ident,
     $heevd_buf:ident, $heevd:ident) => {
        impl SolverScalar for $t {
            type Real = $real;

            unsafe fn getrf_buf(
                h: cusolverDnHandle_t,
                m: c_int,
                n: c_int,
                a: *mut $t,
                lda: c_int,
                lwork: *mut c_int,
            ) -> cusolverStatus_t {
                match cusolver().and_then(|c| c.$getrf_buf()) {
                    Ok(f) => f(h, m, n, a as *mut $raw, lda, lwork),
                    Err(_) => cusolverStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn getrf(
                h: cusolverDnHandle_t,
                m: c_int,
                n: c_int,
                a: *mut $t,
                lda: c_int,
                work: *mut $t,
                ipiv: *mut c_int,
                info: *mut c_int,
            ) -> cusolverStatus_t {
                match cusolver().and_then(|c| c.$getrf()) {
                    Ok(f) => f(
                        h,
                        m,
                        n,
                        a as *mut $raw,
                        lda,
                        work as *mut $raw,
                        ipiv,
                        info,
                    ),
                    Err(_) => cusolverStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn getrs(
                h: cusolverDnHandle_t,
                trans: cublasOperation_t,
                n: c_int,
                nrhs: c_int,
                a: *const $t,
                lda: c_int,
                ipiv: *const c_int,
                b: *mut $t,
                ldb: c_int,
                info: *mut c_int,
            ) -> cusolverStatus_t {
                match cusolver().and_then(|c| c.$getrs()) {
                    Ok(f) => f(
                        h,
                        trans,
                        n,
                        nrhs,
                        a as *const $raw,
                        lda,
                        ipiv,
                        b as *mut $raw,
                        ldb,
                        info,
                    ),
                    Err(_) => cusolverStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn geqrf_buf(
                h: cusolverDnHandle_t,
                m: c_int,
                n: c_int,
                a: *mut $t,
                lda: c_int,
                lwork: *mut c_int,
            ) -> cusolverStatus_t {
                match cusolver().and_then(|c| c.$geqrf_buf()) {
                    Ok(f) => f(h, m, n, a as *mut $raw, lda, lwork),
                    Err(_) => cusolverStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn geqrf(
                h: cusolverDnHandle_t,
                m: c_int,
                n: c_int,
                a: *mut $t,
                lda: c_int,
                tau: *mut $t,
                work: *mut $t,
                lwork: c_int,
                info: *mut c_int,
            ) -> cusolverStatus_t {
                match cusolver().and_then(|c| c.$geqrf()) {
                    Ok(f) => f(
                        h,
                        m,
                        n,
                        a as *mut $raw,
                        lda,
                        tau as *mut $raw,
                        work as *mut $raw,
                        lwork,
                        info,
                    ),
                    Err(_) => cusolverStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn potrf_buf(
                h: cusolverDnHandle_t,
                uplo: cublasFillMode_t,
                n: c_int,
                a: *mut $t,
                lda: c_int,
                lwork: *mut c_int,
            ) -> cusolverStatus_t {
                match cusolver().and_then(|c| c.$potrf_buf()) {
                    Ok(f) => f(h, uplo, n, a as *mut $raw, lda, lwork),
                    Err(_) => cusolverStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn potrf(
                h: cusolverDnHandle_t,
                uplo: cublasFillMode_t,
                n: c_int,
                a: *mut $t,
                lda: c_int,
                work: *mut $t,
                lwork: c_int,
                info: *mut c_int,
            ) -> cusolverStatus_t {
                match cusolver().and_then(|c| c.$potrf()) {
                    Ok(f) => f(
                        h,
                        uplo,
                        n,
                        a as *mut $raw,
                        lda,
                        work as *mut $raw,
                        lwork,
                        info,
                    ),
                    Err(_) => cusolverStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn potrs(
                h: cusolverDnHandle_t,
                uplo: cublasFillMode_t,
                n: c_int,
                nrhs: c_int,
                a: *const $t,
                lda: c_int,
                b: *mut $t,
                ldb: c_int,
                info: *mut c_int,
            ) -> cusolverStatus_t {
                match cusolver().and_then(|c| c.$potrs()) {
                    Ok(f) => f(
                        h,
                        uplo,
                        n,
                        nrhs,
                        a as *const $raw,
                        lda,
                        b as *mut $raw,
                        ldb,
                        info,
                    ),
                    Err(_) => cusolverStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn gesvd_buf(
                h: cusolverDnHandle_t,
                m: c_int,
                n: c_int,
                lwork: *mut c_int,
            ) -> cusolverStatus_t {
                match cusolver().and_then(|c| c.$gesvd_buf()) {
                    Ok(f) => f(h, m, n, lwork),
                    Err(_) => cusolverStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn gesvd(
                h: cusolverDnHandle_t,
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
            ) -> cusolverStatus_t {
                match cusolver().and_then(|c| c.$gesvd()) {
                    Ok(f) => f(
                        h,
                        jobu,
                        jobvt,
                        m,
                        n,
                        a as *mut $raw,
                        lda,
                        s,
                        u as *mut $raw,
                        ldu,
                        vt as *mut $raw,
                        ldvt,
                        work as *mut $raw,
                        lwork,
                        rwork,
                        info,
                    ),
                    Err(_) => cusolverStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn syevd_buf(
                h: cusolverDnHandle_t,
                jobz: cusolverEigMode_t,
                uplo: cublasFillMode_t,
                n: c_int,
                a: *const $t,
                lda: c_int,
                w: *const $real,
                lwork: *mut c_int,
            ) -> cusolverStatus_t {
                match cusolver().and_then(|c| c.$heevd_buf()) {
                    Ok(f) => f(h, jobz, uplo, n, a as *const $raw, lda, w, lwork),
                    Err(_) => cusolverStatus_t::NOT_INITIALIZED,
                }
            }
            unsafe fn syevd(
                h: cusolverDnHandle_t,
                jobz: cusolverEigMode_t,
                uplo: cublasFillMode_t,
                n: c_int,
                a: *mut $t,
                lda: c_int,
                w: *mut $real,
                work: *mut $t,
                lwork: c_int,
                info: *mut c_int,
            ) -> cusolverStatus_t {
                match cusolver().and_then(|c| c.$heevd()) {
                    Ok(f) => f(
                        h,
                        jobz,
                        uplo,
                        n,
                        a as *mut $raw,
                        lda,
                        w,
                        work as *mut $raw,
                        lwork,
                        info,
                    ),
                    Err(_) => cusolverStatus_t::NOT_INITIALIZED,
                }
            }
        }
    };
}

real_impl!(
    f32,
    cusolver_dn_sgetrf_buffer_size,
    cusolver_dn_sgetrf,
    cusolver_dn_sgetrs,
    cusolver_dn_sgeqrf_buffer_size,
    cusolver_dn_sgeqrf,
    cusolver_dn_spotrf_buffer_size,
    cusolver_dn_spotrf,
    cusolver_dn_spotrs,
    cusolver_dn_sgesvd_buffer_size,
    cusolver_dn_sgesvd,
    cusolver_dn_ssyevd_buffer_size,
    cusolver_dn_ssyevd
);

real_impl!(
    f64,
    cusolver_dn_dgetrf_buffer_size,
    cusolver_dn_dgetrf,
    cusolver_dn_dgetrs,
    cusolver_dn_dgeqrf_buffer_size,
    cusolver_dn_dgeqrf,
    cusolver_dn_dpotrf_buffer_size,
    cusolver_dn_dpotrf,
    cusolver_dn_dpotrs,
    cusolver_dn_dgesvd_buffer_size,
    cusolver_dn_dgesvd,
    cusolver_dn_dsyevd_buffer_size,
    cusolver_dn_dsyevd
);

complex_impl!(
    Complex32,
    f32,
    cuComplex,
    cusolver_dn_cgetrf_buffer_size,
    cusolver_dn_cgetrf,
    cusolver_dn_cgetrs,
    cusolver_dn_cgeqrf_buffer_size,
    cusolver_dn_cgeqrf,
    cusolver_dn_cpotrf_buffer_size,
    cusolver_dn_cpotrf,
    cusolver_dn_cpotrs,
    cusolver_dn_cgesvd_buffer_size,
    cusolver_dn_cgesvd,
    cusolver_dn_cheevd_buffer_size,
    cusolver_dn_cheevd
);

complex_impl!(
    Complex64,
    f64,
    cuDoubleComplex,
    cusolver_dn_zgetrf_buffer_size,
    cusolver_dn_zgetrf,
    cusolver_dn_zgetrs,
    cusolver_dn_zgeqrf_buffer_size,
    cusolver_dn_zgeqrf,
    cusolver_dn_zpotrf_buffer_size,
    cusolver_dn_zpotrf,
    cusolver_dn_zpotrs,
    cusolver_dn_zgesvd_buffer_size,
    cusolver_dn_zgesvd,
    cusolver_dn_zheevd_buffer_size,
    cusolver_dn_zheevd
);

// ---- Public API ---------------------------------------------------------

/// In-place LU factorization of a column-major matrix. Overwrites `a`.
#[allow(clippy::too_many_arguments)]
pub fn getrf<T: SolverScalar>(
    handle: &DnHandle,
    m: i32,
    n: i32,
    a: &mut DeviceBuffer<T>,
    lda: i32,
    ipiv: &mut DeviceBuffer<i32>,
    info: &mut DeviceBuffer<i32>,
) -> Result<()> {
    let mut lwork: c_int = 0;
    check(unsafe { T::getrf_buf(handle.handle, m, n, a.as_raw().0 as *mut T, lda, &mut lwork) })?;
    let workspace =
        DeviceBuffer::<T>::new(a.context(), lwork as usize).map_err(alloc_fail)?;
    check(unsafe {
        T::getrf(
            handle.handle,
            m,
            n,
            a.as_raw().0 as *mut T,
            lda,
            workspace.as_raw().0 as *mut T,
            ipiv.as_raw().0 as *mut c_int,
            info.as_raw().0 as *mut c_int,
        )
    })
}

/// Solve `op(A) * X = B` using the LU factorization from [`getrf`].
#[allow(clippy::too_many_arguments)]
pub fn getrs<T: SolverScalar>(
    handle: &DnHandle,
    trans: Op,
    n: i32,
    nrhs: i32,
    a: &DeviceBuffer<T>,
    lda: i32,
    ipiv: &DeviceBuffer<i32>,
    b: &mut DeviceBuffer<T>,
    ldb: i32,
    info: &mut DeviceBuffer<i32>,
) -> Result<()> {
    check(unsafe {
        T::getrs(
            handle.handle,
            trans.raw(),
            n,
            nrhs,
            a.as_raw().0 as *const T,
            lda,
            ipiv.as_raw().0 as *const c_int,
            b.as_raw().0 as *mut T,
            ldb,
            info.as_raw().0 as *mut c_int,
        )
    })
}

/// QR factorization: `A = Q * R`. Overwrites `a` (upper triangle = R,
/// lower = Householder reflectors); `tau` receives reflector scalars.
#[allow(clippy::too_many_arguments)]
pub fn geqrf<T: SolverScalar>(
    handle: &DnHandle,
    m: i32,
    n: i32,
    a: &mut DeviceBuffer<T>,
    lda: i32,
    tau: &mut DeviceBuffer<T>,
    info: &mut DeviceBuffer<i32>,
) -> Result<()> {
    let mut lwork: c_int = 0;
    check(unsafe { T::geqrf_buf(handle.handle, m, n, a.as_raw().0 as *mut T, lda, &mut lwork) })?;
    let workspace =
        DeviceBuffer::<T>::new(a.context(), lwork as usize).map_err(alloc_fail)?;
    check(unsafe {
        T::geqrf(
            handle.handle,
            m,
            n,
            a.as_raw().0 as *mut T,
            lda,
            tau.as_raw().0 as *mut T,
            workspace.as_raw().0 as *mut T,
            lwork,
            info.as_raw().0 as *mut c_int,
        )
    })
}

/// Cholesky factorization: `A = L * Lᵀ` (or `Uᵀ * U`). Overwrites `a`.
pub fn potrf<T: SolverScalar>(
    handle: &DnHandle,
    uplo: Fill,
    n: i32,
    a: &mut DeviceBuffer<T>,
    lda: i32,
    info: &mut DeviceBuffer<i32>,
) -> Result<()> {
    let mut lwork: c_int = 0;
    check(unsafe { T::potrf_buf(handle.handle, uplo, n, a.as_raw().0 as *mut T, lda, &mut lwork) })?;
    let workspace =
        DeviceBuffer::<T>::new(a.context(), lwork as usize).map_err(alloc_fail)?;
    check(unsafe {
        T::potrf(
            handle.handle,
            uplo,
            n,
            a.as_raw().0 as *mut T,
            lda,
            workspace.as_raw().0 as *mut T,
            lwork,
            info.as_raw().0 as *mut c_int,
        )
    })
}

/// Solve `A * X = B` using the Cholesky factorization from [`potrf`].
#[allow(clippy::too_many_arguments)]
pub fn potrs<T: SolverScalar>(
    handle: &DnHandle,
    uplo: Fill,
    n: i32,
    nrhs: i32,
    a: &DeviceBuffer<T>,
    lda: i32,
    b: &mut DeviceBuffer<T>,
    ldb: i32,
    info: &mut DeviceBuffer<i32>,
) -> Result<()> {
    check(unsafe {
        T::potrs(
            handle.handle,
            uplo,
            n,
            nrhs,
            a.as_raw().0 as *const T,
            lda,
            b.as_raw().0 as *mut T,
            ldb,
            info.as_raw().0 as *mut c_int,
        )
    })
}

/// Full SVD: `A = U * Σ * Vᵀ`. `jobu`/`jobvt` are LAPACK-style single-byte
/// selectors (b'A' = all, b'S' = economy, b'N' = none, b'O' = overwrite A).
///
/// `rwork` must be provided for complex element types; pass an empty buffer
/// for real types (pointer is still non-null; cuSOLVER ignores it).
#[allow(clippy::too_many_arguments)]
pub fn gesvd<T: SolverScalar>(
    handle: &DnHandle,
    jobu: u8,
    jobvt: u8,
    m: i32,
    n: i32,
    a: &mut DeviceBuffer<T>,
    lda: i32,
    s: &mut DeviceBuffer<T::Real>,
    u: &mut DeviceBuffer<T>,
    ldu: i32,
    vt: &mut DeviceBuffer<T>,
    ldvt: i32,
    rwork: &mut DeviceBuffer<T::Real>,
    info: &mut DeviceBuffer<i32>,
) -> Result<()> {
    let mut lwork: c_int = 0;
    check(unsafe { T::gesvd_buf(handle.handle, m, n, &mut lwork) })?;
    let workspace =
        DeviceBuffer::<T>::new(a.context(), lwork as usize).map_err(alloc_fail)?;
    check(unsafe {
        T::gesvd(
            handle.handle,
            jobu,
            jobvt,
            m,
            n,
            a.as_raw().0 as *mut T,
            lda,
            s.as_raw().0 as *mut T::Real,
            u.as_raw().0 as *mut T,
            ldu,
            vt.as_raw().0 as *mut T,
            ldvt,
            workspace.as_raw().0 as *mut T,
            lwork,
            rwork.as_raw().0 as *mut T::Real,
            info.as_raw().0 as *mut c_int,
        )
    })
}

/// Symmetric / Hermitian eigenvalue decomposition: `A = Q * diag(w) * Qᵀ`.
#[allow(clippy::too_many_arguments)]
pub fn syevd<T: SolverScalar>(
    handle: &DnHandle,
    jobz: EigMode,
    uplo: Fill,
    n: i32,
    a: &mut DeviceBuffer<T>,
    lda: i32,
    w: &mut DeviceBuffer<T::Real>,
    info: &mut DeviceBuffer<i32>,
) -> Result<()> {
    let mut lwork: c_int = 0;
    check(unsafe {
        T::syevd_buf(
            handle.handle,
            jobz,
            uplo,
            n,
            a.as_raw().0 as *const T,
            lda,
            w.as_raw().0 as *const T::Real,
            &mut lwork,
        )
    })?;
    let workspace =
        DeviceBuffer::<T>::new(a.context(), lwork as usize).map_err(alloc_fail)?;
    check(unsafe {
        T::syevd(
            handle.handle,
            jobz,
            uplo,
            n,
            a.as_raw().0 as *mut T,
            lda,
            w.as_raw().0 as *mut T::Real,
            workspace.as_raw().0 as *mut T,
            lwork,
            info.as_raw().0 as *mut c_int,
        )
    })
}

// ---- Jacobi-based solvers (syevj / gesvdj) -----------------------------

pub use baracuda_cusolver_sys::{gesvdjInfo_t as GesvdjInfoRaw, syevjInfo_t as SyevjInfoRaw};

/// Jacobi-eigen tuning handle (tolerance + max sweeps).
#[derive(Debug)]
pub struct SyevjInfo {
    raw: SyevjInfoRaw,
}

impl SyevjInfo {
    pub fn new() -> Result<Self> {
        let c = cusolver()?;
        let cu = c.cusolver_dn_create_syevj_info()?;
        let mut raw: SyevjInfoRaw = core::ptr::null_mut();
        check(unsafe { cu(&mut raw) })?;
        Ok(Self { raw })
    }

    pub fn set_tolerance(&self, tol: f64) -> Result<()> {
        let c = cusolver()?;
        let cu = c.cusolver_dn_xsyevj_set_tolerance()?;
        check(unsafe { cu(self.raw, tol) })
    }

    pub fn set_max_sweeps(&self, n: i32) -> Result<()> {
        let c = cusolver()?;
        let cu = c.cusolver_dn_xsyevj_set_max_sweeps()?;
        check(unsafe { cu(self.raw, n) })
    }

    pub fn as_raw(&self) -> SyevjInfoRaw {
        self.raw
    }
}

impl Drop for SyevjInfo {
    fn drop(&mut self) {
        if let Ok(c) = cusolver() {
            if let Ok(cu) = c.cusolver_dn_destroy_syevj_info() {
                let _ = unsafe { cu(self.raw) };
            }
        }
    }
}

/// Jacobi-SVD tuning handle.
#[derive(Debug)]
pub struct GesvdjInfo {
    raw: GesvdjInfoRaw,
}

impl GesvdjInfo {
    pub fn new() -> Result<Self> {
        let c = cusolver()?;
        let cu = c.cusolver_dn_create_gesvdj_info()?;
        let mut raw: GesvdjInfoRaw = core::ptr::null_mut();
        check(unsafe { cu(&mut raw) })?;
        Ok(Self { raw })
    }

    pub fn as_raw(&self) -> GesvdjInfoRaw {
        self.raw
    }
}

impl Drop for GesvdjInfo {
    fn drop(&mut self) {
        if let Ok(c) = cusolver() {
            if let Ok(cu) = c.cusolver_dn_destroy_gesvdj_info() {
                let _ = unsafe { cu(self.raw) };
            }
        }
    }
}

/// Jacobi symmetric/Hermitian eigendecomposition (smaller matrices than
/// [`syevd`], faster convergence on well-conditioned problems).
#[allow(clippy::too_many_arguments)]
pub fn syevj<T: SolverScalar>(
    handle: &DnHandle,
    jobz: EigMode,
    uplo: Fill,
    n: i32,
    a: &mut DeviceBuffer<T>,
    lda: i32,
    w: &mut DeviceBuffer<T::Real>,
    info: &mut DeviceBuffer<i32>,
    params: &SyevjInfo,
) -> Result<()> {
    use baracuda_cusolver_sys::{
        cuComplex, cuDoubleComplex,
    };
    use core::mem;

    let mut lwork: c_int = 0;

    // Dispatch is simpler done via a type check, since syevj doesn't share
    // the generic trait shape (extra params: SyevjInfo).
    macro_rules! dispatch_real {
        ($t:ty, $bufsize:ident, $solve:ident) => {{
            let c = cusolver()?;
            check(unsafe {
                (c.$bufsize()?)(
                    handle.as_raw(),
                    jobz,
                    uplo,
                    n,
                    a.as_raw().0 as *const $t,
                    lda,
                    w.as_raw().0 as *const $t,
                    &mut lwork,
                    params.raw,
                )
            })?;
            let workspace =
                DeviceBuffer::<T>::new(a.context(), lwork as usize).map_err(alloc_fail)?;
            check(unsafe {
                (c.$solve()?)(
                    handle.as_raw(),
                    jobz,
                    uplo,
                    n,
                    a.as_raw().0 as *mut $t,
                    lda,
                    w.as_raw().0 as *mut $t,
                    workspace.as_raw().0 as *mut $t,
                    lwork,
                    info.as_raw().0 as *mut c_int,
                    params.raw,
                )
            })
        }};
    }
    macro_rules! dispatch_complex {
        ($t:ty, $real:ty, $raw:ty, $bufsize:ident, $solve:ident) => {{
            let c = cusolver()?;
            check(unsafe {
                (c.$bufsize()?)(
                    handle.as_raw(),
                    jobz,
                    uplo,
                    n,
                    a.as_raw().0 as *const $raw,
                    lda,
                    w.as_raw().0 as *const $real,
                    &mut lwork,
                    params.raw,
                )
            })?;
            let workspace =
                DeviceBuffer::<T>::new(a.context(), lwork as usize).map_err(alloc_fail)?;
            check(unsafe {
                (c.$solve()?)(
                    handle.as_raw(),
                    jobz,
                    uplo,
                    n,
                    a.as_raw().0 as *mut $raw,
                    lda,
                    w.as_raw().0 as *mut $real,
                    workspace.as_raw().0 as *mut $raw,
                    lwork,
                    info.as_raw().0 as *mut c_int,
                    params.raw,
                )
            })
        }};
    }

    if mem::size_of::<T>() == mem::size_of::<f32>() && mem::size_of::<T::Real>() == 4 {
        dispatch_real!(f32, cusolver_dn_ssyevj_buffer_size, cusolver_dn_ssyevj)
    } else if mem::size_of::<T>() == mem::size_of::<f64>() && mem::size_of::<T::Real>() == 8 {
        dispatch_real!(f64, cusolver_dn_dsyevj_buffer_size, cusolver_dn_dsyevj)
    } else if mem::size_of::<T>() == mem::size_of::<Complex32>() {
        dispatch_complex!(
            Complex32,
            f32,
            cuComplex,
            cusolver_dn_cheevj_buffer_size,
            cusolver_dn_cheevj
        )
    } else {
        dispatch_complex!(
            Complex64,
            f64,
            cuDoubleComplex,
            cusolver_dn_zheevj_buffer_size,
            cusolver_dn_zheevj
        )
    }
}

/// Jacobi SVD: `A = U * diag(s) * Vᴴ`. `econ` selects thin-SVD when set.
#[allow(clippy::too_many_arguments)]
pub fn gesvdj<T: SolverScalar>(
    handle: &DnHandle,
    jobz: EigMode,
    econ: bool,
    m: i32,
    n: i32,
    a: &mut DeviceBuffer<T>,
    lda: i32,
    s: &mut DeviceBuffer<T::Real>,
    u: &mut DeviceBuffer<T>,
    ldu: i32,
    v: &mut DeviceBuffer<T>,
    ldv: i32,
    info: &mut DeviceBuffer<i32>,
    params: &GesvdjInfo,
) -> Result<()> {
    use baracuda_cusolver_sys::{cuComplex, cuDoubleComplex};
    use core::mem;

    let mut lwork: c_int = 0;
    let econ_i = if econ { 1 } else { 0 };

    macro_rules! dispatch_real {
        ($t:ty, $bufsize:ident, $solve:ident) => {{
            let c = cusolver()?;
            check(unsafe {
                (c.$bufsize()?)(
                    handle.as_raw(),
                    jobz,
                    econ_i,
                    m,
                    n,
                    a.as_raw().0 as *const $t,
                    lda,
                    s.as_raw().0 as *const $t,
                    u.as_raw().0 as *const $t,
                    ldu,
                    v.as_raw().0 as *const $t,
                    ldv,
                    &mut lwork,
                    params.raw,
                )
            })?;
            let workspace =
                DeviceBuffer::<T>::new(a.context(), lwork as usize).map_err(alloc_fail)?;
            check(unsafe {
                (c.$solve()?)(
                    handle.as_raw(),
                    jobz,
                    econ_i,
                    m,
                    n,
                    a.as_raw().0 as *mut $t,
                    lda,
                    s.as_raw().0 as *mut $t,
                    u.as_raw().0 as *mut $t,
                    ldu,
                    v.as_raw().0 as *mut $t,
                    ldv,
                    workspace.as_raw().0 as *mut $t,
                    lwork,
                    info.as_raw().0 as *mut c_int,
                    params.raw,
                )
            })
        }};
    }
    macro_rules! dispatch_complex {
        ($t:ty, $real:ty, $raw:ty, $bufsize:ident, $solve:ident) => {{
            let c = cusolver()?;
            check(unsafe {
                (c.$bufsize()?)(
                    handle.as_raw(),
                    jobz,
                    econ_i,
                    m,
                    n,
                    a.as_raw().0 as *const $raw,
                    lda,
                    s.as_raw().0 as *const $real,
                    u.as_raw().0 as *const $raw,
                    ldu,
                    v.as_raw().0 as *const $raw,
                    ldv,
                    &mut lwork,
                    params.raw,
                )
            })?;
            let workspace =
                DeviceBuffer::<T>::new(a.context(), lwork as usize).map_err(alloc_fail)?;
            check(unsafe {
                (c.$solve()?)(
                    handle.as_raw(),
                    jobz,
                    econ_i,
                    m,
                    n,
                    a.as_raw().0 as *mut $raw,
                    lda,
                    s.as_raw().0 as *mut $real,
                    u.as_raw().0 as *mut $raw,
                    ldu,
                    v.as_raw().0 as *mut $raw,
                    ldv,
                    workspace.as_raw().0 as *mut $raw,
                    lwork,
                    info.as_raw().0 as *mut c_int,
                    params.raw,
                )
            })
        }};
    }

    if mem::size_of::<T>() == mem::size_of::<f32>() && mem::size_of::<T::Real>() == 4 {
        dispatch_real!(f32, cusolver_dn_sgesvdj_buffer_size, cusolver_dn_sgesvdj)
    } else if mem::size_of::<T>() == mem::size_of::<f64>() && mem::size_of::<T::Real>() == 8 {
        dispatch_real!(f64, cusolver_dn_dgesvdj_buffer_size, cusolver_dn_dgesvdj)
    } else if mem::size_of::<T>() == mem::size_of::<Complex32>() {
        dispatch_complex!(
            Complex32,
            f32,
            cuComplex,
            cusolver_dn_cgesvdj_buffer_size,
            cusolver_dn_cgesvdj
        )
    } else {
        dispatch_complex!(
            Complex64,
            f64,
            cuDoubleComplex,
            cusolver_dn_zgesvdj_buffer_size,
            cusolver_dn_zgesvdj
        )
    }
}

// ---- Generate / apply Q from QR (orgqr / ormqr) -------------------------

/// Generate the orthogonal matrix `Q` from the factorization produced by
/// [`geqrf`]. After this, `a` holds the first `n` columns of `Q`.
#[allow(clippy::too_many_arguments)]
pub fn orgqr<T: SolverScalar>(
    handle: &DnHandle,
    m: i32,
    n: i32,
    k: i32,
    a: &mut DeviceBuffer<T>,
    lda: i32,
    tau: &DeviceBuffer<T>,
    info: &mut DeviceBuffer<i32>,
) -> Result<()> {
    use baracuda_cusolver_sys::{cuComplex, cuDoubleComplex};
    use core::mem;

    let mut lwork: c_int = 0;
    macro_rules! dispatch {
        ($t:ty, $raw:ty, $bufsize:ident, $solve:ident) => {{
            let c = cusolver()?;
            check(unsafe {
                (c.$bufsize()?)(
                    handle.as_raw(),
                    m,
                    n,
                    k,
                    a.as_raw().0 as *const $raw,
                    lda,
                    tau.as_raw().0 as *const $raw,
                    &mut lwork,
                )
            })?;
            let workspace =
                DeviceBuffer::<T>::new(a.context(), lwork as usize).map_err(alloc_fail)?;
            check(unsafe {
                (c.$solve()?)(
                    handle.as_raw(),
                    m,
                    n,
                    k,
                    a.as_raw().0 as *mut $raw,
                    lda,
                    tau.as_raw().0 as *const $raw,
                    workspace.as_raw().0 as *mut $raw,
                    lwork,
                    info.as_raw().0 as *mut c_int,
                )
            })
        }};
    }

    if mem::size_of::<T>() == mem::size_of::<f32>() && mem::size_of::<T::Real>() == 4 {
        dispatch!(f32, f32, cusolver_dn_sorgqr_buffer_size, cusolver_dn_sorgqr)
    } else if mem::size_of::<T>() == mem::size_of::<f64>() && mem::size_of::<T::Real>() == 8 {
        dispatch!(f64, f64, cusolver_dn_dorgqr_buffer_size, cusolver_dn_dorgqr)
    } else if mem::size_of::<T>() == mem::size_of::<Complex32>() {
        dispatch!(
            Complex32,
            cuComplex,
            cusolver_dn_cungqr_buffer_size,
            cusolver_dn_cungqr
        )
    } else {
        dispatch!(
            Complex64,
            cuDoubleComplex,
            cusolver_dn_zungqr_buffer_size,
            cusolver_dn_zungqr
        )
    }
}

/// Side argument for [`ormqr`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Side {
    Left,
    Right,
}

impl Side {
    fn raw(self) -> core::ffi::c_int {
        match self {
            Side::Left => 0,
            Side::Right => 1,
        }
    }
}

/// Apply `op(Q)` to `C`: `C = op(Q) * C` (Left) or `C = C * op(Q)` (Right),
/// where `Q` is packed in `a`+`tau` from [`geqrf`].
#[allow(clippy::too_many_arguments)]
pub fn ormqr<T: SolverScalar>(
    handle: &DnHandle,
    side: Side,
    trans: Op,
    m: i32,
    n: i32,
    k: i32,
    a: &DeviceBuffer<T>,
    lda: i32,
    tau: &DeviceBuffer<T>,
    c_mat: &mut DeviceBuffer<T>,
    ldc: i32,
    info: &mut DeviceBuffer<i32>,
) -> Result<()> {
    use baracuda_cusolver_sys::{cuComplex, cuDoubleComplex};
    use core::mem;

    let mut lwork: c_int = 0;
    let side_i = side.raw();
    macro_rules! dispatch {
        ($t:ty, $raw:ty, $bufsize:ident, $solve:ident) => {{
            let ca = cusolver()?;
            check(unsafe {
                (ca.$bufsize()?)(
                    handle.as_raw(),
                    side_i,
                    trans.raw(),
                    m,
                    n,
                    k,
                    a.as_raw().0 as *const $raw,
                    lda,
                    tau.as_raw().0 as *const $raw,
                    c_mat.as_raw().0 as *const $raw,
                    ldc,
                    &mut lwork,
                )
            })?;
            let workspace =
                DeviceBuffer::<T>::new(c_mat.context(), lwork as usize).map_err(alloc_fail)?;
            check(unsafe {
                (ca.$solve()?)(
                    handle.as_raw(),
                    side_i,
                    trans.raw(),
                    m,
                    n,
                    k,
                    a.as_raw().0 as *const $raw,
                    lda,
                    tau.as_raw().0 as *const $raw,
                    c_mat.as_raw().0 as *mut $raw,
                    ldc,
                    workspace.as_raw().0 as *mut $raw,
                    lwork,
                    info.as_raw().0 as *mut c_int,
                )
            })
        }};
    }

    if mem::size_of::<T>() == mem::size_of::<f32>() && mem::size_of::<T::Real>() == 4 {
        dispatch!(f32, f32, cusolver_dn_sormqr_buffer_size, cusolver_dn_sormqr)
    } else if mem::size_of::<T>() == mem::size_of::<f64>() && mem::size_of::<T::Real>() == 8 {
        dispatch!(f64, f64, cusolver_dn_dormqr_buffer_size, cusolver_dn_dormqr)
    } else if mem::size_of::<T>() == mem::size_of::<Complex32>() {
        dispatch!(
            Complex32,
            cuComplex,
            cusolver_dn_cunmqr_buffer_size,
            cusolver_dn_cunmqr
        )
    } else {
        dispatch!(
            Complex64,
            cuDoubleComplex,
            cusolver_dn_zunmqr_buffer_size,
            cusolver_dn_zunmqr
        )
    }
}

// ---- gels: iterative-refinement least-squares solve ---------------------

/// Solve `A * X = B` in the least-squares sense (iterative-refinement).
/// `A` is `m × n`, `B` is `m × nrhs`, `X` is `n × nrhs`. `A` and `B` may be
/// overwritten. Returns `iter`: number of refinement iterations used (-1 =
/// fallback to full precision).
#[allow(clippy::too_many_arguments)]
pub fn gels<T: SolverScalar>(
    handle: &DnHandle,
    m: i32,
    n: i32,
    nrhs: i32,
    a: &mut DeviceBuffer<T>,
    lda: i32,
    b: &mut DeviceBuffer<T>,
    ldb: i32,
    x: &mut DeviceBuffer<T>,
    ldx: i32,
    info: &mut DeviceBuffer<i32>,
) -> Result<i32> {
    use baracuda_cusolver_sys::{cuComplex, cuDoubleComplex};
    use core::mem;

    let mut bytes: usize = 0;

    macro_rules! dispatch {
        ($t:ty, $raw:ty, $bufsize:ident, $solve:ident) => {{
            let cs = cusolver()?;
            check(unsafe {
                (cs.$bufsize()?)(
                    handle.as_raw(),
                    m,
                    n,
                    nrhs,
                    a.as_raw().0 as *mut $raw,
                    lda,
                    b.as_raw().0 as *mut $raw,
                    ldb,
                    x.as_raw().0 as *mut $raw,
                    ldx,
                    core::ptr::null_mut(),
                    &mut bytes,
                )
            })?;
            // Allocate `bytes` worth of u8 workspace (rounding up to T units).
            let units = bytes.div_ceil(mem::size_of::<T>());
            let workspace =
                DeviceBuffer::<T>::new(a.context(), units).map_err(alloc_fail)?;
            let mut iter: c_int = 0;
            check(unsafe {
                (cs.$solve()?)(
                    handle.as_raw(),
                    m,
                    n,
                    nrhs,
                    a.as_raw().0 as *mut $raw,
                    lda,
                    b.as_raw().0 as *mut $raw,
                    ldb,
                    x.as_raw().0 as *mut $raw,
                    ldx,
                    workspace.as_raw().0 as *mut c_void,
                    bytes,
                    &mut iter,
                    info.as_raw().0 as *mut c_int,
                )
            })?;
            Ok(iter)
        }};
    }

    if mem::size_of::<T>() == mem::size_of::<f32>() && mem::size_of::<T::Real>() == 4 {
        dispatch!(f32, f32, cusolver_dn_ssgels_buffer_size, cusolver_dn_ssgels)
    } else if mem::size_of::<T>() == mem::size_of::<f64>() && mem::size_of::<T::Real>() == 8 {
        dispatch!(f64, f64, cusolver_dn_ddgels_buffer_size, cusolver_dn_ddgels)
    } else if mem::size_of::<T>() == mem::size_of::<Complex32>() {
        dispatch!(
            Complex32,
            cuComplex,
            cusolver_dn_ccgels_buffer_size,
            cusolver_dn_ccgels
        )
    } else {
        dispatch!(
            Complex64,
            cuDoubleComplex,
            cusolver_dn_zzgels_buffer_size,
            cusolver_dn_zzgels
        )
    }
}

// ---- potri: inverse from Cholesky factor --------------------------------

/// Compute `A = (Lᵀ * L)⁻¹` or `A = (U * Uᵀ)⁻¹` given the Cholesky factor
/// already stored in the triangle selected by `uplo`. `a` must hold the
/// output of [`potrf`] in-place.
pub fn potri<T: SolverScalar>(
    handle: &DnHandle,
    uplo: Fill,
    n: i32,
    a: &mut DeviceBuffer<T>,
    lda: i32,
    info: &mut DeviceBuffer<i32>,
) -> Result<()> {
    use baracuda_cusolver_sys::{cuComplex, cuDoubleComplex};
    use core::mem;

    let mut lwork: c_int = 0;
    macro_rules! dispatch {
        ($t:ty, $raw:ty, $bufsize:ident, $solve:ident) => {{
            let cs = cusolver()?;
            check(unsafe {
                (cs.$bufsize()?)(
                    handle.as_raw(),
                    uplo,
                    n,
                    a.as_raw().0 as *mut $raw,
                    lda,
                    &mut lwork,
                )
            })?;
            let workspace =
                DeviceBuffer::<T>::new(a.context(), lwork as usize).map_err(alloc_fail)?;
            check(unsafe {
                (cs.$solve()?)(
                    handle.as_raw(),
                    uplo,
                    n,
                    a.as_raw().0 as *mut $raw,
                    lda,
                    workspace.as_raw().0 as *mut $raw,
                    lwork,
                    info.as_raw().0 as *mut c_int,
                )
            })
        }};
    }

    if mem::size_of::<T>() == mem::size_of::<f32>() && mem::size_of::<T::Real>() == 4 {
        dispatch!(f32, f32, cusolver_dn_spotri_buffer_size, cusolver_dn_spotri)
    } else if mem::size_of::<T>() == mem::size_of::<f64>() && mem::size_of::<T::Real>() == 8 {
        dispatch!(f64, f64, cusolver_dn_dpotri_buffer_size, cusolver_dn_dpotri)
    } else if mem::size_of::<T>() == mem::size_of::<Complex32>() {
        dispatch!(
            Complex32,
            cuComplex,
            cusolver_dn_cpotri_buffer_size,
            cusolver_dn_cpotri
        )
    } else {
        dispatch!(
            Complex64,
            cuDoubleComplex,
            cusolver_dn_zpotri_buffer_size,
            cusolver_dn_zpotri
        )
    }
}

// ---- Batched Jacobi eigen / SVD -----------------------------------------

/// Batched Jacobi symmetric/Hermitian eigendecomposition. Every matrix in
/// the batch is `n × n` and stride `n × n`. `w` holds `n * batch_size`
/// eigenvalues, strided by `n`.
#[allow(clippy::too_many_arguments)]
pub fn syevj_batched<T: SolverScalar>(
    handle: &DnHandle,
    jobz: EigMode,
    uplo: Fill,
    n: i32,
    a: &mut DeviceBuffer<T>,
    lda: i32,
    w: &mut DeviceBuffer<T::Real>,
    info: &mut DeviceBuffer<i32>,
    params: &SyevjInfo,
    batch_size: i32,
) -> Result<()> {
    use baracuda_cusolver_sys::{cuComplex, cuDoubleComplex};
    use core::mem;

    let mut lwork: c_int = 0;
    macro_rules! dispatch_real {
        ($t:ty, $bufsize:ident, $solve:ident) => {{
            let c = cusolver()?;
            check(unsafe {
                (c.$bufsize()?)(
                    handle.as_raw(),
                    jobz,
                    uplo,
                    n,
                    a.as_raw().0 as *const $t,
                    lda,
                    w.as_raw().0 as *const $t,
                    &mut lwork,
                    params.as_raw(),
                    batch_size,
                )
            })?;
            let workspace =
                DeviceBuffer::<T>::new(a.context(), lwork as usize).map_err(alloc_fail)?;
            check(unsafe {
                (c.$solve()?)(
                    handle.as_raw(),
                    jobz,
                    uplo,
                    n,
                    a.as_raw().0 as *mut $t,
                    lda,
                    w.as_raw().0 as *mut $t,
                    workspace.as_raw().0 as *mut $t,
                    lwork,
                    info.as_raw().0 as *mut c_int,
                    params.as_raw(),
                    batch_size,
                )
            })
        }};
    }
    macro_rules! dispatch_complex {
        ($t:ty, $real:ty, $raw:ty, $bufsize:ident, $solve:ident) => {{
            let c = cusolver()?;
            check(unsafe {
                (c.$bufsize()?)(
                    handle.as_raw(),
                    jobz,
                    uplo,
                    n,
                    a.as_raw().0 as *const $raw,
                    lda,
                    w.as_raw().0 as *const $real,
                    &mut lwork,
                    params.as_raw(),
                    batch_size,
                )
            })?;
            let workspace =
                DeviceBuffer::<T>::new(a.context(), lwork as usize).map_err(alloc_fail)?;
            check(unsafe {
                (c.$solve()?)(
                    handle.as_raw(),
                    jobz,
                    uplo,
                    n,
                    a.as_raw().0 as *mut $raw,
                    lda,
                    w.as_raw().0 as *mut $real,
                    workspace.as_raw().0 as *mut $raw,
                    lwork,
                    info.as_raw().0 as *mut c_int,
                    params.as_raw(),
                    batch_size,
                )
            })
        }};
    }

    if mem::size_of::<T>() == mem::size_of::<f32>() && mem::size_of::<T::Real>() == 4 {
        dispatch_real!(
            f32,
            cusolver_dn_ssyevj_batched_buffer_size,
            cusolver_dn_ssyevj_batched
        )
    } else if mem::size_of::<T>() == mem::size_of::<f64>() && mem::size_of::<T::Real>() == 8 {
        dispatch_real!(
            f64,
            cusolver_dn_dsyevj_batched_buffer_size,
            cusolver_dn_dsyevj_batched
        )
    } else if mem::size_of::<T>() == mem::size_of::<Complex32>() {
        dispatch_complex!(
            Complex32,
            f32,
            cuComplex,
            cusolver_dn_cheevj_batched_buffer_size,
            cusolver_dn_cheevj_batched
        )
    } else {
        dispatch_complex!(
            Complex64,
            f64,
            cuDoubleComplex,
            cusolver_dn_zheevj_batched_buffer_size,
            cusolver_dn_zheevj_batched
        )
    }
}

/// Batched Jacobi SVD: batch of `m × n` matrices with stride `m×n`.
#[allow(clippy::too_many_arguments)]
pub fn gesvdj_batched<T: SolverScalar>(
    handle: &DnHandle,
    jobz: EigMode,
    m: i32,
    n: i32,
    a: &mut DeviceBuffer<T>,
    lda: i32,
    s: &mut DeviceBuffer<T::Real>,
    u: &mut DeviceBuffer<T>,
    ldu: i32,
    v: &mut DeviceBuffer<T>,
    ldv: i32,
    info: &mut DeviceBuffer<i32>,
    params: &GesvdjInfo,
    batch_size: i32,
) -> Result<()> {
    use baracuda_cusolver_sys::{cuComplex, cuDoubleComplex};
    use core::mem;

    let mut lwork: c_int = 0;
    macro_rules! dispatch_real {
        ($t:ty, $bufsize:ident, $solve:ident) => {{
            let c = cusolver()?;
            check(unsafe {
                (c.$bufsize()?)(
                    handle.as_raw(),
                    jobz,
                    m,
                    n,
                    a.as_raw().0 as *const $t,
                    lda,
                    s.as_raw().0 as *const $t,
                    u.as_raw().0 as *const $t,
                    ldu,
                    v.as_raw().0 as *const $t,
                    ldv,
                    &mut lwork,
                    params.as_raw(),
                    batch_size,
                )
            })?;
            let workspace =
                DeviceBuffer::<T>::new(a.context(), lwork as usize).map_err(alloc_fail)?;
            check(unsafe {
                (c.$solve()?)(
                    handle.as_raw(),
                    jobz,
                    m,
                    n,
                    a.as_raw().0 as *mut $t,
                    lda,
                    s.as_raw().0 as *mut $t,
                    u.as_raw().0 as *mut $t,
                    ldu,
                    v.as_raw().0 as *mut $t,
                    ldv,
                    workspace.as_raw().0 as *mut $t,
                    lwork,
                    info.as_raw().0 as *mut c_int,
                    params.as_raw(),
                    batch_size,
                )
            })
        }};
    }
    macro_rules! dispatch_complex {
        ($t:ty, $real:ty, $raw:ty, $bufsize:ident, $solve:ident) => {{
            let c = cusolver()?;
            check(unsafe {
                (c.$bufsize()?)(
                    handle.as_raw(),
                    jobz,
                    m,
                    n,
                    a.as_raw().0 as *const $raw,
                    lda,
                    s.as_raw().0 as *const $real,
                    u.as_raw().0 as *const $raw,
                    ldu,
                    v.as_raw().0 as *const $raw,
                    ldv,
                    &mut lwork,
                    params.as_raw(),
                    batch_size,
                )
            })?;
            let workspace =
                DeviceBuffer::<T>::new(a.context(), lwork as usize).map_err(alloc_fail)?;
            check(unsafe {
                (c.$solve()?)(
                    handle.as_raw(),
                    jobz,
                    m,
                    n,
                    a.as_raw().0 as *mut $raw,
                    lda,
                    s.as_raw().0 as *mut $real,
                    u.as_raw().0 as *mut $raw,
                    ldu,
                    v.as_raw().0 as *mut $raw,
                    ldv,
                    workspace.as_raw().0 as *mut $raw,
                    lwork,
                    info.as_raw().0 as *mut c_int,
                    params.as_raw(),
                    batch_size,
                )
            })
        }};
    }

    if mem::size_of::<T>() == mem::size_of::<f32>() && mem::size_of::<T::Real>() == 4 {
        dispatch_real!(
            f32,
            cusolver_dn_sgesvdj_batched_buffer_size,
            cusolver_dn_sgesvdj_batched
        )
    } else if mem::size_of::<T>() == mem::size_of::<f64>() && mem::size_of::<T::Real>() == 8 {
        dispatch_real!(
            f64,
            cusolver_dn_dgesvdj_batched_buffer_size,
            cusolver_dn_dgesvdj_batched
        )
    } else if mem::size_of::<T>() == mem::size_of::<Complex32>() {
        dispatch_complex!(
            Complex32,
            f32,
            cuComplex,
            cusolver_dn_cgesvdj_batched_buffer_size,
            cusolver_dn_cgesvdj_batched
        )
    } else {
        dispatch_complex!(
            Complex64,
            f64,
            cuDoubleComplex,
            cusolver_dn_zgesvdj_batched_buffer_size,
            cusolver_dn_zgesvdj_batched
        )
    }
}

// ---- cuSOLVERMg: multi-GPU dense solvers --------------------------------

pub mod mg {
    //! Multi-GPU dense solvers via `libcusolverMg`. Shares dimensions with
    //! the single-GPU API but takes arrays of device pointers (one per
    //! physical GPU after [`Handle::device_select`]).

    use core::ffi::{c_int, c_void};

    use baracuda_cusolver_sys::{
        cudaDataType, cudaLibMgGrid_t, cudaLibMgMatrixDesc_t, cusolver_mg, cusolverMgHandle_t,
    };

    use super::{alloc_fail, check, EigMode, Fill, Result};

    /// Multi-GPU cuSOLVER handle.
    #[derive(Debug)]
    pub struct Handle {
        raw: cusolverMgHandle_t,
    }

    impl Handle {
        pub fn new() -> Result<Self> {
            let mg = cusolver_mg()?;
            let cu = mg.cusolver_mg_create()?;
            let mut h: cusolverMgHandle_t = core::ptr::null_mut();
            check(unsafe { cu(&mut h) })?;
            Ok(Self { raw: h })
        }

        /// Assign a set of physical CUDA devices to this handle. Future
        /// factorizations will stripe across them.
        pub fn device_select(&self, devices: &[i32]) -> Result<()> {
            let mg = cusolver_mg()?;
            let cu = mg.cusolver_mg_device_select()?;
            check(unsafe { cu(self.raw, devices.len() as c_int, devices.as_ptr()) })
        }

        pub fn as_raw(&self) -> cusolverMgHandle_t {
            self.raw
        }
    }

    impl Drop for Handle {
        fn drop(&mut self) {
            if let Ok(mg) = cusolver_mg() {
                if let Ok(cu) = mg.cusolver_mg_destroy() {
                    let _ = unsafe { cu(self.raw) };
                }
            }
        }
    }

    /// A device grid — assigns distribution roles to physical devices.
    #[derive(Debug)]
    pub struct DeviceGrid {
        raw: cudaLibMgGrid_t,
    }

    impl DeviceGrid {
        /// `mapping` is typically `CUDALIBMG_GRID_MAPPING_COL_MAJOR (1)`.
        pub fn new(num_row_devices: i32, num_col_devices: i32, devices: &[i32], mapping: i32) -> Result<Self> {
            let mg = cusolver_mg()?;
            let cu = mg.cusolver_mg_create_device_grid()?;
            let mut raw: cudaLibMgGrid_t = core::ptr::null_mut();
            check(unsafe {
                cu(
                    &mut raw,
                    num_row_devices,
                    num_col_devices,
                    devices.as_ptr(),
                    mapping,
                )
            })?;
            Ok(Self { raw })
        }

        pub fn as_raw(&self) -> cudaLibMgGrid_t {
            self.raw
        }
    }

    impl Drop for DeviceGrid {
        fn drop(&mut self) {
            if let Ok(mg) = cusolver_mg() {
                if let Ok(cu) = mg.cusolver_mg_destroy_grid() {
                    let _ = unsafe { cu(self.raw) };
                }
            }
        }
    }

    /// Matrix-distribution descriptor.
    #[derive(Debug)]
    pub struct MatrixDesc {
        raw: cudaLibMgMatrixDesc_t,
    }

    impl MatrixDesc {
        pub fn new(
            num_rows: i64,
            num_cols: i64,
            row_block_size: i64,
            col_block_size: i64,
            data_type: cudaDataType,
            grid: &DeviceGrid,
        ) -> Result<Self> {
            let mg = cusolver_mg()?;
            let cu = mg.cusolver_mg_create_matrix_desc()?;
            let mut raw: cudaLibMgMatrixDesc_t = core::ptr::null_mut();
            check(unsafe {
                cu(
                    &mut raw,
                    num_rows,
                    num_cols,
                    row_block_size,
                    col_block_size,
                    data_type,
                    grid.as_raw(),
                )
            })?;
            Ok(Self { raw })
        }

        pub fn as_raw(&self) -> cudaLibMgMatrixDesc_t {
            self.raw
        }
    }

    impl Drop for MatrixDesc {
        fn drop(&mut self) {
            if let Ok(mg) = cusolver_mg() {
                if let Ok(cu) = mg.cusolver_mg_destroy_matrix_desc() {
                    let _ = unsafe { cu(self.raw) };
                }
            }
        }
    }

    /// Multi-GPU LU buffer-size query.
    ///
    /// # Safety
    /// `array_d_a`, `array_d_ipiv` must be host arrays of device pointers
    /// matching the selected devices.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn getrf_buffer_size(
        handle: &Handle,
        m: i32,
        n: i32,
        array_d_a: *mut *mut c_void,
        ia: i32,
        ja: i32,
        desc_a: &MatrixDesc,
        array_d_ipiv: *mut *mut c_int,
        compute_type: cudaDataType,
    ) -> Result<i64> {
        let mg = cusolver_mg()?;
        let cu = mg.cusolver_mg_getrf_buffer_size()?;
        let mut lwork: i64 = 0;
        check(cu(
            handle.as_raw(),
            m,
            n,
            array_d_a,
            ia,
            ja,
            desc_a.as_raw(),
            array_d_ipiv,
            compute_type,
            &mut lwork,
        ))?;
        Ok(lwork)
    }

    /// # Safety
    /// Same pointer-array requirements as [`getrf_buffer_size`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn getrf(
        handle: &Handle,
        m: i32,
        n: i32,
        array_d_a: *mut *mut c_void,
        ia: i32,
        ja: i32,
        desc_a: &MatrixDesc,
        array_d_ipiv: *mut *mut c_int,
        compute_type: cudaDataType,
        array_d_work: *mut *mut c_void,
        lwork: i64,
        info: &mut [c_int],
    ) -> Result<()> {
        let mg = cusolver_mg()?;
        let cu = mg.cusolver_mg_getrf()?;
        let _ = alloc_fail::<()>; // silence unused-import in release builds
        check(cu(
            handle.as_raw(),
            m,
            n,
            array_d_a,
            ia,
            ja,
            desc_a.as_raw(),
            array_d_ipiv,
            compute_type,
            array_d_work,
            lwork,
            info.as_mut_ptr(),
        ))
    }

    /// Multi-GPU Cholesky buffer-size.
    ///
    /// # Safety
    /// Same as [`getrf_buffer_size`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn potrf_buffer_size(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        array_d_a: *mut *mut c_void,
        ia: i32,
        ja: i32,
        desc_a: &MatrixDesc,
        compute_type: cudaDataType,
    ) -> Result<i64> {
        let mg = cusolver_mg()?;
        let cu = mg.cusolver_mg_potrf_buffer_size()?;
        let mut lwork: i64 = 0;
        check(cu(
            handle.as_raw(),
            uplo,
            n,
            array_d_a,
            ia,
            ja,
            desc_a.as_raw(),
            compute_type,
            &mut lwork,
        ))?;
        Ok(lwork)
    }

    /// # Safety
    /// Same as [`getrf_buffer_size`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn potrf(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        array_d_a: *mut *mut c_void,
        ia: i32,
        ja: i32,
        desc_a: &MatrixDesc,
        compute_type: cudaDataType,
        array_d_work: *mut *mut c_void,
        lwork: i64,
        info: &mut [c_int],
    ) -> Result<()> {
        let mg = cusolver_mg()?;
        let cu = mg.cusolver_mg_potrf()?;
        check(cu(
            handle.as_raw(),
            uplo,
            n,
            array_d_a,
            ia,
            ja,
            desc_a.as_raw(),
            compute_type,
            array_d_work,
            lwork,
            info.as_mut_ptr(),
        ))
    }

    /// Multi-GPU symmetric eigendecomposition buffer-size.
    ///
    /// # Safety
    /// Same as [`getrf_buffer_size`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn syevd_buffer_size(
        handle: &Handle,
        jobz: EigMode,
        uplo: Fill,
        n: i32,
        array_d_a: *mut *mut c_void,
        ia: i32,
        ja: i32,
        desc_a: &MatrixDesc,
        w: *mut c_void,
        data_type_w: cudaDataType,
        compute_type: cudaDataType,
    ) -> Result<i64> {
        let mg = cusolver_mg()?;
        let cu = mg.cusolver_mg_syevd_buffer_size()?;
        let mut lwork: i64 = 0;
        check(cu(
            handle.as_raw(),
            jobz,
            uplo,
            n,
            array_d_a,
            ia,
            ja,
            desc_a.as_raw(),
            w,
            data_type_w,
            compute_type,
            &mut lwork,
        ))?;
        Ok(lwork)
    }

    /// # Safety
    /// Same as [`getrf_buffer_size`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn syevd(
        handle: &Handle,
        jobz: EigMode,
        uplo: Fill,
        n: i32,
        array_d_a: *mut *mut c_void,
        ia: i32,
        ja: i32,
        desc_a: &MatrixDesc,
        w: *mut c_void,
        data_type_w: cudaDataType,
        compute_type: cudaDataType,
        array_d_work: *mut *mut c_void,
        lwork: i64,
        info: &mut [c_int],
    ) -> Result<()> {
        let mg = cusolver_mg()?;
        let cu = mg.cusolver_mg_syevd()?;
        check(cu(
            handle.as_raw(),
            jobz,
            uplo,
            n,
            array_d_a,
            ia,
            ja,
            desc_a.as_raw(),
            w,
            data_type_w,
            compute_type,
            array_d_work,
            lwork,
            info.as_mut_ptr(),
        ))
    }
}

// ---- Back-compat: single-precision shortcuts -----------------------------

/// Shortcut for [`getrf`] on `f32`.
pub fn sgetrf(
    handle: &DnHandle,
    m: i32,
    n: i32,
    a: &mut DeviceBuffer<f32>,
    lda: i32,
    ipiv: &mut DeviceBuffer<i32>,
    info: &mut DeviceBuffer<i32>,
) -> Result<()> {
    getrf::<f32>(handle, m, n, a, lda, ipiv, info)
}

/// Shortcut for [`getrs`] on `f32`.
#[allow(clippy::too_many_arguments)]
pub fn sgetrs(
    handle: &DnHandle,
    trans: Op,
    n: i32,
    nrhs: i32,
    a: &DeviceBuffer<f32>,
    lda: i32,
    ipiv: &DeviceBuffer<i32>,
    b: &mut DeviceBuffer<f32>,
    ldb: i32,
    info: &mut DeviceBuffer<i32>,
) -> Result<()> {
    getrs::<f32>(handle, trans, n, nrhs, a, lda, ipiv, b, ldb, info)
}

// ---- Generic X... (64-bit-size, type-erased) ----------------------------

pub mod xapi {
    //! The generic 64-bit cuSOLVER API (`cusolverDnX*`). Matrix dimensions
    //! are `i64`; element types are passed at call-time as
    //! [`cudaDataType`]. Workspace sizes are split between on-device and
    //! on-host buffers.

    use super::*;
    use baracuda_cusolver_sys::{cudaDataType, cusolverDnParams_t};

    #[derive(Debug)]
    pub struct Params {
        raw: cusolverDnParams_t,
    }

    impl Params {
        pub fn new() -> Result<Self> {
            let c = cusolver()?;
            let cu = c.cusolver_dn_create_params()?;
            let mut p: cusolverDnParams_t = core::ptr::null_mut();
            check(unsafe { cu(&mut p) })?;
            Ok(Self { raw: p })
        }

        pub fn as_raw(&self) -> cusolverDnParams_t {
            self.raw
        }
    }

    impl Drop for Params {
        fn drop(&mut self) {
            if let Ok(c) = cusolver() {
                if let Ok(cu) = c.cusolver_dn_destroy_params() {
                    let _ = unsafe { cu(self.raw) };
                }
            }
        }
    }

    /// Buffer-size query for generic LU factorization. Returns
    /// `(workspace_bytes_on_device, workspace_bytes_on_host)`.
    #[allow(clippy::too_many_arguments)]
    pub fn xgetrf_buffer_size(
        handle: &DnHandle,
        params: &Params,
        m: i64,
        n: i64,
        data_type_a: cudaDataType,
        a: *const c_void,
        lda: i64,
        compute_type: cudaDataType,
    ) -> Result<(usize, usize)> {
        let c = cusolver()?;
        let cu = c.cusolver_dn_xgetrf_buffer_size()?;
        let (mut dev, mut host) = (0usize, 0usize);
        check(unsafe {
            cu(
                handle.as_raw(),
                params.raw,
                m,
                n,
                data_type_a,
                a,
                lda,
                compute_type,
                &mut dev,
                &mut host,
            )
        })?;
        Ok((dev, host))
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn xgetrf(
        handle: &DnHandle,
        params: &Params,
        m: i64,
        n: i64,
        data_type_a: cudaDataType,
        a: *mut c_void,
        lda: i64,
        ipiv: *mut i64,
        compute_type: cudaDataType,
        device_buf: *mut c_void,
        device_bytes: usize,
        host_buf: *mut c_void,
        host_bytes: usize,
        info: *mut c_int,
    ) -> Result<()> {
        let c = cusolver()?;
        let cu = c.cusolver_dn_xgetrf()?;
        check(cu(
            handle.as_raw(),
            params.raw,
            m,
            n,
            data_type_a,
            a,
            lda,
            ipiv,
            compute_type,
            device_buf,
            device_bytes,
            host_buf,
            host_bytes,
            info,
        ))
    }
}

// ---- Sparse --------------------------------------------------------------

pub mod sparse {
    //! `cusolverSp*` — solve sparse linear systems via Cholesky or QR.

    use super::*;
    use baracuda_cusolver_sys::cusolverSpHandle_t;
    use core::ffi::c_int;

    #[derive(Debug)]
    pub struct SpHandle {
        raw: cusolverSpHandle_t,
        _not_send: PhantomData<*mut ()>,
    }

    impl SpHandle {
        pub fn new() -> Result<Self> {
            let c = cusolver()?;
            let cu = c.cusolver_sp_create()?;
            let mut h: cusolverSpHandle_t = core::ptr::null_mut();
            check(unsafe { cu(&mut h) })?;
            Ok(Self {
                raw: h,
                _not_send: PhantomData,
            })
        }

        pub fn set_stream(&self, stream: &Stream) -> Result<()> {
            let c = cusolver()?;
            let cu = c.cusolver_sp_set_stream()?;
            check(unsafe { cu(self.raw, stream.as_raw() as _) })
        }

        pub fn as_raw(&self) -> cusolverSpHandle_t {
            self.raw
        }
    }

    impl Drop for SpHandle {
        fn drop(&mut self) {
            if let Ok(c) = cusolver() {
                if let Ok(cu) = c.cusolver_sp_destroy() {
                    let _ = unsafe { cu(self.raw) };
                }
            }
        }
    }

    /// Sparse Cholesky solve: `A * x = b` for SPD `A`.
    ///
    /// # Safety
    /// `descr_a`, CSR arrays, b and x must live on-device (b + x on-device,
    /// CSR arrays + descriptor on-device) and satisfy cuSOLVER sparse
    /// format requirements.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn scsrlsvchol(
        handle: &SpHandle,
        m: i32,
        nnz: i32,
        descr_a: *mut c_void,
        csr_val: *const f32,
        csr_row_ptr: *const c_int,
        csr_col_ind: *const c_int,
        b: *const f32,
        tol: f32,
        reorder: i32,
        x: *mut f32,
        singularity: *mut c_int,
    ) -> Result<()> {
        let c = cusolver()?;
        let cu = c.cusolver_sp_scsrlsvchol()?;
        check(cu(
            handle.raw,
            m,
            nnz,
            descr_a,
            csr_val,
            csr_row_ptr,
            csr_col_ind,
            b,
            tol,
            reorder,
            x,
            singularity,
        ))
    }

    /// Sparse QR solve (least-squares, handles non-SPD systems).
    ///
    /// # Safety
    /// Same as [`scsrlsvchol`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn scsrlsvqr(
        handle: &SpHandle,
        m: i32,
        nnz: i32,
        descr_a: *mut c_void,
        csr_val: *const f32,
        csr_row_ptr: *const c_int,
        csr_col_ind: *const c_int,
        b: *const f32,
        tol: f32,
        reorder: i32,
        x: *mut f32,
        singularity: *mut c_int,
    ) -> Result<()> {
        let c = cusolver()?;
        let cu = c.cusolver_sp_scsrlsvqr()?;
        check(cu(
            handle.raw,
            m,
            nnz,
            descr_a,
            csr_val,
            csr_row_ptr,
            csr_col_ind,
            b,
            tol,
            reorder,
            x,
            singularity,
        ))
    }
}

// ---- Refactor ------------------------------------------------------------

pub mod refactor {
    //! `cusolverRf*` — fast re-factorization given a sparsity pattern, for
    //! solving many systems that differ only in numeric values.

    use super::*;
    use baracuda_cusolver_sys::cusolverRfHandle_t;

    #[derive(Debug)]
    pub struct RfHandle {
        raw: cusolverRfHandle_t,
        _not_send: PhantomData<*mut ()>,
    }

    impl RfHandle {
        pub fn new() -> Result<Self> {
            let c = cusolver()?;
            let cu = c.cusolver_rf_create()?;
            let mut h: cusolverRfHandle_t = core::ptr::null_mut();
            check(unsafe { cu(&mut h) })?;
            Ok(Self {
                raw: h,
                _not_send: PhantomData,
            })
        }

        pub fn as_raw(&self) -> cusolverRfHandle_t {
            self.raw
        }

        pub fn analyze(&self) -> Result<()> {
            let c = cusolver()?;
            let cu = c.cusolver_rf_analyze()?;
            check(unsafe { cu(self.raw) })
        }

        pub fn refactor(&self) -> Result<()> {
            let c = cusolver()?;
            let cu = c.cusolver_rf_refactor()?;
            check(unsafe { cu(self.raw) })
        }
    }

    impl Drop for RfHandle {
        fn drop(&mut self) {
            if let Ok(c) = cusolver() {
                if let Ok(cu) = c.cusolver_rf_destroy() {
                    let _ = unsafe { cu(self.raw) };
                }
            }
        }
    }
}
