//! BLAS-1 routines beyond `axpy`: `dot`, `scal`, `nrm2`, `asum`,
//! `iamax` / `iamin`, and `copy`.
//!
//! Generic over `f32` / `f64` via a private dispatch trait. All vectors
//! are column-based 1-D device buffers; `incx` / `incy` are stride in
//! elements (pass `1` for contiguous).

use baracuda_cublas_sys::{cublas, cublasHandle_t, cublasStatus_t};
use baracuda_driver::{DeviceBuffer, DeviceSlice};
use baracuda_types::DeviceRepr;

use crate::error::{check, Result};

/// Private dispatch trait for L1 ops — implemented for `f32` and `f64`.
pub trait L1Scalar: DeviceRepr + l1_sealed::Sealed {
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn dot_raw(
        handle: cublasHandle_t,
        n: i32,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        result: *mut Self,
    ) -> cublasStatus_t;

    #[doc(hidden)]
    unsafe fn scal_raw(
        handle: cublasHandle_t,
        n: i32,
        alpha: &Self,
        x: *mut Self,
        incx: i32,
    ) -> cublasStatus_t;

    #[doc(hidden)]
    unsafe fn nrm2_raw(
        handle: cublasHandle_t,
        n: i32,
        x: *const Self,
        incx: i32,
        result: *mut Self,
    ) -> cublasStatus_t;

    #[doc(hidden)]
    unsafe fn asum_raw(
        handle: cublasHandle_t,
        n: i32,
        x: *const Self,
        incx: i32,
        result: *mut Self,
    ) -> cublasStatus_t;

    #[doc(hidden)]
    unsafe fn iamax_raw(
        handle: cublasHandle_t,
        n: i32,
        x: *const Self,
        incx: i32,
        result: *mut i32,
    ) -> cublasStatus_t;

    #[doc(hidden)]
    unsafe fn iamin_raw(
        handle: cublasHandle_t,
        n: i32,
        x: *const Self,
        incx: i32,
        result: *mut i32,
    ) -> cublasStatus_t;

    #[doc(hidden)]
    unsafe fn copy_raw(
        handle: cublasHandle_t,
        n: i32,
        x: *const Self,
        incx: i32,
        y: *mut Self,
        incy: i32,
    ) -> cublasStatus_t;
}

impl L1Scalar for f32 {
    unsafe fn dot_raw(
        h: cublasHandle_t,
        n: i32,
        x: *const f32,
        ix: i32,
        y: *const f32,
        iy: i32,
        r: *mut f32,
    ) -> cublasStatus_t {
        match cublas().and_then(|c| c.cublas_sdot()) {
            Ok(f) => f(h, n, x, ix, y, iy, r),
            Err(_) => cublasStatus_t::NOT_INITIALIZED,
        }
    }
    unsafe fn scal_raw(h: cublasHandle_t, n: i32, a: &f32, x: *mut f32, ix: i32) -> cublasStatus_t {
        match cublas().and_then(|c| c.cublas_sscal()) {
            Ok(f) => f(h, n, a, x, ix),
            Err(_) => cublasStatus_t::NOT_INITIALIZED,
        }
    }
    unsafe fn nrm2_raw(
        h: cublasHandle_t,
        n: i32,
        x: *const f32,
        ix: i32,
        r: *mut f32,
    ) -> cublasStatus_t {
        match cublas().and_then(|c| c.cublas_snrm2()) {
            Ok(f) => f(h, n, x, ix, r),
            Err(_) => cublasStatus_t::NOT_INITIALIZED,
        }
    }
    unsafe fn asum_raw(
        h: cublasHandle_t,
        n: i32,
        x: *const f32,
        ix: i32,
        r: *mut f32,
    ) -> cublasStatus_t {
        match cublas().and_then(|c| c.cublas_sasum()) {
            Ok(f) => f(h, n, x, ix, r),
            Err(_) => cublasStatus_t::NOT_INITIALIZED,
        }
    }
    unsafe fn iamax_raw(
        h: cublasHandle_t,
        n: i32,
        x: *const f32,
        ix: i32,
        r: *mut i32,
    ) -> cublasStatus_t {
        match cublas().and_then(|c| c.cublas_isamax()) {
            Ok(f) => f(h, n, x, ix, r),
            Err(_) => cublasStatus_t::NOT_INITIALIZED,
        }
    }
    unsafe fn iamin_raw(
        h: cublasHandle_t,
        n: i32,
        x: *const f32,
        ix: i32,
        r: *mut i32,
    ) -> cublasStatus_t {
        match cublas().and_then(|c| c.cublas_isamin()) {
            Ok(f) => f(h, n, x, ix, r),
            Err(_) => cublasStatus_t::NOT_INITIALIZED,
        }
    }
    unsafe fn copy_raw(
        h: cublasHandle_t,
        n: i32,
        x: *const f32,
        ix: i32,
        y: *mut f32,
        iy: i32,
    ) -> cublasStatus_t {
        match cublas().and_then(|c| c.cublas_scopy()) {
            Ok(f) => f(h, n, x, ix, y, iy),
            Err(_) => cublasStatus_t::NOT_INITIALIZED,
        }
    }
}

impl L1Scalar for f64 {
    unsafe fn dot_raw(
        h: cublasHandle_t,
        n: i32,
        x: *const f64,
        ix: i32,
        y: *const f64,
        iy: i32,
        r: *mut f64,
    ) -> cublasStatus_t {
        match cublas().and_then(|c| c.cublas_ddot()) {
            Ok(f) => f(h, n, x, ix, y, iy, r),
            Err(_) => cublasStatus_t::NOT_INITIALIZED,
        }
    }
    unsafe fn scal_raw(h: cublasHandle_t, n: i32, a: &f64, x: *mut f64, ix: i32) -> cublasStatus_t {
        match cublas().and_then(|c| c.cublas_dscal()) {
            Ok(f) => f(h, n, a, x, ix),
            Err(_) => cublasStatus_t::NOT_INITIALIZED,
        }
    }
    unsafe fn nrm2_raw(
        h: cublasHandle_t,
        n: i32,
        x: *const f64,
        ix: i32,
        r: *mut f64,
    ) -> cublasStatus_t {
        match cublas().and_then(|c| c.cublas_dnrm2()) {
            Ok(f) => f(h, n, x, ix, r),
            Err(_) => cublasStatus_t::NOT_INITIALIZED,
        }
    }
    unsafe fn asum_raw(
        h: cublasHandle_t,
        n: i32,
        x: *const f64,
        ix: i32,
        r: *mut f64,
    ) -> cublasStatus_t {
        match cublas().and_then(|c| c.cublas_dasum()) {
            Ok(f) => f(h, n, x, ix, r),
            Err(_) => cublasStatus_t::NOT_INITIALIZED,
        }
    }
    unsafe fn iamax_raw(
        h: cublasHandle_t,
        n: i32,
        x: *const f64,
        ix: i32,
        r: *mut i32,
    ) -> cublasStatus_t {
        match cublas().and_then(|c| c.cublas_idamax()) {
            Ok(f) => f(h, n, x, ix, r),
            Err(_) => cublasStatus_t::NOT_INITIALIZED,
        }
    }
    unsafe fn iamin_raw(
        h: cublasHandle_t,
        n: i32,
        x: *const f64,
        ix: i32,
        r: *mut i32,
    ) -> cublasStatus_t {
        match cublas().and_then(|c| c.cublas_idamin()) {
            Ok(f) => f(h, n, x, ix, r),
            Err(_) => cublasStatus_t::NOT_INITIALIZED,
        }
    }
    unsafe fn copy_raw(
        h: cublasHandle_t,
        n: i32,
        x: *const f64,
        ix: i32,
        y: *mut f64,
        iy: i32,
    ) -> cublasStatus_t {
        match cublas().and_then(|c| c.cublas_dcopy()) {
            Ok(f) => f(h, n, x, ix, y, iy),
            Err(_) => cublasStatus_t::NOT_INITIALIZED,
        }
    }
}

mod l1_sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Dot product: `x · y`, result in host memory (pointer-mode HOST is the
/// baracuda default).
pub fn dot<T: L1Scalar + Default>(
    handle: &crate::Handle,
    n: i32,
    x: &DeviceSlice<'_, T>,
    incx: i32,
    y: &DeviceSlice<'_, T>,
    incy: i32,
) -> Result<T> {
    let mut result = T::default();
    let status = unsafe {
        T::dot_raw(
            handle.as_raw(),
            n,
            x.as_raw().0 as *const T,
            incx,
            y.as_raw().0 as *const T,
            incy,
            &mut result,
        )
    };
    check(status)?;
    Ok(result)
}

/// Scale in place: `x = alpha * x`.
pub fn scal<T: L1Scalar>(
    handle: &crate::Handle,
    n: i32,
    alpha: T,
    x: &mut DeviceBuffer<T>,
    incx: i32,
) -> Result<()> {
    let status = unsafe { T::scal_raw(handle.as_raw(), n, &alpha, x.as_raw().0 as *mut T, incx) };
    check(status)
}

/// Euclidean norm: `sqrt(sum(x_i^2))`.
pub fn nrm2<T: L1Scalar + Default>(
    handle: &crate::Handle,
    n: i32,
    x: &DeviceSlice<'_, T>,
    incx: i32,
) -> Result<T> {
    let mut result = T::default();
    let status = unsafe {
        T::nrm2_raw(
            handle.as_raw(),
            n,
            x.as_raw().0 as *const T,
            incx,
            &mut result,
        )
    };
    check(status)?;
    Ok(result)
}

/// Sum of absolute values: `sum(|x_i|)`.
pub fn asum<T: L1Scalar + Default>(
    handle: &crate::Handle,
    n: i32,
    x: &DeviceSlice<'_, T>,
    incx: i32,
) -> Result<T> {
    let mut result = T::default();
    let status = unsafe {
        T::asum_raw(
            handle.as_raw(),
            n,
            x.as_raw().0 as *const T,
            incx,
            &mut result,
        )
    };
    check(status)?;
    Ok(result)
}

/// 1-based index of the element with largest absolute value.
/// Subtract 1 to get a 0-based Rust index.
pub fn iamax<T: L1Scalar>(
    handle: &crate::Handle,
    n: i32,
    x: &DeviceSlice<'_, T>,
    incx: i32,
) -> Result<i32> {
    let mut result: i32 = 0;
    let status = unsafe {
        T::iamax_raw(
            handle.as_raw(),
            n,
            x.as_raw().0 as *const T,
            incx,
            &mut result,
        )
    };
    check(status)?;
    Ok(result)
}

/// 1-based index of the element with smallest absolute value.
pub fn iamin<T: L1Scalar>(
    handle: &crate::Handle,
    n: i32,
    x: &DeviceSlice<'_, T>,
    incx: i32,
) -> Result<i32> {
    let mut result: i32 = 0;
    let status = unsafe {
        T::iamin_raw(
            handle.as_raw(),
            n,
            x.as_raw().0 as *const T,
            incx,
            &mut result,
        )
    };
    check(status)?;
    Ok(result)
}

/// Copy: `y = x`. Both vectors live on-device.
pub fn copy<T: L1Scalar>(
    handle: &crate::Handle,
    n: i32,
    x: &DeviceSlice<'_, T>,
    incx: i32,
    y: &mut DeviceBuffer<T>,
    incy: i32,
) -> Result<()> {
    let status = unsafe {
        T::copy_raw(
            handle.as_raw(),
            n,
            x.as_raw().0 as *const T,
            incx,
            y.as_raw().0 as *mut T,
            incy,
        )
    };
    check(status)
}
