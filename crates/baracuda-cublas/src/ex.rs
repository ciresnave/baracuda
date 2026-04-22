//! Mixed-precision BLAS-1 (`axpyEx`, `dotEx`, `nrm2Ex`, `scalEx`, `rotEx`).
//!
//! Unlike the typed [`crate::axpy`] etc. helpers, these accept explicit
//! [`cudaDataType_t`] tags for the alpha scalar, vector element types, and
//! arithmetic precision — matching cuBLAS's type-erased API surface. They
//! are the right tool for fp16/bf16 vector ops with an fp32 accumulator.

use core::ffi::c_void;

use baracuda_cublas_sys::functions::cudaDataType_t;
use baracuda_cublas_sys::cublas;

use crate::error::{check, Result};

/// `y = alpha * x + y` with explicit types. See [`cudaDataType_t`] for tag values.
///
/// # Safety
/// Every pointer must be a valid device pointer of the declared type/length.
#[allow(clippy::too_many_arguments)]
pub unsafe fn axpy(
    handle: &crate::Handle,
    n: i32,
    alpha: *const c_void,
    alpha_type: cudaDataType_t,
    x: *const c_void,
    x_type: cudaDataType_t,
    incx: i32,
    y: *mut c_void,
    y_type: cudaDataType_t,
    incy: i32,
    exec_type: cudaDataType_t,
) -> Result<()> {
    let c = cublas()?;
    let f = c.cublas_axpy_ex()?;
    check(f(
        handle.as_raw(),
        n,
        alpha,
        alpha_type,
        x,
        x_type,
        incx,
        y,
        y_type,
        incy,
        exec_type,
    ))
}

/// `result = x · y` with explicit types.
///
/// # Safety
/// Same as [`axpy`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn dot(
    handle: &crate::Handle,
    n: i32,
    x: *const c_void,
    x_type: cudaDataType_t,
    incx: i32,
    y: *const c_void,
    y_type: cudaDataType_t,
    incy: i32,
    result: *mut c_void,
    result_type: cudaDataType_t,
    exec_type: cudaDataType_t,
) -> Result<()> {
    let c = cublas()?;
    let f = c.cublas_dot_ex()?;
    check(f(
        handle.as_raw(),
        n,
        x,
        x_type,
        incx,
        y,
        y_type,
        incy,
        result,
        result_type,
        exec_type,
    ))
}

/// Conjugate dot product: `result = xᴴ · y` with explicit types.
///
/// # Safety
/// Same as [`axpy`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn dotc(
    handle: &crate::Handle,
    n: i32,
    x: *const c_void,
    x_type: cudaDataType_t,
    incx: i32,
    y: *const c_void,
    y_type: cudaDataType_t,
    incy: i32,
    result: *mut c_void,
    result_type: cudaDataType_t,
    exec_type: cudaDataType_t,
) -> Result<()> {
    let c = cublas()?;
    let f = c.cublas_dotc_ex()?;
    check(f(
        handle.as_raw(),
        n,
        x,
        x_type,
        incx,
        y,
        y_type,
        incy,
        result,
        result_type,
        exec_type,
    ))
}

/// `result = ||x||_2` with explicit types.
///
/// # Safety
/// Same as [`axpy`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn nrm2(
    handle: &crate::Handle,
    n: i32,
    x: *const c_void,
    x_type: cudaDataType_t,
    incx: i32,
    result: *mut c_void,
    result_type: cudaDataType_t,
    exec_type: cudaDataType_t,
) -> Result<()> {
    let c = cublas()?;
    let f = c.cublas_nrm2_ex()?;
    check(f(
        handle.as_raw(),
        n,
        x,
        x_type,
        incx,
        result,
        result_type,
        exec_type,
    ))
}

/// `x = alpha * x` in place with explicit types.
///
/// # Safety
/// Same as [`axpy`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn scal(
    handle: &crate::Handle,
    n: i32,
    alpha: *const c_void,
    alpha_type: cudaDataType_t,
    x: *mut c_void,
    x_type: cudaDataType_t,
    incx: i32,
    exec_type: cudaDataType_t,
) -> Result<()> {
    let c = cublas()?;
    let f = c.cublas_scal_ex()?;
    check(f(
        handle.as_raw(),
        n,
        alpha,
        alpha_type,
        x,
        x_type,
        incx,
        exec_type,
    ))
}

/// Givens rotation: `(x_i, y_i) = (c*x_i + s*y_i, -s*x_i + c*y_i)`.
///
/// # Safety
/// Same as [`axpy`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn rot(
    handle: &crate::Handle,
    n: i32,
    x: *mut c_void,
    x_type: cudaDataType_t,
    incx: i32,
    y: *mut c_void,
    y_type: cudaDataType_t,
    incy: i32,
    c_cos: *const c_void,
    s_sin: *const c_void,
    cs_type: cudaDataType_t,
    exec_type: cudaDataType_t,
) -> Result<()> {
    let c = cublas()?;
    let f = c.cublas_rot_ex()?;
    check(f(
        handle.as_raw(),
        n,
        x,
        x_type,
        incx,
        y,
        y_type,
        incy,
        c_cos,
        s_sin,
        cs_type,
        exec_type,
    ))
}
