//! Shared helpers for the loss-family plans.

use core::ffi::c_void;

use baracuda_cutlass::{Error, Result};
use baracuda_kernels_types::{Element, ElementKind, Workspace};

/// Validate a shape array: dims must be ≥ 0, rank must be ≤ 8.
pub(crate) fn validate_shape(shape: &[i32], rank: usize) -> Result<()> {
    if rank > 8 {
        return Err(Error::Unsupported(
            "baracuda-kernels loss plan: rank > 8 not supported",
        ));
    }
    for &d in shape.iter() {
        if d < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels loss plan: shape dims must be non-negative",
            ));
        }
    }
    Ok(())
}

/// Reject dtypes outside `{f32, f16, bf16, f64}`.
pub(crate) fn check_supported_dtype<T: Element>() -> Result<()> {
    let dtype_in_fp_family = matches!(
        T::KIND,
        ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
    );
    if !dtype_in_fp_family {
        return Err(Error::Unsupported(
            "baracuda-kernels loss plan: only {f32, f16, bf16, f64} wired today",
        ));
    }
    Ok(())
}

/// Resolve a `Workspace` enum to a (ptr, bytes) pair, checking that the
/// provided buffer meets the required size. Returns `(null, 0)` for the
/// zero-bytes case.
pub(crate) fn unpack_workspace<'a>(
    workspace: Workspace<'a>,
    needed: usize,
) -> Result<(*mut c_void, usize)> {
    if needed == 0 {
        return Ok((core::ptr::null_mut(), 0));
    }
    match workspace {
        Workspace::None => Err(Error::WorkspaceTooSmall { needed, got: 0 }),
        Workspace::Borrowed(slice) => {
            if slice.len() < needed {
                return Err(Error::WorkspaceTooSmall {
                    needed,
                    got: slice.len(),
                });
            }
            Ok((slice.as_raw().0 as *mut c_void, slice.len()))
        }
    }
}

/// Map a launcher status code to a `Result<()>`.
pub(crate) fn map_status(code: i32) -> Result<()> {
    match code {
        0 => Ok(()),
        1 => Err(Error::MisalignedOperand),
        2 => Err(Error::InvalidProblem(
            "baracuda-kernels-sys reported invalid problem",
        )),
        3 => Err(Error::Unsupported(
            "baracuda-kernels-sys reported unsupported configuration",
        )),
        4 => Err(Error::WorkspaceTooSmall { needed: 0, got: 0 }),
        n => Err(Error::CutlassInternal(n)),
    }
}
