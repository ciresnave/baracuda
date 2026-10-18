//! Bounds validation for caller-built tensor views.
//!
//! `TensorRef` / `TensorMut` expose `shape` and `stride` as public fields, and
//! many kernels forward both straight to FFI (`base + Σ coord·stride`). A plan's
//! `can_implement` therefore has to prove every view stays inside its
//! `DeviceSlice` before anything is launched; comparing `data.len()` with
//! `numel()` is not enough (`shape = [n]`, `stride = [2]` touches `2n - 1`
//! elements).

use baracuda_cutlass::{Error, Result};
use baracuda_kernels_types::{TensorMut, TensorRef};

/// A tensor view whose reachable extent can be compared with its backing slice.
pub(crate) trait View {
    /// Elements the view can touch (see [`baracuda_kernels_types::required_len`]),
    /// or `None` if it is not addressable inside any slice.
    fn reachable_len(&self) -> Option<usize>;
    /// Length of the backing slice, in elements.
    fn backing_len(&self) -> usize;
}

impl<T: baracuda_types::DeviceRepr + Copy + 'static, const N: usize> View for TensorRef<'_, T, N> {
    fn reachable_len(&self) -> Option<usize> {
        self.required_len()
    }
    fn backing_len(&self) -> usize {
        self.data.len()
    }
}

impl<T: baracuda_types::DeviceRepr + Copy + 'static, const N: usize> View for TensorMut<'_, T, N> {
    fn reachable_len(&self) -> Option<usize> {
        self.required_len()
    }
    fn backing_len(&self) -> usize {
        self.data.len()
    }
}

/// Reject a view whose shape / strides can address memory outside its slice.
///
/// Errors: [`Error::InvalidProblem`] for a negative extent or a stride that
/// reaches before element 0; [`Error::BufferTooSmall`] when the highest
/// reachable element is past the end of the slice.
pub(crate) fn check_fits<V: View>(view: &V) -> Result<()> {
    match view.reachable_len() {
        None => Err(Error::InvalidProblem(
            "tensor view is not addressable (negative extent, or a negative stride \
             that reaches before the start of its buffer)",
        )),
        Some(needed) if needed > view.backing_len() => Err(Error::BufferTooSmall {
            needed,
            got: view.backing_len(),
        }),
        Some(_) => Ok(()),
    }
}
