//! Bounds validation for caller-built tensor views.
//!
//! `TensorRef` / `TensorMut` can only be built through their validating
//! constructors, but the plans re-check every view anyway (defence in depth,
//! and `new_unchecked` / offset views exist). Many kernels forward both straight to FFI (`base + Σ coord·stride`). A plan's
//! `can_implement` therefore has to prove every view stays inside its
//! `DeviceSlice` before anything is launched; comparing `data.len()` with
//! `numel()` is not enough (`shape = [n]`, `stride = [2]` touches `2n - 1`
//! elements).

use baracuda_cutlass::{Error, Result};
use baracuda_kernels_types::{TensorMut, TensorRef, VectorRef, ViewError};

/// A tensor view whose reachable extent can be compared with its backing slice.
pub(crate) trait View {
    /// Elements the view can touch (see [`baracuda_kernels_types::required_len`]),
    /// or `None` if it is not addressable inside any slice.
    fn reachable_len(&self) -> Option<usize>;
    /// Length of the backing slice, in elements.
    fn backing_len(&self) -> usize;
    /// `Π shape` — what a stride-ignoring (dense) kernel will touch — or `None`
    /// for a negative extent / overflow.
    fn dense_len(&self) -> Option<usize>;
}

fn dense_len_of(shape: &[i32]) -> Option<usize> {
    let mut n: u128 = 1;
    for &d in shape {
        n = n.checked_mul(u128::try_from(d).ok()?)?;
    }
    usize::try_from(n).ok()
}

impl<T: baracuda_types::DeviceRepr + Copy + 'static, const N: usize> View for TensorRef<'_, T, N> {
    fn reachable_len(&self) -> Option<usize> {
        self.required_len()
            .map(|n| n.div_ceil(baracuda_kernels_types::elements_per_slot::<T>()))
    }
    fn backing_len(&self) -> usize {
        self.data().len()
    }
    fn dense_len(&self) -> Option<usize> {
        dense_len_of(&self.shape())
            .map(|n| n.div_ceil(baracuda_kernels_types::elements_per_slot::<T>()))
    }
}

impl<T: baracuda_types::DeviceRepr + Copy + 'static, const N: usize> View for TensorMut<'_, T, N> {
    fn reachable_len(&self) -> Option<usize> {
        self.required_len()
            .map(|n| n.div_ceil(baracuda_kernels_types::elements_per_slot::<T>()))
    }
    fn backing_len(&self) -> usize {
        self.data().len()
    }
    fn dense_len(&self) -> Option<usize> {
        dense_len_of(&self.shape())
            .map(|n| n.div_ceil(baracuda_kernels_types::elements_per_slot::<T>()))
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

/// [`check_fits`] for kernels that ignore `stride` and address `numel()`
/// contiguous elements from the slice start: the slice must hold `Π shape`.
pub(crate) fn check_dense<V: View>(view: &V) -> Result<()> {
    match view.dense_len() {
        None => Err(Error::InvalidProblem(
            "tensor view has a negative or overflowing extent",
        )),
        Some(needed) if needed > view.backing_len() => Err(Error::BufferTooSmall {
            needed,
            got: view.backing_len(),
        }),
        Some(_) => Ok(()),
    }
}

/// Bounds check for a matrix operand (`MatrixRef` / `MatrixMut`): `major`
/// lines of `ld` units each, the last one `minor` units long, so the operand
/// touches `(major - 1) * ld + minor` units of `unit_bytes` bytes each. The
/// backing slice holds `backing_len` elements of `backing_elem_bytes` bytes.
/// (Packed int4 / bit operands count `ld` and `minor` in bytes.)
pub(crate) fn check_matrix(
    major: i64,
    minor: i64,
    ld: i64,
    unit_bytes: usize,
    backing_len: usize,
    backing_elem_bytes: usize,
) -> Result<()> {
    if major <= 0 || minor <= 0 {
        return Ok(());
    }
    if ld < 0 {
        return Err(Error::InvalidProblem(
            "matrix operand has a negative leading dimension",
        ));
    }
    let units = (major as i128 - 1) * ld as i128 + minor as i128;
    let need_bytes = units * unit_bytes as i128;
    let got_bytes = backing_len as i128 * backing_elem_bytes as i128;
    if need_bytes > got_bytes {
        let eb = backing_elem_bytes.max(1) as i128;
        return Err(Error::BufferTooSmall {
            needed: usize::try_from((need_bytes + eb - 1) / eb).unwrap_or(usize::MAX),
            got: backing_len,
        });
    }
    Ok(())
}

/// [`check_fits`] for a source view that starts `offset` elements into its
/// slice (`ContiguizePlan`'s `source_offset`): the lowest element reached is
/// `offset + Σ min(0, (shape-1)·stride)` and the highest
/// `offset + Σ max(0, (shape-1)·stride)`; both must lie inside the slice.
pub(crate) fn check_fits_at<T: baracuda_types::DeviceRepr + Copy + 'static, const N: usize>(
    view: &TensorRef<'_, T, N>,
    offset: i64,
) -> Result<()> {
    if view.shape().iter().any(|&d| d < 0) {
        return Err(Error::InvalidProblem(
            "tensor view is not addressable (negative extent)",
        ));
    }
    if view.shape().iter().any(|&d| d == 0) {
        return Ok(());
    }
    let (mut lo, mut hi) = (offset as i128, offset as i128);
    for (&d, &s) in view.shape().iter().zip(view.stride().iter()) {
        let reach = (d as i128 - 1) * s as i128;
        if reach < 0 {
            lo += reach;
        } else {
            hi += reach;
        }
    }
    if lo < 0 {
        return Err(Error::InvalidProblem(
            "tensor view is not addressable (source_offset plus negative strides              reach before the start of its buffer)",
        ));
    }
    let needed = usize::try_from(hi / baracuda_kernels_types::elements_per_slot::<T>() as i128 + 1)
        .unwrap_or(usize::MAX);
    if needed > view.data().len() {
        return Err(Error::BufferTooSmall {
            needed,
            got: view.data().len(),
        });
    }
    Ok(())
}

/// Map a failed view construction inside a plan to the kernels error type.
pub(crate) fn valid<V>(built: core::result::Result<V, ViewError>) -> Result<V> {
    built.map_err(|e| match e {
        ViewError::TooSmall { needed, got } => Error::BufferTooSmall { needed, got },
        _ => Error::InvalidProblem("tensor view is not addressable"),
    })
}

/// Bounds check for a contiguous bias vector: it must declare `n` elements and
/// its slice must hold them (the epilogue reads `bias[0..n]`).
pub(crate) fn check_vector<T: baracuda_types::DeviceRepr + Copy + 'static>(
    v: &VectorRef<'_, T>,
    n: i32,
) -> Result<()> {
    if v.len != n {
        return Err(Error::InvalidProblem("bias length must equal N"));
    }
    let needed = usize::try_from(n).unwrap_or(0);
    if needed > v.data.len() {
        return Err(Error::BufferTooSmall {
            needed,
            got: v.data.len(),
        });
    }
    Ok(())
}
