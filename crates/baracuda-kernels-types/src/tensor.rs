//! Borrowed views of device-resident N-dimensional tensors.
//!
//! Sibling to [`crate::MatrixRef`] / [`crate::MatrixMut`] (which are the
//! 2-D matrix views used by the GEMM kernel family). The tensor views
//! here are used by the elementwise / shape / reduce / norm / attention
//! op families that operate on arbitrary-rank tensors.
//!
//! The rank `N` is a compile-time `const` parameter — the same
//! `Tensor<T, N>` pattern `dfdx` uses. This gives strongest type-safety
//! (rank mismatches are caught at compile time) and keeps the descriptor
//! heap-free.
//!
//! For ops whose input and output rank differ (e.g. `meshgrid` from
//! rank-1 to rank-N, `reduce` from rank-N to rank-`N-D`), the Plan
//! exposes two const-generic rank parameters — one per operand.

use baracuda_driver::{DeviceSlice, DeviceSliceMut};
use baracuda_types::DeviceRepr;

/// Read-only view of a device-resident rank-`N` tensor.
///
/// `shape[i]` is the extent along axis `i`. `stride[i]` is the **element**
/// stride along axis `i` (memory offset to advance one step along that
/// axis). A stride of `0` along an axis signals a broadcast operand —
/// the kernel reads the same memory cell for every step along that
/// axis. The default row-major contiguous stride for shape
/// `[d0, d1, …, dN-1]` is `[d1·d2·…·dN-1, …, dN-1, 1]`.
///
/// `T` is bounded by [`DeviceRepr`] only (not by [`crate::Element`] /
/// [`crate::IntElement`]) so the same view struct can carry any scalar
/// payload — input, output, mask, or auxiliary buffer. The per-op
/// element-class enforcement happens at the Plan layer.
///
/// The fields are private: build a view with [`TensorRef::new`] (checked) or
/// [`TensorRef::contiguous`], and read it back with `data()` / `shape()` /
/// `stride()`. A struct literal does not compile:
///
/// ```compile_fail,E0451
/// use baracuda_driver::DeviceSlice;
/// use baracuda_kernels_types::TensorRef;
///
/// fn forge(data: DeviceSlice<'_, f32>) -> TensorRef<'_, f32, 1> {
///     TensorRef { data, shape: [8], stride: [1_000_000] }
/// }
/// ```
#[derive(Debug, Copy, Clone)]
pub struct TensorRef<'a, T: DeviceRepr + Copy + 'static, const N: usize> {
    // The fields are private on purpose: every constructor checks that the
    // `shape` / `stride` pair stays inside `data`, so a safe plan launch can
    // rely on it (read them back with `data()` / `shape()` / `stride()`).
    data: DeviceSlice<'a, T>,
    shape: [i32; N],
    stride: [i64; N],
}

/// Mutable view of a device-resident rank-`N` tensor.
///
/// See [`TensorRef`] for the rationale behind the relaxed `T` bound and
/// the stride / broadcast contract.
#[derive(Debug)]
pub struct TensorMut<'a, T: DeviceRepr + Copy + 'static, const N: usize> {
    // Private for the same reason as `TensorRef`'s fields.
    data: DeviceSliceMut<'a, T>,
    shape: [i32; N],
    stride: [i64; N],
}

impl<'a, T: DeviceRepr + Copy + 'static, const N: usize> TensorRef<'a, T, N> {
    /// Build a view, checking that `shape` / `stride` stay inside `data`.
    ///
    /// This and [`contiguous`](Self::contiguous) are the only safe ways to
    /// build a view, so a `TensorRef`/`TensorMut` never reaches outside its
    /// slice. (Plans still re-check every view in `can_implement` / `run`.)
    pub fn new(
        data: DeviceSlice<'a, T>,
        shape: [i32; N],
        stride: [i64; N],
    ) -> Result<Self, ViewError> {
        match required_len(&shape, &stride) {
            None => Err(ViewError::NotAddressable),
            Some(needed) if needed > data.len() => Err(ViewError::TooSmall {
                needed,
                got: data.len(),
            }),
            Some(_) => Ok(Self {
                data,
                shape,
                stride,
            }),
        }
    }

    /// Row-major contiguous view of `data` with the given `shape`.
    pub fn contiguous(data: DeviceSlice<'a, T>, shape: [i32; N]) -> Result<Self, ViewError> {
        Self::new(data, shape, contiguous_stride(shape))
    }

    /// The backing device slice.
    #[inline]
    pub fn data(&self) -> DeviceSlice<'a, T> {
        self.data
    }

    /// Extent along each axis (in elements).
    #[inline]
    pub fn shape(&self) -> [i32; N] {
        self.shape
    }

    /// Element stride along each axis. Stride `0` marks a broadcast axis.
    #[inline]
    pub fn stride(&self) -> [i64; N] {
        self.stride
    }

    /// Build a view WITHOUT checking that it stays inside `data`.
    ///
    /// # Safety
    ///
    /// Every element `Σ i_k·stride[k]` for `0 <= i_k < shape[k]` must lie in
    /// `0..data.len()` once the consuming plan has applied any base offset it
    /// takes separately (for example `ContiguizeDescriptor::source_offset`,
    /// which is what makes a negative stride legal); otherwise a plan launch
    /// reads or writes outside `data`.
    #[inline]
    pub unsafe fn new_unchecked(
        data: DeviceSlice<'a, T>,
        shape: [i32; N],
        stride: [i64; N],
    ) -> Self {
        Self {
            data,
            shape,
            stride,
        }
    }

    /// Elements this view can touch; see [`required_len`]. `None` if the
    /// shape / stride pair is not addressable inside any slice.
    #[inline]
    pub fn required_len(&self) -> Option<usize> {
        required_len(&self.shape, &self.stride)
    }

    /// Total number of logical elements (product of `shape`).
    ///
    /// Returns `1` for the rank-0 (scalar) case. Saturates on overflow
    /// rather than wrapping — a tensor with overflowing element count
    /// cannot fit in CUDA's `int64_t` element-count surface anyway, so
    /// reporting `i64::MAX` is a fine sentinel.
    #[inline]
    pub fn numel(&self) -> i64 {
        let mut n: i64 = 1;
        let mut i = 0;
        while i < N {
            n = n.saturating_mul(self.shape[i] as i64);
            i += 1;
        }
        n
    }

    /// `true` iff `stride` matches the standard row-major contiguous
    /// layout (rightmost axis has stride `1`; each prior axis multiplies
    /// by the extent to its right).
    ///
    /// The rank-0 case is always contiguous. A broadcast tensor
    /// (any `stride[i] == 0`) is *not* contiguous.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        if N == 0 {
            return true;
        }
        let mut expected: i64 = 1;
        let mut i = N;
        while i > 0 {
            i -= 1;
            if self.stride[i] != expected {
                return false;
            }
            expected = expected.saturating_mul(self.shape[i] as i64);
        }
        true
    }
}

impl<'a, T: DeviceRepr + Copy + 'static, const N: usize> TensorMut<'a, T, N> {
    /// Build a view, checking that `shape` / `stride` stay inside `data`.
    ///
    /// This and [`contiguous`](Self::contiguous) are the only safe ways to
    /// build a view, so a `TensorRef`/`TensorMut` never reaches outside its
    /// slice. (Plans still re-check every view in `can_implement` / `run`.)
    pub fn new(
        data: DeviceSliceMut<'a, T>,
        shape: [i32; N],
        stride: [i64; N],
    ) -> Result<Self, ViewError> {
        match required_len(&shape, &stride) {
            None => Err(ViewError::NotAddressable),
            Some(needed) if needed > data.len() => Err(ViewError::TooSmall {
                needed,
                got: data.len(),
            }),
            Some(_) => Ok(Self {
                data,
                shape,
                stride,
            }),
        }
    }

    /// Row-major contiguous view of `data` with the given `shape`.
    pub fn contiguous(data: DeviceSliceMut<'a, T>, shape: [i32; N]) -> Result<Self, ViewError> {
        Self::new(data, shape, contiguous_stride(shape))
    }

    /// The backing device slice.
    #[inline]
    pub fn data(&self) -> &DeviceSliceMut<'a, T> {
        &self.data
    }

    /// The backing device slice, mutably.
    #[inline]
    pub fn data_mut(&mut self) -> &mut DeviceSliceMut<'a, T> {
        &mut self.data
    }

    /// Extent along each axis (in elements).
    #[inline]
    pub fn shape(&self) -> [i32; N] {
        self.shape
    }

    /// Element stride along each axis.
    #[inline]
    pub fn stride(&self) -> [i64; N] {
        self.stride
    }

    /// Build a view WITHOUT checking that it stays inside `data`.
    ///
    /// # Safety
    ///
    /// As for [`TensorRef::new_unchecked`].
    #[inline]
    pub unsafe fn new_unchecked(
        data: DeviceSliceMut<'a, T>,
        shape: [i32; N],
        stride: [i64; N],
    ) -> Self {
        Self {
            data,
            shape,
            stride,
        }
    }

    /// Elements this view can touch; see [`required_len`]. `None` if the
    /// shape / stride pair is not addressable inside any slice.
    #[inline]
    pub fn required_len(&self) -> Option<usize> {
        required_len(&self.shape, &self.stride)
    }

    /// See [`TensorRef::numel`].
    #[inline]
    pub fn numel(&self) -> i64 {
        let mut n: i64 = 1;
        let mut i = 0;
        while i < N {
            n = n.saturating_mul(self.shape[i] as i64);
            i += 1;
        }
        n
    }

    /// See [`TensorRef::is_contiguous`].
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        if N == 0 {
            return true;
        }
        let mut expected: i64 = 1;
        let mut i = N;
        while i > 0 {
            i -= 1;
            if self.stride[i] != expected {
                return false;
            }
            expected = expected.saturating_mul(self.shape[i] as i64);
        }
        true
    }
}

/// Why a `shape` / `stride` pair cannot be a valid view of a slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ViewError {
    /// An extent is negative, or a negative stride reaches before element `0`.
    NotAddressable,
    /// The highest element the view can reach is past the end of the slice.
    TooSmall {
        /// Elements the view can reach (see [`required_len`]).
        needed: usize,
        /// Length of the backing slice.
        got: usize,
    },
}

impl core::fmt::Display for ViewError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotAddressable => f.write_str(
                "tensor view is not addressable (negative extent, or a negative stride that reaches before the start of its buffer)",
            ),
            Self::TooSmall { needed, got } => {
                write!(f, "tensor view reaches {needed} elements but its buffer holds {got}")
            }
        }
    }
}

impl core::error::Error for ViewError {}

/// Number of elements a view with this `shape` / `stride` can touch, counted
/// from the start of its backing slice: `1 + Σ (shape[i]-1)·stride[i]` over
/// axes with `shape[i] > 1` (or `0` when any extent is `0`).
///
/// Returns `None` when the view cannot be addressed inside *any* slice: a
/// negative extent, a negative stride that reaches before element `0`, or
/// an offset that overflows. This is the bound a launch must check against
/// `data.len()` before strides are handed to a kernel — `numel()` alone
/// under-counts any view with a stride larger than its extent product (for
/// example `shape = [n]`, `stride = [2]` touches `2n - 1` elements).
///
/// `stride == 0` (broadcast) axes contribute nothing.
#[must_use]
pub fn required_len(shape: &[i32], stride: &[i64]) -> Option<usize> {
    debug_assert_eq!(shape.len(), stride.len());
    let mut lo: i128 = 0;
    let mut hi: i128 = 0;
    let mut empty = false;
    for (&n, &s) in shape.iter().zip(stride) {
        if n < 0 {
            return None;
        }
        if n == 0 {
            empty = true;
            continue;
        }
        let ext = (n as i128 - 1) * s as i128;
        if ext < 0 {
            lo += ext;
        } else {
            hi += ext;
        }
    }
    if lo < 0 {
        return None;
    }
    if empty {
        return Some(0);
    }
    usize::try_from(hi + 1).ok()
}

/// Compute the row-major contiguous stride for the given `shape`.
///
/// Returns `[d1·…·dN-1, d2·…·dN-1, …, dN-1, 1]`. For the rank-0 case
/// returns the empty array.
///
/// Useful for caller-side construction of a [`TensorRef`] / [`TensorMut`]
/// over a contiguous device buffer:
///
/// ```rust,ignore
/// let shape = [8, 128, 128];
/// let stride = baracuda_kernels_types::contiguous_stride(shape);
/// let tref = TensorRef::new(buf.as_slice(), shape, stride)?;
/// ```
#[inline]
pub fn contiguous_stride<const N: usize>(shape: [i32; N]) -> [i64; N] {
    let mut stride = [0i64; N];
    if N == 0 {
        return stride;
    }
    let mut acc: i64 = 1;
    let mut i = N;
    while i > 0 {
        i -= 1;
        stride[i] = acc;
        acc = acc.saturating_mul(shape[i] as i64);
    }
    stride
}

/// Element-wise equality check between two stride arrays.
///
/// Returns `true` iff `a.len() == b.len()` and `a[i] == b[i]` for every
/// `i`. The trivial slice-compare (`a == b`) does the same thing, but
/// this helper exists as a one-place callsite that the
/// **Phase 62 strided in-place aliasing contract** can cite:
///
/// > For unary / binary / ternary / parameterized-unary / affine
/// > strided `_run` launchers, aliasing the output `y` with an input
/// > pointer is safe IF AND ONLY IF the aliased input's stride array
/// > equals `stride_y` element-for-element.
///
/// Callers (Fuel's executor, or any other consumer doing in-place
/// dispatch over a strided forward symbol with `x_ptr == y_ptr` /
/// `a_ptr == y_ptr` / etc.) should invoke this before dispatching to
/// validate the precondition. The kernel does no validation; aliasing
/// with unequal strides is silent data corruption.
///
/// Cheap (single pass over two short arrays), `#[inline]`, no
/// allocations. Trivially `const`-eval-friendly when called with
/// fixed-length arrays.
///
/// # Examples
///
/// ```rust
/// use baracuda_kernels_types::{contiguous_stride, strides_equal};
///
/// let shape = [4, 8];
/// let s_contig = contiguous_stride(shape);
/// let s_other  = [8_i64, 1_i64];  // also contiguous
/// let s_perm   = [1_i64, 4_i64];  // transposed
///
/// assert!(strides_equal(&s_contig, &s_other));
/// assert!(!strides_equal(&s_contig, &s_perm));
/// // Length mismatch is always unequal.
/// assert!(!strides_equal(&[1_i64], &[1_i64, 2_i64]));
/// ```
#[inline]
pub fn strides_equal(a: &[i64], b: &[i64]) -> bool {
    a == b
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- contiguous_stride ----

    #[test]
    fn contiguous_stride_rank0_is_empty() {
        let s: [i64; 0] = contiguous_stride([]);
        assert_eq!(s, [] as [i64; 0]);
    }

    #[test]
    fn contiguous_stride_rank1_is_one() {
        assert_eq!(contiguous_stride([5]), [1]);
        assert_eq!(contiguous_stride([100]), [1]);
    }

    #[test]
    fn contiguous_stride_rank2_row_major() {
        // shape [4, 8] → stride [8, 1]
        assert_eq!(contiguous_stride([4, 8]), [8, 1]);
        // shape [3, 5] → stride [5, 1]
        assert_eq!(contiguous_stride([3, 5]), [5, 1]);
    }

    #[test]
    fn contiguous_stride_rank3() {
        // shape [2, 4, 8] → stride [32, 8, 1]
        assert_eq!(contiguous_stride([2, 4, 8]), [32, 8, 1]);
    }

    #[test]
    fn contiguous_stride_rank4() {
        // shape [2, 3, 5, 7] → stride [105, 35, 7, 1]
        assert_eq!(contiguous_stride([2, 3, 5, 7]), [105, 35, 7, 1]);
    }

    // ---- strides_equal ----
    //
    // This helper anchors the Phase 62 strided in-place aliasing
    // contract — see the docstring on `strides_equal` for the
    // load-bearing semantics. Tests cover the patterns Fuel's
    // executor will exercise.

    #[test]
    fn strides_equal_empty_slices_are_equal() {
        let a: [i64; 0] = [];
        let b: [i64; 0] = [];
        assert!(strides_equal(&a, &b));
    }

    #[test]
    fn strides_equal_identical_arrays() {
        assert!(strides_equal(&[1, 2, 3], &[1, 2, 3]));
        assert!(strides_equal(&[8, 1], &[8, 1]));
        assert!(strides_equal(&[1024, 32, 1], &[1024, 32, 1]));
    }

    #[test]
    fn strides_equal_different_values() {
        assert!(!strides_equal(&[1, 2, 3], &[1, 2, 4]));
        assert!(!strides_equal(&[8, 1], &[1, 8]));
    }

    #[test]
    fn strides_equal_different_lengths() {
        assert!(!strides_equal(&[1, 2], &[1, 2, 3]));
        assert!(!strides_equal(&[1], &[]));
    }

    #[test]
    fn strides_equal_matches_contiguous_stride_output() {
        // Phase 62's intended use: caller computes contiguous_stride
        // for both x and y, then checks equality before in-place
        // dispatch. Round-trip should always be equal.
        for shape in &[[4, 8], [16, 32], [3, 5]] {
            let s = contiguous_stride(*shape);
            assert!(strides_equal(&s, &s));
        }
    }

    #[test]
    fn strides_equal_detects_transpose() {
        // A transposed view (swapped axis order) has a different
        // stride array than the original. This is precisely the case
        // that makes in-place dispatch UNSAFE — strides_equal must
        // return false here.
        let contig = contiguous_stride([4, 8]); // [8, 1]
        let transposed = [1_i64, 4_i64]; // shape [4,8] transposed → stride [1, 4] mapped to [4,1] on T()
        assert!(!strides_equal(&contig, &transposed));
    }

    #[test]
    fn strides_equal_detects_broadcast_zero_stride() {
        // A broadcast operand (one axis with stride 0) is unequal to
        // the regular target — confirms strides_equal will reject
        // aliasing a broadcast input with the output.
        let contig = contiguous_stride([4, 8]); // [8, 1]
        let broadcast = [0_i64, 1_i64]; // broadcast over axis 0
        assert!(!strides_equal(&contig, &broadcast));
    }

    #[test]
    fn strides_equal_detects_negative_stride() {
        // Flipped views have negative strides — different from any
        // positive-stride layout, so aliasing dispatch must reject.
        let contig = contiguous_stride([4]); // [1]
        let flipped = [-1_i64];
        assert!(!strides_equal(&contig, &flipped));
    }

    // ---- TensorRef::numel + is_contiguous parity check ----

    #[test]
    fn numel_matches_shape_product() {
        // Verify shape [4, 8] has numel 32 via direct construction
        // (avoid needing a real DeviceSlice).
        // We can't easily construct TensorRef without device buffers,
        // so just verify the contiguous_stride math composes
        // sensibly: 4 * 8 * 1 = 32 (product of stride[0] * shape[-1]).
        let shape = [4, 8];
        let stride = contiguous_stride(shape);
        // stride[0] * shape[0] should give the total element count.
        assert_eq!(stride[0] * shape[0] as i64, 32);
    }
}

#[cfg(test)]
mod required_len_tests {
    use super::required_len;

    #[test]
    fn contiguous_equals_numel() {
        assert_eq!(
            required_len(&[8, 128, 128], &[128 * 128, 128, 1]),
            Some(8 * 128 * 128)
        );
    }

    #[test]
    fn stride_larger_than_extent_reaches_further_than_numel() {
        // 64 elements at stride 2 touch indices 0..=126.
        assert_eq!(required_len(&[64], &[2]), Some(127));
    }

    #[test]
    fn broadcast_axis_costs_nothing() {
        assert_eq!(required_len(&[4, 16], &[0, 1]), Some(16));
    }

    #[test]
    fn transposed_view_needs_the_whole_buffer() {
        assert_eq!(required_len(&[3, 5], &[1, 3]), Some(15));
    }

    #[test]
    fn empty_view_needs_nothing() {
        assert_eq!(required_len(&[0, 7], &[7, 1]), Some(0));
    }

    #[test]
    fn negative_extent_or_negative_reach_is_unaddressable() {
        assert_eq!(required_len(&[-1], &[1]), None);
        assert_eq!(required_len(&[4], &[-1]), None);
        // a negative stride on a size-1 axis never moves, so it is fine
        assert_eq!(required_len(&[1, 4], &[-9, 1]), Some(4));
    }

    #[test]
    fn overflow_is_unaddressable_not_wrapped() {
        assert_eq!(
            required_len(&[i32::MAX, i32::MAX], &[i64::MAX, i64::MAX]),
            None
        );
    }
}
