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
#[derive(Debug, Copy, Clone)]
pub struct TensorRef<'a, T: DeviceRepr + Copy + 'static, const N: usize> {
    /// Device-resident element storage.
    pub data: DeviceSlice<'a, T>,
    /// Extent along each axis (in elements).
    pub shape: [i32; N],
    /// Element stride along each axis. Stride `0` marks a broadcast axis.
    pub stride: [i64; N],
}

/// Mutable view of a device-resident rank-`N` tensor.
///
/// See [`TensorRef`] for the rationale behind the relaxed `T` bound and
/// the stride / broadcast contract.
#[derive(Debug)]
pub struct TensorMut<'a, T: DeviceRepr + Copy + 'static, const N: usize> {
    /// Device-resident element storage.
    pub data: DeviceSliceMut<'a, T>,
    /// Extent along each axis (in elements).
    pub shape: [i32; N],
    /// Element stride along each axis.
    pub stride: [i64; N],
}

impl<'a, T: DeviceRepr + Copy + 'static, const N: usize> TensorRef<'a, T, N> {
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
/// let tref = TensorRef { data: buf.as_slice(), shape, stride };
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
        let transposed = [1_i64, 4_i64];        // shape [4,8] transposed → stride [1, 4] mapped to [4,1] on T()
        assert!(!strides_equal(&contig, &transposed));
    }

    #[test]
    fn strides_equal_detects_broadcast_zero_stride() {
        // A broadcast operand (one axis with stride 0) is unequal to
        // the regular target — confirms strides_equal will reject
        // aliasing a broadcast input with the output.
        let contig = contiguous_stride([4, 8]); // [8, 1]
        let broadcast = [0_i64, 1_i64];          // broadcast over axis 0
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
