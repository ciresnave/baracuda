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
