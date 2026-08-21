//! The device-view adapter: build a driver-free [`OperandDesc`] from a borrowed
//! device [`TensorRef`].
//!
//! [`OperandDesc`] is defined in the driver-free `unpopped-vocab` crate,
//! so its `from_tensor_ref` constructor — which needs the device-coupled
//! [`TensorRef`] — cannot be an inherent method there (orphan rule + the driver
//! dependency). It lives here as the [`OperandDescExt`] extension trait instead;
//! bring the trait into scope and `OperandDesc::from_tensor_ref(view, align)`
//! keeps working. All [`OperandDesc`] fields are `pub`, so the extension
//! constructs it directly. The driver-free constructor `OperandDesc::new` stays
//! on the type itself.

use baracuda_types::DeviceRepr;
use unpopped_vocab::{KernelDtype, MAX_RANK, OperandDesc};

use crate::TensorRef;

/// Extension trait adding the [`TensorRef`]-based constructor to
/// [`OperandDesc`].
pub trait OperandDescExt {
    /// Build an operand description from a borrowed device tensor view.
    ///
    /// `align_bytes` is supplied by the caller (it knows its allocation /
    /// view alignment — a base `cudaMalloc` is 256-byte aligned, but a sub-view
    /// may be less). dtype is taken from `T` via [`KernelDtype::KIND`].
    ///
    /// # Panics
    /// Panics if `N > MAX_RANK`.
    #[must_use]
    fn from_tensor_ref<T, const N: usize>(view: &TensorRef<'_, T, N>, align_bytes: u32) -> Self
    where
        T: KernelDtype + DeviceRepr + Copy + 'static;
}

impl OperandDescExt for OperandDesc {
    fn from_tensor_ref<T, const N: usize>(view: &TensorRef<'_, T, N>, align_bytes: u32) -> Self
    where
        T: KernelDtype + DeviceRepr + Copy + 'static,
    {
        assert!(N <= MAX_RANK, "rank {N} exceeds MAX_RANK {MAX_RANK}");
        let mut shape = [0i64; MAX_RANK];
        let mut strides = [0i64; MAX_RANK];
        for d in 0..N {
            shape[d] = i64::from(view.shape[d]);
            strides[d] = view.stride[d];
        }
        // `OperandDesc` is `#[non_exhaustive]` in unpopped-vocab 0.2.0 — construct
        // via `new` (the plain non-quant/non-symbolic constructor), which zero-pads
        // shape/strides to MAX_RANK identically to the former struct literal.
        OperandDesc::new(N, &shape[..N], &strides[..N], T::KIND, align_bytes)
    }
}
