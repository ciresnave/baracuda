//! The `KernelArg` trait: a uniform way to hand values to
//! `cuLaunchKernel` / `cudaLaunchKernel`, which expect `void**` — an array
//! of pointers to each argument.
//!
//! Implementors must be [`crate::DeviceRepr`] so we know the layout is
//! ABI-stable, and must produce a pointer whose pointee will remain valid
//! for the duration of the kernel launch.

use core::ffi::c_void;

use crate::device_repr::DeviceRepr;

/// A value that can be marshalled into the `void** kernelParams` slot of
/// `cuLaunchKernel` / `cudaLaunchKernel`.
///
/// # Safety
///
/// Implementors must uphold:
///
/// 1. [`Self::as_kernel_arg_ptr`] returns a pointer whose pointee is a valid
///    `DeviceRepr` value of the correct type for the target kernel slot.
/// 2. The returned pointer remains valid until the kernel launch has been
///    submitted to the stream (not necessarily completed — the runtime
///    copies argument bytes during submission).
/// 3. Concurrent kernel launches using the same argument value must tolerate
///    shared access; in practice this means the referent is either `Copy`
///    or borrowed immutably for the duration of submission.
pub unsafe trait KernelArg {
    fn as_kernel_arg_ptr(&self) -> *mut c_void;
}

// SAFETY: &T where T: DeviceRepr points to a valid, ABI-stable value.
// The launch API reads argument bytes during submission, so an immutable
// borrow for the duration of `launch()` is enough.
unsafe impl<T: DeviceRepr> KernelArg for &T {
    #[inline]
    fn as_kernel_arg_ptr(&self) -> *mut c_void {
        *self as *const T as *mut c_void
    }
}

// SAFETY: same as above; &mut grants even stricter exclusive access.
unsafe impl<T: DeviceRepr> KernelArg for &mut T {
    #[inline]
    fn as_kernel_arg_ptr(&self) -> *mut c_void {
        *self as *const T as *mut c_void
    }
}
