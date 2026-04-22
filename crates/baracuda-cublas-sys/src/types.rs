//! Handle + enum types for cuBLAS.

#![allow(non_camel_case_types)]

use core::ffi::c_void;

/// Opaque cuBLAS handle.
pub type cublasHandle_t = *mut c_void;

/// Transpose selector for matrix arguments.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cublasOperation_t {
    /// No transpose.
    N = 0,
    /// Transpose.
    T = 1,
    /// Conjugate transpose.
    C = 2,
}

/// Pointer-mode selector: scalar alpha/beta arguments can live on host or device.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cublasPointerMode_t {
    Host = 0,
    Device = 1,
}

/// Atomics mode (relevant for some cuBLAS routines that have atomic variants).
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cublasAtomicsMode_t {
    NotAllowed = 0,
    Allowed = 1,
}

/// Math mode / tensor-core enablement.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cublasMath_t {
    Default = 0,
    TensorOpMath = 1,
    Pedantic = 2,
    Tf32TensorOp = 3,
    DisallowReducedPrecisionReduction = 16,
}
