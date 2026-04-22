//! Raw FFI + dynamic loader for NVIDIA cuBLAS.
//!
//! `baracuda-cublas` wraps this with a safe, typed API. Use this crate
//! directly only if you need a function that the safe layer hasn't wrapped
//! yet (in which case please file a bug).

#![allow(non_camel_case_types)]
#![warn(missing_debug_implementations)]

pub mod functions;
pub mod loader;
pub mod status;
pub mod types;

pub use loader::{cublas, Cublas};
pub use status::cublasStatus_t;
pub use types::{
    cublasAtomicsMode_t, cublasHandle_t, cublasMath_t, cublasOperation_t, cublasPointerMode_t,
};
