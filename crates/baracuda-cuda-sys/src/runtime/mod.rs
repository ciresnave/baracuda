#![allow(non_camel_case_types)]

//! Raw FFI + dynamic loader for the CUDA Runtime API (`libcudart`).
//!
//! Parallels the [`crate::driver`] module. Runtime handles
//! (`cudaStream_t`, `cudaEvent_t`, ...) are typedef-compatible with
//! the Driver API's `CUstream` / `CUevent` / ... so
//! `baracuda-runtime` and `baracuda-driver` can freely convert between
//! the two.

pub mod functions;
pub mod loader;
pub mod status;
pub mod types;

pub use loader::{runtime, Runtime};
pub use status::cudaError_t;
pub use types::{
    cudaArray_t, cudaEvent_t, cudaExternalMemory_t, cudaExternalSemaphore_t, cudaGraphExec_t,
    cudaGraphNode_t, cudaGraph_t, cudaKernel_t, cudaLibrary_t, cudaMemPool_t, cudaMemcpyKind,
    cudaStream_t, cudaUserObject_t, dim3,
};
