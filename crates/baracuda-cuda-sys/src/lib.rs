//! Raw FFI + dynamic loader for the CUDA Driver API (and, in later sprint
//! days, the Runtime API). Consumed by the safe wrappers in
//! `baracuda-driver` / `baracuda-runtime`.
//!
//! # Layout
//!
//! - [`types`] — opaque handle types (`CUcontext`, `CUstream`, ...),
//!   integer handle newtypes (`CUdevice`, `CUdeviceptr`), and flag modules.
//! - [`status`] — [`CUresult`] with its [`baracuda_types::CudaStatus`] impl.
//! - [`functions`] — `PFN_*` function-pointer type aliases.
//! - [`mod@driver`] — the [`Driver`] struct and the process-wide [`driver()`]
//!   accessor that loads `libcuda` once via `libloading` and resolves every
//!   other symbol through `cuGetProcAddress`.
//!
//! # Dynamic loading
//!
//! Nothing in this crate is linked against `libcuda` at build time. On
//! machines without CUDA, calling [`driver()`] returns
//! [`baracuda_core::LoaderError::LibraryNotFound`] — callers never crash
//! merely by linking this crate.

#![warn(missing_debug_implementations)]

pub mod driver;
pub mod functions;
pub mod runtime;
pub mod status;
pub mod types;

pub use driver::{driver, Driver};
pub use runtime::{runtime, Runtime};
pub use status::CUresult;
pub use types::*;
