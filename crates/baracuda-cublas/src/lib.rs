//! Safe Rust wrappers for NVIDIA cuBLAS (+ cuBLASLt + cuBLASXt).
//!
//! Layered on top of [`baracuda-cublas-sys`](https://docs.rs/baracuda-cublas-sys).
//! Use this crate directly for typed, RAII-managed cuBLAS handles +
//! ops; reach for `-sys` only when adding a function the safe layer
//! doesn't expose yet.
//!
//! ## Scope
//!
//! - Level-1 BLAS (vector ops: `axpy`, `scal`, `dot`, `nrm2`, `asum`,
//!   `amax`, `amin`, `rot`, `copy`, `swap`) for f32/f64.
//! - Level-2 BLAS (matrix-vector: `gemv`, `gbmv`, `symv`, `trmv`,
//!   `trsv`, etc.).
//! - Level-3 BLAS — the dominant compute path:
//!   - `gemm` for f32 / f64 (`Sgemm` / `Dgemm`)
//!   - `gemmEx` / `gemmStridedBatchedEx` for the mixed-precision
//!     paths (f16/bf16 inputs × f32 accumulator), used by
//!     `baracuda-kernels`'s GEMM dispatcher
//!   - Strided + non-strided batched gemm
//!   - TRSM / TRMM / SYRK / HERK
//! - **cuBLASLt** ([`lt`] module) — modern matmul descriptor API
//!   used by FlashInfer + Megatron-LM TP for higher-throughput
//!   epilogue fusion.
//! - **cuBLASXt** ([`xt`] module) — multi-GPU dense GEMM for
//!   workstation workloads.
//! - **Direct batched** ([`direct_batched`]) — `cublasS/D/CgemmBatched`
//!   for pointer-array batches (used by `baracuda-kernels`'s
//!   `BatchedOrmqrWyPlan`).
//!
//! ## Usage
//!
//! ```no_run
//! use baracuda_driver::{Context, Device, DeviceBuffer};
//! use baracuda_cublas::{gemm, Handle, Op};
//!
//! # fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! let device = Device::get(0)?;
//! let ctx = Context::new(&device)?;
//! let handle = Handle::new()?;
//!
//! // 2×3 × 3×2 = 2×2, column-major.
//! let a_host: Vec<f32> = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]; // 2×3
//! let b_host: Vec<f32> = vec![7.0, 9.0, 11.0, 8.0, 10.0, 12.0]; // 3×2
//! let a = DeviceBuffer::from_slice(&ctx, &a_host)?;
//! let b = DeviceBuffer::from_slice(&ctx, &b_host)?;
//! let mut c: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 4)?;
//!
//! gemm(&handle, Op::N, Op::N, 2, 2, 3, 1.0, &a, 2, &b, 3, 0.0, &mut c, 2)?;
//! # Ok(()) }
//! ```
//!
//! ```no_run
//! use baracuda_driver::{Context, Device, DeviceBuffer};
//! use baracuda_cublas::{gemm, Handle, Op};
//!
//! # fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! let device = Device::get(0)?;
//! let ctx = Context::new(&device)?;
//! let handle = Handle::new()?;
//!
//! // 2×3 × 3×2 = 2×2, column-major.
//! let a_host: Vec<f32> = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]; // 2×3
//! let b_host: Vec<f32> = vec![7.0, 9.0, 11.0, 8.0, 10.0, 12.0]; // 3×2
//! let a = DeviceBuffer::from_slice(&ctx, &a_host)?;
//! let b = DeviceBuffer::from_slice(&ctx, &b_host)?;
//! let mut c: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 4)?;
//!
//! gemm(&handle, Op::N, Op::N, 2, 2, 3, 1.0, &a, 2, &b, 3, 0.0, &mut c, 2)?;
//! # Ok(()) }
//! ```
//!
//! cuBLAS ships as part of the standard CUDA Toolkit — no separate
//! download needed.

#![warn(missing_debug_implementations)]

pub mod batched;
pub mod blas_scalar;
pub mod direct_batched;
pub mod error;
pub mod ex;
pub mod handle;
pub mod level1;
pub mod level2;
pub mod level3;
pub mod lt;
pub mod xt;

pub use baracuda_cublas_sys::functions::{cublasComputeType_t, cudaDataType_t};
pub use baracuda_cublas_sys::{cublasMath_t, cublasPointerMode_t};
pub use batched::{gemm_batched, gemm_ex, gemm_strided_batched_ex, BatchedGemmScalar};
pub use blas_scalar::{axpy, gemm, gemm_strided_batched, BlasScalar, Op};
pub use error::{Error, Result};
pub use handle::Handle;
pub use level1::{asum, copy, dot, iamax, iamin, nrm2, scal, L1Scalar};
pub use level2::{gemv, ger, symv, syr, trmv, trsv, L2Real, L2Scalar};
pub use level3::{
    hemm, herk, symm, syrk, trmm, trsm, Diag, Fill, HermitianScalar, L3Scalar, Side,
};
