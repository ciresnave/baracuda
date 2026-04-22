//! `baracuda` — idiomatic Rust wrappers for the NVIDIA CUDA stack.
//!
//! Umbrella crate. Re-exports individual safe-API crates behind Cargo
//! features. All features are opt-in; defaults enable just `driver` +
//! `runtime` so downstream consumers pay for only what they use.
//!
//! # Feature matrix
//!
//! | Feature     | Re-exports                                    |
//! |-------------|-----------------------------------------------|
//! | `driver`    | [`driver`] — CUDA Driver API                  |
//! | `runtime`   | [`runtime`] — CUDA Runtime API                |
//! | `nvrtc`     | [`nvrtc`] — runtime C++→PTX compiler          |
//! | `nvjitlink` | [`nvjitlink`] — CUDA 12+ JIT linker           |
//! | `cublas`    | [`cublas`] — BLAS                             |
//! | `curand`    | [`curand`] — RNG                              |
//! | `cufft`     | [`cufft`] — FFT                               |
//! | `cusparse`  | [`cusparse`] — sparse linear algebra          |
//! | `cusolver`  | [`cusolver`] — dense/sparse solvers           |
//! | `cudnn`     | [`cudnn`] — deep-learning primitives          |
//! | `nccl`      | [`nccl`] — multi-GPU collectives              |
//! | `npp`       | [`npp`] — performance primitives              |
//! | `nvjpeg`    | [`nvjpeg`] — GPU JPEG codec                   |
//! | `nvcomp`    | [`nvcomp`] — GPU compression (scaffolding)    |
//! | `cvcuda`    | [`cvcuda`] — CV-CUDA (scaffolding)            |
//! | `nvml`      | [`nvml`] — driver-bundled GPU monitoring      |
//! | `cufile`    | [`cufile`] — GPUDirect Storage (Linux only)   |
//!
//! Bundles: `math` = cuBLAS + cuRAND + cuFFT + cuSPARSE + cuSOLVER;
//! `imaging` = NPP + nvJPEG + CV-CUDA; `ml` = driver + runtime + nvrtc +
//! nvjitlink + math + cuDNN + NCCL; `full` = ml + imaging + nvcomp +
//! nvml + cufile.
//!
//! # Quickstart
//!
//! ```toml
//! [dependencies]
//! baracuda = "0.1"
//! ```
//!
//! ```no_run
//! use baracuda::driver::{Context, Device, DeviceBuffer};
//! # fn main() -> baracuda::driver::Result<()> {
//! let device = Device::get(0)?;
//! let ctx = Context::new(&device)?;
//! let data = DeviceBuffer::from_slice(&ctx, &[1.0f32, 2.0, 3.0])?;
//! # let _ = data;
//! # Ok(())
//! # }
//! ```
//!
//! # Shared vocabulary
//!
//! [`types`] (always available, no feature flag) exposes `Half`,
//! `BFloat16`, `Complex32`, `Complex64`, `DeviceRepr`, `CudaVersion`,
//! etc. — the types shared across every safe crate.

#![warn(missing_debug_implementations)]

/// Shared type vocabulary (re-export of `baracuda-types`).
pub use baracuda_types as types;

/// CUDA Driver API (enabled with the `driver` feature).
#[cfg(feature = "driver")]
pub use baracuda_driver as driver;

/// CUDA Runtime API (enabled with the `runtime` feature).
#[cfg(feature = "runtime")]
pub use baracuda_runtime as runtime;

/// NVRTC — runtime CUDA C++ → PTX compiler (`nvrtc` feature).
#[cfg(feature = "nvrtc")]
pub use baracuda_nvrtc as nvrtc;

/// nvJitLink — CUDA 12+ JIT linker (`nvjitlink` feature).
#[cfg(feature = "nvjitlink")]
pub use baracuda_nvjitlink as nvjitlink;

/// cuBLAS — GPU-accelerated BLAS (`cublas` feature).
#[cfg(feature = "cublas")]
pub use baracuda_cublas as cublas;

/// cuRAND — GPU random-number generation (`curand` feature).
#[cfg(feature = "curand")]
pub use baracuda_curand as curand;

/// cuFFT — GPU FFT (`cufft` feature).
#[cfg(feature = "cufft")]
pub use baracuda_cufft as cufft;

/// cuSPARSE — sparse linear algebra (`cusparse` feature).
#[cfg(feature = "cusparse")]
pub use baracuda_cusparse as cusparse;

/// cuSOLVER — dense + sparse solvers (`cusolver` feature).
#[cfg(feature = "cusolver")]
pub use baracuda_cusolver as cusolver;

/// cuDNN — deep-learning primitives (`cudnn` feature).
#[cfg(feature = "cudnn")]
pub use baracuda_cudnn as cudnn;

/// NCCL — multi-GPU collective communication (`nccl` feature).
#[cfg(feature = "nccl")]
pub use baracuda_nccl as nccl;

/// TensorRT — high-performance inference runtime (`tensorrt` feature).
#[cfg(feature = "tensorrt")]
pub use baracuda_tensorrt as tensorrt;

/// cuDF — RAPIDS GPU DataFrames (`cudf` feature; skeleton over emerging libcudf_c).
#[cfg(feature = "cudf")]
pub use baracuda_cudf as cudf;

/// NPP — NVIDIA Performance Primitives (`npp` feature).
#[cfg(feature = "npp")]
pub use baracuda_npp as npp;

/// nvJPEG — GPU JPEG codec (`nvjpeg` feature).
#[cfg(feature = "nvjpeg")]
pub use baracuda_nvjpeg as nvjpeg;

/// nvCOMP — GPU compression (`nvcomp` feature; scaffolding only at v0.1).
#[cfg(feature = "nvcomp")]
pub use baracuda_nvcomp as nvcomp;

/// CV-CUDA — computer-vision operators (`cvcuda` feature; scaffolding only at v0.1).
#[cfg(feature = "cvcuda")]
pub use baracuda_cvcuda as cvcuda;

/// NVML — driver-bundled GPU monitoring (`nvml` feature).
#[cfg(feature = "nvml")]
pub use baracuda_nvml as nvml;

/// cuFile — GPUDirect Storage (`cufile` feature; Linux only).
#[cfg(feature = "cufile")]
pub use baracuda_cufile as cufile;
