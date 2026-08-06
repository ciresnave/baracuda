//! # baracuda-cuda-emit
//!
//! The CUDA backend for the neutral kernel generator [`baracuda_kernelgen`]: the
//! `Cuda` [`Backend`](baracuda_kernelgen::backend::Backend) implementation
//! (IR → `.cu` emitter), the NVRTC on-demand JIT compiler, and the Fuel
//! kernel-seam `Synthesizer`. It supplies the CUDA-specific final emit + compile
//! stage from OUTSIDE the neutral generator core — the swappable-backend model
//! where the neutral `baracuda-kernelgen` owns the IR + transforms and each
//! vendor backend (CUDA here, Slang elsewhere) provides the language emitter.
//!
//! **Scaffold:** the CUDA backend is being carved in from `baracuda-kernelgen`
//! (`cuda.rs` + `NvrtcCompiler` + `BaracudaSynthesizer`), populated incrementally
//! by the extraction. This crate is not yet functional on its own.
