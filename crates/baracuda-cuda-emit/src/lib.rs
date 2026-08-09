//! # baracuda-cuda-emit
//!
//! The CUDA backend for the neutral kernel generator [`unpopped`]: the
//! `Cuda` [`Backend`](unpopped::backend::Backend) implementation
//! (IR → `.cu` emitter), the NVRTC on-demand JIT compiler, and the Fuel
//! kernel-seam `Synthesizer`. It supplies the CUDA-specific final emit + compile
//! stage from OUTSIDE the neutral generator core — the swappable-backend model
//! where the neutral `unpopped` owns the IR + transforms and each
//! vendor backend (CUDA here, Slang elsewhere) provides the language emitter.
//!
//! The CUDA backend was carved out of `baracuda-kernelgen` (carve step 2 of the
//! Unpopped extraction): the [`Cuda`] emitter (`cuda`), the NVRTC JIT compiler
//! (`nvrtc`, `--features nvrtc`), and the Fuel kernel-seam `BaracudaSynthesizer`
//! (`seam`, `--features seam`) all live here now; `unpopped` is the
//! neutral generator.

pub mod cuda;
pub use cuda::{Cuda, emit_cast_helper, emit_coord_unravel_helper, emit_dtype_promote_helper};

/// The NVRTC on-demand JIT compiler (`NvrtcCompiler`) — the production
/// source → PTX [`Compiler`](unpopped::Compiler). Behind `--features
/// nvrtc` because it needs the nvrtc runtime.
#[cfg(feature = "nvrtc")]
pub mod nvrtc;
#[cfg(feature = "nvrtc")]
pub use nvrtc::NvrtcCompiler;

/// The live Fuel kernel-seam `Synthesizer` endpoint (`BaracudaSynthesizer`) —
/// wires the CUDA backend into Fuel's §5 JIT seam. Behind `--features seam`.
#[cfg(feature = "seam")]
pub mod seam;
#[cfg(feature = "seam")]
pub use seam::BaracudaSynthesizer;
