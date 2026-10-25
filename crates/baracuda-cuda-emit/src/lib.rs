//! # baracuda-cuda-emit
//!
//! The CUDA backend for the neutral kernel generator [`unpopped`]: the
//! `Cuda` [`Backend`](unpopped::backend::Backend) implementation
//! (IR → `.cu` emitter) and the NVRTC on-demand JIT compiler. It supplies the CUDA-specific final emit + compile
//! stage from OUTSIDE the neutral generator core — the swappable-backend model
//! where the neutral `unpopped` owns the IR + transforms and each
//! vendor backend (CUDA here, Slang elsewhere) provides the language emitter.
//!
//! The CUDA backend was carved out of `baracuda-kernelgen` (carve step 2 of the
//! Unpopped extraction): the [`Cuda`] emitter (`cuda`), the NVRTC JIT compiler
//! (`nvrtc`, `--features nvrtc`) live here; `unpopped` is the neutral generator.
//!
//! The Fuel kernel-seam `BaracudaSynthesizer` used to be this crate's `seam`
//! feature. It moved to the unpublished `baracuda-cuda-seam` crate because
//! crates.io rejects the git dependency on fuel's seam crates that it needs.

pub mod cuda;
pub use cuda::{Cuda, emit_cast_helper, emit_coord_unravel_helper, emit_dtype_promote_helper};

/// Baracuda's own closed-set copy of 4 CUDA-specific scalar spellers
/// (`scalar_ctype`, `cast_scalar`, `promote_load_f32`, `demote_store_f32`)
/// that used to be imported directly from `unpopped::cfamily`. See the
/// module doc comment for why.
pub mod cfamily_shadow;

/// The CUDA spellings of `Fma`, `Cast` and `Bitcast` (unpopped 0.24.0).
mod typed;

/// The NVRTC on-demand JIT compiler (`NvrtcCompiler`) — the production
/// source → PTX [`Compiler`](unpopped::Compiler). Behind `--features
/// nvrtc` because it needs the nvrtc runtime.
#[cfg(feature = "nvrtc")]
pub mod nvrtc;
#[cfg(feature = "nvrtc")]
pub use nvrtc::NvrtcCompiler;
