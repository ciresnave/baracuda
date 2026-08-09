//! # baracuda-forge
//!
//! Build-time CUDA kernel compiler for the [baracuda] ecosystem. Drop it into
//! your `[build-dependencies]` and turn `.cu` files into a static library or
//! PTX with `nvcc`, with incremental rebuilds, parallel compilation, GPU
//! compute-capability auto-detection, and integrated CUTLASS support.
//!
//! `baracuda-forge` is the **build-time** companion to baracuda's runtime
//! wrappers (`baracuda-driver`, `baracuda-runtime`, ...). Use forge to turn
//! your `.cu` files into a library; use the runtime crates to launch the
//! kernels from Rust.
//!
//! [baracuda]: https://github.com/ciresnave/baracuda
//!
//! ## Features
//!
//! - **Compute Capability Detection** — auto-detect from `nvidia-smi` or
//!   `CUDA_COMPUTE_CAP`, with per-file overrides for mixed architectures.
//! - **Incremental Builds** — only recompile modified kernels using SHA-256
//!   content hashing.
//! - **CUDA Toolkit Auto-Detection** — find `nvcc` and include paths via the
//!   shared [`baracuda_build`] detector.
//! - **C++ Standard Auto-Select** — defaults to `c++20` on CUDA ≥ 12.0,
//!   `c++17` on older toolkits. Override via [`KernelBuilder::cpp_std`].
//! - **External Dependencies** — built-in CUTLASS support, or fetch any git repo.
//! - **Parallel Compilation** — configurable thread percentage for parallel builds.
//! - **Flexible Source Selection** — directory, glob, files, or exclude patterns.
//!
//! ## Controlling build parallelism
//!
//! nvcc/ptxas is memory-heavy, so forge compiles kernels on a pool sized, by
//! default, to **~50% of the logical cores**. Under **concurrent CUDA builds**
//! (several crates or worktrees building at once) that default can oversubscribe
//! host memory and ptxas fails with `Memory allocation failure`. Three knobs, in
//! priority order:
//!
//! - **`BARACUDA_FORGE_THREADS=K`** — set the pool to exactly `K`. The only knob
//!   that can raise the pool ABOVE the 50% default.
//! - **`RAYON_NUM_THREADS=K`** — honoured if the above is unset (it also perturbs
//!   every other rayon consumer in the build; prefer `BARACUDA_FORGE_THREADS`).
//! - **`cargo -j K`** (cargo sets `NUM_JOBS`) — honoured as an **upper cap**: it
//!   only ever *lowers* the pool below the 50% default (cargo defaults `NUM_JOBS`
//!   to the core count when `-j` is omitted, so it never raises it). Lets a
//!   consumer running concurrent builds cap forge with the standard cargo flag.
//!
//! Forge prints a `cargo:warning` naming the thread count and these knobs, and
//! re-surfaces a `ptxas fatal` / `Memory allocation failure` as the build-error
//! headline (it usually means host oversubscription, not a code defect). Note
//! that cargo **caches build-script output** and suppresses the warning on a
//! fresh-but-cached build, so the warning's **absence is not evidence a cap is
//! set** — it only prints on the first compile in a given target dir.
//!
//! ## Quick start
//!
//! ```no_run
//! use baracuda_forge::KernelBuilder;
//!
//! fn main() {
//!     let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR must be set");
//!
//!     KernelBuilder::new()
//!         .source_dir("src/kernels")
//!         .exclude(&["*_test.cu"])
//!         .arg("-O3")
//!         .thread_percentage(0.5)
//!         .build_lib(format!("{}/libkernels.a", out_dir))
//!         .expect("CUDA compilation failed");
//!
//!     println!("cargo:rustc-link-search={}", out_dir);
//!     println!("cargo:rustc-link-lib=kernels");
//! }
//! ```
//!
//! ## Acknowledgments
//!
//! `baracuda-forge` is a vendored fork of [`cudaforge`] by Guoqing Bao,
//! adapted to baracuda's workspace conventions. See the `NOTICE` file at the
//! crate root for full provenance.
//!
//! [`cudaforge`]: https://github.com/guoqingbao/cudaforge

#![deny(missing_docs)]

mod builder;
mod compute_cap;
mod dependency;
mod error;
mod hash;
mod parallel;
mod source;
mod toolkit;

pub use builder::{KernelBuilder, PtxOutput};
pub use compute_cap::{ComputeCapability, GpuArch, detect_compute_cap, get_gpu_arch_string};
pub use dependency::{DependencyManager, ExternalDependency, resolve_cutlass_from_cargo_checkouts};
pub use error::{Error, Result};
pub use hash::BuildCache;
pub use parallel::ParallelConfig;
pub use source::{SourceSelector, collect_headers};
pub use toolkit::CudaToolkit;
