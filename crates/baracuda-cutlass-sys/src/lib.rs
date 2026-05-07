//! # baracuda-cutlass-sys
//!
//! Header acquisition for NVIDIA CUTLASS as a baracuda workspace dependency.
//!
//! At build time, this crate sparse-checks-out CUTLASS headers from
//! <https://github.com/NVIDIA/cutlass> into a file-locked global cache and
//! emits Cargo metadata so downstream `build.rs` files (notably
//! [`baracuda-forge`]) and `cc` / `bindgen` invocations can locate the
//! headers without hand-rolling path discovery.
//!
//! CUTLASS is template-only C++; there is no stable C ABI to bind to.
//! Accordingly, this crate exposes no FFI types — its job is to make the
//! headers available, not to wrap them.
//!
//! See the crate `README.md` for version-selection knobs (the `cutlass-2-11`
//! feature, the `CUTLASS_DIR` and `BARACUDA_CUTLASS_COMMIT` env vars).
//!
//! [`baracuda-forge`]: https://docs.rs/baracuda-forge

#![no_std]
#![deny(missing_docs)]

/// Absolute path to the CUTLASS `include/` directory selected at build time.
///
/// Equivalent to `env!("BARACUDA_CUTLASS_INCLUDE_DIR")`. Useful for runtime
/// code that wants to know where CUTLASS headers live (e.g., a custom NVRTC
/// integration that resolves `#include`s itself).
pub const INCLUDE_DIR: &str = env!("BARACUDA_CUTLASS_INCLUDE_DIR");

/// Absolute path to the CUTLASS checkout root (parent of [`INCLUDE_DIR`]).
pub const ROOT_DIR: &str = env!("BARACUDA_CUTLASS_ROOT");

/// CUTLASS version tag fetched at build time (e.g., `"v4.2.0"`), or the
/// commit hash if `BARACUDA_CUTLASS_COMMIT` was used.
pub const VERSION: &str = env!("BARACUDA_CUTLASS_VERSION");
