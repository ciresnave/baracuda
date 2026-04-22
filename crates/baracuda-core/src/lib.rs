//! Runtime machinery shared across baracuda crates.
//!
//! - [`error`] — loader errors, a generic per-library `Error<S>`, and the
//!   library-erased `BaracudaError`.
//! - [`loader`] — the dynamic loader wrapper used by every `-sys` crate.
//! - [`platform`] — OS detection and default library search paths.
//! - [`stream_mode`] — process-wide default-stream-semantics selector.
//!
//! Depending on this crate pulls in `libloading`. Crates that only need the
//! type vocabulary should depend on `baracuda-types` instead.

#![warn(missing_debug_implementations)]

pub mod error;
pub mod loader;
pub mod platform;
pub mod stream_mode;

pub use baracuda_types::{CudaStatus, CudaVersion, Feature, StreamMode};
pub use error::{BaracudaError, Error, LoaderError};
pub use loader::Library;
