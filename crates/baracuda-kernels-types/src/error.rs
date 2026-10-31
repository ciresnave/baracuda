//! The shared error type for the baracuda kernel facade.
//!
//! Defined here so every per-library wrapper and `baracuda-kernels` share one `Error`;
//! `baracuda_cutlass::Error` and `baracuda_kernels::Error` re-export it.

use thiserror::Error;

/// Crate-local result alias.
pub type Result<T> = core::result::Result<T, Error>;

/// Errors raised by the safe kernel wrappers.
///
/// Re-exported as `baracuda_cutlass::Error` and `baracuda_kernels::Error`; the kernel facade returns the same
/// error surface for every op family (the cuSOLVER / cuDNN / cuFFT
/// facades all map their library-native status codes into one of these
/// variants).
///
/// `#[non_exhaustive]` — error variants have grown every couple of
/// phases as new failure modes surface (Phase 7 added cuDNN-status
/// fallback paths; Phase 22 added the cuSOLVER facade plumbing). Match
/// arms must include a `_ =>` catch-all so adding a new variant
/// doesn't break downstream `match e { ... }` blocks.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// The requested SKU isn't available in this build.
    ///
    /// Either no arch feature is enabled (no kernels were compiled), or the
    /// problem requires a SKU that's outside the curated v0 set
    /// (non-RCR layout, unsupported epilogue, etc.).
    #[error("baracuda-cutlass: requested kernel is unavailable: {0}")]
    Unsupported(&'static str),

    /// A problem dimension or stride is invalid (e.g., M, N, or K is non-positive).
    #[error("baracuda-cutlass: invalid problem: {0}")]
    InvalidProblem(&'static str),

    /// A pointer or stride is misaligned for the selected kernel's tensor-op
    /// instructions.
    #[error("baracuda-cutlass: misaligned operand")]
    MisalignedOperand,

    /// The provided workspace is too small for the selected plan or
    /// `Workspace::None` was passed when scratch was required.
    #[error("baracuda-cutlass: workspace too small (need {needed} bytes, got {got})")]
    WorkspaceTooSmall {
        /// Required workspace size in bytes.
        needed: usize,
        /// Provided workspace size in bytes.
        got: usize,
    },

    /// A device buffer is too small for the declared matrix shape and stride.
    #[error(
        "baracuda-cutlass: buffer too small for declared shape (need {needed} elements, got {got})"
    )]
    BufferTooSmall {
        /// Minimum elements required for `(rows, cols, ld)`.
        needed: usize,
        /// Elements available in the supplied slice.
        got: usize,
    },

    /// CUTLASS reported an internal error during launch (typically a
    /// kernel-launch failure surfaced through `cudaGetLastError`).
    #[error("baracuda-cutlass: kernel internal error (status code {0})")]
    KernelInternal(i32),

    /// Underlying baracuda-driver error (context, stream, etc.).
    #[error("baracuda-cutlass: driver error: {0}")]
    Driver(#[from] baracuda_driver::Error),
}
