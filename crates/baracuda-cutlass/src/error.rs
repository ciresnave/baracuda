//! Error types for `baracuda-cutlass`.

use thiserror::Error;

/// Crate-local result alias.
pub type Result<T> = core::result::Result<T, Error>;

/// Errors raised by the safe CUTLASS wrapper.
///
/// Re-exported by [`baracuda-kernels`](../../baracuda-kernels) as
/// `baracuda_kernels::Error` since the kernel facade returns the same
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
    #[error("baracuda-cutlass: buffer too small for declared shape (need {needed} elements, got {got})")]
    BufferTooSmall {
        /// Minimum elements required for `(rows, cols, ld)`.
        needed: usize,
        /// Elements available in the supplied slice.
        got: usize,
    },

    /// CUTLASS reported an internal error during launch (typically a
    /// kernel-launch failure surfaced through `cudaGetLastError`).
    #[error("baracuda-cutlass: CUTLASS internal error (status code {0})")]
    CutlassInternal(i32),

    /// Underlying baracuda-driver error (context, stream, etc.).
    #[error("baracuda-cutlass: driver error: {0}")]
    Driver(#[from] baracuda_driver::Error),
}

/// Convert a status code returned by a `*_run` / `*_can_implement` extern
/// "C" function into a typed [`Error`].
///
/// Handles status 1, 2, 3, and 5 (and any other non-zero) only. Status 4
/// (workspace too small) is intentionally omitted: by the time the safe
/// layer launches the kernel it has already pre-checked workspace size
/// against the plan's reported requirement and returned a typed
/// [`Error::WorkspaceTooSmall`] *with the actual byte counts*. If a
/// kernel ever returns status 4 here it indicates a CUTLASS internal
/// inconsistency between `get_workspace_size` and the runtime requirement
/// — surfaced as [`Error::CutlassInternal`] to make the bug visible.
pub(crate) fn status_to_result(code: i32) -> Result<()> {
    match code {
        0 => Ok(()),
        1 => Err(Error::MisalignedOperand),
        2 => Err(Error::InvalidProblem("CUTLASS reported invalid problem")),
        3 => Err(Error::Unsupported("CUTLASS reported unsupported configuration")),
        n => Err(Error::CutlassInternal(n)),
    }
}
