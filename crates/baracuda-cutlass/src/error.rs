//! Error types for `baracuda-cutlass`.

// The type itself lives in `baracuda-kernels-types` (SA-35); re-exported here so
// `baracuda_cutlass::Error` / `Result` keep resolving for every consumer.
pub use baracuda_kernels_types::{Error, Result};

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
/// — surfaced as [`Error::KernelInternal`] to make the bug visible.
pub(crate) fn status_to_result(code: i32) -> Result<()> {
    match code {
        0 => Ok(()),
        1 => Err(Error::MisalignedOperand),
        2 => Err(Error::InvalidProblem("CUTLASS reported invalid problem")),
        3 => Err(Error::Unsupported(
            "CUTLASS reported unsupported configuration",
        )),
        n => Err(Error::KernelInternal(n)),
    }
}
