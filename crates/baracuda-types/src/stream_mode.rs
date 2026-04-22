//! The default-stream semantics switch.
//!
//! CUDA's default stream (the null stream) has two legal behaviors:
//!
//! * **Legacy** — the default stream synchronizes implicitly with all other
//!   per-thread streams in the same context. This is the historical default
//!   and is what you get when you don't pass `--default-stream per-thread`
//!   to `nvcc`.
//! * **PerThread** — each host thread has its own default stream and it
//!   does not synchronize with the legacy default stream. This is what
//!   every modern CUDA app should use and what the CUDA SDK has defaulted
//!   to since CUDA 7.
//!
//! The choice is process-global at the C API level: the symbol names that
//! `cuGetProcAddress_v2` resolves depend on which mode we ask for. The
//! enum itself lives here in `-types` (no I/O); the `OnceLock`-backed
//! setter lives in `baracuda-core::stream_mode`.

/// The default-stream semantics the baracuda loader should request from
/// `cuGetProcAddress_v2`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Default)]
pub enum StreamMode {
    /// Historical default: the null stream synchronizes with everything.
    Legacy,
    /// Modern default: each host thread owns its own default stream.
    #[default]
    PerThread,
}
