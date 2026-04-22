//! Process-wide default-stream-semantics selector.
//!
//! The CUDA Driver API exposes two ABI-distinct variants of many functions
//! — one for "legacy default stream" semantics, one for "per-thread default
//! stream" — and the symbol you resolve through `cuGetProcAddress` depends
//! on which you ask for. baracuda treats the choice as a one-time process
//! decision: call [`init`] before the first driver call to override the
//! default, otherwise [`get`] returns [`StreamMode::PerThread`].
//!
//! See the NVIDIA programming guide's "Stream Synchronization Behavior"
//! section for the full semantic difference; the short version is that
//! PerThread is what every modern CUDA app wants.

use std::sync::OnceLock;

pub use baracuda_types::StreamMode;

static STREAM_MODE: OnceLock<StreamMode> = OnceLock::new();

/// Install the process-wide stream mode. Returns `Err` containing the mode
/// that was already installed if `init` (or [`get`]) was called earlier.
///
/// The expected call site is early in `main`, before any CUDA activity.
pub fn init(mode: StreamMode) -> Result<(), StreamMode> {
    STREAM_MODE.set(mode)
}

/// The currently-installed stream mode. If [`init`] has not been called, this
/// latches the default ([`StreamMode::PerThread`]) on first call.
pub fn get() -> StreamMode {
    *STREAM_MODE.get_or_init(StreamMode::default)
}

#[cfg(test)]
mod tests {
    use super::*;

    // NOTE: the stream mode is a `OnceLock` — it's a one-time write for the
    // process. We can't safely test both `init` paths in a single test
    // binary, so we verify `get()` latches to the default without a prior
    // `init`. Each `-sys` crate will have its own integration test exercise
    // the `init` path.
    #[test]
    fn default_is_per_thread() {
        assert_eq!(get(), StreamMode::PerThread);
    }

    #[test]
    fn subsequent_init_fails() {
        // `get()` in the previous test already initialized; a fresh `init`
        // should bounce.
        let prior = get();
        let res = init(StreamMode::Legacy);
        assert!(res.is_err(), "expected re-init to fail; got Ok");
        assert_eq!(get(), prior);
    }
}
