//! Runtime-side profiler start / stop — bracket the region you want
//! Nsight Systems / nvprof to capture.

use baracuda_cuda_sys::runtime::runtime;

use crate::error::{check, Result};

/// `cudaProfilerStart`.
pub fn start() -> Result<()> {
    let r = runtime()?;
    let cu = r.cuda_profiler_start()?;
    check(unsafe { cu() })
}

/// `cudaProfilerStop`.
pub fn stop() -> Result<()> {
    let r = runtime()?;
    let cu = r.cuda_profiler_stop()?;
    check(unsafe { cu() })
}

/// Run `f` with profiling enabled; always stops on return (even on
/// panic unwind via the guard's drop).
pub fn with_profiling<F, R>(f: F) -> Result<R>
where
    F: FnOnce() -> R,
{
    start()?;
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            let _ = stop();
        }
    }
    let _g = Guard;
    Ok(f())
}
