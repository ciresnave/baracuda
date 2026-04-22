//! Thin wrappers over `cuProfilerStart` / `cuProfilerStop`. These tell
//! external profilers (Nsight, CUPTI) which sections of a program are
//! interesting; they do not produce output themselves.
//!
//! Wrap a region with `start()` / `stop()` around the work you want
//! profiled, or use [`section`] which drops an RAII guard:
//!
//! ```no_run
//! # fn demo() -> baracuda_driver::Result<()> {
//! let _guard = baracuda_driver::profiler::section()?;
//! // ... profiled work ...
//! # Ok(())
//! # }
//! ```

use baracuda_cuda_sys::driver;

use crate::error::{check, Result};

/// Request that the external profiler begin sampling now.
pub fn start() -> Result<()> {
    let d = driver()?;
    let cu = d.cu_profiler_start()?;
    check(unsafe { cu() })
}

/// Request that the external profiler stop sampling now.
pub fn stop() -> Result<()> {
    let d = driver()?;
    let cu = d.cu_profiler_stop()?;
    check(unsafe { cu() })
}

/// Convenience guard: starts profiling on construction, stops on drop.
pub fn section() -> Result<Section> {
    start()?;
    Ok(Section { _nonzst: () })
}

#[derive(Debug)]
pub struct Section {
    _nonzst: (),
}

impl Drop for Section {
    fn drop(&mut self) {
        let _ = stop();
    }
}
