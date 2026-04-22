//! Runtime-API events.

use std::sync::Arc;

use baracuda_cuda_sys::runtime::{cudaEvent_t, runtime, types::cudaEventFlags};

use crate::error::{check, Result};
use crate::stream::Stream;

/// A CUDA event (Runtime API).
#[derive(Clone)]
pub struct Event {
    inner: Arc<EventInner>,
}

struct EventInner {
    handle: cudaEvent_t,
}

unsafe impl Send for EventInner {}
unsafe impl Sync for EventInner {}

impl core::fmt::Debug for EventInner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Event")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

impl core::fmt::Debug for Event {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.inner.fmt(f)
    }
}

impl Event {
    /// Create an event with default flags (timing enabled, spinning wait).
    pub fn new() -> Result<Self> {
        Self::with_flags(cudaEventFlags::DEFAULT)
    }

    /// Create an event optimized for synchronization (no timing data).
    pub fn no_timing() -> Result<Self> {
        Self::with_flags(cudaEventFlags::DISABLE_TIMING)
    }

    /// Create an event with raw flags (see [`cudaEventFlags`]).
    pub fn with_flags(flags: u32) -> Result<Self> {
        let r = runtime()?;
        let cu = r.cuda_event_create_with_flags()?;
        let mut event: cudaEvent_t = core::ptr::null_mut();
        check(unsafe { cu(&mut event, flags) })?;
        Ok(Self {
            inner: Arc::new(EventInner { handle: event }),
        })
    }

    /// Record this event on the given stream.
    pub fn record(&self, stream: &Stream) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_event_record()?;
        check(unsafe { cu(self.inner.handle, stream.as_raw()) })
    }

    /// Block the calling host thread until the event has completed.
    pub fn synchronize(&self) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_event_synchronize()?;
        check(unsafe { cu(self.inner.handle) })
    }

    /// `Ok(true)` if the event has completed.
    pub fn is_complete(&self) -> Result<bool> {
        use baracuda_cuda_sys::runtime::cudaError_t;
        let r = runtime()?;
        let cu = r.cuda_event_query()?;
        match unsafe { cu(self.inner.handle) } {
            cudaError_t::Success => Ok(true),
            cudaError_t::NotReady => Ok(false),
            other => Err(crate::error::Error::Status { status: other }),
        }
    }

    /// Milliseconds of device work between `start` and `end`. Both must have
    /// been created with timing enabled.
    pub fn elapsed_time_ms(start: &Event, end: &Event) -> Result<f32> {
        let r = runtime()?;
        let cu = r.cuda_event_elapsed_time()?;
        let mut ms: f32 = 0.0;
        check(unsafe { cu(&mut ms, start.inner.handle, end.inner.handle) })?;
        Ok(ms)
    }

    /// Raw `cudaEvent_t`. Use with care.
    #[inline]
    pub fn as_raw(&self) -> cudaEvent_t {
        self.inner.handle
    }
}

impl Drop for EventInner {
    fn drop(&mut self) {
        if let Ok(r) = runtime() {
            if let Ok(cu) = r.cuda_event_destroy() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}
