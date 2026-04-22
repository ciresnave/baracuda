//! CUDA events — lightweight synchronization objects you can record on
//! a stream and later wait on, or use to measure elapsed device time.

use std::sync::Arc;

use baracuda_cuda_sys::types::CUevent_flags;
use baracuda_cuda_sys::{driver, CUevent};

use crate::context::Context;
use crate::error::{check, Result};
use crate::stream::Stream;

/// A CUDA event.
#[derive(Clone)]
pub struct Event {
    inner: Arc<EventInner>,
}

struct EventInner {
    handle: CUevent,
    context: Context,
}

// SAFETY: CUevent is documented safe for multi-thread use.
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
    /// Create a new event with default flags (timing enabled).
    pub fn new(context: &Context) -> Result<Self> {
        Self::with_flags(context, CUevent_flags::DEFAULT)
    }

    /// Create an event optimized for synchronization (no timing).
    pub fn no_timing(context: &Context) -> Result<Self> {
        Self::with_flags(context, CUevent_flags::DISABLE_TIMING)
    }

    /// Create an event with raw flags (see [`CUevent_flags`]).
    pub fn with_flags(context: &Context, flags: u32) -> Result<Self> {
        context.set_current()?;
        let d = driver()?;
        let cu = d.cu_event_create()?;
        let mut event: CUevent = core::ptr::null_mut();
        check(unsafe { cu(&mut event, flags) })?;
        Ok(Self {
            inner: Arc::new(EventInner {
                handle: event,
                context: context.clone(),
            }),
        })
    }

    /// Record this event on the given stream. The event "happens" when all
    /// prior work on `stream` has completed.
    pub fn record(&self, stream: &Stream) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_event_record()?;
        check(unsafe { cu(self.inner.handle, stream.as_raw()) })
    }

    /// As [`record`](Self::record) but with a raw CUDA event-record flags
    /// bitmask. See `CU_EVENT_RECORD_*` in NVIDIA's headers.
    pub fn record_with_flags(&self, stream: &Stream, flags: u32) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_event_record_with_flags()?;
        check(unsafe { cu(self.inner.handle, stream.as_raw(), flags) })
    }

    /// Block the calling host thread until this event has completed.
    pub fn synchronize(&self) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_event_synchronize()?;
        check(unsafe { cu(self.inner.handle) })
    }

    /// `Ok(true)` if the event has completed.
    pub fn is_complete(&self) -> Result<bool> {
        use baracuda_cuda_sys::CUresult;
        let d = driver()?;
        let cu = d.cu_event_query()?;
        match unsafe { cu(self.inner.handle) } {
            CUresult::SUCCESS => Ok(true),
            CUresult::ERROR_NOT_READY => Ok(false),
            other => Err(crate::error::Error::Status { status: other }),
        }
    }

    /// Elapsed milliseconds of device work between `start` (recorded first)
    /// and `end` (recorded later). Both events must have been created with
    /// timing enabled.
    pub fn elapsed_time_ms(start: &Event, end: &Event) -> Result<f32> {
        let d = driver()?;
        let cu = d.cu_event_elapsed_time()?;
        let mut ms: f32 = 0.0;
        check(unsafe { cu(&mut ms, start.inner.handle, end.inner.handle) })?;
        Ok(ms)
    }

    /// The [`Context`] this event lives in.
    #[inline]
    pub fn context(&self) -> &Context {
        &self.inner.context
    }

    /// Raw `CUevent`. Use with care.
    #[inline]
    pub fn as_raw(&self) -> CUevent {
        self.inner.handle
    }
}

impl Drop for EventInner {
    fn drop(&mut self) {
        if let Ok(d) = driver() {
            if let Ok(cu) = d.cu_event_destroy() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}
