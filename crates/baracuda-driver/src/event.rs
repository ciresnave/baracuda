//! CUDA events — lightweight synchronization objects you can record on
//! a stream and later wait on, or use to measure elapsed device time.

use std::sync::Arc;

use baracuda_cuda_sys::types::CUevent_flags;
use baracuda_cuda_sys::{CUevent, driver};

use crate::context::Context;
use crate::error::{Result, check};
use crate::stream::Stream;

/// A CUDA event.
#[derive(Clone)]
pub struct Event {
    inner: Arc<EventInner>,
}

struct EventInner {
    handle: CUevent,
    context: Context,
    /// When `false`, this is a non-owning (borrowed) wrapper produced by
    /// [`Event::borrow_raw`]: `Drop` will not call `cuEventDestroy`.
    owned: bool,
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
                owned: true,
            }),
        })
    }

    /// Adopt a raw `CUevent`, **transferring ownership to baracuda**: the
    /// returned [`Event`] (and every clone) shares it, and `cuEventDestroy`
    /// runs when the last clone drops. `context` is the [`Context`] the event
    /// belongs to; baracuda keeps a clone of that wrapper. Use this when another
    /// library hands baracuda an event it created and no longer wants to manage.
    ///
    /// To synchronize against an event another library still owns, use
    /// [`Event::borrow_raw`] instead — baracuda won't destroy it.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid, live `CUevent` created on `context` that
    ///   baracuda may take sole ownership of: it must not be destroyed
    ///   elsewhere, nor already be wrapped by another owning baracuda handle
    ///   (either risks a double `cuEventDestroy`).
    /// - The event's underlying CUDA context must outlive the returned `Event`
    ///   and all its clones. Passing an *owning* [`Context`] guarantees that; if
    ///   `context` is itself a [borrowed][`Context::borrow_raw`] (non-owning)
    ///   wrapper, the held clone does **not** extend the real context's
    ///   lifetime — the caller must keep it alive by other means.
    pub unsafe fn from_raw(handle: CUevent, context: &Context) -> Self {
        Self {
            inner: Arc::new(EventInner {
                handle,
                context: context.clone(),
                owned: true,
            }),
        }
    }

    /// Wrap a raw `CUevent` that **baracuda does not own**: `Drop` will not
    /// call `cuEventDestroy`. Use this to let baracuda streams wait on, or
    /// query, an event created and owned by another library (cudarc, cust, a
    /// framework) that outlives this wrapper — the standard cross-library
    /// synchronization path. `context` is the [`Context`] the event belongs to
    /// (wrap a foreign one with [`Context::borrow_raw`] if needed).
    ///
    /// # Safety
    ///
    /// The caller guarantees `handle` is a valid `CUevent` on `context` that
    /// stays live for the entire lifetime of the returned `Event` and all of
    /// its clones. baracuda will not destroy it; the owning library keeps that
    /// responsibility.
    pub unsafe fn borrow_raw(handle: CUevent, context: &Context) -> Self {
        Self {
            inner: Arc::new(EventInner {
                handle,
                context: context.clone(),
                owned: false,
            }),
        }
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
        // A borrowed (non-owning) wrapper must not destroy the foreign handle.
        if !self.owned {
            return;
        }
        if let Ok(d) = driver() {
            if let Ok(cu) = d.cu_event_destroy() {
                // SAFETY: owned handle (cuEventCreate or transferred via from_raw).
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}
