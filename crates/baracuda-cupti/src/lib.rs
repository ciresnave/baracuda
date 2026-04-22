//! Safe Rust wrappers for CUPTI (CUDA Profiling Tools Interface).
//!
//! Four major surfaces are wrapped:
//!
//! - **Activity API** ([`activity`]) — batch-collects records of kernel
//!   launches, memcpys, driver/runtime API calls, etc., into a buffer
//!   you provide. Low overhead; best for post-hoc analysis.
//! - **Callback API** ([`callback`]) — point-in-time hooks on every
//!   driver/runtime call. Higher overhead; best for wrapping a specific
//!   call site.
//! - **Event / Metric APIs** ([`event`], [`metric`]) — legacy GPU
//!   hardware counters. Superseded by the Profiler Host API on modern
//!   GPUs; still useful for older chips.
//! - **Profiler Host API** ([`profiler`]) — modern metric / PM-sampling
//!   session control. Each cuPTI-host call takes an opaque `params`
//!   struct; we expose raw-pointer passthroughs because the struct
//!   layouts are version-sensitive and typically filled by NVIDIA's
//!   [NVPerf SDK](https://developer.nvidia.com/nsight-perf-sdk).

#![warn(missing_debug_implementations)]

use baracuda_cupti_sys::{cupti, CUptiResult};

/// Error type for CUPTI operations.
pub type Error = baracuda_core::Error<CUptiResult>;
/// Result alias.
pub type Result<T, E = Error> = core::result::Result<T, E>;

#[inline]
fn check(status: CUptiResult) -> Result<()> {
    Error::check(status)
}

/// CUPTI library version packed integer (e.g. `24` for CUPTI 24.0).
pub fn version() -> Result<u32> {
    let c = cupti()?;
    let cu = c.cupti_get_version()?;
    let mut v: u32 = 0;
    check(unsafe { cu(&mut v) })?;
    Ok(v)
}

/// GPU timestamp in nanoseconds, matching the one on activity records.
pub fn timestamp() -> Result<u64> {
    let c = cupti()?;
    let cu = c.cupti_get_timestamp()?;
    let mut t: u64 = 0;
    check(unsafe { cu(&mut t) })?;
    Ok(t)
}

/// Human-readable string for a CUPTI status code.
pub fn result_string(status: CUptiResult) -> Result<String> {
    let c = cupti()?;
    let cu = c.cupti_get_result_string()?;
    let mut ptr: *const core::ffi::c_char = core::ptr::null();
    check(unsafe { cu(status, &mut ptr) })?;
    if ptr.is_null() {
        return Ok(String::new());
    }
    let cstr = unsafe { core::ffi::CStr::from_ptr(ptr) };
    Ok(cstr.to_string_lossy().into_owned())
}

/// Activity API — enable kinds, register buffer callbacks, flush.
pub mod activity {
    use super::*;
    pub use baracuda_cupti_sys::{
        CUpti_ActivityKind, CUpti_BuffersCallbackCompleteFunc, CUpti_BuffersCallbackRequestFunc,
    };

    /// `cuptiActivityEnable`.
    pub fn enable(kind: u32) -> Result<()> {
        let c = cupti()?;
        let cu = c.cupti_activity_enable()?;
        check(unsafe { cu(kind) })
    }

    /// `cuptiActivityDisable`.
    pub fn disable(kind: u32) -> Result<()> {
        let c = cupti()?;
        let cu = c.cupti_activity_disable()?;
        check(unsafe { cu(kind) })
    }

    /// Flush pending records. `flags = 0` = default; `1` = force-flush.
    pub fn flush_all(flags: u32) -> Result<()> {
        let c = cupti()?;
        let cu = c.cupti_activity_flush_all()?;
        check(unsafe { cu(flags) })
    }

    /// Register buffer request / complete callbacks.
    ///
    /// # Safety
    ///
    /// The two function pointers must be valid for the lifetime of the
    /// activity session. CUPTI calls them from driver-owned threads.
    pub unsafe fn register_callbacks(
        request_fn: CUpti_BuffersCallbackRequestFunc,
        complete_fn: CUpti_BuffersCallbackCompleteFunc,
    ) -> Result<()> {
        let c = cupti()?;
        let cu = c.cupti_activity_register_callbacks()?;
        check(cu(request_fn, complete_fn))
    }

    /// Walk over activity records in a completed buffer. Returns the
    /// next `*mut CUpti_Activity` pointer, or `None` when exhausted.
    ///
    /// # Safety
    ///
    /// `buffer` + `valid_size` must match what CUPTI handed back to the
    /// complete-callback.
    pub unsafe fn get_next_record(
        buffer: *mut u8,
        valid_size: usize,
        record_inout: &mut *mut core::ffi::c_void,
    ) -> Result<bool> {
        let c = cupti()?;
        let cu = c.cupti_activity_get_next_record()?;
        match cu(buffer, valid_size, record_inout) {
            CUptiResult::SUCCESS => Ok(true),
            // Max reached or queue-empty.
            status if status.0 == 11 || status.0 == 10 => Ok(false),
            err => {
                let _ = check(err);
                Ok(false)
            }
        }
    }
}

/// Callback API — point-in-time hooks on driver / runtime API calls.
pub mod callback {
    use super::*;
    pub use baracuda_cupti_sys::{
        CUpti_CallbackDomain, CUpti_CallbackFunc, CUpti_SubscriberHandle,
    };

    /// A registered callback subscription. Drop unsubscribes.
    #[derive(Debug)]
    pub struct Subscriber {
        handle: CUpti_SubscriberHandle,
    }

    impl Subscriber {
        /// Subscribe a callback `cb` with user-data `user_data`.
        ///
        /// # Safety
        ///
        /// `cb` and `user_data` must remain valid until the returned
        /// subscriber drops.
        pub unsafe fn subscribe(
            cb: CUpti_CallbackFunc,
            user_data: *mut core::ffi::c_void,
        ) -> Result<Self> {
            let c = cupti()?;
            let cu = c.cupti_subscribe()?;
            let mut h: CUpti_SubscriberHandle = core::ptr::null_mut();
            check(cu(&mut h, cb, user_data))?;
            Ok(Self { handle: h })
        }

        /// Enable callbacks for a specific `domain` + `cbid`. Pass
        /// `enable = true` to turn on, `false` to turn off.
        pub fn enable_callback(&self, enable: bool, domain: u32, cbid: u32) -> Result<()> {
            let c = cupti()?;
            let cu = c.cupti_enable_callback()?;
            check(unsafe { cu(enable as u32, self.handle, domain, cbid) })
        }

        /// Enable / disable all callbacks in `domain` (shortcut for
        /// "I want every runtime API entry/exit").
        pub fn enable_domain(&self, enable: bool, domain: u32) -> Result<()> {
            let c = cupti()?;
            let cu = c.cupti_enable_domain()?;
            check(unsafe { cu(enable as u32, self.handle, domain) })
        }

        #[inline]
        pub fn as_raw(&self) -> CUpti_SubscriberHandle {
            self.handle
        }
    }

    impl Drop for Subscriber {
        fn drop(&mut self) {
            if let Ok(c) = cupti() {
                if let Ok(cu) = c.cupti_unsubscribe() {
                    let _ = unsafe { cu(self.handle) };
                }
            }
        }
    }
}

/// Hardware event API — legacy counters (deprecated on modern GPUs, use
/// [`profiler`] instead; still functional on pre-Turing).
pub mod event {
    use super::*;
    use baracuda_cupti_sys::{CUpti_EventGroup, CUpti_EventID};

    /// Number of event domains on `device`.
    pub fn num_domains(device: i32) -> Result<u32> {
        let c = cupti()?;
        let cu = c.cupti_device_get_num_event_domains()?;
        let mut n = 0u32;
        check(unsafe { cu(device, &mut n) })?;
        Ok(n)
    }

    /// Resolve an event name (e.g. `"inst_executed"`) to its ID.
    pub fn id_from_name(device: i32, name: &str) -> Result<CUpti_EventID> {
        let c = cupti()?;
        let cu = c.cupti_event_get_id_from_name()?;
        let cname = std::ffi::CString::new(name).map_err(|_| Error::Status {
            status: CUptiResult::ERROR_INVALID_PARAMETER,
        })?;
        let mut id: CUpti_EventID = 0;
        check(unsafe { cu(device, cname.as_ptr(), &mut id) })?;
        Ok(id)
    }

    /// Owned event-group handle. Drop destroys; add events then
    /// [`Group::enable`] to start collection.
    #[derive(Debug)]
    pub struct Group {
        raw: CUpti_EventGroup,
    }

    impl Group {
        /// # Safety
        /// `ctx` must be a valid `CUcontext` (use [`baracuda_driver::Context::as_raw`]).
        pub unsafe fn new(ctx: *mut core::ffi::c_void) -> Result<Self> {
            let c = cupti()?;
            let cu = c.cupti_event_group_create()?;
            let mut raw: CUpti_EventGroup = core::ptr::null_mut();
            check(cu(ctx, &mut raw, 0))?;
            Ok(Self { raw })
        }

        pub fn add(&self, event: CUpti_EventID) -> Result<()> {
            let c = cupti()?;
            let cu = c.cupti_event_group_add_event()?;
            check(unsafe { cu(self.raw, event) })
        }

        pub fn remove(&self, event: CUpti_EventID) -> Result<()> {
            let c = cupti()?;
            let cu = c.cupti_event_group_remove_event()?;
            check(unsafe { cu(self.raw, event) })
        }

        pub fn enable(&self) -> Result<()> {
            let c = cupti()?;
            let cu = c.cupti_event_group_enable()?;
            check(unsafe { cu(self.raw) })
        }

        pub fn disable(&self) -> Result<()> {
            let c = cupti()?;
            let cu = c.cupti_event_group_disable()?;
            check(unsafe { cu(self.raw) })
        }

        /// Read the latest counter value for `event`. Returns the single
        /// first bucket; for per-SM multi-value reads use the raw sys
        /// function directly.
        pub fn read(&self, event: CUpti_EventID) -> Result<u64> {
            let c = cupti()?;
            let cu = c.cupti_event_group_read_event()?;
            let mut buf = [0u64; 1];
            let mut size = core::mem::size_of::<[u64; 1]>();
            check(unsafe {
                cu(self.raw, 0, event, &mut size, buf.as_mut_ptr())
            })?;
            Ok(buf[0])
        }

        pub fn as_raw(&self) -> CUpti_EventGroup {
            self.raw
        }
    }

    impl Drop for Group {
        fn drop(&mut self) {
            if let Ok(c) = cupti() {
                if let Ok(cu) = c.cupti_event_group_destroy() {
                    let _ = unsafe { cu(self.raw) };
                }
            }
        }
    }
}

/// Hardware metric API — higher-level metrics that CUPTI computes from
/// one or more events. See [`event`] for the lower-level surface.
pub mod metric {
    use super::*;
    use baracuda_cupti_sys::CUpti_MetricID;

    pub fn id_from_name(device: i32, name: &str) -> Result<CUpti_MetricID> {
        let c = cupti()?;
        let cu = c.cupti_metric_get_id_from_name()?;
        let cname = std::ffi::CString::new(name).map_err(|_| Error::Status {
            status: CUptiResult::ERROR_INVALID_PARAMETER,
        })?;
        let mut id: CUpti_MetricID = 0;
        check(unsafe { cu(device, cname.as_ptr(), &mut id) })?;
        Ok(id)
    }

    /// Generic attribute accessor. `attrib` values come from CUPTI's
    /// `CUpti_MetricAttribute` enum.
    ///
    /// # Safety
    /// `value` must be writable for at least `size_bytes` bytes of the
    /// correct type for `attrib`.
    pub unsafe fn get_attribute(
        metric: CUpti_MetricID,
        attrib: u32,
        size: &mut usize,
        value: *mut core::ffi::c_void,
    ) -> Result<()> {
        let c = cupti()?;
        let cu = c.cupti_metric_get_attribute()?;
        check(cu(metric, attrib, size, value))
    }

    /// Compute a metric's value from collected event values and a time
    /// duration (ns).
    ///
    /// # Safety
    /// `event_ids` and `event_values` must have equal length; `metric_value`
    /// must be a writable buffer sized for the metric's value type.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn get_value(
        device: i32,
        metric: CUpti_MetricID,
        event_ids: &mut [baracuda_cupti_sys::CUpti_EventID],
        event_values: &mut [u64],
        time_duration_ns: u64,
        metric_value: *mut core::ffi::c_void,
    ) -> Result<()> {
        assert_eq!(event_ids.len(), event_values.len());
        let c = cupti()?;
        let cu = c.cupti_metric_get_value()?;
        check(cu(
            device,
            metric,
            core::mem::size_of_val(event_ids),
            event_ids.as_mut_ptr(),
            core::mem::size_of_val(event_values),
            event_values.as_mut_ptr(),
            time_duration_ns,
            metric_value,
        ))
    }
}

/// Modern Profiler Host API. Each call takes a `*mut c_void` params
/// struct; use the layouts from NVIDIA's NVPerf SDK to fill them. This
/// crate simply plumbs the calls through.
pub mod profiler {
    use super::*;

    /// # Safety
    /// See NVPerf SDK for the expected `params` struct.
    pub unsafe fn initialize(params: *mut core::ffi::c_void) -> Result<()> {
        let c = cupti()?;
        check((c.cupti_profiler_initialize()?)(params))
    }
    /// # Safety
    /// See NVPerf SDK.
    pub unsafe fn deinitialize(params: *mut core::ffi::c_void) -> Result<()> {
        let c = cupti()?;
        check((c.cupti_profiler_deinitialize()?)(params))
    }
    /// # Safety
    /// See NVPerf SDK.
    pub unsafe fn begin_session(params: *mut core::ffi::c_void) -> Result<()> {
        let c = cupti()?;
        check((c.cupti_profiler_begin_session()?)(params))
    }
    /// # Safety
    /// See NVPerf SDK.
    pub unsafe fn end_session(params: *mut core::ffi::c_void) -> Result<()> {
        let c = cupti()?;
        check((c.cupti_profiler_end_session()?)(params))
    }
    /// # Safety
    /// See NVPerf SDK.
    pub unsafe fn set_config(params: *mut core::ffi::c_void) -> Result<()> {
        let c = cupti()?;
        check((c.cupti_profiler_set_config()?)(params))
    }
    /// # Safety
    /// See NVPerf SDK.
    pub unsafe fn unset_config(params: *mut core::ffi::c_void) -> Result<()> {
        let c = cupti()?;
        check((c.cupti_profiler_unset_config()?)(params))
    }
    /// # Safety
    /// See NVPerf SDK.
    pub unsafe fn begin_pass(params: *mut core::ffi::c_void) -> Result<()> {
        let c = cupti()?;
        check((c.cupti_profiler_begin_pass()?)(params))
    }
    /// # Safety
    /// See NVPerf SDK.
    pub unsafe fn end_pass(params: *mut core::ffi::c_void) -> Result<()> {
        let c = cupti()?;
        check((c.cupti_profiler_end_pass()?)(params))
    }
    /// # Safety
    /// See NVPerf SDK.
    pub unsafe fn enable_profiling(params: *mut core::ffi::c_void) -> Result<()> {
        let c = cupti()?;
        check((c.cupti_profiler_enable_profiling()?)(params))
    }
    /// # Safety
    /// See NVPerf SDK.
    pub unsafe fn disable_profiling(params: *mut core::ffi::c_void) -> Result<()> {
        let c = cupti()?;
        check((c.cupti_profiler_disable_profiling()?)(params))
    }
    /// # Safety
    /// See NVPerf SDK.
    pub unsafe fn push_range(params: *mut core::ffi::c_void) -> Result<()> {
        let c = cupti()?;
        check((c.cupti_profiler_push_range()?)(params))
    }
    /// # Safety
    /// See NVPerf SDK.
    pub unsafe fn pop_range(params: *mut core::ffi::c_void) -> Result<()> {
        let c = cupti()?;
        check((c.cupti_profiler_pop_range()?)(params))
    }
    /// # Safety
    /// See NVPerf SDK.
    pub unsafe fn flush_counter_data(params: *mut core::ffi::c_void) -> Result<()> {
        let c = cupti()?;
        check((c.cupti_profiler_flush_counter_data()?)(params))
    }
    /// # Safety
    /// See NVPerf SDK.
    pub unsafe fn get_counter_availability(params: *mut core::ffi::c_void) -> Result<()> {
        let c = cupti()?;
        check((c.cupti_profiler_get_counter_availability()?)(params))
    }
}
