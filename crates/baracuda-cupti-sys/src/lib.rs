//! Raw FFI + dynamic loader skeleton for CUPTI (CUDA Profiling Tools Interface).
//!
//! `baracuda-cupti` wraps this with a safe, typed API. Use this crate
//! directly only if you need a function that the safe layer hasn't
//! wrapped yet (in which case please file a bug).
//!
//! CUPTI's surface is huge (activity API, callback API, event/metric APIs,
//! PC sampling, the modern profiler host API). v0.1 ships the loader +
//! status/version types so the crate compiles; concrete profiler wrappers
//! will land as CI gains matching tooling.

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
#![warn(missing_debug_implementations)]

use core::ffi::c_void;
use std::path::PathBuf;
use std::sync::OnceLock;

use baracuda_core::{Library, LoaderError};
use baracuda_types::CudaStatus;

/// CUPTI status code.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct CUptiResult(pub i32);

impl CUptiResult {
    /// success
    pub const SUCCESS: Self = Self(0);
    /// invalid parameter
    pub const ERROR_INVALID_PARAMETER: Self = Self(1);
    /// invalid device
    pub const ERROR_INVALID_DEVICE: Self = Self(2);
    /// invalid CUDA context
    pub const ERROR_INVALID_CONTEXT: Self = Self(3);
    /// invalid event ID
    pub const ERROR_INVALID_EVENT_ID: Self = Self(4);
    /// invalid event name
    pub const ERROR_INVALID_EVENT_NAME: Self = Self(5);
    /// invalid operation
    pub const ERROR_INVALID_OPERATION: Self = Self(6);
    /// out of memory
    pub const ERROR_OUT_OF_MEMORY: Self = Self(7);
    /// hardware error
    pub const ERROR_HARDWARE: Self = Self(8);
    /// CUPTI not initialized
    pub const ERROR_NOT_INITIALIZED: Self = Self(13);
    /// operation not supported
    pub const ERROR_NOT_SUPPORTED: Self = Self(14);
    /// unknown CUPTI error
    pub const ERROR_UNKNOWN: Self = Self(999);

    /// returns true when the result code is `CUPTI_SUCCESS`
    pub const fn is_success(self) -> bool {
        self.0 == 0
    }
}

impl CudaStatus for CUptiResult {
    fn code(self) -> i32 {
        self.0
    }
    fn name(self) -> &'static str {
        match self.0 {
            0 => "CUPTI_SUCCESS",
            1 => "CUPTI_ERROR_INVALID_PARAMETER",
            6 => "CUPTI_ERROR_INVALID_OPERATION",
            7 => "CUPTI_ERROR_OUT_OF_MEMORY",
            13 => "CUPTI_ERROR_NOT_INITIALIZED",
            14 => "CUPTI_ERROR_NOT_SUPPORTED",
            _ => "CUPTI_ERROR_UNRECOGNIZED",
        }
    }
    fn description(self) -> &'static str {
        match self.0 {
            0 => "success",
            1 => "invalid parameter",
            6 => "invalid operation",
            7 => "out of memory",
            13 => "CUPTI not initialized",
            14 => "operation not supported",
            _ => "unrecognized CUPTI status code",
        }
    }
    fn is_success(self) -> bool {
        CUptiResult::is_success(self)
    }
    fn library(self) -> &'static str {
        "cupti"
    }
}

// ---- activity API (the main "just collect kernel records" surface) ----

/// `CUpti_ActivityKind` — selector for activity records.
#[allow(non_snake_case)]
pub mod CUpti_ActivityKind {
    /// invalid / sentinel
    pub const INVALID: u32 = 0;
    /// memcpy activity record
    pub const MEMCPY: u32 = 1;
    /// memset activity record
    pub const MEMSET: u32 = 2;
    /// kernel activity record
    pub const KERNEL: u32 = 3;
    /// driver API activity record
    pub const DRIVER: u32 = 4;
    /// runtime API activity record
    pub const RUNTIME: u32 = 5;
    /// event activity record
    pub const EVENT: u32 = 6;
    /// metric activity record
    pub const METRIC: u32 = 7;
    /// device activity record
    pub const DEVICE: u32 = 8;
    /// context activity record
    pub const CONTEXT: u32 = 9;
    /// concurrent-kernel activity record
    pub const CONCURRENT_KERNEL: u32 = 10;
    /// name activity record
    pub const NAME: u32 = 11;
    /// marker activity record
    pub const MARKER: u32 = 12;
    /// CUPTI overhead activity record
    pub const OVERHEAD: u32 = 14;
    /// CUDA-Dynamic-Parallelism kernel activity record
    pub const CDP_KERNEL: u32 = 20;
}

/// Callback supplied by the user to provide a new activity buffer.
pub type CUpti_BuffersCallbackRequestFunc = unsafe extern "C" fn(
    buffer_out: *mut *mut u8,
    size_out: *mut usize,
    max_num_records_out: *mut usize,
);

/// Callback supplied by the user, fired when CUPTI fills a buffer.
pub type CUpti_BuffersCallbackCompleteFunc = unsafe extern "C" fn(
    ctx: *mut c_void, // CUcontext
    stream_id: u32,
    buffer: *mut u8,
    size: usize,
    valid_size: usize,
);

// ---- callback API (point-in-time hooks on driver/runtime API calls) ----

/// `CUpti_CallbackDomain`.
#[allow(non_snake_case)]
pub mod CUpti_CallbackDomain {
    /// invalid / sentinel
    pub const INVALID: u32 = 0;
    /// callbacks for CUDA driver API calls
    pub const DRIVER_API: u32 = 1;
    /// callbacks for CUDA runtime API calls
    pub const RUNTIME_API: u32 = 2;
    /// callbacks for resource creation/destruction
    pub const RESOURCE: u32 = 3;
    /// callbacks for synchronization points
    pub const SYNCHRONIZE: u32 = 4;
    /// callbacks for NVTX events
    pub const NVTX: u32 = 5;
    /// callbacks for CUPTI state changes
    pub const STATE: u32 = 6;
}

/// Callback signature for `cuptiSubscribe`.
pub type CUpti_CallbackFunc =
    unsafe extern "C" fn(user_data: *mut c_void, domain: u32, cbid: u32, cb_info: *const c_void);

/// Opaque CUPTI subscriber handle.
pub type CUpti_SubscriberHandle = *mut c_void;

// ---- PFN types ----

/// function pointer for `cuptiGetVersion`
pub type PFN_cuptiGetVersion = unsafe extern "C" fn(version: *mut u32) -> CUptiResult;
/// function pointer for `cuptiGetLastError`
pub type PFN_cuptiGetLastError = unsafe extern "C" fn() -> CUptiResult;
/// function pointer for `cuptiGetResultString`
pub type PFN_cuptiGetResultString =
    unsafe extern "C" fn(result: CUptiResult, string: *mut *const core::ffi::c_char) -> CUptiResult;

/// function pointer for `cuptiGetTimestamp`
pub type PFN_cuptiGetTimestamp = unsafe extern "C" fn(ts_out: *mut u64) -> CUptiResult;

/// function pointer for `cuptiActivityEnable` — enable activity kind tracking
pub type PFN_cuptiActivityEnable = unsafe extern "C" fn(kind: u32) -> CUptiResult;
/// function pointer for `cuptiActivityDisable`
pub type PFN_cuptiActivityDisable = unsafe extern "C" fn(kind: u32) -> CUptiResult;
/// function pointer for `cuptiActivityFlushAll`
pub type PFN_cuptiActivityFlushAll = unsafe extern "C" fn(flags: u32) -> CUptiResult;
/// function pointer for `cuptiActivityRegisterCallbacks`
pub type PFN_cuptiActivityRegisterCallbacks = unsafe extern "C" fn(
    request_fn: CUpti_BuffersCallbackRequestFunc,
    complete_fn: CUpti_BuffersCallbackCompleteFunc,
) -> CUptiResult;

/// function pointer for `cuptiActivityGetNextRecord`
pub type PFN_cuptiActivityGetNextRecord = unsafe extern "C" fn(
    buffer: *mut u8,
    valid_buffer_size_bytes: usize,
    record_out: *mut *mut c_void,
) -> CUptiResult;

/// function pointer for `cuptiSubscribe`
pub type PFN_cuptiSubscribe = unsafe extern "C" fn(
    subscriber_out: *mut CUpti_SubscriberHandle,
    callback: CUpti_CallbackFunc,
    user_data: *mut c_void,
) -> CUptiResult;

/// function pointer for `cuptiUnsubscribe`
pub type PFN_cuptiUnsubscribe =
    unsafe extern "C" fn(subscriber: CUpti_SubscriberHandle) -> CUptiResult;

/// function pointer for `cuptiEnableCallback` — enable callback for domain
pub type PFN_cuptiEnableCallback = unsafe extern "C" fn(
    enable: u32,
    subscriber: CUpti_SubscriberHandle,
    domain: u32,
    cbid: u32,
) -> CUptiResult;

/// function pointer for `cuptiEnableDomain`
pub type PFN_cuptiEnableDomain = unsafe extern "C" fn(
    enable: u32,
    subscriber: CUpti_SubscriberHandle,
    domain: u32,
) -> CUptiResult;

// ---- Activity record structs (partial — KERNEL + MEMCPY) ------------------

/// `CUpti_ActivityKernel9` — matches CUPTI 12.x+ layout. Older drivers may
/// emit smaller records; check the leading `kind` field before reading past
/// the size indicated by the buffer.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct CUpti_ActivityKernel9 {
    /// activity-kind discriminant
    pub kind: u32,
    /// requested L1 cache configuration
    pub cache_config: u32,
    /// shared-memory bank configuration
    pub shared_memory_config: u8,
    /// registers allocated per thread
    pub registers_per_thread: u16,
    /// requested partitioned-cache mode
    pub partitioned_global_cache_requested: u8,
    /// actual partitioned-cache mode
    pub partitioned_global_cache_executed: u8,
    /// padding for ABI alignment
    pub reserved_padding: [u8; 3],
    /// kernel start timestamp (ns since epoch)
    pub start: u64,
    /// kernel end timestamp (ns since epoch)
    pub end: u64,
    /// kernel completion timestamp (ns since epoch)
    pub completed: u64,
    /// device that ran the kernel
    pub device_id: u32,
    /// CUDA context ID
    pub context_id: u32,
    /// CUDA stream ID
    pub stream_id: u32,
    /// grid dimension x
    pub grid_x: u32,
    /// grid dimension y
    pub grid_y: u32,
    /// grid dimension z
    pub grid_z: u32,
    /// block dimension x
    pub block_x: u32,
    /// block dimension y
    pub block_y: u32,
    /// block dimension z
    pub block_z: u32,
    /// static shared-memory bytes
    pub static_shared_memory: i64,
    /// dynamic shared-memory bytes
    pub dynamic_shared_memory: i64,
    /// local-memory bytes per thread
    pub local_memory_per_thread: u32,
    /// total local-memory bytes for the launch
    pub local_memory_total: u32,
    /// correlation ID linking driver/runtime/kernel records
    pub correlation_id: u32,
    /// correlation ID for the runtime API call
    pub runtime_correlation_id: u32,
    /// timestamp at which the launch was queued
    pub queued: u64,
    /// timestamp at which the launch was submitted to the device
    pub submitted: u64,
    /// kernel name (interned C string)
    pub name: *const core::ffi::c_char,
    /// reserved; do not read
    pub reserved0: *mut c_void,
    /// globally-unique grid ID
    pub grid_id: u64,
    /// CUDA-graph node ID (if launched from a graph)
    pub graph_node_id: u64,
    /// engine channel ID
    pub channel_id: u32,
    /// thread-block-cluster dimension x
    pub cluster_x: u32,
    /// thread-block-cluster dimension y
    pub cluster_y: u32,
    /// thread-block-cluster dimension z
    pub cluster_z: u32,
    /// cluster scheduling policy
    pub cluster_scheduling_policy: u32,
    /// total local-memory bytes (v2; 64-bit)
    pub local_memory_total_v2: u64,
}

/// cupti activitymemcpy5
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct CUpti_ActivityMemcpy5 {
    /// activity-kind discriminant
    pub kind: u32,
    /// memcpy kind (H2D, D2H, …)
    pub copy_kind: u8,
    /// source memory kind
    pub src_kind: u8,
    /// destination memory kind
    pub dst_kind: u8,
    /// memcpy flags
    pub flags: u8,
    /// bytes transferred
    pub bytes: u64,
    /// kernel start timestamp (ns since epoch)
    pub start: u64,
    /// kernel end timestamp (ns since epoch)
    pub end: u64,
    /// device that ran the kernel
    pub device_id: u32,
    /// CUDA context ID
    pub context_id: u32,
    /// CUDA stream ID
    pub stream_id: u32,
    /// correlation ID linking driver/runtime/kernel records
    pub correlation_id: u32,
    /// correlation ID for the runtime API call
    pub runtime_correlation_id: u32,
    /// CUDA-graph node ID (if launched from a graph)
    pub graph_node_id: u64,
    /// engine channel ID
    pub channel_id: u32,
    /// engine channel type
    pub channel_type: u32,
}

// ---- Event API (legacy, per-SM hardware counters) -----------------------

/// Opaque CUPTI event-group handle.
pub type CUpti_EventGroup = *mut c_void;
/// Opaque CUPTI event-group-sets handle.
pub type CUpti_EventGroupSets = *mut c_void;
/// CUPTI event identifier.
pub type CUpti_EventID = u32;
/// CUPTI metric identifier.
pub type CUpti_MetricID = u32;

/// function pointer for `cuptiDeviceGetNumEventDomains`
pub type PFN_cuptiDeviceGetNumEventDomains = unsafe extern "C" fn(
    device: i32,
    num_domains: *mut u32,
) -> CUptiResult;

/// function pointer for `cuptiDeviceEnumEventDomains`
pub type PFN_cuptiDeviceEnumEventDomains = unsafe extern "C" fn(
    device: i32,
    array_size_bytes: *mut usize,
    domain_array: *mut u32,
) -> CUptiResult;

/// function pointer for `cuptiEventGroupCreate`
pub type PFN_cuptiEventGroupCreate = unsafe extern "C" fn(
    ctx: *mut c_void,
    event_group: *mut CUpti_EventGroup,
    flags: u32,
) -> CUptiResult;

/// function pointer for `cuptiEventGroupDestroy`
pub type PFN_cuptiEventGroupDestroy = unsafe extern "C" fn(
    event_group: CUpti_EventGroup,
) -> CUptiResult;

/// function pointer for `cuptiEventGroupAddEvent`
pub type PFN_cuptiEventGroupAddEvent = unsafe extern "C" fn(
    event_group: CUpti_EventGroup,
    event: CUpti_EventID,
) -> CUptiResult;

/// function pointer for `cuptiEventGroupRemoveEvent`
pub type PFN_cuptiEventGroupRemoveEvent = unsafe extern "C" fn(
    event_group: CUpti_EventGroup,
    event: CUpti_EventID,
) -> CUptiResult;

/// function pointer for `cuptiEventGroupEnable`
pub type PFN_cuptiEventGroupEnable = unsafe extern "C" fn(
    event_group: CUpti_EventGroup,
) -> CUptiResult;

/// function pointer for `cuptiEventGroupDisable`
pub type PFN_cuptiEventGroupDisable = unsafe extern "C" fn(
    event_group: CUpti_EventGroup,
) -> CUptiResult;

/// function pointer for `cuptiEventGroupReadEvent`
pub type PFN_cuptiEventGroupReadEvent = unsafe extern "C" fn(
    event_group: CUpti_EventGroup,
    flags: u32,
    event: CUpti_EventID,
    event_value_buffer_size_bytes: *mut usize,
    event_value_buffer: *mut u64,
) -> CUptiResult;

/// function pointer for `cuptiEventGetIdFromName`
pub type PFN_cuptiEventGetIdFromName = unsafe extern "C" fn(
    device: i32,
    event_name: *const core::ffi::c_char,
    event: *mut CUpti_EventID,
) -> CUptiResult;

/// function pointer for `cuptiEventGetAttribute`
pub type PFN_cuptiEventGetAttribute = unsafe extern "C" fn(
    event: CUpti_EventID,
    attrib: u32,
    value_size: *mut usize,
    value: *mut c_void,
) -> CUptiResult;

// ---- Metric API ----------------------------------------------------------

/// function pointer for `cuptiMetricGetIdFromName`
pub type PFN_cuptiMetricGetIdFromName = unsafe extern "C" fn(
    device: i32,
    metric_name: *const core::ffi::c_char,
    metric: *mut CUpti_MetricID,
) -> CUptiResult;

/// function pointer for `cuptiMetricGetAttribute`
pub type PFN_cuptiMetricGetAttribute = unsafe extern "C" fn(
    metric: CUpti_MetricID,
    attrib: u32,
    value_size: *mut usize,
    value: *mut c_void,
) -> CUptiResult;

/// function pointer for `cuptiMetricCreateEventGroupSets`
pub type PFN_cuptiMetricCreateEventGroupSets = unsafe extern "C" fn(
    ctx: *mut c_void,
    metric_id_array_size_bytes: usize,
    metric_id_array: *mut CUpti_MetricID,
    event_group_passes: *mut *mut CUpti_EventGroupSets,
) -> CUptiResult;

/// function pointer for `cuptiMetricGetValue`
pub type PFN_cuptiMetricGetValue = unsafe extern "C" fn(
    device: i32,
    metric: CUpti_MetricID,
    event_id_array_size_bytes: usize,
    event_id_array: *mut CUpti_EventID,
    event_value_array_size_bytes: usize,
    event_value_array: *mut u64,
    time_duration: u64,
    metric_value: *mut c_void,
) -> CUptiResult;

/// function pointer for `cuptiEventGroupSetsDestroy`
pub type PFN_cuptiEventGroupSetsDestroy = unsafe extern "C" fn(
    event_group_sets: *mut CUpti_EventGroupSets,
) -> CUptiResult;

// ---- Profiler Host API (modern, replaces event/metric) ------------------

/// function pointer for `cuptiProfilerInitialize`
pub type PFN_cuptiProfilerInitialize = unsafe extern "C" fn(params: *mut c_void) -> CUptiResult;
/// function pointer for `cuptiProfilerDeInitialize`
pub type PFN_cuptiProfilerDeInitialize = unsafe extern "C" fn(params: *mut c_void) -> CUptiResult;
/// function pointer for `cuptiProfilerBeginSession`
pub type PFN_cuptiProfilerBeginSession = unsafe extern "C" fn(params: *mut c_void) -> CUptiResult;
/// function pointer for `cuptiProfilerEndSession`
pub type PFN_cuptiProfilerEndSession = unsafe extern "C" fn(params: *mut c_void) -> CUptiResult;
/// function pointer for `cuptiProfilerSetConfig`
pub type PFN_cuptiProfilerSetConfig = unsafe extern "C" fn(params: *mut c_void) -> CUptiResult;
/// function pointer for `cuptiProfilerUnsetConfig`
pub type PFN_cuptiProfilerUnsetConfig = unsafe extern "C" fn(params: *mut c_void) -> CUptiResult;
/// function pointer for `cuptiProfilerBeginPass`
pub type PFN_cuptiProfilerBeginPass = unsafe extern "C" fn(params: *mut c_void) -> CUptiResult;
/// function pointer for `cuptiProfilerEndPass`
pub type PFN_cuptiProfilerEndPass = unsafe extern "C" fn(params: *mut c_void) -> CUptiResult;
/// function pointer for `cuptiProfilerEnableProfiling`
pub type PFN_cuptiProfilerEnableProfiling =
    unsafe extern "C" fn(params: *mut c_void) -> CUptiResult;
/// function pointer for `cuptiProfilerDisableProfiling`
pub type PFN_cuptiProfilerDisableProfiling =
    unsafe extern "C" fn(params: *mut c_void) -> CUptiResult;
/// function pointer for `cuptiProfilerPushRange`
pub type PFN_cuptiProfilerPushRange =
    unsafe extern "C" fn(params: *mut c_void) -> CUptiResult;
/// function pointer for `cuptiProfilerPopRange`
pub type PFN_cuptiProfilerPopRange = unsafe extern "C" fn(params: *mut c_void) -> CUptiResult;
/// function pointer for `cuptiProfilerGetCounterAvailability`
pub type PFN_cuptiProfilerGetCounterAvailability =
    unsafe extern "C" fn(params: *mut c_void) -> CUptiResult;
/// function pointer for `cuptiProfilerFlushCounterData`
pub type PFN_cuptiProfilerFlushCounterData =
    unsafe extern "C" fn(params: *mut c_void) -> CUptiResult;

// ---- PC sampling --------------------------------------------------------

/// function pointer for `cuptiActivityConfigurePCSampling`
pub type PFN_cuptiActivityConfigurePCSampling = unsafe extern "C" fn(
    ctx: *mut c_void,
    config: *mut c_void,
) -> CUptiResult;

// ---- Loader ----

macro_rules! cupti_fns {
    ($($(#[$attr:meta])* fn $name:ident as $sym:literal : $pfn:ty;)*) => {
        /// Dynamic loader handle for CUPTI.
        pub struct Cupti {
            /// Underlying dynamic-library handle.
            pub lib: Library,
            $(
                $name: OnceLock<$pfn>,
            )*
        }

        impl core::fmt::Debug for Cupti {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct("Cupti").field("lib", &self.lib).finish_non_exhaustive()
            }
        }

        impl Cupti {
            fn empty(lib: Library) -> Self {
                Self { lib, $($name: OnceLock::new(),)* }
            }
            $(
                $(#[$attr])*
                #[doc = concat!("Resolve `", $sym, "`.")]
                pub fn $name(&self) -> Result<$pfn, LoaderError> {
                    if let Some(&p) = self.$name.get() { return Ok(p); }
                    let raw: *mut () = unsafe { self.lib.raw_symbol($sym)? };
                    let p: $pfn = unsafe { core::mem::transmute_copy::<*mut (), $pfn>(&raw) };
                    let _ = self.$name.set(p);
                    Ok(p)
                }
            )*
        }
    };
}

cupti_fns! {
    fn cupti_get_version as "cuptiGetVersion": PFN_cuptiGetVersion;
    fn cupti_get_last_error as "cuptiGetLastError": PFN_cuptiGetLastError;
    fn cupti_get_result_string as "cuptiGetResultString": PFN_cuptiGetResultString;
    fn cupti_get_timestamp as "cuptiGetTimestamp": PFN_cuptiGetTimestamp;
    fn cupti_activity_enable as "cuptiActivityEnable": PFN_cuptiActivityEnable;
    fn cupti_activity_disable as "cuptiActivityDisable": PFN_cuptiActivityDisable;
    fn cupti_activity_flush_all as "cuptiActivityFlushAll": PFN_cuptiActivityFlushAll;
    fn cupti_activity_register_callbacks as "cuptiActivityRegisterCallbacks":
        PFN_cuptiActivityRegisterCallbacks;
    fn cupti_activity_get_next_record as "cuptiActivityGetNextRecord":
        PFN_cuptiActivityGetNextRecord;
    fn cupti_subscribe as "cuptiSubscribe": PFN_cuptiSubscribe;
    fn cupti_unsubscribe as "cuptiUnsubscribe": PFN_cuptiUnsubscribe;
    fn cupti_enable_callback as "cuptiEnableCallback": PFN_cuptiEnableCallback;
    fn cupti_enable_domain as "cuptiEnableDomain": PFN_cuptiEnableDomain;

    // Event API
    fn cupti_device_get_num_event_domains as "cuptiDeviceGetNumEventDomains": PFN_cuptiDeviceGetNumEventDomains;
    fn cupti_device_enum_event_domains as "cuptiDeviceEnumEventDomains": PFN_cuptiDeviceEnumEventDomains;
    fn cupti_event_group_create as "cuptiEventGroupCreate": PFN_cuptiEventGroupCreate;
    fn cupti_event_group_destroy as "cuptiEventGroupDestroy": PFN_cuptiEventGroupDestroy;
    fn cupti_event_group_add_event as "cuptiEventGroupAddEvent": PFN_cuptiEventGroupAddEvent;
    fn cupti_event_group_remove_event as "cuptiEventGroupRemoveEvent": PFN_cuptiEventGroupRemoveEvent;
    fn cupti_event_group_enable as "cuptiEventGroupEnable": PFN_cuptiEventGroupEnable;
    fn cupti_event_group_disable as "cuptiEventGroupDisable": PFN_cuptiEventGroupDisable;
    fn cupti_event_group_read_event as "cuptiEventGroupReadEvent": PFN_cuptiEventGroupReadEvent;
    fn cupti_event_get_id_from_name as "cuptiEventGetIdFromName": PFN_cuptiEventGetIdFromName;
    fn cupti_event_get_attribute as "cuptiEventGetAttribute": PFN_cuptiEventGetAttribute;

    // Metric API
    fn cupti_metric_get_id_from_name as "cuptiMetricGetIdFromName": PFN_cuptiMetricGetIdFromName;
    fn cupti_metric_get_attribute as "cuptiMetricGetAttribute": PFN_cuptiMetricGetAttribute;
    fn cupti_metric_create_event_group_sets as "cuptiMetricCreateEventGroupSets": PFN_cuptiMetricCreateEventGroupSets;
    fn cupti_metric_get_value as "cuptiMetricGetValue": PFN_cuptiMetricGetValue;
    fn cupti_event_group_sets_destroy as "cuptiEventGroupSetsDestroy": PFN_cuptiEventGroupSetsDestroy;

    // Profiler Host API
    fn cupti_profiler_initialize as "cuptiProfilerInitialize": PFN_cuptiProfilerInitialize;
    fn cupti_profiler_deinitialize as "cuptiProfilerDeInitialize": PFN_cuptiProfilerDeInitialize;
    fn cupti_profiler_begin_session as "cuptiProfilerBeginSession": PFN_cuptiProfilerBeginSession;
    fn cupti_profiler_end_session as "cuptiProfilerEndSession": PFN_cuptiProfilerEndSession;
    fn cupti_profiler_set_config as "cuptiProfilerSetConfig": PFN_cuptiProfilerSetConfig;
    fn cupti_profiler_unset_config as "cuptiProfilerUnsetConfig": PFN_cuptiProfilerUnsetConfig;
    fn cupti_profiler_begin_pass as "cuptiProfilerBeginPass": PFN_cuptiProfilerBeginPass;
    fn cupti_profiler_end_pass as "cuptiProfilerEndPass": PFN_cuptiProfilerEndPass;
    fn cupti_profiler_enable_profiling as "cuptiProfilerEnableProfiling": PFN_cuptiProfilerEnableProfiling;
    fn cupti_profiler_disable_profiling as "cuptiProfilerDisableProfiling": PFN_cuptiProfilerDisableProfiling;
    fn cupti_profiler_push_range as "cuptiProfilerPushRange": PFN_cuptiProfilerPushRange;
    fn cupti_profiler_pop_range as "cuptiProfilerPopRange": PFN_cuptiProfilerPopRange;
    fn cupti_profiler_get_counter_availability as "cuptiProfilerGetCounterAvailability": PFN_cuptiProfilerGetCounterAvailability;
    fn cupti_profiler_flush_counter_data as "cuptiProfilerFlushCounterData": PFN_cuptiProfilerFlushCounterData;

    // PC sampling
    fn cupti_activity_configure_pc_sampling as "cuptiActivityConfigurePCSampling": PFN_cuptiActivityConfigurePCSampling;
}

fn cupti_candidates() -> &'static [&'static str] {
    #[cfg(target_os = "linux")]
    {
        &["libcupti.so.13", "libcupti.so.12", "libcupti.so"]
    }
    #[cfg(target_os = "windows")]
    {
        // CUPTI DLLs on Windows often have version suffixes.
        &["cupti64_2025.1.0.dll", "cupti64_2024.1.0.dll", "cupti.dll"]
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        &[]
    }
}

fn cupti_extra_dirs() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if cfg!(target_os = "windows") {
        for var in ["CUDA_PATH", "CUDA_HOME"] {
            if let Ok(raw) = std::env::var(var) {
                out.push(
                    PathBuf::from(&raw)
                        .join("extras")
                        .join("CUPTI")
                        .join("lib64"),
                );
                out.push(PathBuf::from(raw).join("extras").join("CUPTI").join("lib"));
            }
        }
    }
    out
}

/// Open (or return the cached) CUPTI dynamic library.
pub fn cupti() -> Result<&'static Cupti, LoaderError> {
    static CUPTI: OnceLock<Cupti> = OnceLock::new();
    if let Some(c) = CUPTI.get() {
        return Ok(c);
    }
    let lib = match Library::open("cupti", cupti_candidates()) {
        Ok(l) => l,
        Err(_) => {
            let mut found: Option<Library> = None;
            for dir in cupti_extra_dirs() {
                for candidate in cupti_candidates() {
                    let full = dir.join(candidate);
                    if let Ok(l) = Library::open_at("cupti", &full) {
                        found = Some(l);
                        break;
                    }
                }
                if found.is_some() {
                    break;
                }
            }
            found.ok_or_else(|| {
                LoaderError::library_not_found_with_search("cupti", cupti_candidates(), 2)
            })?
        }
    };
    let _ = CUPTI.set(Cupti::empty(lib));
    Ok(CUPTI.get().expect("OnceLock set or lost race"))
}

#[allow(dead_code)]
fn _touch() -> *mut c_void {
    core::ptr::null_mut()
}
