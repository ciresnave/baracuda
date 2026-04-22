//! Raw FFI + dynamic loader for the NVIDIA Management Library (NVML).
//!
//! NVML ships with the NVIDIA driver (not the CUDA toolkit), so it's
//! available on any host with a functional NVIDIA GPU — no separate
//! install needed.

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
#![warn(missing_debug_implementations)]

use core::ffi::{c_char, c_uint, c_void};
use std::path::PathBuf;
use std::sync::OnceLock;

use baracuda_core::{Library, LoaderError};
use baracuda_types::CudaStatus;

/// Opaque NVML device handle.
pub type nvmlDevice_t = *mut c_void;
/// Opaque NVML unit handle (for NVSwitch-based systems).
pub type nvmlUnit_t = *mut c_void;
/// Opaque NVML event-set handle.
pub type nvmlEventSet_t = *mut c_void;

/// Temperature sensor selector.
pub const NVML_TEMPERATURE_GPU: c_uint = 0;

/// Clock-type selector: 0 graphics, 1 SM, 2 memory, 3 video.
#[repr(u32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum nvmlClockType_t {
    Graphics = 0,
    Sm = 1,
    Mem = 2,
    Video = 3,
}

/// Performance-state enum.
#[repr(u32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum nvmlPstates_t {
    P0 = 0, P1 = 1, P2 = 2, P3 = 3, P4 = 4, P5 = 5, P6 = 6, P7 = 7,
    P8 = 8, P9 = 9, P10 = 10, P11 = 11, P12 = 12, P13 = 13, P14 = 14, P15 = 15,
    Unknown = 32,
}

/// ECC counter type: 0 volatile, 1 aggregate.
#[repr(u32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum nvmlEccCounterType_t {
    Volatile = 0,
    Aggregate = 1,
}

/// Memory error type: 0 corrected, 1 uncorrected.
#[repr(u32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum nvmlMemoryErrorType_t {
    Corrected = 0,
    Uncorrected = 1,
}

/// Memory location in the ECC hierarchy.
#[repr(u32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum nvmlMemoryLocation_t {
    L1Cache = 0,
    L2Cache = 1,
    DeviceMemory = 2,
    RegisterFile = 3,
    TextureMemory = 4,
    SharedMemory = 5,
    Cbu = 6,
    Sram = 7,
}

/// Compute mode. Controls context concurrency on the device.
#[repr(u32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum nvmlComputeMode_t {
    Default = 0,
    ExclusiveThread = 1,
    Prohibited = 2,
    ExclusiveProcess = 3,
}

/// `nvmlMemoryInfo_v2_t` — present on newer drivers; falls back to v1.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct nvmlMemory_v2_t {
    pub version: c_uint,
    pub total: u64,
    pub reserved: u64,
    pub free: u64,
    pub used: u64,
}

/// PCIe info.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct nvmlPciInfo_t {
    pub bus_id_legacy: [c_char; 16],
    pub domain: c_uint,
    pub bus: c_uint,
    pub device: c_uint,
    pub pci_device_id: c_uint,
    pub pci_subsystem_id: c_uint,
    pub bus_id: [c_char; 32],
}

/// Per-process running on the GPU (with compute context).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct nvmlProcessInfo_t {
    pub pid: c_uint,
    pub used_gpu_memory: u64,
    pub gpu_instance_id: c_uint,
    pub compute_instance_id: c_uint,
}

/// Field-value query union.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct nvmlFieldValue_t {
    pub field_id: c_uint,
    pub scope_id: c_uint,
    pub timestamp: i64,
    pub latency_usec: i64,
    pub value_type: c_uint,
    pub nvml_return: nvmlReturn_t,
    pub value: [u64; 2],
}

impl Default for nvmlFieldValue_t {
    fn default() -> Self {
        Self {
            field_id: 0,
            scope_id: 0,
            timestamp: 0,
            latency_usec: 0,
            value_type: 0,
            nvml_return: nvmlReturn_t(0),
            value: [0; 2],
        }
    }
}

/// Memory info (`total`/`free`/`used` in bytes).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct nvmlMemory_t {
    pub total: u64,
    pub free: u64,
    pub used: u64,
}

/// GPU/memory utilization (0–100, integer percent).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct nvmlUtilization_t {
    pub gpu: c_uint,
    pub memory: c_uint,
}

// ---- status ---------------------------------------------------------------

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct nvmlReturn_t(pub i32);

impl nvmlReturn_t {
    pub const SUCCESS: Self = Self(0);
    pub const UNINITIALIZED: Self = Self(1);
    pub const INVALID_ARGUMENT: Self = Self(2);
    pub const NOT_SUPPORTED: Self = Self(3);
    pub const NO_PERMISSION: Self = Self(4);
    pub const ALREADY_INITIALIZED: Self = Self(5);
    pub const NOT_FOUND: Self = Self(6);
    pub const INSUFFICIENT_SIZE: Self = Self(7);
    pub const INSUFFICIENT_POWER: Self = Self(8);
    pub const DRIVER_NOT_LOADED: Self = Self(9);
    pub const TIMEOUT: Self = Self(10);
    pub const GPU_IS_LOST: Self = Self(15);
    pub const RESET_REQUIRED: Self = Self(16);
    pub const UNKNOWN: Self = Self(999);

    pub const fn is_success(self) -> bool {
        self.0 == 0
    }
}

impl CudaStatus for nvmlReturn_t {
    fn code(self) -> i32 {
        self.0
    }
    fn name(self) -> &'static str {
        match self.0 {
            0 => "NVML_SUCCESS",
            1 => "NVML_ERROR_UNINITIALIZED",
            2 => "NVML_ERROR_INVALID_ARGUMENT",
            3 => "NVML_ERROR_NOT_SUPPORTED",
            4 => "NVML_ERROR_NO_PERMISSION",
            6 => "NVML_ERROR_NOT_FOUND",
            9 => "NVML_ERROR_DRIVER_NOT_LOADED",
            15 => "NVML_ERROR_GPU_IS_LOST",
            16 => "NVML_ERROR_RESET_REQUIRED",
            999 => "NVML_ERROR_UNKNOWN",
            _ => "NVML_ERROR_UNRECOGNIZED",
        }
    }
    fn description(self) -> &'static str {
        match self.0 {
            0 => "success",
            1 => "NVML not initialized; call nvmlInit first",
            3 => "query not supported on this device",
            9 => "NVIDIA driver not loaded",
            15 => "GPU is lost (unresponsive)",
            16 => "GPU requires a reset",
            _ => "unrecognized NVML status code",
        }
    }
    fn is_success(self) -> bool {
        nvmlReturn_t::is_success(self)
    }
    fn library(self) -> &'static str {
        "nvml"
    }
}

// ---- function-pointer types ----------------------------------------------

pub type PFN_nvmlInit = unsafe extern "C" fn() -> nvmlReturn_t;
pub type PFN_nvmlShutdown = unsafe extern "C" fn() -> nvmlReturn_t;
pub type PFN_nvmlSystemGetDriverVersion =
    unsafe extern "C" fn(version: *mut c_char, length: c_uint) -> nvmlReturn_t;
pub type PFN_nvmlSystemGetCudaDriverVersion =
    unsafe extern "C" fn(cuda_driver_version: *mut core::ffi::c_int) -> nvmlReturn_t;
pub type PFN_nvmlDeviceGetCount = unsafe extern "C" fn(count: *mut c_uint) -> nvmlReturn_t;
pub type PFN_nvmlDeviceGetHandleByIndex =
    unsafe extern "C" fn(index: c_uint, device: *mut nvmlDevice_t) -> nvmlReturn_t;
pub type PFN_nvmlDeviceGetName =
    unsafe extern "C" fn(device: nvmlDevice_t, name: *mut c_char, length: c_uint) -> nvmlReturn_t;
pub type PFN_nvmlDeviceGetMemoryInfo =
    unsafe extern "C" fn(device: nvmlDevice_t, memory: *mut nvmlMemory_t) -> nvmlReturn_t;
pub type PFN_nvmlDeviceGetTemperature = unsafe extern "C" fn(
    device: nvmlDevice_t,
    sensor_type: c_uint,
    temp: *mut c_uint,
) -> nvmlReturn_t;
pub type PFN_nvmlDeviceGetPowerUsage =
    unsafe extern "C" fn(device: nvmlDevice_t, power: *mut c_uint) -> nvmlReturn_t;
pub type PFN_nvmlDeviceGetFanSpeed =
    unsafe extern "C" fn(device: nvmlDevice_t, speed: *mut c_uint) -> nvmlReturn_t;
pub type PFN_nvmlDeviceGetUtilizationRates =
    unsafe extern "C" fn(device: nvmlDevice_t, util: *mut nvmlUtilization_t) -> nvmlReturn_t;

// ---- Clocks / power / performance ---------------------------------------

pub type PFN_nvmlDeviceGetClockInfo = unsafe extern "C" fn(
    device: nvmlDevice_t,
    ty: nvmlClockType_t,
    clock_mhz: *mut c_uint,
) -> nvmlReturn_t;

pub type PFN_nvmlDeviceGetMaxClockInfo = unsafe extern "C" fn(
    device: nvmlDevice_t,
    ty: nvmlClockType_t,
    clock_mhz: *mut c_uint,
) -> nvmlReturn_t;

pub type PFN_nvmlDeviceGetApplicationsClock = unsafe extern "C" fn(
    device: nvmlDevice_t,
    ty: nvmlClockType_t,
    clock_mhz: *mut c_uint,
) -> nvmlReturn_t;

pub type PFN_nvmlDeviceGetDefaultApplicationsClock = unsafe extern "C" fn(
    device: nvmlDevice_t,
    ty: nvmlClockType_t,
    clock_mhz: *mut c_uint,
) -> nvmlReturn_t;

pub type PFN_nvmlDeviceSetApplicationsClocks = unsafe extern "C" fn(
    device: nvmlDevice_t,
    mem_mhz: c_uint,
    graphics_mhz: c_uint,
) -> nvmlReturn_t;

pub type PFN_nvmlDeviceGetPowerManagementLimit = unsafe extern "C" fn(
    device: nvmlDevice_t,
    limit: *mut c_uint,
) -> nvmlReturn_t;

pub type PFN_nvmlDeviceGetPowerManagementLimitConstraints = unsafe extern "C" fn(
    device: nvmlDevice_t,
    min_limit: *mut c_uint,
    max_limit: *mut c_uint,
) -> nvmlReturn_t;

pub type PFN_nvmlDeviceSetPowerManagementLimit = unsafe extern "C" fn(
    device: nvmlDevice_t,
    limit: c_uint,
) -> nvmlReturn_t;

pub type PFN_nvmlDeviceGetPowerState = unsafe extern "C" fn(
    device: nvmlDevice_t,
    p_state: *mut nvmlPstates_t,
) -> nvmlReturn_t;

pub type PFN_nvmlDeviceGetPerformanceState = unsafe extern "C" fn(
    device: nvmlDevice_t,
    p_state: *mut nvmlPstates_t,
) -> nvmlReturn_t;

pub type PFN_nvmlDeviceGetTemperatureThreshold = unsafe extern "C" fn(
    device: nvmlDevice_t,
    threshold_type: c_uint,
    temp: *mut c_uint,
) -> nvmlReturn_t;

// ---- ECC -----------------------------------------------------------------

pub type PFN_nvmlDeviceGetMemoryErrorCounter = unsafe extern "C" fn(
    device: nvmlDevice_t,
    error_type: nvmlMemoryErrorType_t,
    counter_type: nvmlEccCounterType_t,
    location: nvmlMemoryLocation_t,
    count: *mut u64,
) -> nvmlReturn_t;

pub type PFN_nvmlDeviceGetTotalEccErrors = unsafe extern "C" fn(
    device: nvmlDevice_t,
    error_type: nvmlMemoryErrorType_t,
    counter_type: nvmlEccCounterType_t,
    count: *mut u64,
) -> nvmlReturn_t;

pub type PFN_nvmlDeviceGetEccMode = unsafe extern "C" fn(
    device: nvmlDevice_t,
    current: *mut c_uint,
    pending: *mut c_uint,
) -> nvmlReturn_t;

pub type PFN_nvmlDeviceSetEccMode =
    unsafe extern "C" fn(device: nvmlDevice_t, ecc: c_uint) -> nvmlReturn_t;

// ---- PCIe / NVLink -------------------------------------------------------

pub type PFN_nvmlDeviceGetPciInfo_v3 = unsafe extern "C" fn(
    device: nvmlDevice_t,
    pci: *mut nvmlPciInfo_t,
) -> nvmlReturn_t;

pub type PFN_nvmlDeviceGetCurrPcieLinkGeneration = unsafe extern "C" fn(
    device: nvmlDevice_t,
    gen: *mut c_uint,
) -> nvmlReturn_t;

pub type PFN_nvmlDeviceGetCurrPcieLinkWidth = unsafe extern "C" fn(
    device: nvmlDevice_t,
    width: *mut c_uint,
) -> nvmlReturn_t;

pub type PFN_nvmlDeviceGetPcieThroughput = unsafe extern "C" fn(
    device: nvmlDevice_t,
    counter: c_uint,
    value: *mut c_uint,
) -> nvmlReturn_t;

pub type PFN_nvmlDeviceGetNvLinkState = unsafe extern "C" fn(
    device: nvmlDevice_t,
    link: c_uint,
    is_active: *mut c_uint,
) -> nvmlReturn_t;

pub type PFN_nvmlDeviceGetNvLinkVersion = unsafe extern "C" fn(
    device: nvmlDevice_t,
    link: c_uint,
    version: *mut c_uint,
) -> nvmlReturn_t;

pub type PFN_nvmlDeviceGetNvLinkCapability = unsafe extern "C" fn(
    device: nvmlDevice_t,
    link: c_uint,
    capability: c_uint,
    cap_result: *mut c_uint,
) -> nvmlReturn_t;

// ---- Processes -----------------------------------------------------------

pub type PFN_nvmlDeviceGetComputeRunningProcesses_v3 = unsafe extern "C" fn(
    device: nvmlDevice_t,
    info_count: *mut c_uint,
    infos: *mut nvmlProcessInfo_t,
) -> nvmlReturn_t;

pub type PFN_nvmlDeviceGetGraphicsRunningProcesses_v3 = unsafe extern "C" fn(
    device: nvmlDevice_t,
    info_count: *mut c_uint,
    infos: *mut nvmlProcessInfo_t,
) -> nvmlReturn_t;

// ---- Compute mode --------------------------------------------------------

pub type PFN_nvmlDeviceGetComputeMode = unsafe extern "C" fn(
    device: nvmlDevice_t,
    mode: *mut nvmlComputeMode_t,
) -> nvmlReturn_t;

pub type PFN_nvmlDeviceSetComputeMode = unsafe extern "C" fn(
    device: nvmlDevice_t,
    mode: nvmlComputeMode_t,
) -> nvmlReturn_t;

// ---- Identity / UUID / serial --------------------------------------------

pub type PFN_nvmlDeviceGetUUID =
    unsafe extern "C" fn(device: nvmlDevice_t, uuid: *mut c_char, length: c_uint) -> nvmlReturn_t;

pub type PFN_nvmlDeviceGetSerial =
    unsafe extern "C" fn(device: nvmlDevice_t, serial: *mut c_char, length: c_uint) -> nvmlReturn_t;

pub type PFN_nvmlDeviceGetIndex =
    unsafe extern "C" fn(device: nvmlDevice_t, index: *mut c_uint) -> nvmlReturn_t;

pub type PFN_nvmlDeviceGetMinorNumber =
    unsafe extern "C" fn(device: nvmlDevice_t, minor: *mut c_uint) -> nvmlReturn_t;

pub type PFN_nvmlDeviceGetHandleByUUID = unsafe extern "C" fn(
    uuid: *const c_char,
    device: *mut nvmlDevice_t,
) -> nvmlReturn_t;

pub type PFN_nvmlDeviceGetHandleByPciBusId_v2 = unsafe extern "C" fn(
    pci_bus_id: *const c_char,
    device: *mut nvmlDevice_t,
) -> nvmlReturn_t;

// ---- Event set -----------------------------------------------------------

pub type PFN_nvmlEventSetCreate = unsafe extern "C" fn(set: *mut nvmlEventSet_t) -> nvmlReturn_t;
pub type PFN_nvmlEventSetFree = unsafe extern "C" fn(set: nvmlEventSet_t) -> nvmlReturn_t;
pub type PFN_nvmlDeviceRegisterEvents = unsafe extern "C" fn(
    device: nvmlDevice_t,
    event_types: u64,
    set: nvmlEventSet_t,
) -> nvmlReturn_t;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct nvmlEventData_t {
    pub device: nvmlDevice_t,
    pub event_type: u64,
    pub event_data: u64,
    pub gpu_instance_id: c_uint,
    pub compute_instance_id: c_uint,
}

pub type PFN_nvmlEventSetWait_v2 = unsafe extern "C" fn(
    set: nvmlEventSet_t,
    data: *mut nvmlEventData_t,
    timeout_ms: c_uint,
) -> nvmlReturn_t;

// ---- Field values --------------------------------------------------------

pub type PFN_nvmlDeviceGetFieldValues = unsafe extern "C" fn(
    device: nvmlDevice_t,
    values_count: core::ffi::c_int,
    values: *mut nvmlFieldValue_t,
) -> nvmlReturn_t;

// ---- loader --------------------------------------------------------------

/// NVML ships with the NVIDIA driver. Windows puts it in `System32\nvml.dll`
/// or `NVIDIA Corporation\NVSMI\nvml.dll`; Linux ships `libnvidia-ml.so.1`.
fn nvml_candidates() -> &'static [&'static str] {
    #[cfg(target_os = "linux")]
    {
        &["libnvidia-ml.so.1", "libnvidia-ml.so"]
    }
    #[cfg(target_os = "windows")]
    {
        &["nvml.dll"]
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        &[]
    }
}

fn nvml_extra_search_dirs() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if cfg!(target_os = "windows") {
        if let Ok(pf) = std::env::var("ProgramFiles") {
            // NVSMI is where older drivers kept nvml.dll.
            out.push(PathBuf::from(pf).join("NVIDIA Corporation").join("NVSMI"));
        }
    }
    out
}

macro_rules! nvml_fns {
    ($($name:ident as $sym:literal : $pfn:ty);* $(;)?) => {
        pub struct Nvml {
            lib: Library,
            $($name: OnceLock<$pfn>,)*
        }
        impl core::fmt::Debug for Nvml {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct("Nvml").field("lib", &self.lib).finish_non_exhaustive()
            }
        }
        impl Nvml {
            $(
                pub fn $name(&self) -> Result<$pfn, LoaderError> {
                    if let Some(&p) = self.$name.get() { return Ok(p); }
                    let raw: *mut () = unsafe { self.lib.raw_symbol($sym)? };
                    let p: $pfn = unsafe { core::mem::transmute_copy::<*mut (), $pfn>(&raw) };
                    let _ = self.$name.set(p);
                    Ok(p)
                }
            )*
            fn empty(lib: Library) -> Self {
                Self { lib, $($name: OnceLock::new(),)* }
            }
        }
    };
}

nvml_fns! {
    // Some symbols have `_v2` suffix — prefer those where available for the
    // modern ABI. NVML's dlsym'd symbols include both.
    nvml_init as "nvmlInit_v2": PFN_nvmlInit;
    nvml_shutdown as "nvmlShutdown": PFN_nvmlShutdown;
    nvml_system_get_driver_version as "nvmlSystemGetDriverVersion": PFN_nvmlSystemGetDriverVersion;
    nvml_system_get_cuda_driver_version as "nvmlSystemGetCudaDriverVersion": PFN_nvmlSystemGetCudaDriverVersion;
    nvml_device_get_count as "nvmlDeviceGetCount_v2": PFN_nvmlDeviceGetCount;
    nvml_device_get_handle_by_index as "nvmlDeviceGetHandleByIndex_v2": PFN_nvmlDeviceGetHandleByIndex;
    nvml_device_get_name as "nvmlDeviceGetName": PFN_nvmlDeviceGetName;
    nvml_device_get_memory_info as "nvmlDeviceGetMemoryInfo": PFN_nvmlDeviceGetMemoryInfo;
    nvml_device_get_temperature as "nvmlDeviceGetTemperature": PFN_nvmlDeviceGetTemperature;
    nvml_device_get_power_usage as "nvmlDeviceGetPowerUsage": PFN_nvmlDeviceGetPowerUsage;
    nvml_device_get_fan_speed as "nvmlDeviceGetFanSpeed": PFN_nvmlDeviceGetFanSpeed;
    nvml_device_get_utilization_rates as "nvmlDeviceGetUtilizationRates": PFN_nvmlDeviceGetUtilizationRates;

    // Clocks / power / perf state
    nvml_device_get_clock_info as "nvmlDeviceGetClockInfo": PFN_nvmlDeviceGetClockInfo;
    nvml_device_get_max_clock_info as "nvmlDeviceGetMaxClockInfo": PFN_nvmlDeviceGetMaxClockInfo;
    nvml_device_get_applications_clock as "nvmlDeviceGetApplicationsClock": PFN_nvmlDeviceGetApplicationsClock;
    nvml_device_get_default_applications_clock as "nvmlDeviceGetDefaultApplicationsClock": PFN_nvmlDeviceGetDefaultApplicationsClock;
    nvml_device_set_applications_clocks as "nvmlDeviceSetApplicationsClocks": PFN_nvmlDeviceSetApplicationsClocks;
    nvml_device_get_power_management_limit as "nvmlDeviceGetPowerManagementLimit": PFN_nvmlDeviceGetPowerManagementLimit;
    nvml_device_get_power_management_limit_constraints as "nvmlDeviceGetPowerManagementLimitConstraints": PFN_nvmlDeviceGetPowerManagementLimitConstraints;
    nvml_device_set_power_management_limit as "nvmlDeviceSetPowerManagementLimit": PFN_nvmlDeviceSetPowerManagementLimit;
    nvml_device_get_power_state as "nvmlDeviceGetPowerState": PFN_nvmlDeviceGetPowerState;
    nvml_device_get_performance_state as "nvmlDeviceGetPerformanceState": PFN_nvmlDeviceGetPerformanceState;
    nvml_device_get_temperature_threshold as "nvmlDeviceGetTemperatureThreshold": PFN_nvmlDeviceGetTemperatureThreshold;

    // ECC
    nvml_device_get_memory_error_counter as "nvmlDeviceGetMemoryErrorCounter": PFN_nvmlDeviceGetMemoryErrorCounter;
    nvml_device_get_total_ecc_errors as "nvmlDeviceGetTotalEccErrors": PFN_nvmlDeviceGetTotalEccErrors;
    nvml_device_get_ecc_mode as "nvmlDeviceGetEccMode": PFN_nvmlDeviceGetEccMode;
    nvml_device_set_ecc_mode as "nvmlDeviceSetEccMode": PFN_nvmlDeviceSetEccMode;

    // PCIe / NVLink
    nvml_device_get_pci_info_v3 as "nvmlDeviceGetPciInfo_v3": PFN_nvmlDeviceGetPciInfo_v3;
    nvml_device_get_curr_pcie_link_generation as "nvmlDeviceGetCurrPcieLinkGeneration": PFN_nvmlDeviceGetCurrPcieLinkGeneration;
    nvml_device_get_curr_pcie_link_width as "nvmlDeviceGetCurrPcieLinkWidth": PFN_nvmlDeviceGetCurrPcieLinkWidth;
    nvml_device_get_pcie_throughput as "nvmlDeviceGetPcieThroughput": PFN_nvmlDeviceGetPcieThroughput;
    nvml_device_get_nvlink_state as "nvmlDeviceGetNvLinkState": PFN_nvmlDeviceGetNvLinkState;
    nvml_device_get_nvlink_version as "nvmlDeviceGetNvLinkVersion": PFN_nvmlDeviceGetNvLinkVersion;
    nvml_device_get_nvlink_capability as "nvmlDeviceGetNvLinkCapability": PFN_nvmlDeviceGetNvLinkCapability;

    // Processes
    nvml_device_get_compute_running_processes_v3 as "nvmlDeviceGetComputeRunningProcesses_v3": PFN_nvmlDeviceGetComputeRunningProcesses_v3;
    nvml_device_get_graphics_running_processes_v3 as "nvmlDeviceGetGraphicsRunningProcesses_v3": PFN_nvmlDeviceGetGraphicsRunningProcesses_v3;

    // Compute mode
    nvml_device_get_compute_mode as "nvmlDeviceGetComputeMode": PFN_nvmlDeviceGetComputeMode;
    nvml_device_set_compute_mode as "nvmlDeviceSetComputeMode": PFN_nvmlDeviceSetComputeMode;

    // Identity
    nvml_device_get_uuid as "nvmlDeviceGetUUID": PFN_nvmlDeviceGetUUID;
    nvml_device_get_serial as "nvmlDeviceGetSerial": PFN_nvmlDeviceGetSerial;
    nvml_device_get_index as "nvmlDeviceGetIndex": PFN_nvmlDeviceGetIndex;
    nvml_device_get_minor_number as "nvmlDeviceGetMinorNumber": PFN_nvmlDeviceGetMinorNumber;
    nvml_device_get_handle_by_uuid as "nvmlDeviceGetHandleByUUID": PFN_nvmlDeviceGetHandleByUUID;
    nvml_device_get_handle_by_pci_bus_id_v2 as "nvmlDeviceGetHandleByPciBusId_v2": PFN_nvmlDeviceGetHandleByPciBusId_v2;

    // Event set
    nvml_event_set_create as "nvmlEventSetCreate": PFN_nvmlEventSetCreate;
    nvml_event_set_free as "nvmlEventSetFree": PFN_nvmlEventSetFree;
    nvml_device_register_events as "nvmlDeviceRegisterEvents": PFN_nvmlDeviceRegisterEvents;
    nvml_event_set_wait_v2 as "nvmlEventSetWait_v2": PFN_nvmlEventSetWait_v2;

    // Field values
    nvml_device_get_field_values as "nvmlDeviceGetFieldValues": PFN_nvmlDeviceGetFieldValues;
}

pub fn nvml() -> Result<&'static Nvml, LoaderError> {
    static NVML: OnceLock<Nvml> = OnceLock::new();
    if let Some(n) = NVML.get() {
        return Ok(n);
    }
    let lib = match Library::open("nvml", nvml_candidates()) {
        Ok(l) => l,
        Err(_) => {
            // Fall back to explicit NVSMI path on Windows.
            let mut found: Option<Library> = None;
            for dir in nvml_extra_search_dirs() {
                for candidate in nvml_candidates() {
                    let full = dir.join(candidate);
                    if let Ok(l) = Library::open_at("nvml", &full) {
                        found = Some(l);
                        break;
                    }
                }
                if found.is_some() {
                    break;
                }
            }
            found.ok_or_else(|| {
                LoaderError::library_not_found_with_search("nvml", nvml_candidates(), 1)
            })?
        }
    };
    let n = Nvml::empty(lib);
    let _ = NVML.set(n);
    Ok(NVML.get().expect("OnceLock set or lost race"))
}
