//! Safe Rust wrappers for the NVIDIA Management Library (NVML).
//!
//! NVML is bundled with the NVIDIA driver (not the CUDA toolkit), so it's
//! always available on hosts with a working GPU. Use it to monitor GPU
//! health (temperature, power, utilization) and inspect the driver version.
//!
//! The library requires a one-time `nvmlInit_v2` / `nvmlShutdown` pair;
//! this crate does that automatically on first use via a process-global
//! guard that shuts down when the process exits.
//!
//! ```no_run
//! # fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! let nvml = baracuda_nvml::Nvml::init()?;
//! for dev in nvml.devices()? {
//!     println!(
//!         "{}: {}°C, {}W, {:.2} GiB used / {:.2} GiB total",
//!         dev.name()?,
//!         dev.temperature()?,
//!         dev.power_usage_watts()?,
//!         dev.memory_info()?.used as f64 / (1024.0 * 1024.0 * 1024.0),
//!         dev.memory_info()?.total as f64 / (1024.0 * 1024.0 * 1024.0),
//!     );
//! }
//! # Ok(()) }
//! ```

#![warn(missing_debug_implementations)]

use core::ffi::c_char;
use std::sync::OnceLock;

use baracuda_nvml_sys::{
    nvml, nvmlDevice_t, nvmlEventData_t, nvmlEventSet_t, nvmlFieldValue_t, nvmlMemory_t,
    nvmlPciInfo_t, nvmlProcessInfo_t, nvmlReturn_t, nvmlUtilization_t,
};

pub use baracuda_nvml_sys::nvmlClockType_t as ClockType;
pub use baracuda_nvml_sys::nvmlComputeMode_t as ComputeMode;
pub use baracuda_nvml_sys::nvmlEccCounterType_t as EccCounter;
pub use baracuda_nvml_sys::nvmlMemoryErrorType_t as MemoryError;
pub use baracuda_nvml_sys::nvmlMemoryLocation_t as MemoryLocation;
pub use baracuda_nvml_sys::nvmlPstates_t as PState;

/// Error type for NVML operations.
pub type Error = baracuda_core::Error<nvmlReturn_t>;
/// Result alias.
pub type Result<T, E = Error> = core::result::Result<T, E>;

#[inline]
fn check(status: nvmlReturn_t) -> Result<()> {
    Error::check(status)
}

/// RAII handle representing an initialized NVML instance.
///
/// A single instance is shared process-wide; clones are cheap and increment
/// a static guard. Actual shutdown happens when the last reference drops
/// (which also finalizes any Debug-derived handles NVML held).
#[derive(Clone)]
pub struct Nvml {
    // Unit marker — the underlying initialization lives in a OnceLock.
    _marker: (),
}

impl core::fmt::Debug for Nvml {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Nvml").finish_non_exhaustive()
    }
}

static INIT: OnceLock<Result<(), Error>> = OnceLock::new();

impl Nvml {
    /// Initialize NVML (idempotent across the process).
    pub fn init() -> Result<Self> {
        // Memoize init; first caller runs nvmlInit_v2.
        let result = INIT.get_or_init(|| -> Result<(), Error> {
            let n = nvml()?;
            let init = n.nvml_init()?;
            check(unsafe { init() })?;
            Ok(())
        });
        match result {
            Ok(()) => Ok(Self { _marker: () }),
            Err(e) => Err(clone_error(e)),
        }
    }

    /// The NVIDIA driver version string (e.g. `"595.79"`).
    pub fn driver_version(&self) -> Result<String> {
        let n = nvml()?;
        let cu = n.nvml_system_get_driver_version()?;
        let mut buf = [0i8; 80];
        check(unsafe { cu(buf.as_mut_ptr(), buf.len() as u32) })?;
        Ok(cstr_to_string(&buf))
    }

    /// CUDA-driver version reported by NVML, as a packed integer.
    pub fn cuda_driver_version(&self) -> Result<i32> {
        let n = nvml()?;
        let cu = n.nvml_system_get_cuda_driver_version()?;
        let mut v: core::ffi::c_int = 0;
        check(unsafe { cu(&mut v) })?;
        Ok(v)
    }

    /// All visible NVIDIA devices, in `nvmlDeviceGetHandleByIndex_v2` order.
    pub fn devices(&self) -> Result<Vec<Device>> {
        let n = nvml()?;
        let cu_count = n.nvml_device_get_count()?;
        let mut count: core::ffi::c_uint = 0;
        check(unsafe { cu_count(&mut count) })?;
        let cu_handle = n.nvml_device_get_handle_by_index()?;
        let mut out = Vec::with_capacity(count as usize);
        for i in 0..count {
            let mut h: nvmlDevice_t = core::ptr::null_mut();
            check(unsafe { cu_handle(i, &mut h) })?;
            out.push(Device { handle: h });
        }
        Ok(out)
    }
}

/// A physical NVIDIA GPU as seen by NVML. `Clone` is a cheap handle copy.
#[derive(Copy, Clone)]
pub struct Device {
    handle: nvmlDevice_t,
}

unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl core::fmt::Debug for Device {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("nvml::Device")
            .field("handle", &self.handle)
            .finish()
    }
}

impl Device {
    /// Human-readable GPU name.
    pub fn name(&self) -> Result<String> {
        let n = nvml()?;
        let cu = n.nvml_device_get_name()?;
        let mut buf = [0i8; 96];
        check(unsafe { cu(self.handle, buf.as_mut_ptr(), buf.len() as u32) })?;
        Ok(cstr_to_string(&buf))
    }

    /// GPU memory usage.
    pub fn memory_info(&self) -> Result<nvmlMemory_t> {
        let n = nvml()?;
        let cu = n.nvml_device_get_memory_info()?;
        let mut mem = nvmlMemory_t::default();
        check(unsafe { cu(self.handle, &mut mem) })?;
        Ok(mem)
    }

    /// Current GPU core temperature in °C.
    pub fn temperature(&self) -> Result<u32> {
        let n = nvml()?;
        let cu = n.nvml_device_get_temperature()?;
        let mut t: core::ffi::c_uint = 0;
        check(unsafe { cu(self.handle, baracuda_nvml_sys::NVML_TEMPERATURE_GPU, &mut t) })?;
        Ok(t)
    }

    /// Current instantaneous power draw in watts.
    pub fn power_usage_watts(&self) -> Result<f32> {
        let n = nvml()?;
        let cu = n.nvml_device_get_power_usage()?;
        let mut mw: core::ffi::c_uint = 0;
        check(unsafe { cu(self.handle, &mut mw) })?;
        Ok(mw as f32 / 1000.0)
    }

    /// Fan speed as a percentage (0–100). Returns
    /// `NVML_ERROR_NOT_SUPPORTED` on GPUs without user-visible fans
    /// (e.g. laptop MXM modules).
    pub fn fan_speed_percent(&self) -> Result<u32> {
        let n = nvml()?;
        let cu = n.nvml_device_get_fan_speed()?;
        let mut pct: core::ffi::c_uint = 0;
        check(unsafe { cu(self.handle, &mut pct) })?;
        Ok(pct)
    }

    /// GPU and memory utilization, each as an integer percent (0–100).
    pub fn utilization(&self) -> Result<Utilization> {
        let n = nvml()?;
        let cu = n.nvml_device_get_utilization_rates()?;
        let mut u = nvmlUtilization_t::default();
        check(unsafe { cu(self.handle, &mut u) })?;
        Ok(Utilization {
            gpu: u.gpu,
            memory: u.memory,
        })
    }

    /// Raw `nvmlDevice_t`. Use with care.
    #[inline]
    pub fn as_raw(&self) -> nvmlDevice_t {
        self.handle
    }

    // ---- clocks / power / perf ---------------------------------------

    /// Current clock (MHz) for the given engine.
    pub fn clock(&self, ty: ClockType) -> Result<u32> {
        let n = nvml()?;
        let cu = n.nvml_device_get_clock_info()?;
        let mut v = 0u32;
        check(unsafe { cu(self.handle, ty, &mut v) })?;
        Ok(v)
    }

    /// Maximum clock (MHz) for the given engine.
    pub fn max_clock(&self, ty: ClockType) -> Result<u32> {
        let n = nvml()?;
        let cu = n.nvml_device_get_max_clock_info()?;
        let mut v = 0u32;
        check(unsafe { cu(self.handle, ty, &mut v) })?;
        Ok(v)
    }

    /// Currently-configured applications clock (MHz).
    pub fn applications_clock(&self, ty: ClockType) -> Result<u32> {
        let n = nvml()?;
        let cu = n.nvml_device_get_applications_clock()?;
        let mut v = 0u32;
        check(unsafe { cu(self.handle, ty, &mut v) })?;
        Ok(v)
    }

    /// Default applications clock (MHz).
    pub fn default_applications_clock(&self, ty: ClockType) -> Result<u32> {
        let n = nvml()?;
        let cu = n.nvml_device_get_default_applications_clock()?;
        let mut v = 0u32;
        check(unsafe { cu(self.handle, ty, &mut v) })?;
        Ok(v)
    }

    /// Set both memory and graphics applications clocks (MHz).
    pub fn set_applications_clocks(&self, mem_mhz: u32, graphics_mhz: u32) -> Result<()> {
        let n = nvml()?;
        let cu = n.nvml_device_set_applications_clocks()?;
        check(unsafe { cu(self.handle, mem_mhz, graphics_mhz) })
    }

    /// Current power-management limit (mW).
    pub fn power_limit_mw(&self) -> Result<u32> {
        let n = nvml()?;
        let cu = n.nvml_device_get_power_management_limit()?;
        let mut v = 0u32;
        check(unsafe { cu(self.handle, &mut v) })?;
        Ok(v)
    }

    /// Min/max allowed power-management limit (mW).
    pub fn power_limit_range_mw(&self) -> Result<(u32, u32)> {
        let n = nvml()?;
        let cu = n.nvml_device_get_power_management_limit_constraints()?;
        let (mut lo, mut hi) = (0u32, 0u32);
        check(unsafe { cu(self.handle, &mut lo, &mut hi) })?;
        Ok((lo, hi))
    }

    /// Set the power-management limit (mW, within `power_limit_range_mw()`).
    pub fn set_power_limit_mw(&self, mw: u32) -> Result<()> {
        let n = nvml()?;
        let cu = n.nvml_device_set_power_management_limit()?;
        check(unsafe { cu(self.handle, mw) })
    }

    /// Current hardware power state (P-state).
    pub fn power_state(&self) -> Result<PState> {
        let n = nvml()?;
        let cu = n.nvml_device_get_power_state()?;
        let mut p = PState::Unknown;
        check(unsafe { cu(self.handle, &mut p) })?;
        Ok(p)
    }

    /// Current performance state (P-state) — identical content to
    /// [`Device::power_state`] on most drivers.
    pub fn performance_state(&self) -> Result<PState> {
        let n = nvml()?;
        let cu = n.nvml_device_get_performance_state()?;
        let mut p = PState::Unknown;
        check(unsafe { cu(self.handle, &mut p) })?;
        Ok(p)
    }

    /// Temperature threshold for a given threshold type (NVML's own
    /// `NVML_TEMPERATURE_THRESHOLD_*` integers).
    pub fn temperature_threshold(&self, threshold: u32) -> Result<u32> {
        let n = nvml()?;
        let cu = n.nvml_device_get_temperature_threshold()?;
        let mut v = 0u32;
        check(unsafe { cu(self.handle, threshold, &mut v) })?;
        Ok(v)
    }

    // ---- ECC ---------------------------------------------------------

    /// Corrected or uncorrected ECC count at a specific memory location.
    pub fn memory_error_count(
        &self,
        error: MemoryError,
        counter: EccCounter,
        location: MemoryLocation,
    ) -> Result<u64> {
        let n = nvml()?;
        let cu = n.nvml_device_get_memory_error_counter()?;
        let mut v = 0u64;
        check(unsafe { cu(self.handle, error, counter, location, &mut v) })?;
        Ok(v)
    }

    /// Sum of ECC errors of a given type across every location.
    pub fn total_ecc_errors(&self, error: MemoryError, counter: EccCounter) -> Result<u64> {
        let n = nvml()?;
        let cu = n.nvml_device_get_total_ecc_errors()?;
        let mut v = 0u64;
        check(unsafe { cu(self.handle, error, counter, &mut v) })?;
        Ok(v)
    }

    /// Returns `(current_enabled, pending_enabled)`.
    pub fn ecc_mode(&self) -> Result<(bool, bool)> {
        let n = nvml()?;
        let cu = n.nvml_device_get_ecc_mode()?;
        let (mut cur, mut pend) = (0u32, 0u32);
        check(unsafe { cu(self.handle, &mut cur, &mut pend) })?;
        Ok((cur != 0, pend != 0))
    }

    pub fn set_ecc_mode(&self, enable: bool) -> Result<()> {
        let n = nvml()?;
        let cu = n.nvml_device_set_ecc_mode()?;
        check(unsafe { cu(self.handle, enable as u32) })
    }

    // ---- PCIe / NVLink ----------------------------------------------

    pub fn pci_info(&self) -> Result<nvmlPciInfo_t> {
        let n = nvml()?;
        let cu = n.nvml_device_get_pci_info_v3()?;
        let mut pci = nvmlPciInfo_t::default();
        check(unsafe { cu(self.handle, &mut pci) })?;
        Ok(pci)
    }

    pub fn pcie_link_generation(&self) -> Result<u32> {
        let n = nvml()?;
        let cu = n.nvml_device_get_curr_pcie_link_generation()?;
        let mut v = 0u32;
        check(unsafe { cu(self.handle, &mut v) })?;
        Ok(v)
    }

    pub fn pcie_link_width(&self) -> Result<u32> {
        let n = nvml()?;
        let cu = n.nvml_device_get_curr_pcie_link_width()?;
        let mut v = 0u32;
        check(unsafe { cu(self.handle, &mut v) })?;
        Ok(v)
    }

    /// PCIe throughput for counter `n` (NVML `NVML_PCIE_UTIL_*` values).
    pub fn pcie_throughput(&self, counter: u32) -> Result<u32> {
        let n = nvml()?;
        let cu = n.nvml_device_get_pcie_throughput()?;
        let mut v = 0u32;
        check(unsafe { cu(self.handle, counter, &mut v) })?;
        Ok(v)
    }

    /// Whether a specific NVLink is currently active.
    pub fn nvlink_active(&self, link: u32) -> Result<bool> {
        let n = nvml()?;
        let cu = n.nvml_device_get_nvlink_state()?;
        let mut v = 0u32;
        check(unsafe { cu(self.handle, link, &mut v) })?;
        Ok(v != 0)
    }

    pub fn nvlink_version(&self, link: u32) -> Result<u32> {
        let n = nvml()?;
        let cu = n.nvml_device_get_nvlink_version()?;
        let mut v = 0u32;
        check(unsafe { cu(self.handle, link, &mut v) })?;
        Ok(v)
    }

    // ---- processes ----------------------------------------------------

    /// All processes holding a compute context on this GPU.
    pub fn compute_processes(&self) -> Result<Vec<nvmlProcessInfo_t>> {
        let n = nvml()?;
        let cu = n.nvml_device_get_compute_running_processes_v3()?;
        let mut count: u32 = 0;
        // First call with count=0 to learn size.
        let _ = unsafe { cu(self.handle, &mut count, core::ptr::null_mut()) };
        let mut buf = vec![nvmlProcessInfo_t::default(); count as usize];
        check(unsafe { cu(self.handle, &mut count, buf.as_mut_ptr()) })?;
        buf.truncate(count as usize);
        Ok(buf)
    }

    /// All processes with a graphics context on this GPU.
    pub fn graphics_processes(&self) -> Result<Vec<nvmlProcessInfo_t>> {
        let n = nvml()?;
        let cu = n.nvml_device_get_graphics_running_processes_v3()?;
        let mut count: u32 = 0;
        let _ = unsafe { cu(self.handle, &mut count, core::ptr::null_mut()) };
        let mut buf = vec![nvmlProcessInfo_t::default(); count as usize];
        check(unsafe { cu(self.handle, &mut count, buf.as_mut_ptr()) })?;
        buf.truncate(count as usize);
        Ok(buf)
    }

    // ---- compute mode --------------------------------------------------

    pub fn compute_mode(&self) -> Result<ComputeMode> {
        let n = nvml()?;
        let cu = n.nvml_device_get_compute_mode()?;
        let mut m = ComputeMode::Default;
        check(unsafe { cu(self.handle, &mut m) })?;
        Ok(m)
    }

    pub fn set_compute_mode(&self, mode: ComputeMode) -> Result<()> {
        let n = nvml()?;
        let cu = n.nvml_device_set_compute_mode()?;
        check(unsafe { cu(self.handle, mode) })
    }

    // ---- identity ------------------------------------------------------

    pub fn uuid(&self) -> Result<String> {
        let n = nvml()?;
        let cu = n.nvml_device_get_uuid()?;
        let mut buf = [0i8; 96];
        check(unsafe { cu(self.handle, buf.as_mut_ptr(), buf.len() as u32) })?;
        Ok(cstr_to_string(&buf))
    }

    pub fn serial(&self) -> Result<String> {
        let n = nvml()?;
        let cu = n.nvml_device_get_serial()?;
        let mut buf = [0i8; 64];
        check(unsafe { cu(self.handle, buf.as_mut_ptr(), buf.len() as u32) })?;
        Ok(cstr_to_string(&buf))
    }

    pub fn index(&self) -> Result<u32> {
        let n = nvml()?;
        let cu = n.nvml_device_get_index()?;
        let mut v = 0u32;
        check(unsafe { cu(self.handle, &mut v) })?;
        Ok(v)
    }

    pub fn minor_number(&self) -> Result<u32> {
        let n = nvml()?;
        let cu = n.nvml_device_get_minor_number()?;
        let mut v = 0u32;
        check(unsafe { cu(self.handle, &mut v) })?;
        Ok(v)
    }

    /// Populate the `value` field of each [`nvmlFieldValue_t`]. Values are
    /// valid only where `field.nvml_return == SUCCESS`.
    pub fn field_values(&self, values: &mut [nvmlFieldValue_t]) -> Result<()> {
        let n = nvml()?;
        let cu = n.nvml_device_get_field_values()?;
        check(unsafe {
            cu(
                self.handle,
                values.len() as core::ffi::c_int,
                values.as_mut_ptr(),
            )
        })
    }

    /// Register for events on this device. Use with [`EventSet`].
    pub fn register_events(&self, event_types: u64, set: &EventSet) -> Result<()> {
        let n = nvml()?;
        let cu = n.nvml_device_register_events()?;
        check(unsafe { cu(self.handle, event_types, set.raw) })
    }
}

/// Look up a device by its NVML UUID.
pub fn device_by_uuid(nvml_instance: &Nvml, uuid: &str) -> Result<Device> {
    let _ = nvml_instance;
    let n = nvml()?;
    let cu = n.nvml_device_get_handle_by_uuid()?;
    let c = std::ffi::CString::new(uuid).map_err(|_| Error::Status {
        status: nvmlReturn_t::INVALID_ARGUMENT,
    })?;
    let mut h: nvmlDevice_t = core::ptr::null_mut();
    check(unsafe { cu(c.as_ptr(), &mut h) })?;
    Ok(Device { handle: h })
}

/// Look up a device by its PCI bus ID string (e.g. `"00000000:01:00.0"`).
pub fn device_by_pci_bus_id(nvml_instance: &Nvml, pci_bus_id: &str) -> Result<Device> {
    let _ = nvml_instance;
    let n = nvml()?;
    let cu = n.nvml_device_get_handle_by_pci_bus_id_v2()?;
    let c = std::ffi::CString::new(pci_bus_id).map_err(|_| Error::Status {
        status: nvmlReturn_t::INVALID_ARGUMENT,
    })?;
    let mut h: nvmlDevice_t = core::ptr::null_mut();
    check(unsafe { cu(c.as_ptr(), &mut h) })?;
    Ok(Device { handle: h })
}

// ---- Event set ------------------------------------------------------

/// An NVML event set. Register devices on it with
/// [`Device::register_events`], then poll [`EventSet::wait`].
#[derive(Debug)]
pub struct EventSet {
    raw: nvmlEventSet_t,
}

impl EventSet {
    pub fn new() -> Result<Self> {
        let n = nvml()?;
        let cu = n.nvml_event_set_create()?;
        let mut raw: nvmlEventSet_t = core::ptr::null_mut();
        check(unsafe { cu(&mut raw) })?;
        Ok(Self { raw })
    }

    /// Wait up to `timeout_ms` for the next event. `timeout_ms == u32::MAX`
    /// waits forever.
    pub fn wait(&self, timeout_ms: u32) -> Result<nvmlEventData_t> {
        let n = nvml()?;
        let cu = n.nvml_event_set_wait_v2()?;
        let mut data = nvmlEventData_t::default();
        check(unsafe { cu(self.raw, &mut data, timeout_ms) })?;
        Ok(data)
    }

    pub fn as_raw(&self) -> nvmlEventSet_t {
        self.raw
    }
}

impl Drop for EventSet {
    fn drop(&mut self) {
        if let Ok(n) = nvml() {
            if let Ok(cu) = n.nvml_event_set_free() {
                let _ = unsafe { cu(self.raw) };
            }
        }
    }
}

/// GPU and memory utilization, 0–100 integer percent.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Utilization {
    pub gpu: u32,
    pub memory: u32,
}

fn cstr_to_string(buf: &[c_char]) -> String {
    let as_u8: &[u8] = unsafe { core::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len()) };
    let end = as_u8.iter().position(|&b| b == 0).unwrap_or(as_u8.len());
    String::from_utf8_lossy(&as_u8[..end]).into_owned()
}

fn clone_error(e: &Error) -> Error {
    match e {
        Error::Status { status } => Error::Status { status: *status },
        Error::Loader(l) => Error::Loader(clone_loader(l)),
        Error::FeatureNotSupported { api, since } => {
            Error::FeatureNotSupported { api, since: *since }
        }
    }
}

fn clone_loader(l: &baracuda_core::LoaderError) -> baracuda_core::LoaderError {
    use baracuda_core::LoaderError as L;
    match l {
        L::LibraryNotFound {
            library,
            candidates,
            search_paths,
        } => L::LibraryNotFound {
            library,
            candidates: candidates.clone(),
            search_paths: *search_paths,
        },
        L::SymbolNotFound { library, symbol } => L::SymbolNotFound { library, symbol },
        L::VersionTooOld {
            symbol,
            required,
            installed,
        } => L::VersionTooOld {
            symbol,
            required: *required,
            installed: *installed,
        },
        L::UnsupportedPlatform { platform } => L::UnsupportedPlatform { platform },
        L::Libloading(_) => L::SymbolNotFound {
            library: "nvml",
            symbol: "(libloading error; see first error)",
        },
    }
}
