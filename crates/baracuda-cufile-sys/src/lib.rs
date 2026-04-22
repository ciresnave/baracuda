//! Raw FFI + dynamic loader for NVIDIA cuFile (GPUDirect Storage).
//!
//! Linux-only: cuFile calls `ibverbs` + NVIDIA's GDS kernel driver and has
//! no Windows/macOS analogue. On non-Linux platforms the loader returns
//! [`LoaderError::UnsupportedPlatform`].

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
#![warn(missing_debug_implementations)]

use core::ffi::{c_char, c_int, c_uint, c_void};
use std::sync::OnceLock;

use baracuda_core::{Library, LoaderError};
use baracuda_types::CudaStatus;

/// cuFile operation status (a.k.a. `CUfileOpError`).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct CUfileOpError(pub i32);

impl CUfileOpError {
    pub const SUCCESS: Self = Self(0);
    pub const INTERNAL: Self = Self(5001);
    pub const DRIVER_NOT_INITIALIZED: Self = Self(5002);
    pub const IO_NOT_SUPPORTED: Self = Self(5003);
    pub const NOT_REGISTERED: Self = Self(5004);
    pub const INVALID_FILE_HANDLE: Self = Self(5005);

    pub const fn is_success(self) -> bool {
        self.0 == 0
    }
}

impl CudaStatus for CUfileOpError {
    fn code(self) -> i32 {
        self.0
    }
    fn name(self) -> &'static str {
        match self.0 {
            0 => "CU_FILE_SUCCESS",
            5001 => "CU_FILE_INTERNAL_ERROR",
            5002 => "CU_FILE_DRIVER_NOT_INITIALIZED",
            5003 => "CU_FILE_IO_NOT_SUPPORTED",
            5004 => "CU_FILE_NOT_REGISTERED",
            5005 => "CU_FILE_INVALID_FILE_HANDLE",
            _ => "CU_FILE_UNRECOGNIZED",
        }
    }
    fn description(self) -> &'static str {
        match self.0 {
            0 => "success",
            5002 => "cuFile driver not initialized — call cuFileDriverOpen first",
            5003 => "I/O not supported on this file / filesystem",
            _ => "unrecognized cuFile status code",
        }
    }
    fn is_success(self) -> bool {
        CUfileOpError::is_success(self)
    }
    fn library(self) -> &'static str {
        "cufile"
    }
}

/// `CUfileError_t` — status plus a CUDA error code (for I/O that
/// dispatched through the GPU).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct CUfileError_t {
    pub err: CUfileOpError,
    pub cu_err: c_int, // CUresult (driver-API error)
}

/// `CUfileHandle_t` — opaque handle returned by `cuFileHandleRegister`.
pub type CUfileHandle_t = *mut c_void;

/// `CUfileDescr_t` — struct describing a file (Unix fd or Win32 handle)
/// for registration. cuFile's own layout is an outer tagged union; we
/// model only the Unix-fd path since this API is Linux-only anyway.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct CUfileDescr_t {
    /// Handle-type selector: 1 = CU_FILE_HANDLE_TYPE_OPAQUE_FD.
    pub handle_type: c_int,
    pub handle_fd: c_int,
    /// 40-byte reserved tail; cuFile's real union is larger for Win32 +
    /// opaque handles but Linux-only builds don't touch it.
    pub _reserved: [u8; 40],
    pub fs_ops: *mut c_void,
}

impl Default for CUfileDescr_t {
    fn default() -> Self {
        Self {
            handle_type: 1, // OPAQUE_FD
            handle_fd: -1,
            _reserved: [0; 40],
            fs_ops: core::ptr::null_mut(),
        }
    }
}

// ---- PFN types ----

pub type PFN_cuFileDriverOpen = unsafe extern "C" fn() -> CUfileError_t;
pub type PFN_cuFileDriverClose = unsafe extern "C" fn() -> CUfileError_t;
pub type PFN_cuFileDriverGetProperties = unsafe extern "C" fn(props: *mut c_void) -> CUfileError_t;
pub type PFN_cuFileDriverSetPollMode =
    unsafe extern "C" fn(poll: bool, poll_threshold_size: usize) -> CUfileError_t;

pub type PFN_cuFileHandleRegister =
    unsafe extern "C" fn(fh: *mut CUfileHandle_t, descr: *mut CUfileDescr_t) -> CUfileError_t;
pub type PFN_cuFileHandleDeregister = unsafe extern "C" fn(fh: CUfileHandle_t);

pub type PFN_cuFileBufRegister =
    unsafe extern "C" fn(buf_ptr: *mut c_void, length: usize, flags: c_int) -> CUfileError_t;
pub type PFN_cuFileBufDeregister = unsafe extern "C" fn(buf_ptr: *mut c_void) -> CUfileError_t;

pub type PFN_cuFileRead = unsafe extern "C" fn(
    fh: CUfileHandle_t,
    buf_ptr: *mut c_void,
    size: usize,
    file_offset: i64,
    buf_ptr_offset: i64,
) -> isize;
pub type PFN_cuFileWrite = unsafe extern "C" fn(
    fh: CUfileHandle_t,
    buf_ptr: *const c_void,
    size: usize,
    file_offset: i64,
    buf_ptr_offset: i64,
) -> isize;

pub type PFN_cuFileGetVersion = unsafe extern "C" fn(version: *mut c_int) -> CUfileError_t;

pub type PFN_cuFileOpStatusError = unsafe extern "C" fn(status: CUfileOpError) -> *const c_char;

// ---- Stream-registered (v1.6+) async APIs ----

pub type PFN_cuFileReadAsync = unsafe extern "C" fn(
    fh: CUfileHandle_t,
    buf_ptr: *mut c_void,
    size_p: *mut usize,
    file_offset_p: *mut i64,
    buf_ptr_offset_p: *mut i64,
    bytes_read: *mut isize,
    stream: *mut c_void,
) -> CUfileError_t;

pub type PFN_cuFileWriteAsync = unsafe extern "C" fn(
    fh: CUfileHandle_t,
    buf_ptr: *const c_void,
    size_p: *mut usize,
    file_offset_p: *mut i64,
    buf_ptr_offset_p: *mut i64,
    bytes_written: *mut isize,
    stream: *mut c_void,
) -> CUfileError_t;

pub type PFN_cuFileStreamRegister =
    unsafe extern "C" fn(stream: *mut c_void, flags: c_uint) -> CUfileError_t;

pub type PFN_cuFileStreamDeregister = unsafe extern "C" fn(stream: *mut c_void) -> CUfileError_t;

// ---- Batched I/O (v1.6+) ----

/// Opcode selector for `CUfileIOParams_t`.
#[allow(non_snake_case)]
pub mod CUfileOpcode {
    pub const READ: i32 = 0;
    pub const WRITE: i32 = 1;
}

/// `CUfileIOParams_t` — 1 entry in a batched-IO request.
///
/// The real C struct is a tagged union — we expose the most-common
/// "opaque FD + device memory" flavor via `fh`, `dev_ptr`, `file_offset`,
/// `dev_ptr_offset`, `size`, and `opcode`. `mode` selector is always 0
/// (BATCH, the only shipped flavor today).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct CUfileIOParams_t {
    pub mode: c_int, // always 0 == CUFILE_BATCH
    pub fh: CUfileHandle_t,
    pub opcode: c_int,
    pub cookie: *mut c_void,
    pub dev_ptr_base: *mut c_void,
    pub file_offset: i64,
    pub dev_ptr_offset: i64,
    pub size: usize,
}

impl Default for CUfileIOParams_t {
    fn default() -> Self {
        Self {
            mode: 0,
            fh: core::ptr::null_mut(),
            opcode: 0,
            cookie: core::ptr::null_mut(),
            dev_ptr_base: core::ptr::null_mut(),
            file_offset: 0,
            dev_ptr_offset: 0,
            size: 0,
        }
    }
}

/// `CUfileIOEvents_t` — status for a single batched-I/O entry.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CUfileIOEvents_t {
    pub cookie: *mut c_void,
    pub status: c_int,
    pub ret: usize,
}

unsafe impl Send for CUfileIOEvents_t {}

/// Opaque batch-handle (maps to `CUfileBatchHandle_t`).
pub type CUfileBatchHandle_t = *mut c_void;

pub type PFN_cuFileBatchIOSetUp = unsafe extern "C" fn(
    batch_handle_out: *mut CUfileBatchHandle_t,
    num_batches: c_uint,
) -> CUfileError_t;

pub type PFN_cuFileBatchIOSubmit = unsafe extern "C" fn(
    batch_handle: CUfileBatchHandle_t,
    num_entries: c_uint,
    io_batch_params: *mut CUfileIOParams_t,
    flags: c_uint,
) -> CUfileError_t;

pub type PFN_cuFileBatchIOGetStatus = unsafe extern "C" fn(
    batch_handle: CUfileBatchHandle_t,
    min_nr: c_uint,
    nr: *mut c_uint,
    io_batch_events: *mut CUfileIOEvents_t,
    timeout: *mut c_void, // struct timespec* — opaque on Linux
) -> CUfileError_t;

pub type PFN_cuFileBatchIOCancel =
    unsafe extern "C" fn(batch_handle: CUfileBatchHandle_t) -> CUfileError_t;

pub type PFN_cuFileBatchIODestroy =
    unsafe extern "C" fn(batch_handle: CUfileBatchHandle_t) -> CUfileError_t;

// ---- Compatibility-mode helpers ----

pub type PFN_cuFileUseCount = unsafe extern "C" fn(fh: CUfileHandle_t) -> c_int;

// ---- Driver-tuning setters (v1.6+) ----

pub type PFN_cuFileDriverSetMaxDirectIOSize =
    unsafe extern "C" fn(max_direct_io_size_kb: usize) -> CUfileError_t;

pub type PFN_cuFileDriverSetMaxCacheSize =
    unsafe extern "C" fn(max_cache_size_kb: usize) -> CUfileError_t;

pub type PFN_cuFileDriverSetMaxPinnedMemSize =
    unsafe extern "C" fn(max_pinned_size_kb: usize) -> CUfileError_t;

// ---- Loader ----

macro_rules! cufile_fns {
    ($($(#[$attr:meta])* fn $name:ident as $sym:literal : $pfn:ty;)*) => {
        pub struct Cufile {
            pub lib: Library,
            $(
                $name: OnceLock<$pfn>,
            )*
        }

        impl core::fmt::Debug for Cufile {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct("Cufile").field("lib", &self.lib).finish_non_exhaustive()
            }
        }

        impl Cufile {
            #[allow(dead_code)]
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

cufile_fns! {
    fn cu_file_driver_open as "cuFileDriverOpen": PFN_cuFileDriverOpen;
    fn cu_file_driver_close as "cuFileDriverClose": PFN_cuFileDriverClose;
    fn cu_file_driver_get_properties as "cuFileDriverGetProperties":
        PFN_cuFileDriverGetProperties;
    fn cu_file_driver_set_poll_mode as "cuFileDriverSetPollMode":
        PFN_cuFileDriverSetPollMode;
    fn cu_file_handle_register as "cuFileHandleRegister": PFN_cuFileHandleRegister;
    fn cu_file_handle_deregister as "cuFileHandleDeregister": PFN_cuFileHandleDeregister;
    fn cu_file_buf_register as "cuFileBufRegister": PFN_cuFileBufRegister;
    fn cu_file_buf_deregister as "cuFileBufDeregister": PFN_cuFileBufDeregister;
    fn cu_file_read as "cuFileRead": PFN_cuFileRead;
    fn cu_file_write as "cuFileWrite": PFN_cuFileWrite;
    fn cu_file_get_version as "cuFileGetVersion": PFN_cuFileGetVersion;
    fn cu_file_op_status_error as "cuFileGetOpStatusErrorString":
        PFN_cuFileOpStatusError;

    // Async I/O (stream-registered, v1.6+)
    fn cu_file_read_async as "cuFileReadAsync": PFN_cuFileReadAsync;
    fn cu_file_write_async as "cuFileWriteAsync": PFN_cuFileWriteAsync;
    fn cu_file_stream_register as "cuFileStreamRegister": PFN_cuFileStreamRegister;
    fn cu_file_stream_deregister as "cuFileStreamDeregister": PFN_cuFileStreamDeregister;

    // Batched I/O (v1.6+)
    fn cu_file_batch_io_set_up as "cuFileBatchIOSetUp": PFN_cuFileBatchIOSetUp;
    fn cu_file_batch_io_submit as "cuFileBatchIOSubmit": PFN_cuFileBatchIOSubmit;
    fn cu_file_batch_io_get_status as "cuFileBatchIOGetStatus": PFN_cuFileBatchIOGetStatus;
    fn cu_file_batch_io_cancel as "cuFileBatchIOCancel": PFN_cuFileBatchIOCancel;
    fn cu_file_batch_io_destroy as "cuFileBatchIODestroy": PFN_cuFileBatchIODestroy;

    // Compat / diagnostic
    fn cu_file_use_count as "cuFileUseCount": PFN_cuFileUseCount;

    // Driver tuning
    fn cu_file_driver_set_max_direct_io_size as "cuFileDriverSetMaxDirectIOSize":
        PFN_cuFileDriverSetMaxDirectIOSize;
    fn cu_file_driver_set_max_cache_size as "cuFileDriverSetMaxCacheSize":
        PFN_cuFileDriverSetMaxCacheSize;
    fn cu_file_driver_set_max_pinned_mem_size as "cuFileDriverSetMaxPinnedMemSize":
        PFN_cuFileDriverSetMaxPinnedMemSize;
}

#[cfg(target_os = "linux")]
fn cufile_candidates() -> &'static [&'static str] {
    &["libcufile.so.0", "libcufile.so"]
}

pub fn cufile() -> Result<&'static Cufile, LoaderError> {
    static CUFILE: OnceLock<Cufile> = OnceLock::new();
    if let Some(c) = CUFILE.get() {
        return Ok(c);
    }
    #[cfg(not(target_os = "linux"))]
    {
        Err(LoaderError::UnsupportedPlatform {
            platform: std::env::consts::OS,
        })
    }
    #[cfg(target_os = "linux")]
    {
        let lib = Library::open("cufile", cufile_candidates())?;
        let _ = CUFILE.set(Cufile::empty(lib));
        Ok(CUFILE.get().expect("OnceLock set or lost race"))
    }
}
