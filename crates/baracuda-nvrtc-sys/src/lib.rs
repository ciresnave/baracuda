//! Raw FFI + dynamic loader for NVIDIA NVRTC (runtime CUDA C++→PTX compiler).

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
#![warn(missing_debug_implementations)]

use core::ffi::{c_char, c_void};
use std::sync::OnceLock;

use baracuda_core::{Library, LoaderError};
use baracuda_types::CudaStatus;

/// Opaque NVRTC program handle.
pub type nvrtcProgram = *mut c_void;

/// Return code from an NVRTC call.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct nvrtcResult(pub i32);

impl nvrtcResult {
    pub const SUCCESS: Self = Self(0);
    pub const OUT_OF_MEMORY: Self = Self(1);
    pub const PROGRAM_CREATION_FAILURE: Self = Self(2);
    pub const INVALID_INPUT: Self = Self(3);
    pub const INVALID_PROGRAM: Self = Self(4);
    pub const INVALID_OPTION: Self = Self(5);
    pub const COMPILATION: Self = Self(6);
    pub const BUILTIN_OPERATION_FAILURE: Self = Self(7);
    pub const NO_NAME_EXPRESSIONS_AFTER_COMPILATION: Self = Self(8);
    pub const NO_LOWERED_NAMES_BEFORE_COMPILATION: Self = Self(9);
    pub const NAME_EXPRESSION_NOT_VALID: Self = Self(10);
    pub const INTERNAL_ERROR: Self = Self(11);

    pub const fn is_success(self) -> bool {
        self.0 == 0
    }
}

impl CudaStatus for nvrtcResult {
    fn code(self) -> i32 {
        self.0
    }
    fn name(self) -> &'static str {
        match self.0 {
            0 => "NVRTC_SUCCESS",
            1 => "NVRTC_ERROR_OUT_OF_MEMORY",
            2 => "NVRTC_ERROR_PROGRAM_CREATION_FAILURE",
            3 => "NVRTC_ERROR_INVALID_INPUT",
            4 => "NVRTC_ERROR_INVALID_PROGRAM",
            5 => "NVRTC_ERROR_INVALID_OPTION",
            6 => "NVRTC_ERROR_COMPILATION",
            7 => "NVRTC_ERROR_BUILTIN_OPERATION_FAILURE",
            11 => "NVRTC_ERROR_INTERNAL_ERROR",
            _ => "NVRTC_ERROR_UNRECOGNIZED",
        }
    }
    fn description(self) -> &'static str {
        match self.0 {
            0 => "success",
            1 => "out of memory",
            2 => "program creation failure",
            3 => "invalid input",
            6 => "compilation failed (fetch program log for details)",
            _ => "unrecognized NVRTC status code",
        }
    }
    fn is_success(self) -> bool {
        nvrtcResult::is_success(self)
    }
    fn library(self) -> &'static str {
        "nvrtc"
    }
}

// ---- function-pointer types ----------------------------------------------

pub type PFN_nvrtcVersion =
    unsafe extern "C" fn(major: *mut core::ffi::c_int, minor: *mut core::ffi::c_int) -> nvrtcResult;
pub type PFN_nvrtcCreateProgram = unsafe extern "C" fn(
    prog: *mut nvrtcProgram,
    src: *const c_char,
    name: *const c_char,
    num_headers: core::ffi::c_int,
    headers: *const *const c_char,
    include_names: *const *const c_char,
) -> nvrtcResult;
pub type PFN_nvrtcDestroyProgram = unsafe extern "C" fn(prog: *mut nvrtcProgram) -> nvrtcResult;
pub type PFN_nvrtcCompileProgram = unsafe extern "C" fn(
    prog: nvrtcProgram,
    num_options: core::ffi::c_int,
    options: *const *const c_char,
) -> nvrtcResult;
pub type PFN_nvrtcGetPTXSize =
    unsafe extern "C" fn(prog: nvrtcProgram, size: *mut usize) -> nvrtcResult;
pub type PFN_nvrtcGetPTX =
    unsafe extern "C" fn(prog: nvrtcProgram, ptx: *mut c_char) -> nvrtcResult;
pub type PFN_nvrtcGetProgramLogSize =
    unsafe extern "C" fn(prog: nvrtcProgram, size: *mut usize) -> nvrtcResult;
pub type PFN_nvrtcGetProgramLog =
    unsafe extern "C" fn(prog: nvrtcProgram, log: *mut c_char) -> nvrtcResult;
pub type PFN_nvrtcGetErrorString = unsafe extern "C" fn(result: nvrtcResult) -> *const c_char;

// ---- loader --------------------------------------------------------------

fn nvrtc_candidates() -> &'static [&'static str] {
    #[cfg(target_os = "linux")]
    {
        &[
            "libnvrtc.so.13",
            "libnvrtc.so.12",
            "libnvrtc.so.11.2",
            "libnvrtc.so",
        ]
    }
    #[cfg(target_os = "windows")]
    {
        // NVRTC DLL naming: nvrtc64_<major><minor>_0.dll (e.g. nvrtc64_130_0.dll).
        &[
            "nvrtc64_130_0.dll",
            "nvrtc64_128_0.dll",
            "nvrtc64_126_0.dll",
            "nvrtc64_123_0.dll",
            "nvrtc64_120_0.dll",
            "nvrtc64_112_0.dll",
        ]
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        &[]
    }
}

macro_rules! nvrtc_fns {
    ($($name:ident as $sym:literal : $pfn:ty);* $(;)?) => {
        pub struct Nvrtc {
            lib: Library,
            $($name: OnceLock<$pfn>,)*
        }
        impl core::fmt::Debug for Nvrtc {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct("Nvrtc").field("lib", &self.lib).finish_non_exhaustive()
            }
        }
        impl Nvrtc {
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

nvrtc_fns! {
    nvrtc_version as "nvrtcVersion": PFN_nvrtcVersion;
    nvrtc_create_program as "nvrtcCreateProgram": PFN_nvrtcCreateProgram;
    nvrtc_destroy_program as "nvrtcDestroyProgram": PFN_nvrtcDestroyProgram;
    nvrtc_compile_program as "nvrtcCompileProgram": PFN_nvrtcCompileProgram;
    nvrtc_get_ptx_size as "nvrtcGetPTXSize": PFN_nvrtcGetPTXSize;
    nvrtc_get_ptx as "nvrtcGetPTX": PFN_nvrtcGetPTX;
    nvrtc_get_program_log_size as "nvrtcGetProgramLogSize": PFN_nvrtcGetProgramLogSize;
    nvrtc_get_program_log as "nvrtcGetProgramLog": PFN_nvrtcGetProgramLog;
    nvrtc_get_error_string as "nvrtcGetErrorString": PFN_nvrtcGetErrorString;
}

pub fn nvrtc() -> Result<&'static Nvrtc, LoaderError> {
    static NVRTC: OnceLock<Nvrtc> = OnceLock::new();
    if let Some(n) = NVRTC.get() {
        return Ok(n);
    }
    let lib = Library::open("nvrtc", nvrtc_candidates())?;
    let n = Nvrtc::empty(lib);
    let _ = NVRTC.set(n);
    Ok(NVRTC.get().expect("OnceLock set or lost race"))
}
