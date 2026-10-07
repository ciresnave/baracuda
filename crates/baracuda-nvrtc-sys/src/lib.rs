//! Raw FFI + dynamic loader for NVIDIA NVRTC (runtime CUDA C++→PTX compiler).
//!
//! `baracuda-nvrtc` wraps this with a safe, typed API. Use this crate
//! directly only if you need a function that the safe layer hasn't
//! wrapped yet (in which case please file a bug).

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
    /// success
    pub const SUCCESS: Self = Self(0);
    /// out of memory
    pub const OUT_OF_MEMORY: Self = Self(1);
    /// program creation failed
    pub const PROGRAM_CREATION_FAILURE: Self = Self(2);
    /// invalid input pointer or length
    pub const INVALID_INPUT: Self = Self(3);
    /// invalid `nvrtcProgram` handle
    pub const INVALID_PROGRAM: Self = Self(4);
    /// unrecognized or malformed compile option
    pub const INVALID_OPTION: Self = Self(5);
    /// compilation failed — fetch the program log
    pub const COMPILATION: Self = Self(6);
    /// an NVRTC builtin operation failed
    pub const BUILTIN_OPERATION_FAILURE: Self = Self(7);
    /// name expressions added after compilation
    pub const NO_NAME_EXPRESSIONS_AFTER_COMPILATION: Self = Self(8);
    /// lowered-name lookup attempted before compilation
    pub const NO_LOWERED_NAMES_BEFORE_COMPILATION: Self = Self(9);
    /// name expression is not valid
    pub const NAME_EXPRESSION_NOT_VALID: Self = Self(10);
    /// internal NVRTC error
    pub const INTERNAL_ERROR: Self = Self(11);

    /// returns true when the result code is `NVRTC_SUCCESS`
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

/// function pointer for `nvrtcVersion`
pub type PFN_nvrtcVersion =
    unsafe extern "C" fn(major: *mut core::ffi::c_int, minor: *mut core::ffi::c_int) -> nvrtcResult;
/// function pointer for `nvrtcCreateProgram`
pub type PFN_nvrtcCreateProgram = unsafe extern "C" fn(
    prog: *mut nvrtcProgram,
    src: *const c_char,
    name: *const c_char,
    num_headers: core::ffi::c_int,
    headers: *const *const c_char,
    include_names: *const *const c_char,
) -> nvrtcResult;
/// function pointer for `nvrtcDestroyProgram`
pub type PFN_nvrtcDestroyProgram = unsafe extern "C" fn(prog: *mut nvrtcProgram) -> nvrtcResult;
/// function pointer for `nvrtcCompileProgram`
pub type PFN_nvrtcCompileProgram = unsafe extern "C" fn(
    prog: nvrtcProgram,
    num_options: core::ffi::c_int,
    options: *const *const c_char,
) -> nvrtcResult;
/// function pointer for `nvrtcGetPTXSize`
pub type PFN_nvrtcGetPTXSize =
    unsafe extern "C" fn(prog: nvrtcProgram, size: *mut usize) -> nvrtcResult;
/// function pointer for `nvrtcGetPTX`
pub type PFN_nvrtcGetPTX =
    unsafe extern "C" fn(prog: nvrtcProgram, ptx: *mut c_char) -> nvrtcResult;
/// function pointer for `nvrtcGetProgramLogSize`
pub type PFN_nvrtcGetProgramLogSize =
    unsafe extern "C" fn(prog: nvrtcProgram, size: *mut usize) -> nvrtcResult;
/// function pointer for `nvrtcGetProgramLog`
pub type PFN_nvrtcGetProgramLog =
    unsafe extern "C" fn(prog: nvrtcProgram, log: *mut c_char) -> nvrtcResult;
/// function pointer for `nvrtcGetErrorString`
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
        /// Dynamic loader handle for NVRTC.
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
                #[doc = concat!("Resolve `", $sym, "`.")]
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

/// Env var that, if set, names the exact NVRTC library path to load
/// (checked before the hardcoded candidate list). Useful when more than one CUDA
/// toolkit is installed and the one on `PATH`/`CUDA_PATH` isn't the one
/// whose NVRTC should be used (e.g. building for sm_61, which needs a
/// pre-CUDA-13 toolkit's `nvrtc64_*.dll` even on a box whose default
/// toolkit is 13.x). An override that's set but fails to load is reported
/// as [`LoaderError::EnvOverrideUnusable`], not silently ignored.
pub const BARACUDA_NVRTC_PATH_ENV: &str = "BARACUDA_NVRTC_PATH";

/// Open (or return the cached) NVRTC dynamic library.
pub fn nvrtc() -> Result<&'static Nvrtc, LoaderError> {
    static NVRTC: OnceLock<Nvrtc> = OnceLock::new();
    if let Some(n) = NVRTC.get() {
        return Ok(n);
    }
    let lib =
        Library::open_with_env_override("nvrtc", BARACUDA_NVRTC_PATH_ENV, nvrtc_candidates())?;
    let n = Nvrtc::empty(lib);
    let _ = NVRTC.set(n);
    Ok(NVRTC.get().expect("OnceLock set or lost race"))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Both scenarios below mutate the process-global `BARACUDA_NVRTC_PATH`
    // env var, which races against any other test doing the same in the
    // same process -- kept as one sequential test (not two `#[test]`s) so
    // Rust's default parallel test execution can't interleave them.
    #[test]
    fn env_override_wiring() {
        // Exercises the same `Library::open_with_env_override` call
        // `nvrtc()` makes, without going through `nvrtc()` itself --
        // `nvrtc()` caches its result process-wide in a `OnceLock`, so a
        // second scenario in the same test binary would just see the first
        // call's cached outcome.
        unsafe {
            std::env::set_var(
                BARACUDA_NVRTC_PATH_ENV,
                "/definitely/does/not/exist/nvrtc.bogus",
            );
        }
        let bogus =
            Library::open_with_env_override("nvrtc", BARACUDA_NVRTC_PATH_ENV, nvrtc_candidates());
        match bogus {
            Err(LoaderError::EnvOverrideUnusable {
                library, env_var, ..
            }) => {
                assert_eq!(library, "nvrtc");
                assert_eq!(env_var, BARACUDA_NVRTC_PATH_ENV);
            }
            other => panic!(
                "expected EnvOverrideUnusable (the override must short-circuit, never fall back to the real candidates), got {other:?}"
            ),
        }

        unsafe {
            std::env::remove_var(BARACUDA_NVRTC_PATH_ENV);
        }
        // Can't assert success here (no CUDA guaranteed on every CI
        // runner), only that an unset override does not itself produce
        // `EnvOverrideUnusable` -- that variant must only ever come from a
        // *set* override.
        let unset =
            Library::open_with_env_override("nvrtc", BARACUDA_NVRTC_PATH_ENV, nvrtc_candidates());
        if let Err(LoaderError::EnvOverrideUnusable { .. }) = unset {
            panic!("an unset override must never produce EnvOverrideUnusable");
        }
    }
}
