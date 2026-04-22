//! Raw FFI + dynamic loader for NVIDIA nvJitLink (CUDA 12.0+).

#![allow(non_camel_case_types)]
#![warn(missing_debug_implementations)]

use core::ffi::{c_char, c_int, c_uint, c_void};
use std::sync::OnceLock;

use baracuda_core::{Library, LoaderError};
use baracuda_types::CudaStatus;

/// Opaque nvJitLink handle.
pub type nvJitLinkHandle = *mut c_void;

/// Input blob type for `nvJitLinkAddData`.
#[repr(u32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum nvJitLinkInputType {
    None = 0,
    Cubin = 1,
    Ptx = 2,
    LtoIr = 3,
    Fatbin = 4,
    Object = 5,
    Library = 6,
}

/// Return code from an nvJitLink call.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct nvJitLinkResult(pub i32);

impl nvJitLinkResult {
    pub const SUCCESS: Self = Self(0);
    pub const ERROR_UNRECOGNIZED_OPTION: Self = Self(1);
    pub const ERROR_MISSING_ARCH: Self = Self(2);
    pub const ERROR_INVALID_INPUT: Self = Self(3);
    pub const ERROR_PTX_COMPILE: Self = Self(4);
    pub const ERROR_NVVM_COMPILE: Self = Self(5);
    pub const ERROR_INTERNAL: Self = Self(6);
    pub const ERROR_THREADPOOL: Self = Self(7);
    pub const ERROR_UNRECOGNIZED_INPUT: Self = Self(8);
    pub const ERROR_FINALIZE: Self = Self(9);

    pub const fn is_success(self) -> bool {
        self.0 == 0
    }
}

impl CudaStatus for nvJitLinkResult {
    fn code(self) -> i32 {
        self.0
    }
    fn name(self) -> &'static str {
        match self.0 {
            0 => "NVJITLINK_SUCCESS",
            1 => "NVJITLINK_ERROR_UNRECOGNIZED_OPTION",
            2 => "NVJITLINK_ERROR_MISSING_ARCH",
            3 => "NVJITLINK_ERROR_INVALID_INPUT",
            4 => "NVJITLINK_ERROR_PTX_COMPILE",
            5 => "NVJITLINK_ERROR_NVVM_COMPILE",
            6 => "NVJITLINK_ERROR_INTERNAL",
            7 => "NVJITLINK_ERROR_THREADPOOL",
            8 => "NVJITLINK_ERROR_UNRECOGNIZED_INPUT",
            9 => "NVJITLINK_ERROR_FINALIZE",
            _ => "NVJITLINK_ERROR_UNRECOGNIZED",
        }
    }
    fn description(self) -> &'static str {
        match self.0 {
            0 => "success",
            1 => "unrecognized linker option",
            2 => "missing required -arch= option",
            3 => "invalid input blob",
            4 => "PTX compilation failed",
            5 => "NVVM compilation failed",
            6 => "internal nvJitLink error",
            7 => "thread-pool failure",
            8 => "unrecognized input type",
            9 => "finalization failed",
            _ => "unrecognized nvJitLink status code",
        }
    }
    fn is_success(self) -> bool {
        nvJitLinkResult::is_success(self)
    }
    fn library(self) -> &'static str {
        "nvjitlink"
    }
}

// ---- function-pointer types ----

pub type PFN_nvJitLinkCreate = unsafe extern "C" fn(
    handle: *mut nvJitLinkHandle,
    num_options: u32,
    options: *const *const c_char,
) -> nvJitLinkResult;
/// `nvJitLinkDestroy` — note the C API takes `nvJitLinkHandle*` (a pointer
/// to the handle slot, which it zeroes out), not the handle by value.
pub type PFN_nvJitLinkDestroy =
    unsafe extern "C" fn(handle: *mut nvJitLinkHandle) -> nvJitLinkResult;
pub type PFN_nvJitLinkAddData = unsafe extern "C" fn(
    handle: nvJitLinkHandle,
    input_type: nvJitLinkInputType,
    data: *const c_void,
    size: usize,
    name: *const c_char,
) -> nvJitLinkResult;
pub type PFN_nvJitLinkAddFile = unsafe extern "C" fn(
    handle: nvJitLinkHandle,
    input_type: nvJitLinkInputType,
    file_name: *const c_char,
) -> nvJitLinkResult;
pub type PFN_nvJitLinkComplete = unsafe extern "C" fn(handle: nvJitLinkHandle) -> nvJitLinkResult;
pub type PFN_nvJitLinkGetLinkedCubinSize =
    unsafe extern "C" fn(handle: nvJitLinkHandle, size: *mut usize) -> nvJitLinkResult;
pub type PFN_nvJitLinkGetLinkedCubin =
    unsafe extern "C" fn(handle: nvJitLinkHandle, cubin: *mut c_void) -> nvJitLinkResult;
pub type PFN_nvJitLinkGetLinkedPtxSize =
    unsafe extern "C" fn(handle: nvJitLinkHandle, size: *mut usize) -> nvJitLinkResult;
pub type PFN_nvJitLinkGetLinkedPtx =
    unsafe extern "C" fn(handle: nvJitLinkHandle, ptx: *mut c_char) -> nvJitLinkResult;
pub type PFN_nvJitLinkGetErrorLogSize =
    unsafe extern "C" fn(handle: nvJitLinkHandle, size: *mut usize) -> nvJitLinkResult;
pub type PFN_nvJitLinkGetErrorLog =
    unsafe extern "C" fn(handle: nvJitLinkHandle, log: *mut c_char) -> nvJitLinkResult;
pub type PFN_nvJitLinkGetInfoLogSize =
    unsafe extern "C" fn(handle: nvJitLinkHandle, size: *mut usize) -> nvJitLinkResult;
pub type PFN_nvJitLinkGetInfoLog =
    unsafe extern "C" fn(handle: nvJitLinkHandle, log: *mut c_char) -> nvJitLinkResult;
pub type PFN_nvJitLinkVersion =
    unsafe extern "C" fn(major: *mut c_uint, minor: *mut c_uint) -> nvJitLinkResult;

// Silence unused-import warnings for c_int.
#[allow(dead_code)]
type _CInt = c_int;

// ---- loader ----

const fn nvjitlink_candidates() -> &'static [&'static str] {
    #[cfg(target_os = "linux")]
    {
        &[
            "libnvJitLink.so.13",
            "libnvJitLink.so.12",
            "libnvJitLink.so",
        ]
    }
    #[cfg(target_os = "windows")]
    {
        // Windows DLL uses `nvJitLink_<major>_<minor>_0.dll`.
        &[
            "nvJitLink_130_0.dll",
            "nvJitLink_128_0.dll",
            "nvJitLink_126_0.dll",
            "nvJitLink_123_0.dll",
            "nvJitLink_120_0.dll",
        ]
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        &[]
    }
}

macro_rules! nvjitlink_fns {
    ($($name:ident as $sym:literal : $pfn:ty);* $(;)?) => {
        pub struct NvJitLink {
            lib: Library,
            $($name: OnceLock<$pfn>,)*
        }

        impl core::fmt::Debug for NvJitLink {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct("NvJitLink").field("lib", &self.lib).finish_non_exhaustive()
            }
        }

        impl NvJitLink {
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
                Self {
                    lib,
                    $($name: OnceLock::new(),)*
                }
            }
        }
    };
}

nvjitlink_fns! {
    nv_jit_link_create as "nvJitLinkCreate": PFN_nvJitLinkCreate;
    nv_jit_link_destroy as "nvJitLinkDestroy": PFN_nvJitLinkDestroy;
    nv_jit_link_add_data as "nvJitLinkAddData": PFN_nvJitLinkAddData;
    nv_jit_link_add_file as "nvJitLinkAddFile": PFN_nvJitLinkAddFile;
    nv_jit_link_complete as "nvJitLinkComplete": PFN_nvJitLinkComplete;
    nv_jit_link_get_linked_cubin_size as "nvJitLinkGetLinkedCubinSize": PFN_nvJitLinkGetLinkedCubinSize;
    nv_jit_link_get_linked_cubin as "nvJitLinkGetLinkedCubin": PFN_nvJitLinkGetLinkedCubin;
    nv_jit_link_get_linked_ptx_size as "nvJitLinkGetLinkedPtxSize": PFN_nvJitLinkGetLinkedPtxSize;
    nv_jit_link_get_linked_ptx as "nvJitLinkGetLinkedPtx": PFN_nvJitLinkGetLinkedPtx;
    nv_jit_link_get_error_log_size as "nvJitLinkGetErrorLogSize": PFN_nvJitLinkGetErrorLogSize;
    nv_jit_link_get_error_log as "nvJitLinkGetErrorLog": PFN_nvJitLinkGetErrorLog;
    nv_jit_link_get_info_log_size as "nvJitLinkGetInfoLogSize": PFN_nvJitLinkGetInfoLogSize;
    nv_jit_link_get_info_log as "nvJitLinkGetInfoLog": PFN_nvJitLinkGetInfoLog;
    nv_jit_link_version as "nvJitLinkVersion": PFN_nvJitLinkVersion;
}

pub fn nvjitlink() -> Result<&'static NvJitLink, LoaderError> {
    static NVJITLINK: OnceLock<NvJitLink> = OnceLock::new();
    if let Some(n) = NVJITLINK.get() {
        return Ok(n);
    }
    let lib = Library::open("nvjitlink", nvjitlink_candidates())?;
    let n = NvJitLink::empty(lib);
    let _ = NVJITLINK.set(n);
    Ok(NVJITLINK.get().expect("OnceLock set or lost race"))
}
