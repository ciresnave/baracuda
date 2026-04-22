//! Raw FFI + dynamic loader for NVIDIA NPP (Performance Primitives).
//!
//! NPP ships as ~10 separate DLLs on Windows (nppc, nppial, nppicc, nppidei,
//! nppif, nppig, nppim, nppist, nppisu, nppitc, npps). v0.1 exposes `nppc`
//! (core) and `npps` (signal) — enough for version queries and a demo
//! signal-arithmetic op. Image-processing DLLs (nppi*) land in follow-ups.

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
#![warn(missing_debug_implementations)]

use core::ffi::{c_int, c_void};
use std::sync::OnceLock;

use baracuda_core::{platform, Library, LoaderError};
use baracuda_types::CudaStatus;

/// NPP library version (returned by `nppGetLibVersion`).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct NppLibraryVersion {
    pub major: c_int,
    pub minor: c_int,
    pub build: c_int,
}

/// NPP status code. Negative values are errors; positive values are warnings.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct NppStatus(pub i32);

impl NppStatus {
    pub const NPP_SUCCESS: Self = Self(0);
    pub const NPP_NULL_POINTER_ERROR: Self = Self(-4);
    pub const NPP_INVALID_ARGUMENT_ERROR: Self = Self(-14);
    pub const NPP_SIZE_ERROR: Self = Self(-21);
    pub const NPP_STEP_ERROR: Self = Self(-14);

    pub const fn is_success(self) -> bool {
        self.0 == 0
    }
    pub const fn is_error(self) -> bool {
        self.0 < 0
    }
}

impl CudaStatus for NppStatus {
    fn code(self) -> i32 {
        self.0
    }
    fn name(self) -> &'static str {
        match self.0 {
            0 => "NPP_SUCCESS",
            -4 => "NPP_NULL_POINTER_ERROR",
            -14 => "NPP_INVALID_ARGUMENT_ERROR_OR_STEP_ERROR",
            -21 => "NPP_SIZE_ERROR",
            -23 => "NPP_BAD_ARGUMENT_ERROR",
            _ if self.0 < 0 => "NPP_ERROR",
            _ => "NPP_WARNING",
        }
    }
    fn description(self) -> &'static str {
        match self.0 {
            0 => "success",
            -4 => "null pointer argument",
            _ if self.0 < 0 => "NPP error",
            _ => "NPP warning",
        }
    }
    fn is_success(self) -> bool {
        NppStatus::is_success(self)
    }
    fn library(self) -> &'static str {
        "npp"
    }
}

// ---- common NPP image-processing types -----------------------------------

/// `NppiSize` — width × height in pixels.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct NppiSize {
    pub width: c_int,
    pub height: c_int,
}

/// `NppiRect` — x, y, width, height region of interest.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct NppiRect {
    pub x: c_int,
    pub y: c_int,
    pub width: c_int,
    pub height: c_int,
}

/// `NppiPoint` — 2D integer point.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct NppiPoint {
    pub x: c_int,
    pub y: c_int,
}

/// `NppiInterpolationMode` — resampling kernel for geometric ops.
#[allow(non_snake_case)]
pub mod NppiInterpolationMode {
    pub const NN: i32 = 1;
    pub const LINEAR: i32 = 2;
    pub const CUBIC: i32 = 4;
    pub const CUBIC2P_BSPLINE: i32 = 5;
    pub const CUBIC2P_CATMULLROM: i32 = 6;
    pub const CUBIC2P_B05C03: i32 = 7;
    pub const SUPER: i32 = 8;
    pub const LANCZOS: i32 = 16;
    pub const LANCZOS3_ADVANCED: i32 = 17;
}

// ---- function-pointer types ----------------------------------------------

pub type PFN_nppGetLibVersion = unsafe extern "C" fn() -> *const NppLibraryVersion;

// --- Signal arithmetic (npps) ---
pub type PFN_nppsAdd_32f_I =
    unsafe extern "C" fn(p_src: *const f32, p_src_dst: *mut f32, n_length: c_int) -> NppStatus;

pub type PFN_nppsSub_32f_I =
    unsafe extern "C" fn(p_src: *const f32, p_src_dst: *mut f32, n_length: c_int) -> NppStatus;

pub type PFN_nppsMul_32f_I =
    unsafe extern "C" fn(p_src: *const f32, p_src_dst: *mut f32, n_length: c_int) -> NppStatus;

pub type PFN_nppsSum_32f = unsafe extern "C" fn(
    p_src: *const f32,
    n_length: c_int,
    p_sum: *mut f32,
    p_device_buffer: *mut u8,
) -> NppStatus;

pub type PFN_nppsSumGetBufferSize_32f =
    unsafe extern "C" fn(n_length: c_int, h_buffer_size: *mut c_int) -> NppStatus;

pub type PFN_nppsMinMax_32f = unsafe extern "C" fn(
    p_src: *const f32,
    n_length: c_int,
    p_min: *mut f32,
    p_max: *mut f32,
    p_device_buffer: *mut u8,
) -> NppStatus;

pub type PFN_nppsMinMaxGetBufferSize_32f =
    unsafe extern "C" fn(n_length: c_int, h_buffer_size: *mut c_int) -> NppStatus;

// --- Image arithmetic (nppial) — single-channel 8u + 32f ---
pub type PFN_nppiAdd_8u_C1RSfs = unsafe extern "C" fn(
    p_src1: *const u8,
    n_src1_step: c_int,
    p_src2: *const u8,
    n_src2_step: c_int,
    p_dst: *mut u8,
    n_dst_step: c_int,
    size: NppiSize,
    n_scale_factor: c_int,
) -> NppStatus;

pub type PFN_nppiAdd_32f_C1R = unsafe extern "C" fn(
    p_src1: *const f32,
    n_src1_step: c_int,
    p_src2: *const f32,
    n_src2_step: c_int,
    p_dst: *mut f32,
    n_dst_step: c_int,
    size: NppiSize,
) -> NppStatus;

pub type PFN_nppiMul_32f_C1R = unsafe extern "C" fn(
    p_src1: *const f32,
    n_src1_step: c_int,
    p_src2: *const f32,
    n_src2_step: c_int,
    p_dst: *mut f32,
    n_dst_step: c_int,
    size: NppiSize,
) -> NppStatus;

// --- Image geometry (nppig) — resize ---
pub type PFN_nppiResize_8u_C1R = unsafe extern "C" fn(
    p_src: *const u8,
    n_src_step: c_int,
    src_size: NppiSize,
    src_rect: NppiRect,
    p_dst: *mut u8,
    n_dst_step: c_int,
    dst_size: NppiSize,
    dst_rect: NppiRect,
    interpolation: c_int,
) -> NppStatus;

pub type PFN_nppiResize_32f_C1R = unsafe extern "C" fn(
    p_src: *const f32,
    n_src_step: c_int,
    src_size: NppiSize,
    src_rect: NppiRect,
    p_dst: *mut f32,
    n_dst_step: c_int,
    dst_size: NppiSize,
    dst_rect: NppiRect,
    interpolation: c_int,
) -> NppStatus;

// --- Image color conversion (nppicc) — RGB → grayscale ---
pub type PFN_nppiRGBToGray_8u_C3C1R = unsafe extern "C" fn(
    p_src: *const u8,
    n_src_step: c_int,
    p_dst: *mut u8,
    n_dst_step: c_int,
    size: NppiSize,
) -> NppStatus;

pub type PFN_nppiBGRToGray_8u_C3C1R = unsafe extern "C" fn(
    p_src: *const u8,
    n_src_step: c_int,
    p_dst: *mut u8,
    n_dst_step: c_int,
    size: NppiSize,
) -> NppStatus;

// --- Image filters (nppif) — box + gaussian ---
pub type PFN_nppiFilterBox_8u_C1R = unsafe extern "C" fn(
    p_src: *const u8,
    n_src_step: c_int,
    p_dst: *mut u8,
    n_dst_step: c_int,
    dst_roi: NppiSize,
    mask_size: NppiSize,
    anchor: NppiPoint,
) -> NppStatus;

// --- Image statistics (nppist) — sum + min/max ---
pub type PFN_nppiSum_32f_C1R = unsafe extern "C" fn(
    p_src: *const f32,
    n_src_step: c_int,
    roi: NppiSize,
    p_device_buffer: *mut u8,
    p_sum: *mut f64,
) -> NppStatus;

pub type PFN_nppiSumGetBufferHostSize_32f_C1R =
    unsafe extern "C" fn(roi: NppiSize, h_buffer_size: *mut c_int) -> NppStatus;

// ---- loaders --------------------------------------------------------------

/// Load `nppc` — the NPP core shared library (also home to
/// `nppGetLibVersion`).
fn nppc_candidates() -> Vec<String> {
    platform::versioned_library_candidates("nppc", &["13", "12", "11"])
}

/// Load `npps` — the NPP signal-processing library.
fn npps_candidates() -> Vec<String> {
    platform::versioned_library_candidates("npps", &["13", "12", "11"])
}

macro_rules! npp_fns {
    ($struct:ident ($candidates:expr) { $($name:ident as $sym:literal : $pfn:ty);* $(;)? }) => {
        pub struct $struct {
            lib: Library,
            $($name: OnceLock<$pfn>,)*
        }
        impl core::fmt::Debug for $struct {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct(stringify!($struct)).field("lib", &self.lib).finish_non_exhaustive()
            }
        }
        impl $struct {
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

            fn load_library(name: &'static str, cands: Vec<String>) -> Result<Library, LoaderError> {
                let leaked: Vec<&'static str> = cands
                    .into_iter()
                    .map(|s| Box::leak(s.into_boxed_str()) as &'static str)
                    .collect();
                let slice: &'static [&'static str] = Box::leak(leaked.into_boxed_slice());
                Library::open(name, slice)
            }
        }
    };
}

npp_fns!(Nppc(nppc_candidates()) {
    npp_get_lib_version as "nppGetLibVersion": PFN_nppGetLibVersion;
});

npp_fns!(Npps(npps_candidates()) {
    npps_add_32f_i as "nppsAdd_32f_I": PFN_nppsAdd_32f_I;
    npps_sub_32f_i as "nppsSub_32f_I": PFN_nppsSub_32f_I;
    npps_mul_32f_i as "nppsMul_32f_I": PFN_nppsMul_32f_I;
    npps_sum_32f as "nppsSum_32f": PFN_nppsSum_32f;
    npps_sum_get_buffer_size_32f as "nppsSumGetBufferSize_32f":
        PFN_nppsSumGetBufferSize_32f;
    npps_min_max_32f as "nppsMinMax_32f": PFN_nppsMinMax_32f;
    npps_min_max_get_buffer_size_32f as "nppsMinMaxGetBufferSize_32f":
        PFN_nppsMinMaxGetBufferSize_32f;
});

fn nppial_candidates() -> Vec<String> {
    platform::versioned_library_candidates("nppial", &["13", "12", "11"])
}
fn nppig_candidates() -> Vec<String> {
    platform::versioned_library_candidates("nppig", &["13", "12", "11"])
}
fn nppicc_candidates() -> Vec<String> {
    platform::versioned_library_candidates("nppicc", &["13", "12", "11"])
}
fn nppif_candidates() -> Vec<String> {
    platform::versioned_library_candidates("nppif", &["13", "12", "11"])
}
fn nppist_candidates() -> Vec<String> {
    platform::versioned_library_candidates("nppist", &["13", "12", "11"])
}

npp_fns!(Nppial(nppial_candidates()) {
    nppi_add_8u_c1r_sfs as "nppiAdd_8u_C1RSfs": PFN_nppiAdd_8u_C1RSfs;
    nppi_add_32f_c1r as "nppiAdd_32f_C1R": PFN_nppiAdd_32f_C1R;
    nppi_mul_32f_c1r as "nppiMul_32f_C1R": PFN_nppiMul_32f_C1R;
});

npp_fns!(Nppig(nppig_candidates()) {
    nppi_resize_8u_c1r as "nppiResize_8u_C1R": PFN_nppiResize_8u_C1R;
    nppi_resize_32f_c1r as "nppiResize_32f_C1R": PFN_nppiResize_32f_C1R;
});

npp_fns!(Nppicc(nppicc_candidates()) {
    nppi_rgb_to_gray_8u_c3c1r as "nppiRGBToGray_8u_C3C1R": PFN_nppiRGBToGray_8u_C3C1R;
    nppi_bgr_to_gray_8u_c3c1r as "nppiBGRToGray_8u_C3C1R": PFN_nppiBGRToGray_8u_C3C1R;
});

npp_fns!(Nppif(nppif_candidates()) {
    nppi_filter_box_8u_c1r as "nppiFilterBox_8u_C1R": PFN_nppiFilterBox_8u_C1R;
});

npp_fns!(Nppist(nppist_candidates()) {
    nppi_sum_32f_c1r as "nppiSum_32f_C1R": PFN_nppiSum_32f_C1R;
    nppi_sum_get_buffer_host_size_32f_c1r as "nppiSumGetBufferHostSize_32f_C1R":
        PFN_nppiSumGetBufferHostSize_32f_C1R;
});

pub fn nppc() -> Result<&'static Nppc, LoaderError> {
    static NPPC: OnceLock<Nppc> = OnceLock::new();
    if let Some(n) = NPPC.get() {
        return Ok(n);
    }
    let lib = Nppc::load_library("nppc", nppc_candidates())?;
    let n = Nppc::empty(lib);
    let _ = NPPC.set(n);
    Ok(NPPC.get().expect("OnceLock set or lost race"))
}

pub fn npps() -> Result<&'static Npps, LoaderError> {
    static NPPS: OnceLock<Npps> = OnceLock::new();
    if let Some(n) = NPPS.get() {
        return Ok(n);
    }
    let lib = Npps::load_library("npps", npps_candidates())?;
    let n = Npps::empty(lib);
    let _ = NPPS.set(n);
    Ok(NPPS.get().expect("OnceLock set or lost race"))
}

pub fn nppial() -> Result<&'static Nppial, LoaderError> {
    static L: OnceLock<Nppial> = OnceLock::new();
    if let Some(n) = L.get() {
        return Ok(n);
    }
    let lib = Nppial::load_library("nppial", nppial_candidates())?;
    let _ = L.set(Nppial::empty(lib));
    Ok(L.get().expect("OnceLock set or lost race"))
}

pub fn nppig() -> Result<&'static Nppig, LoaderError> {
    static L: OnceLock<Nppig> = OnceLock::new();
    if let Some(n) = L.get() {
        return Ok(n);
    }
    let lib = Nppig::load_library("nppig", nppig_candidates())?;
    let _ = L.set(Nppig::empty(lib));
    Ok(L.get().expect("OnceLock set or lost race"))
}

pub fn nppicc() -> Result<&'static Nppicc, LoaderError> {
    static L: OnceLock<Nppicc> = OnceLock::new();
    if let Some(n) = L.get() {
        return Ok(n);
    }
    let lib = Nppicc::load_library("nppicc", nppicc_candidates())?;
    let _ = L.set(Nppicc::empty(lib));
    Ok(L.get().expect("OnceLock set or lost race"))
}

pub fn nppif() -> Result<&'static Nppif, LoaderError> {
    static L: OnceLock<Nppif> = OnceLock::new();
    if let Some(n) = L.get() {
        return Ok(n);
    }
    let lib = Nppif::load_library("nppif", nppif_candidates())?;
    let _ = L.set(Nppif::empty(lib));
    Ok(L.get().expect("OnceLock set or lost race"))
}

pub fn nppist() -> Result<&'static Nppist, LoaderError> {
    static L: OnceLock<Nppist> = OnceLock::new();
    if let Some(n) = L.get() {
        return Ok(n);
    }
    let lib = Nppist::load_library("nppist", nppist_candidates())?;
    let _ = L.set(Nppist::empty(lib));
    Ok(L.get().expect("OnceLock set or lost race"))
}

// Placeholder silences an unused import lint when only one of the libs is used.
#[allow(dead_code)]
fn _touch() -> *mut c_void {
    core::ptr::null_mut()
}
