//! Raw FFI + dynamic loader for NVIDIA cuFFT.
//!
//! `baracuda-cufft` wraps this with a safe, typed API. Use this crate
//! directly only if you need a function that the safe layer hasn't
//! wrapped yet (in which case please file a bug).

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
#![warn(missing_debug_implementations)]

use core::ffi::c_int;
use std::sync::OnceLock;

use baracuda_core::{platform, Library, LoaderError};
use baracuda_cuda_sys::runtime::cudaStream_t;
use baracuda_types::CudaStatus;

/// A cuFFT plan handle. Historically typedef'd as `int`.
pub type cufftHandle = c_int;

/// cuFFT transform direction.
pub const CUFFT_FORWARD: c_int = -1;
/// Inverse-transform direction flag for `cufftExec*` calls.
pub const CUFFT_INVERSE: c_int = 1;

/// cuFFT transform type.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cufftType {
    /// real-to-complex (single-precision) transform.
    R2C = 0x2A,
    /// complex-to-real (single-precision) transform.
    C2R = 0x2C,
    /// complex-to-complex (single-precision) transform.
    C2C = 0x29,
    /// real-to-complex (double-precision) transform.
    D2Z = 0x6A,
    /// complex-to-real (double-precision) transform.
    Z2D = 0x6C,
    /// complex-to-complex (double-precision) transform.
    Z2Z = 0x69,
}

/// cuFFT complex (single precision) — layout-compatible with
/// `baracuda_types::Complex32`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct cufftComplex {
    /// `x` component.
    pub x: f32,
    /// `y` component.
    pub y: f32,
}

/// cuFFT complex (double precision) — layout-compatible with `baracuda_types::Complex64`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct cufftDoubleComplex {
    /// `x` component.
    pub x: f64,
    /// `y` component.
    pub y: f64,
}

/// Return code from a cuFFT call.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct cufftResult(pub i32);

impl cufftResult {
    /// `CUFFT_SUCCESS` — operation succeeded.
    pub const SUCCESS: Self = Self(0);
    /// `CUFFT_INVALID_PLAN` — the plan handle is invalid.
    pub const INVALID_PLAN: Self = Self(1);
    /// `CUFFT_ALLOC_FAILED` — an allocation failed.
    pub const ALLOC_FAILED: Self = Self(2);
    /// `CUFFT_INVALID_TYPE` — the requested transform type is invalid.
    pub const INVALID_TYPE: Self = Self(3);
    /// `CUFFT_INVALID_VALUE` — an argument has an invalid value.
    pub const INVALID_VALUE: Self = Self(4);
    /// `CUFFT_INTERNAL_ERROR` — an internal cuFFT error occurred.
    pub const INTERNAL_ERROR: Self = Self(5);
    /// `CUFFT_EXEC_FAILED` — transform execution failed.
    pub const EXEC_FAILED: Self = Self(6);
    /// `CUFFT_SETUP_FAILED` — library setup failed.
    pub const SETUP_FAILED: Self = Self(7);
    /// `CUFFT_INVALID_SIZE` — the requested transform size is invalid.
    pub const INVALID_SIZE: Self = Self(8);
    /// `CUFFT_UNALIGNED_DATA` — input/output data is misaligned.
    pub const UNALIGNED_DATA: Self = Self(9);
    /// `CUFFT_INVALID_DEVICE` — the active CUDA device is invalid for cuFFT.
    pub const INVALID_DEVICE: Self = Self(11);
    /// `CUFFT_NOT_IMPLEMENTED` — the requested feature is not implemented.
    pub const NOT_IMPLEMENTED: Self = Self(14);
    /// `CUFFT_NOT_SUPPORTED` — the requested feature is not supported on this device.
    pub const NOT_SUPPORTED: Self = Self(16);

    /// Return `true` if the status code denotes success.
    pub const fn is_success(self) -> bool {
        self.0 == 0
    }
}

impl CudaStatus for cufftResult {
    fn code(self) -> i32 {
        self.0
    }
    fn name(self) -> &'static str {
        match self.0 {
            0 => "CUFFT_SUCCESS",
            1 => "CUFFT_INVALID_PLAN",
            2 => "CUFFT_ALLOC_FAILED",
            3 => "CUFFT_INVALID_TYPE",
            4 => "CUFFT_INVALID_VALUE",
            5 => "CUFFT_INTERNAL_ERROR",
            6 => "CUFFT_EXEC_FAILED",
            7 => "CUFFT_SETUP_FAILED",
            8 => "CUFFT_INVALID_SIZE",
            9 => "CUFFT_UNALIGNED_DATA",
            11 => "CUFFT_INVALID_DEVICE",
            14 => "CUFFT_NOT_IMPLEMENTED",
            16 => "CUFFT_NOT_SUPPORTED",
            _ => "CUFFT_ERROR_UNRECOGNIZED",
        }
    }
    fn description(self) -> &'static str {
        match self.0 {
            0 => "success",
            1 => "invalid plan handle",
            2 => "allocation failed",
            4 => "invalid argument",
            6 => "transform execution failed",
            8 => "invalid transform size",
            9 => "misaligned data",
            14 => "feature not implemented in this version",
            16 => "feature not supported on the target device",
            _ => "unrecognized cuFFT status code",
        }
    }
    fn is_success(self) -> bool {
        cufftResult::is_success(self)
    }
    fn library(self) -> &'static str {
        "cufft"
    }
}

// ---- function-pointer types ----

/// Function-pointer type for `cufftCreate` (create cuFFT plan handle). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftCreate = unsafe extern "C" fn(plan: *mut cufftHandle) -> cufftResult;
/// Function-pointer type for `cufftDestroy` (destroy cuFFT plan handle). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftDestroy = unsafe extern "C" fn(plan: cufftHandle) -> cufftResult;
/// Function-pointer type for `cufftPlan1d` (create 1D FFT plan). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftPlan1d = unsafe extern "C" fn(
    plan: *mut cufftHandle,
    nx: c_int,
    ty: cufftType,
    batch: c_int,
) -> cufftResult;
/// Function-pointer type for `cufftPlan2d` (create 2D FFT plan). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftPlan2d = unsafe extern "C" fn(
    plan: *mut cufftHandle,
    nx: c_int,
    ny: c_int,
    ty: cufftType,
) -> cufftResult;
/// Function-pointer type for `cufftPlan3d` (create 3D FFT plan). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftPlan3d = unsafe extern "C" fn(
    plan: *mut cufftHandle,
    nx: c_int,
    ny: c_int,
    nz: c_int,
    ty: cufftType,
) -> cufftResult;
/// Function-pointer type for `cufftSetStream` (bind a CUDA stream to a cuFFT plan). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftSetStream =
    unsafe extern "C" fn(plan: cufftHandle, stream: cudaStream_t) -> cufftResult;
/// Function-pointer type for `cufftGetVersion` (query cuFFT library version). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftGetVersion = unsafe extern "C" fn(version: *mut c_int) -> cufftResult;

/// Function-pointer type for `cufftExecR2C` (execute real-to-complex (single-precision) FFT). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftExecR2C = unsafe extern "C" fn(
    plan: cufftHandle,
    input: *mut f32,
    output: *mut cufftComplex,
) -> cufftResult;
/// Function-pointer type for `cufftExecC2R` (execute complex-to-real (single-precision) FFT). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftExecC2R = unsafe extern "C" fn(
    plan: cufftHandle,
    input: *mut cufftComplex,
    output: *mut f32,
) -> cufftResult;
/// Function-pointer type for `cufftExecC2C` (execute complex-to-complex (single-precision) FFT). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftExecC2C = unsafe extern "C" fn(
    plan: cufftHandle,
    input: *mut cufftComplex,
    output: *mut cufftComplex,
    direction: c_int,
) -> cufftResult;

// ---- Double-precision exec paths ----

/// Function-pointer type for `cufftExecD2Z` (execute real-to-complex (double-precision) FFT). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftExecD2Z = unsafe extern "C" fn(
    plan: cufftHandle,
    input: *mut f64,
    output: *mut cufftDoubleComplex,
) -> cufftResult;

/// Function-pointer type for `cufftExecZ2D` (execute complex-to-real (double-precision) FFT). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftExecZ2D = unsafe extern "C" fn(
    plan: cufftHandle,
    input: *mut cufftDoubleComplex,
    output: *mut f64,
) -> cufftResult;

/// Function-pointer type for `cufftExecZ2Z` (execute complex-to-complex (double-precision) FFT). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftExecZ2Z = unsafe extern "C" fn(
    plan: cufftHandle,
    input: *mut cufftDoubleComplex,
    output: *mut cufftDoubleComplex,
    direction: c_int,
) -> cufftResult;

// ---- Batched / many plans ----

/// Function-pointer type for `cufftPlanMany` (create batched / strided FFT plan). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftPlanMany = unsafe extern "C" fn(
    plan: *mut cufftHandle,
    rank: c_int,
    n: *mut c_int,
    inembed: *mut c_int,
    istride: c_int,
    idist: c_int,
    onembed: *mut c_int,
    ostride: c_int,
    odist: c_int,
    ty: cufftType,
    batch: c_int,
) -> cufftResult;

/// Function-pointer type for `cufftMakePlan1d` (configure an existing handle as a 1D FFT plan). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftMakePlan1d = unsafe extern "C" fn(
    plan: cufftHandle,
    nx: c_int,
    ty: cufftType,
    batch: c_int,
    work_size: *mut usize,
) -> cufftResult;

/// Function-pointer type for `cufftMakePlan2d` (configure an existing handle as a 2D FFT plan). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftMakePlan2d = unsafe extern "C" fn(
    plan: cufftHandle,
    nx: c_int,
    ny: c_int,
    ty: cufftType,
    work_size: *mut usize,
) -> cufftResult;

/// Function-pointer type for `cufftMakePlan3d` (configure an existing handle as a 3D FFT plan). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftMakePlan3d = unsafe extern "C" fn(
    plan: cufftHandle,
    nx: c_int,
    ny: c_int,
    nz: c_int,
    ty: cufftType,
    work_size: *mut usize,
) -> cufftResult;

/// Function-pointer type for `cufftMakePlanMany` (configure an existing handle as a batched/strided FFT plan). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftMakePlanMany = unsafe extern "C" fn(
    plan: cufftHandle,
    rank: c_int,
    n: *mut c_int,
    inembed: *mut c_int,
    istride: c_int,
    idist: c_int,
    onembed: *mut c_int,
    ostride: c_int,
    odist: c_int,
    ty: cufftType,
    batch: c_int,
    work_size: *mut usize,
) -> cufftResult;

/// Function-pointer type for `cufftMakePlanMany64` (configure an existing handle as a 64-bit batched/strided FFT plan). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftMakePlanMany64 = unsafe extern "C" fn(
    plan: cufftHandle,
    rank: c_int,
    n: *mut i64,
    inembed: *mut i64,
    istride: i64,
    idist: i64,
    onembed: *mut i64,
    ostride: i64,
    odist: i64,
    ty: cufftType,
    batch: i64,
    work_size: *mut usize,
) -> cufftResult;

/// Function-pointer type for `cufftEstimate1d` (estimate workspace size for a 1D FFT plan). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftEstimate1d = unsafe extern "C" fn(
    nx: c_int,
    ty: cufftType,
    batch: c_int,
    work_size: *mut usize,
) -> cufftResult;

/// Function-pointer type for `cufftEstimate2d` (estimate workspace size for a 2D FFT plan). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftEstimate2d = unsafe extern "C" fn(
    nx: c_int,
    ny: c_int,
    ty: cufftType,
    work_size: *mut usize,
) -> cufftResult;

/// Function-pointer type for `cufftEstimate3d` (estimate workspace size for a 3D FFT plan). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftEstimate3d = unsafe extern "C" fn(
    nx: c_int,
    ny: c_int,
    nz: c_int,
    ty: cufftType,
    work_size: *mut usize,
) -> cufftResult;

/// Function-pointer type for `cufftEstimateMany` (estimate workspace size for a batched/strided FFT plan). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftEstimateMany = unsafe extern "C" fn(
    rank: c_int,
    n: *mut c_int,
    inembed: *mut c_int,
    istride: c_int,
    idist: c_int,
    onembed: *mut c_int,
    ostride: c_int,
    odist: c_int,
    ty: cufftType,
    batch: c_int,
    work_size: *mut usize,
) -> cufftResult;

/// Function-pointer type for `cufftGetSize1d` (query exact workspace size for a 1D plan). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftGetSize1d = unsafe extern "C" fn(
    plan: cufftHandle,
    nx: c_int,
    ty: cufftType,
    batch: c_int,
    work_size: *mut usize,
) -> cufftResult;

/// Function-pointer type for `cufftGetSize` (query exact workspace size for a configured plan). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftGetSize = unsafe extern "C" fn(plan: cufftHandle, work_size: *mut usize) -> cufftResult;

// ---- Work-area management ----

/// Function-pointer type for `cufftSetWorkArea` (supply caller-managed work area for a plan). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftSetWorkArea = unsafe extern "C" fn(
    plan: cufftHandle,
    work_area: *mut core::ffi::c_void,
) -> cufftResult;

/// Function-pointer type for `cufftSetAutoAllocation` (toggle automatic workspace allocation). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftSetAutoAllocation =
    unsafe extern "C" fn(plan: cufftHandle, auto_allocate: c_int) -> cufftResult;

// ---- Properties ----

/// Function-pointer type for `cufftGetProperty` (query a cuFFT property value). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftGetProperty =
    unsafe extern "C" fn(prop: c_int, value_out: *mut c_int) -> cufftResult;

// ---- Multi-GPU (XT) ----

/// Function-pointer type for `cufftXtSetGPUs` (select GPUs for multi-GPU XT execution). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftXtSetGPUs =
    unsafe extern "C" fn(plan: cufftHandle, n: c_int, which_gpus: *mut c_int) -> cufftResult;

/// Function-pointer type for `cufftXtMakePlanMany` (configure a multi-GPU XT batched/strided plan). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftXtMakePlanMany = unsafe extern "C" fn(
    plan: cufftHandle,
    rank: c_int,
    n: *mut i64,
    inembed: *mut i64,
    istride: i64,
    idist: i64,
    input_type: cudaDataType,
    onembed: *mut i64,
    ostride: i64,
    odist: i64,
    output_type: cudaDataType,
    batch: i64,
    work_size: *mut usize,
    execution_type: cudaDataType,
) -> cufftResult;

/// Function-pointer type for `cufftXtMalloc` (allocate multi-GPU XT descriptor buffer). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftXtMalloc = unsafe extern "C" fn(
    plan: cufftHandle,
    desc_out: *mut *mut core::ffi::c_void, // cudaLibXtDesc**
    subformat: c_int,
) -> cufftResult;

/// Function-pointer type for `cufftXtFree` (free a multi-GPU XT descriptor buffer). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftXtFree = unsafe extern "C" fn(desc: *mut core::ffi::c_void) -> cufftResult;

/// Function-pointer type for `cufftXtMemcpy` (copy data to/from a multi-GPU XT descriptor). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftXtMemcpy = unsafe extern "C" fn(
    plan: cufftHandle,
    dst: *mut core::ffi::c_void,
    src: *mut core::ffi::c_void,
    ty: c_int, // cufftXtCopyType
) -> cufftResult;

/// Function-pointer type for `cufftXtExec` (execute multi-GPU XT plan). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftXtExec = unsafe extern "C" fn(
    plan: cufftHandle,
    input: *mut core::ffi::c_void,
    output: *mut core::ffi::c_void,
    direction: c_int,
) -> cufftResult;

/// Function-pointer type for `cufftXtExecDescriptor` (execute multi-GPU XT plan over descriptor buffers). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftXtExecDescriptor = unsafe extern "C" fn(
    plan: cufftHandle,
    input: *mut core::ffi::c_void,
    output: *mut core::ffi::c_void,
    direction: c_int,
) -> cufftResult;

/// Function-pointer type for `cufftXtQueryPlan` (query multi-GPU XT plan properties). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftXtQueryPlan = unsafe extern "C" fn(
    plan: cufftHandle,
    query_struct: *mut core::ffi::c_void,
    query_type: c_int,
) -> cufftResult;

/// Function-pointer type for `cufftXtSetCallback` (register a load/store callback on an XT plan). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftXtSetCallback = unsafe extern "C" fn(
    plan: cufftHandle,
    callback_routine: *mut *mut core::ffi::c_void,
    cb_type: c_int,
    caller_info: *mut *mut core::ffi::c_void,
) -> cufftResult;

/// Function-pointer type for `cufftXtClearCallback` (clear a load/store callback on an XT plan). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftXtClearCallback =
    unsafe extern "C" fn(plan: cufftHandle, cb_type: c_int) -> cufftResult;

/// Function-pointer type for `cufftXtSetCallbackSharedSize` (set shared-memory size for a callback). See <https://docs.nvidia.com/cuda/cufft/index.html>.
pub type PFN_cufftXtSetCallbackSharedSize = unsafe extern "C" fn(
    plan: cufftHandle,
    cb_type: c_int,
    shared_size: usize,
) -> cufftResult;

/// `cudaDataType` forward-declared for cuFFT signatures. Matches
/// `baracuda_cuda_sys::runtime::types::cudaDataType` but kept local to
/// avoid a heavier dep.
pub type cudaDataType = c_int;

// ---- loader ----

fn cufft_candidates() -> Vec<String> {
    platform::versioned_library_candidates("cufft", &["13", "12", "11"])
}

macro_rules! cufft_fns {
    ($($name:ident as $sym:literal : $pfn:ty);* $(;)?) => {
        /// Lazily-resolved cuFFT function-pointer table.
        pub struct Cufft {
            lib: Library,
            $($name: OnceLock<$pfn>,)*
        }
        impl core::fmt::Debug for Cufft {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct("Cufft").field("lib", &self.lib).finish_non_exhaustive()
            }
        }
        impl Cufft {
            $(
                /// `func` (func).
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

cufft_fns! {
    // Lifecycle + plan construction
    cufft_create as "cufftCreate": PFN_cufftCreate;
    cufft_destroy as "cufftDestroy": PFN_cufftDestroy;
    cufft_plan_1d as "cufftPlan1d": PFN_cufftPlan1d;
    cufft_plan_2d as "cufftPlan2d": PFN_cufftPlan2d;
    cufft_plan_3d as "cufftPlan3d": PFN_cufftPlan3d;
    cufft_plan_many as "cufftPlanMany": PFN_cufftPlanMany;
    cufft_make_plan_1d as "cufftMakePlan1d": PFN_cufftMakePlan1d;
    cufft_make_plan_2d as "cufftMakePlan2d": PFN_cufftMakePlan2d;
    cufft_make_plan_3d as "cufftMakePlan3d": PFN_cufftMakePlan3d;
    cufft_make_plan_many as "cufftMakePlanMany": PFN_cufftMakePlanMany;
    cufft_make_plan_many64 as "cufftMakePlanMany64": PFN_cufftMakePlanMany64;

    // Sizing
    cufft_estimate_1d as "cufftEstimate1d": PFN_cufftEstimate1d;
    cufft_estimate_2d as "cufftEstimate2d": PFN_cufftEstimate2d;
    cufft_estimate_3d as "cufftEstimate3d": PFN_cufftEstimate3d;
    cufft_estimate_many as "cufftEstimateMany": PFN_cufftEstimateMany;
    cufft_get_size_1d as "cufftGetSize1d": PFN_cufftGetSize1d;
    cufft_get_size as "cufftGetSize": PFN_cufftGetSize;

    // Work-area
    cufft_set_work_area as "cufftSetWorkArea": PFN_cufftSetWorkArea;
    cufft_set_auto_allocation as "cufftSetAutoAllocation": PFN_cufftSetAutoAllocation;

    // Stream
    cufft_set_stream as "cufftSetStream": PFN_cufftSetStream;

    // Version / property
    cufft_get_version as "cufftGetVersion": PFN_cufftGetVersion;
    cufft_get_property as "cufftGetProperty": PFN_cufftGetProperty;

    // Exec single-precision
    cufft_exec_r2c as "cufftExecR2C": PFN_cufftExecR2C;
    cufft_exec_c2r as "cufftExecC2R": PFN_cufftExecC2R;
    cufft_exec_c2c as "cufftExecC2C": PFN_cufftExecC2C;

    // Exec double-precision
    cufft_exec_d2z as "cufftExecD2Z": PFN_cufftExecD2Z;
    cufft_exec_z2d as "cufftExecZ2D": PFN_cufftExecZ2D;
    cufft_exec_z2z as "cufftExecZ2Z": PFN_cufftExecZ2Z;

    // Multi-GPU (XT)
    cufft_xt_set_gpus as "cufftXtSetGPUs": PFN_cufftXtSetGPUs;
    cufft_xt_make_plan_many as "cufftXtMakePlanMany": PFN_cufftXtMakePlanMany;
    cufft_xt_malloc as "cufftXtMalloc": PFN_cufftXtMalloc;
    cufft_xt_free as "cufftXtFree": PFN_cufftXtFree;
    cufft_xt_memcpy as "cufftXtMemcpy": PFN_cufftXtMemcpy;
    cufft_xt_exec as "cufftXtExec": PFN_cufftXtExec;
    cufft_xt_exec_descriptor as "cufftXtExecDescriptor": PFN_cufftXtExecDescriptor;
    cufft_xt_query_plan as "cufftXtQueryPlan": PFN_cufftXtQueryPlan;

    // Callbacks
    cufft_xt_set_callback as "cufftXtSetCallback": PFN_cufftXtSetCallback;
    cufft_xt_clear_callback as "cufftXtClearCallback": PFN_cufftXtClearCallback;
    cufft_xt_set_callback_shared_size as "cufftXtSetCallbackSharedSize":
        PFN_cufftXtSetCallbackSharedSize;
}

/// Return the lazily-loaded cuFFT library accessor.
pub fn cufft() -> Result<&'static Cufft, LoaderError> {
    static CUFFT: OnceLock<Cufft> = OnceLock::new();
    if let Some(c) = CUFFT.get() {
        return Ok(c);
    }
    let candidates: Vec<&'static str> = cufft_candidates()
        .into_iter()
        .map(|s| Box::leak(s.into_boxed_str()) as &'static str)
        .collect();
    let candidates_leaked: &'static [&'static str] = Box::leak(candidates.into_boxed_slice());
    let lib = Library::open("cufft", candidates_leaked)?;
    let c = Cufft::empty(lib);
    let _ = CUFFT.set(c);
    Ok(CUFFT.get().expect("OnceLock set or lost race"))
}
