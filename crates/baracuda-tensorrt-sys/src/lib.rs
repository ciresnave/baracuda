//! Raw FFI + dynamic loader for NVIDIA TensorRT (C API surface).
//!
//! TensorRT's native public API is C++; NVIDIA ships a partial C-ABI surface
//! suitable for language bindings in `NvInferRuntimeCAPI.h` (TRT 10+). This
//! crate wraps that surface for runtime deserialization and inference. The
//! builder side of TensorRT remains C++-only; use the TRT `trtexec` tool or
//! the Python bindings to produce serialized engines, then load them here.

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
#![warn(missing_debug_implementations)]

use core::ffi::{c_char, c_int, c_void};
use std::sync::OnceLock;

use baracuda_core::{platform, Library, LoaderError};
use baracuda_cuda_sys::runtime::cudaStream_t;
use baracuda_types::CudaStatus;

// ---- opaque handles ------------------------------------------------------

pub type trtIRuntime_t = *mut c_void;
pub type trtICudaEngine_t = *mut c_void;
pub type trtIExecutionContext_t = *mut c_void;
pub type trtILogger_t = *mut c_void;
pub type trtIPluginRegistry_t = *mut c_void;
pub type trtIHostMemory_t = *mut c_void;

// ---- enums ---------------------------------------------------------------

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum trtDataType_t {
    Float = 0,
    Half = 1,
    Int8 = 2,
    Int32 = 3,
    Bool = 4,
    Uint8 = 5,
    Fp8 = 6,
    BFloat16 = 7,
    Int64 = 8,
    Int4 = 9,
    Fp4 = 10,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum trtTensorIOMode_t {
    None = 0,
    Input = 1,
    Output = 2,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum trtSeverity_t {
    InternalError = 0,
    Error = 1,
    Warning = 2,
    Info = 3,
    Verbose = 4,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum trtExecutionContextAllocationStrategy_t {
    Static = 0,
    OnProfileChange = 1,
    UserManaged = 2,
}

// ---- Dims container ------------------------------------------------------

/// Analog of `nvinfer1::Dims` — up to 8 dimensions.
pub const TRT_MAX_DIMS: usize = 8;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct trtDims_t {
    pub nb_dims: i32,
    pub d: [i64; TRT_MAX_DIMS],
}

// ---- status --------------------------------------------------------------

/// TensorRT C API returns `bool` (0/1) or `int32_t` status codes depending on the
/// function. We provide a thin `trtStatus_t` newtype for the error-reporting
/// subset so it implements [`CudaStatus`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct trtStatus_t(pub i32);

impl trtStatus_t {
    pub const SUCCESS: Self = Self(0);
    pub const FAILURE: Self = Self(-1);

    pub const fn is_success(self) -> bool {
        self.0 == 0
    }
}

impl CudaStatus for trtStatus_t {
    fn code(self) -> i32 {
        self.0
    }
    fn name(self) -> &'static str {
        match self.0 {
            0 => "TRT_SUCCESS",
            -1 => "TRT_FAILURE",
            _ => "TRT_UNRECOGNIZED",
        }
    }
    fn description(self) -> &'static str {
        match self.0 {
            0 => "success",
            -1 => "TensorRT call failed (check logger output)",
            _ => "unrecognized TensorRT status code",
        }
    }
    fn is_success(self) -> bool {
        trtStatus_t::is_success(self)
    }
    fn library(self) -> &'static str {
        "tensorrt"
    }
}

// ---- function-pointer types ----------------------------------------------

pub type PFN_getInferLibVersion = unsafe extern "C" fn() -> i32;

/// Logger callback signature (matches `nvinfer1::ILogger::log`).
pub type trtLogCallback =
    unsafe extern "C" fn(severity: trtSeverity_t, msg: *const c_char, user: *mut c_void);

pub type PFN_createInferRuntime =
    unsafe extern "C" fn(logger: trtILogger_t) -> trtIRuntime_t;
pub type PFN_destroyInferRuntime = unsafe extern "C" fn(runtime: trtIRuntime_t);

pub type PFN_deserializeCudaEngine = unsafe extern "C" fn(
    runtime: trtIRuntime_t,
    blob: *const c_void,
    size: usize,
) -> trtICudaEngine_t;
pub type PFN_destroyCudaEngine = unsafe extern "C" fn(engine: trtICudaEngine_t);

pub type PFN_engineGetNbIOTensors =
    unsafe extern "C" fn(engine: trtICudaEngine_t) -> i32;
pub type PFN_engineGetIOTensorName =
    unsafe extern "C" fn(engine: trtICudaEngine_t, index: i32) -> *const c_char;
pub type PFN_engineGetTensorIOMode = unsafe extern "C" fn(
    engine: trtICudaEngine_t,
    name: *const c_char,
) -> trtTensorIOMode_t;
pub type PFN_engineGetTensorDataType = unsafe extern "C" fn(
    engine: trtICudaEngine_t,
    name: *const c_char,
) -> trtDataType_t;
pub type PFN_engineGetTensorShape =
    unsafe extern "C" fn(engine: trtICudaEngine_t, name: *const c_char) -> trtDims_t;
pub type PFN_engineGetTensorBytesPerComponent =
    unsafe extern "C" fn(engine: trtICudaEngine_t, name: *const c_char) -> i32;
pub type PFN_engineCreateExecutionContext =
    unsafe extern "C" fn(engine: trtICudaEngine_t) -> trtIExecutionContext_t;
pub type PFN_engineCreateExecutionContextWithStrategy = unsafe extern "C" fn(
    engine: trtICudaEngine_t,
    strategy: trtExecutionContextAllocationStrategy_t,
) -> trtIExecutionContext_t;
pub type PFN_destroyExecutionContext = unsafe extern "C" fn(ctx: trtIExecutionContext_t);

pub type PFN_contextSetInputShape = unsafe extern "C" fn(
    ctx: trtIExecutionContext_t,
    name: *const c_char,
    dims: *const trtDims_t,
) -> bool;
pub type PFN_contextGetTensorShape = unsafe extern "C" fn(
    ctx: trtIExecutionContext_t,
    name: *const c_char,
) -> trtDims_t;
pub type PFN_contextSetTensorAddress = unsafe extern "C" fn(
    ctx: trtIExecutionContext_t,
    name: *const c_char,
    data: *mut c_void,
) -> bool;
pub type PFN_contextGetTensorAddress = unsafe extern "C" fn(
    ctx: trtIExecutionContext_t,
    name: *const c_char,
) -> *mut c_void;
pub type PFN_contextEnqueueV3 =
    unsafe extern "C" fn(ctx: trtIExecutionContext_t, stream: cudaStream_t) -> bool;

pub type PFN_engineGetName =
    unsafe extern "C" fn(engine: trtICudaEngine_t) -> *const c_char;
pub type PFN_engineGetNbOptimizationProfiles =
    unsafe extern "C" fn(engine: trtICudaEngine_t) -> i32;

pub type PFN_engineSerialize = unsafe extern "C" fn(engine: trtICudaEngine_t) -> trtIHostMemory_t;
pub type PFN_hostMemoryData = unsafe extern "C" fn(mem: trtIHostMemory_t) -> *mut c_void;
pub type PFN_hostMemorySize = unsafe extern "C" fn(mem: trtIHostMemory_t) -> usize;
pub type PFN_hostMemoryDestroy = unsafe extern "C" fn(mem: trtIHostMemory_t);

// ---- loader --------------------------------------------------------------

fn tensorrt_candidates() -> Vec<String> {
    // TensorRT 10 ships libnvinfer.so.10 / nvinfer_10.dll; 8 uses "8".
    platform::versioned_library_candidates("nvinfer", &["10", "9", "8"])
}

macro_rules! trt_fns {
    ($($name:ident as $sym:literal : $pfn:ty);* $(;)?) => {
        pub struct TensorRt {
            lib: Library,
            $($name: OnceLock<$pfn>,)*
        }
        impl core::fmt::Debug for TensorRt {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct("TensorRt").field("lib", &self.lib).finish_non_exhaustive()
            }
        }
        impl TensorRt {
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

trt_fns! {
    // The symbol names below mirror the C API exported by TensorRT 10
    // (`NvInferRuntimeCAPI.h`). Symbol-name mismatches against older TRT
    // versions fall back to `LoaderError::SymbolUnavailable`, which the safe
    // crate maps to `Error::FeatureNotSupported`.
    get_infer_lib_version as "getInferLibVersion": PFN_getInferLibVersion;
    create_infer_runtime as "createInferRuntime_INTERNAL": PFN_createInferRuntime;
    destroy_infer_runtime as "destroyInferRuntime": PFN_destroyInferRuntime;
    deserialize_cuda_engine as "trtRuntimeDeserializeCudaEngine": PFN_deserializeCudaEngine;
    destroy_cuda_engine as "trtCudaEngineDestroy": PFN_destroyCudaEngine;
    engine_get_nb_io_tensors as "trtCudaEngineGetNbIOTensors": PFN_engineGetNbIOTensors;
    engine_get_io_tensor_name as "trtCudaEngineGetIOTensorName": PFN_engineGetIOTensorName;
    engine_get_tensor_io_mode as "trtCudaEngineGetTensorIOMode": PFN_engineGetTensorIOMode;
    engine_get_tensor_data_type as "trtCudaEngineGetTensorDataType": PFN_engineGetTensorDataType;
    engine_get_tensor_shape as "trtCudaEngineGetTensorShape": PFN_engineGetTensorShape;
    engine_get_tensor_bytes_per_component as "trtCudaEngineGetTensorBytesPerComponent": PFN_engineGetTensorBytesPerComponent;
    engine_create_execution_context as "trtCudaEngineCreateExecutionContext": PFN_engineCreateExecutionContext;
    engine_create_execution_context_with_strategy as "trtCudaEngineCreateExecutionContextWithStrategy": PFN_engineCreateExecutionContextWithStrategy;
    destroy_execution_context as "trtExecutionContextDestroy": PFN_destroyExecutionContext;
    context_set_input_shape as "trtExecutionContextSetInputShape": PFN_contextSetInputShape;
    context_get_tensor_shape as "trtExecutionContextGetTensorShape": PFN_contextGetTensorShape;
    context_set_tensor_address as "trtExecutionContextSetTensorAddress": PFN_contextSetTensorAddress;
    context_get_tensor_address as "trtExecutionContextGetTensorAddress": PFN_contextGetTensorAddress;
    context_enqueue_v3 as "trtExecutionContextEnqueueV3": PFN_contextEnqueueV3;
    engine_get_name as "trtCudaEngineGetName": PFN_engineGetName;
    engine_get_nb_optimization_profiles as "trtCudaEngineGetNbOptimizationProfiles": PFN_engineGetNbOptimizationProfiles;
    engine_serialize as "trtCudaEngineSerialize": PFN_engineSerialize;
    host_memory_data as "trtHostMemoryData": PFN_hostMemoryData;
    host_memory_size as "trtHostMemorySize": PFN_hostMemorySize;
    host_memory_destroy as "trtHostMemoryDestroy": PFN_hostMemoryDestroy;
}

pub fn tensorrt() -> Result<&'static TensorRt, LoaderError> {
    static TRT: OnceLock<TensorRt> = OnceLock::new();
    if let Some(c) = TRT.get() {
        return Ok(c);
    }
    let candidates: Vec<&'static str> = tensorrt_candidates()
        .into_iter()
        .map(|s| Box::leak(s.into_boxed_str()) as &'static str)
        .collect();
    let candidates_leaked: &'static [&'static str] = Box::leak(candidates.into_boxed_slice());
    let lib = Library::open("nvinfer", candidates_leaked)?;
    let c = TensorRt::empty(lib);
    let _ = TRT.set(c);
    Ok(TRT.get().expect("OnceLock set or lost race"))
}

// A pointer placeholder that satisfies the `c_int` dependency used by
// cross-checking crates (silences unused-import lint on some configs).
#[doc(hidden)]
pub const _UNUSED_C_INT_MARKER: c_int = 0;
