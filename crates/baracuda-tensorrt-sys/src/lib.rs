//! Raw FFI + dynamic loader for NVIDIA TensorRT (C API surface).
//!
//! `baracuda-tensorrt` wraps this with a safe, typed API. Use this
//! crate directly only if you need a function that the safe layer
//! hasn't wrapped yet (in which case please file a bug).
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

/// TensorRT IRuntime opaque handle.
pub type trtIRuntime_t = *mut c_void;
/// TensorRT ICudaEngine opaque handle.
pub type trtICudaEngine_t = *mut c_void;
/// TensorRT IExecutionContext opaque handle.
pub type trtIExecutionContext_t = *mut c_void;
/// TensorRT ILogger opaque handle.
pub type trtILogger_t = *mut c_void;
/// TensorRT IPluginRegistry opaque handle.
pub type trtIPluginRegistry_t = *mut c_void;
/// TensorRT IHostMemory opaque handle.
pub type trtIHostMemory_t = *mut c_void;

// ---- enums ---------------------------------------------------------------

/// TensorRT tensor element type (mirrors `nvinfer1::DataType`).
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum trtDataType_t {
    /// Float variant.
    Float = 0,
    /// Half variant.
    Half = 1,
    /// Int8 variant.
    Int8 = 2,
    /// Int32 variant.
    Int32 = 3,
    /// Bool variant.
    Bool = 4,
    /// Uint8 variant.
    Uint8 = 5,
    /// Fp8 variant.
    Fp8 = 6,
    /// B Float16 variant.
    BFloat16 = 7,
    /// Int64 variant.
    Int64 = 8,
    /// Int4 variant.
    Int4 = 9,
    /// Fp4 variant.
    Fp4 = 10,
}

/// TensorRT tensor I/O mode (input / output / none).
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum trtTensorIOMode_t {
    /// None variant.
    None = 0,
    /// Input variant.
    Input = 1,
    /// Output variant.
    Output = 2,
}

/// TensorRT logger severity level.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum trtSeverity_t {
    /// Internal error error code.
    InternalError = 0,
    /// Error error code.
    Error = 1,
    /// Warning variant.
    Warning = 2,
    /// Info variant.
    Info = 3,
    /// Verbose variant.
    Verbose = 4,
}

/// TensorRT execution-context allocation strategy.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum trtExecutionContextAllocationStrategy_t {
    /// Static variant.
    Static = 0,
    /// On Profile Change variant.
    OnProfileChange = 1,
    /// User Managed variant.
    UserManaged = 2,
}

// ---- Dims container ------------------------------------------------------

/// Analog of `nvinfer1::Dims` — up to 8 dimensions.
pub const TRT_MAX_DIMS: usize = 8;

/// TensorRT dimensions container (analog of `nvinfer1::Dims`).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct trtDims_t {
    /// Nb dims field.
    pub nb_dims: i32,
    /// D field.
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
    /// TensorRT status code `SUCCESS`.
    pub const SUCCESS: Self = Self(0);
    /// TensorRT status code `FAILURE`.
    pub const FAILURE: Self = Self(-1);

    /// `is_success` method on `trtStatus_t`.
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

/// get TensorRT inference library version.
pub type PFN_getInferLibVersion = unsafe extern "C" fn() -> i32;

/// Logger callback signature (matches `nvinfer1::ILogger::log`).
pub type trtLogCallback =
    unsafe extern "C" fn(severity: trtSeverity_t, msg: *const c_char, user: *mut c_void);

/// `createInferRuntime_INTERNAL(void* logger, int32_t version)` — the real
/// `extern "C"` factory exported by `libnvinfer`. The `version` argument must
/// equal the runtime's own version (the inline C++ wrapper passes
/// `NV_TENSORRT_VERSION`); pass [`PFN_getInferLibVersion`]'s result, which is
/// that value for the loaded library.
pub type PFN_createInferRuntime =
    unsafe extern "C" fn(logger: trtILogger_t, version: c_int) -> trtIRuntime_t;

// ---- shim-backed runtime operations --------------------------------------
//
// TensorRT exposes no flat C ABI for the runtime methods (deserialize, the
// engine/context getters, tensor binding, enqueueV3, …). `libnvinfer` only
// exports `getInferLibVersion` + `createInferRuntime_INTERNAL` as `extern "C"`.
//
// With the `shim` feature, `shim/trt_shim.cpp` is compiled and statically
// linked, defining the `trt*` symbols below as `extern "C"` forwarders over
// TensorRT's C++ vtable API (verified against the TRT 10.7 headers). The shim
// references no libnvinfer symbol — it does pure vtable dispatch on the opaque
// pointers handed in from Rust, so libnvinfer stays dynamically loaded at
// runtime via the loader above.
//
// Without the feature (default), the stubs return null/false/0 so the safe
// crate degrades gracefully and the workspace builds with no TensorRT SDK.

/// `true` when the C++ shim (the `shim` feature) was compiled in. The safe
/// crate consults this to return a clear error instead of a null-handle when
/// the runtime path is exercised on a shim-less build.
pub const SHIM_BUILT: bool = cfg!(feature = "shim");

#[cfg(feature = "shim")]
extern "C" {
    /// TensorRT shim: Runtime Deserialize Cuda Engine.
    pub fn trtRuntimeDeserializeCudaEngine(
        runtime: trtIRuntime_t,
        blob: *const c_void,
        size: usize,
    ) -> trtICudaEngine_t;
    /// TensorRT shim: Runtime Destroy.
    pub fn trtRuntimeDestroy(runtime: trtIRuntime_t);

    /// TensorRT shim: Cuda Engine Destroy.
    pub fn trtCudaEngineDestroy(engine: trtICudaEngine_t);
    /// TensorRT shim: Cuda Engine Get Nb I O Tensors.
    pub fn trtCudaEngineGetNbIOTensors(engine: trtICudaEngine_t) -> i32;
    /// TensorRT shim: Cuda Engine Get I O Tensor Name.
    pub fn trtCudaEngineGetIOTensorName(engine: trtICudaEngine_t, index: i32) -> *const c_char;
    /// TensorRT shim: Cuda Engine Get Tensor I O Mode.
    pub fn trtCudaEngineGetTensorIOMode(
        engine: trtICudaEngine_t,
        name: *const c_char,
    ) -> trtTensorIOMode_t;
    /// TensorRT shim: Cuda Engine Get Tensor Data Type.
    pub fn trtCudaEngineGetTensorDataType(
        engine: trtICudaEngine_t,
        name: *const c_char,
    ) -> trtDataType_t;
    /// TensorRT shim: Cuda Engine Get Tensor Shape.
    pub fn trtCudaEngineGetTensorShape(
        engine: trtICudaEngine_t,
        name: *const c_char,
    ) -> trtDims_t;
    /// TensorRT shim: Cuda Engine Get Tensor Bytes Per Component.
    pub fn trtCudaEngineGetTensorBytesPerComponent(
        engine: trtICudaEngine_t,
        name: *const c_char,
    ) -> i32;
    /// TensorRT shim: Cuda Engine Create Execution Context.
    pub fn trtCudaEngineCreateExecutionContext(
        engine: trtICudaEngine_t,
    ) -> trtIExecutionContext_t;
    /// TensorRT shim: Cuda Engine Create Execution Context With Strategy.
    pub fn trtCudaEngineCreateExecutionContextWithStrategy(
        engine: trtICudaEngine_t,
        strategy: i32,
    ) -> trtIExecutionContext_t;
    /// TensorRT shim: Cuda Engine Get Name.
    pub fn trtCudaEngineGetName(engine: trtICudaEngine_t) -> *const c_char;
    /// TensorRT shim: Cuda Engine Get Nb Optimization Profiles.
    pub fn trtCudaEngineGetNbOptimizationProfiles(engine: trtICudaEngine_t) -> i32;
    /// TensorRT shim: Cuda Engine Serialize.
    pub fn trtCudaEngineSerialize(engine: trtICudaEngine_t) -> trtIHostMemory_t;

    /// TensorRT shim: Execution Context Destroy.
    pub fn trtExecutionContextDestroy(ctx: trtIExecutionContext_t);
    /// TensorRT shim: Execution Context Set Input Shape.
    pub fn trtExecutionContextSetInputShape(
        ctx: trtIExecutionContext_t,
        name: *const c_char,
        dims: *const trtDims_t,
    ) -> bool;
    /// TensorRT shim: Execution Context Get Tensor Shape.
    pub fn trtExecutionContextGetTensorShape(
        ctx: trtIExecutionContext_t,
        name: *const c_char,
    ) -> trtDims_t;
    /// TensorRT shim: Execution Context Set Tensor Address.
    pub fn trtExecutionContextSetTensorAddress(
        ctx: trtIExecutionContext_t,
        name: *const c_char,
        data: *mut c_void,
    ) -> bool;
    /// TensorRT shim: Execution Context Get Tensor Address.
    pub fn trtExecutionContextGetTensorAddress(
        ctx: trtIExecutionContext_t,
        name: *const c_char,
    ) -> *mut c_void;
    /// TensorRT shim: Execution Context Enqueue V3.
    pub fn trtExecutionContextEnqueueV3(ctx: trtIExecutionContext_t, stream: cudaStream_t) -> bool;

    /// TensorRT shim: Host Memory Data.
    pub fn trtHostMemoryData(mem: trtIHostMemory_t) -> *mut c_void;
    /// TensorRT shim: Host Memory Size.
    pub fn trtHostMemorySize(mem: trtIHostMemory_t) -> usize;
    /// TensorRT shim: Host Memory Destroy.
    pub fn trtHostMemoryDestroy(mem: trtIHostMemory_t);
}

/// Feature-off stubs: no shim was compiled, so the runtime operations are
/// unavailable. They return null/false/0 (never call into anything) so the
/// safe crate maps the result to a clear "shim not built" error.
#[cfg(not(feature = "shim"))]
mod shim_stubs {
    use super::*;

    /// TensorRT shim: Runtime Deserialize Cuda Engine.
    #[inline]
    pub unsafe fn trtRuntimeDeserializeCudaEngine(
        _runtime: trtIRuntime_t,
        _blob: *const c_void,
        _size: usize,
    ) -> trtICudaEngine_t {
        core::ptr::null_mut()
    }
    /// TensorRT shim: Runtime Destroy.
    #[inline]
    pub unsafe fn trtRuntimeDestroy(_runtime: trtIRuntime_t) {}

    /// TensorRT shim: Cuda Engine Destroy.
    #[inline]
    pub unsafe fn trtCudaEngineDestroy(_engine: trtICudaEngine_t) {}
    /// TensorRT shim: Cuda Engine Get Nb I O Tensors.
    #[inline]
    pub unsafe fn trtCudaEngineGetNbIOTensors(_engine: trtICudaEngine_t) -> i32 {
        0
    }
    /// TensorRT shim: Cuda Engine Get I O Tensor Name.
    #[inline]
    pub unsafe fn trtCudaEngineGetIOTensorName(
        _engine: trtICudaEngine_t,
        _index: i32,
    ) -> *const c_char {
        core::ptr::null()
    }
    /// TensorRT shim: Cuda Engine Get Tensor I O Mode.
    #[inline]
    pub unsafe fn trtCudaEngineGetTensorIOMode(
        _engine: trtICudaEngine_t,
        _name: *const c_char,
    ) -> trtTensorIOMode_t {
        trtTensorIOMode_t::None
    }
    /// TensorRT shim: Cuda Engine Get Tensor Data Type.
    #[inline]
    pub unsafe fn trtCudaEngineGetTensorDataType(
        _engine: trtICudaEngine_t,
        _name: *const c_char,
    ) -> trtDataType_t {
        trtDataType_t::Float
    }
    /// TensorRT shim: Cuda Engine Get Tensor Shape.
    #[inline]
    pub unsafe fn trtCudaEngineGetTensorShape(
        _engine: trtICudaEngine_t,
        _name: *const c_char,
    ) -> trtDims_t {
        trtDims_t {
            nb_dims: -1,
            d: [0; TRT_MAX_DIMS],
        }
    }
    /// TensorRT shim: Cuda Engine Get Tensor Bytes Per Component.
    #[inline]
    pub unsafe fn trtCudaEngineGetTensorBytesPerComponent(
        _engine: trtICudaEngine_t,
        _name: *const c_char,
    ) -> i32 {
        0
    }
    /// TensorRT shim: Cuda Engine Create Execution Context.
    #[inline]
    pub unsafe fn trtCudaEngineCreateExecutionContext(
        _engine: trtICudaEngine_t,
    ) -> trtIExecutionContext_t {
        core::ptr::null_mut()
    }
    /// TensorRT shim: Cuda Engine Create Execution Context With Strategy.
    #[inline]
    pub unsafe fn trtCudaEngineCreateExecutionContextWithStrategy(
        _engine: trtICudaEngine_t,
        _strategy: i32,
    ) -> trtIExecutionContext_t {
        core::ptr::null_mut()
    }
    /// TensorRT shim: Cuda Engine Get Name.
    #[inline]
    pub unsafe fn trtCudaEngineGetName(_engine: trtICudaEngine_t) -> *const c_char {
        core::ptr::null()
    }
    /// TensorRT shim: Cuda Engine Get Nb Optimization Profiles.
    #[inline]
    pub unsafe fn trtCudaEngineGetNbOptimizationProfiles(_engine: trtICudaEngine_t) -> i32 {
        0
    }
    /// TensorRT shim: Cuda Engine Serialize.
    #[inline]
    pub unsafe fn trtCudaEngineSerialize(_engine: trtICudaEngine_t) -> trtIHostMemory_t {
        core::ptr::null_mut()
    }

    /// TensorRT shim: Execution Context Destroy.
    #[inline]
    pub unsafe fn trtExecutionContextDestroy(_ctx: trtIExecutionContext_t) {}
    /// TensorRT shim: Execution Context Set Input Shape.
    #[inline]
    pub unsafe fn trtExecutionContextSetInputShape(
        _ctx: trtIExecutionContext_t,
        _name: *const c_char,
        _dims: *const trtDims_t,
    ) -> bool {
        false
    }
    /// TensorRT shim: Execution Context Get Tensor Shape.
    #[inline]
    pub unsafe fn trtExecutionContextGetTensorShape(
        _ctx: trtIExecutionContext_t,
        _name: *const c_char,
    ) -> trtDims_t {
        trtDims_t {
            nb_dims: -1,
            d: [0; TRT_MAX_DIMS],
        }
    }
    /// TensorRT shim: Execution Context Set Tensor Address.
    #[inline]
    pub unsafe fn trtExecutionContextSetTensorAddress(
        _ctx: trtIExecutionContext_t,
        _name: *const c_char,
        _data: *mut c_void,
    ) -> bool {
        false
    }
    /// TensorRT shim: Execution Context Get Tensor Address.
    #[inline]
    pub unsafe fn trtExecutionContextGetTensorAddress(
        _ctx: trtIExecutionContext_t,
        _name: *const c_char,
    ) -> *mut c_void {
        core::ptr::null_mut()
    }
    /// TensorRT shim: Execution Context Enqueue V3.
    #[inline]
    pub unsafe fn trtExecutionContextEnqueueV3(
        _ctx: trtIExecutionContext_t,
        _stream: cudaStream_t,
    ) -> bool {
        false
    }

    /// TensorRT shim: Host Memory Data.
    #[inline]
    pub unsafe fn trtHostMemoryData(_mem: trtIHostMemory_t) -> *mut c_void {
        core::ptr::null_mut()
    }
    /// TensorRT shim: Host Memory Size.
    #[inline]
    pub unsafe fn trtHostMemorySize(_mem: trtIHostMemory_t) -> usize {
        0
    }
    /// TensorRT shim: Host Memory Destroy.
    #[inline]
    pub unsafe fn trtHostMemoryDestroy(_mem: trtIHostMemory_t) {}
}

#[cfg(not(feature = "shim"))]
pub use shim_stubs::*;

// ---- loader --------------------------------------------------------------

fn tensorrt_candidates() -> Vec<String> {
    // TensorRT 10 ships libnvinfer.so.10 / nvinfer_10.dll; 8 uses "8".
    platform::versioned_library_candidates("nvinfer", &["10", "9", "8"])
}

macro_rules! trt_fns {
    ($($name:ident as $sym:literal : $pfn:ty);* $(;)?) => {
        /// TensorRT dynamic-loader handle (libnvinfer).
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

trt_fns! {
    // Only these two are real `extern "C"` exports of `libnvinfer`. Everything
    // else on the runtime path is a C++ vtable method with no flat symbol — it
    // is reached through the `shim` functions above, not the dynamic loader.
    // `createInferRuntime_INTERNAL` takes `(logger, version)`; pass
    // `get_infer_lib_version()` as the version.
    get_infer_lib_version as "getInferLibVersion": PFN_getInferLibVersion;
    create_infer_runtime as "createInferRuntime_INTERNAL": PFN_createInferRuntime;
}

/// resolve and return the process-wide TensorRT loader.
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
