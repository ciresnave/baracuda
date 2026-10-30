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

#![allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    reason = "names mirror the vendor C API (FFI bindings)"
)]
#![warn(missing_debug_implementations)]

use core::ffi::{c_char, c_int, c_void};
use std::sync::OnceLock;

use baracuda_core::{Library, LoaderError, platform};
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
/// TensorRT IBuilder opaque handle.
pub type trtIBuilder_t = *mut c_void;
/// TensorRT IBuilderConfig opaque handle.
pub type trtIBuilderConfig_t = *mut c_void;
/// TensorRT INetworkDefinition opaque handle.
pub type trtINetworkDefinition_t = *mut c_void;
/// TensorRT ITensor opaque handle (a network tensor).
pub type trtITensor_t = *mut c_void;
/// TensorRT ILayer opaque handle (always an `ILayer*`, never a subclass pointer).
pub type trtILayer_t = *mut c_void;
/// TensorRT IOptimizationProfile opaque handle (owned by the builder that made it).
pub type trtIOptimizationProfile_t = *mut c_void;
/// `nvonnxparser::IParser` opaque handle.
pub type trtIParser_t = *mut c_void;

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

/// Logger callback signature (matches `nvinfer1::ILogger::log`). `severity` is a
/// plain integer, not [`trtSeverity_t`]: the value comes from C++, and a Rust enum
/// holding a value outside its variants is undefined behaviour, so the callee
/// validates it (see `Severity::from_raw` in `baracuda-tensorrt`).
pub type trtLogCallback =
    unsafe extern "C" fn(severity: i32, msg: *const c_char, user: *mut c_void);

/// `createInferRuntime_INTERNAL(void* logger, int32_t version)` — the real
/// `extern "C"` factory exported by `libnvinfer`. The `version` argument must
/// equal the runtime's own version (the inline C++ wrapper passes
/// `NV_TENSORRT_VERSION`); pass [`PFN_getInferLibVersion`]'s result, which is
/// that value for the loaded library.
pub type PFN_createInferRuntime =
    unsafe extern "C" fn(logger: trtILogger_t, version: c_int) -> trtIRuntime_t;

/// `getPluginRegistry()`, the `extern "C"` accessor `libnvinfer` exports for its
/// process-wide plugin registry.
pub type PFN_getPluginRegistry = unsafe extern "C" fn() -> trtIPluginRegistry_t;

/// `createInferBuilder_INTERNAL(void* logger, int32_t version)` — the
/// `extern "C"` builder factory exported by `libnvinfer`; same contract as
/// [`PFN_createInferRuntime`].
pub type PFN_createInferBuilder =
    unsafe extern "C" fn(logger: trtILogger_t, version: c_int) -> trtIBuilder_t;

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
unsafe extern "C" {
    /// TensorRT shim: create an `ILogger` that calls `cb(severity, msg, user)`
    /// for every message. Returns null on allocation failure. Free with
    /// [`trtLoggerDestroy`], after everything built from it is gone.
    pub fn trtLoggerCreate(cb: trtLogCallback, user: *mut c_void) -> trtILogger_t;
    /// TensorRT shim: destroy a logger made by [`trtLoggerCreate`].
    pub fn trtLoggerDestroy(logger: trtILogger_t);
}

#[cfg(feature = "shim")]
unsafe extern "C" {
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
    pub fn trtCudaEngineGetTensorShape(engine: trtICudaEngine_t, name: *const c_char) -> trtDims_t;
    /// TensorRT shim: Cuda Engine Get Tensor Bytes Per Component.
    pub fn trtCudaEngineGetTensorBytesPerComponent(
        engine: trtICudaEngine_t,
        name: *const c_char,
    ) -> i32;
    /// TensorRT shim: Cuda Engine Create Execution Context.
    pub fn trtCudaEngineCreateExecutionContext(engine: trtICudaEngine_t) -> trtIExecutionContext_t;
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
    /// TensorRT shim: Cuda Engine Get Profile Shape.
    pub fn trtCudaEngineGetProfileShape(
        engine: trtICudaEngine_t,
        name: *const c_char,
        profile: i32,
        selector: i32,
    ) -> trtDims_t;
    /// TensorRT shim: Execution Context Set Optimization Profile Async.
    pub fn trtExecutionContextSetOptimizationProfileAsync(
        ctx: trtIExecutionContext_t,
        profile: i32,
        stream: cudaStream_t,
    ) -> bool;
    /// TensorRT shim: Execution Context Get Optimization Profile.
    pub fn trtExecutionContextGetOptimizationProfile(ctx: trtIExecutionContext_t) -> i32;

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
    ///
    /// # Safety
    ///
    /// Shim no-op mirroring the raw TensorRT C-API signature; callers must
    /// uphold the same contract as the real symbol (valid handles/pointers).
    #[inline]
    pub unsafe fn trtRuntimeDeserializeCudaEngine(
        _runtime: trtIRuntime_t,
        _blob: *const c_void,
        _size: usize,
    ) -> trtICudaEngine_t {
        core::ptr::null_mut()
    }
    /// TensorRT shim: Runtime Destroy.
    ///
    /// # Safety
    ///
    /// Shim no-op mirroring the raw TensorRT C-API signature; callers must
    /// uphold the same contract as the real symbol (valid handles/pointers).
    #[inline]
    pub unsafe fn trtRuntimeDestroy(_runtime: trtIRuntime_t) {}

    /// TensorRT shim: Cuda Engine Destroy.
    ///
    /// # Safety
    ///
    /// Shim no-op mirroring the raw TensorRT C-API signature; callers must
    /// uphold the same contract as the real symbol (valid handles/pointers).
    #[inline]
    pub unsafe fn trtCudaEngineDestroy(_engine: trtICudaEngine_t) {}
    /// TensorRT shim: Cuda Engine Get Nb I O Tensors.
    ///
    /// # Safety
    ///
    /// Shim no-op mirroring the raw TensorRT C-API signature; callers must
    /// uphold the same contract as the real symbol (valid handles/pointers).
    #[inline]
    pub unsafe fn trtCudaEngineGetNbIOTensors(_engine: trtICudaEngine_t) -> i32 {
        0
    }
    /// TensorRT shim: Cuda Engine Get I O Tensor Name.
    ///
    /// # Safety
    ///
    /// Shim no-op mirroring the raw TensorRT C-API signature; callers must
    /// uphold the same contract as the real symbol (valid handles/pointers).
    #[inline]
    pub unsafe fn trtCudaEngineGetIOTensorName(
        _engine: trtICudaEngine_t,
        _index: i32,
    ) -> *const c_char {
        core::ptr::null()
    }
    /// TensorRT shim: Cuda Engine Get Tensor I O Mode.
    ///
    /// # Safety
    ///
    /// Shim no-op mirroring the raw TensorRT C-API signature; callers must
    /// uphold the same contract as the real symbol (valid handles/pointers).
    #[inline]
    pub unsafe fn trtCudaEngineGetTensorIOMode(
        _engine: trtICudaEngine_t,
        _name: *const c_char,
    ) -> trtTensorIOMode_t {
        trtTensorIOMode_t::None
    }
    /// TensorRT shim: Cuda Engine Get Tensor Data Type.
    ///
    /// # Safety
    ///
    /// Shim no-op mirroring the raw TensorRT C-API signature; callers must
    /// uphold the same contract as the real symbol (valid handles/pointers).
    #[inline]
    pub unsafe fn trtCudaEngineGetTensorDataType(
        _engine: trtICudaEngine_t,
        _name: *const c_char,
    ) -> trtDataType_t {
        trtDataType_t::Float
    }
    /// TensorRT shim: Cuda Engine Get Tensor Shape.
    ///
    /// # Safety
    ///
    /// Shim no-op mirroring the raw TensorRT C-API signature; callers must
    /// uphold the same contract as the real symbol (valid handles/pointers).
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
    ///
    /// # Safety
    ///
    /// Shim no-op mirroring the raw TensorRT C-API signature; callers must
    /// uphold the same contract as the real symbol (valid handles/pointers).
    #[inline]
    pub unsafe fn trtCudaEngineGetTensorBytesPerComponent(
        _engine: trtICudaEngine_t,
        _name: *const c_char,
    ) -> i32 {
        0
    }
    /// TensorRT shim: Cuda Engine Create Execution Context.
    ///
    /// # Safety
    ///
    /// Shim no-op mirroring the raw TensorRT C-API signature; callers must
    /// uphold the same contract as the real symbol (valid handles/pointers).
    #[inline]
    pub unsafe fn trtCudaEngineCreateExecutionContext(
        _engine: trtICudaEngine_t,
    ) -> trtIExecutionContext_t {
        core::ptr::null_mut()
    }
    /// TensorRT shim: Cuda Engine Create Execution Context With Strategy.
    ///
    /// # Safety
    ///
    /// Shim no-op mirroring the raw TensorRT C-API signature; callers must
    /// uphold the same contract as the real symbol (valid handles/pointers).
    #[inline]
    pub unsafe fn trtCudaEngineCreateExecutionContextWithStrategy(
        _engine: trtICudaEngine_t,
        _strategy: i32,
    ) -> trtIExecutionContext_t {
        core::ptr::null_mut()
    }
    /// TensorRT shim: Cuda Engine Get Name.
    ///
    /// # Safety
    ///
    /// Shim no-op mirroring the raw TensorRT C-API signature; callers must
    /// uphold the same contract as the real symbol (valid handles/pointers).
    #[inline]
    pub unsafe fn trtCudaEngineGetName(_engine: trtICudaEngine_t) -> *const c_char {
        core::ptr::null()
    }
    /// TensorRT shim: Cuda Engine Get Nb Optimization Profiles.
    ///
    /// # Safety
    ///
    /// Shim no-op mirroring the raw TensorRT C-API signature; callers must
    /// uphold the same contract as the real symbol (valid handles/pointers).
    #[inline]
    pub unsafe fn trtCudaEngineGetNbOptimizationProfiles(_engine: trtICudaEngine_t) -> i32 {
        0
    }
    /// TensorRT shim: Cuda Engine Serialize.
    ///
    /// # Safety
    ///
    /// Shim no-op mirroring the raw TensorRT C-API signature; callers must
    /// uphold the same contract as the real symbol (valid handles/pointers).
    #[inline]
    pub unsafe fn trtCudaEngineSerialize(_engine: trtICudaEngine_t) -> trtIHostMemory_t {
        core::ptr::null_mut()
    }

    /// TensorRT shim: Execution Context Destroy.
    ///
    /// # Safety
    ///
    /// Shim no-op mirroring the raw TensorRT C-API signature; callers must
    /// uphold the same contract as the real symbol (valid handles/pointers).
    #[inline]
    pub unsafe fn trtExecutionContextDestroy(_ctx: trtIExecutionContext_t) {}
    /// TensorRT shim: Execution Context Set Input Shape.
    ///
    /// # Safety
    ///
    /// Shim no-op mirroring the raw TensorRT C-API signature; callers must
    /// uphold the same contract as the real symbol (valid handles/pointers).
    #[inline]
    pub unsafe fn trtExecutionContextSetInputShape(
        _ctx: trtIExecutionContext_t,
        _name: *const c_char,
        _dims: *const trtDims_t,
    ) -> bool {
        false
    }
    /// TensorRT shim: Execution Context Get Tensor Shape.
    ///
    /// # Safety
    ///
    /// Shim no-op mirroring the raw TensorRT C-API signature; callers must
    /// uphold the same contract as the real symbol (valid handles/pointers).
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
    ///
    /// # Safety
    ///
    /// Shim no-op mirroring the raw TensorRT C-API signature; callers must
    /// uphold the same contract as the real symbol (valid handles/pointers).
    #[inline]
    pub unsafe fn trtExecutionContextSetTensorAddress(
        _ctx: trtIExecutionContext_t,
        _name: *const c_char,
        _data: *mut c_void,
    ) -> bool {
        false
    }
    /// TensorRT shim: Execution Context Get Tensor Address.
    ///
    /// # Safety
    ///
    /// Shim no-op mirroring the raw TensorRT C-API signature; callers must
    /// uphold the same contract as the real symbol (valid handles/pointers).
    #[inline]
    pub unsafe fn trtExecutionContextGetTensorAddress(
        _ctx: trtIExecutionContext_t,
        _name: *const c_char,
    ) -> *mut c_void {
        core::ptr::null_mut()
    }
    /// TensorRT shim: Execution Context Enqueue V3.
    ///
    /// # Safety
    ///
    /// Shim no-op mirroring the raw TensorRT C-API signature; callers must
    /// uphold the same contract as the real symbol (valid handles/pointers).
    #[inline]
    pub unsafe fn trtExecutionContextEnqueueV3(
        _ctx: trtIExecutionContext_t,
        _stream: cudaStream_t,
    ) -> bool {
        false
    }

    /// TensorRT shim: Host Memory Data.
    ///
    /// # Safety
    ///
    /// Shim no-op mirroring the raw TensorRT C-API signature; callers must
    /// uphold the same contract as the real symbol (valid handles/pointers).
    #[inline]
    pub unsafe fn trtHostMemoryData(_mem: trtIHostMemory_t) -> *mut c_void {
        core::ptr::null_mut()
    }
    /// TensorRT shim: Host Memory Size.
    ///
    /// # Safety
    ///
    /// Shim no-op mirroring the raw TensorRT C-API signature; callers must
    /// uphold the same contract as the real symbol (valid handles/pointers).
    #[inline]
    pub unsafe fn trtHostMemorySize(_mem: trtIHostMemory_t) -> usize {
        0
    }
    /// TensorRT shim: Host Memory Destroy.
    ///
    /// # Safety
    ///
    /// Shim no-op mirroring the raw TensorRT C-API signature; callers must
    /// uphold the same contract as the real symbol (valid handles/pointers).
    #[inline]
    pub unsafe fn trtHostMemoryDestroy(_mem: trtIHostMemory_t) {}

    /// TensorRT shim: Cuda Engine Get Profile Shape (no-op without the shim).
    ///
    /// # Safety
    ///
    /// Shim no-op mirroring the real symbol's signature.
    #[inline]
    pub unsafe fn trtCudaEngineGetProfileShape(
        _engine: trtICudaEngine_t,
        _name: *const c_char,
        _profile: i32,
        _selector: i32,
    ) -> trtDims_t {
        trtDims_t {
            nb_dims: 0,
            d: [0; TRT_MAX_DIMS],
        }
    }
    /// TensorRT shim: Execution Context Set Optimization Profile Async (no-op without the shim).
    ///
    /// # Safety
    ///
    /// Shim no-op mirroring the real symbol's signature.
    #[inline]
    pub unsafe fn trtExecutionContextSetOptimizationProfileAsync(
        _ctx: trtIExecutionContext_t,
        _profile: i32,
        _stream: cudaStream_t,
    ) -> bool {
        false
    }
    /// TensorRT shim: Execution Context Get Optimization Profile (no-op without the shim).
    ///
    /// # Safety
    ///
    /// Shim no-op mirroring the real symbol's signature.
    #[inline]
    pub unsafe fn trtExecutionContextGetOptimizationProfile(_ctx: trtIExecutionContext_t) -> i32 {
        -1
    }

    /// TensorRT shim: create a logger (no-op: always null without the shim).
    ///
    /// # Safety
    ///
    /// Shim no-op mirroring the real symbol's signature.
    #[inline]
    pub unsafe fn trtLoggerCreate(_cb: trtLogCallback, _user: *mut c_void) -> trtILogger_t {
        core::ptr::null_mut()
    }
    /// TensorRT shim: destroy a logger (no-op without the shim).
    ///
    /// # Safety
    ///
    /// Shim no-op mirroring the real symbol's signature.
    #[inline]
    pub unsafe fn trtLoggerDestroy(_logger: trtILogger_t) {}
}

#[cfg(not(feature = "shim"))]
pub use shim_stubs::*;

/// Plugin-registry shim symbols (`nvinfer1::IPluginRegistry` and the creators it
/// holds). Only present with the `shim` feature, like [`builder`].
#[cfg(feature = "shim")]
#[allow(
    missing_docs,
    reason = "raw FFI mirror of the shim; each name is the TensorRT method it forwards to"
)]
pub mod registry {
    use super::*;

    unsafe extern "C" {
        /// 1 = `IPluginCreator` (V2 / V1 interface), 2 = `IPluginCreatorV3One`, 0 = unknown.
        pub fn trtPluginCreatorKind(creator: *mut c_void) -> i32;
        pub fn trtPluginCreatorName(creator: *mut c_void) -> *const c_char;
        pub fn trtPluginCreatorVersion(creator: *mut c_void) -> *const c_char;
        pub fn trtPluginCreatorNamespace(creator: *mut c_void) -> *const c_char;
        pub fn trtPluginRegistryGetAllCreators(
            registry: trtIPluginRegistry_t,
            count: *mut i32,
        ) -> *const *mut c_void;
        pub fn trtPluginRegistryGetCreator(
            registry: trtIPluginRegistry_t,
            name: *const c_char,
            version: *const c_char,
            namespace: *const c_char,
        ) -> *mut c_void;
        pub fn trtPluginRegistryLoadLibrary(
            registry: trtIPluginRegistry_t,
            path: *const c_char,
        ) -> *mut c_void;
        pub fn trtPluginRegistryDeregisterLibrary(
            registry: trtIPluginRegistry_t,
            handle: *mut c_void,
        );
        pub fn trtPluginRegistryDeregisterCreator(
            registry: trtIPluginRegistry_t,
            creator: *mut c_void,
        ) -> bool;
    }
}

/// `initLibNvInferPlugins(void* logger, const char* libNamespace)`, exported by
/// the TensorRT plugin library; registers TensorRT's built-in plugins.
pub type PFN_initLibNvInferPlugins =
    unsafe extern "C" fn(logger: trtILogger_t, namespace: *const c_char) -> bool;

/// Loader handle for TensorRT's plugin library (`nvinfer_plugin_10.dll` /
/// `libnvinfer_plugin.so.10`).
pub struct PluginLib {
    _lib: Library,
    init: PFN_initLibNvInferPlugins,
}

impl core::fmt::Debug for PluginLib {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PluginLib").finish_non_exhaustive()
    }
}

impl PluginLib {
    /// The `initLibNvInferPlugins` entry point.
    pub fn init_lib_nvinfer_plugins(&self) -> PFN_initLibNvInferPlugins {
        self.init
    }
}

/// Resolve and return the process-wide TensorRT plugin library.
pub fn plugin_lib() -> Result<&'static PluginLib, LoaderError> {
    static LIB: OnceLock<PluginLib> = OnceLock::new();
    if let Some(l) = LIB.get() {
        return Ok(l);
    }
    let candidates: &'static [&'static str] = Box::leak(
        if cfg!(windows) {
            vec!["nvinfer_plugin_10.dll", "nvinfer_plugin.dll"]
        } else {
            vec!["libnvinfer_plugin.so.10", "libnvinfer_plugin.so"]
        }
        .into_boxed_slice(),
    );
    let lib = Library::open("nvinfer_plugin", candidates)?;
    // SAFETY: the symbol is `bool initLibNvInferPlugins(void*, const char*)` with C linkage.
    let raw: *mut () = unsafe { lib.raw_symbol("initLibNvInferPlugins")? };
    // SAFETY: `raw` is the non-null address of that function, and a data pointer and a function pointer
    // have the same size on every platform this crate supports (the loader macro above relies on the same).
    let init: PFN_initLibNvInferPlugins =
        unsafe { core::mem::transmute_copy::<*mut (), PFN_initLibNvInferPlugins>(&raw) };
    let _ = LIB.set(PluginLib { _lib: lib, init });
    Ok(LIB.get().expect("OnceLock set or lost race"))
}

/// Builder-side shim symbols (`IBuilder`, `IBuilderConfig`, `INetworkDefinition`,
/// layers and tensors). Only present with the `shim` feature: there is no
/// no-op fallback, because a builder that silently does nothing is worse than
/// one that is absent (`baracuda-tensorrt-builder` gates its API on the same
/// feature).
#[cfg(feature = "shim")]
#[allow(
    missing_docs,
    reason = "raw FFI mirror of the shim; each name is the TensorRT method it forwards to"
)]
pub mod builder {
    use super::*;

    unsafe extern "C" {
        // IBuilder
        pub fn trtBuilderDestroy(builder: trtIBuilder_t);
        pub fn trtBuilderCreateNetwork(
            builder: trtIBuilder_t,
            flags: u32,
        ) -> trtINetworkDefinition_t;
        pub fn trtBuilderCreateConfig(builder: trtIBuilder_t) -> trtIBuilderConfig_t;
        pub fn trtBuilderBuildSerialized(
            builder: trtIBuilder_t,
            network: trtINetworkDefinition_t,
            config: trtIBuilderConfig_t,
        ) -> trtIHostMemory_t;

        // IBuilderConfig
        pub fn trtConfigDestroy(config: trtIBuilderConfig_t);
        pub fn trtConfigSetMemoryPoolLimit(config: trtIBuilderConfig_t, pool: i32, size: usize);
        pub fn trtConfigGetMemoryPoolLimit(config: trtIBuilderConfig_t, pool: i32) -> usize;
        pub fn trtConfigSetFlag(config: trtIBuilderConfig_t, flag: i32, on: bool);
        pub fn trtConfigGetFlag(config: trtIBuilderConfig_t, flag: i32) -> bool;
        pub fn trtConfigSetOptimizationLevel(config: trtIBuilderConfig_t, level: i32);
        pub fn trtConfigGetOptimizationLevel(config: trtIBuilderConfig_t) -> i32;

        // INetworkDefinition
        pub fn trtNetworkDestroy(network: trtINetworkDefinition_t);
        pub fn trtNetworkAddInput(
            network: trtINetworkDefinition_t,
            name: *const c_char,
            dtype: i32,
            dims: *const trtDims_t,
        ) -> trtITensor_t;
        pub fn trtNetworkMarkOutput(network: trtINetworkDefinition_t, tensor: trtITensor_t);
        pub fn trtNetworkGetNbInputs(network: trtINetworkDefinition_t) -> i32;
        pub fn trtNetworkGetNbOutputs(network: trtINetworkDefinition_t) -> i32;
        pub fn trtNetworkGetNbLayers(network: trtINetworkDefinition_t) -> i32;
        pub fn trtNetworkAddElementWise(
            network: trtINetworkDefinition_t,
            a: trtITensor_t,
            b: trtITensor_t,
            op: i32,
        ) -> trtILayer_t;
        pub fn trtNetworkAddActivation(
            network: trtINetworkDefinition_t,
            input: trtITensor_t,
            kind: i32,
        ) -> trtILayer_t;
        pub fn trtNetworkAddUnary(
            network: trtINetworkDefinition_t,
            input: trtITensor_t,
            op: i32,
        ) -> trtILayer_t;
        pub fn trtNetworkAddMatrixMultiply(
            network: trtINetworkDefinition_t,
            a: trtITensor_t,
            op_a: i32,
            b: trtITensor_t,
            op_b: i32,
        ) -> trtILayer_t;
        pub fn trtNetworkAddConstant(
            network: trtINetworkDefinition_t,
            dims: *const trtDims_t,
            dtype: i32,
            values: *const c_void,
            count: i64,
        ) -> trtILayer_t;
        pub fn trtNetworkAddShuffle(
            network: trtINetworkDefinition_t,
            input: trtITensor_t,
        ) -> trtILayer_t;
        pub fn trtNetworkAddReduce(
            network: trtINetworkDefinition_t,
            input: trtITensor_t,
            op: i32,
            axes: u32,
            keep_dims: bool,
        ) -> trtILayer_t;
        pub fn trtNetworkAddSoftMax(
            network: trtINetworkDefinition_t,
            input: trtITensor_t,
        ) -> trtILayer_t;
        pub fn trtNetworkAddConcatenation(
            network: trtINetworkDefinition_t,
            inputs: *const trtITensor_t,
            n: i32,
        ) -> trtILayer_t;
        pub fn trtNetworkAddCast(
            network: trtINetworkDefinition_t,
            input: trtITensor_t,
            dtype: i32,
        ) -> trtILayer_t;
        pub fn trtNetworkAddIdentity(
            network: trtINetworkDefinition_t,
            input: trtITensor_t,
        ) -> trtILayer_t;
        pub fn trtNetworkAddConvolutionNd(
            network: trtINetworkDefinition_t,
            input: trtITensor_t,
            nb_output_maps: i64,
            kernel_size: *const trtDims_t,
            kernel: *const c_void,
            kernel_count: i64,
            bias: *const c_void,
            bias_count: i64,
        ) -> trtILayer_t;
        pub fn trtNetworkAddDeconvolutionNd(
            network: trtINetworkDefinition_t,
            input: trtITensor_t,
            nb_output_maps: i64,
            kernel_size: *const trtDims_t,
            kernel: *const c_void,
            kernel_count: i64,
            bias: *const c_void,
            bias_count: i64,
        ) -> trtILayer_t;
        pub fn trtNetworkAddGather(
            network: trtINetworkDefinition_t,
            data: trtITensor_t,
            indices: trtITensor_t,
            axis: i32,
        ) -> trtILayer_t;
        pub fn trtNetworkAddSlice(
            network: trtINetworkDefinition_t,
            input: trtITensor_t,
            start: *const trtDims_t,
            size: *const trtDims_t,
            stride: *const trtDims_t,
        ) -> trtILayer_t;
        pub fn trtNetworkAddPoolingNd(
            network: trtINetworkDefinition_t,
            input: trtITensor_t,
            pooling_type: i32,
            window_size: *const trtDims_t,
        ) -> trtILayer_t;

        // ILayer / ITensor
        pub fn trtLayerGetOutput(layer: trtILayer_t, index: i32) -> trtITensor_t;
        pub fn trtLayerGetNbOutputs(layer: trtILayer_t) -> i32;
        pub fn trtLayerSetName(layer: trtILayer_t, name: *const c_char);
        pub fn trtTensorSetName(tensor: trtITensor_t, name: *const c_char);
        pub fn trtTensorGetName(tensor: trtITensor_t) -> *const c_char;
        pub fn trtTensorGetDimensions(tensor: trtITensor_t) -> trtDims_t;
        pub fn trtTensorGetType(tensor: trtITensor_t) -> i32;

        // layer-specific setters
        pub fn trtActivationSetAlpha(layer: trtILayer_t, alpha: f32);
        pub fn trtActivationSetBeta(layer: trtILayer_t, beta: f32);
        pub fn trtShuffleSetReshapeDims(layer: trtILayer_t, dims: *const trtDims_t);
        pub fn trtShuffleSetFirstTranspose(layer: trtILayer_t, order: *const i32);
        pub fn trtShuffleSetSecondTranspose(layer: trtILayer_t, order: *const i32);
        pub fn trtSoftMaxSetAxes(layer: trtILayer_t, axes: u32);
        pub fn trtConcatenationSetAxis(layer: trtILayer_t, axis: i32);
        pub fn trtConvolutionSetStrideNd(layer: trtILayer_t, dims: *const trtDims_t);
        pub fn trtConvolutionSetPaddingNd(layer: trtILayer_t, dims: *const trtDims_t);
        pub fn trtConvolutionSetDilationNd(layer: trtILayer_t, dims: *const trtDims_t);
        pub fn trtConvolutionSetNbGroups(layer: trtILayer_t, groups: i64);
        pub fn trtDeconvolutionSetStrideNd(layer: trtILayer_t, dims: *const trtDims_t);
        pub fn trtDeconvolutionSetPaddingNd(layer: trtILayer_t, dims: *const trtDims_t);
        pub fn trtDeconvolutionSetDilationNd(layer: trtILayer_t, dims: *const trtDims_t);
        pub fn trtDeconvolutionSetNbGroups(layer: trtILayer_t, groups: i64);
        pub fn trtPoolingSetStrideNd(layer: trtILayer_t, dims: *const trtDims_t);
        pub fn trtPoolingSetPaddingNd(layer: trtILayer_t, dims: *const trtDims_t);

        // network inputs / outputs by index
        pub fn trtNetworkGetInput(network: trtINetworkDefinition_t, index: i32) -> trtITensor_t;
        pub fn trtNetworkGetOutput(network: trtINetworkDefinition_t, index: i32) -> trtITensor_t;

        // ONNX parser
        pub fn trtOnnxParserDestroy(parser: trtIParser_t);
        pub fn trtOnnxParserParse(
            parser: trtIParser_t,
            data: *const c_void,
            size: usize,
            path: *const c_char,
        ) -> bool;
        pub fn trtOnnxParserGetNbErrors(parser: trtIParser_t) -> i32;
        pub fn trtOnnxParserClearErrors(parser: trtIParser_t);
        pub fn trtOnnxParserSupportsOperator(parser: trtIParser_t, op: *const c_char) -> bool;
        pub fn trtOnnxParserErrorCode(parser: trtIParser_t, index: i32) -> i32;
        pub fn trtOnnxParserErrorDesc(parser: trtIParser_t, index: i32) -> *const c_char;
        pub fn trtOnnxParserErrorNode(parser: trtIParser_t, index: i32) -> i32;
        pub fn trtOnnxParserErrorNodeName(parser: trtIParser_t, index: i32) -> *const c_char;
        pub fn trtOnnxParserErrorNodeOperator(parser: trtIParser_t, index: i32) -> *const c_char;

        // optimization profiles
        pub fn trtBuilderCreateOptimizationProfile(
            builder: trtIBuilder_t,
        ) -> trtIOptimizationProfile_t;
        pub fn trtProfileSetDimensions(
            profile: trtIOptimizationProfile_t,
            name: *const c_char,
            selector: i32,
            dims: *const trtDims_t,
        ) -> bool;
        pub fn trtProfileGetDimensions(
            profile: trtIOptimizationProfile_t,
            name: *const c_char,
            selector: i32,
        ) -> trtDims_t;
        pub fn trtProfileIsValid(profile: trtIOptimizationProfile_t) -> bool;
        pub fn trtConfigAddOptimizationProfile(
            config: trtIBuilderConfig_t,
            profile: trtIOptimizationProfile_t,
        ) -> i32;
        pub fn trtConfigGetNbOptimizationProfiles(config: trtIBuilderConfig_t) -> i32;
    }
}

// ---- loader --------------------------------------------------------------

fn tensorrt_candidates() -> Vec<String> {
    // TensorRT 10 ships libnvinfer.so.10 / nvinfer_10.dll; 8 uses "8".
    let mut out = platform::versioned_library_candidates("nvinfer", &["10", "9", "8"]);
    // TensorRT's Windows DLLs are `nvinfer_10.dll` (no `64`), unlike the
    // `cublas64_12.dll` convention the helper above spells.
    if cfg!(windows) {
        out.splice(
            0..0,
            ["10", "9", "8"].map(|major| format!("nvinfer_{major}.dll")),
        );
    }
    out
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
    create_infer_builder as "createInferBuilder_INTERNAL": PFN_createInferBuilder;
    get_plugin_registry as "getPluginRegistry": PFN_getPluginRegistry;
}

/// `createNvOnnxParser_INTERNAL(void* network, void* logger, int version)`, the
/// `extern "C"` factory exported by the ONNX parser library.
pub type PFN_createNvOnnxParser = unsafe extern "C" fn(
    network: trtINetworkDefinition_t,
    logger: trtILogger_t,
    version: c_int,
) -> trtIParser_t;

/// `NV_ONNX_PARSER_VERSION` from `NvOnnxParser.h` (0.1.0 encoded as `1 * 100`).
pub const NV_ONNX_PARSER_VERSION: c_int = 100;

/// Loader handle for the ONNX parser library (`nvonnxparser_10.dll` /
/// `libnvonnxparser.so.10`), which ships next to `libnvinfer`.
pub struct OnnxParserLib {
    _lib: Library,
    create: PFN_createNvOnnxParser,
}

impl core::fmt::Debug for OnnxParserLib {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("OnnxParserLib").finish_non_exhaustive()
    }
}

impl OnnxParserLib {
    /// The parser factory.
    pub fn create_parser(&self) -> PFN_createNvOnnxParser {
        self.create
    }
}

fn onnx_parser_candidates() -> Vec<&'static str> {
    // TensorRT 10 ships libnvonnxparser.so.10 / nvonnxparser_10.dll.
    if cfg!(windows) {
        vec!["nvonnxparser_10.dll", "nvonnxparser.dll"]
    } else {
        vec!["libnvonnxparser.so.10", "libnvonnxparser.so"]
    }
}

/// Resolve and return the process-wide ONNX parser library.
pub fn onnx_parser() -> Result<&'static OnnxParserLib, LoaderError> {
    static LIB: OnceLock<OnnxParserLib> = OnceLock::new();
    if let Some(l) = LIB.get() {
        return Ok(l);
    }
    let candidates: &'static [&'static str] =
        Box::leak(onnx_parser_candidates().into_boxed_slice());
    let lib = Library::open("nvonnxparser", candidates)?;
    let raw: *mut () = unsafe { lib.raw_symbol("createNvOnnxParser_INTERNAL")? };
    let create: PFN_createNvOnnxParser =
        unsafe { core::mem::transmute_copy::<*mut (), PFN_createNvOnnxParser>(&raw) };
    let _ = LIB.set(OnnxParserLib { _lib: lib, create });
    Ok(LIB.get().expect("OnceLock set or lost race"))
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
