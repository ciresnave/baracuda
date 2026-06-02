//! Safe-ish TensorRT runtime-side bindings.
//!
//! Scope is inference only: load a serialized engine (e.g. produced by
//! `trtexec` or the TRT Python API), build an execution context, bind tensor
//! addresses, and enqueue execution on a CUDA stream. Engine construction
//! (the builder / network definition API) is C++-only and is not wrapped here.
//!
//! # How it links: dynamic libnvinfer + a C++ shim
//!
//! **TensorRT exposes no flat C ABI.** Its public headers (`NvInfer.h`,
//! `NvInferRuntime.h`, …) are C++-only; the only `extern "C"` exports in
//! `libnvinfer` are `createInferRuntime_INTERNAL` and `getInferLibVersion`.
//! Those two are resolved at runtime via [`libloading`](baracuda_core) (no
//! link-time dependency on TensorRT). Everything else on the runtime path
//! (deserialize, engine/context getters, tensor binding, `enqueueV3`, …) is a
//! C++ vtable method with no flat symbol; it is reached through a small C++
//! shim (`baracuda-tensorrt-sys/shim/trt_shim.cpp`) that forwards flat `trt*`
//! symbols to the C++ API. The shim does pure vtable dispatch on the pointers
//! handed in from Rust, so it references no libnvinfer symbol — `libnvinfer`
//! stays dynamically loaded at runtime.
//!
//! The shim is compiled (and statically linked) only when the `shim` feature
//! is enabled on `baracuda-tensorrt-sys`, which requires the TensorRT SDK
//! headers at build time. Without it, [`version`] and [`Runtime`] construction
//! still work, but [`Runtime::deserialize_engine`] returns
//! [`Error::ShimNotBuilt`] — query [`shim_built`] to detect this. See
//! `crates/baracuda-tensorrt/AUDIT.md` for the full symbol map and the build
//! requirement.

#![warn(missing_debug_implementations, rust_2018_idioms)]

use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::os::raw::c_void;

use baracuda_cuda_sys::runtime::cudaStream_t;
use baracuda_driver::Stream;
use baracuda_tensorrt_sys as sys;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("TensorRT loader: {0}")]
    Loader(#[from] baracuda_core::LoaderError),
    #[error("TensorRT returned null for {op}")]
    NullHandle { op: &'static str },
    #[error("TensorRT call failed: {op}")]
    Call { op: &'static str },
    #[error("invalid C string: {0}")]
    Utf8(#[from] std::ffi::NulError),
    #[error(
        "TensorRT C-ABI shim not built — rebuild baracuda-tensorrt-sys with the `shim` \
         feature (needs the TensorRT SDK headers; see crates/baracuda-tensorrt/AUDIT.md)"
    )]
    ShimNotBuilt,
}

pub type Result<T> = std::result::Result<T, Error>;

/// Whether the C++ runtime shim was compiled in (the `shim` feature on
/// `baracuda-tensorrt-sys`). When `false`, only [`version`] and [`Runtime`]
/// construction work; deserializing an engine returns [`Error::ShimNotBuilt`].
pub fn shim_built() -> bool {
    sys::SHIM_BUILT
}

pub use sys::{
    trtDataType_t as DataType, trtExecutionContextAllocationStrategy_t as AllocStrategy,
    trtSeverity_t as Severity, trtTensorIOMode_t as IoMode,
};

/// TensorRT library version, encoded as `MAJOR * 1000 + MINOR * 100 + PATCH`.
pub fn version() -> Result<i32> {
    let t = sys::tensorrt()?;
    Ok(unsafe { t.get_infer_lib_version()?() })
}

/// A dimension list up to 8 axes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Dims {
    pub dims: [i64; sys::TRT_MAX_DIMS],
    pub rank: usize,
}

impl Dims {
    pub fn new(dims: &[i64]) -> Self {
        let mut out = Dims {
            dims: [0; sys::TRT_MAX_DIMS],
            rank: dims.len().min(sys::TRT_MAX_DIMS),
        };
        out.dims[..out.rank].copy_from_slice(&dims[..out.rank]);
        out
    }
    pub fn as_slice(&self) -> &[i64] {
        &self.dims[..self.rank]
    }
    fn to_raw(self) -> sys::trtDims_t {
        sys::trtDims_t {
            nb_dims: self.rank as i32,
            d: self.dims,
        }
    }
    fn from_raw(raw: sys::trtDims_t) -> Self {
        let mut out = Dims {
            dims: [0; sys::TRT_MAX_DIMS],
            rank: raw.nb_dims.max(0) as usize,
        };
        for i in 0..out.rank {
            out.dims[i] = raw.d[i];
        }
        out
    }
}

/// Owned TensorRT runtime. Created around a user-supplied logger (the logger
/// pointer is passed verbatim; safety responsibility is on the caller).
#[derive(Debug)]
pub struct Runtime {
    raw: sys::trtIRuntime_t,
}

impl Runtime {
    /// # Safety
    /// `logger` must be a valid `nvinfer1::ILogger*` (typically obtained from
    /// C++ or passed through a thin shim). Use [`Runtime::with_null_logger`]
    /// if no logging is desired (TRT allows `nullptr` in recent versions).
    pub unsafe fn new(logger: sys::trtILogger_t) -> Result<Self> { unsafe {
        let t = sys::tensorrt()?;
        // `createInferRuntime_INTERNAL(logger, version)` — the version must
        // match the loaded library (the C++ inline wrapper passes
        // NV_TENSORRT_VERSION). Use the runtime's own reported version.
        let version = (t.get_infer_lib_version()?)();
        let raw = (t.create_infer_runtime()?)(logger, version);
        if raw.is_null() {
            return Err(Error::NullHandle {
                op: "createInferRuntime",
            });
        }
        Ok(Self { raw })
    }}

    /// Construct without a logger. Supported on recent TensorRT; older
    /// versions may refuse and return null.
    pub fn with_null_logger() -> Result<Self> {
        unsafe { Self::new(core::ptr::null_mut()) }
    }

    /// Deserialize a serialized TensorRT engine blob (produced by `trtexec`,
    /// the TRT Python API, or [`Engine::serialize`]) into a runnable
    /// [`Engine`] borrowed from this runtime.
    pub fn deserialize_engine(&self, blob: &[u8]) -> Result<Engine<'_>> {
        // Engine deserialization is the gateway to the whole runtime path; it
        // (and everything reachable from the returned Engine) is provided by
        // the C++ shim. Fail clearly here rather than returning a null handle.
        if !sys::SHIM_BUILT {
            return Err(Error::ShimNotBuilt);
        }
        let raw = unsafe {
            sys::trtRuntimeDeserializeCudaEngine(
                self.raw,
                blob.as_ptr() as *const c_void,
                blob.len(),
            )
        };
        if raw.is_null() {
            return Err(Error::NullHandle {
                op: "deserializeCudaEngine",
            });
        }
        Ok(Engine {
            raw,
            _owner: PhantomData,
        })
    }

    /// Alias for [`Runtime::deserialize_engine`] kept for source-compat.
    #[inline]
    pub fn deserialize(&self, blob: &[u8]) -> Result<Engine<'_>> {
        self.deserialize_engine(blob)
    }

    pub fn as_raw(&self) -> sys::trtIRuntime_t {
        self.raw
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        // `trtRuntimeDestroy` is a shim symbol (a no-op stub when the shim is
        // absent — in which case `self.raw` was never a real runtime anyway).
        unsafe { sys::trtRuntimeDestroy(self.raw) };
    }
}

/// Represents a deserialized TensorRT engine borrowed from its parent runtime.
#[derive(Debug)]
pub struct Engine<'rt> {
    raw: sys::trtICudaEngine_t,
    _owner: PhantomData<&'rt Runtime>,
}

impl Engine<'_> {
    pub fn as_raw(&self) -> sys::trtICudaEngine_t {
        self.raw
    }

    pub fn num_io_tensors(&self) -> Result<i32> {
        Ok(unsafe { sys::trtCudaEngineGetNbIOTensors(self.raw) })
    }

    pub fn io_tensor_name(&self, index: i32) -> Result<String> {
        let cstr = unsafe { sys::trtCudaEngineGetIOTensorName(self.raw, index) };
        if cstr.is_null() {
            return Err(Error::NullHandle {
                op: "getIOTensorName",
            });
        }
        Ok(unsafe { CStr::from_ptr(cstr) }.to_string_lossy().into_owned())
    }

    pub fn tensor_io_mode(&self, name: &str) -> Result<IoMode> {
        let c = CString::new(name)?;
        Ok(unsafe { sys::trtCudaEngineGetTensorIOMode(self.raw, c.as_ptr()) })
    }

    pub fn tensor_data_type(&self, name: &str) -> Result<DataType> {
        let c = CString::new(name)?;
        Ok(unsafe { sys::trtCudaEngineGetTensorDataType(self.raw, c.as_ptr()) })
    }

    pub fn tensor_shape(&self, name: &str) -> Result<Dims> {
        let c = CString::new(name)?;
        let raw = unsafe { sys::trtCudaEngineGetTensorShape(self.raw, c.as_ptr()) };
        Ok(Dims::from_raw(raw))
    }

    /// Bytes per vectorized component for a tensor (1 for non-vectorized
    /// formats). Useful when computing device buffer sizes for bindings.
    pub fn tensor_bytes_per_component(&self, name: &str) -> Result<i32> {
        let c = CString::new(name)?;
        Ok(unsafe { sys::trtCudaEngineGetTensorBytesPerComponent(self.raw, c.as_ptr()) })
    }

    pub fn create_execution_context(&self) -> Result<ExecutionContext<'_>> {
        let raw = unsafe { sys::trtCudaEngineCreateExecutionContext(self.raw) };
        if raw.is_null() {
            return Err(Error::NullHandle {
                op: "createExecutionContext",
            });
        }
        Ok(ExecutionContext {
            raw,
            _owner: PhantomData,
        })
    }

    /// Create an execution context with a user-chosen allocation strategy.
    /// Use [`AllocStrategy::UserManaged`] when you intend to supply a
    /// scratch-allocator yourself; [`AllocStrategy::Static`] preallocates
    /// the maximum workspace at context-creation time.
    pub fn create_execution_context_with_strategy(
        &self,
        strategy: AllocStrategy,
    ) -> Result<ExecutionContext<'_>> {
        let raw = unsafe {
            sys::trtCudaEngineCreateExecutionContextWithStrategy(self.raw, strategy as i32)
        };
        if raw.is_null() {
            return Err(Error::NullHandle {
                op: "createExecutionContextWithStrategy",
            });
        }
        Ok(ExecutionContext {
            raw,
            _owner: PhantomData,
        })
    }

    /// Engine name as set in the TensorRT builder.
    pub fn name(&self) -> Result<String> {
        let cstr = unsafe { sys::trtCudaEngineGetName(self.raw) };
        if cstr.is_null() {
            return Err(Error::NullHandle { op: "engineGetName" });
        }
        Ok(unsafe { CStr::from_ptr(cstr) }
            .to_string_lossy()
            .into_owned())
    }

    /// Number of optimization profiles that were baked into the engine.
    pub fn num_optimization_profiles(&self) -> Result<i32> {
        Ok(unsafe { sys::trtCudaEngineGetNbOptimizationProfiles(self.raw) })
    }

    /// Serialize this engine back into a byte blob you can round-trip to
    /// disk. The returned [`HostMemory`] owns TensorRT-allocated storage;
    /// use [`HostMemory::as_slice`] to copy.
    pub fn serialize(&self) -> Result<HostMemory> {
        let raw = unsafe { sys::trtCudaEngineSerialize(self.raw) };
        if raw.is_null() {
            return Err(Error::NullHandle {
                op: "engineSerialize",
            });
        }
        Ok(HostMemory { raw })
    }
}

/// TensorRT-owned host buffer (as returned by [`Engine::serialize`]).
#[derive(Debug)]
pub struct HostMemory {
    raw: sys::trtIHostMemory_t,
}

impl HostMemory {
    pub fn len(&self) -> Result<usize> {
        Ok(unsafe { sys::trtHostMemorySize(self.raw) })
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    pub fn as_slice(&self) -> Result<&[u8]> {
        let ptr = unsafe { sys::trtHostMemoryData(self.raw) };
        let len = self.len()?;
        if ptr.is_null() || len == 0 {
            return Ok(&[]);
        }
        Ok(unsafe { core::slice::from_raw_parts(ptr as *const u8, len) })
    }
}

impl Drop for HostMemory {
    fn drop(&mut self) {
        unsafe { sys::trtHostMemoryDestroy(self.raw) };
    }
}

impl Drop for Engine<'_> {
    fn drop(&mut self) {
        unsafe { sys::trtCudaEngineDestroy(self.raw) };
    }
}

#[derive(Debug)]
pub struct ExecutionContext<'e> {
    raw: sys::trtIExecutionContext_t,
    _owner: PhantomData<&'e Engine<'e>>,
}

impl ExecutionContext<'_> {
    pub fn as_raw(&self) -> sys::trtIExecutionContext_t {
        self.raw
    }

    pub fn set_input_shape(&self, name: &str, dims: Dims) -> Result<()> {
        let c = CString::new(name)?;
        let raw_dims = dims.to_raw();
        let ok = unsafe { sys::trtExecutionContextSetInputShape(self.raw, c.as_ptr(), &raw_dims) };
        if !ok {
            return Err(Error::Call {
                op: "setInputShape",
            });
        }
        Ok(())
    }

    /// Bind a device pointer to a named input/output tensor on this
    /// execution context. The pointer is forwarded to TensorRT's
    /// `setTensorAddress`, which uses it during `enqueueV3` execution.
    ///
    /// # Safety
    ///
    /// `addr` must point to device memory that:
    /// - is large enough for the tensor's bound shape and data type, and
    /// - remains valid (not freed, not unmapped) for the duration of any
    ///   `enqueueV3` call that runs after this binding and before the
    ///   stream completes.
    pub unsafe fn set_tensor_address(&self, name: &str, addr: *mut c_void) -> Result<()> {
        let c = CString::new(name)?;
        let ok = unsafe { sys::trtExecutionContextSetTensorAddress(self.raw, c.as_ptr(), addr) };
        if !ok {
            return Err(Error::Call {
                op: "setTensorAddress",
            });
        }
        Ok(())
    }

    pub fn tensor_shape(&self, name: &str) -> Result<Dims> {
        let c = CString::new(name)?;
        let raw = unsafe { sys::trtExecutionContextGetTensorShape(self.raw, c.as_ptr()) };
        Ok(Dims::from_raw(raw))
    }

    /// Read the current bound device address for a tensor (null if unset).
    pub fn tensor_address(&self, name: &str) -> Result<*mut c_void> {
        let c = CString::new(name)?;
        Ok(unsafe { sys::trtExecutionContextGetTensorAddress(self.raw, c.as_ptr()) })
    }

    /// Enqueue the inference on a baracuda [`Stream`]. This is the preferred
    /// entry point — it accepts the same [`Stream`] type the rest of the
    /// baracuda stack uses, so callers never touch a raw `cudaStream_t`.
    ///
    /// Returns Ok if TRT reports success; the stream is still responsible for
    /// ordering, and the caller must ensure all tensor addresses have been set.
    ///
    /// # Safety
    /// All device pointers bound via [`set_tensor_address`] must still point to
    /// valid device memory of sufficient size for the bound shapes, and must
    /// remain valid until the stream completes this enqueue. (The [`Stream`]
    /// itself is valid by construction.)
    ///
    /// [`set_tensor_address`]: ExecutionContext::set_tensor_address
    pub unsafe fn enqueue_v3(&self, stream: &Stream) -> Result<()> {
        // CUstream and cudaStream_t are both `*mut CUstream_st`; the cast is a
        // no-op reinterpret (same pattern as baracuda-cudnn's `set_stream`).
        unsafe { self.enqueue_v3_raw(stream.as_raw() as cudaStream_t) }
    }

    /// Enqueue on a raw `cudaStream_t`. Prefer [`enqueue_v3`] with a baracuda
    /// [`Stream`]; this lower-level form exists for callers that already hold a
    /// raw runtime-API stream handle from elsewhere.
    ///
    /// # Safety
    /// In addition to the tensor-address requirements of [`enqueue_v3`],
    /// `stream` must be a valid `cudaStream_t` that outlives the enqueue.
    ///
    /// [`enqueue_v3`]: ExecutionContext::enqueue_v3
    pub unsafe fn enqueue_v3_raw(&self, stream: cudaStream_t) -> Result<()> {
        let ok = unsafe { sys::trtExecutionContextEnqueueV3(self.raw, stream) };
        if !ok {
            return Err(Error::Call { op: "enqueueV3" });
        }
        Ok(())
    }
}

impl Drop for ExecutionContext<'_> {
    fn drop(&mut self) {
        unsafe { sys::trtExecutionContextDestroy(self.raw) };
    }
}
