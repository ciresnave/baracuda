//! Safe-ish TensorRT runtime-side bindings.
//!
//! Scope is inference only: load a serialized engine (e.g. produced by
//! `trtexec` or the TRT Python API), build an execution context, bind tensor
//! addresses, and enqueue execution on a CUDA stream. Engine construction
//! (the builder / network definition API) is C++-only and is not wrapped here.

#![warn(missing_debug_implementations, rust_2018_idioms)]

use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::os::raw::c_void;

use baracuda_cuda_sys::runtime::cudaStream_t;
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
}

pub type Result<T> = std::result::Result<T, Error>;

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
    pub unsafe fn new(logger: sys::trtILogger_t) -> Result<Self> {
        let t = sys::tensorrt()?;
        let raw = (t.create_infer_runtime()?)(logger);
        if raw.is_null() {
            return Err(Error::NullHandle {
                op: "createInferRuntime",
            });
        }
        Ok(Self { raw })
    }

    /// Construct without a logger. Supported on recent TensorRT; older
    /// versions may refuse and return null.
    pub fn with_null_logger() -> Result<Self> {
        unsafe { Self::new(core::ptr::null_mut()) }
    }

    pub fn deserialize(&self, blob: &[u8]) -> Result<Engine<'_>> {
        let t = sys::tensorrt()?;
        let raw = unsafe {
            (t.deserialize_cuda_engine()?)(self.raw, blob.as_ptr() as *const c_void, blob.len())
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

    pub fn as_raw(&self) -> sys::trtIRuntime_t {
        self.raw
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        if let Ok(t) = sys::tensorrt() {
            if let Ok(f) = t.destroy_infer_runtime() {
                unsafe { f(self.raw) };
            }
        }
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
        let t = sys::tensorrt()?;
        Ok(unsafe { (t.engine_get_nb_io_tensors()?)(self.raw) })
    }

    pub fn io_tensor_name(&self, index: i32) -> Result<String> {
        let t = sys::tensorrt()?;
        let cstr = unsafe { (t.engine_get_io_tensor_name()?)(self.raw, index) };
        if cstr.is_null() {
            return Err(Error::NullHandle {
                op: "getIOTensorName",
            });
        }
        Ok(unsafe { CStr::from_ptr(cstr) }.to_string_lossy().into_owned())
    }

    pub fn tensor_io_mode(&self, name: &str) -> Result<IoMode> {
        let t = sys::tensorrt()?;
        let c = CString::new(name)?;
        Ok(unsafe { (t.engine_get_tensor_io_mode()?)(self.raw, c.as_ptr()) })
    }

    pub fn tensor_data_type(&self, name: &str) -> Result<DataType> {
        let t = sys::tensorrt()?;
        let c = CString::new(name)?;
        Ok(unsafe { (t.engine_get_tensor_data_type()?)(self.raw, c.as_ptr()) })
    }

    pub fn tensor_shape(&self, name: &str) -> Result<Dims> {
        let t = sys::tensorrt()?;
        let c = CString::new(name)?;
        let raw = unsafe { (t.engine_get_tensor_shape()?)(self.raw, c.as_ptr()) };
        Ok(Dims::from_raw(raw))
    }

    pub fn create_execution_context(&self) -> Result<ExecutionContext<'_>> {
        let t = sys::tensorrt()?;
        let raw = unsafe { (t.engine_create_execution_context()?)(self.raw) };
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
        let t = sys::tensorrt()?;
        let raw = unsafe {
            (t.engine_create_execution_context_with_strategy()?)(self.raw, strategy)
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
        let t = sys::tensorrt()?;
        let cstr = unsafe { (t.engine_get_name()?)(self.raw) };
        if cstr.is_null() {
            return Err(Error::NullHandle { op: "engineGetName" });
        }
        Ok(unsafe { CStr::from_ptr(cstr) }
            .to_string_lossy()
            .into_owned())
    }

    /// Number of optimization profiles that were baked into the engine.
    pub fn num_optimization_profiles(&self) -> Result<i32> {
        let t = sys::tensorrt()?;
        Ok(unsafe { (t.engine_get_nb_optimization_profiles()?)(self.raw) })
    }

    /// Serialize this engine back into a byte blob you can round-trip to
    /// disk. The returned [`HostMemory`] owns TensorRT-allocated storage;
    /// use [`HostMemory::as_slice`] to copy.
    pub fn serialize(&self) -> Result<HostMemory> {
        let t = sys::tensorrt()?;
        let raw = unsafe { (t.engine_serialize()?)(self.raw) };
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
        let t = sys::tensorrt()?;
        Ok(unsafe { (t.host_memory_size()?)(self.raw) })
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    pub fn as_slice(&self) -> Result<&[u8]> {
        let t = sys::tensorrt()?;
        let ptr = unsafe { (t.host_memory_data()?)(self.raw) };
        let len = self.len()?;
        if ptr.is_null() || len == 0 {
            return Ok(&[]);
        }
        Ok(unsafe { core::slice::from_raw_parts(ptr as *const u8, len) })
    }
}

impl Drop for HostMemory {
    fn drop(&mut self) {
        if let Ok(t) = sys::tensorrt() {
            if let Ok(d) = t.host_memory_destroy() {
                unsafe { d(self.raw) };
            }
        }
    }
}

impl Drop for Engine<'_> {
    fn drop(&mut self) {
        if let Ok(t) = sys::tensorrt() {
            if let Ok(f) = t.destroy_cuda_engine() {
                unsafe { f(self.raw) };
            }
        }
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
        let t = sys::tensorrt()?;
        let c = CString::new(name)?;
        let raw_dims = dims.to_raw();
        let ok = unsafe { (t.context_set_input_shape()?)(self.raw, c.as_ptr(), &raw_dims) };
        if !ok {
            return Err(Error::Call {
                op: "setInputShape",
            });
        }
        Ok(())
    }

    pub fn set_tensor_address(&self, name: &str, addr: *mut c_void) -> Result<()> {
        let t = sys::tensorrt()?;
        let c = CString::new(name)?;
        let ok = unsafe { (t.context_set_tensor_address()?)(self.raw, c.as_ptr(), addr) };
        if !ok {
            return Err(Error::Call {
                op: "setTensorAddress",
            });
        }
        Ok(())
    }

    pub fn tensor_shape(&self, name: &str) -> Result<Dims> {
        let t = sys::tensorrt()?;
        let c = CString::new(name)?;
        let raw = unsafe { (t.context_get_tensor_shape()?)(self.raw, c.as_ptr()) };
        Ok(Dims::from_raw(raw))
    }

    /// Read the current bound device address for a tensor (null if unset).
    pub fn tensor_address(&self, name: &str) -> Result<*mut c_void> {
        let t = sys::tensorrt()?;
        let c = CString::new(name)?;
        Ok(unsafe { (t.context_get_tensor_address()?)(self.raw, c.as_ptr()) })
    }

    /// Enqueue the inference on the given CUDA stream. Returns Ok if TRT
    /// reports success; the stream is still responsible for ordering, and the
    /// caller must ensure all tensor addresses have been set.
    ///
    /// # Safety
    /// `stream` must be a valid `cudaStream_t` that outlives the enqueue.
    pub unsafe fn enqueue_v3(&self, stream: cudaStream_t) -> Result<()> {
        let t = sys::tensorrt()?;
        let ok = (t.context_enqueue_v3()?)(self.raw, stream);
        if !ok {
            return Err(Error::Call { op: "enqueueV3" });
        }
        Ok(())
    }
}

impl Drop for ExecutionContext<'_> {
    fn drop(&mut self) {
        if let Ok(t) = sys::tensorrt() {
            if let Ok(f) = t.destroy_execution_context() {
                unsafe { f(self.raw) };
            }
        }
    }
}
