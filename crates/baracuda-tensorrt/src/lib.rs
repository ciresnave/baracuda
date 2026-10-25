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
use std::os::raw::{c_char, c_void};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use baracuda_cuda_sys::runtime::cudaStream_t;
use baracuda_driver::Stream;
use baracuda_tensorrt_sys as sys;

/// Error type for TensorRT operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The TensorRT dynamic library or one of its symbols could not be loaded.
    #[error("TensorRT loader: {0}")]
    Loader(#[from] baracuda_core::LoaderError),
    /// A TensorRT entry point returned a null handle (failed allocation /
    /// invalid arguments).
    #[error("TensorRT returned null for {op}")]
    NullHandle {
        /// Name of the entry point that returned null.
        op: &'static str,
    },
    /// A TensorRT entry point returned a false / non-success status.
    #[error("TensorRT call failed: {op}")]
    Call {
        /// Name of the entry point that failed.
        op: &'static str,
    },
    /// A Rust string contained an interior NUL byte and could not be passed to C.
    #[error("invalid C string: {0}")]
    Utf8(#[from] std::ffi::NulError),
    /// The TensorRT C-ABI shim was not compiled into `baracuda-tensorrt-sys`.
    /// See the crate-level docs for how to rebuild with the `shim` feature.
    #[error(
        "TensorRT C-ABI shim not built — rebuild baracuda-tensorrt-sys with the `shim` \
         feature (needs the TensorRT SDK headers; see crates/baracuda-tensorrt/AUDIT.md)"
    )]
    ShimNotBuilt,
}

/// Result alias for TensorRT operations.
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
    /// Fixed-size backing storage; only the first `rank` entries are meaningful.
    pub dims: [i64; sys::TRT_MAX_DIMS],
    /// Number of axes actually populated (`0..=TRT_MAX_DIMS`).
    pub rank: usize,
}

impl Dims {
    /// Construct from a Rust slice; extra axes are truncated, missing axes
    /// are zero-padded.
    pub fn new(dims: &[i64]) -> Self {
        let mut out = Dims {
            dims: [0; sys::TRT_MAX_DIMS],
            rank: dims.len().min(sys::TRT_MAX_DIMS),
        };
        out.dims[..out.rank].copy_from_slice(&dims[..out.rank]);
        out
    }
    /// View the populated axes as a slice.
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

type LogFn = dyn Fn(Severity, &str) + Send + Sync + 'static;

/// The loggers currently alive, newest last. The one process-wide `ILogger`
/// forwards every TensorRT message to all of them.
static LOGGERS: Mutex<Vec<(u64, Arc<LogFn>)>> = Mutex::new(Vec::new());
static NEXT_LOGGER_ID: AtomicU64 = AtomicU64::new(0);

/// The single C++ `ILogger` handed to TensorRT, created on first use and never
/// freed (stored as an address so the static is `Sync`).
static GLOBAL_ILOGGER: OnceLock<usize> = OnceLock::new();

unsafe extern "C" fn log_trampoline(severity: Severity, msg: *const c_char, _user: *mut c_void) {
    if msg.is_null() {
        return;
    }
    let text = unsafe { CStr::from_ptr(msg) }.to_string_lossy();
    // Snapshot under the lock and call outside it, so a callback that itself
    // creates or drops a `Logger` cannot deadlock.
    let snapshot: Vec<Arc<LogFn>> = match LOGGERS.lock() {
        Ok(guard) => guard.iter().map(|(_, f)| Arc::clone(f)).collect(),
        Err(poisoned) => poisoned
            .into_inner()
            .iter()
            .map(|(_, f)| Arc::clone(f))
            .collect(),
    };
    for callback in snapshot {
        let _ =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| callback(severity, &text)));
    }
}

fn global_ilogger() -> Result<sys::trtILogger_t> {
    if let Some(&addr) = GLOBAL_ILOGGER.get() {
        return Ok(addr as sys::trtILogger_t);
    }
    let raw = unsafe { sys::trtLoggerCreate(log_trampoline, core::ptr::null_mut()) };
    if raw.is_null() {
        return Err(Error::NullHandle {
            op: "trtLoggerCreate",
        });
    }
    // If two threads race, one `ILogger` is created twice; keep the winner and
    // free the loser (it was never handed to TensorRT).
    match GLOBAL_ILOGGER.set(raw as usize) {
        Ok(()) => Ok(raw),
        Err(_) => {
            unsafe { sys::trtLoggerDestroy(raw) };
            Ok(*GLOBAL_ILOGGER.get().expect("set by the racing thread") as sys::trtILogger_t)
        }
    }
}

/// A safe TensorRT logger: a Rust closure that receives TensorRT's messages.
///
/// TensorRT 10 refuses a null logger, so every [`Runtime`] and builder needs
/// one. TensorRT also keeps **one logger per process**: a second logger passed
/// to `createInferRuntime` / `createInferBuilder` is ignored while any builder,
/// runtime or refitter is alive (measured on 10.16: the "logger ... differs
/// from one already registered" warning). So all [`Logger`]s share one
/// process-wide C++ `ILogger`, created on first use and never freed, and every
/// message is delivered to every live [`Logger`]. Dropping a [`Logger`] stops
/// delivery to its closure and cannot leave TensorRT holding a dangling logger.
///
/// TensorRT may call it from any thread, hence the `Send + Sync` bound. A
/// panic in the closure is caught at the boundary (unwinding into C++ would be
/// undefined behaviour). Hand the logger to [`Runtime::with_logger`]; the
/// runtime keeps it alive for as long as it lives.
#[derive(Debug)]
pub struct Logger {
    id: u64,
    raw: sys::trtILogger_t,
}

// SAFETY: `raw` is the process-wide `ILogger*`, which is never freed and only
// read; the closures live in `LOGGERS` behind a mutex and are `Send + Sync`.
unsafe impl Send for Logger {}
unsafe impl Sync for Logger {}

impl Logger {
    /// Build a logger that calls `callback(severity, message)` for every
    /// TensorRT message. Needs the shim: returns [`Error::ShimNotBuilt`]
    /// without it.
    pub fn new(callback: impl Fn(Severity, &str) + Send + Sync + 'static) -> Result<Self> {
        if !sys::SHIM_BUILT {
            return Err(Error::ShimNotBuilt);
        }
        let raw = global_ilogger()?;
        let id = NEXT_LOGGER_ID.fetch_add(1, Ordering::Relaxed);
        let callback: Arc<LogFn> = Arc::new(callback);
        LOGGERS
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push((id, callback));
        Ok(Self { id, raw })
    }

    /// A logger that writes messages at or above `min` (more severe = lower
    /// number, so `Severity::Warning` keeps warnings, errors and internal
    /// errors) to standard error.
    pub fn stderr(min: Severity) -> Result<Self> {
        Self::new(move |severity, msg| {
            if (severity as i32) <= (min as i32) {
                eprintln!("[TensorRT {severity:?}] {msg}");
            }
        })
    }

    /// Raw `ILogger*`: the process-wide logger, valid for the life of the process.
    pub fn as_raw(&self) -> sys::trtILogger_t {
        self.raw
    }
}

impl Drop for Logger {
    fn drop(&mut self) {
        LOGGERS
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .retain(|(id, _)| *id != self.id);
    }
}

/// Owned TensorRT runtime. Built around a [`Logger`] ([`Runtime::with_logger`])
/// or, for callers who manage their own `ILogger*`, [`Runtime::new`].
#[derive(Debug)]
pub struct Runtime {
    raw: sys::trtIRuntime_t,
    // TensorRT keeps the logger pointer, so the logger must outlive the
    // runtime: `Drop for Runtime` destroys `raw` before this field drops.
    _logger: Option<Arc<Logger>>,
}

impl Runtime {
    /// Create a runtime that reports through `logger`. The runtime shares
    /// ownership of the logger, so it stays alive as long as the runtime does.
    pub fn with_logger(logger: Arc<Logger>) -> Result<Self> {
        let mut rt = unsafe { Self::new(logger.as_raw())? };
        rt._logger = Some(logger);
        Ok(rt)
    }

    /// # Safety
    /// `logger` must be a valid `nvinfer1::ILogger*` (typically obtained from
    /// C++ or passed through a thin shim). Use [`Runtime::with_null_logger`]
    /// if no logging is desired (TRT allows `nullptr` in recent versions).
    pub unsafe fn new(logger: sys::trtILogger_t) -> Result<Self> {
        unsafe {
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
            Ok(Self { raw, _logger: None })
        }
    }

    /// Construct without a logger. TensorRT 10 refuses a null logger and this
    /// returns [`Error::NullHandle`] there (measured against TensorRT 10.16);
    /// use [`Runtime::with_logger`].
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

    /// Raw `trtIRuntime_t`. Use with care.
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
    /// Raw `trtICudaEngine_t`. Use with care.
    pub fn as_raw(&self) -> sys::trtICudaEngine_t {
        self.raw
    }

    /// Number of I/O tensors the engine exposes (sum of inputs + outputs).
    pub fn num_io_tensors(&self) -> Result<i32> {
        Ok(unsafe { sys::trtCudaEngineGetNbIOTensors(self.raw) })
    }

    /// Look up the I/O tensor name at `index` (`0..num_io_tensors()`).
    pub fn io_tensor_name(&self, index: i32) -> Result<String> {
        let cstr = unsafe { sys::trtCudaEngineGetIOTensorName(self.raw, index) };
        if cstr.is_null() {
            return Err(Error::NullHandle {
                op: "getIOTensorName",
            });
        }
        Ok(unsafe { CStr::from_ptr(cstr) }
            .to_string_lossy()
            .into_owned())
    }

    /// Whether a named tensor is an input, output, or unused binding.
    pub fn tensor_io_mode(&self, name: &str) -> Result<IoMode> {
        let c = CString::new(name)?;
        Ok(unsafe { sys::trtCudaEngineGetTensorIOMode(self.raw, c.as_ptr()) })
    }

    /// Element data type for a named tensor.
    pub fn tensor_data_type(&self, name: &str) -> Result<DataType> {
        let c = CString::new(name)?;
        Ok(unsafe { sys::trtCudaEngineGetTensorDataType(self.raw, c.as_ptr()) })
    }

    /// Engine-time shape for a named tensor (may contain `-1` for dynamic axes).
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

    /// Wraps `engine->createExecutionContext()` with the default allocation
    /// strategy; the returned [`ExecutionContext`] is borrowed from this engine.
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
            return Err(Error::NullHandle {
                op: "engineGetName",
            });
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
    /// Size of the buffer in bytes.
    pub fn len(&self) -> Result<usize> {
        Ok(unsafe { sys::trtHostMemorySize(self.raw) })
    }

    /// `true` if [`HostMemory::len`] is zero.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Borrow the buffer as a Rust byte slice.
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

/// A TensorRT execution context — the per-call state needed to actually run
/// inference against an [`Engine`]. Borrows the engine for its lifetime.
#[derive(Debug)]
pub struct ExecutionContext<'e> {
    raw: sys::trtIExecutionContext_t,
    _owner: PhantomData<&'e Engine<'e>>,
}

impl ExecutionContext<'_> {
    /// Raw `trtIExecutionContext_t`. Use with care.
    pub fn as_raw(&self) -> sys::trtIExecutionContext_t {
        self.raw
    }

    /// Set the runtime shape of an input tensor. Required before
    /// [`Self::enqueue_v3`] when the engine has dynamic input shapes.
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

    /// Read the currently-bound shape for a named tensor on this context
    /// (post-`set_input_shape` for inputs; engine-time shape for outputs).
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
