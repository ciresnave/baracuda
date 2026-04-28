//! Compiled module loading (PTX, CUBIN, fatbin) and kernel entry-point lookup.

use core::ffi::{c_char, c_void};
use std::ffi::CString;
use std::sync::Arc;

use baracuda_cuda_sys::{driver, CUdeviceptr, CUfunction, CUmodule};

use crate::context::Context;
use crate::error::{check, Result};

/// A loaded CUDA module (e.g. compiled PTX).
#[derive(Clone)]
pub struct Module {
    inner: Arc<ModuleInner>,
}

struct ModuleInner {
    handle: CUmodule,
    context: Context,
}

unsafe impl Send for ModuleInner {}
unsafe impl Sync for ModuleInner {}

impl core::fmt::Debug for ModuleInner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Module")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

impl core::fmt::Debug for Module {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.inner.fmt(f)
    }
}

impl Module {
    /// Load a module from a raw binary image (CUBIN, fatbin, or PTX text with a trailing NUL).
    ///
    /// For PTX, the bytes must be a null-terminated UTF-8 string matching the
    /// `ptx` file on disk. [`Module::load_ptx`] is a convenience wrapper that
    /// adds the NUL for you.
    pub fn load_raw(context: &Context, image: &[u8]) -> Result<Self> {
        context.set_current()?;
        let d = driver()?;
        let cu = d.cu_module_load_data()?;
        let mut module: CUmodule = core::ptr::null_mut();
        // SAFETY: `image.as_ptr()` is valid for reads of the image bytes.
        check(unsafe { cu(&mut module, image.as_ptr() as *const c_void) })?;
        Ok(Self {
            inner: Arc::new(ModuleInner {
                handle: module,
                context: context.clone(),
            }),
        })
    }

    /// Load a module from a PTX source string.
    pub fn load_ptx(context: &Context, ptx_source: &str) -> Result<Self> {
        // cuModuleLoadData expects a null-terminated buffer for PTX.
        let c_src = CString::new(ptx_source).map_err(|_| {
            crate::error::Error::Loader(baracuda_core::LoaderError::SymbolNotFound {
                library: "cuda-driver",
                symbol: "cuModuleLoadData(PTX input contained a NUL byte)",
            })
        })?;
        Self::load_raw(context, c_src.as_bytes_with_nul())
    }

    /// Look up a `__device__` global variable by name. Returns
    /// `(device_ptr, size_in_bytes)`.
    pub fn get_global(&self, name: &str) -> Result<(CUdeviceptr, usize)> {
        let d = driver()?;
        let cu = d.cu_module_get_global()?;
        let c_name = CString::new(name).map_err(|_| {
            crate::error::Error::Loader(baracuda_core::LoaderError::SymbolNotFound {
                library: "cuda-driver",
                symbol: "cuModuleGetGlobal(name contained a NUL byte)",
            })
        })?;
        let mut dptr = CUdeviceptr(0);
        let mut bytes: usize = 0;
        check(unsafe {
            cu(
                &mut dptr,
                &mut bytes,
                self.inner.handle,
                c_name.as_ptr() as *const c_char,
            )
        })?;
        Ok((dptr, bytes))
    }

    /// Look up a kernel entry point by name.
    pub fn get_function(&self, name: &str) -> Result<Function> {
        let d = driver()?;
        let cu = d.cu_module_get_function()?;
        let c_name = CString::new(name).map_err(|_| {
            crate::error::Error::Loader(baracuda_core::LoaderError::SymbolNotFound {
                library: "cuda-driver",
                symbol: "cuModuleGetFunction(kernel name contained a NUL byte)",
            })
        })?;
        let mut func: CUfunction = core::ptr::null_mut();
        // SAFETY: `func` writable; `self.inner.handle` owned by this Arc;
        // `c_name.as_ptr()` is null-terminated.
        check(unsafe {
            cu(
                &mut func,
                self.inner.handle,
                c_name.as_ptr() as *const c_char,
            )
        })?;
        Ok(Function {
            handle: func,
            _owner: FunctionOwner::Module(self.clone()),
        })
    }

    /// Raw `CUmodule` handle. Use with care.
    #[inline]
    pub fn as_raw(&self) -> CUmodule {
        self.inner.handle
    }

    /// Return the current process-wide module loading mode (eager vs. lazy).
    /// Compare against
    /// [`baracuda_cuda_sys::types::CUmoduleLoadingMode`] constants.
    pub fn loading_mode() -> Result<i32> {
        let d = driver()?;
        let cu = d.cu_module_get_loading_mode()?;
        let mut mode: core::ffi::c_int = 0;
        check(unsafe { cu(&mut mode) })?;
        Ok(mode)
    }

    /// Load a module from a raw image with extra JIT compiler options —
    /// the typical use is capturing the JIT log when a PTX module
    /// fails to compile. `options` and `option_values` are parallel
    /// arrays whose entries follow the `CUjit_option` ABI (see the
    /// CUDA driver reference). For PTX, the bytes must be a
    /// null-terminated UTF-8 string.
    ///
    /// # Safety
    ///
    /// Each `option_value` must point at a value of the type the
    /// matching `CUjit_option` expects (some are pointers, some are
    /// integers cast to `*mut c_void`). The arrays must have the same
    /// length.
    pub unsafe fn load_data_ex(
        context: &Context,
        image: &[u8],
        options: &mut [i32],
        option_values: &mut [*mut core::ffi::c_void],
    ) -> Result<Self> {
        assert_eq!(
            options.len(),
            option_values.len(),
            "load_data_ex: options and option_values must have the same length"
        );
        context.set_current()?;
        let d = driver()?;
        let cu = d.cu_module_load_data_ex()?;
        let mut module: CUmodule = core::ptr::null_mut();
        check(cu(
            &mut module,
            image.as_ptr() as *const c_void,
            options.len() as core::ffi::c_uint,
            options.as_mut_ptr(),
            option_values.as_mut_ptr(),
        ))?;
        Ok(Self {
            inner: Arc::new(ModuleInner {
                handle: module,
                context: context.clone(),
            }),
        })
    }

    /// The [`Context`] this module was loaded into.
    #[inline]
    pub fn context(&self) -> &Context {
        &self.inner.context
    }
}

impl Drop for ModuleInner {
    fn drop(&mut self) {
        if let Ok(d) = driver() {
            if let Ok(cu) = d.cu_module_unload() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

/// A kernel entry point — either inside a [`Module`] (classic
/// Driver API) or materialized from a [`crate::library::Kernel`] (CUDA
/// 12.0+ library API). Either way it keeps the parent alive via an Arc
/// so the kernel stays valid for as long as any [`Function`] handle exists.
#[derive(Clone, Debug)]
pub struct Function {
    handle: CUfunction,
    _owner: FunctionOwner,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
enum FunctionOwner {
    /// Owned by a `Module` (classic Driver API flow).
    Module(Module),
    /// Owned by a `Library` (CUDA 12.0+ cuLibrary flow).
    Library(crate::library::Library),
}

impl Function {
    /// Construct from an already-resolved `CUfunction` plus the parent
    /// library that owns it. Intended for `library::Kernel`'s
    /// `function_for_current_context`.
    pub(crate) fn from_raw_with_library(
        handle: CUfunction,
        library: crate::library::Library,
    ) -> Self {
        Self {
            handle,
            _owner: FunctionOwner::Library(library),
        }
    }
}

unsafe impl Send for Function {}
unsafe impl Sync for Function {}

impl Function {
    /// Raw `CUfunction`. Use with care.
    #[inline]
    pub fn as_raw(&self) -> CUfunction {
        self.handle
    }

    /// The [`Module`] this kernel lives in, if it was obtained through
    /// `Module::get_function`. Returns `None` for kernels materialized
    /// from a `library::Kernel`.
    #[inline]
    pub fn module(&self) -> Option<&Module> {
        match &self._owner {
            FunctionOwner::Module(m) => Some(m),
            FunctionOwner::Library(_) => None,
        }
    }

    /// Query a kernel attribute (see
    /// [`baracuda_cuda_sys::types::CUfunction_attribute`]).
    pub fn get_attribute(&self, attribute: i32) -> Result<i32> {
        let d = driver()?;
        let cu = d.cu_func_get_attribute()?;
        let mut v: core::ffi::c_int = 0;
        check(unsafe { cu(&mut v, attribute, self.handle) })?;
        Ok(v)
    }

    /// Return the demangled kernel name reported by the driver.
    pub fn name(&self) -> Result<String> {
        let d = driver()?;
        let cu = d.cu_func_get_name()?;
        let mut p: *const core::ffi::c_char = core::ptr::null();
        check(unsafe { cu(&mut p, self.handle) })?;
        if p.is_null() {
            return Ok(String::new());
        }
        let cstr = unsafe { core::ffi::CStr::from_ptr(p) };
        Ok(cstr.to_string_lossy().into_owned())
    }

    /// Return `(offset_in_bytes, size_in_bytes)` for the `index`-th
    /// parameter in this function's ABI signature.
    pub fn param_info(&self, index: usize) -> Result<(usize, usize)> {
        let d = driver()?;
        let cu = d.cu_func_get_param_info()?;
        let mut off: usize = 0;
        let mut sz: usize = 0;
        check(unsafe { cu(self.handle, index, &mut off, &mut sz) })?;
        Ok((off, sz))
    }

    /// Return the raw `CUmodule` this function was loaded from, if any.
    pub fn module_raw(&self) -> Result<baracuda_cuda_sys::CUmodule> {
        let d = driver()?;
        let cu = d.cu_func_get_module()?;
        let mut m: baracuda_cuda_sys::CUmodule = core::ptr::null_mut();
        check(unsafe { cu(&mut m, self.handle) })?;
        Ok(m)
    }

    /// Set a kernel attribute. Only a subset is writable (notably
    /// `MAX_DYNAMIC_SHARED_SIZE_BYTES` and
    /// `PREFERRED_SHARED_MEMORY_CARVEOUT`).
    pub fn set_attribute(&self, attribute: i32, value: i32) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_func_set_attribute()?;
        check(unsafe { cu(self.handle, attribute, value) })
    }

    // Convenience named accessors for the most-read attributes.

    /// Maximum threads per block this kernel supports on the current device.
    pub fn max_threads_per_block(&self) -> Result<i32> {
        use baracuda_cuda_sys::types::CUfunction_attribute as A;
        self.get_attribute(A::MAX_THREADS_PER_BLOCK)
    }

    /// Size of per-block statically-allocated shared memory (bytes).
    pub fn shared_size_bytes(&self) -> Result<i32> {
        use baracuda_cuda_sys::types::CUfunction_attribute as A;
        self.get_attribute(A::SHARED_SIZE_BYTES)
    }

    /// Number of registers used per thread.
    pub fn num_regs(&self) -> Result<i32> {
        use baracuda_cuda_sys::types::CUfunction_attribute as A;
        self.get_attribute(A::NUM_REGS)
    }

    /// Per-thread local-memory footprint (bytes).
    pub fn local_size_bytes(&self) -> Result<i32> {
        use baracuda_cuda_sys::types::CUfunction_attribute as A;
        self.get_attribute(A::LOCAL_SIZE_BYTES)
    }

    /// PTX version this kernel was compiled from, as `major*10 + minor`.
    pub fn ptx_version(&self) -> Result<i32> {
        use baracuda_cuda_sys::types::CUfunction_attribute as A;
        self.get_attribute(A::PTX_VERSION)
    }

    /// SM-architecture this kernel was compiled for, as `major*10 + minor`.
    pub fn binary_version(&self) -> Result<i32> {
        use baracuda_cuda_sys::types::CUfunction_attribute as A;
        self.get_attribute(A::BINARY_VERSION)
    }
}
