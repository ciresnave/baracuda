//! Driver-API library + kernel management (CUDA 12.0+).
//!
//! Unlike [`crate::Module`] which loads into a specific [`crate::Context`],
//! a [`Library`] is context-independent — one `cuLibraryLoadData` call
//! produces a handle that can be queried for kernels from any context on
//! any device. This is the modern preferred way to ship precompiled
//! PTX / CUBIN / fatbin in a reusable form.
//!
//! Requires CUDA 12.0+ at runtime. On older drivers the loader reports
//! `LoaderError::SymbolNotFound` on first use.

use core::ffi::{c_char, c_void};
use std::ffi::CString;
use std::sync::Arc;

use baracuda_cuda_sys::{driver, CUdeviceptr, CUfunction, CUkernel, CUlibrary};

use crate::error::{check, Result};
use crate::module::Function;

/// A loaded CUDA library (CUDA 12.0+).
#[derive(Clone)]
pub struct Library {
    inner: Arc<LibraryInner>,
}

struct LibraryInner {
    handle: CUlibrary,
}

unsafe impl Send for LibraryInner {}
unsafe impl Sync for LibraryInner {}

impl core::fmt::Debug for LibraryInner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Library")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

impl core::fmt::Debug for Library {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.inner.fmt(f)
    }
}

impl Library {
    /// Load a library from a raw image (CUBIN, fatbin, or null-terminated PTX)
    /// with no JIT/library options.
    pub fn load_raw(image: &[u8]) -> Result<Self> {
        let d = driver()?;
        let cu = d.cu_library_load_data()?;
        let mut lib: CUlibrary = core::ptr::null_mut();
        check(unsafe {
            cu(
                &mut lib,
                image.as_ptr() as *const c_void,
                core::ptr::null_mut(), // jit_options
                core::ptr::null_mut(), // jit_option_values
                0,                     // num_jit_options
                core::ptr::null_mut(), // library_options
                core::ptr::null_mut(), // library_option_values
                0,                     // num_library_options
            )
        })?;
        Ok(Self {
            inner: Arc::new(LibraryInner { handle: lib }),
        })
    }

    /// Load a library from a PTX source string (NUL-terminates internally).
    pub fn load_ptx(ptx: &str) -> Result<Self> {
        let c_src = CString::new(ptx).map_err(|_| {
            crate::error::Error::Loader(baracuda_core::LoaderError::SymbolNotFound {
                library: "cuda-driver",
                symbol: "cuLibraryLoadData(PTX contained a NUL byte)",
            })
        })?;
        Self::load_raw(c_src.as_bytes_with_nul())
    }

    /// Look up a kernel entry point by name.
    pub fn get_kernel(&self, name: &str) -> Result<Kernel> {
        let d = driver()?;
        let cu = d.cu_library_get_kernel()?;
        let c_name = CString::new(name).map_err(|_| {
            crate::error::Error::Loader(baracuda_core::LoaderError::SymbolNotFound {
                library: "cuda-driver",
                symbol: "cuLibraryGetKernel(name contained a NUL byte)",
            })
        })?;
        let mut kernel: CUkernel = core::ptr::null_mut();
        check(unsafe {
            cu(
                &mut kernel,
                self.inner.handle,
                c_name.as_ptr() as *const c_char,
            )
        })?;
        Ok(Kernel {
            handle: kernel,
            _library: self.clone(),
        })
    }

    /// Count of kernels this library exposes (CUDA 12.4+).
    pub fn kernel_count(&self) -> Result<u32> {
        let d = driver()?;
        let cu = d.cu_library_get_kernel_count()?;
        let mut n: core::ffi::c_uint = 0;
        check(unsafe { cu(&mut n, self.inner.handle) })?;
        Ok(n)
    }

    /// Enumerate every kernel in the library (CUDA 12.4+). Allocates the
    /// result vector at the count reported by [`Self::kernel_count`].
    pub fn enumerate_kernels(&self) -> Result<Vec<Kernel>> {
        let d = driver()?;
        let n = self.kernel_count()?;
        let cu = d.cu_library_enumerate_kernels()?;
        let mut raw: Vec<baracuda_cuda_sys::CUkernel> = vec![core::ptr::null_mut(); n as usize];
        if n > 0 {
            check(unsafe { cu(raw.as_mut_ptr(), n, self.inner.handle) })?;
        }
        Ok(raw
            .into_iter()
            .map(|h| Kernel {
                handle: h,
                _library: self.clone(),
            })
            .collect())
    }

    /// Retrieve the underlying `CUmodule` backing this library, if any
    /// (CUDA 12.4+). Not all libraries have an addressable module — some
    /// ship compiled-kernel images only.
    pub fn module_raw(&self) -> Result<baracuda_cuda_sys::CUmodule> {
        let d = driver()?;
        let cu = d.cu_library_get_module()?;
        let mut m: baracuda_cuda_sys::CUmodule = core::ptr::null_mut();
        check(unsafe { cu(&mut m, self.inner.handle) })?;
        Ok(m)
    }

    /// Look up a managed-memory global by name (CUDA 12.2+).
    pub fn get_managed(&self, name: &str) -> Result<(CUdeviceptr, usize)> {
        let d = driver()?;
        let cu = d.cu_library_get_managed()?;
        let c_name = CString::new(name).map_err(|_| {
            crate::error::Error::Loader(baracuda_core::LoaderError::SymbolNotFound {
                library: "cuda-driver",
                symbol: "cuLibraryGetManaged(name contained a NUL byte)",
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

    /// Look up a unified-function pointer by name (CUDA 12.4+).
    /// Returns the raw host-side function pointer; the caller is
    /// responsible for casting it to the right signature before calling.
    pub fn get_unified_function(&self, name: &str) -> Result<*mut core::ffi::c_void> {
        let d = driver()?;
        let cu = d.cu_library_get_unified_function()?;
        let c_name = CString::new(name).map_err(|_| {
            crate::error::Error::Loader(baracuda_core::LoaderError::SymbolNotFound {
                library: "cuda-driver",
                symbol: "cuLibraryGetUnifiedFunction(name contained a NUL byte)",
            })
        })?;
        let mut fptr: *mut core::ffi::c_void = core::ptr::null_mut();
        check(unsafe {
            cu(
                &mut fptr,
                self.inner.handle,
                c_name.as_ptr() as *const c_char,
            )
        })?;
        Ok(fptr)
    }

    /// Look up a `__device__` global variable by name across contexts.
    /// Returns `(device_ptr, size_in_bytes)`. The returned pointer is valid
    /// in whatever context is current when the caller dereferences it.
    pub fn get_global(&self, name: &str) -> Result<(CUdeviceptr, usize)> {
        let d = driver()?;
        let cu = d.cu_library_get_global()?;
        let c_name = CString::new(name).map_err(|_| {
            crate::error::Error::Loader(baracuda_core::LoaderError::SymbolNotFound {
                library: "cuda-driver",
                symbol: "cuLibraryGetGlobal(name contained a NUL byte)",
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

    /// Raw `CUlibrary` handle. Use with care.
    #[inline]
    pub fn as_raw(&self) -> CUlibrary {
        self.inner.handle
    }
}

impl Drop for LibraryInner {
    fn drop(&mut self) {
        if let Ok(d) = driver() {
            if let Ok(cu) = d.cu_library_unload() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

/// A kernel from a [`Library`]. Library kernels are per-library but not
/// per-context; use [`Kernel::function_for_current_context`] to materialize
/// a [`Function`] for the active context before launching.
#[derive(Clone, Debug)]
pub struct Kernel {
    handle: CUkernel,
    _library: Library,
}

unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

impl Kernel {
    /// Raw `CUkernel`. Use with care.
    #[inline]
    pub fn as_raw(&self) -> CUkernel {
        self.handle
    }

    /// Materialize this library kernel into a [`Function`] suitable for
    /// the caller's currently-current CUDA context. The returned
    /// `Function` keeps a clone of the parent library alive.
    pub fn function_for_current_context(&self) -> Result<Function> {
        let d = driver()?;
        let cu = d.cu_kernel_get_function()?;
        let mut f: CUfunction = core::ptr::null_mut();
        check(unsafe { cu(&mut f, self.handle) })?;
        // Reuse the public module-less Function constructor via from_raw.
        Ok(Function::from_raw_with_library(f, self._library.clone()))
    }

    /// Query a `CUfunction_attribute` on this kernel for a specific device.
    pub fn attribute(&self, attr: i32, device: &crate::Device) -> Result<i32> {
        let d = driver()?;
        let cu = d.cu_kernel_get_attribute()?;
        let mut v: core::ffi::c_int = 0;
        check(unsafe { cu(&mut v, attr, self.handle, device.as_raw()) })?;
        Ok(v)
    }

    /// Set a (writable) `CUfunction_attribute` on this kernel for a
    /// specific device. Common writables: `MAX_DYNAMIC_SHARED_SIZE_BYTES`,
    /// `PREFERRED_SHARED_MEMORY_CARVEOUT`.
    pub fn set_attribute(&self, attr: i32, value: i32, device: &crate::Device) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_kernel_set_attribute()?;
        check(unsafe { cu(attr, value, self.handle, device.as_raw()) })
    }

    /// Return the kernel's demangled name as reported by the driver.
    pub fn name(&self) -> Result<String> {
        let d = driver()?;
        let cu = d.cu_kernel_get_name()?;
        let mut p: *const core::ffi::c_char = core::ptr::null();
        check(unsafe { cu(&mut p, self.handle) })?;
        if p.is_null() {
            return Ok(String::new());
        }
        // SAFETY: driver returns a NUL-terminated static string; we copy to owned.
        let cstr = unsafe { core::ffi::CStr::from_ptr(p) };
        Ok(cstr.to_string_lossy().into_owned())
    }

    /// Set the preferred L1 vs shared-memory cache config for this kernel
    /// on `device`. Pass one of the
    /// [`baracuda_cuda_sys::types::CUfunc_cache`] constants (PREFER_NONE,
    /// PREFER_SHARED, PREFER_L1, PREFER_EQUAL).
    pub fn set_cache_config(&self, config: u32, device: &crate::Device) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_kernel_set_cache_config()?;
        check(unsafe { cu(self.handle, config as core::ffi::c_int, device.as_raw()) })
    }

    /// Return `(offset_in_bytes, size_in_bytes)` for the `index`-th
    /// parameter in the kernel's ABI signature. Useful for reflective
    /// launches and matching Rust structs to kernel parameters.
    pub fn param_info(&self, index: usize) -> Result<(usize, usize)> {
        let d = driver()?;
        let cu = d.cu_kernel_get_param_info()?;
        let mut off: usize = 0;
        let mut sz: usize = 0;
        check(unsafe { cu(self.handle, index, &mut off, &mut sz) })?;
        Ok((off, sz))
    }

    /// Return the library that owns this kernel.
    pub fn library(&self) -> Library {
        self._library.clone()
    }
}
