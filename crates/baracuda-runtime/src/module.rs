//! Runtime-API library + kernel loading.
//!
//! Uses `cudaLibraryLoadData` (CUDA 12.0+) to load PTX / CUBIN at runtime,
//! exposing kernels as [`Kernel`] handles. On CUDA < 12 this returns
//! [`crate::Error::FeatureNotSupported`].

use core::ffi::{c_char, c_void};
use std::ffi::CString;
use std::sync::Arc;

use baracuda_cuda_sys::runtime::{cudaKernel_t, cudaLibrary_t, runtime};
use baracuda_types::{supports, CudaVersion, Feature};

use crate::error::{check, Error, Result};

/// A loaded CUDA library (CUDA 12.0+).
#[derive(Clone)]
pub struct Library {
    inner: Arc<LibraryInner>,
}

struct LibraryInner {
    handle: cudaLibrary_t,
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
    /// Load a library from a raw binary image (CUBIN, fatbin, or null-terminated PTX).
    pub fn load_raw(image: &[u8]) -> Result<Self> {
        let installed = crate::init::driver_version()?;
        if !supports(installed, Feature::LibraryManagement) {
            return Err(Error::FeatureNotSupported {
                api: "cudaLibraryLoadData",
                since: Feature::LibraryManagement.required_version(),
            });
        }

        let r = runtime()?;
        let cu = r.cuda_library_load_data()?;
        let mut lib: cudaLibrary_t = core::ptr::null_mut();
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

    /// Load a library from a PTX source string.
    pub fn load_ptx(ptx_source: &str) -> Result<Self> {
        let c_src = CString::new(ptx_source).map_err(|_| {
            Error::Loader(baracuda_core::LoaderError::SymbolNotFound {
                library: "cuda-runtime",
                symbol: "cudaLibraryLoadData(PTX input contained a NUL byte)",
            })
        })?;
        Self::load_raw(c_src.as_bytes_with_nul())
    }

    /// Look up a kernel entry point by name.
    pub fn get_kernel(&self, name: &str) -> Result<Kernel> {
        let r = runtime()?;
        let cu = r.cuda_library_get_kernel()?;
        let c_name = CString::new(name).map_err(|_| {
            Error::Loader(baracuda_core::LoaderError::SymbolNotFound {
                library: "cuda-runtime",
                symbol: "cudaLibraryGetKernel(kernel name contained a NUL byte)",
            })
        })?;
        let mut kernel: cudaKernel_t = core::ptr::null_mut();
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

    /// Raw `cudaLibrary_t`. Use with care.
    #[inline]
    pub fn as_raw(&self) -> cudaLibrary_t {
        self.inner.handle
    }
}

impl Drop for LibraryInner {
    fn drop(&mut self) {
        if let Ok(r) = runtime() {
            if let Ok(cu) = r.cuda_library_unload() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

/// A kernel entry point inside a [`Library`].
#[derive(Clone, Debug)]
pub struct Kernel {
    handle: cudaKernel_t,
    // Keeps the library alive for the lifetime of the kernel.
    _library: Library,
}

unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

impl Kernel {
    /// Raw `cudaKernel_t`. Use with care.
    #[inline]
    pub fn as_raw(&self) -> cudaKernel_t {
        self.handle
    }

    /// Returns the raw kernel handle cast to a `const void*` — the form
    /// expected by `cudaLaunchKernel`. Library-loaded kernels can be
    /// launched through the standard runtime launch function by
    /// passing this pointer.
    #[inline]
    pub fn as_launch_ptr(&self) -> *const c_void {
        self.handle as *const c_void
    }

    /// `cudaOccupancyMaxActiveBlocksPerMultiprocessor` — how many blocks
    /// of size `block_size` can run concurrently per SM given
    /// `dynamic_smem_bytes` of dynamic shared memory.
    pub fn max_active_blocks_per_multiprocessor(
        &self,
        block_size: i32,
        dynamic_smem_bytes: usize,
    ) -> Result<i32> {
        let r = runtime()?;
        let cu = r.cuda_occupancy_max_active_blocks_per_multiprocessor()?;
        let mut n: core::ffi::c_int = 0;
        check(unsafe { cu(&mut n, self.as_launch_ptr(), block_size, dynamic_smem_bytes) })?;
        Ok(n)
    }

    /// `cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags` — same
    /// as [`Self::max_active_blocks_per_multiprocessor`] but accepting
    /// occupancy-flag bits (0 = default, 1 = disable shared-memory
    /// carveout adjustment).
    pub fn max_active_blocks_per_multiprocessor_with_flags(
        &self,
        block_size: i32,
        dynamic_smem_bytes: usize,
        flags: u32,
    ) -> Result<i32> {
        let r = runtime()?;
        let cu = r.cuda_occupancy_max_active_blocks_per_multiprocessor_with_flags()?;
        let mut n: core::ffi::c_int = 0;
        check(unsafe {
            cu(
                &mut n,
                self.as_launch_ptr(),
                block_size,
                dynamic_smem_bytes,
                flags,
            )
        })?;
        Ok(n)
    }

    /// `cudaOccupancyAvailableDynamicSMemPerBlock` — how much dynamic
    /// shared memory can each block use if `num_blocks` of `block_size`
    /// threads run concurrently on each SM.
    pub fn available_dynamic_smem_per_block(
        &self,
        num_blocks: i32,
        block_size: i32,
    ) -> Result<usize> {
        let r = runtime()?;
        let cu = r.cuda_occupancy_available_dynamic_smem_per_block()?;
        let mut n: usize = 0;
        check(unsafe { cu(&mut n, self.as_launch_ptr(), num_blocks, block_size) })?;
        Ok(n)
    }

    /// Set a writable `cudaFuncAttribute`. The common one is
    /// `cudaFuncAttributeMaxDynamicSharedMemorySize = 8`.
    pub fn set_attribute(&self, attr: i32, value: i32) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_func_set_attribute()?;
        check(unsafe { cu(self.as_launch_ptr(), attr, value) })
    }
}

/// Unused helper to check `CudaVersion` availability from outside this module.
#[allow(dead_code)]
fn require_library_management(installed: CudaVersion) -> Result<()> {
    if supports(installed, Feature::LibraryManagement) {
        Ok(())
    } else {
        Err(Error::FeatureNotSupported {
            api: "cudaLibraryLoadData",
            since: Feature::LibraryManagement.required_version(),
        })
    }
}
