//! Safe Rust wrappers for NVIDIA nvJitLink — the modern CUDA 12+ JIT linker.
//!
//! Supersedes the driver API's `cuLink*` family with a cleaner interface
//! and LTO IR support.
//!
//! ```no_run
//! use baracuda_nvjitlink::{InputType, Linker};
//!
//! # fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! let mut linker = Linker::new(&["-arch=sm_89"])?;
//! let ptx_source = b"/* some PTX... */";
//! linker.add_data(InputType::Ptx, ptx_source, "kernel.ptx")?;
//! linker.complete()?;
//! let cubin = linker.linked_cubin()?;
//! # let _ = cubin; Ok(()) }
//! ```

#![warn(missing_debug_implementations)]

use std::ffi::CString;

use baracuda_nvjitlink_sys::{nvJitLinkHandle, nvJitLinkInputType, nvJitLinkResult, nvjitlink};

/// Error type for nvJitLink operations.
pub type Error = baracuda_core::Error<nvJitLinkResult>;
/// Result alias.
pub type Result<T, E = Error> = core::result::Result<T, E>;

#[inline]
fn check(status: nvJitLinkResult) -> Result<()> {
    Error::check(status)
}

/// Kind of blob passed to [`Linker::add_data`] / [`Linker::add_file`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum InputType {
    Cubin,
    Ptx,
    LtoIr,
    Fatbin,
    Object,
    Library,
}

impl InputType {
    #[inline]
    fn raw(self) -> nvJitLinkInputType {
        match self {
            InputType::Cubin => nvJitLinkInputType::Cubin,
            InputType::Ptx => nvJitLinkInputType::Ptx,
            InputType::LtoIr => nvJitLinkInputType::LtoIr,
            InputType::Fatbin => nvJitLinkInputType::Fatbin,
            InputType::Object => nvJitLinkInputType::Object,
            InputType::Library => nvJitLinkInputType::Library,
        }
    }
}

/// An nvJitLink linker session. Dropped = destroyed.
pub struct Linker {
    handle: nvJitLinkHandle,
    /// Keep the options C-strings alive for at least as long as the linker.
    _option_storage: Vec<CString>,
}

unsafe impl Send for Linker {}

impl core::fmt::Debug for Linker {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Linker")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

impl Linker {
    /// Create a new linker with `options` (e.g. `"-arch=sm_89"`,
    /// `"-lto"`, `"-g"`).
    pub fn new(options: &[&str]) -> Result<Self> {
        let n = nvjitlink()?;
        let cu = n.nv_jit_link_create()?;

        let storage: Vec<CString> = options
            .iter()
            .map(|s| CString::new(*s).expect("nvJitLink option must not contain a NUL byte"))
            .collect();
        let ptrs: Vec<*const core::ffi::c_char> = storage.iter().map(|s| s.as_ptr()).collect();

        let mut handle: nvJitLinkHandle = core::ptr::null_mut();
        check(unsafe { cu(&mut handle, ptrs.len() as u32, ptrs.as_ptr()) })?;
        Ok(Self {
            handle,
            _option_storage: storage,
        })
    }

    /// Add an in-memory blob to the link.
    pub fn add_data(&mut self, ty: InputType, data: &[u8], name: &str) -> Result<()> {
        let n = nvjitlink()?;
        let cu = n.nv_jit_link_add_data()?;
        let c_name = CString::new(name).expect("input name must not contain a NUL byte");
        check(unsafe {
            cu(
                self.handle,
                ty.raw(),
                data.as_ptr() as *const core::ffi::c_void,
                data.len(),
                c_name.as_ptr(),
            )
        })
    }

    /// Add a file from disk to the link.
    pub fn add_file(&mut self, ty: InputType, path: &str) -> Result<()> {
        let n = nvjitlink()?;
        let cu = n.nv_jit_link_add_file()?;
        let c_path = CString::new(path).expect("path must not contain a NUL byte");
        check(unsafe { cu(self.handle, ty.raw(), c_path.as_ptr()) })
    }

    /// Finalize the link. Call before retrieving the output.
    pub fn complete(&mut self) -> Result<()> {
        let n = nvjitlink()?;
        let cu = n.nv_jit_link_complete()?;
        check(unsafe { cu(self.handle) })
    }

    /// Retrieve the linked CUBIN as a byte vector. Call after [`Self::complete`].
    pub fn linked_cubin(&self) -> Result<Vec<u8>> {
        let n = nvjitlink()?;
        let size_fn = n.nv_jit_link_get_linked_cubin_size()?;
        let get_fn = n.nv_jit_link_get_linked_cubin()?;
        let mut size: usize = 0;
        check(unsafe { size_fn(self.handle, &mut size) })?;
        let mut buf = vec![0u8; size];
        check(unsafe { get_fn(self.handle, buf.as_mut_ptr() as *mut core::ffi::c_void) })?;
        Ok(buf)
    }

    /// Retrieve the linker's error log (populated on failure).
    pub fn error_log(&self) -> Result<String> {
        let n = nvjitlink()?;
        let size_fn = n.nv_jit_link_get_error_log_size()?;
        let get_fn = n.nv_jit_link_get_error_log()?;
        let mut size: usize = 0;
        check(unsafe { size_fn(self.handle, &mut size) })?;
        if size == 0 {
            return Ok(String::new());
        }
        let mut buf = vec![0u8; size];
        check(unsafe { get_fn(self.handle, buf.as_mut_ptr() as *mut core::ffi::c_char) })?;
        // Trim trailing NUL.
        if buf.last() == Some(&0) {
            buf.pop();
        }
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    /// Raw `nvJitLinkHandle`. Use with care.
    #[inline]
    pub fn as_raw(&self) -> nvJitLinkHandle {
        self.handle
    }
}

impl Drop for Linker {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        if let Ok(n) = nvjitlink() {
            if let Ok(cu) = n.nv_jit_link_destroy() {
                // nvJitLinkDestroy takes `nvJitLinkHandle*` — pass the
                // address of our handle slot; the driver zeroes it.
                let _ = unsafe { cu(&mut self.handle) };
            }
        }
    }
}

/// nvJitLink library version as `(major, minor)`.
pub fn version() -> Result<(u32, u32)> {
    let n = nvjitlink()?;
    let cu = n.nv_jit_link_version()?;
    let mut major: core::ffi::c_uint = 0;
    let mut minor: core::ffi::c_uint = 0;
    check(unsafe { cu(&mut major, &mut minor) })?;
    Ok((major, minor))
}
