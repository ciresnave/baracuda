//! Safe Rust wrappers for NVIDIA NVRTC — runtime CUDA-C++-to-PTX compiler.
//!
//! ```no_run
//! use baracuda_nvrtc::Program;
//!
//! # fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! let ptx = Program::compile(
//!     r#"
//!     extern "C" __global__ void saxpy(float a, float* x, float* y, int n) {
//!         int i = blockIdx.x * blockDim.x + threadIdx.x;
//!         if (i < n) y[i] = a * x[i] + y[i];
//!     }
//!     "#,
//!     "saxpy.cu",
//!     &["--gpu-architecture=compute_80"],
//! )?;
//! # let _ = ptx; Ok(()) }
//! ```

#![warn(missing_debug_implementations)]

use std::ffi::CString;

use baracuda_nvrtc_sys::{nvrtc, nvrtcProgram, nvrtcResult};

/// Error type for NVRTC operations.
pub type Error = baracuda_core::Error<nvrtcResult>;
/// Result alias.
pub type Result<T, E = Error> = core::result::Result<T, E>;

#[inline]
fn check(status: nvrtcResult) -> Result<()> {
    Error::check(status)
}

/// An NVRTC program. Dropped = destroyed.
pub struct Program {
    prog: nvrtcProgram,
}

unsafe impl Send for Program {}

impl core::fmt::Debug for Program {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Program")
            .field("prog", &self.prog)
            .finish_non_exhaustive()
    }
}

impl Program {
    /// Create a new program from CUDA-C++ source.
    pub fn new(source: &str, name: &str) -> Result<Self> {
        let n = nvrtc()?;
        let cu = n.nvrtc_create_program()?;
        let c_src = CString::new(source).expect("source must not contain NUL");
        let c_name = CString::new(name).expect("name must not contain NUL");
        let mut prog: nvrtcProgram = core::ptr::null_mut();
        check(unsafe {
            cu(
                &mut prog,
                c_src.as_ptr(),
                c_name.as_ptr(),
                0,
                core::ptr::null(),
                core::ptr::null(),
            )
        })?;
        Ok(Self { prog })
    }

    /// Compile the program with the given command-line options
    /// (e.g. `"--gpu-architecture=compute_80"`).
    pub fn compile_raw(&self, options: &[&str]) -> Result<()> {
        let n = nvrtc()?;
        let cu = n.nvrtc_compile_program()?;
        let storage: Vec<CString> = options
            .iter()
            .map(|s| CString::new(*s).expect("NVRTC option must not contain NUL"))
            .collect();
        let ptrs: Vec<*const core::ffi::c_char> = storage.iter().map(|s| s.as_ptr()).collect();
        check(unsafe { cu(self.prog, ptrs.len() as core::ffi::c_int, ptrs.as_ptr()) })
    }

    /// Retrieve compiled PTX as a UTF-8 string.
    pub fn ptx(&self) -> Result<String> {
        let n = nvrtc()?;
        let size_fn = n.nvrtc_get_ptx_size()?;
        let get_fn = n.nvrtc_get_ptx()?;
        let mut size: usize = 0;
        check(unsafe { size_fn(self.prog, &mut size) })?;
        let mut buf = vec![0u8; size];
        check(unsafe { get_fn(self.prog, buf.as_mut_ptr() as *mut core::ffi::c_char) })?;
        // PTX is NUL-terminated; drop the trailing NUL.
        if buf.last() == Some(&0) {
            buf.pop();
        }
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    /// Retrieve the compiler's log (populated on failure, and sometimes on success).
    pub fn log(&self) -> Result<String> {
        let n = nvrtc()?;
        let size_fn = n.nvrtc_get_program_log_size()?;
        let get_fn = n.nvrtc_get_program_log()?;
        let mut size: usize = 0;
        check(unsafe { size_fn(self.prog, &mut size) })?;
        if size <= 1 {
            return Ok(String::new());
        }
        let mut buf = vec![0u8; size];
        check(unsafe { get_fn(self.prog, buf.as_mut_ptr() as *mut core::ffi::c_char) })?;
        if buf.last() == Some(&0) {
            buf.pop();
        }
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    /// One-shot convenience: create + compile + fetch PTX.
    ///
    /// If compilation fails, the returned error's source is an NVRTC
    /// `COMPILATION` status; call [`Program::new`] manually if you need
    /// to inspect the build log on error.
    pub fn compile(source: &str, name: &str, options: &[&str]) -> Result<String> {
        let prog = Self::new(source, name)?;
        prog.compile_raw(options)?;
        prog.ptx()
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        if let Ok(n) = nvrtc() {
            if let Ok(cu) = n.nvrtc_destroy_program() {
                let _ = unsafe { cu(&mut self.prog) };
            }
        }
    }
}

/// NVRTC library version as `(major, minor)`.
pub fn version() -> Result<(i32, i32)> {
    let n = nvrtc()?;
    let cu = n.nvrtc_version()?;
    let mut major: core::ffi::c_int = 0;
    let mut minor: core::ffi::c_int = 0;
    check(unsafe { cu(&mut major, &mut minor) })?;
    Ok((major, minor))
}
