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

/// Typed builder for the NVRTC `--flag=value` CLI options.
///
/// Raw string flags are still accepted via
/// [`Program::compile_raw`] / [`Program::compile`]. Use this struct when
/// you want compile-time typo protection and IDE completion; its
/// [`Self::into_raw`] converts to the `Vec<String>` the underlying API
/// expects.
///
/// # Example
///
/// ```no_run
/// use baracuda_nvrtc::{CompileOptions, Program};
///
/// # fn demo() -> baracuda_nvrtc::Result<()> {
/// let opts = CompileOptions::new()
///     .arch("sm_80")
///     .use_fast_math(true)
///     .line_info(true)
///     .define("TILE_M", Some("128"));
///
/// let ptx = Program::compile_with("/* source */", "kernel.cu", &opts)?;
/// # let _ = ptx; Ok(()) }
/// ```
#[derive(Clone, Debug, Default)]
pub struct CompileOptions {
    /// `--gpu-architecture=sm_XX` or `compute_XX`.
    pub arch: Option<String>,
    /// `--std=c++17` etc.
    pub cpp_standard: Option<String>,
    /// `--use_fast_math` toggle.
    pub use_fast_math: Option<bool>,
    /// `--line-info` toggle (preserve source location for debuggers).
    pub line_info: Option<bool>,
    /// `--generate-line-info` (alias; same as [`Self::line_info`]).
    pub device_debug: Option<bool>,
    /// `--relocatable-device-code=true|false`.
    pub rdc: Option<bool>,
    /// `--extensible-whole-program=true|false`.
    pub extensible_whole_program: Option<bool>,
    /// `--ftz=true|false`.
    pub ftz: Option<bool>,
    /// `-I <dir>` include-path entries.
    pub include_paths: Vec<String>,
    /// `-D NAME` or `-D NAME=value` macro definitions.
    pub defines: Vec<(String, Option<String>)>,
    /// Any extra raw flags not covered by a typed setter.
    pub extra: Vec<String>,
}

impl CompileOptions {
    /// Fresh `CompileOptions` with every field at its default (unset).
    pub fn new() -> Self {
        Self::default()
    }

    /// `--gpu-architecture=<arch>`. Use `sm_80`, `compute_90a`, etc.
    pub fn arch(mut self, arch: impl Into<String>) -> Self {
        self.arch = Some(arch.into());
        self
    }

    /// `--std=<standard>`, e.g. `"c++17"` or `"c++20"`.
    pub fn cpp_standard(mut self, std: impl Into<String>) -> Self {
        self.cpp_standard = Some(std.into());
        self
    }

    /// Turn on (or off) `--use_fast_math`.
    pub fn use_fast_math(mut self, enable: bool) -> Self {
        self.use_fast_math = Some(enable);
        self
    }

    /// Turn on (or off) `--generate-line-info`.
    pub fn line_info(mut self, enable: bool) -> Self {
        self.line_info = Some(enable);
        self
    }

    /// Alias matching NVIDIA's `-G` flag. Use for debug builds.
    pub fn device_debug(mut self, enable: bool) -> Self {
        self.device_debug = Some(enable);
        self
    }

    /// Turn relocatable device code on or off.
    pub fn rdc(mut self, enable: bool) -> Self {
        self.rdc = Some(enable);
        self
    }

    /// Extensible-whole-program mode. Usually only needed for libraries that
    /// dynamically dispatch kernels.
    pub fn extensible_whole_program(mut self, enable: bool) -> Self {
        self.extensible_whole_program = Some(enable);
        self
    }

    /// Flush-to-zero for denormals. Off by default.
    pub fn ftz(mut self, enable: bool) -> Self {
        self.ftz = Some(enable);
        self
    }

    /// Add a `-I <dir>` include-path entry.
    pub fn include(mut self, dir: impl Into<String>) -> Self {
        self.include_paths.push(dir.into());
        self
    }

    /// Add a `-D NAME` or `-D NAME=value` preprocessor macro.
    pub fn define(mut self, name: impl Into<String>, value: Option<impl Into<String>>) -> Self {
        self.defines.push((name.into(), value.map(Into::into)));
        self
    }

    /// Any raw flag NVRTC understands. Escape hatch for options we haven't
    /// grown a typed setter for yet.
    pub fn extra(mut self, flag: impl Into<String>) -> Self {
        self.extra.push(flag.into());
        self
    }

    /// Render to the `Vec<String>` form `nvrtcCompileProgram` wants.
    pub fn to_strings(&self) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(a) = &self.arch {
            out.push(format!("--gpu-architecture={a}"));
        }
        if let Some(s) = &self.cpp_standard {
            out.push(format!("--std={s}"));
        }
        if self.use_fast_math == Some(true) {
            out.push("--use_fast_math".to_string());
        }
        if self.line_info == Some(true) {
            out.push("--generate-line-info".to_string());
        }
        if self.device_debug == Some(true) {
            out.push("-G".to_string());
        }
        if let Some(rdc) = self.rdc {
            out.push(format!("--relocatable-device-code={rdc}"));
        }
        if let Some(ewp) = self.extensible_whole_program {
            out.push(format!("--extensible-whole-program={ewp}"));
        }
        if let Some(ftz) = self.ftz {
            out.push(format!("--ftz={ftz}"));
        }
        for inc in &self.include_paths {
            out.push(format!("-I{inc}"));
        }
        for (name, value) in &self.defines {
            match value {
                Some(v) => out.push(format!("-D{name}={v}")),
                None => out.push(format!("-D{name}")),
            }
        }
        out.extend(self.extra.iter().cloned());
        out
    }

    /// Render to a freshly-allocated `Vec<String>` that can back a slice of
    /// `&str` for [`Program::compile`] / [`Program::compile_raw`]:
    ///
    /// ```no_run
    /// # use baracuda_nvrtc::{CompileOptions, Program};
    /// # fn demo() -> baracuda_nvrtc::Result<()> {
    /// let opts = CompileOptions::new().arch("sm_80").use_fast_math(true);
    /// let owned = opts.to_str_refs();
    /// let borrowed: Vec<&str> = owned.iter().map(String::as_str).collect();
    /// let ptx = Program::compile("/* ... */", "kernel.cu", &borrowed)?;
    /// # let _ = ptx; Ok(()) }
    /// ```
    pub fn to_str_refs(&self) -> Vec<String> {
        self.to_strings()
    }
}

impl Program {
    /// Compile with a typed [`CompileOptions`] instead of a raw `&[&str]`.
    /// Equivalent to [`Program::compile`] after rendering the options.
    pub fn compile_with(source: &str, name: &str, options: &CompileOptions) -> Result<String> {
        let owned = options.to_strings();
        let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
        Self::compile(source, name, &refs)
    }
}
