//! Build-script helpers shared by every `baracuda-*-sys` crate.
//!
//! Typical usage from a `-sys` crate's `build.rs`:
//!
//! ```text
//! fn main() {
//!     baracuda_build::emit_rerun_hints();
//!     // The rest is only relevant when the crate's `bindgen` feature is on:
//!     #[cfg(feature = "bindgen")] {
//!         if let Some(install) = baracuda_build::detect_cuda() {
//!             let bindings = baracuda_build::bindgen_builder(&install)
//!                 .header(install.include.join("cuda.h").to_string_lossy().to_string())
//!                 .allowlist_function("cu.*")
//!                 .allowlist_type("CU.*")
//!                 .generate()
//!                 .expect("bindgen");
//!             // ...
//!         }
//!     }
//! }
//! ```
//!
//! Nothing in this module shells out to `nvcc`, reads NVIDIA's headers, or
//! requires a CUDA install to be present at build time. The functions
//! return `Option` / empty results when CUDA can't be found.

use std::path::{Path, PathBuf};

/// Where CUDA is installed and what version is on disk.
#[derive(Clone, Debug)]
pub struct CudaInstall {
    /// The toolkit root (contains `include/` and `lib*/` subdirs).
    pub root: PathBuf,
    /// Parsed `cuda.h` version, e.g. `(12, 6)`. `None` if we found the
    /// toolkit but couldn't parse `cuda.h` (unusual).
    pub version: Option<(u32, u32)>,
    /// `$root/include`.
    pub include: PathBuf,
    /// `$root/lib64` (Linux) or `$root/lib/x64` (Windows), whichever exists.
    pub lib: PathBuf,
    /// `$root/bin/nvcc[.exe]` if present. `None` for runtime-only installs
    /// that ship the libraries but no compiler (rare on a normal toolkit
    /// install but possible in stripped Docker images).
    pub nvcc: Option<PathBuf>,
}

/// Emit `cargo:rerun-if-env-changed=` lines for every env var the detector looks at.
///
/// Call this from `build.rs` unconditionally — it's cheap and lets Cargo
/// re-run the build script if the user changes `CUDA_PATH`, etc.
pub fn emit_rerun_hints() {
    for var in [
        "CUDA_PATH",
        "CUDA_HOME",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
        "CUDNN_LIB",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }
}

/// Search the usual environment variables and OS-default paths for a CUDA
/// install. Returns `None` if none is found.
pub fn detect_cuda() -> Option<CudaInstall> {
    for var in [
        "CUDA_PATH",
        "CUDA_HOME",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
    ] {
        if let Ok(raw) = std::env::var(var) {
            if let Some(install) = probe_root(&PathBuf::from(raw)) {
                return Some(install);
            }
        }
    }

    let candidates: &[&str] = if cfg!(target_os = "linux") {
        &["/usr/local/cuda", "/opt/cuda"]
    } else if cfg!(target_os = "windows") {
        // Callers typically set CUDA_PATH on Windows; fall back via CUDA_PATH_V*_*.
        &[]
    } else {
        &[]
    };
    for c in candidates {
        if let Some(install) = probe_root(&PathBuf::from(c)) {
            return Some(install);
        }
    }

    None
}

fn probe_root(root: &Path) -> Option<CudaInstall> {
    if !root.is_dir() {
        return None;
    }
    let include = root.join("include");
    if !include.is_dir() {
        return None;
    }
    let lib = pick_lib_dir(root)?;
    let version = read_cuda_h_version(&include.join("cuda.h"));
    let nvcc = pick_nvcc(root);
    Some(CudaInstall {
        root: root.to_path_buf(),
        version,
        include,
        lib,
        nvcc,
    })
}

fn pick_nvcc(root: &Path) -> Option<PathBuf> {
    let exe = if cfg!(target_os = "windows") {
        "nvcc.exe"
    } else {
        "nvcc"
    };
    let p = root.join("bin").join(exe);
    if p.is_file() {
        Some(p)
    } else {
        None
    }
}

fn pick_lib_dir(root: &Path) -> Option<PathBuf> {
    // Return the directory that actually holds the import/link libraries, in
    // priority order. On Windows that is `lib\x64` — the bare `lib` is only its
    // parent and contains no `.lib`, so `lib\x64` must be tried *before* `lib`
    // (an earlier ordering returned `lib`, breaking `cudart.lib` resolution and
    // contradicting `CudaInstall::lib`'s documented contract). On Linux the
    // libraries live in `lib64` or the target-triple dir. `bin` is a last-ditch
    // fallback for unusual layouts. Linux ordering is preserved exactly.
    let candidates: &[&str] = if cfg!(target_os = "windows") {
        &["lib/x64", "lib", "bin"]
    } else {
        &["lib64", "lib", "targets/x86_64-linux/lib", "bin"]
    };
    for sub in candidates {
        let p = root.join(sub);
        if p.is_dir() {
            return Some(p);
        }
    }
    None
}

fn read_cuda_h_version(cuda_h: &Path) -> Option<(u32, u32)> {
    let src = std::fs::read_to_string(cuda_h).ok()?;
    // Look for `#define CUDA_VERSION <N>` where N is major*1000 + minor*10.
    for line in src.lines() {
        let line = line.trim_start();
        if let Some(rest) = line.strip_prefix("#define CUDA_VERSION") {
            let number: u32 = rest
                .trim()
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse()
                .ok()?;
            let major = number / 1000;
            let minor = (number % 1000) / 10;
            return Some((major, minor));
        }
    }
    None
}

/// Return the path to `libcuda`, `libcudart`, `libcublas`, ... within a given install.
///
/// `stem` should be the library's base name without `lib` prefix or `.so` /
/// `.dll` suffix — e.g. `"cudart"`, `"cublas"`, `"cudnn"`.
pub fn find_library(install: &CudaInstall, stem: &str) -> Option<PathBuf> {
    let names: Vec<String> = if cfg!(target_os = "windows") {
        let major = install.version.map(|(m, _)| m).unwrap_or(12);
        vec![format!("{stem}64_{major}.dll"), format!("{stem}.dll")]
    } else {
        let major = install.version.map(|(m, _)| m).unwrap_or(12);
        vec![format!("lib{stem}.so.{major}"), format!("lib{stem}.so")]
    };
    for name in &names {
        let p = install.lib.join(name);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Locate `nvcc` on the system, in this order:
///
/// 1. `$NVCC` environment variable (if it points at an existing file).
/// 2. The `nvcc` field of [`detect_cuda`]'s result.
/// 3. The first `nvcc[.exe]` found by walking `$PATH`.
///
/// Returns `None` when no compiler is found. Used by `baracuda-forge` for
/// kernel compilation; the runtime `-sys` crates don't need this.
pub fn find_nvcc() -> Option<PathBuf> {
    if let Ok(raw) = std::env::var("NVCC") {
        let p = PathBuf::from(raw);
        if p.is_file() {
            return Some(p);
        }
    }

    if let Some(install) = detect_cuda() {
        if let Some(nvcc) = install.nvcc {
            return Some(nvcc);
        }
    }

    let exe = if cfg!(target_os = "windows") {
        "nvcc.exe"
    } else {
        "nvcc"
    };
    if let Some(path_var) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path_var) {
            let candidate = dir.join(exe);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    None
}

/// Parse `nvcc --version` stdout into `(major, minor)`.
///
/// nvcc prints a line like `Cuda compilation tools, release 12.6, V12.6.85`;
/// this returns `Some((12, 6))` for that input. Returns `None` if no
/// `release X.Y` token is found.
pub fn parse_nvcc_version(stdout: &str) -> Option<(u32, u32)> {
    for line in stdout.lines() {
        if let Some(rest) = line.split_once("release ").map(|(_, r)| r) {
            let token: String = rest
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            let mut parts = token.split('.');
            let major = parts.next()?.parse().ok()?;
            let minor = parts.next().unwrap_or("0").parse().ok()?;
            return Some((major, minor));
        }
    }
    None
}

/// Emit `cargo:rustc-cfg=cuda_<major>_<minor>` so downstream crates can
/// conditionally compile based on the detected toolkit version. Ignored when
/// the install's `cuda.h` didn't expose `CUDA_VERSION`.
pub fn emit_version_cfg(install: &CudaInstall) {
    if let Some((major, minor)) = install.version {
        println!("cargo:rustc-cfg=cuda_{major}_{minor}");
        println!("cargo:rustc-cfg=cuda_{major}");
    }
}

/// Build a `bindgen::Builder` pre-configured for CUDA headers.
///
/// Only available with the `bindgen` feature.
#[cfg(feature = "bindgen")]
pub fn bindgen_builder(install: &CudaInstall) -> bindgen::Builder {
    bindgen::Builder::default()
        .clang_arg(format!("-I{}", install.include.display()))
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++17")
        // CUDA headers use `__host__` / `__device__` etc. — strip them.
        .clang_arg("-D__CUDACC__")
        .clang_arg("-D__host__=")
        .clang_arg("-D__device__=")
        .clang_arg("-D__global__=")
        .clang_arg("-D__shared__=")
        .clang_arg("-D__forceinline__=")
        // Layout guarantees baracuda relies on.
        .derive_debug(true)
        .derive_default(true)
        .derive_eq(true)
        .derive_hash(true)
        .derive_partialeq(true)
        .derive_partialord(true)
        .default_enum_style(bindgen::EnumVariation::NewType {
            is_bitfield: false,
            is_global: false,
        })
        .layout_tests(false)
        .generate_comments(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn parse_cuda_version_from_synthetic_header() {
        let tmp = std::env::temp_dir().join(format!("baracuda-build-test-{}", std::process::id()));
        let _ = fs::create_dir_all(&tmp);
        let cuda_h = tmp.join("cuda.h");
        fs::write(&cuda_h, "/* header */\n#define CUDA_VERSION 12060\n").unwrap();
        let v = read_cuda_h_version(&cuda_h);
        assert_eq!(v, Some((12, 6)));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn missing_root_returns_none() {
        let missing = PathBuf::from("/definitely-not-a-cuda-install-yolo");
        assert!(probe_root(&missing).is_none());
    }

    #[test]
    fn parse_nvcc_version_extracts_major_minor() {
        let out = "nvcc: NVIDIA (R) Cuda compiler driver\nCopyright (c) 2005-2024 NVIDIA Corporation\nBuilt on Some_Date\nCuda compilation tools, release 12.6, V12.6.85\nBuild cuda_12.6.r12.6/compiler.34714021_0\n";
        assert_eq!(parse_nvcc_version(out), Some((12, 6)));
    }

    #[test]
    fn parse_nvcc_version_handles_minor_zero() {
        let out = "Cuda compilation tools, release 11, V11.0.0\n";
        assert_eq!(parse_nvcc_version(out), Some((11, 0)));
    }

    #[test]
    fn parse_nvcc_version_returns_none_for_unrelated_output() {
        assert_eq!(parse_nvcc_version("hello world"), None);
    }

    #[test]
    fn pick_lib_dir_prefers_arch_specific_subdir() {
        let tmp = std::env::temp_dir()
            .join(format!("baracuda-build-libdir-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);

        if cfg!(target_os = "windows") {
            // A real Windows toolkit has both `lib` and `lib\x64`; the import
            // libraries (e.g. cudart.lib) live in `lib\x64`. Regression guard:
            // the bare `lib` must NOT win over `lib\x64`.
            fs::create_dir_all(tmp.join("lib").join("x64")).unwrap();
            assert_eq!(pick_lib_dir(&tmp), Some(tmp.join("lib").join("x64")));
        } else {
            // Linux: `lib64` is the canonical lib dir and must win over `lib`.
            fs::create_dir_all(tmp.join("lib64")).unwrap();
            fs::create_dir_all(tmp.join("lib")).unwrap();
            assert_eq!(pick_lib_dir(&tmp), Some(tmp.join("lib64")));
        }

        let _ = fs::remove_dir_all(&tmp);
    }
}
