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
    Some(CudaInstall {
        root: root.to_path_buf(),
        version,
        include,
        lib,
    })
}

fn pick_lib_dir(root: &Path) -> Option<PathBuf> {
    for sub in &["lib64", "lib", "targets/x86_64-linux/lib", "lib/x64", "bin"] {
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
}
