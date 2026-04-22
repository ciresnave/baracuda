//! Platform detection and default library search paths.

use std::path::PathBuf;

/// Broad host-OS classification used by the loader.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum OsFamily {
    Linux,
    Windows,
    /// Anything else (e.g. macOS). baracuda refuses to load on these.
    Unsupported,
}

/// Detect the current OS family at runtime (cheap; branches on `cfg!`).
pub const fn os_family() -> OsFamily {
    if cfg!(target_os = "linux") {
        OsFamily::Linux
    } else if cfg!(target_os = "windows") {
        OsFamily::Windows
    } else {
        OsFamily::Unsupported
    }
}

/// `true` when running under the Windows Subsystem for Linux (WSL2).
///
/// Cheap probe: reads `/proc/version`. Safe to call on non-Linux hosts; it
/// simply returns `false`.
pub fn is_wsl2() -> bool {
    if !matches!(os_family(), OsFamily::Linux) {
        return false;
    }
    std::fs::read_to_string("/proc/version")
        .map(|s| {
            let s = s.to_ascii_lowercase();
            s.contains("microsoft") || s.contains("wsl")
        })
        .unwrap_or(false)
}

/// Directories the loader will search for NVIDIA shared libraries.
///
/// Order (first hit wins):
///
/// 1. `$CUDA_PATH`, `$CUDA_HOME`, `$CUDA_ROOT`, `$CUDA_TOOLKIT_ROOT_DIR`
///    (each joined with the OS-appropriate `lib`/`bin` subdirectory).
/// 2. OS defaults: `/usr/local/cuda/*`, `/usr/local/cuda/compat`,
///    `/usr/lib/x86_64-linux-gnu`, `/usr/lib/wsl/lib` on Linux;
///    `%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v*\bin` on Windows.
///
/// The returned paths are candidates only; the loader silently skips any
/// that don't exist.
pub fn library_search_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    for var in [
        "CUDA_PATH",
        "CUDA_HOME",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
    ] {
        if let Ok(raw) = std::env::var(var) {
            let base = PathBuf::from(raw);
            push_os_subdirs(&base, &mut paths);
        }
    }

    match os_family() {
        OsFamily::Linux => {
            for base in ["/usr/local/cuda", "/opt/cuda"] {
                push_os_subdirs(&PathBuf::from(base), &mut paths);
            }
            paths.push(PathBuf::from("/usr/local/cuda/compat"));
            paths.push(PathBuf::from("/usr/lib/x86_64-linux-gnu"));
            paths.push(PathBuf::from("/usr/lib/aarch64-linux-gnu"));
            paths.push(PathBuf::from("/usr/lib/wsl/lib"));
            paths.push(PathBuf::from("/lib/wsl/lib"));
        }
        OsFamily::Windows => {
            if let Ok(pf) = std::env::var("ProgramFiles") {
                let toolkit = PathBuf::from(pf)
                    .join("NVIDIA GPU Computing Toolkit")
                    .join("CUDA");
                // We can't glob at const time; caller pairs this with specific
                // CUDA_PATH_V12_6 env vars. Keep the prefix as a hint.
                paths.push(toolkit);
            }
            for var in [
                "CUDA_PATH_V13_0",
                "CUDA_PATH_V12_8",
                "CUDA_PATH_V12_6",
                "CUDA_PATH_V12_3",
                "CUDA_PATH_V12_0",
                "CUDA_PATH_V11_8",
                "CUDA_PATH_V11_4",
            ] {
                if let Ok(raw) = std::env::var(var) {
                    let base = PathBuf::from(raw);
                    push_os_subdirs(&base, &mut paths);
                }
            }
        }
        OsFamily::Unsupported => {}
    }

    paths
}

fn push_os_subdirs(base: &std::path::Path, out: &mut Vec<PathBuf>) {
    match os_family() {
        OsFamily::Linux => {
            out.push(base.join("lib64"));
            out.push(base.join("lib"));
            out.push(base.join("targets/x86_64-linux/lib"));
            out.push(base.join("lib/stubs"));
            out.push(base.join("lib64/stubs"));
        }
        OsFamily::Windows => {
            out.push(base.join("bin"));
            out.push(base.join("lib").join("x64"));
        }
        OsFamily::Unsupported => {}
    }
}

/// The most common `libcuda` filenames to probe, in preference order.
pub const fn driver_library_candidates() -> &'static [&'static str] {
    #[cfg(target_os = "linux")]
    {
        &["libcuda.so.1", "libcuda.so"]
    }
    #[cfg(target_os = "windows")]
    {
        &["nvcuda.dll"]
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        &[]
    }
}

/// The most common `libcudart` filenames to probe, in preference order.
pub const fn runtime_library_candidates() -> &'static [&'static str] {
    #[cfg(target_os = "linux")]
    {
        &[
            "libcudart.so.13",
            "libcudart.so.12",
            "libcudart.so.11.0",
            "libcudart.so",
        ]
    }
    #[cfg(target_os = "windows")]
    {
        &[
            "cudart64_13.dll",
            "cudart64_12.dll",
            "cudart64_110.dll",
            "cudart64_101.dll",
        ]
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        &[]
    }
}

/// Build the list of probe filenames for a generic CUDA library, across the
/// major versions baracuda targets. For example,
/// `versioned_library_candidates("cublas", "12", "11.0")` yields
/// `libcublas.so.12, libcublas.so.11.0, libcublas.so` on Linux and
/// `cublas64_12.dll, cublas64_11.dll, cublas64_110.dll` on Windows.
///
/// Individual `-sys` crates provide their own curated candidate lists;
/// this helper is for opportunistic probing during development.
pub fn versioned_library_candidates(name: &str, preferred_majors: &[&str]) -> Vec<String> {
    let mut out = Vec::with_capacity(preferred_majors.len() + 2);
    match os_family() {
        OsFamily::Linux => {
            for major in preferred_majors {
                out.push(format!("lib{name}.so.{major}"));
            }
            out.push(format!("lib{name}.so"));
        }
        OsFamily::Windows => {
            for major in preferred_majors {
                // Windows CUDA DLL convention: cublas64_12.dll, cublas64_11.dll, ...
                // Keep only leading digit(s) before a dot.
                let numeric = major.split('.').next().unwrap_or(major);
                out.push(format!("{name}64_{numeric}.dll"));
            }
            out.push(format!("{name}64.dll"));
        }
        OsFamily::Unsupported => {}
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn os_family_matches_cfg() {
        let f = os_family();
        if cfg!(target_os = "linux") {
            assert_eq!(f, OsFamily::Linux);
        } else if cfg!(target_os = "windows") {
            assert_eq!(f, OsFamily::Windows);
        } else {
            assert_eq!(f, OsFamily::Unsupported);
        }
    }

    #[test]
    fn driver_candidates_nonempty_on_supported() {
        if matches!(os_family(), OsFamily::Linux | OsFamily::Windows) {
            assert!(!driver_library_candidates().is_empty());
            assert!(!runtime_library_candidates().is_empty());
        }
    }

    #[test]
    fn versioned_candidates_linux_shape() {
        if matches!(os_family(), OsFamily::Linux) {
            let v = versioned_library_candidates("cublas", &["12", "11.0"]);
            assert!(v.iter().any(|s| s == "libcublas.so.12"));
            assert!(v.iter().any(|s| s == "libcublas.so"));
        }
    }

    #[test]
    fn versioned_candidates_windows_shape() {
        if matches!(os_family(), OsFamily::Windows) {
            let v = versioned_library_candidates("cublas", &["12", "11"]);
            assert!(v.iter().any(|s| s == "cublas64_12.dll"));
        }
    }

    #[test]
    fn search_paths_include_env() {
        std::env::set_var("CUDA_PATH", "/tmp/test-cuda-path");
        let paths = library_search_paths();
        let has = paths.iter().any(|p| p.starts_with("/tmp/test-cuda-path"));
        std::env::remove_var("CUDA_PATH");
        assert!(
            has,
            "CUDA_PATH environment should show up in the search list"
        );
    }
}
