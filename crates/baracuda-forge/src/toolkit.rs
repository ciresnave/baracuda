//! CUDA toolkit auto-detection.
//!
//! Thin wrapper around [`baracuda_build::detect_cuda`] that adds nvcc-specific
//! conveniences (architecture listing) the runtime `-sys` crates don't need.

use crate::error::{Error, Result};
use std::path::PathBuf;
use std::process::Command;

/// CUDA toolkit information.
#[derive(Debug, Clone)]
pub struct CudaToolkit {
    /// Path to the `nvcc` binary.
    pub nvcc_path: PathBuf,
    /// CUDA include directory.
    pub include_dir: PathBuf,
    /// CUDA lib directory.
    pub lib_dir: PathBuf,
    /// CUDA version `(major, minor)` if detected (e.g., `(12, 6)`).
    pub version: Option<(u32, u32)>,
}

impl CudaToolkit {
    /// Auto-detect CUDA toolkit installation.
    ///
    /// Defers to [`baracuda_build::detect_cuda`] for install discovery, then
    /// resolves nvcc via [`baracuda_build::find_nvcc`] (which adds `$NVCC` /
    /// `$PATH` lookup as fallbacks).
    pub fn detect() -> Result<Self> {
        let install = baracuda_build::detect_cuda();
        let nvcc_path = baracuda_build::find_nvcc().ok_or_else(|| {
            Error::NvccNotFound(
                "No nvcc found via $NVCC, CUDA install dirs, or $PATH".to_string(),
            )
        })?;

        if let Some(install) = install {
            let version = install.version;
            return Ok(Self {
                nvcc_path,
                include_dir: install.include,
                lib_dir: install.lib,
                version,
            });
        }

        Self::from_nvcc_path(nvcc_path)
    }

    /// Create toolkit from explicit nvcc path.
    pub fn from_nvcc_path(nvcc_path: PathBuf) -> Result<Self> {
        if !nvcc_path.exists() {
            return Err(Error::NvccNotFound(nvcc_path.display().to_string()));
        }

        let cuda_root = nvcc_path
            .parent()
            .and_then(|p| p.parent())
            .ok_or_else(|| Error::CudaToolkitNotFound(nvcc_path.clone()))?;

        let include_dir = cuda_root.join("include");
        let lib_dir = if cfg!(target_os = "windows") {
            cuda_root.join("lib").join("x64")
        } else {
            cuda_root.join("lib64")
        };

        let version = nvcc_version(&nvcc_path);

        Ok(Self {
            nvcc_path,
            include_dir,
            lib_dir,
            version,
        })
    }

    /// Get supported GPU architectures by querying nvcc.
    pub fn supported_architectures(&self) -> Vec<usize> {
        let output = Command::new(&self.nvcc_path)
            .arg("--list-gpu-code")
            .output();

        if let Ok(output) = output {
            let stdout = String::from_utf8_lossy(&output.stdout);
            parse_gpu_codes(&stdout)
        } else {
            Vec::new()
        }
    }
}

fn nvcc_version(nvcc_path: &PathBuf) -> Option<(u32, u32)> {
    let output = Command::new(nvcc_path).arg("--version").output().ok()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    baracuda_build::parse_nvcc_version(&stdout)
}

fn parse_gpu_codes(output: &str) -> Vec<usize> {
    let mut codes = Vec::new();
    for line in output.lines() {
        let parts: Vec<&str> = line.split('_').collect();
        if parts.len() >= 2 && parts.contains(&"sm") {
            if let Ok(code) = parts[1].parse::<usize>() {
                codes.push(code);
            }
        }
    }
    codes.sort();
    codes.dedup();
    codes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_gpu_codes() {
        let output = "sm_52\nsm_60\nsm_70\nsm_75\nsm_80\nsm_86\nsm_89\nsm_90";
        let codes = parse_gpu_codes(output);
        assert!(codes.contains(&80));
        assert!(codes.contains(&90));
    }
}
