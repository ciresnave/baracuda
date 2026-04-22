//! `cublasStatus_t` + `CudaStatus` impl.

#![allow(non_camel_case_types)]

use baracuda_types::CudaStatus;

/// Return code from a cuBLAS call.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct cublasStatus_t(pub i32);

#[allow(non_upper_case_globals)]
impl cublasStatus_t {
    pub const SUCCESS: Self = Self(0);
    pub const NOT_INITIALIZED: Self = Self(1);
    pub const ALLOC_FAILED: Self = Self(3);
    pub const INVALID_VALUE: Self = Self(7);
    pub const ARCH_MISMATCH: Self = Self(8);
    pub const MAPPING_ERROR: Self = Self(11);
    pub const EXECUTION_FAILED: Self = Self(13);
    pub const INTERNAL_ERROR: Self = Self(14);
    pub const NOT_SUPPORTED: Self = Self(15);
    pub const LICENSE_ERROR: Self = Self(16);

    pub const fn is_success(self) -> bool {
        self.0 == 0
    }
}

impl CudaStatus for cublasStatus_t {
    fn code(self) -> i32 {
        self.0
    }
    fn name(self) -> &'static str {
        match self.0 {
            0 => "CUBLAS_STATUS_SUCCESS",
            1 => "CUBLAS_STATUS_NOT_INITIALIZED",
            3 => "CUBLAS_STATUS_ALLOC_FAILED",
            7 => "CUBLAS_STATUS_INVALID_VALUE",
            8 => "CUBLAS_STATUS_ARCH_MISMATCH",
            11 => "CUBLAS_STATUS_MAPPING_ERROR",
            13 => "CUBLAS_STATUS_EXECUTION_FAILED",
            14 => "CUBLAS_STATUS_INTERNAL_ERROR",
            15 => "CUBLAS_STATUS_NOT_SUPPORTED",
            16 => "CUBLAS_STATUS_LICENSE_ERROR",
            _ => "CUBLAS_STATUS_UNRECOGNIZED",
        }
    }
    fn description(self) -> &'static str {
        match self.0 {
            0 => "success",
            1 => "cuBLAS library not initialized",
            3 => "resource allocation failed",
            7 => "invalid argument value",
            8 => "architecture mismatch",
            11 => "texture mapping error",
            13 => "GPU kernel execution failed",
            14 => "internal cuBLAS error",
            15 => "requested functionality not supported",
            16 => "cuBLAS license error",
            _ => "unrecognized cuBLAS status code",
        }
    }
    fn is_success(self) -> bool {
        cublasStatus_t::is_success(self)
    }
    fn library(self) -> &'static str {
        "cublas"
    }
}
