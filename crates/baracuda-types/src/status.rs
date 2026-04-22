//! A uniform interface every NVIDIA-library status/return-code enum implements.
//!
//! `CudaStatus` lets higher-level code print or log an error from any of the
//! ~15 separate status enums (`CUresult`, `cudaError_t`, `cublasStatus_t`,
//! `cufftResult`, `curandStatus_t`, `cusparseStatus_t`, `cusolverStatus_t`,
//! `cudnnStatus_t`, `ncclResult_t`, `nvrtcResult`, `nvjpegStatus_t`,
//! `cutensorStatus_t`, `nppStatus`, `nvmlReturn_t`, ...) without having to
//! special-case each one at the call site.
//!
//! `-sys` crates implement `CudaStatus` on their repr-`i32` status enums.

/// A status code returned by an NVIDIA library call.
pub trait CudaStatus: Copy + core::fmt::Debug + Eq {
    /// The integer value of this status, as returned from the C API.
    fn code(self) -> i32;

    /// The stable symbol name of this status
    /// (e.g. `"CUDA_SUCCESS"`, `"CUBLAS_STATUS_NOT_INITIALIZED"`).
    fn name(self) -> &'static str;

    /// A human-readable description of the status.
    ///
    /// Implementations should prefer the strings returned by the library
    /// itself (`cuGetErrorString`, `cudnnGetErrorString`, ...) when available.
    fn description(self) -> &'static str;

    /// `true` if this status represents successful completion.
    fn is_success(self) -> bool;

    /// Which NVIDIA library produced this status (e.g. `"cuda-driver"`,
    /// `"cublas"`). Used for composing cross-library error messages.
    fn library(self) -> &'static str;
}
