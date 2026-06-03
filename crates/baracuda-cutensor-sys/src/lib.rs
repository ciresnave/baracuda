//! Raw FFI + dynamic loader skeleton for NVIDIA cuTENSOR.
//!
//! `baracuda-cutensor` wraps this with a safe, typed API. Use this
//! crate directly only if you need a function that the safe layer
//! hasn't wrapped yet (in which case please file a bug).
//!
//! cuTENSOR is a separately-installed NVIDIA library for high-performance
//! tensor contraction, reduction, and element-wise ops. v0.1 ships the
//! loader + status enum; concrete contraction/permutation/reduction
//! wrappers follow once CI has a cuTENSOR install.

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
#![warn(missing_debug_implementations)]

use std::sync::OnceLock;

use baracuda_core::{Library, LoaderError};
use baracuda_types::CudaStatus;

/// cuTENSOR status code.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct cutensorStatus_t(pub i32);

impl cutensorStatus_t {
    /// `CUTENSOR_STATUS_SUCCESS` — operation succeeded.
    pub const SUCCESS: Self = Self(0);
    /// `CUTENSOR_STATUS_NOT_INITIALIZED` — the library is not initialized.
    pub const NOT_INITIALIZED: Self = Self(1);
    /// `CUTENSOR_STATUS_ALLOC_FAILED` — an allocation failed.
    pub const ALLOC_FAILED: Self = Self(3);
    /// `CUTENSOR_STATUS_INVALID_VALUE` — an argument has an invalid value.
    pub const INVALID_VALUE: Self = Self(7);
    /// `CUTENSOR_STATUS_ARCH_MISMATCH` — the device architecture is unsupported.
    pub const ARCH_MISMATCH: Self = Self(8);
    /// `CUTENSOR_STATUS_MAPPING_ERROR` — a host/device memory mapping error occurred.
    pub const MAPPING_ERROR: Self = Self(11);
    /// `CUTENSOR_STATUS_EXECUTION_FAILED` — kernel execution failed.
    pub const EXECUTION_FAILED: Self = Self(13);
    /// `CUTENSOR_STATUS_INTERNAL_ERROR` — an internal cuTENSOR error occurred.
    pub const INTERNAL_ERROR: Self = Self(14);
    /// `CUTENSOR_STATUS_NOT_SUPPORTED` — the requested feature is not supported.
    pub const NOT_SUPPORTED: Self = Self(15);
    /// `CUTENSOR_STATUS_LICENSE_ERROR` — license check failed.
    pub const LICENSE_ERROR: Self = Self(16);
    /// `CUTENSOR_STATUS_CUBLAS_ERROR` — an internal cuBLAS call failed.
    pub const CUBLAS_ERROR: Self = Self(17);
    /// `CUTENSOR_STATUS_CUDA_ERROR` — an internal CUDA call failed.
    pub const CUDA_ERROR: Self = Self(18);
    /// `CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE` — supplied workspace buffer is too small.
    pub const INSUFFICIENT_WORKSPACE: Self = Self(19);
    /// `CUTENSOR_STATUS_INSUFFICIENT_DRIVER` — installed CUDA driver is too old.
    pub const INSUFFICIENT_DRIVER: Self = Self(20);
    /// `CUTENSOR_STATUS_IO_ERROR` — an I/O error occurred (cache read/write, logger, ...).
    pub const IO_ERROR: Self = Self(21);

    /// Return `true` if the status code denotes success.
    pub const fn is_success(self) -> bool {
        self.0 == 0
    }
}

impl CudaStatus for cutensorStatus_t {
    fn code(self) -> i32 {
        self.0
    }
    fn name(self) -> &'static str {
        match self.0 {
            0 => "CUTENSOR_STATUS_SUCCESS",
            1 => "CUTENSOR_STATUS_NOT_INITIALIZED",
            3 => "CUTENSOR_STATUS_ALLOC_FAILED",
            7 => "CUTENSOR_STATUS_INVALID_VALUE",
            13 => "CUTENSOR_STATUS_EXECUTION_FAILED",
            15 => "CUTENSOR_STATUS_NOT_SUPPORTED",
            19 => "CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE",
            _ => "CUTENSOR_STATUS_UNRECOGNIZED",
        }
    }
    fn description(self) -> &'static str {
        match self.0 {
            0 => "success",
            15 => "operation not supported",
            19 => "workspace buffer too small",
            _ => "unrecognized cuTENSOR status code",
        }
    }
    fn is_success(self) -> bool {
        cutensorStatus_t::is_success(self)
    }
    fn library(self) -> &'static str {
        "cutensor"
    }
}

// ---- Handle + descriptor types ----

/// Opaque cuTENSOR handle.
pub type cutensorHandle_t = *mut core::ffi::c_void;

/// Opaque tensor descriptor.
pub type cutensorTensorDescriptor_t = *mut core::ffi::c_void;

/// Opaque contraction-plan descriptor.
pub type cutensorOperationDescriptor_t = *mut core::ffi::c_void;

/// Opaque plan preference handle.
pub type cutensorPlanPreference_t = *mut core::ffi::c_void;

/// Opaque plan (built from an operation descriptor + preference).
pub type cutensorPlan_t = *mut core::ffi::c_void;

/// `cutensorDataType_t` — element type enum.
#[allow(non_snake_case)]
pub mod cutensorDataType {
    /// `CUTENSOR_R_16F` — real fp16.
    pub const R_16F: i32 = 2; // fp16
    /// `CUTENSOR_R_16BF` — real bfloat16.
    pub const R_16BF: i32 = 14; // bfloat16
    /// `CUTENSOR_R_32F` — real fp32.
    pub const R_32F: i32 = 0; // float
    /// `CUTENSOR_R_64F` — real fp64.
    pub const R_64F: i32 = 1; // double
    /// `CUTENSOR_C_32F` — complex fp32.
    pub const C_32F: i32 = 4;
    /// `CUTENSOR_C_64F` — complex fp64.
    pub const C_64F: i32 = 5;
    /// `CUTENSOR_R_8I` — real signed 8-bit integer.
    pub const R_8I: i32 = 3;
    /// `CUTENSOR_R_8U` — real unsigned 8-bit integer.
    pub const R_8U: i32 = 8;
    /// `CUTENSOR_R_32I` — real signed 32-bit integer.
    pub const R_32I: i32 = 10;
    /// `CUTENSOR_R_32U` — real unsigned 32-bit integer.
    pub const R_32U: i32 = 12;
}

/// `cutensorComputeDescriptor_t` — the compute-precision descriptor
/// used on modern cuTENSOR (v2+). Opaque pointer.
///
/// In v2 this must NOT be null when building operation descriptors.
/// Get a valid pointer via [`Cutensor::compute_desc_32f`] and friends —
/// these resolve the library's pre-defined descriptor globals
/// (`CUTENSOR_COMPUTE_DESC_32F` etc.).
pub type cutensorComputeDescriptor_t = *const core::ffi::c_void;

impl Cutensor {
    /// Return a pre-defined compute descriptor resolved from the cuTENSOR
    /// shared library's exported data symbols. These symbols are global
    /// pointer variables (`CUTENSOR_COMPUTE_DESC_*`) that the library
    /// initializes — we read the pointer value at the symbol's address.
    fn compute_desc_by_name(
        &self,
        name: &'static str,
    ) -> Result<cutensorComputeDescriptor_t, LoaderError> {
        // SAFETY: the symbol resolves to a `cutensorComputeDescriptor_t*`
        // that cuTENSOR initializes on library load.
        let raw: *mut () = unsafe { self.lib.raw_symbol(name)? };
        let ptr_ptr = raw as *const cutensorComputeDescriptor_t;
        Ok(unsafe { *ptr_ptr })
    }

    /// `CUTENSOR_COMPUTE_DESC_32F` — 32-bit float compute.
    pub fn compute_desc_32f(&self) -> Result<cutensorComputeDescriptor_t, LoaderError> {
        self.compute_desc_by_name("CUTENSOR_COMPUTE_DESC_32F")
    }
    /// `CUTENSOR_COMPUTE_DESC_64F` — 64-bit float compute.
    pub fn compute_desc_64f(&self) -> Result<cutensorComputeDescriptor_t, LoaderError> {
        self.compute_desc_by_name("CUTENSOR_COMPUTE_DESC_64F")
    }
    /// `CUTENSOR_COMPUTE_DESC_16F` — 16-bit float compute.
    pub fn compute_desc_16f(&self) -> Result<cutensorComputeDescriptor_t, LoaderError> {
        self.compute_desc_by_name("CUTENSOR_COMPUTE_DESC_16F")
    }
    /// `CUTENSOR_COMPUTE_DESC_16BF` — bfloat16 compute.
    pub fn compute_desc_16bf(&self) -> Result<cutensorComputeDescriptor_t, LoaderError> {
        self.compute_desc_by_name("CUTENSOR_COMPUTE_DESC_16BF")
    }
    /// `CUTENSOR_COMPUTE_DESC_TF32` — TensorFloat32 compute.
    pub fn compute_desc_tf32(&self) -> Result<cutensorComputeDescriptor_t, LoaderError> {
        self.compute_desc_by_name("CUTENSOR_COMPUTE_DESC_TF32")
    }
    /// `CUTENSOR_COMPUTE_DESC_3XTF32` — 3× TF32 mantissa-extended compute.
    pub fn compute_desc_3xtf32(&self) -> Result<cutensorComputeDescriptor_t, LoaderError> {
        self.compute_desc_by_name("CUTENSOR_COMPUTE_DESC_3XTF32")
    }
    /// `CUTENSOR_COMPUTE_DESC_4X16F` — 4× FP16 mantissa-extended compute.
    pub fn compute_desc_4x16f(&self) -> Result<cutensorComputeDescriptor_t, LoaderError> {
        self.compute_desc_by_name("CUTENSOR_COMPUTE_DESC_4X16F")
    }
    /// `CUTENSOR_COMPUTE_DESC_8XINT8` — 8× INT8 packed compute.
    pub fn compute_desc_8xint8(&self) -> Result<cutensorComputeDescriptor_t, LoaderError> {
        self.compute_desc_by_name("CUTENSOR_COMPUTE_DESC_8XINT8")
    }
    /// `CUTENSOR_COMPUTE_DESC_9X16BF` — 9× BF16 mantissa-extended compute.
    pub fn compute_desc_9x16bf(&self) -> Result<cutensorComputeDescriptor_t, LoaderError> {
        self.compute_desc_by_name("CUTENSOR_COMPUTE_DESC_9X16BF")
    }
}

/// `cutensorOperator_t` — element-wise op selector.
#[allow(non_snake_case)]
pub mod cutensorOperator {
    /// `CUTENSOR_OP_IDENTITY` — identity (pass-through).
    pub const IDENTITY: i32 = 1;
    /// `CUTENSOR_OP_SQRT` — element-wise square root.
    pub const SQRT: i32 = 2;
    /// `CUTENSOR_OP_RELU` — element-wise ReLU.
    pub const RELU: i32 = 8;
    /// `CUTENSOR_OP_CONJ` — complex conjugate.
    pub const CONJ: i32 = 9;
    /// `CUTENSOR_OP_RCP` — element-wise reciprocal.
    pub const RCP: i32 = 10;
    /// `CUTENSOR_OP_SIGMOID` — element-wise sigmoid.
    pub const SIGMOID: i32 = 11;
    /// `CUTENSOR_OP_TANH` — element-wise hyperbolic tangent.
    pub const TANH: i32 = 12;
    /// `CUTENSOR_OP_ADD` — binary add (combines operands).
    pub const ADD: i32 = 3;
    /// `CUTENSOR_OP_MUL` — binary multiply (combines operands).
    pub const MUL: i32 = 5;
    /// `CUTENSOR_OP_MAX` — binary max (combines operands).
    pub const MAX: i32 = 6;
    /// `CUTENSOR_OP_MIN` — binary min (combines operands).
    pub const MIN: i32 = 7;
}

/// `cutensorAlgo_t` — algorithm selector for contraction planning.
#[allow(non_snake_case)]
pub mod cutensorAlgo {
    /// `CUTENSOR_ALGO_DEFAULT` — library-chosen algorithm.
    pub const DEFAULT: i32 = -1;
    /// `CUTENSOR_ALGO_GETT` — GETT (general tensor-tensor) algorithm.
    pub const GETT: i32 = -4;
    /// `CUTENSOR_ALGO_TGETT` — transposed-GETT algorithm.
    pub const TGETT: i32 = -3;
    /// `CUTENSOR_ALGO_TTGT` — transpose-transpose-GEMM-transpose algorithm.
    pub const TTGT: i32 = -2;
}

/// `cutensorJitMode_t` — Just-in-time-compile selector (cuTENSOR 2.x).
#[allow(non_snake_case)]
pub mod cutensorJitMode {
    /// `CUTENSOR_JIT_MODE_NONE` — JIT compilation disabled.
    pub const NONE: i32 = 0;
    /// `CUTENSOR_JIT_MODE_DEFAULT` — library-chosen JIT default.
    pub const DEFAULT: i32 = 1;
}

/// `cutensorWorksizePreference_t`.
#[allow(non_snake_case)]
pub mod cutensorWorksizePreference {
    /// `CUTENSOR_WORKSPACE_MIN` — request minimum workspace.
    pub const MIN: i32 = 1;
    /// `CUTENSOR_WORKSPACE_DEFAULT` — request library default workspace.
    pub const DEFAULT: i32 = 2;
    /// `CUTENSOR_WORKSPACE_MAX` — request maximum workspace.
    pub const MAX: i32 = 3;
}

// ---- PFN types ----

/// Function-pointer type for `cutensorCreate` (create cuTENSOR library handle). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorCreate =
    unsafe extern "C" fn(handle_out: *mut cutensorHandle_t) -> cutensorStatus_t;
/// Function-pointer type for `cutensorDestroy` (destroy cuTENSOR library handle). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorDestroy = unsafe extern "C" fn(handle: cutensorHandle_t) -> cutensorStatus_t;

/// Function-pointer type for `cutensorCreateTensorDescriptor` (create a tensor descriptor). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorCreateTensorDescriptor = unsafe extern "C" fn(
    handle: cutensorHandle_t,
    desc_out: *mut cutensorTensorDescriptor_t,
    num_modes: u32,
    extents: *const i64,
    strides: *const i64,
    data_type: i32,
    alignment_bytes: u32,
) -> cutensorStatus_t;
/// Function-pointer type for `cutensorDestroyTensorDescriptor` (destroy a tensor descriptor). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorDestroyTensorDescriptor =
    unsafe extern "C" fn(desc: cutensorTensorDescriptor_t) -> cutensorStatus_t;

/// Function-pointer type for `cutensorCreateContraction` (build an operation descriptor for a tensor contraction). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorCreateContraction = unsafe extern "C" fn(
    handle: cutensorHandle_t,
    op_desc_out: *mut cutensorOperationDescriptor_t,
    desc_a: cutensorTensorDescriptor_t,
    modes_a: *const i32,
    op_a: i32,
    desc_b: cutensorTensorDescriptor_t,
    modes_b: *const i32,
    op_b: i32,
    desc_c: cutensorTensorDescriptor_t,
    modes_c: *const i32,
    op_c: i32,
    desc_d: cutensorTensorDescriptor_t,
    modes_d: *const i32,
    compute_desc: cutensorComputeDescriptor_t,
) -> cutensorStatus_t;

/// Function-pointer type for `cutensorDestroyOperationDescriptor` (destroy an operation descriptor). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorDestroyOperationDescriptor =
    unsafe extern "C" fn(desc: cutensorOperationDescriptor_t) -> cutensorStatus_t;

/// Function-pointer type for `cutensorCreatePlanPreference` (create a plan-preference object). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorCreatePlanPreference = unsafe extern "C" fn(
    handle: cutensorHandle_t,
    pref_out: *mut cutensorPlanPreference_t,
    algo: i32,
    jit_mode: i32,
) -> cutensorStatus_t;
/// Function-pointer type for `cutensorDestroyPlanPreference` (destroy a plan-preference object). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorDestroyPlanPreference =
    unsafe extern "C" fn(pref: cutensorPlanPreference_t) -> cutensorStatus_t;

/// Function-pointer type for `cutensorEstimateWorkspaceSize` (estimate workspace bytes required by a plan). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorEstimateWorkspaceSize = unsafe extern "C" fn(
    handle: cutensorHandle_t,
    op_desc: cutensorOperationDescriptor_t,
    pref: cutensorPlanPreference_t,
    workspace_pref: i32,
    workspace_size_bytes_out: *mut u64,
) -> cutensorStatus_t;

/// Function-pointer type for `cutensorCreatePlan` (build an execution plan from an operation descriptor). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorCreatePlan = unsafe extern "C" fn(
    handle: cutensorHandle_t,
    plan_out: *mut cutensorPlan_t,
    op_desc: cutensorOperationDescriptor_t,
    pref: cutensorPlanPreference_t,
    workspace_size_limit: u64,
) -> cutensorStatus_t;
/// Function-pointer type for `cutensorDestroyPlan` (destroy an execution plan). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorDestroyPlan = unsafe extern "C" fn(plan: cutensorPlan_t) -> cutensorStatus_t;

/// Function-pointer type for `cutensorContract` (execute tensor contraction). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorContract = unsafe extern "C" fn(
    handle: cutensorHandle_t,
    plan: cutensorPlan_t,
    alpha: *const core::ffi::c_void,
    a: *const core::ffi::c_void,
    b: *const core::ffi::c_void,
    beta: *const core::ffi::c_void,
    c: *const core::ffi::c_void,
    d: *mut core::ffi::c_void,
    workspace: *mut core::ffi::c_void,
    workspace_size_bytes: u64,
    stream: *mut core::ffi::c_void, // cudaStream_t
) -> cutensorStatus_t;

/// Function-pointer type for `cutensorGetVersion` (query cuTENSOR library version). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorGetVersion = unsafe extern "C" fn() -> usize;
/// Function-pointer type for `cutensorGetCudartVersion` (query the CUDA Runtime version cuTENSOR was built against). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorGetCudartVersion = unsafe extern "C" fn() -> usize;
/// Function-pointer type for `cutensorGetErrorString` (decode a cutensorStatus_t into a static C string). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorGetErrorString =
    unsafe extern "C" fn(status: cutensorStatus_t) -> *const core::ffi::c_char;

// ---- Compute descriptor (opaque — built-in ones are statically exported from cuTENSOR) ----

/// Function-pointer type for `cutensorCreateElementwiseBinary` (build an operation descriptor for an element-wise binary op). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorCreateElementwiseBinary = unsafe extern "C" fn(
    handle: cutensorHandle_t,
    op_desc_out: *mut cutensorOperationDescriptor_t,
    desc_a: cutensorTensorDescriptor_t,
    modes_a: *const i32,
    op_a: i32,
    desc_c: cutensorTensorDescriptor_t,
    modes_c: *const i32,
    op_c: i32,
    desc_d: cutensorTensorDescriptor_t,
    modes_d: *const i32,
    op_ac: i32, // op_ac is the binary operator between A and C
    compute_desc: cutensorComputeDescriptor_t,
) -> cutensorStatus_t;

/// Function-pointer type for `cutensorElementwiseBinaryExecute` (execute an element-wise binary plan). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorElementwiseBinaryExecute = unsafe extern "C" fn(
    handle: cutensorHandle_t,
    plan: cutensorPlan_t,
    alpha: *const core::ffi::c_void,
    a: *const core::ffi::c_void,
    gamma: *const core::ffi::c_void,
    c: *const core::ffi::c_void,
    d: *mut core::ffi::c_void,
    stream: *mut core::ffi::c_void,
) -> cutensorStatus_t;

/// Function-pointer type for `cutensorCreateElementwiseTrinary` (build an operation descriptor for an element-wise trinary op). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorCreateElementwiseTrinary = unsafe extern "C" fn(
    handle: cutensorHandle_t,
    op_desc_out: *mut cutensorOperationDescriptor_t,
    desc_a: cutensorTensorDescriptor_t,
    modes_a: *const i32,
    op_a: i32,
    desc_b: cutensorTensorDescriptor_t,
    modes_b: *const i32,
    op_b: i32,
    desc_c: cutensorTensorDescriptor_t,
    modes_c: *const i32,
    op_c: i32,
    desc_d: cutensorTensorDescriptor_t,
    modes_d: *const i32,
    op_ab: i32,
    op_abc: i32,
    compute_desc: cutensorComputeDescriptor_t,
) -> cutensorStatus_t;

/// Function-pointer type for `cutensorElementwiseTrinaryExecute` (execute an element-wise trinary plan). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorElementwiseTrinaryExecute = unsafe extern "C" fn(
    handle: cutensorHandle_t,
    plan: cutensorPlan_t,
    alpha: *const core::ffi::c_void,
    a: *const core::ffi::c_void,
    beta: *const core::ffi::c_void,
    b: *const core::ffi::c_void,
    gamma: *const core::ffi::c_void,
    c: *const core::ffi::c_void,
    d: *mut core::ffi::c_void,
    stream: *mut core::ffi::c_void,
) -> cutensorStatus_t;

/// Function-pointer type for `cutensorCreatePermutation` (build an operation descriptor for a tensor permutation). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorCreatePermutation = unsafe extern "C" fn(
    handle: cutensorHandle_t,
    op_desc_out: *mut cutensorOperationDescriptor_t,
    desc_a: cutensorTensorDescriptor_t,
    modes_a: *const i32,
    op_a: i32,
    desc_b: cutensorTensorDescriptor_t,
    modes_b: *const i32,
    compute_desc: cutensorComputeDescriptor_t,
) -> cutensorStatus_t;

/// Function-pointer type for `cutensorPermute` (execute tensor permutation). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorPermute = unsafe extern "C" fn(
    handle: cutensorHandle_t,
    plan: cutensorPlan_t,
    alpha: *const core::ffi::c_void,
    a: *const core::ffi::c_void,
    b: *mut core::ffi::c_void,
    stream: *mut core::ffi::c_void,
) -> cutensorStatus_t;

/// Function-pointer type for `cutensorCreateReduction` (build an operation descriptor for a tensor reduction). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorCreateReduction = unsafe extern "C" fn(
    handle: cutensorHandle_t,
    op_desc_out: *mut cutensorOperationDescriptor_t,
    desc_a: cutensorTensorDescriptor_t,
    modes_a: *const i32,
    op_a: i32,
    desc_c: cutensorTensorDescriptor_t,
    modes_c: *const i32,
    op_c: i32,
    desc_d: cutensorTensorDescriptor_t,
    modes_d: *const i32,
    op_reduce: i32, // ADD / MUL / MAX / MIN
    compute_desc: cutensorComputeDescriptor_t,
) -> cutensorStatus_t;

/// Function-pointer type for `cutensorReduce` (execute tensor reduction). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorReduce = unsafe extern "C" fn(
    handle: cutensorHandle_t,
    plan: cutensorPlan_t,
    alpha: *const core::ffi::c_void,
    a: *const core::ffi::c_void,
    beta: *const core::ffi::c_void,
    c: *const core::ffi::c_void,
    d: *mut core::ffi::c_void,
    workspace: *mut core::ffi::c_void,
    workspace_size: u64,
    stream: *mut core::ffi::c_void,
) -> cutensorStatus_t;

// ---- Attribute getters / setters for operation descriptors + plan preferences ----

/// Function-pointer type for `cutensorOperationDescriptorGetAttribute` (get an attribute on an operation descriptor). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorOperationDescriptorGetAttribute = unsafe extern "C" fn(
    handle: cutensorHandle_t,
    op_desc: cutensorOperationDescriptor_t,
    attr: i32,
    buf: *mut core::ffi::c_void,
    size_in_bytes: usize,
) -> cutensorStatus_t;

/// Function-pointer type for `cutensorOperationDescriptorSetAttribute` (set an attribute on an operation descriptor). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorOperationDescriptorSetAttribute = unsafe extern "C" fn(
    handle: cutensorHandle_t,
    op_desc: cutensorOperationDescriptor_t,
    attr: i32,
    buf: *const core::ffi::c_void,
    size_in_bytes: usize,
) -> cutensorStatus_t;

/// Function-pointer type for `cutensorPlanPreferenceSetAttribute` (set an attribute on a plan-preference object). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorPlanPreferenceSetAttribute = unsafe extern "C" fn(
    handle: cutensorHandle_t,
    pref: cutensorPlanPreference_t,
    attr: i32,
    buf: *const core::ffi::c_void,
    size_in_bytes: usize,
) -> cutensorStatus_t;

/// Function-pointer type for `cutensorPlanGetAttribute` (get an attribute on a plan). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorPlanGetAttribute = unsafe extern "C" fn(
    handle: cutensorHandle_t,
    plan: cutensorPlan_t,
    attr: i32,
    buf: *mut core::ffi::c_void,
    size_in_bytes: usize,
) -> cutensorStatus_t;

/// Function-pointer type for `cutensorTensorDescriptorGetAttribute` (get an attribute on a tensor descriptor). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorTensorDescriptorGetAttribute = unsafe extern "C" fn(
    handle: cutensorHandle_t,
    desc: cutensorTensorDescriptor_t,
    attr: i32,
    buf: *mut core::ffi::c_void,
    size_in_bytes: usize,
) -> cutensorStatus_t;

// ---- Plan-cache management ----

/// Function-pointer type for `cutensorHandleResizePlanCache` (resize a handle's plan cache). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorHandleResizePlanCache =
    unsafe extern "C" fn(handle: cutensorHandle_t, num_entries: u32) -> cutensorStatus_t;

/// Function-pointer type for `cutensorHandleReadCacheFromFile` (read a plan/kernel cache from a file). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorHandleReadCacheFromFile = unsafe extern "C" fn(
    handle: cutensorHandle_t,
    filename: *const core::ffi::c_char,
) -> cutensorStatus_t;

/// Function-pointer type for `cutensorHandleWriteCacheToFile` (write a plan/kernel cache to a file). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorHandleWriteCacheToFile = unsafe extern "C" fn(
    handle: cutensorHandle_t,
    filename: *const core::ffi::c_char,
) -> cutensorStatus_t;

// ---- Trinary contraction (3-tensor chains) ----

/// Function-pointer type for `cutensorCreateContractionTrinary` (build an operation descriptor for a three-tensor contraction). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorCreateContractionTrinary = unsafe extern "C" fn(
    handle: cutensorHandle_t,
    op_desc_out: *mut cutensorOperationDescriptor_t,
    desc_a: cutensorTensorDescriptor_t,
    modes_a: *const i32,
    op_a: i32,
    desc_b: cutensorTensorDescriptor_t,
    modes_b: *const i32,
    op_b: i32,
    desc_c: cutensorTensorDescriptor_t,
    modes_c: *const i32,
    op_c: i32,
    desc_d: cutensorTensorDescriptor_t,
    modes_d: *const i32,
    op_d: i32,
    desc_e: cutensorTensorDescriptor_t,
    modes_e: *const i32,
    compute_desc: cutensorComputeDescriptor_t,
) -> cutensorStatus_t;

/// Function-pointer type for `cutensorContractTrinary` (execute three-tensor contraction). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorContractTrinary = unsafe extern "C" fn(
    handle: cutensorHandle_t,
    plan: cutensorPlan_t,
    alpha: *const core::ffi::c_void,
    a: *const core::ffi::c_void,
    b: *const core::ffi::c_void,
    c: *const core::ffi::c_void,
    beta: *const core::ffi::c_void,
    d: *const core::ffi::c_void,
    e: *mut core::ffi::c_void,
    workspace: *mut core::ffi::c_void,
    workspace_size: u64,
    stream: *mut core::ffi::c_void,
) -> cutensorStatus_t;

// ---- Custom compute descriptor lifecycle ----

/// Function-pointer type for `cutensorCreateComputeDescriptor` (create a compute-precision descriptor). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorCreateComputeDescriptor = unsafe extern "C" fn(
    handle: cutensorHandle_t,
    desc_out: *mut cutensorComputeDescriptor_t,
) -> cutensorStatus_t;

/// Function-pointer type for `cutensorDestroyComputeDescriptor` (destroy a compute-precision descriptor). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorDestroyComputeDescriptor =
    unsafe extern "C" fn(desc: cutensorComputeDescriptor_t) -> cutensorStatus_t;

/// Function-pointer type for `cutensorComputeDescriptorGetAttribute` (get an attribute on a compute descriptor). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorComputeDescriptorGetAttribute = unsafe extern "C" fn(
    handle: cutensorHandle_t,
    desc: cutensorComputeDescriptor_t,
    attr: i32,
    buf: *mut core::ffi::c_void,
    size_in_bytes: usize,
) -> cutensorStatus_t;

/// Function-pointer type for `cutensorComputeDescriptorSetAttribute` (set an attribute on a compute descriptor). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorComputeDescriptorSetAttribute = unsafe extern "C" fn(
    handle: cutensorHandle_t,
    desc: cutensorComputeDescriptor_t,
    attr: i32,
    buf: *const core::ffi::c_void,
    size_in_bytes: usize,
) -> cutensorStatus_t;

// ---- Additional attribute getters / setters ----

/// Function-pointer type for `cutensorTensorDescriptorSetAttribute` (set an attribute on a tensor descriptor). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorTensorDescriptorSetAttribute = unsafe extern "C" fn(
    handle: cutensorHandle_t,
    desc: cutensorTensorDescriptor_t,
    attr: i32,
    buf: *const core::ffi::c_void,
    size_in_bytes: usize,
) -> cutensorStatus_t;

/// Function-pointer type for `cutensorPlanPreferenceGetAttribute` (get an attribute on a plan-preference object). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorPlanPreferenceGetAttribute = unsafe extern "C" fn(
    handle: cutensorHandle_t,
    pref: cutensorPlanPreference_t,
    attr: i32,
    buf: *mut core::ffi::c_void,
    size_in_bytes: usize,
) -> cutensorStatus_t;

// ---- Operation-level introspection ----

/// Function-pointer type for `cutensorOperationEstimateRuntime` (estimate runtime in milliseconds for a planned operation). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorOperationEstimateRuntime = unsafe extern "C" fn(
    handle: cutensorHandle_t,
    op_desc: cutensorOperationDescriptor_t,
    pref: cutensorPlanPreference_t,
    algo: i32,
    runtime_ms_out: *mut f32,
) -> cutensorStatus_t;

/// Function-pointer type for `cutensorOperationNumAlgos` (query the number of algorithms available for an operation). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorOperationNumAlgos = unsafe extern "C" fn(
    op_desc: cutensorOperationDescriptor_t,
    num_algos_out: *mut i32,
) -> cutensorStatus_t;

// ---- Logging ----

/// Function-pointer type for `cutensorLoggerSetLevel` (set logger verbosity level). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorLoggerSetLevel = unsafe extern "C" fn(level: i32) -> cutensorStatus_t;

/// Function-pointer type for `cutensorLoggerSetMask` (set logger category mask). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorLoggerSetMask = unsafe extern "C" fn(mask: i32) -> cutensorStatus_t;

/// Function-pointer type for `cutensorLoggerOpenFile` (open a logger output file by path). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorLoggerOpenFile =
    unsafe extern "C" fn(path: *const core::ffi::c_char) -> cutensorStatus_t;

/// Function-pointer type for `cutensorLoggerSetFile` (redirect logger output to an open FILE*). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorLoggerSetFile =
    unsafe extern "C" fn(file: *mut core::ffi::c_void) -> cutensorStatus_t;

/// Function-pointer type for `cutensorLoggerSetCallback` (register a logger callback). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorLoggerSetCallback = unsafe extern "C" fn(
    callback: Option<unsafe extern "C" fn(i32, *const core::ffi::c_char, *const core::ffi::c_char)>,
) -> cutensorStatus_t;

/// Function-pointer type for `cutensorLoggerForceDisable` (force-disable the logger). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorLoggerForceDisable = unsafe extern "C" fn() -> cutensorStatus_t;

// ---- Block-sparse contraction (cuTENSOR 2.x) ----

/// Opaque block-sparse tensor descriptor.
pub type cutensorBlockSparseTensorDescriptor_t = *mut core::ffi::c_void;

/// Function-pointer type for `cutensorCreateBlockSparseTensorDescriptor` (create a block-sparse tensor descriptor). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorCreateBlockSparseTensorDescriptor = unsafe extern "C" fn(
    handle: cutensorHandle_t,
    desc_out: *mut cutensorBlockSparseTensorDescriptor_t,
    num_modes: u32,
    extents: *const i64,
    block_size: *const i64,
    strides: *const i64,
    block_index_count: i64,
    block_indices: *const i32,
    data_type: i32,
    alignment_bytes: u32,
) -> cutensorStatus_t;

/// Function-pointer type for `cutensorDestroyBlockSparseTensorDescriptor` (destroy a block-sparse tensor descriptor). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorDestroyBlockSparseTensorDescriptor =
    unsafe extern "C" fn(desc: cutensorBlockSparseTensorDescriptor_t) -> cutensorStatus_t;

/// Function-pointer type for `cutensorCreateBlockSparseContraction` (build an operation descriptor for a block-sparse contraction). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorCreateBlockSparseContraction = unsafe extern "C" fn(
    handle: cutensorHandle_t,
    op_desc_out: *mut cutensorOperationDescriptor_t,
    desc_a: cutensorBlockSparseTensorDescriptor_t,
    modes_a: *const i32,
    op_a: i32,
    desc_b: cutensorTensorDescriptor_t,
    modes_b: *const i32,
    op_b: i32,
    desc_c: cutensorTensorDescriptor_t,
    modes_c: *const i32,
    op_c: i32,
    desc_d: cutensorTensorDescriptor_t,
    modes_d: *const i32,
    compute_desc: cutensorComputeDescriptor_t,
) -> cutensorStatus_t;

/// Function-pointer type for `cutensorBlockSparseContract` (execute block-sparse tensor contraction). See <https://docs.nvidia.com/cuda/cutensor/index.html>.
pub type PFN_cutensorBlockSparseContract = unsafe extern "C" fn(
    handle: cutensorHandle_t,
    plan: cutensorPlan_t,
    alpha: *const core::ffi::c_void,
    a: *const core::ffi::c_void,
    b: *const core::ffi::c_void,
    beta: *const core::ffi::c_void,
    c: *const core::ffi::c_void,
    d: *mut core::ffi::c_void,
    workspace: *mut core::ffi::c_void,
    workspace_size: u64,
    stream: *mut core::ffi::c_void,
) -> cutensorStatus_t;

// ---- Loader ----

macro_rules! cutensor_fns {
    ($($(#[$attr:meta])* fn $name:ident as $sym:literal : $pfn:ty;)*) => {
        /// Lazily-resolved cuTENSOR function-pointer table.
        pub struct Cutensor {
            pub lib: Library,
            $(
                $name: OnceLock<$pfn>,
            )*
        }

        impl core::fmt::Debug for Cutensor {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct("Cutensor").field("lib", &self.lib).finish_non_exhaustive()
            }
        }

        impl Cutensor {
            fn empty(lib: Library) -> Self {
                Self { lib, $($name: OnceLock::new(),)* }
            }
            $(
                $(#[$attr])*
                #[doc = concat!("Resolve `", $sym, "`.")]
                pub fn $name(&self) -> Result<$pfn, LoaderError> {
                    if let Some(&p) = self.$name.get() { return Ok(p); }
                    let raw: *mut () = unsafe { self.lib.raw_symbol($sym)? };
                    let p: $pfn = unsafe { core::mem::transmute_copy::<*mut (), $pfn>(&raw) };
                    let _ = self.$name.set(p);
                    Ok(p)
                }
            )*
        }
    };
}

cutensor_fns! {
    fn cutensor_create as "cutensorCreate": PFN_cutensorCreate;
    fn cutensor_destroy as "cutensorDestroy": PFN_cutensorDestroy;
    fn cutensor_create_tensor_descriptor as "cutensorCreateTensorDescriptor":
        PFN_cutensorCreateTensorDescriptor;
    fn cutensor_destroy_tensor_descriptor as "cutensorDestroyTensorDescriptor":
        PFN_cutensorDestroyTensorDescriptor;
    fn cutensor_create_contraction as "cutensorCreateContraction": PFN_cutensorCreateContraction;
    fn cutensor_destroy_operation_descriptor as "cutensorDestroyOperationDescriptor":
        PFN_cutensorDestroyOperationDescriptor;
    fn cutensor_create_plan_preference as "cutensorCreatePlanPreference":
        PFN_cutensorCreatePlanPreference;
    fn cutensor_destroy_plan_preference as "cutensorDestroyPlanPreference":
        PFN_cutensorDestroyPlanPreference;
    fn cutensor_estimate_workspace_size as "cutensorEstimateWorkspaceSize":
        PFN_cutensorEstimateWorkspaceSize;
    fn cutensor_create_plan as "cutensorCreatePlan": PFN_cutensorCreatePlan;
    fn cutensor_destroy_plan as "cutensorDestroyPlan": PFN_cutensorDestroyPlan;
    fn cutensor_contract as "cutensorContract": PFN_cutensorContract;
    fn cutensor_get_version as "cutensorGetVersion": PFN_cutensorGetVersion;
    fn cutensor_get_cudart_version as "cutensorGetCudartVersion": PFN_cutensorGetCudartVersion;
    fn cutensor_get_error_string as "cutensorGetErrorString": PFN_cutensorGetErrorString;

    // Elementwise binary (A op C → D)
    fn cutensor_create_elementwise_binary as "cutensorCreateElementwiseBinary":
        PFN_cutensorCreateElementwiseBinary;
    fn cutensor_elementwise_binary_execute as "cutensorElementwiseBinaryExecute":
        PFN_cutensorElementwiseBinaryExecute;

    // Elementwise trinary ((A op B) op C → D)
    fn cutensor_create_elementwise_trinary as "cutensorCreateElementwiseTrinary":
        PFN_cutensorCreateElementwiseTrinary;
    fn cutensor_elementwise_trinary_execute as "cutensorElementwiseTrinaryExecute":
        PFN_cutensorElementwiseTrinaryExecute;

    // Permutation
    fn cutensor_create_permutation as "cutensorCreatePermutation":
        PFN_cutensorCreatePermutation;
    fn cutensor_permute as "cutensorPermute": PFN_cutensorPermute;

    // Reduction
    fn cutensor_create_reduction as "cutensorCreateReduction": PFN_cutensorCreateReduction;
    fn cutensor_reduce as "cutensorReduce": PFN_cutensorReduce;

    // Attributes
    fn cutensor_operation_descriptor_get_attribute as "cutensorOperationDescriptorGetAttribute":
        PFN_cutensorOperationDescriptorGetAttribute;
    fn cutensor_operation_descriptor_set_attribute as "cutensorOperationDescriptorSetAttribute":
        PFN_cutensorOperationDescriptorSetAttribute;
    fn cutensor_plan_preference_set_attribute as "cutensorPlanPreferenceSetAttribute":
        PFN_cutensorPlanPreferenceSetAttribute;
    fn cutensor_plan_get_attribute as "cutensorPlanGetAttribute":
        PFN_cutensorPlanGetAttribute;
    fn cutensor_tensor_descriptor_get_attribute as "cutensorTensorDescriptorGetAttribute":
        PFN_cutensorTensorDescriptorGetAttribute;

    // Plan cache
    fn cutensor_handle_resize_plan_cache as "cutensorHandleResizePlanCache":
        PFN_cutensorHandleResizePlanCache;
    fn cutensor_handle_read_plan_cache_from_file as "cutensorHandleReadPlanCacheFromFile":
        PFN_cutensorHandleReadCacheFromFile;
    fn cutensor_handle_write_plan_cache_to_file as "cutensorHandleWritePlanCacheToFile":
        PFN_cutensorHandleWriteCacheToFile;
    fn cutensor_read_kernel_cache_from_file as "cutensorReadKernelCacheFromFile":
        PFN_cutensorHandleReadCacheFromFile;
    fn cutensor_write_kernel_cache_to_file as "cutensorWriteKernelCacheToFile":
        PFN_cutensorHandleWriteCacheToFile;

    // Trinary contraction
    fn cutensor_create_contraction_trinary as "cutensorCreateContractionTrinary":
        PFN_cutensorCreateContractionTrinary;
    fn cutensor_contract_trinary as "cutensorContractTrinary": PFN_cutensorContractTrinary;

    // Custom compute descriptors
    fn cutensor_create_compute_descriptor as "cutensorCreateComputeDescriptor":
        PFN_cutensorCreateComputeDescriptor;
    fn cutensor_destroy_compute_descriptor as "cutensorDestroyComputeDescriptor":
        PFN_cutensorDestroyComputeDescriptor;
    fn cutensor_compute_descriptor_get_attribute as "cutensorComputeDescriptorGetAttribute":
        PFN_cutensorComputeDescriptorGetAttribute;
    fn cutensor_compute_descriptor_set_attribute as "cutensorComputeDescriptorSetAttribute":
        PFN_cutensorComputeDescriptorSetAttribute;

    // Additional attributes
    fn cutensor_tensor_descriptor_set_attribute as "cutensorTensorDescriptorSetAttribute":
        PFN_cutensorTensorDescriptorSetAttribute;
    fn cutensor_plan_preference_get_attribute as "cutensorPlanPreferenceGetAttribute":
        PFN_cutensorPlanPreferenceGetAttribute;

    // Introspection
    fn cutensor_operation_estimate_runtime as "cutensorOperationEstimateRuntime":
        PFN_cutensorOperationEstimateRuntime;
    fn cutensor_operation_num_algos as "cutensorOperationNumAlgos":
        PFN_cutensorOperationNumAlgos;

    // Logging
    fn cutensor_logger_set_level as "cutensorLoggerSetLevel": PFN_cutensorLoggerSetLevel;
    fn cutensor_logger_set_mask as "cutensorLoggerSetMask": PFN_cutensorLoggerSetMask;
    fn cutensor_logger_open_file as "cutensorLoggerOpenFile": PFN_cutensorLoggerOpenFile;
    fn cutensor_logger_set_file as "cutensorLoggerSetFile": PFN_cutensorLoggerSetFile;
    fn cutensor_logger_set_callback as "cutensorLoggerSetCallback":
        PFN_cutensorLoggerSetCallback;
    fn cutensor_logger_force_disable as "cutensorLoggerForceDisable":
        PFN_cutensorLoggerForceDisable;

    // Block-sparse contraction
    fn cutensor_create_block_sparse_tensor_descriptor as "cutensorCreateBlockSparseTensorDescriptor":
        PFN_cutensorCreateBlockSparseTensorDescriptor;
    fn cutensor_destroy_block_sparse_tensor_descriptor
        as "cutensorDestroyBlockSparseTensorDescriptor":
        PFN_cutensorDestroyBlockSparseTensorDescriptor;
    fn cutensor_create_block_sparse_contraction as "cutensorCreateBlockSparseContraction":
        PFN_cutensorCreateBlockSparseContraction;
    fn cutensor_block_sparse_contract as "cutensorBlockSparseContract":
        PFN_cutensorBlockSparseContract;
}

fn cutensor_candidates() -> &'static [&'static str] {
    #[cfg(target_os = "linux")]
    {
        &["libcutensor.so.2", "libcutensor.so.1", "libcutensor.so"]
    }
    #[cfg(target_os = "windows")]
    {
        &["cutensor.dll"]
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        &[]
    }
}

/// Extra directories to search for cuTENSOR on Windows — NVIDIA's
/// installer places it in a non-CUDA-Toolkit location.
#[cfg(target_os = "windows")]
fn cutensor_extra_dirs() -> Vec<std::path::PathBuf> {
    use std::path::PathBuf;
    let mut out = Vec::new();

    let progfiles = std::env::var("ProgramFiles").unwrap_or_else(|_| "C:\\Program Files".into());

    // Stand-alone cuTENSOR installs.
    let stand_alone_roots = [
        format!("{progfiles}\\NVIDIA cuTENSOR"),
        format!("{progfiles}\\NVIDIA\\cuTENSOR"),
    ];
    for root in &stand_alone_roots {
        // Typical layouts:
        //   <root>\<ver>\bin\<cuda-major>\cutensor.dll
        //   <root>\<ver>\lib\<cuda-major>\cutensor.dll
        //   <root>\bin\cutensor.dll
        let root_pb = PathBuf::from(root);
        if let Ok(top) = std::fs::read_dir(&root_pb) {
            for ent in top.flatten() {
                let p = ent.path();
                if p.is_dir() {
                    out.push(p.join("bin"));
                    for sub in [
                        "bin\\12", "bin\\13", "bin\\11", "lib\\12", "lib\\13", "lib\\11",
                    ] {
                        out.push(p.join(sub));
                    }
                }
            }
        }
        out.push(root_pb.join("bin"));
    }

    // Also fall back to the CUDA Toolkit's own bin dir (some installers
    // drop a stub there).
    for var in ["CUDA_PATH", "CUDA_HOME"] {
        if let Ok(p) = std::env::var(var) {
            out.push(PathBuf::from(p).join("bin"));
        }
    }

    out
}

/// Return the lazily-loaded cuTENSOR library accessor.
pub fn cutensor() -> Result<&'static Cutensor, LoaderError> {
    static CUTENSOR: OnceLock<Cutensor> = OnceLock::new();
    if let Some(c) = CUTENSOR.get() {
        return Ok(c);
    }
    let lib = match Library::open("cutensor", cutensor_candidates()) {
        Ok(l) => l,
        Err(e) => {
            #[cfg(target_os = "windows")]
            {
                let mut found: Option<Library> = None;
                for dir in cutensor_extra_dirs() {
                    for candidate in cutensor_candidates() {
                        let full = dir.join(candidate);
                        if let Ok(l) = Library::open_at("cutensor", &full) {
                            found = Some(l);
                            break;
                        }
                    }
                    if found.is_some() {
                        break;
                    }
                }
                match found {
                    Some(l) => l,
                    None => return Err(e),
                }
            }
            #[cfg(not(target_os = "windows"))]
            {
                return Err(e);
            }
        }
    };
    let _ = CUTENSOR.set(Cutensor::empty(lib));
    Ok(CUTENSOR.get().expect("OnceLock set or lost race"))
}
