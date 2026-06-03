//! Raw FFI + dynamic loader for NVIDIA cuSPARSE (generic API subset).
//!
//! `baracuda-cusparse` wraps this with a safe, typed API. Use this
//! crate directly only if you need a function that the safe layer
//! hasn't wrapped yet (in which case please file a bug).

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
#![warn(missing_debug_implementations)]

use core::ffi::{c_int, c_void};
use std::sync::OnceLock;

use baracuda_core::{platform, Library, LoaderError};
use baracuda_cuda_sys::runtime::cudaStream_t;
use baracuda_types::CudaStatus;

// ---- handles --------------------------------------------------------------

/// Opaque cuSPARSE handle.
pub type cusparseHandle_t = *mut c_void;
/// Opaque sparse-matrix descriptor (CSR / CSC / COO / BSR).
pub type cusparseSpMatDescr_t = *mut c_void;
/// Opaque dense-matrix descriptor.
pub type cusparseDnMatDescr_t = *mut c_void;
/// Opaque dense-vector descriptor.
pub type cusparseDnVecDescr_t = *mut c_void;
/// Opaque SpGEMM intermediate-state descriptor.
pub type cusparseSpGEMMDescr_t = *mut c_void;
/// Opaque SpSV intermediate-state descriptor.
pub type cusparseSpSVDescr_t = *mut c_void;
/// Opaque SpSM intermediate-state descriptor.
pub type cusparseSpSMDescr_t = *mut c_void;
/// Opaque legacy-API matrix descriptor.
pub type cusparseMatDescr_t = *mut c_void;

// ---- enums ----------------------------------------------------------------

/// Transpose selector for cuSPARSE routines (matches cuBLAS values).
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusparseOperation_t {
    /// No transpose.
    N = 0,
    /// Transpose.
    T = 1,
    /// Conjugate transpose.
    C = 2,
}

/// Index-element dtype for sparse-matrix offsets and column indices.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusparseIndexType_t {
    /// Unsigned 16-bit indices.
    I16U = 1,
    /// Signed 32-bit indices.
    I32I = 2,
    /// Signed 64-bit indices.
    I64I = 3,
}

/// Zero- vs one-based indexing for sparse-matrix index arrays.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusparseIndexBase_t {
    /// Zero-based indexing.
    Zero = 0,
    /// One-based indexing (Fortran convention).
    One = 1,
}

/// Row-major vs column-major dense storage order.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusparseOrder_t {
    /// Row-major dense storage.
    Row = 1,
    /// Column-major dense storage.
    Col = 2,
}

/// Algorithm selector for `cusparseSpMV`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusparseSpMVAlg_t {
    /// Driver-chosen default.
    Default = 0,
    /// CSR algorithm 1 (deterministic).
    CsrAlg1 = 2,
    /// CSR algorithm 2 (higher throughput).
    CsrAlg2 = 3,
    /// COO algorithm 1.
    CooAlg1 = 1,
    /// COO algorithm 2.
    CooAlg2 = 4,
}

/// Algorithm selector for `cusparseSpMM`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusparseSpMMAlg_t {
    /// Driver-chosen default.
    Default = 0,
    /// COO algorithm 1.
    CooAlg1 = 1,
    /// CSR algorithm 1.
    CsrAlg1 = 2,
    /// COO algorithm 2.
    CooAlg2 = 3,
    /// COO algorithm 3.
    CooAlg3 = 4,
    /// CSR algorithm 2.
    CsrAlg2 = 5,
    /// CSR algorithm 3.
    CsrAlg3 = 6,
    /// BSR algorithm.
    Bsr = 7,
    /// CSR algorithm 4.
    CsrAlg4 = 8,
}

/// Algorithm selector for `cusparseSpGEMM`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusparseSpGEMMAlg_t {
    /// Driver-chosen default.
    Default = 0,
    /// SpGEMM algorithm 1.
    Alg1 = 1,
    /// SpGEMM algorithm 2.
    Alg2 = 2,
    /// SpGEMM algorithm 3.
    Alg3 = 3,
    /// Memory-saving default for CSR inputs.
    CsrMemoryDefault = 4,
}

/// Algorithm selector for `cusparseSpSV`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusparseSpSVAlg_t {
    /// Driver-chosen default.
    Default = 0,
}

/// Algorithm selector for `cusparseSpSM`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusparseSpSMAlg_t {
    /// Driver-chosen default.
    Default = 0,
}

/// Algorithm selector for `cusparseSDDMM`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusparseSDDMMAlg_t {
    /// Driver-chosen default.
    Default = 0,
}

/// Algorithm selector for CSR-to-CSC conversion.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusparseCsr2CscAlg_t {
    /// CSR-to-CSC algorithm 1.
    Alg1 = 1,
    /// CSR-to-CSC algorithm 2.
    Alg2 = 2,
}

impl cusparseCsr2CscAlg_t {
    /// Alias for `cusparseCsr2CscAlg_t::Alg1`.
    pub const DEFAULT: Self = Self::Alg1;
}

/// Triangular fill mode for sparse triangular routines.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusparseFillMode_t {
    /// Lower-triangular.
    Lower = 0,
    /// Upper-triangular.
    Upper = 1,
}

/// Diagonal-unit selector for sparse triangular routines.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusparseDiagType_t {
    /// Non-unit diagonal.
    NonUnit = 0,
    /// Unit diagonal.
    Unit = 1,
}

/// Attribute key for `cusparseSpMatSetAttribute` / `*GetAttribute`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusparseSpMatAttribute_t {
    /// Fill-mode attribute (lower / upper triangular).
    FillMode = 0,
    /// Diagonal-type attribute (unit / non-unit).
    DiagType = 1,
}

/// `cudaDataType` values used by cuSPARSE / cuSOLVER's generic APIs. Only
/// the subset we actually use at v0.1.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudaDataType {
    /// 32-bit real (`f32`).
    R_32F = 0,
    /// 64-bit real (`f64`).
    R_64F = 1,
    /// 16-bit real (IEEE half / `f16`).
    R_16F = 2,
    /// 32-bit complex (`Complex<f32>`).
    C_32F = 4,
    /// 64-bit complex (`Complex<f64>`).
    C_64F = 5,
    /// 16-bit real bfloat16.
    R_16BF = 14,
}

// ---- status ---------------------------------------------------------------

/// Status / error code returned by cuSPARSE FFI calls.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct cusparseStatus_t(pub i32);

impl cusparseStatus_t {
    /// Status: success.
    pub const SUCCESS: Self = Self(0);
    /// Status: not initialized.
    pub const NOT_INITIALIZED: Self = Self(1);
    /// Status: alloc failed.
    pub const ALLOC_FAILED: Self = Self(2);
    /// Status: invalid value.
    pub const INVALID_VALUE: Self = Self(3);
    /// Status: arch mismatch.
    pub const ARCH_MISMATCH: Self = Self(4);
    /// Status: mapping error.
    pub const MAPPING_ERROR: Self = Self(5);
    /// Status: execution failed.
    pub const EXECUTION_FAILED: Self = Self(6);
    /// Status: internal error.
    pub const INTERNAL_ERROR: Self = Self(7);
    /// Status: matrix type not supported.
    pub const MATRIX_TYPE_NOT_SUPPORTED: Self = Self(8);
    /// Status: zero pivot.
    pub const ZERO_PIVOT: Self = Self(9);
    /// Status: not supported.
    pub const NOT_SUPPORTED: Self = Self(10);
    /// Status: insufficient resources.
    pub const INSUFFICIENT_RESOURCES: Self = Self(11);

    /// Returns `true` when this is the success status code.
    pub const fn is_success(self) -> bool {
        self.0 == 0
    }
}

impl CudaStatus for cusparseStatus_t {
    fn code(self) -> i32 {
        self.0
    }
    fn name(self) -> &'static str {
        match self.0 {
            0 => "CUSPARSE_STATUS_SUCCESS",
            1 => "CUSPARSE_STATUS_NOT_INITIALIZED",
            2 => "CUSPARSE_STATUS_ALLOC_FAILED",
            3 => "CUSPARSE_STATUS_INVALID_VALUE",
            4 => "CUSPARSE_STATUS_ARCH_MISMATCH",
            6 => "CUSPARSE_STATUS_EXECUTION_FAILED",
            7 => "CUSPARSE_STATUS_INTERNAL_ERROR",
            10 => "CUSPARSE_STATUS_NOT_SUPPORTED",
            _ => "CUSPARSE_STATUS_UNRECOGNIZED",
        }
    }
    fn description(self) -> &'static str {
        match self.0 {
            0 => "success",
            1 => "cuSPARSE handle not initialized",
            2 => "allocation failed",
            3 => "invalid argument",
            6 => "GPU execution failed",
            10 => "operation not supported",
            _ => "unrecognized cuSPARSE status code",
        }
    }
    fn is_success(self) -> bool {
        cusparseStatus_t::is_success(self)
    }
    fn library(self) -> &'static str {
        "cusparse"
    }
}

// ---- function-pointer types ----------------------------------------------

/// cuSPARSE: create a cuSPARSE handle. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseCreate =
    unsafe extern "C" fn(handle: *mut cusparseHandle_t) -> cusparseStatus_t;
/// cuSPARSE: destroy a cuSPARSE handle. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseDestroy = unsafe extern "C" fn(handle: cusparseHandle_t) -> cusparseStatus_t;
/// cuSPARSE: bind a CUDA stream to a cuSPARSE handle. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseSetStream =
    unsafe extern "C" fn(handle: cusparseHandle_t, stream: cudaStream_t) -> cusparseStatus_t;
/// cuSPARSE: return the cuSPARSE library version. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseGetVersion =
    unsafe extern "C" fn(handle: cusparseHandle_t, version: *mut c_int) -> cusparseStatus_t;

/// cuSPARSE: create a CSR sparse-matrix descriptor. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseCreateCsr = unsafe extern "C" fn(
    sp_mat: *mut cusparseSpMatDescr_t,
    rows: i64,
    cols: i64,
    nnz: i64,
    csr_row_offsets: *mut c_void,
    csr_col_ind: *mut c_void,
    csr_values: *mut c_void,
    csr_row_offsets_type: cusparseIndexType_t,
    csr_col_ind_type: cusparseIndexType_t,
    idx_base: cusparseIndexBase_t,
    value_type: cudaDataType,
) -> cusparseStatus_t;
/// cuSPARSE: destroy a sparse-matrix descriptor. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseDestroySpMat =
    unsafe extern "C" fn(descr: cusparseSpMatDescr_t) -> cusparseStatus_t;

/// cuSPARSE: create a dense-vector descriptor. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseCreateDnVec = unsafe extern "C" fn(
    descr: *mut cusparseDnVecDescr_t,
    size: i64,
    values: *mut c_void,
    value_type: cudaDataType,
) -> cusparseStatus_t;
/// cuSPARSE: destroy a dense-vector descriptor. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseDestroyDnVec =
    unsafe extern "C" fn(descr: cusparseDnVecDescr_t) -> cusparseStatus_t;

/// cuSPARSE: workspace-size query for sparse matrix-vector multiplication. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseSpMV_bufferSize = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    op: cusparseOperation_t,
    alpha: *const c_void,
    mat_a: cusparseSpMatDescr_t,
    vec_x: cusparseDnVecDescr_t,
    beta: *const c_void,
    vec_y: cusparseDnVecDescr_t,
    compute_type: cudaDataType,
    alg: cusparseSpMVAlg_t,
    buffer_size: *mut usize,
) -> cusparseStatus_t;

/// cuSPARSE: sparse matrix-vector multiplication. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseSpMV = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    op: cusparseOperation_t,
    alpha: *const c_void,
    mat_a: cusparseSpMatDescr_t,
    vec_x: cusparseDnVecDescr_t,
    beta: *const c_void,
    vec_y: cusparseDnVecDescr_t,
    compute_type: cudaDataType,
    alg: cusparseSpMVAlg_t,
    external_buffer: *mut c_void,
) -> cusparseStatus_t;

// ---- CSC / COO / BSR / Dense descriptors ---------------------------------

/// cuSPARSE: create a CSC sparse-matrix descriptor. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseCreateCsc = unsafe extern "C" fn(
    sp_mat: *mut cusparseSpMatDescr_t,
    rows: i64,
    cols: i64,
    nnz: i64,
    csc_col_offsets: *mut c_void,
    csc_row_ind: *mut c_void,
    csc_values: *mut c_void,
    csc_col_offsets_type: cusparseIndexType_t,
    csc_row_ind_type: cusparseIndexType_t,
    idx_base: cusparseIndexBase_t,
    value_type: cudaDataType,
) -> cusparseStatus_t;

/// cuSPARSE: create a COO sparse-matrix descriptor. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseCreateCoo = unsafe extern "C" fn(
    sp_mat: *mut cusparseSpMatDescr_t,
    rows: i64,
    cols: i64,
    nnz: i64,
    coo_row_ind: *mut c_void,
    coo_col_ind: *mut c_void,
    coo_values: *mut c_void,
    coo_idx_type: cusparseIndexType_t,
    idx_base: cusparseIndexBase_t,
    value_type: cudaDataType,
) -> cusparseStatus_t;

/// cuSPARSE: create a BSR sparse-matrix descriptor. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseCreateBsr = unsafe extern "C" fn(
    sp_mat: *mut cusparseSpMatDescr_t,
    brows: i64,
    bcols: i64,
    bnnz: i64,
    row_block_dim: i64,
    col_block_dim: i64,
    bsr_row_offsets: *mut c_void,
    bsr_col_ind: *mut c_void,
    bsr_values: *mut c_void,
    bsr_row_offsets_type: cusparseIndexType_t,
    bsr_col_ind_type: cusparseIndexType_t,
    idx_base: cusparseIndexBase_t,
    value_type: cudaDataType,
    order: cusparseOrder_t,
) -> cusparseStatus_t;

/// cuSPARSE: create a dense-matrix descriptor. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseCreateDnMat = unsafe extern "C" fn(
    descr: *mut cusparseDnMatDescr_t,
    rows: i64,
    cols: i64,
    ld: i64,
    values: *mut c_void,
    value_type: cudaDataType,
    order: cusparseOrder_t,
) -> cusparseStatus_t;

/// cuSPARSE: destroy a dense-matrix descriptor. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseDestroyDnMat =
    unsafe extern "C" fn(descr: cusparseDnMatDescr_t) -> cusparseStatus_t;

/// cuSPARSE: query rows, cols, and nnz of a sparse-matrix descriptor. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseSpMatGetSize = unsafe extern "C" fn(
    sp_mat: cusparseSpMatDescr_t,
    rows: *mut i64,
    cols: *mut i64,
    nnz: *mut i64,
) -> cusparseStatus_t;

/// cuSPARSE: set an attribute (fill mode, diag type) on a sparse-matrix descriptor. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseSpMatSetAttribute = unsafe extern "C" fn(
    sp_mat: cusparseSpMatDescr_t,
    attribute: cusparseSpMatAttribute_t,
    data: *const c_void,
    data_size: usize,
) -> cusparseStatus_t;

/// cuSPARSE: replace the row/column/value pointers on a CSR descriptor. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseCsrSetPointers = unsafe extern "C" fn(
    sp_mat: cusparseSpMatDescr_t,
    csr_row_offsets: *mut c_void,
    csr_col_ind: *mut c_void,
    csr_values: *mut c_void,
) -> cusparseStatus_t;

/// cuSPARSE: replace the column/row/value pointers on a CSC descriptor. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseCscSetPointers = unsafe extern "C" fn(
    sp_mat: cusparseSpMatDescr_t,
    csc_col_offsets: *mut c_void,
    csc_row_ind: *mut c_void,
    csc_values: *mut c_void,
) -> cusparseStatus_t;

/// cuSPARSE: replace the row/column/value pointers on a COO descriptor. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseCooSetPointers = unsafe extern "C" fn(
    sp_mat: cusparseSpMatDescr_t,
    coo_row_ind: *mut c_void,
    coo_col_ind: *mut c_void,
    coo_values: *mut c_void,
) -> cusparseStatus_t;

// ---- SpMM (sparse × dense = dense) ---------------------------------------

/// cuSPARSE: workspace-size query for sparse matrix × dense matrix multiplication. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseSpMM_bufferSize = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    op_a: cusparseOperation_t,
    op_b: cusparseOperation_t,
    alpha: *const c_void,
    mat_a: cusparseSpMatDescr_t,
    mat_b: cusparseDnMatDescr_t,
    beta: *const c_void,
    mat_c: cusparseDnMatDescr_t,
    compute_type: cudaDataType,
    alg: cusparseSpMMAlg_t,
    buffer_size: *mut usize,
) -> cusparseStatus_t;

/// cuSPARSE: preprocess stage of sparse matrix × dense matrix multiplication. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseSpMM_preprocess = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    op_a: cusparseOperation_t,
    op_b: cusparseOperation_t,
    alpha: *const c_void,
    mat_a: cusparseSpMatDescr_t,
    mat_b: cusparseDnMatDescr_t,
    beta: *const c_void,
    mat_c: cusparseDnMatDescr_t,
    compute_type: cudaDataType,
    alg: cusparseSpMMAlg_t,
    external_buffer: *mut c_void,
) -> cusparseStatus_t;

/// cuSPARSE: sparse matrix × dense matrix multiplication. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseSpMM = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    op_a: cusparseOperation_t,
    op_b: cusparseOperation_t,
    alpha: *const c_void,
    mat_a: cusparseSpMatDescr_t,
    mat_b: cusparseDnMatDescr_t,
    beta: *const c_void,
    mat_c: cusparseDnMatDescr_t,
    compute_type: cudaDataType,
    alg: cusparseSpMMAlg_t,
    external_buffer: *mut c_void,
) -> cusparseStatus_t;

// ---- SpGEMM (sparse × sparse = sparse) -----------------------------------

/// cuSPARSE: create an opaque descriptor for sparse matrix × sparse matrix multiplication. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseSpGEMM_createDescr =
    unsafe extern "C" fn(descr: *mut cusparseSpGEMMDescr_t) -> cusparseStatus_t;
/// cuSPARSE: destroy an opaque descriptor for sparse matrix × sparse matrix multiplication. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseSpGEMM_destroyDescr =
    unsafe extern "C" fn(descr: cusparseSpGEMMDescr_t) -> cusparseStatus_t;

/// cuSPARSE: work-estimation stage of sparse matrix × sparse matrix multiplication. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseSpGEMM_workEstimation = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    op_a: cusparseOperation_t,
    op_b: cusparseOperation_t,
    alpha: *const c_void,
    mat_a: cusparseSpMatDescr_t,
    mat_b: cusparseSpMatDescr_t,
    beta: *const c_void,
    mat_c: cusparseSpMatDescr_t,
    compute_type: cudaDataType,
    alg: cusparseSpGEMMAlg_t,
    descr: cusparseSpGEMMDescr_t,
    buffer_size1: *mut usize,
    external_buffer1: *mut c_void,
) -> cusparseStatus_t;

/// cuSPARSE: compute stage of sparse matrix × sparse matrix multiplication. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseSpGEMM_compute = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    op_a: cusparseOperation_t,
    op_b: cusparseOperation_t,
    alpha: *const c_void,
    mat_a: cusparseSpMatDescr_t,
    mat_b: cusparseSpMatDescr_t,
    beta: *const c_void,
    mat_c: cusparseSpMatDescr_t,
    compute_type: cudaDataType,
    alg: cusparseSpGEMMAlg_t,
    descr: cusparseSpGEMMDescr_t,
    buffer_size2: *mut usize,
    external_buffer2: *mut c_void,
) -> cusparseStatus_t;

/// cuSPARSE: copy stage of sparse matrix × sparse matrix multiplication. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseSpGEMM_copy = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    op_a: cusparseOperation_t,
    op_b: cusparseOperation_t,
    alpha: *const c_void,
    mat_a: cusparseSpMatDescr_t,
    mat_b: cusparseSpMatDescr_t,
    beta: *const c_void,
    mat_c: cusparseSpMatDescr_t,
    compute_type: cudaDataType,
    alg: cusparseSpGEMMAlg_t,
    descr: cusparseSpGEMMDescr_t,
) -> cusparseStatus_t;

// ---- SpSV (sparse triangular solve, vector) ------------------------------

/// cuSPARSE: create an opaque descriptor for sparse triangular linear solve (single right-hand side). See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseSpSV_createDescr =
    unsafe extern "C" fn(descr: *mut cusparseSpSVDescr_t) -> cusparseStatus_t;
/// cuSPARSE: destroy an opaque descriptor for sparse triangular linear solve (single right-hand side). See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseSpSV_destroyDescr =
    unsafe extern "C" fn(descr: cusparseSpSVDescr_t) -> cusparseStatus_t;

/// cuSPARSE: workspace-size query for sparse triangular linear solve (single right-hand side). See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseSpSV_bufferSize = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    op_a: cusparseOperation_t,
    alpha: *const c_void,
    mat_a: cusparseSpMatDescr_t,
    vec_x: cusparseDnVecDescr_t,
    vec_y: cusparseDnVecDescr_t,
    compute_type: cudaDataType,
    alg: cusparseSpSVAlg_t,
    descr: cusparseSpSVDescr_t,
    buffer_size: *mut usize,
) -> cusparseStatus_t;

/// cuSPARSE: analysis stage of sparse triangular linear solve (single right-hand side). See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseSpSV_analysis = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    op_a: cusparseOperation_t,
    alpha: *const c_void,
    mat_a: cusparseSpMatDescr_t,
    vec_x: cusparseDnVecDescr_t,
    vec_y: cusparseDnVecDescr_t,
    compute_type: cudaDataType,
    alg: cusparseSpSVAlg_t,
    descr: cusparseSpSVDescr_t,
    external_buffer: *mut c_void,
) -> cusparseStatus_t;

/// cuSPARSE: solve stage of sparse triangular linear solve (single right-hand side). See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseSpSV_solve = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    op_a: cusparseOperation_t,
    alpha: *const c_void,
    mat_a: cusparseSpMatDescr_t,
    vec_x: cusparseDnVecDescr_t,
    vec_y: cusparseDnVecDescr_t,
    compute_type: cudaDataType,
    alg: cusparseSpSVAlg_t,
    descr: cusparseSpSVDescr_t,
) -> cusparseStatus_t;

// ---- SpSM (sparse triangular solve, matrix) ------------------------------

/// cuSPARSE: create an opaque descriptor for sparse triangular linear solve (multiple right-hand sides). See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseSpSM_createDescr =
    unsafe extern "C" fn(descr: *mut cusparseSpSMDescr_t) -> cusparseStatus_t;
/// cuSPARSE: destroy an opaque descriptor for sparse triangular linear solve (multiple right-hand sides). See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseSpSM_destroyDescr =
    unsafe extern "C" fn(descr: cusparseSpSMDescr_t) -> cusparseStatus_t;

/// cuSPARSE: workspace-size query for sparse triangular linear solve (multiple right-hand sides). See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseSpSM_bufferSize = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    op_a: cusparseOperation_t,
    op_b: cusparseOperation_t,
    alpha: *const c_void,
    mat_a: cusparseSpMatDescr_t,
    mat_b: cusparseDnMatDescr_t,
    mat_c: cusparseDnMatDescr_t,
    compute_type: cudaDataType,
    alg: cusparseSpSMAlg_t,
    descr: cusparseSpSMDescr_t,
    buffer_size: *mut usize,
) -> cusparseStatus_t;

/// cuSPARSE: analysis stage of sparse triangular linear solve (multiple right-hand sides). See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseSpSM_analysis = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    op_a: cusparseOperation_t,
    op_b: cusparseOperation_t,
    alpha: *const c_void,
    mat_a: cusparseSpMatDescr_t,
    mat_b: cusparseDnMatDescr_t,
    mat_c: cusparseDnMatDescr_t,
    compute_type: cudaDataType,
    alg: cusparseSpSMAlg_t,
    descr: cusparseSpSMDescr_t,
    external_buffer: *mut c_void,
) -> cusparseStatus_t;

/// cuSPARSE: solve stage of sparse triangular linear solve (multiple right-hand sides). See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseSpSM_solve = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    op_a: cusparseOperation_t,
    op_b: cusparseOperation_t,
    alpha: *const c_void,
    mat_a: cusparseSpMatDescr_t,
    mat_b: cusparseDnMatDescr_t,
    mat_c: cusparseDnMatDescr_t,
    compute_type: cudaDataType,
    alg: cusparseSpSMAlg_t,
    descr: cusparseSpSMDescr_t,
) -> cusparseStatus_t;

// ---- SDDMM (sampled dense-dense matmul) ----------------------------------

/// cuSPARSE: workspace-size query for sampled dense-dense matrix multiplication. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseSDDMM_bufferSize = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    op_a: cusparseOperation_t,
    op_b: cusparseOperation_t,
    alpha: *const c_void,
    mat_a: cusparseDnMatDescr_t,
    mat_b: cusparseDnMatDescr_t,
    beta: *const c_void,
    mat_c: cusparseSpMatDescr_t,
    compute_type: cudaDataType,
    alg: cusparseSDDMMAlg_t,
    buffer_size: *mut usize,
) -> cusparseStatus_t;

/// cuSPARSE: preprocess stage of sampled dense-dense matrix multiplication. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseSDDMM_preprocess = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    op_a: cusparseOperation_t,
    op_b: cusparseOperation_t,
    alpha: *const c_void,
    mat_a: cusparseDnMatDescr_t,
    mat_b: cusparseDnMatDescr_t,
    beta: *const c_void,
    mat_c: cusparseSpMatDescr_t,
    compute_type: cudaDataType,
    alg: cusparseSDDMMAlg_t,
    external_buffer: *mut c_void,
) -> cusparseStatus_t;

/// cuSPARSE: sampled dense-dense matrix multiplication. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseSDDMM = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    op_a: cusparseOperation_t,
    op_b: cusparseOperation_t,
    alpha: *const c_void,
    mat_a: cusparseDnMatDescr_t,
    mat_b: cusparseDnMatDescr_t,
    beta: *const c_void,
    mat_c: cusparseSpMatDescr_t,
    compute_type: cudaDataType,
    alg: cusparseSDDMMAlg_t,
    external_buffer: *mut c_void,
) -> cusparseStatus_t;

// ---- CSR ↔ CSC conversion -------------------------------------------------

/// cuSPARSE: workspace-size query for `cusparseCsr2cscEx2`. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseCsr2cscEx2_bufferSize = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    m: c_int,
    n: c_int,
    nnz: c_int,
    csr_val: *const c_void,
    csr_row_ptr: *const c_int,
    csr_col_ind: *const c_int,
    csc_val: *mut c_void,
    csc_col_ptr: *mut c_int,
    csc_row_ind: *mut c_int,
    value_type: cudaDataType,
    copy_values: c_int,
    idx_base: cusparseIndexBase_t,
    alg: cusparseCsr2CscAlg_t,
    buffer_size: *mut usize,
) -> cusparseStatus_t;

/// cuSPARSE: convert a CSR matrix to CSC (extended API). See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseCsr2cscEx2 = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    m: c_int,
    n: c_int,
    nnz: c_int,
    csr_val: *const c_void,
    csr_row_ptr: *const c_int,
    csr_col_ind: *const c_int,
    csc_val: *mut c_void,
    csc_col_ptr: *mut c_int,
    csc_row_ind: *mut c_int,
    value_type: cudaDataType,
    copy_values: c_int,
    idx_base: cusparseIndexBase_t,
    alg: cusparseCsr2CscAlg_t,
    buffer: *mut c_void,
) -> cusparseStatus_t;

// ---- Sparse↔Dense conversion ---------------------------------------------

/// cuSPARSE: workspace-size query for sparse-to-dense conversion. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseSparseToDense_bufferSize = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    mat_a: cusparseSpMatDescr_t,
    mat_b: cusparseDnMatDescr_t,
    alg: c_int,
    buffer_size: *mut usize,
) -> cusparseStatus_t;

/// cuSPARSE: materialize a sparse matrix into a dense one. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseSparseToDense = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    mat_a: cusparseSpMatDescr_t,
    mat_b: cusparseDnMatDescr_t,
    alg: c_int,
    external_buffer: *mut c_void,
) -> cusparseStatus_t;

/// cuSPARSE: workspace-size query for dense-to-sparse conversion. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseDenseToSparse_bufferSize = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    mat_a: cusparseDnMatDescr_t,
    mat_b: cusparseSpMatDescr_t,
    alg: c_int,
    buffer_size: *mut usize,
) -> cusparseStatus_t;

/// cuSPARSE: analysis stage for dense-to-sparse conversion. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseDenseToSparse_analysis = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    mat_a: cusparseDnMatDescr_t,
    mat_b: cusparseSpMatDescr_t,
    alg: c_int,
    external_buffer: *mut c_void,
) -> cusparseStatus_t;

/// cuSPARSE: execute stage for dense-to-sparse conversion. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseDenseToSparse_convert = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    mat_a: cusparseDnMatDescr_t,
    mat_b: cusparseSpMatDescr_t,
    alg: c_int,
    external_buffer: *mut c_void,
) -> cusparseStatus_t;

// ---- Axpby / Gather / Scatter / Rot (sparse BLAS L1) --------------------

/// cuSPARSE: scaled vector addition (alpha*x + beta*y → y). See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseAxpby = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    alpha: *const c_void,
    vec_x: cusparseDnVecDescr_t,
    beta: *const c_void,
    vec_y: cusparseDnVecDescr_t,
) -> cusparseStatus_t;

/// cuSPARSE: gather dense entries into a sparse vector. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseGather = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    vec_y: cusparseDnVecDescr_t,
    vec_x: cusparseDnVecDescr_t,
) -> cusparseStatus_t;

/// cuSPARSE: scatter sparse vector entries into a dense vector. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseScatter = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    vec_x: cusparseDnVecDescr_t,
    vec_y: cusparseDnVecDescr_t,
) -> cusparseStatus_t;

/// cuSPARSE: Givens-rotation on sparse and dense vector pair. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
pub type PFN_cusparseRot = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    c: *const c_void,
    s: *const c_void,
    vec_x: cusparseDnVecDescr_t,
    vec_y: cusparseDnVecDescr_t,
) -> cusparseStatus_t;

// ---- loader --------------------------------------------------------------

fn cusparse_candidates() -> Vec<String> {
    platform::versioned_library_candidates("cusparse", &["13", "12", "11"])
}

macro_rules! cusparse_fns {
    ($($(#[$m:meta])* $name:ident as $sym:literal : $pfn:ty);* $(;)?) => {
        /// Loaded cuSPARSE shared library plus a per-symbol `OnceLock` of function pointers.
        pub struct Cusparse {
            lib: Library,
            $($name: OnceLock<$pfn>,)*
        }
        impl core::fmt::Debug for Cusparse {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct("Cusparse").field("lib", &self.lib).finish_non_exhaustive()
            }
        }
        impl Cusparse {
            $(
                $(#[$m])*
                pub fn $name(&self) -> Result<$pfn, LoaderError> {
                    if let Some(&p) = self.$name.get() { return Ok(p); }
                    let raw: *mut () = unsafe { self.lib.raw_symbol($sym)? };
                    let p: $pfn = unsafe { core::mem::transmute_copy::<*mut (), $pfn>(&raw) };
                    let _ = self.$name.set(p);
                    Ok(p)
                }
            )*
            fn empty(lib: Library) -> Self {
                Self { lib, $($name: OnceLock::new(),)* }
            }
        }
    };
}

cusparse_fns! {
    /// cuSPARSE: create a cuSPARSE handle. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_create as "cusparseCreate": PFN_cusparseCreate;
    /// cuSPARSE: destroy a cuSPARSE handle. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_destroy as "cusparseDestroy": PFN_cusparseDestroy;
    /// cuSPARSE: bind a CUDA stream to a cuSPARSE handle. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_set_stream as "cusparseSetStream": PFN_cusparseSetStream;
    /// cuSPARSE: return the cuSPARSE library version. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_get_version as "cusparseGetVersion": PFN_cusparseGetVersion;
    // Sparse-matrix descriptors
    /// cuSPARSE: create a CSR sparse-matrix descriptor. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_create_csr as "cusparseCreateCsr": PFN_cusparseCreateCsr;
    /// cuSPARSE: create a CSC sparse-matrix descriptor. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_create_csc as "cusparseCreateCsc": PFN_cusparseCreateCsc;
    /// cuSPARSE: create a COO sparse-matrix descriptor. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_create_coo as "cusparseCreateCoo": PFN_cusparseCreateCoo;
    /// cuSPARSE: create a BSR sparse-matrix descriptor. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_create_bsr as "cusparseCreateBsr": PFN_cusparseCreateBsr;
    /// cuSPARSE: destroy a sparse-matrix descriptor. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_destroy_sp_mat as "cusparseDestroySpMat": PFN_cusparseDestroySpMat;
    /// cuSPARSE: query rows, cols, and nnz of a sparse-matrix descriptor. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_sp_mat_get_size as "cusparseSpMatGetSize": PFN_cusparseSpMatGetSize;
    /// cuSPARSE: set an attribute (fill mode, diag type) on a sparse-matrix descriptor. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_sp_mat_set_attribute as "cusparseSpMatSetAttribute": PFN_cusparseSpMatSetAttribute;
    /// cuSPARSE: replace the row/column/value pointers on a CSR descriptor. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_csr_set_pointers as "cusparseCsrSetPointers": PFN_cusparseCsrSetPointers;
    /// cuSPARSE: replace the column/row/value pointers on a CSC descriptor. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_csc_set_pointers as "cusparseCscSetPointers": PFN_cusparseCscSetPointers;
    /// cuSPARSE: replace the row/column/value pointers on a COO descriptor. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_coo_set_pointers as "cusparseCooSetPointers": PFN_cusparseCooSetPointers;
    // Dense descriptors
    /// cuSPARSE: create a dense-vector descriptor. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_create_dn_vec as "cusparseCreateDnVec": PFN_cusparseCreateDnVec;
    /// cuSPARSE: destroy a dense-vector descriptor. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_destroy_dn_vec as "cusparseDestroyDnVec": PFN_cusparseDestroyDnVec;
    /// cuSPARSE: create a dense-matrix descriptor. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_create_dn_mat as "cusparseCreateDnMat": PFN_cusparseCreateDnMat;
    /// cuSPARSE: destroy a dense-matrix descriptor. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_destroy_dn_mat as "cusparseDestroyDnMat": PFN_cusparseDestroyDnMat;
    // SpMV
    /// cuSPARSE: workspace-size query for sparse matrix-vector multiplication. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_spmv_buffer_size as "cusparseSpMV_bufferSize": PFN_cusparseSpMV_bufferSize;
    /// cuSPARSE: sparse matrix-vector multiplication. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_spmv as "cusparseSpMV": PFN_cusparseSpMV;
    // SpMM
    /// cuSPARSE: workspace-size query for sparse matrix × dense matrix multiplication. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_spmm_buffer_size as "cusparseSpMM_bufferSize": PFN_cusparseSpMM_bufferSize;
    /// cuSPARSE: preprocess stage of sparse matrix × dense matrix multiplication. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_spmm_preprocess as "cusparseSpMM_preprocess": PFN_cusparseSpMM_preprocess;
    /// cuSPARSE: sparse matrix × dense matrix multiplication. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_spmm as "cusparseSpMM": PFN_cusparseSpMM;
    // SpGEMM
    /// cuSPARSE: create an opaque descriptor for sparse matrix × sparse matrix multiplication. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_spgemm_create_descr as "cusparseSpGEMM_createDescr": PFN_cusparseSpGEMM_createDescr;
    /// cuSPARSE: destroy an opaque descriptor for sparse matrix × sparse matrix multiplication. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_spgemm_destroy_descr as "cusparseSpGEMM_destroyDescr": PFN_cusparseSpGEMM_destroyDescr;
    /// cuSPARSE: work-estimation stage of sparse matrix × sparse matrix multiplication. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_spgemm_work_estimation as "cusparseSpGEMM_workEstimation": PFN_cusparseSpGEMM_workEstimation;
    /// cuSPARSE: compute stage of sparse matrix × sparse matrix multiplication. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_spgemm_compute as "cusparseSpGEMM_compute": PFN_cusparseSpGEMM_compute;
    /// cuSPARSE: copy stage of sparse matrix × sparse matrix multiplication. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_spgemm_copy as "cusparseSpGEMM_copy": PFN_cusparseSpGEMM_copy;
    // SpSV
    /// cuSPARSE: create an opaque descriptor for sparse triangular linear solve (single right-hand side). See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_spsv_create_descr as "cusparseSpSV_createDescr": PFN_cusparseSpSV_createDescr;
    /// cuSPARSE: destroy an opaque descriptor for sparse triangular linear solve (single right-hand side). See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_spsv_destroy_descr as "cusparseSpSV_destroyDescr": PFN_cusparseSpSV_destroyDescr;
    /// cuSPARSE: workspace-size query for sparse triangular linear solve (single right-hand side). See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_spsv_buffer_size as "cusparseSpSV_bufferSize": PFN_cusparseSpSV_bufferSize;
    /// cuSPARSE: analysis stage of sparse triangular linear solve (single right-hand side). See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_spsv_analysis as "cusparseSpSV_analysis": PFN_cusparseSpSV_analysis;
    /// cuSPARSE: solve stage of sparse triangular linear solve (single right-hand side). See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_spsv_solve as "cusparseSpSV_solve": PFN_cusparseSpSV_solve;
    // SpSM
    /// cuSPARSE: create an opaque descriptor for sparse triangular linear solve (multiple right-hand sides). See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_spsm_create_descr as "cusparseSpSM_createDescr": PFN_cusparseSpSM_createDescr;
    /// cuSPARSE: destroy an opaque descriptor for sparse triangular linear solve (multiple right-hand sides). See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_spsm_destroy_descr as "cusparseSpSM_destroyDescr": PFN_cusparseSpSM_destroyDescr;
    /// cuSPARSE: workspace-size query for sparse triangular linear solve (multiple right-hand sides). See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_spsm_buffer_size as "cusparseSpSM_bufferSize": PFN_cusparseSpSM_bufferSize;
    /// cuSPARSE: analysis stage of sparse triangular linear solve (multiple right-hand sides). See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_spsm_analysis as "cusparseSpSM_analysis": PFN_cusparseSpSM_analysis;
    /// cuSPARSE: solve stage of sparse triangular linear solve (multiple right-hand sides). See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_spsm_solve as "cusparseSpSM_solve": PFN_cusparseSpSM_solve;
    // SDDMM
    /// cuSPARSE: workspace-size query for sampled dense-dense matrix multiplication. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_sddmm_buffer_size as "cusparseSDDMM_bufferSize": PFN_cusparseSDDMM_bufferSize;
    /// cuSPARSE: preprocess stage of sampled dense-dense matrix multiplication. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_sddmm_preprocess as "cusparseSDDMM_preprocess": PFN_cusparseSDDMM_preprocess;
    /// cuSPARSE: sampled dense-dense matrix multiplication. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_sddmm as "cusparseSDDMM": PFN_cusparseSDDMM;
    // CSR ↔ CSC
    /// cuSPARSE: workspace-size query for `cusparseCsr2cscEx2`. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_csr2csc_ex2_buffer_size as "cusparseCsr2cscEx2_bufferSize": PFN_cusparseCsr2cscEx2_bufferSize;
    /// cuSPARSE: convert a CSR matrix to CSC (extended API). See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_csr2csc_ex2 as "cusparseCsr2cscEx2": PFN_cusparseCsr2cscEx2;
    // Sparse ↔ Dense
    /// cuSPARSE: workspace-size query for sparse-to-dense conversion. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_sparse_to_dense_buffer_size as "cusparseSparseToDense_bufferSize": PFN_cusparseSparseToDense_bufferSize;
    /// cuSPARSE: materialize a sparse matrix into a dense one. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_sparse_to_dense as "cusparseSparseToDense": PFN_cusparseSparseToDense;
    /// cuSPARSE: workspace-size query for dense-to-sparse conversion. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_dense_to_sparse_buffer_size as "cusparseDenseToSparse_bufferSize": PFN_cusparseDenseToSparse_bufferSize;
    /// cuSPARSE: analysis stage for dense-to-sparse conversion. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_dense_to_sparse_analysis as "cusparseDenseToSparse_analysis": PFN_cusparseDenseToSparse_analysis;
    /// cuSPARSE: execute stage for dense-to-sparse conversion. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_dense_to_sparse_convert as "cusparseDenseToSparse_convert": PFN_cusparseDenseToSparse_convert;
    // Sparse BLAS L1
    /// cuSPARSE: scaled vector addition (alpha*x + beta*y → y). See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_axpby as "cusparseAxpby": PFN_cusparseAxpby;
    /// cuSPARSE: gather dense entries into a sparse vector. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_gather as "cusparseGather": PFN_cusparseGather;
    /// cuSPARSE: scatter sparse vector entries into a dense vector. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_scatter as "cusparseScatter": PFN_cusparseScatter;
    /// cuSPARSE: Givens-rotation on sparse and dense vector pair. See <https://docs.nvidia.com/cuda/cusparse/index.html>.
    cusparse_rot as "cusparseRot": PFN_cusparseRot;
}

/// Lazy-load the cuSPARSE shared library and return its function-pointer table.
pub fn cusparse() -> Result<&'static Cusparse, LoaderError> {
    static CUSPARSE: OnceLock<Cusparse> = OnceLock::new();
    if let Some(c) = CUSPARSE.get() {
        return Ok(c);
    }
    let candidates: Vec<&'static str> = cusparse_candidates()
        .into_iter()
        .map(|s| Box::leak(s.into_boxed_str()) as &'static str)
        .collect();
    let candidates_leaked: &'static [&'static str] = Box::leak(candidates.into_boxed_slice());
    let lib = Library::open("cusparse", candidates_leaked)?;
    let c = Cusparse::empty(lib);
    let _ = CUSPARSE.set(c);
    Ok(CUSPARSE.get().expect("OnceLock set or lost race"))
}
