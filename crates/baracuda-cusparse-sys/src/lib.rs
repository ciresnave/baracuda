//! Raw FFI + dynamic loader for NVIDIA cuSPARSE (generic API subset).

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
#![warn(missing_debug_implementations)]

use core::ffi::{c_int, c_void};
use std::sync::OnceLock;

use baracuda_core::{platform, Library, LoaderError};
use baracuda_cuda_sys::runtime::cudaStream_t;
use baracuda_types::CudaStatus;

// ---- handles --------------------------------------------------------------

pub type cusparseHandle_t = *mut c_void;
pub type cusparseSpMatDescr_t = *mut c_void;
pub type cusparseDnMatDescr_t = *mut c_void;
pub type cusparseDnVecDescr_t = *mut c_void;
pub type cusparseSpGEMMDescr_t = *mut c_void;
pub type cusparseSpSVDescr_t = *mut c_void;
pub type cusparseSpSMDescr_t = *mut c_void;
pub type cusparseMatDescr_t = *mut c_void;

// ---- enums ----------------------------------------------------------------

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusparseOperation_t {
    N = 0,
    T = 1,
    C = 2,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusparseIndexType_t {
    I16U = 1,
    I32I = 2,
    I64I = 3,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusparseIndexBase_t {
    Zero = 0,
    One = 1,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusparseOrder_t {
    Row = 1,
    Col = 2,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusparseSpMVAlg_t {
    /// Driver-chosen default.
    Default = 0,
    /// CSR algorithm 1 (deterministic).
    CsrAlg1 = 2,
    /// CSR algorithm 2 (higher throughput).
    CsrAlg2 = 3,
    CooAlg1 = 1,
    CooAlg2 = 4,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusparseSpMMAlg_t {
    Default = 0,
    CooAlg1 = 1,
    CsrAlg1 = 2,
    CooAlg2 = 3,
    CooAlg3 = 4,
    CsrAlg2 = 5,
    CsrAlg3 = 6,
    Bsr = 7,
    CsrAlg4 = 8,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusparseSpGEMMAlg_t {
    Default = 0,
    Alg1 = 1,
    Alg2 = 2,
    Alg3 = 3,
    CsrMemoryDefault = 4,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusparseSpSVAlg_t {
    Default = 0,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusparseSpSMAlg_t {
    Default = 0,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusparseSDDMMAlg_t {
    Default = 0,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusparseCsr2CscAlg_t {
    Alg1 = 1,
    Alg2 = 2,
}

impl cusparseCsr2CscAlg_t {
    pub const DEFAULT: Self = Self::Alg1;
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusparseFillMode_t {
    Lower = 0,
    Upper = 1,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusparseDiagType_t {
    NonUnit = 0,
    Unit = 1,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cusparseSpMatAttribute_t {
    FillMode = 0,
    DiagType = 1,
}

/// `cudaDataType` values used by cuSPARSE / cuSOLVER's generic APIs. Only
/// the subset we actually use at v0.1.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudaDataType {
    R_32F = 0,
    R_64F = 1,
    R_16F = 2,
    C_32F = 4,
    C_64F = 5,
    R_16BF = 14,
}

// ---- status ---------------------------------------------------------------

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct cusparseStatus_t(pub i32);

impl cusparseStatus_t {
    pub const SUCCESS: Self = Self(0);
    pub const NOT_INITIALIZED: Self = Self(1);
    pub const ALLOC_FAILED: Self = Self(2);
    pub const INVALID_VALUE: Self = Self(3);
    pub const ARCH_MISMATCH: Self = Self(4);
    pub const MAPPING_ERROR: Self = Self(5);
    pub const EXECUTION_FAILED: Self = Self(6);
    pub const INTERNAL_ERROR: Self = Self(7);
    pub const MATRIX_TYPE_NOT_SUPPORTED: Self = Self(8);
    pub const ZERO_PIVOT: Self = Self(9);
    pub const NOT_SUPPORTED: Self = Self(10);
    pub const INSUFFICIENT_RESOURCES: Self = Self(11);

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

pub type PFN_cusparseCreate =
    unsafe extern "C" fn(handle: *mut cusparseHandle_t) -> cusparseStatus_t;
pub type PFN_cusparseDestroy = unsafe extern "C" fn(handle: cusparseHandle_t) -> cusparseStatus_t;
pub type PFN_cusparseSetStream =
    unsafe extern "C" fn(handle: cusparseHandle_t, stream: cudaStream_t) -> cusparseStatus_t;
pub type PFN_cusparseGetVersion =
    unsafe extern "C" fn(handle: cusparseHandle_t, version: *mut c_int) -> cusparseStatus_t;

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
pub type PFN_cusparseDestroySpMat =
    unsafe extern "C" fn(descr: cusparseSpMatDescr_t) -> cusparseStatus_t;

pub type PFN_cusparseCreateDnVec = unsafe extern "C" fn(
    descr: *mut cusparseDnVecDescr_t,
    size: i64,
    values: *mut c_void,
    value_type: cudaDataType,
) -> cusparseStatus_t;
pub type PFN_cusparseDestroyDnVec =
    unsafe extern "C" fn(descr: cusparseDnVecDescr_t) -> cusparseStatus_t;

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

pub type PFN_cusparseCreateDnMat = unsafe extern "C" fn(
    descr: *mut cusparseDnMatDescr_t,
    rows: i64,
    cols: i64,
    ld: i64,
    values: *mut c_void,
    value_type: cudaDataType,
    order: cusparseOrder_t,
) -> cusparseStatus_t;

pub type PFN_cusparseDestroyDnMat =
    unsafe extern "C" fn(descr: cusparseDnMatDescr_t) -> cusparseStatus_t;

pub type PFN_cusparseSpMatGetSize = unsafe extern "C" fn(
    sp_mat: cusparseSpMatDescr_t,
    rows: *mut i64,
    cols: *mut i64,
    nnz: *mut i64,
) -> cusparseStatus_t;

pub type PFN_cusparseSpMatSetAttribute = unsafe extern "C" fn(
    sp_mat: cusparseSpMatDescr_t,
    attribute: cusparseSpMatAttribute_t,
    data: *const c_void,
    data_size: usize,
) -> cusparseStatus_t;

pub type PFN_cusparseCsrSetPointers = unsafe extern "C" fn(
    sp_mat: cusparseSpMatDescr_t,
    csr_row_offsets: *mut c_void,
    csr_col_ind: *mut c_void,
    csr_values: *mut c_void,
) -> cusparseStatus_t;

pub type PFN_cusparseCscSetPointers = unsafe extern "C" fn(
    sp_mat: cusparseSpMatDescr_t,
    csc_col_offsets: *mut c_void,
    csc_row_ind: *mut c_void,
    csc_values: *mut c_void,
) -> cusparseStatus_t;

pub type PFN_cusparseCooSetPointers = unsafe extern "C" fn(
    sp_mat: cusparseSpMatDescr_t,
    coo_row_ind: *mut c_void,
    coo_col_ind: *mut c_void,
    coo_values: *mut c_void,
) -> cusparseStatus_t;

// ---- SpMM (sparse × dense = dense) ---------------------------------------

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

pub type PFN_cusparseSpGEMM_createDescr =
    unsafe extern "C" fn(descr: *mut cusparseSpGEMMDescr_t) -> cusparseStatus_t;
pub type PFN_cusparseSpGEMM_destroyDescr =
    unsafe extern "C" fn(descr: cusparseSpGEMMDescr_t) -> cusparseStatus_t;

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

pub type PFN_cusparseSpSV_createDescr =
    unsafe extern "C" fn(descr: *mut cusparseSpSVDescr_t) -> cusparseStatus_t;
pub type PFN_cusparseSpSV_destroyDescr =
    unsafe extern "C" fn(descr: cusparseSpSVDescr_t) -> cusparseStatus_t;

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

pub type PFN_cusparseSpSM_createDescr =
    unsafe extern "C" fn(descr: *mut cusparseSpSMDescr_t) -> cusparseStatus_t;
pub type PFN_cusparseSpSM_destroyDescr =
    unsafe extern "C" fn(descr: cusparseSpSMDescr_t) -> cusparseStatus_t;

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

pub type PFN_cusparseSparseToDense_bufferSize = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    mat_a: cusparseSpMatDescr_t,
    mat_b: cusparseDnMatDescr_t,
    alg: c_int,
    buffer_size: *mut usize,
) -> cusparseStatus_t;

pub type PFN_cusparseSparseToDense = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    mat_a: cusparseSpMatDescr_t,
    mat_b: cusparseDnMatDescr_t,
    alg: c_int,
    external_buffer: *mut c_void,
) -> cusparseStatus_t;

pub type PFN_cusparseDenseToSparse_bufferSize = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    mat_a: cusparseDnMatDescr_t,
    mat_b: cusparseSpMatDescr_t,
    alg: c_int,
    buffer_size: *mut usize,
) -> cusparseStatus_t;

pub type PFN_cusparseDenseToSparse_analysis = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    mat_a: cusparseDnMatDescr_t,
    mat_b: cusparseSpMatDescr_t,
    alg: c_int,
    external_buffer: *mut c_void,
) -> cusparseStatus_t;

pub type PFN_cusparseDenseToSparse_convert = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    mat_a: cusparseDnMatDescr_t,
    mat_b: cusparseSpMatDescr_t,
    alg: c_int,
    external_buffer: *mut c_void,
) -> cusparseStatus_t;

// ---- Axpby / Gather / Scatter / Rot (sparse BLAS L1) --------------------

pub type PFN_cusparseAxpby = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    alpha: *const c_void,
    vec_x: cusparseDnVecDescr_t,
    beta: *const c_void,
    vec_y: cusparseDnVecDescr_t,
) -> cusparseStatus_t;

pub type PFN_cusparseGather = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    vec_y: cusparseDnVecDescr_t,
    vec_x: cusparseDnVecDescr_t,
) -> cusparseStatus_t;

pub type PFN_cusparseScatter = unsafe extern "C" fn(
    handle: cusparseHandle_t,
    vec_x: cusparseDnVecDescr_t,
    vec_y: cusparseDnVecDescr_t,
) -> cusparseStatus_t;

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
    ($($name:ident as $sym:literal : $pfn:ty);* $(;)?) => {
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
    cusparse_create as "cusparseCreate": PFN_cusparseCreate;
    cusparse_destroy as "cusparseDestroy": PFN_cusparseDestroy;
    cusparse_set_stream as "cusparseSetStream": PFN_cusparseSetStream;
    cusparse_get_version as "cusparseGetVersion": PFN_cusparseGetVersion;
    // Sparse-matrix descriptors
    cusparse_create_csr as "cusparseCreateCsr": PFN_cusparseCreateCsr;
    cusparse_create_csc as "cusparseCreateCsc": PFN_cusparseCreateCsc;
    cusparse_create_coo as "cusparseCreateCoo": PFN_cusparseCreateCoo;
    cusparse_create_bsr as "cusparseCreateBsr": PFN_cusparseCreateBsr;
    cusparse_destroy_sp_mat as "cusparseDestroySpMat": PFN_cusparseDestroySpMat;
    cusparse_sp_mat_get_size as "cusparseSpMatGetSize": PFN_cusparseSpMatGetSize;
    cusparse_sp_mat_set_attribute as "cusparseSpMatSetAttribute": PFN_cusparseSpMatSetAttribute;
    cusparse_csr_set_pointers as "cusparseCsrSetPointers": PFN_cusparseCsrSetPointers;
    cusparse_csc_set_pointers as "cusparseCscSetPointers": PFN_cusparseCscSetPointers;
    cusparse_coo_set_pointers as "cusparseCooSetPointers": PFN_cusparseCooSetPointers;
    // Dense descriptors
    cusparse_create_dn_vec as "cusparseCreateDnVec": PFN_cusparseCreateDnVec;
    cusparse_destroy_dn_vec as "cusparseDestroyDnVec": PFN_cusparseDestroyDnVec;
    cusparse_create_dn_mat as "cusparseCreateDnMat": PFN_cusparseCreateDnMat;
    cusparse_destroy_dn_mat as "cusparseDestroyDnMat": PFN_cusparseDestroyDnMat;
    // SpMV
    cusparse_spmv_buffer_size as "cusparseSpMV_bufferSize": PFN_cusparseSpMV_bufferSize;
    cusparse_spmv as "cusparseSpMV": PFN_cusparseSpMV;
    // SpMM
    cusparse_spmm_buffer_size as "cusparseSpMM_bufferSize": PFN_cusparseSpMM_bufferSize;
    cusparse_spmm_preprocess as "cusparseSpMM_preprocess": PFN_cusparseSpMM_preprocess;
    cusparse_spmm as "cusparseSpMM": PFN_cusparseSpMM;
    // SpGEMM
    cusparse_spgemm_create_descr as "cusparseSpGEMM_createDescr": PFN_cusparseSpGEMM_createDescr;
    cusparse_spgemm_destroy_descr as "cusparseSpGEMM_destroyDescr": PFN_cusparseSpGEMM_destroyDescr;
    cusparse_spgemm_work_estimation as "cusparseSpGEMM_workEstimation": PFN_cusparseSpGEMM_workEstimation;
    cusparse_spgemm_compute as "cusparseSpGEMM_compute": PFN_cusparseSpGEMM_compute;
    cusparse_spgemm_copy as "cusparseSpGEMM_copy": PFN_cusparseSpGEMM_copy;
    // SpSV
    cusparse_spsv_create_descr as "cusparseSpSV_createDescr": PFN_cusparseSpSV_createDescr;
    cusparse_spsv_destroy_descr as "cusparseSpSV_destroyDescr": PFN_cusparseSpSV_destroyDescr;
    cusparse_spsv_buffer_size as "cusparseSpSV_bufferSize": PFN_cusparseSpSV_bufferSize;
    cusparse_spsv_analysis as "cusparseSpSV_analysis": PFN_cusparseSpSV_analysis;
    cusparse_spsv_solve as "cusparseSpSV_solve": PFN_cusparseSpSV_solve;
    // SpSM
    cusparse_spsm_create_descr as "cusparseSpSM_createDescr": PFN_cusparseSpSM_createDescr;
    cusparse_spsm_destroy_descr as "cusparseSpSM_destroyDescr": PFN_cusparseSpSM_destroyDescr;
    cusparse_spsm_buffer_size as "cusparseSpSM_bufferSize": PFN_cusparseSpSM_bufferSize;
    cusparse_spsm_analysis as "cusparseSpSM_analysis": PFN_cusparseSpSM_analysis;
    cusparse_spsm_solve as "cusparseSpSM_solve": PFN_cusparseSpSM_solve;
    // SDDMM
    cusparse_sddmm_buffer_size as "cusparseSDDMM_bufferSize": PFN_cusparseSDDMM_bufferSize;
    cusparse_sddmm_preprocess as "cusparseSDDMM_preprocess": PFN_cusparseSDDMM_preprocess;
    cusparse_sddmm as "cusparseSDDMM": PFN_cusparseSDDMM;
    // CSR ↔ CSC
    cusparse_csr2csc_ex2_buffer_size as "cusparseCsr2cscEx2_bufferSize": PFN_cusparseCsr2cscEx2_bufferSize;
    cusparse_csr2csc_ex2 as "cusparseCsr2cscEx2": PFN_cusparseCsr2cscEx2;
    // Sparse ↔ Dense
    cusparse_sparse_to_dense_buffer_size as "cusparseSparseToDense_bufferSize": PFN_cusparseSparseToDense_bufferSize;
    cusparse_sparse_to_dense as "cusparseSparseToDense": PFN_cusparseSparseToDense;
    cusparse_dense_to_sparse_buffer_size as "cusparseDenseToSparse_bufferSize": PFN_cusparseDenseToSparse_bufferSize;
    cusparse_dense_to_sparse_analysis as "cusparseDenseToSparse_analysis": PFN_cusparseDenseToSparse_analysis;
    cusparse_dense_to_sparse_convert as "cusparseDenseToSparse_convert": PFN_cusparseDenseToSparse_convert;
    // Sparse BLAS L1
    cusparse_axpby as "cusparseAxpby": PFN_cusparseAxpby;
    cusparse_gather as "cusparseGather": PFN_cusparseGather;
    cusparse_scatter as "cusparseScatter": PFN_cusparseScatter;
    cusparse_rot as "cusparseRot": PFN_cusparseRot;
}

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
