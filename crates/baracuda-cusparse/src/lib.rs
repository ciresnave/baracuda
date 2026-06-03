//! Safe Rust wrappers for NVIDIA cuSPARSE.
//!
//! Covers the modern generic-API surface: `Handle`, `SpMat` (CSR/CSC/COO/BSR),
//! `DnMat`, `DnVec`, and the family of op algorithms — SpMV, SpMM, SpGEMM,
//! SpSV, SpSM, SDDMM — plus CSR↔CSC and sparse↔dense conversions, and the
//! sparse BLAS-1 helpers (`axpby`, `gather`, `scatter`, `rot`).
//!
//! All matrix/vector descriptors borrow the underlying [`DeviceBuffer`]s and
//! tie their lifetime to them so the buffers can't be freed while cuSPARSE
//! is still holding references.
//!
//! # Examples
//!
//! **SpMV (sparse-times-dense vector):** `y = A · x` for a 4×4 CSR matrix
//! with 6 non-zeros.
//!
//! ```no_run
//! use baracuda_cusparse::{spmv, spmv_buffer_size, DnVec, Handle, Op, SpMVAlg, SpMat};
//! use baracuda_driver::{Context, Device, DeviceBuffer};
//!
//! # fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! baracuda_driver::init()?;
//! let device = Device::get(0)?;
//! let ctx = Context::new(&device)?;
//! let handle = Handle::new()?;
//!
//! // A = [[1, 0, 0, 2], [0, 3, 0, 0], [0, 0, 0, 4], [5, 0, 6, 0]], nnz = 6
//! let mut row_offsets = DeviceBuffer::from_slice(&ctx, &[0i32, 2, 3, 4, 6])?;
//! let mut col_indices = DeviceBuffer::from_slice(&ctx, &[0i32, 3, 1, 3, 0, 2])?;
//! let mut values     = DeviceBuffer::from_slice(&ctx, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])?;
//! let mut x          = DeviceBuffer::from_slice(&ctx, &[1.0f32, 2.0, 3.0, 4.0])?;
//! let mut y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 4)?;
//!
//! let a = SpMat::csr(4, 4, 6, &mut row_offsets, &mut col_indices, &mut values)?;
//! let x_d = DnVec::new(&mut x)?;
//! let mut y_d = DnVec::new(&mut y)?;
//!
//! let bytes = spmv_buffer_size::<f32>(
//!     &handle, Op::N, &1.0, &a, &x_d, &0.0, &y_d, SpMVAlg::Default)?;
//! let mut workspace: DeviceBuffer<u8> = DeviceBuffer::new(&ctx, bytes.max(1))?;
//! spmv::<f32>(&handle, Op::N, &1.0, &a, &x_d, &0.0, &mut y_d, SpMVAlg::Default, &mut workspace)?;
//! # Ok(()) }
//! ```
//!
//! **COO descriptor build** — feed an unsorted `(row, col, value)` triple
//! stream into cuSPARSE without pre-converting to CSR:
//!
//! ```no_run
//! use baracuda_cusparse::SpMat;
//! use baracuda_driver::{Context, Device, DeviceBuffer};
//!
//! # fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! baracuda_driver::init()?;
//! let device = Device::get(0)?;
//! let ctx = Context::new(&device)?;
//!
//! // Same 4×4 matrix as above, expressed as (row, col, value) triples.
//! let mut row_idx = DeviceBuffer::from_slice(&ctx, &[0i32, 0, 1, 2, 3, 3])?;
//! let mut col_idx = DeviceBuffer::from_slice(&ctx, &[0i32, 3, 1, 3, 0, 2])?;
//! let mut values  = DeviceBuffer::from_slice(&ctx, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])?;
//!
//! let _coo = SpMat::coo(4, 4, 6, &mut row_idx, &mut col_idx, &mut values)?;
//! // Pass `_coo` to `spmv`, `spmm`, or `sparse_to_dense` just like a CSR descriptor.
//! # Ok(()) }
//! ```
//!
//! **SpMM (sparse-times-dense matrix):** `C = A · B` where `A` is sparse,
//! `B` and `C` are dense column-major.
//!
//! ```no_run
//! use baracuda_cusparse::{spmm, spmm_buffer_size, DnMat, Handle, Op, Order, SpMMAlg, SpMat};
//! use baracuda_driver::{Context, Device, DeviceBuffer};
//!
//! # fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! baracuda_driver::init()?;
//! let device = Device::get(0)?;
//! let ctx = Context::new(&device)?;
//! let handle = Handle::new()?;
//!
//! // 4×4 CSR A (same as the SpMV example) times a 4×2 dense B → 4×2 dense C.
//! let mut row_offsets = DeviceBuffer::from_slice(&ctx, &[0i32, 2, 3, 4, 6])?;
//! let mut col_indices = DeviceBuffer::from_slice(&ctx, &[0i32, 3, 1, 3, 0, 2])?;
//! let mut values     = DeviceBuffer::from_slice(&ctx, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])?;
//! let mut b_data: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 4 * 2)?;
//! let mut c_data: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 4 * 2)?;
//!
//! let a = SpMat::csr(4, 4, 6, &mut row_offsets, &mut col_indices, &mut values)?;
//! let b = DnMat::new(4, 2, /* ld */ 4, Order::Col, &mut b_data)?;
//! let mut c = DnMat::new(4, 2, /* ld */ 4, Order::Col, &mut c_data)?;
//!
//! let bytes = spmm_buffer_size::<f32>(
//!     &handle, Op::N, Op::N, &1.0, &a, &b, &0.0, &c, SpMMAlg::Default)?;
//! let mut workspace: DeviceBuffer<u8> = DeviceBuffer::new(&ctx, bytes.max(1))?;
//! spmm::<f32>(&handle, Op::N, Op::N, &1.0, &a, &b, &0.0, &mut c, SpMMAlg::Default,
//!     &mut workspace)?;
//! # Ok(()) }
//! ```

#![warn(missing_debug_implementations)]

use core::ffi::c_void;
use std::marker::PhantomData;

use baracuda_cusparse_sys::{
    cudaDataType, cusparse, cusparseDiagType_t, cusparseDnMatDescr_t, cusparseDnVecDescr_t,
    cusparseFillMode_t, cusparseHandle_t, cusparseIndexBase_t, cusparseIndexType_t,
    cusparseOperation_t, cusparseOrder_t, cusparseSpGEMMDescr_t, cusparseSpMatAttribute_t,
    cusparseSpMatDescr_t, cusparseSpSMDescr_t, cusparseSpSVDescr_t, cusparseStatus_t,
};
use baracuda_driver::{DeviceBuffer, Stream};
use baracuda_types::{Complex32, Complex64};

pub use baracuda_cusparse_sys::{
    cusparseCsr2CscAlg_t as Csr2CscAlg, cusparseIndexBase_t as IndexBase,
    cusparseSDDMMAlg_t as SDDMMAlg, cusparseSpGEMMAlg_t as SpGEMMAlg,
    cusparseSpMMAlg_t as SpMMAlg, cusparseSpMVAlg_t as SpMVAlg, cusparseSpSMAlg_t as SpSMAlg,
    cusparseSpSVAlg_t as SpSVAlg,
};

/// Error type for cuSPARSE operations.
pub type Error = baracuda_core::Error<cusparseStatus_t>;
/// Result alias.
pub type Result<T, E = Error> = core::result::Result<T, E>;

#[inline]
fn check(status: cusparseStatus_t) -> Result<()> {
    Error::check(status)
}

// ---- scalar <-> cudaDataType --------------------------------------------

/// Types that cuSPARSE's generic API accepts as element / compute type.
pub trait SparseScalar: sealed::Sealed + Copy + 'static {
    /// cuSPARSE / cuBLAS element-type tag.
    fn data_type() -> cudaDataType;
}

impl SparseScalar for f32 {
    fn data_type() -> cudaDataType {
        cudaDataType::R_32F
    }
}
impl SparseScalar for f64 {
    fn data_type() -> cudaDataType {
        cudaDataType::R_64F
    }
}
impl SparseScalar for Complex32 {
    fn data_type() -> cudaDataType {
        cudaDataType::C_32F
    }
}
impl SparseScalar for Complex64 {
    fn data_type() -> cudaDataType {
        cudaDataType::C_64F
    }
}

mod sealed {
    use baracuda_types::{Complex32, Complex64};
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for Complex32 {}
    impl Sealed for Complex64 {}
}

// ---- Op / Order / Fill / Diag wrappers ----------------------------------

/// Transpose / conjugate-transpose selector.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum Op {
    /// No transpose.
    #[default]
    N,
    /// Transpose.
    T,
    /// Conjugate transpose (real types: same as `T`).
    C,
}

impl Op {
    fn raw(self) -> cusparseOperation_t {
        match self {
            Op::N => cusparseOperation_t::N,
            Op::T => cusparseOperation_t::T,
            Op::C => cusparseOperation_t::C,
        }
    }
}

/// Dense-matrix storage order.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Order {
    /// Row-major (C-style) storage.
    Row,
    /// Column-major (Fortran-style) storage.
    Col,
}

impl Order {
    fn raw(self) -> cusparseOrder_t {
        match self {
            Order::Row => cusparseOrder_t::Row,
            Order::Col => cusparseOrder_t::Col,
        }
    }
}

/// Triangular storage mode for sparse matrices.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Fill {
    /// Lower triangle is the populated half.
    Lower,
    /// Upper triangle is the populated half.
    Upper,
}

impl Fill {
    fn raw(self) -> cusparseFillMode_t {
        match self {
            Fill::Lower => cusparseFillMode_t::Lower,
            Fill::Upper => cusparseFillMode_t::Upper,
        }
    }
}

/// Unit-diagonal flag for triangular sparse matrices.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Diag {
    /// Diagonal entries are read from the matrix.
    NonUnit,
    /// Diagonal entries are implicitly 1 (not read).
    Unit,
}

impl Diag {
    fn raw(self) -> cusparseDiagType_t {
        match self {
            Diag::NonUnit => cusparseDiagType_t::NonUnit,
            Diag::Unit => cusparseDiagType_t::Unit,
        }
    }
}

// ---- Handle -------------------------------------------------------------

/// Owned cuSPARSE handle.
pub struct Handle {
    handle: cusparseHandle_t,
}

unsafe impl Send for Handle {}

impl core::fmt::Debug for Handle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("cusparse::Handle")
            .field("handle", &self.handle)
            .finish()
    }
}

impl Handle {
    /// Create a new cuSPARSE handle (`cusparseCreate`).
    pub fn new() -> Result<Self> {
        let c = cusparse()?;
        let cu = c.cusparse_create()?;
        let mut h: cusparseHandle_t = core::ptr::null_mut();
        check(unsafe { cu(&mut h) })?;
        Ok(Self { handle: h })
    }

    /// Bind this handle to `stream`; subsequent cuSPARSE calls dispatch on
    /// it (`cusparseSetStream`).
    pub fn set_stream(&self, stream: &Stream) -> Result<()> {
        let c = cusparse()?;
        let cu = c.cusparse_set_stream()?;
        check(unsafe { cu(self.handle, stream.as_raw() as _) })
    }

    /// cuSPARSE library version (`cusparseGetVersion`),
    /// e.g. `12604` for cuSPARSE 12.6.4.
    pub fn version(&self) -> Result<i32> {
        let c = cusparse()?;
        let cu = c.cusparse_get_version()?;
        let mut v: core::ffi::c_int = 0;
        check(unsafe { cu(self.handle, &mut v) })?;
        Ok(v)
    }

    /// Raw `cusparseHandle_t`. Use with care.
    #[inline]
    pub fn as_raw(&self) -> cusparseHandle_t {
        self.handle
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if let Ok(c) = cusparse() {
            if let Ok(cu) = c.cusparse_destroy() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

// ---- Sparse matrix descriptor -------------------------------------------

/// A sparse-matrix descriptor (CSR / CSC / COO / BSR). The descriptor keeps
/// pointers to externally-owned device buffers; the lifetime parameter ties
/// those buffers to the descriptor.
pub struct SpMat<'buf, T> {
    descr: cusparseSpMatDescr_t,
    _markers: PhantomData<&'buf mut T>,
}

unsafe impl<T> Send for SpMat<'_, T> {}

impl<T> core::fmt::Debug for SpMat<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SpMat")
            .field("descr", &self.descr)
            .finish_non_exhaustive()
    }
}

impl<'buf, T: SparseScalar + baracuda_types::DeviceRepr> SpMat<'buf, T> {
    /// Build a CSR (compressed sparse row) descriptor.
    ///
    /// `row_offsets.len()` must equal `rows + 1`; `col_indices.len()` and
    /// `values.len()` must equal `nnz`.
    pub fn csr(
        rows: i64,
        cols: i64,
        nnz: i64,
        row_offsets: &'buf mut DeviceBuffer<i32>,
        col_indices: &'buf mut DeviceBuffer<i32>,
        values: &'buf mut DeviceBuffer<T>,
    ) -> Result<Self> {
        let c = cusparse()?;
        let cu = c.cusparse_create_csr()?;
        let mut descr: cusparseSpMatDescr_t = core::ptr::null_mut();
        check(unsafe {
            cu(
                &mut descr,
                rows,
                cols,
                nnz,
                row_offsets.as_raw().0 as *mut c_void,
                col_indices.as_raw().0 as *mut c_void,
                values.as_raw().0 as *mut c_void,
                cusparseIndexType_t::I32I,
                cusparseIndexType_t::I32I,
                cusparseIndexBase_t::Zero,
                T::data_type(),
            )
        })?;
        Ok(Self {
            descr,
            _markers: PhantomData,
        })
    }

    /// Build a CSC (compressed sparse column) descriptor.
    pub fn csc(
        rows: i64,
        cols: i64,
        nnz: i64,
        col_offsets: &'buf mut DeviceBuffer<i32>,
        row_indices: &'buf mut DeviceBuffer<i32>,
        values: &'buf mut DeviceBuffer<T>,
    ) -> Result<Self> {
        let c = cusparse()?;
        let cu = c.cusparse_create_csc()?;
        let mut descr: cusparseSpMatDescr_t = core::ptr::null_mut();
        check(unsafe {
            cu(
                &mut descr,
                rows,
                cols,
                nnz,
                col_offsets.as_raw().0 as *mut c_void,
                row_indices.as_raw().0 as *mut c_void,
                values.as_raw().0 as *mut c_void,
                cusparseIndexType_t::I32I,
                cusparseIndexType_t::I32I,
                cusparseIndexBase_t::Zero,
                T::data_type(),
            )
        })?;
        Ok(Self {
            descr,
            _markers: PhantomData,
        })
    }

    /// Build a BSR (block-sparse-row) descriptor.
    #[allow(clippy::too_many_arguments)]
    pub fn bsr(
        brows: i64,
        bcols: i64,
        bnnz: i64,
        row_block_dim: i64,
        col_block_dim: i64,
        order: Order,
        row_offsets: &'buf mut DeviceBuffer<i32>,
        col_indices: &'buf mut DeviceBuffer<i32>,
        values: &'buf mut DeviceBuffer<T>,
    ) -> Result<Self> {
        let c = cusparse()?;
        let cu = c.cusparse_create_bsr()?;
        let mut descr: cusparseSpMatDescr_t = core::ptr::null_mut();
        check(unsafe {
            cu(
                &mut descr,
                brows,
                bcols,
                bnnz,
                row_block_dim,
                col_block_dim,
                row_offsets.as_raw().0 as *mut c_void,
                col_indices.as_raw().0 as *mut c_void,
                values.as_raw().0 as *mut c_void,
                cusparseIndexType_t::I32I,
                cusparseIndexType_t::I32I,
                cusparseIndexBase_t::Zero,
                T::data_type(),
                order.raw(),
            )
        })?;
        Ok(Self {
            descr,
            _markers: PhantomData,
        })
    }

    /// Build a COO (coordinate) descriptor.
    pub fn coo(
        rows: i64,
        cols: i64,
        nnz: i64,
        row_indices: &'buf mut DeviceBuffer<i32>,
        col_indices: &'buf mut DeviceBuffer<i32>,
        values: &'buf mut DeviceBuffer<T>,
    ) -> Result<Self> {
        let c = cusparse()?;
        let cu = c.cusparse_create_coo()?;
        let mut descr: cusparseSpMatDescr_t = core::ptr::null_mut();
        check(unsafe {
            cu(
                &mut descr,
                rows,
                cols,
                nnz,
                row_indices.as_raw().0 as *mut c_void,
                col_indices.as_raw().0 as *mut c_void,
                values.as_raw().0 as *mut c_void,
                cusparseIndexType_t::I32I,
                cusparseIndexBase_t::Zero,
                T::data_type(),
            )
        })?;
        Ok(Self {
            descr,
            _markers: PhantomData,
        })
    }
}

impl<T> SpMat<'_, T> {
    /// Sparse matrix dimensions: `(rows, cols, nnz)`.
    pub fn shape(&self) -> Result<(i64, i64, i64)> {
        let c = cusparse()?;
        let cu = c.cusparse_sp_mat_get_size()?;
        let (mut r, mut col, mut nz) = (0i64, 0i64, 0i64);
        check(unsafe { cu(self.descr, &mut r, &mut col, &mut nz) })?;
        Ok((r, col, nz))
    }

    /// Rebind a CSR descriptor's underlying device pointers without
    /// rebuilding it. Saves descriptor recreation when the same shape
    /// is reused with new data.
    ///
    /// # Safety
    ///
    /// All three pointers must be live device allocations matching the
    /// original `(rows + 1, nnz, nnz)` element counts and the original
    /// element types. They must stay valid until the next operation
    /// on this descriptor completes.
    pub unsafe fn set_csr_pointers(
        &self,
        row_offsets: *mut c_void,
        col_indices: *mut c_void,
        values: *mut c_void,
    ) -> Result<()> { unsafe {
        let c = cusparse()?;
        let cu = c.cusparse_csr_set_pointers()?;
        check(cu(self.descr, row_offsets, col_indices, values))
    }}

    /// Rebind a CSC descriptor's underlying device pointers.
    ///
    /// # Safety
    ///
    /// See [`Self::set_csr_pointers`].
    pub unsafe fn set_csc_pointers(
        &self,
        col_offsets: *mut c_void,
        row_indices: *mut c_void,
        values: *mut c_void,
    ) -> Result<()> { unsafe {
        let c = cusparse()?;
        let cu = c.cusparse_csc_set_pointers()?;
        check(cu(self.descr, col_offsets, row_indices, values))
    }}

    /// Rebind a COO descriptor's underlying device pointers.
    ///
    /// # Safety
    ///
    /// See [`Self::set_csr_pointers`].
    pub unsafe fn set_coo_pointers(
        &self,
        row_indices: *mut c_void,
        col_indices: *mut c_void,
        values: *mut c_void,
    ) -> Result<()> { unsafe {
        let c = cusparse()?;
        let cu = c.cusparse_coo_set_pointers()?;
        check(cu(self.descr, row_indices, col_indices, values))
    }}

    /// Set the fill-triangle attribute (for triangular solves).
    pub fn set_fill(&self, fill: Fill) -> Result<()> {
        let c = cusparse()?;
        let cu = c.cusparse_sp_mat_set_attribute()?;
        let raw = fill.raw();
        check(unsafe {
            cu(
                self.descr,
                cusparseSpMatAttribute_t::FillMode,
                &raw as *const _ as *const c_void,
                core::mem::size_of::<cusparseFillMode_t>(),
            )
        })
    }

    /// Set the diagonal-type attribute (unit vs non-unit, for triangular solves).
    pub fn set_diag(&self, diag: Diag) -> Result<()> {
        let c = cusparse()?;
        let cu = c.cusparse_sp_mat_set_attribute()?;
        let raw = diag.raw();
        check(unsafe {
            cu(
                self.descr,
                cusparseSpMatAttribute_t::DiagType,
                &raw as *const _ as *const c_void,
                core::mem::size_of::<cusparseDiagType_t>(),
            )
        })
    }

    /// Raw `cusparseSpMatDescr_t`. Use with care.
    #[inline]
    pub fn as_raw(&self) -> cusparseSpMatDescr_t {
        self.descr
    }
}

impl<T> Drop for SpMat<'_, T> {
    fn drop(&mut self) {
        if let Ok(c) = cusparse() {
            if let Ok(cu) = c.cusparse_destroy_sp_mat() {
                let _ = unsafe { cu(self.descr) };
            }
        }
    }
}

// ---- Dense vector / matrix -----------------------------------------------

/// Dense vector descriptor for cuSPARSE's generic API.
///
/// Borrows the underlying [`DeviceBuffer`]; the buffer cannot be freed while
/// the descriptor is alive.
pub struct DnVec<'buf, T> {
    descr: cusparseDnVecDescr_t,
    _marker: PhantomData<&'buf mut T>,
}

unsafe impl<T> Send for DnVec<'_, T> {}

impl<T> core::fmt::Debug for DnVec<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DnVec")
            .field("descr", &self.descr)
            .finish_non_exhaustive()
    }
}

impl<'buf, T: SparseScalar + baracuda_types::DeviceRepr> DnVec<'buf, T> {
    /// Build a dense-vector descriptor (`cusparseCreateDnVec`)
    /// over the buffer `values`.
    pub fn new(values: &'buf mut DeviceBuffer<T>) -> Result<Self> {
        let c = cusparse()?;
        let cu = c.cusparse_create_dn_vec()?;
        let mut descr: cusparseDnVecDescr_t = core::ptr::null_mut();
        check(unsafe {
            cu(
                &mut descr,
                values.len() as i64,
                values.as_raw().0 as *mut c_void,
                T::data_type(),
            )
        })?;
        Ok(Self {
            descr,
            _marker: PhantomData,
        })
    }
}

impl<T> DnVec<'_, T> {
    /// Raw `cusparseDnVecDescr_t`. Use with care.
    #[inline]
    pub fn as_raw(&self) -> cusparseDnVecDescr_t {
        self.descr
    }
}

impl<T> Drop for DnVec<'_, T> {
    fn drop(&mut self) {
        if let Ok(c) = cusparse() {
            if let Ok(cu) = c.cusparse_destroy_dn_vec() {
                let _ = unsafe { cu(self.descr) };
            }
        }
    }
}

/// Dense matrix descriptor for cuSPARSE's generic API.
///
/// Borrows the underlying [`DeviceBuffer`]; the buffer cannot be freed while
/// the descriptor is alive.
pub struct DnMat<'buf, T> {
    descr: cusparseDnMatDescr_t,
    _marker: PhantomData<&'buf mut T>,
}

unsafe impl<T> Send for DnMat<'_, T> {}

impl<T> core::fmt::Debug for DnMat<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DnMat")
            .field("descr", &self.descr)
            .finish_non_exhaustive()
    }
}

impl<'buf, T: SparseScalar + baracuda_types::DeviceRepr> DnMat<'buf, T> {
    /// Build a dense-matrix descriptor (`cusparseCreateDnMat`) for the
    /// given shape, leading dimension, and storage order.
    pub fn new(
        rows: i64,
        cols: i64,
        ld: i64,
        order: Order,
        values: &'buf mut DeviceBuffer<T>,
    ) -> Result<Self> {
        let c = cusparse()?;
        let cu = c.cusparse_create_dn_mat()?;
        let mut descr: cusparseDnMatDescr_t = core::ptr::null_mut();
        check(unsafe {
            cu(
                &mut descr,
                rows,
                cols,
                ld,
                values.as_raw().0 as *mut c_void,
                T::data_type(),
                order.raw(),
            )
        })?;
        Ok(Self {
            descr,
            _marker: PhantomData,
        })
    }
}

impl<T> DnMat<'_, T> {
    /// Raw `cusparseDnMatDescr_t`. Use with care.
    #[inline]
    pub fn as_raw(&self) -> cusparseDnMatDescr_t {
        self.descr
    }
}

impl<T> Drop for DnMat<'_, T> {
    fn drop(&mut self) {
        if let Ok(c) = cusparse() {
            if let Ok(cu) = c.cusparse_destroy_dn_mat() {
                let _ = unsafe { cu(self.descr) };
            }
        }
    }
}

// ---- SpMV ---------------------------------------------------------------

/// Query buffer-size for `y = alpha * op(A) * x + beta * y`.
#[allow(clippy::too_many_arguments)]
pub fn spmv_buffer_size<T: SparseScalar>(
    handle: &Handle,
    op: Op,
    alpha: &T,
    a: &SpMat<'_, T>,
    x: &DnVec<'_, T>,
    beta: &T,
    y: &DnVec<'_, T>,
    alg: SpMVAlg,
) -> Result<usize> {
    let c = cusparse()?;
    let cu = c.cusparse_spmv_buffer_size()?;
    let mut size: usize = 0;
    check(unsafe {
        cu(
            handle.as_raw(),
            op.raw(),
            alpha as *const T as *const c_void,
            a.descr,
            x.descr,
            beta as *const T as *const c_void,
            y.descr,
            T::data_type(),
            alg,
            &mut size,
        )
    })?;
    Ok(size)
}

/// `y = alpha * op(A) * x + beta * y`.
#[allow(clippy::too_many_arguments)]
pub fn spmv<T: SparseScalar>(
    handle: &Handle,
    op: Op,
    alpha: &T,
    a: &SpMat<'_, T>,
    x: &DnVec<'_, T>,
    beta: &T,
    y: &mut DnVec<'_, T>,
    alg: SpMVAlg,
    workspace: &mut DeviceBuffer<u8>,
) -> Result<()> {
    let c = cusparse()?;
    let cu = c.cusparse_spmv()?;
    check(unsafe {
        cu(
            handle.as_raw(),
            op.raw(),
            alpha as *const T as *const c_void,
            a.descr,
            x.descr,
            beta as *const T as *const c_void,
            y.descr,
            T::data_type(),
            alg,
            workspace.as_raw().0 as *mut c_void,
        )
    })
}

// ---- SpMM ---------------------------------------------------------------

/// Query buffer-size for `C = alpha * op(A) * op(B) + beta * C`, `A` sparse.
#[allow(clippy::too_many_arguments)]
pub fn spmm_buffer_size<T: SparseScalar>(
    handle: &Handle,
    op_a: Op,
    op_b: Op,
    alpha: &T,
    a: &SpMat<'_, T>,
    b: &DnMat<'_, T>,
    beta: &T,
    c: &DnMat<'_, T>,
    alg: SpMMAlg,
) -> Result<usize> {
    let c_api = cusparse()?;
    let cu = c_api.cusparse_spmm_buffer_size()?;
    let mut size = 0usize;
    check(unsafe {
        cu(
            handle.as_raw(),
            op_a.raw(),
            op_b.raw(),
            alpha as *const T as *const c_void,
            a.descr,
            b.descr,
            beta as *const T as *const c_void,
            c.descr,
            T::data_type(),
            alg,
            &mut size,
        )
    })?;
    Ok(size)
}

/// `C = alpha * op(A) * op(B) + beta * C` with `A` sparse, `B`/`C` dense
/// (`cusparseSpMM`).
#[allow(clippy::too_many_arguments)]
pub fn spmm<T: SparseScalar>(
    handle: &Handle,
    op_a: Op,
    op_b: Op,
    alpha: &T,
    a: &SpMat<'_, T>,
    b: &DnMat<'_, T>,
    beta: &T,
    c: &mut DnMat<'_, T>,
    alg: SpMMAlg,
    workspace: &mut DeviceBuffer<u8>,
) -> Result<()> {
    let c_api = cusparse()?;
    let cu = c_api.cusparse_spmm()?;
    check(unsafe {
        cu(
            handle.as_raw(),
            op_a.raw(),
            op_b.raw(),
            alpha as *const T as *const c_void,
            a.descr,
            b.descr,
            beta as *const T as *const c_void,
            c.descr,
            T::data_type(),
            alg,
            workspace.as_raw().0 as *mut c_void,
        )
    })
}

/// One-time preprocessing before [`spmm`]. Pre-computes algorithm-specific
/// state into `workspace` so subsequent [`spmm`] calls (with the same A
/// sparsity pattern + dimensions) are faster. Use this when the same
/// matrix is multiplied many times.
#[allow(clippy::too_many_arguments)]
pub fn spmm_preprocess<T: SparseScalar>(
    handle: &Handle,
    op_a: Op,
    op_b: Op,
    alpha: &T,
    a: &SpMat<'_, T>,
    b: &DnMat<'_, T>,
    beta: &T,
    c: &mut DnMat<'_, T>,
    alg: SpMMAlg,
    workspace: &mut DeviceBuffer<u8>,
) -> Result<()> {
    let c_api = cusparse()?;
    let cu = c_api.cusparse_spmm_preprocess()?;
    check(unsafe {
        cu(
            handle.as_raw(),
            op_a.raw(),
            op_b.raw(),
            alpha as *const T as *const c_void,
            a.descr,
            b.descr,
            beta as *const T as *const c_void,
            c.descr,
            T::data_type(),
            alg,
            workspace.as_raw().0 as *mut c_void,
        )
    })
}

// ---- SpGEMM -------------------------------------------------------------

/// Per-plan handle for a 3-phase SpGEMM computation.
#[derive(Debug)]
pub struct SpGEMMPlan {
    raw: cusparseSpGEMMDescr_t,
}

impl SpGEMMPlan {
    /// Create a new SpGEMM plan descriptor
    /// (`cusparseSpGEMM_createDescr`).
    pub fn new() -> Result<Self> {
        let c = cusparse()?;
        let cu = c.cusparse_spgemm_create_descr()?;
        let mut d: cusparseSpGEMMDescr_t = core::ptr::null_mut();
        check(unsafe { cu(&mut d) })?;
        Ok(Self { raw: d })
    }
}

impl Drop for SpGEMMPlan {
    fn drop(&mut self) {
        if let Ok(c) = cusparse() {
            if let Ok(cu) = c.cusparse_spgemm_destroy_descr() {
                let _ = unsafe { cu(self.raw) };
            }
        }
    }
}

/// Phase 1: work-estimation. The caller provides `buffer1` whose size is
/// returned in `size1`; pass `null` the first time, then allocate & re-call.
///
/// # Safety
///
/// `buffer1` must be either null (size-query mode) or a valid device
/// pointer to at least `*size1` bytes of writable scratch memory that
/// remains live for the duration of the underlying cuSPARSE call.
#[allow(clippy::too_many_arguments)]
pub unsafe fn spgemm_work_estimation<T: SparseScalar>(
    handle: &Handle,
    op_a: Op,
    op_b: Op,
    alpha: &T,
    a: &SpMat<'_, T>,
    b: &SpMat<'_, T>,
    beta: &T,
    c: &mut SpMat<'_, T>,
    alg: SpGEMMAlg,
    plan: &SpGEMMPlan,
    size1: &mut usize,
    buffer1: *mut c_void,
) -> Result<()> { unsafe {
    let c_api = cusparse()?;
    let cu = c_api.cusparse_spgemm_work_estimation()?;
    check(cu(
        handle.as_raw(),
        op_a.raw(),
        op_b.raw(),
        alpha as *const T as *const c_void,
        a.descr,
        b.descr,
        beta as *const T as *const c_void,
        c.descr,
        T::data_type(),
        alg,
        plan.raw,
        size1,
        buffer1,
    ))
}}

/// Phase 2: compute. Same two-step pattern for `buffer2`.
///
/// # Safety
///
/// `buffer2` must be either null (size-query mode) or a valid device
/// pointer to at least `*size2` bytes of writable scratch memory that
/// remains live for the duration of the underlying cuSPARSE call.
#[allow(clippy::too_many_arguments)]
pub unsafe fn spgemm_compute<T: SparseScalar>(
    handle: &Handle,
    op_a: Op,
    op_b: Op,
    alpha: &T,
    a: &SpMat<'_, T>,
    b: &SpMat<'_, T>,
    beta: &T,
    c: &mut SpMat<'_, T>,
    alg: SpGEMMAlg,
    plan: &SpGEMMPlan,
    size2: &mut usize,
    buffer2: *mut c_void,
) -> Result<()> { unsafe {
    let c_api = cusparse()?;
    let cu = c_api.cusparse_spgemm_compute()?;
    check(cu(
        handle.as_raw(),
        op_a.raw(),
        op_b.raw(),
        alpha as *const T as *const c_void,
        a.descr,
        b.descr,
        beta as *const T as *const c_void,
        c.descr,
        T::data_type(),
        alg,
        plan.raw,
        size2,
        buffer2,
    ))
}}

/// Phase 3: write output arrays into the pre-populated output `SpMat`.
#[allow(clippy::too_many_arguments)]
pub fn spgemm_copy<T: SparseScalar>(
    handle: &Handle,
    op_a: Op,
    op_b: Op,
    alpha: &T,
    a: &SpMat<'_, T>,
    b: &SpMat<'_, T>,
    beta: &T,
    c: &mut SpMat<'_, T>,
    alg: SpGEMMAlg,
    plan: &SpGEMMPlan,
) -> Result<()> {
    let c_api = cusparse()?;
    let cu = c_api.cusparse_spgemm_copy()?;
    check(unsafe {
        cu(
            handle.as_raw(),
            op_a.raw(),
            op_b.raw(),
            alpha as *const T as *const c_void,
            a.descr,
            b.descr,
            beta as *const T as *const c_void,
            c.descr,
            T::data_type(),
            alg,
            plan.raw,
        )
    })
}

// ---- SpSV / SpSM --------------------------------------------------------

/// Per-plan handle for an SpSV (sparse triangular solve, single RHS).
#[derive(Debug)]
pub struct SpSVPlan {
    raw: cusparseSpSVDescr_t,
}

impl SpSVPlan {
    /// Create a new SpSV plan descriptor
    /// (`cusparseSpSV_createDescr`).
    pub fn new() -> Result<Self> {
        let c = cusparse()?;
        let cu = c.cusparse_spsv_create_descr()?;
        let mut d: cusparseSpSVDescr_t = core::ptr::null_mut();
        check(unsafe { cu(&mut d) })?;
        Ok(Self { raw: d })
    }
}

impl Drop for SpSVPlan {
    fn drop(&mut self) {
        if let Ok(c) = cusparse() {
            if let Ok(cu) = c.cusparse_spsv_destroy_descr() {
                let _ = unsafe { cu(self.raw) };
            }
        }
    }
}

/// Query buffer-size for a sparse triangular solve
/// (`cusparseSpSV_bufferSize`).
#[allow(clippy::too_many_arguments)]
pub fn spsv_buffer_size<T: SparseScalar>(
    handle: &Handle,
    op: Op,
    alpha: &T,
    a: &SpMat<'_, T>,
    x: &DnVec<'_, T>,
    y: &DnVec<'_, T>,
    alg: SpSVAlg,
    plan: &SpSVPlan,
) -> Result<usize> {
    let c = cusparse()?;
    let cu = c.cusparse_spsv_buffer_size()?;
    let mut size = 0usize;
    check(unsafe {
        cu(
            handle.as_raw(),
            op.raw(),
            alpha as *const T as *const c_void,
            a.descr,
            x.descr,
            y.descr,
            T::data_type(),
            alg,
            plan.raw,
            &mut size,
        )
    })?;
    Ok(size)
}

/// Run the SpSV analysis phase
/// (`cusparseSpSV_analysis`) — must be called once before [`spsv_solve`].
#[allow(clippy::too_many_arguments)]
pub fn spsv_analysis<T: SparseScalar>(
    handle: &Handle,
    op: Op,
    alpha: &T,
    a: &SpMat<'_, T>,
    x: &DnVec<'_, T>,
    y: &DnVec<'_, T>,
    alg: SpSVAlg,
    plan: &SpSVPlan,
    workspace: &mut DeviceBuffer<u8>,
) -> Result<()> {
    let c = cusparse()?;
    let cu = c.cusparse_spsv_analysis()?;
    check(unsafe {
        cu(
            handle.as_raw(),
            op.raw(),
            alpha as *const T as *const c_void,
            a.descr,
            x.descr,
            y.descr,
            T::data_type(),
            alg,
            plan.raw,
            workspace.as_raw().0 as *mut c_void,
        )
    })
}

/// Execute the sparse triangular solve `op(A) * y = alpha * x`
/// (`cusparseSpSV_solve`). Requires a prior [`spsv_analysis`] call.
#[allow(clippy::too_many_arguments)]
pub fn spsv_solve<T: SparseScalar>(
    handle: &Handle,
    op: Op,
    alpha: &T,
    a: &SpMat<'_, T>,
    x: &DnVec<'_, T>,
    y: &mut DnVec<'_, T>,
    alg: SpSVAlg,
    plan: &SpSVPlan,
) -> Result<()> {
    let c = cusparse()?;
    let cu = c.cusparse_spsv_solve()?;
    check(unsafe {
        cu(
            handle.as_raw(),
            op.raw(),
            alpha as *const T as *const c_void,
            a.descr,
            x.descr,
            y.descr,
            T::data_type(),
            alg,
            plan.raw,
        )
    })
}

/// Per-plan handle for an SpSM (sparse triangular solve, multiple RHS).
#[derive(Debug)]
pub struct SpSMPlan {
    raw: cusparseSpSMDescr_t,
}

impl SpSMPlan {
    /// Create a new SpSM plan descriptor
    /// (`cusparseSpSM_createDescr`).
    pub fn new() -> Result<Self> {
        let c = cusparse()?;
        let cu = c.cusparse_spsm_create_descr()?;
        let mut d: cusparseSpSMDescr_t = core::ptr::null_mut();
        check(unsafe { cu(&mut d) })?;
        Ok(Self { raw: d })
    }
}

impl Drop for SpSMPlan {
    fn drop(&mut self) {
        if let Ok(c) = cusparse() {
            if let Ok(cu) = c.cusparse_spsm_destroy_descr() {
                let _ = unsafe { cu(self.raw) };
            }
        }
    }
}

/// Query buffer-size for a multi-RHS sparse triangular solve
/// (`cusparseSpSM_bufferSize`).
#[allow(clippy::too_many_arguments)]
pub fn spsm_buffer_size<T: SparseScalar>(
    handle: &Handle,
    op_a: Op,
    op_b: Op,
    alpha: &T,
    a: &SpMat<'_, T>,
    b: &DnMat<'_, T>,
    c: &DnMat<'_, T>,
    alg: SpSMAlg,
    plan: &SpSMPlan,
) -> Result<usize> {
    let c_api = cusparse()?;
    let cu = c_api.cusparse_spsm_buffer_size()?;
    let mut size = 0usize;
    check(unsafe {
        cu(
            handle.as_raw(),
            op_a.raw(),
            op_b.raw(),
            alpha as *const T as *const c_void,
            a.descr,
            b.descr,
            c.descr,
            T::data_type(),
            alg,
            plan.raw,
            &mut size,
        )
    })?;
    Ok(size)
}

/// Run the SpSM analysis phase
/// (`cusparseSpSM_analysis`) — must be called once before [`spsm_solve`].
#[allow(clippy::too_many_arguments)]
pub fn spsm_analysis<T: SparseScalar>(
    handle: &Handle,
    op_a: Op,
    op_b: Op,
    alpha: &T,
    a: &SpMat<'_, T>,
    b: &DnMat<'_, T>,
    c: &DnMat<'_, T>,
    alg: SpSMAlg,
    plan: &SpSMPlan,
    workspace: &mut DeviceBuffer<u8>,
) -> Result<()> {
    let c_api = cusparse()?;
    let cu = c_api.cusparse_spsm_analysis()?;
    check(unsafe {
        cu(
            handle.as_raw(),
            op_a.raw(),
            op_b.raw(),
            alpha as *const T as *const c_void,
            a.descr,
            b.descr,
            c.descr,
            T::data_type(),
            alg,
            plan.raw,
            workspace.as_raw().0 as *mut c_void,
        )
    })
}

/// Execute the multi-RHS sparse triangular solve `op(A) * C = alpha * op(B)`
/// (`cusparseSpSM_solve`).
#[allow(clippy::too_many_arguments)]
pub fn spsm_solve<T: SparseScalar>(
    handle: &Handle,
    op_a: Op,
    op_b: Op,
    alpha: &T,
    a: &SpMat<'_, T>,
    b: &DnMat<'_, T>,
    c: &mut DnMat<'_, T>,
    alg: SpSMAlg,
    plan: &SpSMPlan,
) -> Result<()> {
    let c_api = cusparse()?;
    let cu = c_api.cusparse_spsm_solve()?;
    check(unsafe {
        cu(
            handle.as_raw(),
            op_a.raw(),
            op_b.raw(),
            alpha as *const T as *const c_void,
            a.descr,
            b.descr,
            c.descr,
            T::data_type(),
            alg,
            plan.raw,
        )
    })
}

// ---- SDDMM -------------------------------------------------------------

/// Query buffer-size for the sampled dense-dense matmul
/// (`cusparseSDDMM_bufferSize`).
#[allow(clippy::too_many_arguments)]
pub fn sddmm_buffer_size<T: SparseScalar>(
    handle: &Handle,
    op_a: Op,
    op_b: Op,
    alpha: &T,
    a: &DnMat<'_, T>,
    b: &DnMat<'_, T>,
    beta: &T,
    c: &SpMat<'_, T>,
    alg: SDDMMAlg,
) -> Result<usize> {
    let c_api = cusparse()?;
    let cu = c_api.cusparse_sddmm_buffer_size()?;
    let mut size = 0usize;
    check(unsafe {
        cu(
            handle.as_raw(),
            op_a.raw(),
            op_b.raw(),
            alpha as *const T as *const c_void,
            a.descr,
            b.descr,
            beta as *const T as *const c_void,
            c.descr,
            T::data_type(),
            alg,
            &mut size,
        )
    })?;
    Ok(size)
}

/// Sampled dense-dense matrix multiply (`cusparseSDDMM`): computes
/// `C = alpha * op(A) * op(B) + beta * C` at the sparsity pattern of `C`.
#[allow(clippy::too_many_arguments)]
pub fn sddmm<T: SparseScalar>(
    handle: &Handle,
    op_a: Op,
    op_b: Op,
    alpha: &T,
    a: &DnMat<'_, T>,
    b: &DnMat<'_, T>,
    beta: &T,
    c: &mut SpMat<'_, T>,
    alg: SDDMMAlg,
    workspace: &mut DeviceBuffer<u8>,
) -> Result<()> {
    let c_api = cusparse()?;
    let cu = c_api.cusparse_sddmm()?;
    check(unsafe {
        cu(
            handle.as_raw(),
            op_a.raw(),
            op_b.raw(),
            alpha as *const T as *const c_void,
            a.descr,
            b.descr,
            beta as *const T as *const c_void,
            c.descr,
            T::data_type(),
            alg,
            workspace.as_raw().0 as *mut c_void,
        )
    })
}

/// One-time preprocessing before [`sddmm`]. See [`spmm_preprocess`] for
/// the rationale.
#[allow(clippy::too_many_arguments)]
pub fn sddmm_preprocess<T: SparseScalar>(
    handle: &Handle,
    op_a: Op,
    op_b: Op,
    alpha: &T,
    a: &DnMat<'_, T>,
    b: &DnMat<'_, T>,
    beta: &T,
    c: &mut SpMat<'_, T>,
    alg: SDDMMAlg,
    workspace: &mut DeviceBuffer<u8>,
) -> Result<()> {
    let c_api = cusparse()?;
    let cu = c_api.cusparse_sddmm_preprocess()?;
    check(unsafe {
        cu(
            handle.as_raw(),
            op_a.raw(),
            op_b.raw(),
            alpha as *const T as *const c_void,
            a.descr,
            b.descr,
            beta as *const T as *const c_void,
            c.descr,
            T::data_type(),
            alg,
            workspace.as_raw().0 as *mut c_void,
        )
    })
}

// ---- Sparse / dense conversions ----------------------------------------

/// Query buffer-size for a sparse → dense conversion
/// (`cusparseSparseToDense_bufferSize`).
pub fn sparse_to_dense_buffer_size<T: SparseScalar>(
    handle: &Handle,
    sp: &SpMat<'_, T>,
    dn: &DnMat<'_, T>,
) -> Result<usize> {
    let c = cusparse()?;
    let cu = c.cusparse_sparse_to_dense_buffer_size()?;
    let mut size = 0usize;
    check(unsafe { cu(handle.as_raw(), sp.descr, dn.descr, 0, &mut size) })?;
    Ok(size)
}

/// Convert a sparse matrix into a dense one
/// (`cusparseSparseToDense`).
pub fn sparse_to_dense<T: SparseScalar>(
    handle: &Handle,
    sp: &SpMat<'_, T>,
    dn: &mut DnMat<'_, T>,
    workspace: &mut DeviceBuffer<u8>,
) -> Result<()> {
    let c = cusparse()?;
    let cu = c.cusparse_sparse_to_dense()?;
    check(unsafe {
        cu(
            handle.as_raw(),
            sp.descr,
            dn.descr,
            0,
            workspace.as_raw().0 as *mut c_void,
        )
    })
}

/// Query buffer-size for a dense → sparse conversion
/// (`cusparseDenseToSparse_bufferSize`).
pub fn dense_to_sparse_buffer_size<T: SparseScalar>(
    handle: &Handle,
    dn: &DnMat<'_, T>,
    sp: &SpMat<'_, T>,
) -> Result<usize> {
    let c = cusparse()?;
    let cu = c.cusparse_dense_to_sparse_buffer_size()?;
    let mut size = 0usize;
    check(unsafe { cu(handle.as_raw(), dn.descr, sp.descr, 0, &mut size) })?;
    Ok(size)
}

/// Run the dense → sparse analysis phase
/// (`cusparseDenseToSparse_analysis`); fills in the sparsity pattern.
pub fn dense_to_sparse_analysis<T: SparseScalar>(
    handle: &Handle,
    dn: &DnMat<'_, T>,
    sp: &SpMat<'_, T>,
    workspace: &mut DeviceBuffer<u8>,
) -> Result<()> {
    let c = cusparse()?;
    let cu = c.cusparse_dense_to_sparse_analysis()?;
    check(unsafe {
        cu(
            handle.as_raw(),
            dn.descr,
            sp.descr,
            0,
            workspace.as_raw().0 as *mut c_void,
        )
    })
}

/// Execute the dense → sparse conversion
/// (`cusparseDenseToSparse_convert`); fills in the sparse values.
pub fn dense_to_sparse_convert<T: SparseScalar>(
    handle: &Handle,
    dn: &DnMat<'_, T>,
    sp: &mut SpMat<'_, T>,
    workspace: &mut DeviceBuffer<u8>,
) -> Result<()> {
    let c = cusparse()?;
    let cu = c.cusparse_dense_to_sparse_convert()?;
    check(unsafe {
        cu(
            handle.as_raw(),
            dn.descr,
            sp.descr,
            0,
            workspace.as_raw().0 as *mut c_void,
        )
    })
}

/// Workspace size in bytes for [`csr2csc_ex2`].
#[allow(clippy::too_many_arguments)]
pub fn csr2csc_ex2_buffer_size<T: SparseScalar + baracuda_types::DeviceRepr>(
    handle: &Handle,
    m: i32,
    n: i32,
    nnz: i32,
    csr_val: &DeviceBuffer<T>,
    csr_row_ptr: &DeviceBuffer<i32>,
    csr_col_ind: &DeviceBuffer<i32>,
    csc_val: &mut DeviceBuffer<T>,
    csc_col_ptr: &mut DeviceBuffer<i32>,
    csc_row_ind: &mut DeviceBuffer<i32>,
    copy_values: bool,
    idx_base: IndexBase,
    alg: Csr2CscAlg,
) -> Result<usize> {
    let c = cusparse()?;
    let cu = c.cusparse_csr2csc_ex2_buffer_size()?;
    let mut size = 0usize;
    check(unsafe {
        cu(
            handle.as_raw(),
            m,
            n,
            nnz,
            csr_val.as_raw().0 as *const c_void,
            csr_row_ptr.as_raw().0 as *const i32,
            csr_col_ind.as_raw().0 as *const i32,
            csc_val.as_raw().0 as *mut c_void,
            csc_col_ptr.as_raw().0 as *mut i32,
            csc_row_ind.as_raw().0 as *mut i32,
            T::data_type(),
            copy_values as i32,
            idx_base,
            alg,
            &mut size,
        )
    })?;
    Ok(size)
}

/// Convert a CSR matrix to CSC format using the modern Ex2 entry point —
/// supports algorithm selection (`alg`) and arbitrary value types.
#[allow(clippy::too_many_arguments)]
pub fn csr2csc_ex2<T: SparseScalar + baracuda_types::DeviceRepr>(
    handle: &Handle,
    m: i32,
    n: i32,
    nnz: i32,
    csr_val: &DeviceBuffer<T>,
    csr_row_ptr: &DeviceBuffer<i32>,
    csr_col_ind: &DeviceBuffer<i32>,
    csc_val: &mut DeviceBuffer<T>,
    csc_col_ptr: &mut DeviceBuffer<i32>,
    csc_row_ind: &mut DeviceBuffer<i32>,
    copy_values: bool,
    idx_base: IndexBase,
    alg: Csr2CscAlg,
    workspace: &mut DeviceBuffer<u8>,
) -> Result<()> {
    let c = cusparse()?;
    let cu = c.cusparse_csr2csc_ex2()?;
    check(unsafe {
        cu(
            handle.as_raw(),
            m,
            n,
            nnz,
            csr_val.as_raw().0 as *const c_void,
            csr_row_ptr.as_raw().0 as *const i32,
            csr_col_ind.as_raw().0 as *const i32,
            csc_val.as_raw().0 as *mut c_void,
            csc_col_ptr.as_raw().0 as *mut i32,
            csc_row_ind.as_raw().0 as *mut i32,
            T::data_type(),
            copy_values as i32,
            idx_base,
            alg,
            workspace.as_raw().0 as *mut c_void,
        )
    })
}

// ---- Sparse BLAS-1 helpers ---------------------------------------------

/// Sparse AXPBY: `y = alpha * x + beta * y` over dense vectors
/// (`cusparseAxpby`).
pub fn axpby<T: SparseScalar>(
    handle: &Handle,
    alpha: &T,
    x: &DnVec<'_, T>,
    beta: &T,
    y: &mut DnVec<'_, T>,
) -> Result<()> {
    let c = cusparse()?;
    let cu = c.cusparse_axpby()?;
    check(unsafe {
        cu(
            handle.as_raw(),
            alpha as *const T as *const c_void,
            x.descr,
            beta as *const T as *const c_void,
            y.descr,
        )
    })
}

/// Sparse gather: `x = y[indices]` (`cusparseGather`).
pub fn gather<T: SparseScalar>(
    handle: &Handle,
    y: &DnVec<'_, T>,
    x: &mut DnVec<'_, T>,
) -> Result<()> {
    let c = cusparse()?;
    let cu = c.cusparse_gather()?;
    check(unsafe { cu(handle.as_raw(), y.descr, x.descr) })
}

/// Sparse scatter: `y[indices] = x` (`cusparseScatter`).
pub fn scatter<T: SparseScalar>(
    handle: &Handle,
    x: &DnVec<'_, T>,
    y: &mut DnVec<'_, T>,
) -> Result<()> {
    let c = cusparse()?;
    let cu = c.cusparse_scatter()?;
    check(unsafe { cu(handle.as_raw(), x.descr, y.descr) })
}

/// Givens rotation applied to a pair of sparse/dense vectors
/// (`cusparseRot`).
pub fn rot<T: SparseScalar>(
    handle: &Handle,
    c_cos: &T,
    s_sin: &T,
    x: &mut DnVec<'_, T>,
    y: &mut DnVec<'_, T>,
) -> Result<()> {
    let c_api = cusparse()?;
    let cu = c_api.cusparse_rot()?;
    check(unsafe {
        cu(
            handle.as_raw(),
            c_cos as *const T as *const c_void,
            s_sin as *const T as *const c_void,
            x.descr,
            y.descr,
        )
    })
}

// ---- Back-compat re-exports for existing users --------------------------

/// Legacy alias kept for callers from v0.1 — prefer [`SpMat::csr`].
pub type CsrMatrix<'buf> = SpMat<'buf, f32>;
/// Legacy alias — prefer [`DnVec`].
pub type DenseVector<'buf, T> = DnVec<'buf, T>;
