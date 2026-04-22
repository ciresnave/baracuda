//! Safe cuDF bindings for the RAPIDS `libcudf` C ABI.
//!
//! This crate wraps the subset of cuDF that NVIDIA exposes through the
//! emerging C ABI — enough to load a CSV / Parquet file into a device-side
//! [`Table`], inspect or iterate its columns, write a `Table` back to
//! Parquet, and hand-roll empty [`Column`]s of any supported [`TypeId`].
//!
//! Functions RAPIDS has not yet exposed through the C ABI (joins, groupbys,
//! aggregations, complex per-column operations) remain C++-only at the
//! cuDF level. Baracuda's dynamic loader maps any such missing symbol to
//! `Error::FeatureNotSupported`, so code written against this crate will
//! continue to compile as RAPIDS grows the surface and you just need to
//! run against a newer libcudf.
//!
//! # Example
//!
//! ```no_run
//! use baracuda_cudf::{Table, TypeId};
//!
//! # fn demo() -> baracuda_cudf::Result<()> {
//! let t = Table::from_csv("data/input.csv")?;
//! println!("{} cols × {} rows", t.num_columns()?, t.num_rows()?);
//! for i in 0..t.num_columns()? {
//!     let col = t.column(i)?;
//!     println!("col {i}: {:?}", col.type_id()?);
//! }
//! t.write_parquet("data/output.parquet")?;
//! # Ok(()) }
//! ```

#![warn(missing_debug_implementations, rust_2018_idioms)]

use std::ffi::CString;
use std::path::Path;

use baracuda_cudf_sys as sys;
use baracuda_types::CudaStatus;

pub use sys::cudfTypeId_t as TypeId;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("cuDF loader: {0}")]
    Loader(#[from] baracuda_core::LoaderError),
    #[error("cuDF status {0:?}: {1}")]
    Status(sys::cudfStatus_t, &'static str),
    #[error("invalid path: {0}")]
    Path(String),
    #[error("invalid C string: {0}")]
    Nul(#[from] std::ffi::NulError),
}

pub type Result<T> = std::result::Result<T, Error>;

fn check(status: sys::cudfStatus_t) -> Result<()> {
    if status.is_success() {
        Ok(())
    } else {
        Err(Error::Status(status, status.description()))
    }
}

pub fn version() -> Result<i32> {
    let c = sys::cudf()?;
    let mut v = 0;
    unsafe { check((c.cudf_get_version()?)(&mut v))? };
    Ok(v)
}

#[derive(Debug)]
pub struct Column {
    raw: sys::cudfColumn_t,
    owned: bool,
}

impl Column {
    pub fn new_empty(type_id: TypeId, size: usize) -> Result<Self> {
        let c = sys::cudf()?;
        let mut raw: sys::cudfColumn_t = core::ptr::null_mut();
        unsafe { check((c.cudf_column_create_empty()?)(&mut raw, type_id, size))? };
        Ok(Self { raw, owned: true })
    }

    pub fn len(&self) -> Result<usize> {
        let c = sys::cudf()?;
        let mut n = 0usize;
        unsafe { check((c.cudf_column_get_size()?)(self.raw, &mut n))? };
        Ok(n)
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    pub fn type_id(&self) -> Result<TypeId> {
        let c = sys::cudf()?;
        let mut t = TypeId::Empty;
        unsafe { check((c.cudf_column_get_type()?)(self.raw, &mut t))? };
        Ok(t)
    }

    pub fn as_raw(&self) -> sys::cudfColumn_t {
        self.raw
    }
}

impl Drop for Column {
    fn drop(&mut self) {
        if !self.owned {
            return;
        }
        if let Ok(c) = sys::cudf() {
            if let Ok(f) = c.cudf_column_destroy() {
                unsafe {
                    let _ = f(self.raw);
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct Table {
    raw: sys::cudfTable_t,
}

impl Table {
    pub fn from_csv(path: impl AsRef<Path>) -> Result<Self> {
        let c = sys::cudf()?;
        let p = path
            .as_ref()
            .to_str()
            .ok_or_else(|| Error::Path(path.as_ref().display().to_string()))?;
        let cstr = CString::new(p)?;
        let mut raw: sys::cudfTable_t = core::ptr::null_mut();
        unsafe { check((c.cudf_read_csv()?)(cstr.as_ptr(), &mut raw))? };
        Ok(Self { raw })
    }

    pub fn from_parquet(path: impl AsRef<Path>) -> Result<Self> {
        let c = sys::cudf()?;
        let p = path
            .as_ref()
            .to_str()
            .ok_or_else(|| Error::Path(path.as_ref().display().to_string()))?;
        let cstr = CString::new(p)?;
        let mut raw: sys::cudfTable_t = core::ptr::null_mut();
        unsafe { check((c.cudf_read_parquet()?)(cstr.as_ptr(), &mut raw))? };
        Ok(Self { raw })
    }

    pub fn write_parquet(&self, path: impl AsRef<Path>) -> Result<()> {
        let c = sys::cudf()?;
        let p = path
            .as_ref()
            .to_str()
            .ok_or_else(|| Error::Path(path.as_ref().display().to_string()))?;
        let cstr = CString::new(p)?;
        unsafe { check((c.cudf_write_parquet()?)(cstr.as_ptr(), self.raw))? };
        Ok(())
    }

    pub fn num_columns(&self) -> Result<usize> {
        let c = sys::cudf()?;
        let mut n = 0usize;
        unsafe { check((c.cudf_table_get_num_columns()?)(self.raw, &mut n))? };
        Ok(n)
    }

    pub fn num_rows(&self) -> Result<usize> {
        let c = sys::cudf()?;
        let mut n = 0usize;
        unsafe { check((c.cudf_table_get_num_rows()?)(self.raw, &mut n))? };
        Ok(n)
    }

    /// Borrows a column from the table without taking ownership.
    pub fn column(&self, index: usize) -> Result<Column> {
        let c = sys::cudf()?;
        let mut col: sys::cudfColumn_t = core::ptr::null_mut();
        unsafe { check((c.cudf_table_get_column()?)(self.raw, index, &mut col))? };
        Ok(Column {
            raw: col,
            owned: false,
        })
    }

    pub fn as_raw(&self) -> sys::cudfTable_t {
        self.raw
    }

    /// Borrow every column in order. Columns are non-owning views into this
    /// table; drop them before dropping the [`Table`].
    pub fn columns(&self) -> Result<Vec<Column>> {
        let n = self.num_columns()?;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(self.column(i)?);
        }
        Ok(out)
    }
}

impl Drop for Table {
    fn drop(&mut self) {
        if let Ok(c) = sys::cudf() {
            if let Ok(f) = c.cudf_table_destroy() {
                unsafe {
                    let _ = f(self.raw);
                }
            }
        }
    }
}
