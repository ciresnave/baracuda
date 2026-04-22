//! Raw FFI + dynamic loader for NVIDIA libcudf (RAPIDS cuDF GPU dataframes).
//!
//! **Status: skeleton.** libcudf is a C++ library; RAPIDS has a nascent C ABI
//! (`cudf_c`) that is still evolving. This crate gives you a loader and a
//! small surface covering I/O (CSV/Parquet reader handles) and columnar
//! allocation. Most of cuDF's API remains C++; production users today embed a
//! thin C shim over the C++ API and link that shim here.
//!
//! The symbol names below reflect the RAPIDS **libcudf_c** proposal and are
//! feature-gated so missing entries at runtime map to
//! `Error::FeatureNotSupported`. Extend the `cudf_fns!` table as RAPIDS
//! promotes more APIs.

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
#![warn(missing_debug_implementations)]

use core::ffi::{c_char, c_int, c_void};
use std::sync::OnceLock;

use baracuda_core::{platform, Library, LoaderError};
use baracuda_types::CudaStatus;

// ---- opaque handles -----------------------------------------------------

pub type cudfColumn_t = *mut c_void;
pub type cudfTable_t = *mut c_void;
pub type cudfScalar_t = *mut c_void;

// ---- enums --------------------------------------------------------------

/// Matches `cudf::type_id` — the wire type of a column's elements.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudfTypeId_t {
    Empty = 0,
    Int8 = 1,
    Int16 = 2,
    Int32 = 3,
    Int64 = 4,
    Uint8 = 5,
    Uint16 = 6,
    Uint32 = 7,
    Uint64 = 8,
    Float32 = 9,
    Float64 = 10,
    Bool8 = 11,
    Timestamp_Days = 12,
    Timestamp_Seconds = 13,
    Timestamp_Milliseconds = 14,
    Timestamp_Microseconds = 15,
    Timestamp_Nanoseconds = 16,
    Duration_Days = 17,
    Duration_Seconds = 18,
    Duration_Milliseconds = 19,
    Duration_Microseconds = 20,
    Duration_Nanoseconds = 21,
    Dictionary32 = 22,
    String = 23,
    List = 24,
    Decimal32 = 25,
    Decimal64 = 26,
    Decimal128 = 27,
    Struct = 28,
}

// ---- status -------------------------------------------------------------

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct cudfStatus_t(pub i32);

impl cudfStatus_t {
    pub const SUCCESS: Self = Self(0);
    pub const LOGIC_ERROR: Self = Self(1);
    pub const CUDA_ERROR: Self = Self(2);
    pub const OUT_OF_MEMORY: Self = Self(3);
    pub const NOT_IMPLEMENTED: Self = Self(4);

    pub const fn is_success(self) -> bool {
        self.0 == 0
    }
}

impl CudaStatus for cudfStatus_t {
    fn code(self) -> i32 {
        self.0
    }
    fn name(self) -> &'static str {
        match self.0 {
            0 => "CUDF_STATUS_SUCCESS",
            1 => "CUDF_STATUS_LOGIC_ERROR",
            2 => "CUDF_STATUS_CUDA_ERROR",
            3 => "CUDF_STATUS_OUT_OF_MEMORY",
            4 => "CUDF_STATUS_NOT_IMPLEMENTED",
            _ => "CUDF_STATUS_UNRECOGNIZED",
        }
    }
    fn description(self) -> &'static str {
        match self.0 {
            0 => "success",
            1 => "logic error in cuDF call",
            2 => "CUDA failure bubbled up from cuDF",
            3 => "out of device memory",
            4 => "not implemented in this cuDF build",
            _ => "unrecognized cuDF status code",
        }
    }
    fn is_success(self) -> bool {
        cudfStatus_t::is_success(self)
    }
    fn library(self) -> &'static str {
        "cudf"
    }
}

// ---- function-pointer types ----------------------------------------------

pub type PFN_cudfGetVersion = unsafe extern "C" fn(version: *mut c_int) -> cudfStatus_t;

pub type PFN_cudfColumnCreateEmpty = unsafe extern "C" fn(
    out: *mut cudfColumn_t,
    type_id: cudfTypeId_t,
    size: usize,
) -> cudfStatus_t;

pub type PFN_cudfColumnDestroy = unsafe extern "C" fn(column: cudfColumn_t) -> cudfStatus_t;

pub type PFN_cudfColumnGetType =
    unsafe extern "C" fn(column: cudfColumn_t, type_id: *mut cudfTypeId_t) -> cudfStatus_t;

pub type PFN_cudfColumnGetSize =
    unsafe extern "C" fn(column: cudfColumn_t, size: *mut usize) -> cudfStatus_t;

pub type PFN_cudfTableCreate = unsafe extern "C" fn(
    out: *mut cudfTable_t,
    columns: *const cudfColumn_t,
    n_columns: usize,
) -> cudfStatus_t;

pub type PFN_cudfTableDestroy = unsafe extern "C" fn(table: cudfTable_t) -> cudfStatus_t;

pub type PFN_cudfTableGetNumColumns =
    unsafe extern "C" fn(table: cudfTable_t, n: *mut usize) -> cudfStatus_t;

pub type PFN_cudfTableGetNumRows =
    unsafe extern "C" fn(table: cudfTable_t, n: *mut usize) -> cudfStatus_t;

pub type PFN_cudfTableGetColumn = unsafe extern "C" fn(
    table: cudfTable_t,
    index: usize,
    column: *mut cudfColumn_t,
) -> cudfStatus_t;

pub type PFN_cudfReadCsv = unsafe extern "C" fn(
    filepath: *const c_char,
    out_table: *mut cudfTable_t,
) -> cudfStatus_t;

pub type PFN_cudfReadParquet = unsafe extern "C" fn(
    filepath: *const c_char,
    out_table: *mut cudfTable_t,
) -> cudfStatus_t;

pub type PFN_cudfWriteParquet = unsafe extern "C" fn(
    filepath: *const c_char,
    table: cudfTable_t,
) -> cudfStatus_t;

// ---- loader --------------------------------------------------------------

fn cudf_candidates() -> Vec<String> {
    // RAPIDS ships versioned libcudf.so; the C shim name varies by build.
    // Cover the common conventions.
    let mut v = platform::versioned_library_candidates("cudf", &["25", "24", "23", "22"]);
    // Plain soname fallbacks.
    #[cfg(target_os = "linux")]
    {
        v.push("libcudf.so".to_string());
        v.push("libcudf_c.so".to_string());
    }
    #[cfg(target_os = "windows")]
    {
        v.push("cudf.dll".to_string());
        v.push("cudf_c.dll".to_string());
    }
    v
}

macro_rules! cudf_fns {
    ($($name:ident as $sym:literal : $pfn:ty);* $(;)?) => {
        pub struct Cudf {
            lib: Library,
            $($name: OnceLock<$pfn>,)*
        }
        impl core::fmt::Debug for Cudf {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct("Cudf").field("lib", &self.lib).finish_non_exhaustive()
            }
        }
        impl Cudf {
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

cudf_fns! {
    cudf_get_version as "cudfGetVersion": PFN_cudfGetVersion;
    cudf_column_create_empty as "cudfColumnCreateEmpty": PFN_cudfColumnCreateEmpty;
    cudf_column_destroy as "cudfColumnDestroy": PFN_cudfColumnDestroy;
    cudf_column_get_type as "cudfColumnGetType": PFN_cudfColumnGetType;
    cudf_column_get_size as "cudfColumnGetSize": PFN_cudfColumnGetSize;
    cudf_table_create as "cudfTableCreate": PFN_cudfTableCreate;
    cudf_table_destroy as "cudfTableDestroy": PFN_cudfTableDestroy;
    cudf_table_get_num_columns as "cudfTableGetNumColumns": PFN_cudfTableGetNumColumns;
    cudf_table_get_num_rows as "cudfTableGetNumRows": PFN_cudfTableGetNumRows;
    cudf_table_get_column as "cudfTableGetColumn": PFN_cudfTableGetColumn;
    cudf_read_csv as "cudfReadCsv": PFN_cudfReadCsv;
    cudf_read_parquet as "cudfReadParquet": PFN_cudfReadParquet;
    cudf_write_parquet as "cudfWriteParquet": PFN_cudfWriteParquet;
}

pub fn cudf() -> Result<&'static Cudf, LoaderError> {
    static CUDF: OnceLock<Cudf> = OnceLock::new();
    if let Some(c) = CUDF.get() {
        return Ok(c);
    }
    let candidates: Vec<&'static str> = cudf_candidates()
        .into_iter()
        .map(|s| Box::leak(s.into_boxed_str()) as &'static str)
        .collect();
    let candidates_leaked: &'static [&'static str] = Box::leak(candidates.into_boxed_slice());
    let lib = Library::open("cudf", candidates_leaked)?;
    let c = Cudf::empty(lib);
    let _ = CUDF.set(c);
    Ok(CUDF.get().expect("OnceLock set or lost race"))
}
