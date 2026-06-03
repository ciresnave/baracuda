//! Smoke tests for `baracuda-cudf`.
//!
//! The crate is a thin safe wrapper around the (still evolving) RAPIDS
//! libcudf C ABI. Most of its public surface only does meaningful work
//! when libcudf is present at runtime — but several invariants are pure
//! static checks (enum discriminants matching `cudf::type_id`, types
//! implementing the expected derive traits, error formatting, path
//! validation, etc.). Those host-only checks ship as default and run on
//! every `cargo test -p baracuda-cudf` even without a GPU or RAPIDS install.
//!
//! GPU/library-touching checks are marked `#[ignore]` so they run only
//! under `cargo test -p baracuda-cudf -- --include-ignored` on a machine
//! that has `libcudf` on the loader path.

use baracuda_cudf::{Column, Error, Result, Table, TypeId, version};
use baracuda_cudf_sys::cudfStatus_t;

// ---------------------------------------------------------------------------
// Host-only checks: run everywhere, no libcudf needed.
// ---------------------------------------------------------------------------

/// Cross-check `TypeId` discriminants against the documented `cudf::type_id`
/// numeric values (see `cudf/include/cudf/types.hpp`). Any silent renumber
/// in the `-sys` crate's enum would break wire compatibility with libcudf
/// columns coming back from the library.
#[test]
fn type_id_discriminants_match_upstream() {
    // The integer values come from RAPIDS' `cudf::type_id` enum, which is
    // the wire contract of a column's element type. Re-asserted here so
    // any reorder would fail loudly rather than silently mistyping data.
    assert_eq!(TypeId::Empty as i32, 0);
    assert_eq!(TypeId::Int8 as i32, 1);
    assert_eq!(TypeId::Int16 as i32, 2);
    assert_eq!(TypeId::Int32 as i32, 3);
    assert_eq!(TypeId::Int64 as i32, 4);
    assert_eq!(TypeId::Uint8 as i32, 5);
    assert_eq!(TypeId::Uint16 as i32, 6);
    assert_eq!(TypeId::Uint32 as i32, 7);
    assert_eq!(TypeId::Uint64 as i32, 8);
    assert_eq!(TypeId::Float32 as i32, 9);
    assert_eq!(TypeId::Float64 as i32, 10);
    assert_eq!(TypeId::Bool8 as i32, 11);
    assert_eq!(TypeId::Timestamp_Days as i32, 12);
    assert_eq!(TypeId::Timestamp_Nanoseconds as i32, 16);
    assert_eq!(TypeId::Duration_Days as i32, 17);
    assert_eq!(TypeId::Duration_Nanoseconds as i32, 21);
    assert_eq!(TypeId::Dictionary32 as i32, 22);
    assert_eq!(TypeId::String as i32, 23);
    assert_eq!(TypeId::List as i32, 24);
    assert_eq!(TypeId::Decimal32 as i32, 25);
    assert_eq!(TypeId::Decimal64 as i32, 26);
    assert_eq!(TypeId::Decimal128 as i32, 27);
    assert_eq!(TypeId::Struct as i32, 28);
}

/// `TypeId` derives `Copy + Clone + Debug + Eq + PartialEq` in the `-sys`
/// crate — exercise each so a stray attribute removal fails compilation.
#[test]
fn type_id_has_expected_derives() {
    let a = TypeId::Float32;
    let b = a; // Copy
    let c = a.clone(); // Clone
    assert_eq!(a, b);
    assert_eq!(a, c);
    assert_ne!(a, TypeId::Float64);
    // Debug
    let s = format!("{a:?}");
    assert!(s.contains("Float32"), "Debug output unexpected: {s}");
}

/// `cudfStatus_t` is the `-sys` status wrapper; the safe crate maps it
/// onto `Error::Status`. Check the documented success/failure constants
/// stay aligned.
#[test]
fn status_constants_are_stable() {
    assert!(cudfStatus_t::SUCCESS.is_success());
    assert!(!cudfStatus_t::LOGIC_ERROR.is_success());
    assert!(!cudfStatus_t::CUDA_ERROR.is_success());
    assert!(!cudfStatus_t::OUT_OF_MEMORY.is_success());
    assert!(!cudfStatus_t::NOT_IMPLEMENTED.is_success());
    assert_eq!(cudfStatus_t::SUCCESS.0, 0);
    assert_eq!(cudfStatus_t::LOGIC_ERROR.0, 1);
    assert_eq!(cudfStatus_t::CUDA_ERROR.0, 2);
    assert_eq!(cudfStatus_t::OUT_OF_MEMORY.0, 3);
    assert_eq!(cudfStatus_t::NOT_IMPLEMENTED.0, 4);
}

/// `Error::Path` and `Error::Nul` are the host-side validation arms — they
/// are reachable without ever touching libcudf, so we can exercise their
/// `Display` formatting and the `From<NulError>` round-trip here.
#[test]
fn error_display_round_trips() {
    let e = Error::Path("/no/such/file".to_string());
    let s = format!("{e}");
    assert!(s.contains("invalid path"), "unexpected Display: {s}");
    assert!(s.contains("/no/such/file"), "Display missed payload: {s}");

    // Trigger Error::Nul via CString construction.
    let nul = std::ffi::CString::new(b"contains\0nul".to_vec()).unwrap_err();
    let e: Error = nul.into();
    let s = format!("{e}");
    assert!(s.contains("invalid C string"), "unexpected Display: {s}");

    // Synthesised status arm formats the underlying description.
    let e = Error::Status(cudfStatus_t::OUT_OF_MEMORY, "out of device memory");
    let s = format!("{e}");
    assert!(
        s.contains("OUT_OF_MEMORY") || s.contains("out of device memory"),
        "unexpected status Display: {s}"
    );
}

/// The public types should `impl Debug` (enforced by
/// `#![warn(missing_debug_implementations)]` in the crate). Verify by
/// invoking debug-format on the type names through `std::any`. Owned
/// instances would need a live libcudf, so we only check the type-level
/// guarantee via a trait bound.
#[test]
fn public_types_impl_debug() {
    fn assert_debug<T: std::fmt::Debug>() {}
    assert_debug::<TypeId>();
    assert_debug::<Error>();
    assert_debug::<Column>();
    assert_debug::<Table>();
}

/// `Result<T>` is the documented return alias; exercise the OK path.
#[test]
fn result_alias_is_usable() {
    fn ok() -> Result<i32> {
        Ok(42)
    }
    assert_eq!(ok().unwrap(), 42);
}

// ---------------------------------------------------------------------------
// GPU / libcudf-required checks. Skipped by default. Run with:
//   cargo test -p baracuda-cudf -- --include-ignored
// on a host that has libcudf installed.
// ---------------------------------------------------------------------------

/// Loader probe + version query. Confirms `cudf()` can find a candidate
/// libcudf and that `cudfGetVersion` returns successfully.
#[test]
#[ignore]
fn loader_returns_version() {
    let v = version().expect("cudf loader/version failed");
    // The C ABI returns an integer such as 25_06_00; assert it's
    // monotonic with the oldest supported RAPIDS release we target.
    assert!(v >= 22_00_00, "unexpectedly old libcudf: {v}");
}

/// Allocate an empty column of every supported `TypeId`, query its size
/// and type back, then let the destructor run. Catches handle-lifecycle
/// bugs in the safe wrapper.
#[test]
#[ignore]
fn create_empty_column_lifecycle() {
    let types = [
        TypeId::Int8,
        TypeId::Int32,
        TypeId::Int64,
        TypeId::Uint32,
        TypeId::Float32,
        TypeId::Float64,
        TypeId::Bool8,
    ];
    for t in types {
        let col = Column::new_empty(t, 16).expect("create_empty failed");
        assert_eq!(col.len().unwrap(), 16);
        assert!(!col.is_empty().unwrap());
        assert_eq!(col.type_id().unwrap(), t);
        assert!(!col.as_raw().is_null());
        // Drop runs here.
    }
}

/// `from_csv`/`from_parquet` on a non-existent path should surface as a
/// non-success `cudfStatus_t`, not panic.
#[test]
#[ignore]
fn from_csv_missing_file_returns_error() {
    let r = Table::from_csv("/definitely/does/not/exist.csv");
    assert!(r.is_err(), "expected error from missing csv path");
}
