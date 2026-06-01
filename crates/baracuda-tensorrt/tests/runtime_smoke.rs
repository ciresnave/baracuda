//! Runtime-path smoke tests for `baracuda-tensorrt`.
//!
//! The inference tests are `#[ignore]`d: they require both a working TensorRT
//! install (`libnvinfer` resolvable on the loader path) **and** the C-ABI shim
//! that exports the baracuda-defined `trt*` symbols (see `AUDIT.md`). Neither is
//! present on a stock machine, so these are opt-in:
//!
//! ```text
//! cargo test -p baracuda-tensorrt -- --ignored
//! ```
//!
//! The non-ignored test exercises the pure-Rust `Dims` helper and always runs.

use baracuda_tensorrt::{Dims, Runtime};

#[test]
fn dims_new_and_slice_roundtrip() {
    let d = Dims::new(&[1, 3, 224, 224]);
    assert_eq!(d.rank, 4);
    assert_eq!(d.as_slice(), &[1, 3, 224, 224]);

    // Rank is clamped to TRT_MAX_DIMS (8); extra axes are dropped.
    let too_many = Dims::new(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    assert_eq!(too_many.rank, 8);
    assert_eq!(too_many.as_slice(), &[1, 2, 3, 4, 5, 6, 7, 8]);

    // Empty dims are valid (scalar / unranked).
    let empty = Dims::new(&[]);
    assert_eq!(empty.rank, 0);
    assert!(empty.as_slice().is_empty());
}

#[test]
#[ignore = "requires a working TensorRT install + the C-ABI shim (see AUDIT.md)"]
fn version_reports_trt10_or_newer() {
    let v = baracuda_tensorrt::version().expect("getInferLibVersion");
    // TRT version is MAJOR*1000 + MINOR*100 + PATCH; we wrap TRT 10+.
    assert!(v >= 10_000, "expected TensorRT >= 10, got encoded {v}");
}

#[test]
#[ignore = "requires a working TensorRT install + the C-ABI shim (see AUDIT.md)"]
fn deserialize_garbage_blob_is_graceful() {
    // A runtime built without a logger; older TRT may refuse (returns null),
    // which surfaces as Err — that is itself an acceptable graceful outcome.
    let rt = match Runtime::with_null_logger() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("runtime creation refused without logger (acceptable): {e}");
            return;
        }
    };
    // Feeding a non-engine blob must error, never panic / UB.
    let bogus = [0u8; 32];
    let res = rt.deserialize_engine(&bogus);
    assert!(res.is_err(), "deserializing garbage must fail gracefully");
}
