//! Error-type plumbing edge cases. All host-only; no CUDA needed.

use baracuda_core::{Error, LoaderError};
use baracuda_types::{CudaStatus, CudaVersion};

// A throwaway status type with the bare minimum `CudaStatus` impl.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct FakeStatus(i32);

impl CudaStatus for FakeStatus {
    fn code(self) -> i32 {
        self.0
    }
    fn name(self) -> &'static str {
        "FAKE_STATUS"
    }
    fn description(self) -> &'static str {
        "fake status used in tests"
    }
    fn is_success(self) -> bool {
        self.0 == 0
    }
    fn library(self) -> &'static str {
        "fake"
    }
}

#[test]
fn error_check_success_is_ok() {
    let r = Error::check(FakeStatus(0));
    assert!(r.is_ok());
}

#[test]
fn error_check_non_zero_is_err() {
    let r = Error::check(FakeStatus(5));
    let e = r.expect_err("non-zero status must map to Err");
    match e {
        Error::Status { status } => assert_eq!(status, FakeStatus(5)),
        other => panic!("unexpected variant: {other:?}"),
    }
}

#[test]
fn loader_error_display_mentions_library() {
    let e = LoaderError::library_not_found("pretend-cuda", &["libpretend.so"]);
    let msg = format!("{e}");
    assert!(msg.contains("pretend-cuda"));
    assert!(msg.contains("libpretend.so"));
}

#[test]
fn loader_error_symbol_not_found() {
    let e = LoaderError::SymbolNotFound {
        library: "pretend",
        symbol: "missingFunc",
    };
    let msg = format!("{e}");
    assert!(msg.contains("pretend"));
    assert!(msg.contains("missingFunc"));
}

#[test]
fn loader_error_version_too_old() {
    let e = LoaderError::VersionTooOld {
        symbol: "cuNewThing",
        required: CudaVersion::CUDA_13_0,
        installed: CudaVersion::CUDA_12_0,
    };
    let msg = format!("{e}");
    assert!(msg.contains("cuNewThing"));
    assert!(msg.contains("13"));
    assert!(msg.contains("12"));
}

#[test]
fn feature_not_supported_formats_cleanly() {
    let e: Error<FakeStatus> = Error::FeatureNotSupported {
        api: "cuSomethingExotic",
        since: CudaVersion::CUDA_12_6,
    };
    let msg = format!("{e}");
    assert!(msg.contains("cuSomethingExotic"));
    assert!(msg.contains("12"));
    assert!(msg.contains("6"));
}
