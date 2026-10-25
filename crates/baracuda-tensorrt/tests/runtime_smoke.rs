//! Runtime-path smoke tests for `baracuda-tensorrt`.
//!
//! The inference tests are gated behind the `device-tests` feature: they require a working TensorRT
//! install (`libnvinfer` resolvable on the loader path) **and** the C-ABI shim,
//! which is compiled only with the `shim` feature (needs the TensorRT SDK
//! headers; see `AUDIT.md`). Neither is present on a stock machine, so these are
//! opt-in:
//!
//! ```text
//! cargo test -p baracuda-tensorrt --features shim --features device-tests
//! ```
//!
//! The non-ignored test exercises the pure-Rust `Dims` helper and always runs.

#[cfg(feature = "device-tests")]
use baracuda_driver::require_optional;
use baracuda_tensorrt::Dims;
#[cfg(feature = "device-tests")]
use baracuda_tensorrt::{Logger, Runtime, Severity};
#[cfg(feature = "device-tests")]
use std::sync::{Arc, Mutex};

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
#[cfg(feature = "device-tests")]
fn version_reports_trt10_or_newer() {
    // TensorRT absent (libnvinfer not loadable) is a legitimate skip under
    // `cargo gpu-test`, but a non-Loader error means TRT is present and
    // misbehaving -- a real regression -- so only a Loader error declares-skip;
    // anything else panics (don't let require_optional! swallow it).
    let probe = match baracuda_tensorrt::version() {
        Ok(v) => Some(v),
        Err(baracuda_tensorrt::Error::Loader(_)) => None,
        Err(e) => panic!("getInferLibVersion returned a non-absence error: {e}"),
    };
    let v = require_optional!(probe, "TensorRT runtime (getInferLibVersion)");
    // TRT version is MAJOR*1000 + MINOR*100 + PATCH; we wrap TRT 10+.
    assert!(v >= 10_000, "expected TensorRT >= 10, got encoded {v}");
}

#[test]
#[cfg(feature = "device-tests")]
fn deserialize_garbage_blob_is_graceful() {
    let logger = require_optional!(
        Logger::stderr(Severity::Warning).map(Arc::new),
        "C-ABI shim (Logger::new)"
    );
    let rt = require_optional!(
        Runtime::with_logger(logger),
        "TensorRT runtime (Runtime::with_logger)"
    );
    // Feeding a non-engine blob must error, never panic / UB.
    let bogus = [0u8; 32];
    let res = rt.deserialize_engine(&bogus);
    assert!(res.is_err(), "deserializing garbage must fail gracefully");
}

/// The logger's callback really runs: deserializing garbage makes TensorRT log
/// an error, and that message reaches the Rust closure.
#[test]
#[cfg(feature = "device-tests")]
fn logger_forwards_messages_to_the_closure() {
    let seen: Arc<Mutex<Vec<(Severity, String)>>> = Arc::default();
    let sink = Arc::clone(&seen);
    let logger = require_optional!(
        Logger::new(move |severity, msg| sink.lock().unwrap().push((severity, msg.to_owned())))
            .map(Arc::new),
        "C-ABI shim (Logger::new)"
    );
    let rt = require_optional!(
        Runtime::with_logger(logger),
        "TensorRT runtime (Runtime::with_logger)"
    );
    assert!(rt.deserialize_engine(&[0u8; 32]).is_err());
    let seen = seen.lock().unwrap();
    assert!(
        seen.iter()
            .any(|(s, _)| (*s as i32) <= (Severity::Error as i32)),
        "no error-or-worse message reached the closure; saw {seen:?}"
    );
}

/// TensorRT 10 refuses a null logger. The positive control is the same call
/// with a real logger, which succeeds in the same process.
#[test]
#[cfg(feature = "device-tests")]
fn null_logger_is_refused_but_a_real_logger_is_accepted() {
    let logger = require_optional!(
        Logger::stderr(Severity::Error).map(Arc::new),
        "C-ABI shim (Logger::new)"
    );
    require_optional!(
        Runtime::with_logger(logger),
        "TensorRT runtime (Runtime::with_logger)"
    );
    assert!(
        matches!(
            Runtime::with_null_logger(),
            Err(baracuda_tensorrt::Error::NullHandle { .. })
        ),
        "TensorRT 10 is expected to refuse a null logger"
    );
}

/// A panicking closure must not unwind into C++.
#[test]
#[cfg(feature = "device-tests")]
fn panicking_logger_does_not_unwind_into_cpp() {
    let logger = require_optional!(
        Logger::new(|_, _| panic!("logger panic")).map(Arc::new),
        "C-ABI shim (Logger::new)"
    );
    let rt = require_optional!(
        Runtime::with_logger(logger),
        "TensorRT runtime (Runtime::with_logger)"
    );
    assert!(rt.deserialize_engine(&[0u8; 32]).is_err());
}

/// TensorRT keeps one logger per process and ignores later ones while a runtime
/// exists. Everything built from the first logger is dropped here while the
/// second runtime is still alive; the second logger must still hear TensorRT,
/// and TensorRT must not be left calling a freed logger.
#[test]
#[cfg(feature = "device-tests")]
fn a_later_logger_survives_the_first_one_being_dropped() {
    let first = require_optional!(
        Logger::stderr(Severity::Error).map(Arc::new),
        "C-ABI shim (Logger::new)"
    );
    let first_rt = require_optional!(
        Runtime::with_logger(first),
        "TensorRT runtime (Runtime::with_logger)"
    );

    let seen: Arc<Mutex<Vec<(Severity, String)>>> = Arc::default();
    let sink = Arc::clone(&seen);
    let second =
        Logger::new(move |severity, msg| sink.lock().unwrap().push((severity, msg.to_owned())))
            .map(Arc::new)
            .expect("second logger");
    let second_rt = Runtime::with_logger(second).expect("second runtime");

    // Drop the first runtime and, with it, the last handle to the first logger.
    drop(first_rt);

    assert!(second_rt.deserialize_engine(&[0u8; 32]).is_err());
    let seen = seen.lock().unwrap();
    assert!(
        seen.iter()
            .any(|(s, _)| (*s as i32) <= (Severity::Error as i32)),
        "the second logger heard nothing after the first was dropped; saw {seen:?}"
    );
}

/// TensorRT ignores a second logger while a runtime made with the first is
/// alive, so with one `ILogger` per `Logger` the second logger heard nothing.
/// Both loggers must hear a message raised through the second runtime.
#[test]
#[cfg(feature = "device-tests")]
fn every_live_logger_hears_a_message_from_any_runtime() {
    let seen_a: Arc<Mutex<Vec<Severity>>> = Arc::default();
    let seen_b: Arc<Mutex<Vec<Severity>>> = Arc::default();
    let (sink_a, sink_b) = (Arc::clone(&seen_a), Arc::clone(&seen_b));
    let logger_a = require_optional!(
        Logger::new(move |severity, _| sink_a.lock().unwrap().push(severity)).map(Arc::new),
        "C-ABI shim (Logger::new)"
    );
    let logger_b = Logger::new(move |severity, _| sink_b.lock().unwrap().push(severity))
        .map(Arc::new)
        .expect("second logger");
    let _rt_a = require_optional!(
        Runtime::with_logger(logger_a),
        "TensorRT runtime (Runtime::with_logger)"
    );
    let rt_b = Runtime::with_logger(logger_b).expect("second runtime");

    assert!(rt_b.deserialize_engine(&[0u8; 32]).is_err());
    let errors = |seen: &Mutex<Vec<Severity>>| {
        seen.lock()
            .unwrap()
            .iter()
            .any(|s| (*s as i32) <= (Severity::Error as i32))
    };
    assert!(errors(&seen_b), "the runtime's own logger heard no error");
    assert!(errors(&seen_a), "the other live logger heard no error");
}
