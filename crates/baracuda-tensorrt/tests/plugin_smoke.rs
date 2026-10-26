//! The plugin registry against a real TensorRT: register the built-in plugins,
//! list and look up creators, and refuse a library that is not there.
//!
//! ```text
//! cargo gpu-test -p baracuda-tensorrt --features shim,device-tests
//! ```

#![cfg(all(feature = "shim", feature = "device-tests"))]

use std::sync::{Arc, OnceLock};

use baracuda_driver::require_optional;
use baracuda_tensorrt::{
    Logger, PluginCreatorKind, PluginRegistry, Severity, init_builtin_plugins,
};

/// Register the built-in plugins once per process; `None` when TensorRT or its
/// plugin library is not here.
fn registry() -> Option<PluginRegistry> {
    static INIT: OnceLock<Option<PluginRegistry>> = OnceLock::new();
    *INIT.get_or_init(|| {
        let logger = Arc::new(Logger::stderr(Severity::Error).ok()?);
        let ok = init_builtin_plugins(&logger, "").ok()?;
        // The process-wide `ILogger` outlives this handle; keep the Rust side alive too.
        std::mem::forget(logger);
        ok.then(|| PluginRegistry::global().ok()).flatten()
    })
}

#[test]
fn builtin_plugins_register_and_are_listed() {
    let reg = require_optional!(registry(), "TensorRT plugin library + shim");
    let creators = reg.creators().expect("list creators");
    assert!(
        creators.len() >= 5,
        "expected the built-in plugins, found {}",
        creators.len()
    );
    assert!(creators.iter().all(|c| !c.name.is_empty()), "{creators:?}");
    assert!(
        creators
            .iter()
            .any(|c| c.kind != PluginCreatorKind::Unknown),
        "no creator reported a readable interface kind: {creators:?}"
    );
}

#[test]
fn a_listed_creator_is_found_by_its_triple_and_a_missing_one_is_not() {
    let reg = require_optional!(registry(), "TensorRT plugin library + shim");
    let creators = reg.creators().expect("list creators");
    let first = creators
        .iter()
        .find(|c| c.kind != PluginCreatorKind::Unknown)
        .expect("a readable creator");
    // Control: the lookup finds what the listing shows.
    let found = reg
        .creator(&first.name, &first.version, &first.namespace)
        .expect("lookup")
        .expect("listed creator must be found");
    assert_eq!(&found, first);
    // The question being asked: a name nobody registered.
    assert!(
        reg.creator("NoSuchPlugin_BARACUDA", "1", "")
            .expect("lookup")
            .is_none()
    );
}

#[test]
fn a_missing_plugin_library_is_refused() {
    let reg = require_optional!(registry(), "TensorRT plugin library + shim");
    let err = reg
        .load_library("C:/definitely/not/a/plugin_library.dll")
        .expect_err("a missing library must not load");
    assert!(matches!(err, baracuda_tensorrt::Error::NullHandle { .. }));
}
