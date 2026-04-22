//! GPU-gated NPP smoke test: query version.
//!
//! NPP's signal/image functions moved to `_Ctx`-suffixed variants in
//! CUDA 12+ (they now require an explicit `NppStreamContext`). v0.1 of
//! `baracuda-npp` only proves the loader works via `nppGetLibVersion`;
//! wrapping the `_Ctx` family with an ergonomic context builder lands in
//! baracuda-npp v0.2.

use baracuda_npp::version;

#[test]
#[ignore = "requires an NVIDIA GPU with NPP installed"]
fn query_version() {
    let v = version().expect("nppGetLibVersion");
    eprintln!("NPP {}.{}.{}", v.major, v.minor, v.build);
    assert!(
        v.major >= 11,
        "unexpectedly old NPP major version: {}",
        v.major
    );
}
