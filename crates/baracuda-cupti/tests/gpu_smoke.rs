//! CUPTI presence + version probe.

use baracuda_driver::require_optional;

#[test]
fn cupti_version_query() {
    // Absent library: a declared skip (visible to `cargo gpu-test`), not a silent pass.
    let v = require_optional!(baracuda_cupti::version(), "CUPTI");
    eprintln!("CUPTI version: {v}");
}
