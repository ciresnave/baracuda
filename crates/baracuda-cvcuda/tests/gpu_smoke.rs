//! CV-CUDA presence probe.

use baracuda_driver::require_optional;

#[test]
fn cvcuda_probe() {
    // Absent library: a declared skip (visible to `cargo gpu-test`), not a silent pass.
    require_optional!(baracuda_cvcuda::probe(), "CV-CUDA");
    eprintln!("CV-CUDA loaded");
}
