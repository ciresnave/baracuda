//! nvCOMP presence probe.

use baracuda_driver::require_optional;

#[test]
fn nvcomp_probe() {
    // Absent library: a declared skip (visible to `cargo gpu-test`), not a silent pass.
    require_optional!(baracuda_nvcomp::probe(), "nvCOMP");
    eprintln!("nvCOMP loaded");
}
