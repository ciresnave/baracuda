//! cuTENSOR presence probe. cuTENSOR is a separate NVIDIA download and
//! is not part of the default CUDA Toolkit install — the test gracefully
//! skips if the library isn't available.

use baracuda_driver::require_optional;

#[test]
fn cutensor_probe() {
    // Absent library: a declared skip (visible to `cargo gpu-test`), not a silent pass.
    require_optional!(baracuda_cutensor::probe(), "cuTENSOR");
    eprintln!("cuTENSOR loaded");
}
