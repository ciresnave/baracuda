//! cuFile (GPUDirect Storage) presence probe. Linux-only in practice —
//! the Windows driver doesn't ship libcufile.

use baracuda_driver::require_optional;

#[test]
fn cufile_probe() {
    // Absent library: a declared skip (visible to `cargo gpu-test`), not a silent pass.
    require_optional!(baracuda_cufile::probe(), "cuFile (GPUDirect Storage)");
    eprintln!("cuFile loaded");
}
