//! CV-CUDA presence probe.

#[test]
fn cvcuda_probe() {
    match baracuda_cvcuda::probe() {
        Ok(()) => eprintln!("CV-CUDA loaded"),
        Err(e) => eprintln!("CV-CUDA not available on this host: {e:?}"),
    }
}
