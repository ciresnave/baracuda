//! cuTENSOR presence probe. cuTENSOR is a separate NVIDIA download and
//! is not part of the default CUDA Toolkit install — the test gracefully
//! skips if the library isn't available.

#[test]
fn cutensor_probe() {
    match baracuda_cutensor::probe() {
        Ok(()) => eprintln!("cuTENSOR loaded"),
        Err(e) => eprintln!("cuTENSOR not available on this host: {e:?}"),
    }
}
