//! nvCOMP presence probe.

#[test]
fn nvcomp_probe() {
    match baracuda_nvcomp::probe() {
        Ok(()) => eprintln!("nvCOMP loaded"),
        Err(e) => eprintln!("nvCOMP not available on this host: {e:?}"),
    }
}
