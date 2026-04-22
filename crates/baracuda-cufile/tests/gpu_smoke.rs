//! cuFile (GPUDirect Storage) presence probe. Linux-only in practice —
//! the Windows driver doesn't ship libcufile.

#[test]
fn cufile_probe() {
    match baracuda_cufile::probe() {
        Ok(()) => eprintln!("cuFile loaded"),
        Err(e) => eprintln!("cuFile not available on this host: {e:?}"),
    }
}
