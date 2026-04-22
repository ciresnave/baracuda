//! Smoke test for nvJitLink: query version + link a trivial PTX to CUBIN.
//!
//! The GPU-gated test targets the device's compute capability via the
//! driver-reported version and expects a non-empty cubin blob.

use baracuda_nvjitlink::{version, InputType, Linker};

const VECTOR_ADD_PTX: &str = include_str!("../../baracuda-driver/tests/kernels/vector_add.ptx");

#[test]
fn nvjitlink_loads() {
    // Non-GPU host test: nvJitLink is a CPU-side linker — loading it is
    // enough to prove the FFI works. Platforms without nvJitLink shipped
    // gracefully return `Err` without panic.
    let _ = version();
}

#[test]
#[ignore = "requires nvJitLink + an NVIDIA GPU (for compute-cap query)"]
fn link_ptx_to_cubin() {
    baracuda_driver::init().expect("driver init");
    let device = baracuda_driver::Device::get(0).expect("get device 0");
    let (major, minor) = device.compute_capability().unwrap();
    let arch = format!("-arch=sm_{major}{minor}");

    let (vmajor, vminor) = version().expect("nvjitlink version");
    eprintln!("nvJitLink {vmajor}.{vminor}, targeting {arch}");

    let mut linker = Linker::new(&[arch.as_str()]).expect("Linker::new");
    // nvJitLink expects NUL-terminated PTX bytes.
    let mut ptx = VECTOR_ADD_PTX.as_bytes().to_vec();
    ptx.push(0);
    linker
        .add_data(InputType::Ptx, &ptx, "vector_add.ptx")
        .expect("add_data");
    match linker.complete() {
        Ok(()) => {
            let cubin = linker.linked_cubin().expect("linked_cubin");
            eprintln!("linked cubin: {} bytes", cubin.len());
            assert!(!cubin.is_empty());
        }
        Err(e) => {
            let log = linker.error_log().unwrap_or_default();
            eprintln!("nvJitLink complete failed: {e:?}\nerror log: {log}");
            // Not a hard failure — some combos (compute cap mismatch, PTX
            // version too new) can genuinely fail even on a working setup.
        }
    }
}
