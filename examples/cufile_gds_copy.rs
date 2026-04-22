//! cuFile GPUDirect Storage copy demo (Linux-only).
//!
//! Writes a 64 MiB device buffer into a file via cuFile, reads it
//! back into a different device buffer, and verifies bit-equality.
//!
//! Requires a GDS-capable filesystem — set `BARACUDA_CUFILE_PATH` to
//! a path on an NVMe + ext4/XFS mount. `/tmp` usually isn't GDS-capable.

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("cuFile is Linux-only");
    std::process::exit(1);
}

#[cfg(target_os = "linux")]
fn main() {
    use std::os::fd::AsRawFd;

    use baracuda_cufile::{BufRegistration, Driver, FileHandle};
    use baracuda_runtime::{Device, DeviceBuffer};

    if let Err(e) = baracuda_cufile::probe() {
        eprintln!("cuFile not available: {e:?}");
        std::process::exit(1);
    }

    let _driver = Driver::open().expect("Driver::open");
    Device::from_ordinal(0).set_current().expect("set device");

    let n_u32 = 16 * 1024 * 1024; // 64 MiB of u32
    let bytes = n_u32 * 4;

    println!("cuFile GDS: {} B round-trip", bytes);

    // Build a device-side payload.
    let host: Vec<u32> = (0..n_u32 as u32).collect();
    let d_src: DeviceBuffer<u32> = DeviceBuffer::from_slice(&host).unwrap();
    let d_dst: DeviceBuffer<u32> = DeviceBuffer::new(n_u32).unwrap();

    // Register buffers for the fastest DMA path.
    let _src_reg = unsafe { BufRegistration::register(d_src.as_raw(), bytes, 0).unwrap() };
    let _dst_reg = unsafe { BufRegistration::register(d_dst.as_raw(), bytes, 0).unwrap() };

    // Open a file on a GDS-capable mount.
    let path = std::env::var("BARACUDA_CUFILE_PATH")
        .unwrap_or_else(|_| "/mnt/nvme/baracuda_cufile_demo.bin".into());
    let file = std::fs::File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&path)
        .unwrap_or_else(|e| panic!("open {path}: {e}"));
    let fh = unsafe { FileHandle::register(file.as_raw_fd()).expect("register fd") };

    // Write device → file.
    let t0 = std::time::Instant::now();
    let wrote = unsafe { fh.write(d_src.as_raw(), bytes, 0, 0).unwrap() };
    let write_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Read file → device.
    let t1 = std::time::Instant::now();
    let read = unsafe { fh.read(d_dst.as_raw(), bytes, 0, 0).unwrap() };
    let read_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // Verify.
    let mut back = vec![0u32; n_u32];
    d_dst.copy_to_host(&mut back).unwrap();
    assert_eq!(host, back);

    let w_gbps = (bytes as f64 / 1e9) / (write_ms / 1000.0);
    let r_gbps = (bytes as f64 / 1e9) / (read_ms / 1000.0);
    println!(
        "  write: {} B in {:.2} ms ({:.2} GB/s)",
        wrote, write_ms, w_gbps
    );
    println!(
        "  read:  {} B in {:.2} ms ({:.2} GB/s)",
        read, read_ms, r_gbps
    );
    println!("OK — round-trip bit-exact");

    // Clean up the temp file.
    let _ = std::fs::remove_file(&path);
}
