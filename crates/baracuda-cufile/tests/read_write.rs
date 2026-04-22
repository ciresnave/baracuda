//! cuFile integration test — register a temp file, write device data,
//! read it back. Linux-only (cuFile has no Windows/macOS support).

#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;

#[cfg(target_os = "linux")]
use baracuda_runtime::{Device, DeviceBuffer};

#[test]
#[ignore = "requires cuFile + GDS-capable filesystem (Linux-only)"]
#[cfg(target_os = "linux")]
fn cufile_write_read_roundtrip() {
    if baracuda_cufile::probe().is_err() {
        eprintln!("cuFile not loadable on this host — skipping");
        return;
    }

    let _driver = match baracuda_cufile::Driver::open() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Driver::open failed ({e:?}) — likely no GDS-capable filesystem; skipping");
            return;
        }
    };

    Device::from_ordinal(0).set_current().unwrap();

    // Prepare host data, upload to device.
    let n = 4096usize;
    let host: Vec<u32> = (0..n as u32).collect();
    let d_src: DeviceBuffer<u32> = DeviceBuffer::from_slice(&host).unwrap();
    let d_dst: DeviceBuffer<u32> = DeviceBuffer::new(n).unwrap();

    // Temp file — we need a GDS-capable FS; typically /mnt/nvme or /tmp on
    // ext4 with nvme (not all /tmp supports GDS).
    let tmp_path = std::env::var("BARACUDA_CUFILE_TEST_PATH")
        .unwrap_or_else(|_| "/tmp/baracuda_cufile_test.bin".into());
    let file = match std::fs::File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&tmp_path)
    {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Can't open {tmp_path}: {e} — skipping");
            return;
        }
    };

    let fh = match unsafe { baracuda_cufile::FileHandle::register(file.as_raw_fd()) } {
        Ok(h) => h,
        Err(e) => {
            eprintln!("FileHandle::register: {e:?} — likely GDS-incompatible FS; skipping");
            return;
        }
    };

    let bytes = (n * core::mem::size_of::<u32>()) as usize;

    // Write from device to file.
    unsafe { fh.write(d_src.as_raw() as *const _, bytes, 0, 0) }.unwrap();

    // Read back into a different device buffer.
    unsafe { fh.read(d_dst.as_raw(), bytes, 0, 0) }.unwrap();

    let mut got = vec![0u32; n];
    d_dst.copy_to_host(&mut got).unwrap();
    assert_eq!(host, got);
    eprintln!("cuFile write+read round-trip OK ({bytes} B)");
}

#[cfg(not(target_os = "linux"))]
#[test]
fn cufile_not_supported_on_non_linux() {
    assert!(
        baracuda_cufile::probe().is_err(),
        "expected cuFile to be unavailable on non-Linux"
    );
}
