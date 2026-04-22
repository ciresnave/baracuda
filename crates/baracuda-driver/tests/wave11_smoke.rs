//! GPU-gated integration tests for Wave-11 Driver-API additions:
//! pinned host memory (`cuMemHostAlloc` + `cuMemHostRegister` paths).

use baracuda_driver::pinned::{flags, PinnedBuffer, PinnedRegistration};
use baracuda_driver::{Context, Device, DeviceBuffer};

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn pinned_buffer_allocates_and_readbacks() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();

    let n = 4096usize;
    let mut host = PinnedBuffer::<f32>::new(&ctx, n).unwrap();
    assert_eq!(host.len(), n);

    // Fill via deref, copy to device, copy back, compare.
    for (i, x) in host.iter_mut().enumerate() {
        *x = i as f32 * 0.75;
    }

    let d_buf = DeviceBuffer::<f32>::new(&ctx, n).unwrap();
    d_buf.copy_from_host(&host).unwrap();

    let mut back = PinnedBuffer::<f32>::new(&ctx, n).unwrap();
    d_buf.copy_to_host(&mut back).unwrap();

    for (a, b) in host.iter().zip(back.iter()) {
        assert_eq!(a, b);
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn pinned_registration_pins_existing_vec() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();

    let n = 1024usize;
    let expected: Vec<u32> = (0..n as u32).collect();
    let mut host: Vec<u32> = expected.clone();

    // Pin, round-trip, unpin — scope the borrow so we can compare after.
    {
        let reg = PinnedRegistration::register_with_flags(&mut host, flags::PORTABLE).unwrap();
        assert_eq!(reg.len(), n);
    }

    // Round trip via an unrelated device buffer — only proves the host
    // memory remains usable after unpinning.
    let d = DeviceBuffer::<u32>::from_slice(&ctx, &host).unwrap();
    let mut back = vec![0u32; n];
    d.copy_to_host(&mut back).unwrap();
    assert_eq!(expected, back);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn pinned_buffer_with_device_map() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();

    let n = 256usize;
    let buf = PinnedBuffer::<u32>::with_flags(&ctx, n, flags::DEVICEMAP).unwrap();

    // Flags queried back should include DEVICEMAP.
    let f = buf.flags().unwrap();
    eprintln!("pinned buffer flags = {f:#x}");
    assert!(f & flags::DEVICEMAP != 0, "DEVICEMAP flag lost");

    // Device pointer should be non-zero when DEVICEMAP is on.
    let dptr = buf.device_ptr().unwrap();
    assert_ne!(dptr.0, 0, "device pointer should be mapped");
}
