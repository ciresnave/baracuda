//! GPU-gated integration tests for Wave-14 Driver-API additions:
//! miscellaneous memcpy + memset variants.

use baracuda_cuda_sys::driver;
use baracuda_driver::memory::{memset_u16, memset_u16_async, memset_u32_async, memset_u8_async};
use baracuda_driver::{Context, Device, DeviceBuffer, Stream};

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn dtod_async_copy() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = Stream::new(&ctx).unwrap();

    let n = 2048usize;
    let host: Vec<u32> = (0..n as u32).collect();
    let a = DeviceBuffer::from_slice(&ctx, &host).unwrap();
    let b: DeviceBuffer<u32> = DeviceBuffer::new(&ctx, n).unwrap();

    a.copy_to_device_async(&b, &stream).unwrap();
    stream.synchronize().unwrap();

    let mut back = vec![0u32; n];
    b.copy_to_host(&mut back).unwrap();
    assert_eq!(host, back);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn memset_16_and_async_variants() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = Stream::new(&ctx).unwrap();

    let n = 512usize;
    let buf: DeviceBuffer<u16> = DeviceBuffer::new(&ctx, n).unwrap();
    memset_u16(buf.as_raw(), 0xBEEF, n).unwrap();
    let mut back = vec![0u16; n];
    buf.copy_to_host(&mut back).unwrap();
    assert!(back.iter().all(|&v| v == 0xBEEF));

    let buf32: DeviceBuffer<u32> = DeviceBuffer::new(&ctx, n).unwrap();
    memset_u32_async(buf32.as_raw(), 0xCAFE_BABE, n, &stream).unwrap();
    let buf16: DeviceBuffer<u16> = DeviceBuffer::new(&ctx, n).unwrap();
    memset_u16_async(buf16.as_raw(), 0xDEAD, n, &stream).unwrap();
    let buf8: DeviceBuffer<u8> = DeviceBuffer::new(&ctx, n).unwrap();
    memset_u8_async(buf8.as_raw(), 0xA5, n, &stream).unwrap();
    stream.synchronize().unwrap();

    let mut b32 = vec![0u32; n];
    let mut b16 = vec![0u16; n];
    let mut b8 = vec![0u8; n];
    buf32.copy_to_host(&mut b32).unwrap();
    buf16.copy_to_host(&mut b16).unwrap();
    buf8.copy_to_host(&mut b8).unwrap();
    assert!(b32.iter().all(|&v| v == 0xCAFE_BABE));
    assert!(b16.iter().all(|&v| v == 0xDEAD));
    assert!(b8.iter().all(|&v| v == 0xA5));
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn generic_cumemcpy_smoke() {
    // cuMemcpy is the "smart" generic routing: src/dst are device
    // pointers but can refer to unified/host-mapped addresses.
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();

    let n = 256usize;
    let host: Vec<u32> = (0..n as u32).map(|i| i * 3).collect();
    let a = DeviceBuffer::from_slice(&ctx, &host).unwrap();
    let b: DeviceBuffer<u32> = DeviceBuffer::new(&ctx, n).unwrap();

    let d = driver().unwrap();
    let cu = d.cu_memcpy().unwrap();
    let bytes = n * core::mem::size_of::<u32>();
    let rc = unsafe { cu(b.as_raw(), a.as_raw(), bytes) };
    assert!(rc.is_success());
    ctx.synchronize().unwrap();

    let mut back = vec![0u32; n];
    b.copy_to_host(&mut back).unwrap();
    assert_eq!(host, back);
}
