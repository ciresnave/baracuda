//! GPU-gated integration test for Wave-25 Driver-API additions:
//! stream memory ops (`cuStreamWriteValue32/64_v2`,
//! `cuStreamWaitValue32/64_v2`, `cuStreamBatchMemOp_v2`).

use baracuda_cuda_sys::types::{CUstreamBatchMemOpParams, CUstreamWaitValue_flags};
use baracuda_driver::{Context, Device, DeviceBuffer, Stream};

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn stream_write_value_32_writes_through() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = Stream::new(&ctx).unwrap();

    let buf: DeviceBuffer<u32> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    stream.write_value_32(buf.as_raw(), 0xCAFE_BABE, 0).unwrap();
    stream.synchronize().unwrap();

    let mut got = [0u32; 1];
    buf.copy_to_host(&mut got).unwrap();
    assert_eq!(got[0], 0xCAFE_BABE);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn stream_wait_value_32_releases_when_value_matches() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = Stream::new(&ctx).unwrap();

    // Pre-populate the flag memory with the value we'll wait for.
    let flag: DeviceBuffer<u32> = DeviceBuffer::from_slice(&ctx, &[0x1u32]).unwrap();
    let payload: DeviceBuffer<u32> = DeviceBuffer::zeros(&ctx, 1).unwrap();

    // Wait with EQ 1 — should pass immediately since we've already
    // written 1.
    stream
        .wait_value_32(flag.as_raw(), 0x1, CUstreamWaitValue_flags::EQ)
        .unwrap();
    // Then write a sentinel to `payload` so we can detect the wait released.
    stream
        .write_value_32(payload.as_raw(), 0xDEAD_BEEF, 0)
        .unwrap();
    stream.synchronize().unwrap();

    let mut got = [0u32; 1];
    payload.copy_to_host(&mut got).unwrap();
    assert_eq!(got[0], 0xDEAD_BEEF);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn stream_batch_mem_op_applies_all_entries() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = Stream::new(&ctx).unwrap();

    let buf: DeviceBuffer<u32> = DeviceBuffer::zeros(&ctx, 4).unwrap();
    // Addresses of the four slots as device pointers.
    let bp = buf.as_raw();
    let addr = |i: usize| baracuda_cuda_sys::CUdeviceptr(bp.0 + (i * 4) as u64);

    let mut ops = [
        CUstreamBatchMemOpParams::write_value_32(addr(0), 0x11, 0),
        CUstreamBatchMemOpParams::write_value_32(addr(1), 0x22, 0),
        CUstreamBatchMemOpParams::write_value_32(addr(2), 0x33, 0),
        CUstreamBatchMemOpParams::write_value_32(addr(3), 0x44, 0),
    ];
    stream.batch_mem_op(&mut ops, 0).unwrap();
    stream.synchronize().unwrap();

    let mut got = [0u32; 4];
    buf.copy_to_host(&mut got).unwrap();
    assert_eq!(got, [0x11, 0x22, 0x33, 0x44]);
}
