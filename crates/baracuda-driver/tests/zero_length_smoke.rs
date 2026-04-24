//! Zero-length DeviceBuffer short-circuit — GPU-gated.
//!
//! The short-circuit itself is pure bookkeeping (null pointer + len = 0),
//! but verifying it end-to-end requires a live context, so the test is
//! ignored by default. Run with `BARACUDA_GPU_TESTS=1` on a machine with
//! a working NVIDIA driver.

use baracuda_driver::{Context, Device, DeviceBuffer, PinnedBuffer, Stream};

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn zero_length_buffer_round_trips_without_cuda_calls() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = Stream::new(&ctx).unwrap();

    // All three constructors should succeed even though CUDA itself
    // rejects 0-byte cuMemAlloc.
    let empty: DeviceBuffer<f32> = DeviceBuffer::new(&ctx, 0).unwrap();
    assert_eq!(empty.len(), 0);
    assert!(empty.is_empty());

    let zeroed: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, 0).unwrap();
    assert_eq!(zeroed.len(), 0);

    let empty_slice: Vec<i64> = Vec::new();
    let from_slice: DeviceBuffer<i64> = DeviceBuffer::from_slice(&ctx, &empty_slice).unwrap();
    assert_eq!(from_slice.len(), 0);

    // Copies on zero-length buffers must be no-ops, not CUDA errors.
    let mut host_out: Vec<f32> = Vec::new();
    empty.copy_to_host(&mut host_out).unwrap();
    assert!(host_out.is_empty());

    let empty_src: Vec<f32> = Vec::new();
    empty.copy_from_host(&empty_src).unwrap();

    let empty_async: DeviceBuffer<f32> = DeviceBuffer::new(&ctx, 0).unwrap();
    empty.copy_to_device_async(&empty_async, &stream).unwrap();
    stream.synchronize().unwrap();
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn nonempty_buffer_still_works_after_short_circuit_additions() {
    // Regression guard: the short-circuit must not affect non-empty paths.
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();

    let host = vec![1.0f32, 2.0, 3.0, 4.0];
    let buf = DeviceBuffer::from_slice(&ctx, &host).unwrap();
    let mut back = vec![0.0f32; host.len()];
    buf.copy_to_host(&mut back).unwrap();
    assert_eq!(host, back);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn zero_length_pinned_buffer_is_sound() {
    // cuMemHostAlloc(0) returns a null pointer; without the short-circuit,
    // PinnedBuffer's Deref would trigger UB in slice::from_raw_parts.
    // Construct an empty PinnedBuffer, deref it, and verify it round-trips.
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();

    let empty: PinnedBuffer<f32> = PinnedBuffer::new(&ctx, 0).unwrap();
    assert_eq!(empty.len(), 0);
    assert!(empty.is_empty());

    // Deref must be sound — slice::from_raw_parts(dangling, 0) is OK.
    let slice: &[f32] = &empty;
    assert_eq!(slice.len(), 0);

    // device_ptr / flags degrade gracefully.
    assert_eq!(empty.device_ptr().unwrap().0, 0);
    assert_eq!(empty.flags().unwrap(), 0);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn nonempty_pinned_buffer_still_works() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();

    let mut pinned: PinnedBuffer<f32> = PinnedBuffer::new(&ctx, 4).unwrap();
    pinned.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(&pinned[..], &[1.0, 2.0, 3.0, 4.0]);
}
