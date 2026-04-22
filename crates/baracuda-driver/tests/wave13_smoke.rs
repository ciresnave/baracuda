//! GPU-gated integration tests for Wave-13 Driver-API additions:
//! stream extras (`cuStreamGetId`, `cuStreamGetCaptureInfo`,
//! `cuStreamCopyAttributes`, `cuStreamAttachMemAsync`).

use baracuda_cuda_sys::CUdeviceptr;
use baracuda_driver::{CaptureMode, Context, Device, ManagedBuffer, Stream};

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn stream_get_id_and_copy_attributes() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();

    let s1 = Stream::new(&ctx).unwrap();
    let s2 = Stream::new(&ctx).unwrap();

    let id1 = s1.id().unwrap();
    let id2 = s2.id().unwrap();
    eprintln!("stream ids: s1={id1}, s2={id2}");
    assert_ne!(id1, id2, "different streams should have different ids");

    // Just exercise copy_attributes_from — there's no observable side
    // effect without setting an access-policy window first.
    s1.copy_attributes_from(&s2).unwrap();
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn stream_capture_info_reflects_state() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = Stream::new(&ctx).unwrap();

    // Before begin_capture: status should be inactive.
    let (active, _, _) = stream.capture_info().unwrap();
    assert!(!active);

    stream.begin_capture(CaptureMode::ThreadLocal).unwrap();
    let (active, _id, _g) = stream.capture_info().unwrap();
    assert!(active, "capture_info should report active during capture");
    let _graph = stream.end_capture().unwrap();

    let (active, _, _) = stream.capture_info().unwrap();
    assert!(!active);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn stream_attach_mem_async_on_managed() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = Stream::new(&ctx).unwrap();

    let buf = ManagedBuffer::<u32>::new(&ctx, 1024).unwrap();
    // CU_MEM_ATTACH_SINGLE = 4 — restrict the managed range to this stream.
    let raw: CUdeviceptr = buf.as_raw();
    stream.attach_mem_async(raw, 1024 * 4, 4).unwrap();
    stream.synchronize().unwrap();
}
