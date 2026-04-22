//! GPU-gated integration test for Wave-15 Driver-API additions:
//! batched pointer-attribute + range-attribute queries.

use baracuda_cuda_sys::types::CUpointer_attribute;
use baracuda_driver::pointer;
use baracuda_driver::{Context, Device, DeviceBuffer};

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn pointer_attributes_batched_returns_memory_type_and_ordinal() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();

    let buf: DeviceBuffer<f32> = DeviceBuffer::new(&ctx, 1024).unwrap();

    let mut mem_type: u32 = 0;
    let mut ordinal: i32 = -1;

    let mut attrs = [
        CUpointer_attribute::MEMORY_TYPE,
        CUpointer_attribute::DEVICE_ORDINAL,
    ];
    let mut data: [*mut core::ffi::c_void; 2] = [
        &mut mem_type as *mut u32 as *mut core::ffi::c_void,
        &mut ordinal as *mut i32 as *mut core::ffi::c_void,
    ];
    // SAFETY: both slots are the correct sizes (u32 / i32).
    unsafe {
        pointer::raw_attributes_batched(&mut attrs, &mut data, buf.as_raw()).unwrap();
    }

    // Device memory: type=2 (DEVICE), ordinal=0.
    assert_eq!(mem_type, baracuda_cuda_sys::types::CUmemorytype::DEVICE);
    assert_eq!(ordinal, 0);
}
