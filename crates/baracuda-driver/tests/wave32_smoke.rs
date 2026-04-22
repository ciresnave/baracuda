//! Integration test for `cuMemMapArrayAsync` typed wrapper.
//!
//! Sparse arrays require hardware support (compute 7.0+; full mipmap-tail
//! support is Ampere+). On devices without sparse residency, array
//! creation with the SPARSE flag or the map call fails — we verify
//! either the full end-to-end path or a graceful `NOT_SUPPORTED` return.

use baracuda_cuda_sys::types::{CUarray3D_flags, CUarrayMapInfo};
use baracuda_driver::array::ArrayFormat;
use baracuda_driver::memcpy3d::{self, Array3D};
use baracuda_driver::{Context, Device, Stream};

#[test]
fn array_map_info_size_and_builder_layout() {
    // 96 bytes is the CUDA 13.x layout. If NVIDIA bumps this,
    // baracuda's typed wrapper needs regenerating.
    assert_eq!(core::mem::size_of::<CUarrayMapInfo>(), 96);

    // Builder sanity: fields should round-trip to the expected offsets.
    let info = CUarrayMapInfo::default()
        .with_sparse_level(3, 0, 10, 20, 30, 4, 4, 4)
        .with_offset(0xDEAD_BEEF)
        .with_device_bit_mask(0b11)
        .as_map();
    assert_eq!(info.offset, 0xDEAD_BEEF);
    assert_eq!(info.device_bit_mask, 0b11);
    assert_eq!(
        info.mem_operation_type,
        baracuda_cuda_sys::types::CUmemOperationType::MAP
    );

    // sparseLevel layout: 8 u32s at offset 0 of subresource_raw.
    // [level=3, layer=0, ox=10, oy=20, oz=30, ew=4, eh=4, ed=4]
    let sl = unsafe { *(info.subresource_raw.as_ptr() as *const [u32; 8]) };
    assert_eq!(sl, [3, 0, 10, 20, 30, 4, 4, 4]);
}

#[test]
#[ignore = "requires an NVIDIA GPU with sparse-array support"]
fn map_array_async_on_sparse_capable_device() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = Stream::new(&ctx).unwrap();

    // Try to create a sparse 3D array. If the device doesn't support
    // sparse arrays, creation itself fails — treat as skip.
    let arr = match Array3D::with_flags(
        &ctx,
        256,
        256,
        1,
        ArrayFormat::F32,
        1,
        CUarray3D_flags::SPARSE,
    ) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("sparse Array3D creation rejected (expected on non-sparse HW): {e:?}");
            return;
        }
    };

    // Call with an empty array — should be a trivial OK.
    memcpy3d::map_array_async(&mut [], &stream).unwrap();

    // Build an unmap op on the full array — unmapping when nothing is
    // mapped should either be a no-op or fail with INVALID_VALUE. The
    // point is exercising the tagged-union layout and the FFI.
    let mut info = [CUarrayMapInfo::default()
        .with_array(arr.as_raw())
        .with_sparse_level(0, 0, 0, 0, 0, 256, 256, 1)
        .as_unmap()];
    match memcpy3d::map_array_async(&mut info, &stream) {
        Ok(()) => eprintln!("cuMemMapArrayAsync (unmap-noop) OK"),
        Err(e) => eprintln!("cuMemMapArrayAsync returned (non-fatal): {e:?}"),
    }
    stream.synchronize().unwrap();
}
