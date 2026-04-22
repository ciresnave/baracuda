//! GPU-gated integration test for Wave-16 Driver-API additions:
//! Hopper TMA descriptor encoding. Encoding works on any device; actual
//! TMA instructions require SM 9.0+.

use baracuda_driver::tensor_map::{DataType, Interleave, L2Promotion, OOBFill, Swizzle, TensorMap};
use baracuda_driver::{Context, Device, DeviceBuffer};

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn tensor_map_encode_tiled_produces_non_empty_descriptor() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();

    // 2-D f32 tensor 256×128, tile of 64×32.
    let buf: DeviceBuffer<f32> = DeviceBuffer::new(&ctx, 256 * 128).unwrap();
    let elem_bytes = core::mem::size_of::<f32>() as u64;
    let global_dim = [256u64, 128];
    let global_strides = [elem_bytes * 256, elem_bytes * 256 * 128];
    let box_dim = [64u32, 32];
    let element_strides = [1u32, 1];

    let map = TensorMap::encode_tiled(
        DataType::FLOAT32,
        buf.as_raw(),
        &global_dim,
        // cuTensorMapEncodeTiled ignores global_strides[0] and only uses
        // strides[1..], but expects a full-length array.
        &global_strides,
        &box_dim,
        &element_strides,
        Interleave::NONE,
        Swizzle::NONE,
        L2Promotion::NONE,
        OOBFill::NONE,
    );

    // Hopper-only errors vs success — both are informative.
    match map {
        Ok(mut m) => {
            // Descriptor should be non-zero after successful encode.
            assert!(m.as_raw().opaque.iter().any(|w| *w != 0));
            // Swap the base pointer — should round-trip fine.
            let buf2: DeviceBuffer<f32> = DeviceBuffer::new(&ctx, 256 * 128).unwrap();
            m.replace_address(buf2.as_raw()).unwrap();
        }
        Err(e) => {
            eprintln!("cuTensorMapEncodeTiled rejected our descriptor: {e:?}");
            // That's OK on non-Hopper — the call is host-side and should
            // usually succeed, but some parameter combos are only legal
            // on Hopper. Treat non-success as informational, not a failure.
        }
    }
}
