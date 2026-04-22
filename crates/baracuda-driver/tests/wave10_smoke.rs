//! GPU-gated integration tests for Wave-10 Driver-API additions:
//! 3-D arrays + `cuMemcpy3D` + mipmapped arrays.

use baracuda_driver::array::ArrayFormat;
use baracuda_driver::memcpy3d::{Array3D, MipmappedArray};
use baracuda_driver::{Context, Device};

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn array3d_host_roundtrip_f32() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();

    let w = 7usize;
    let h = 5usize;
    let d = 3usize;
    let src: Vec<f32> = (0..(w * h * d)).map(|i| (i as f32) * 0.125).collect();

    let arr = Array3D::new(&ctx, w, h, d, ArrayFormat::F32, 1).unwrap();
    arr.copy_from_host(&src).unwrap();

    let mut back = vec![0.0f32; w * h * d];
    arr.copy_to_host(&mut back).unwrap();
    ctx.synchronize().unwrap();

    assert_eq!(src, back);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn mipmapped_array_level_dimensions() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();

    // 64×64 base, 7 levels -> sizes 64, 32, 16, 8, 4, 2, 1.
    let mip = MipmappedArray::new(&ctx, 64, 64, 0, ArrayFormat::F32, 1, 7, 0).unwrap();
    assert_eq!(mip.num_levels(), 7);
    let lvl0 = mip.level(0).unwrap();
    let lvl3 = mip.level(3).unwrap();
    let lvl6 = mip.level(6).unwrap();
    assert_eq!(lvl0.width(), 64);
    assert_eq!(lvl3.width(), 8);
    assert_eq!(lvl6.width(), 1);
    drop(lvl0);
    drop(lvl3);
    drop(lvl6);
    drop(mip);
}
