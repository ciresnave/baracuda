//! GPU-gated integration tests for Wave-6 Driver-API additions:
//! `cuArrayCreate_v2` + 2-D memcpy to/from CUarray,
//! `cuTexObjectCreate/Destroy`, `cuSurfObjectCreate/Destroy`.

use baracuda_driver::{
    Array, ArrayFormat, Context, Device, SurfaceObject, TextureDesc, TextureFilterMode,
    TextureObject,
};

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn array_host_roundtrip_f32x1() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();

    let width = 17usize;
    let height = 11usize;
    let src: Vec<f32> = (0..(width * height)).map(|i| i as f32 * 0.25).collect();

    let arr = Array::new(&ctx, width, height, ArrayFormat::F32, 1).unwrap();
    arr.copy_from_host(&src).unwrap();

    let mut back = vec![0.0f32; width * height];
    arr.copy_to_host(&mut back).unwrap();
    ctx.synchronize().unwrap();

    assert_eq!(src, back);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn array_host_roundtrip_u8x4() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();

    let width = 8usize;
    let height = 6usize;
    // 4 channels per texel -> 4 bytes per texel at U8.
    #[repr(C)]
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    struct Rgba(u8, u8, u8, u8);
    // SAFETY: Rgba is #[repr(C)] with 4 u8s, no padding.
    unsafe impl baracuda_types::DeviceRepr for Rgba {}

    let src: Vec<Rgba> = (0..(width * height))
        .map(|i| Rgba(i as u8, (i + 1) as u8, (i + 2) as u8, (i + 3) as u8))
        .collect();

    let arr = Array::new(&ctx, width, height, ArrayFormat::U8, 4).unwrap();
    arr.copy_from_host(&src).unwrap();

    let mut back = vec![Rgba(0, 0, 0, 0); width * height];
    arr.copy_to_host(&mut back).unwrap();
    ctx.synchronize().unwrap();

    assert_eq!(src, back);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn tex_and_surf_object_lifecycle() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();

    let arr = Array::new(&ctx, 16, 16, ArrayFormat::F32, 1).unwrap();

    // Texture object with linear filtering + wrap addressing.
    let tex = TextureObject::with_desc(
        &arr,
        TextureDesc {
            address_mode: [
                baracuda_driver::TextureAddressMode::Wrap,
                baracuda_driver::TextureAddressMode::Clamp,
                baracuda_driver::TextureAddressMode::Clamp,
            ],
            filter_mode: TextureFilterMode::Linear,
            read_normalized: true,
            normalized_coords: false,
        },
    )
    .unwrap();
    assert_ne!(tex.as_raw(), 0, "tex object handle should be non-zero");

    // Surface object on the same array.
    let surf = SurfaceObject::new(&arr).unwrap();
    assert_ne!(surf.as_raw(), 0, "surf object handle should be non-zero");

    // Drop order: drop surf and tex first so Array outlives both (it's
    // cloned internally, so this is purely a readability assertion — it
    // works either way).
    drop(tex);
    drop(surf);
    drop(arr);
    drop(ctx);
}
