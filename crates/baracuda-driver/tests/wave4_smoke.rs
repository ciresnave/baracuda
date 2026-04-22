//! GPU-gated integration test for Wave-4 Driver-API additions:
//! `cuMemAllocPitch` + `cuMemcpy2D` round-trip.

use baracuda_driver::{memcpy2d, Context, Device};

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn pitched_2d_roundtrip() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();

    // 8-row × 13-column f32 grid (deliberately non-multiple-of-32 width so
    // the driver picks a non-trivial pitch).
    let width_elems = 13usize;
    let height = 8usize;

    let host_src: Vec<f32> = (0..(width_elems * height))
        .map(|i| (i as f32) * 0.5)
        .collect();
    let host_src_pitch = width_elems * core::mem::size_of::<f32>();

    let d_buf = memcpy2d::PitchedBuffer::<f32>::new(&ctx, width_elems, height).unwrap();
    eprintln!(
        "pitched {}×{} f32: pitch = {} bytes (nominal {} bytes)",
        width_elems,
        height,
        d_buf.pitch_bytes(),
        host_src_pitch,
    );
    assert!(d_buf.pitch_bytes() >= host_src_pitch);

    memcpy2d::copy_h_to_d_2d(&host_src, host_src_pitch, &d_buf, width_elems, height).unwrap();

    let mut host_back = vec![0.0f32; width_elems * height];
    memcpy2d::copy_d_to_h_2d(&d_buf, &mut host_back, host_src_pitch, width_elems, height).unwrap();
    ctx.synchronize().unwrap();

    assert_eq!(host_src, host_back);
}
