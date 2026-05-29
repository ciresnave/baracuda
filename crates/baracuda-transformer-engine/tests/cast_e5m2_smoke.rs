//! E5M2 round-trip smoke test.
//!
//! E5M2 trades 1 bit of mantissa for 1 bit of exponent vs E4M3:
//!   - mantissa: 2 bits (vs 3 for E4M3)
//!   - exponent: 5 bits (vs 4 for E4M3)
//!   - max finite: 57344.0 (vs 448.0 for E4M3)
//!
//! Wider dynamic range, lower precision. The roundtrip tolerance is
//! looser than E4M3's accordingly.

use baracuda_driver::{Context, Device, DeviceBuffer, Stream};
use baracuda_transformer_engine::{Fp8CastPlan, Fp8DequantPlan, Fp8Format, Fp8Recipe};

const HIST_LEN: usize = 16;
const N_ELEMS: usize = 4096;

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn e5m2_roundtrip_bf16_within_dynamic_range() {
    baracuda_driver::init().expect("driver init");
    let ctx = Context::new(&Device::get(0).expect("device")).expect("ctx");
    let stream = Stream::new(&ctx).expect("stream");

    // Wider input range than the E4M3 test — E5M2 handles it.
    let host_x: Vec<half::bf16> = (0..N_ELEMS)
        .map(|i| {
            let v = ((i as f32) * 0.011).cos() * 100.0;
            half::bf16::from_f32(v)
        })
        .collect();

    let x: DeviceBuffer<half::bf16> =
        DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut x_fp8: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, N_ELEMS).expect("alloc fp8");
    let mut y: DeviceBuffer<half::bf16> = DeviceBuffer::zeros(&ctx, N_ELEMS).expect("alloc y");

    let mut recipe =
        Fp8Recipe::new(&ctx, &stream, Fp8Format::E5M2, HIST_LEN).expect("recipe new");

    let cast_plan: Fp8CastPlan<half::bf16> = Fp8CastPlan::select().expect("cast plan");
    for _ in 0..HIST_LEN {
        cast_plan
            .run(&x, &mut x_fp8, &mut recipe, &stream)
            .expect("cast");
        recipe.update_after_pass(&stream).expect("update");
    }

    cast_plan
        .run(&x, &mut x_fp8, &mut recipe, &stream)
        .expect("cast final");

    let dequant_plan: Fp8DequantPlan<half::bf16> = Fp8DequantPlan::select().expect("dequant plan");
    dequant_plan
        .run(&x_fp8, &mut y, &recipe, &stream)
        .expect("dequant");

    stream.synchronize().expect("sync");

    let mut got = vec![half::bf16::ZERO; N_ELEMS];
    y.copy_to_host(&mut got).expect("D2H got");

    let scale = recipe.scale_host(&stream).expect("scale");
    assert!(scale.is_finite() && scale > 0.0);

    // E5M2 with 2 mantissa bits — 25% worst case at granular edges,
    // typical ~10-15%. Allow up to 5% of cells at the loose budget.
    let mut bad = 0usize;
    let mut worst = 0.0f32;
    for (a, b) in got.iter().zip(host_x.iter()) {
        let g = a.to_f32();
        let r = b.to_f32();
        let denom = r.abs().max(1e-4);
        let rel = (g - r).abs() / denom;
        worst = worst.max(rel);
        if rel > 0.25 {
            bad += 1;
        }
    }
    let bad_frac = (bad as f32) / (N_ELEMS as f32);
    assert!(
        bad_frac < 0.05,
        "E5M2 roundtrip: {}/{} cells exceeded 25% relative error (worst {:.4}); \
         scale={}, max_repr=57344.0",
        bad, N_ELEMS, worst, scale,
    );
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn e5m2_wider_dynamic_range_than_e4m3() {
    // Verify that E5M2 can roundtrip values that overflow E4M3:
    // ~1000.0 is fine for E5M2 (max 57344) but >448 for E4M3.
    baracuda_driver::init().expect("driver init");
    let ctx = Context::new(&Device::get(0).expect("device")).expect("ctx");
    let stream = Stream::new(&ctx).expect("stream");

    let host_x: Vec<f32> = vec![1000.0f32; 128];
    let x: DeviceBuffer<f32> = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload");
    let mut x_fp8: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, 128).expect("alloc fp8");
    let mut y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 128).expect("alloc y");

    let mut recipe = Fp8Recipe::new(&ctx, &stream, Fp8Format::E5M2, 16).expect("recipe");
    let cast_plan: Fp8CastPlan<f32> = Fp8CastPlan::select().expect("cast plan");
    // Run twice + update so the recipe stabilizes.
    for _ in 0..3 {
        cast_plan
            .run(&x, &mut x_fp8, &mut recipe, &stream)
            .expect("cast");
        recipe.update_after_pass(&stream).expect("update");
    }
    cast_plan
        .run(&x, &mut x_fp8, &mut recipe, &stream)
        .expect("cast");
    let dequant_plan: Fp8DequantPlan<f32> = Fp8DequantPlan::select().expect("dequant");
    dequant_plan
        .run(&x_fp8, &mut y, &recipe, &stream)
        .expect("dequant");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0f32; 128];
    y.copy_to_host(&mut got).expect("D2H");

    // The value 1000.0 should round-trip to within ~12% (E5M2 mantissa
    // granularity at this magnitude — 2^9 = 512 to 2^10 = 1024 bucket,
    // ulp = 2^(10-2) = 256).
    for v in &got {
        assert!(v.is_finite(), "non-finite {}", v);
        let rel = ((*v) - 1000.0).abs() / 1000.0;
        assert!(
            rel < 0.30,
            "E5M2 roundtrip of 1000.0 too far: got {} (rel err {})", v, rel,
        );
    }
}
