//! E4M3 round-trip smoke test.
//!
//! Verifies that values within E4M3's representable range survive a
//! cast → dequant round trip with acceptable precision loss.
//!
//! E4M3 has 4-bit exponent + 3-bit mantissa, finite max 448.0.
//! With the recipe stabilized to `scale = 448 / max_amax`, values
//! in `[-max_amax, max_amax]` map into E4M3's representable range
//! without saturation, and the round trip recovers them to within
//! ~3 bits of mantissa precision (relative ~6e-2 in the worst case,
//! tighter for values not at the granular edges).

use baracuda_driver::{Context, Device, DeviceBuffer, Stream};
use baracuda_transformer_engine::{Fp8CastPlan, Fp8DequantPlan, Fp8Format, Fp8Recipe};

const HIST_LEN: usize = 16;
const N_ELEMS: usize = 4096;

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn e4m3_roundtrip_f16_within_dynamic_range() {
    baracuda_driver::init().expect("driver init");
    let ctx = Context::new(&Device::get(0).expect("device")).expect("ctx");
    let stream = Stream::new(&ctx).expect("stream");

    // Build a deterministic input — sin/cos pattern in [-1.0, 1.0].
    // Well within E4M3's [-448, 448] range so we don't need to wait
    // for the recipe to stabilize.
    let host_x: Vec<half::f16> = (0..N_ELEMS)
        .map(|i| {
            let v = ((i as f32) * 0.013).sin() * 0.95;
            half::f16::from_f32(v)
        })
        .collect();

    let x: DeviceBuffer<half::f16> =
        DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut x_fp8: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, N_ELEMS).expect("alloc fp8");
    let mut y: DeviceBuffer<half::f16> = DeviceBuffer::zeros(&ctx, N_ELEMS).expect("alloc y");

    let mut recipe =
        Fp8Recipe::new(&ctx, &stream, Fp8Format::E4M3, HIST_LEN).expect("recipe new");

    // Stabilize the recipe — run the cast a few times so the amax
    // history populates, then update so `scale` reflects the actual
    // input range.
    let cast_plan: Fp8CastPlan<half::f16> = Fp8CastPlan::select().expect("cast plan");
    for _ in 0..HIST_LEN {
        cast_plan
            .run(&x, &mut x_fp8, &mut recipe, &stream)
            .expect("cast");
        recipe.update_after_pass(&stream).expect("update");
    }

    // Final pass at the stabilized scale.
    cast_plan
        .run(&x, &mut x_fp8, &mut recipe, &stream)
        .expect("cast final");

    let dequant_plan: Fp8DequantPlan<half::f16> = Fp8DequantPlan::select().expect("dequant plan");
    dequant_plan
        .run(&x_fp8, &mut y, &recipe, &stream)
        .expect("dequant");

    stream.synchronize().expect("sync");

    // Pull back to host and measure.
    let mut got = vec![half::f16::ZERO; N_ELEMS];
    y.copy_to_host(&mut got).expect("D2H got");

    let scale = recipe.scale_host(&stream).expect("scale");
    assert!(
        scale.is_finite() && scale > 0.0,
        "scale must be finite and positive, got {}", scale,
    );

    // Per-element relative error budget for E4M3 at well-conditioned
    // inputs: 3 bits of mantissa = ~12.5% worst case, but typical
    // is far better (most values aren't at granular edges). We use
    // 8% as the cell-level cap with up to 2% of cells allowed at
    // the loose budget (the granular-edge ones).
    let mut bad = 0usize;
    let mut worst = 0.0f32;
    for (a, b) in got.iter().zip(host_x.iter()) {
        let g = a.to_f32();
        let r = b.to_f32();
        let denom = r.abs().max(1e-6);
        let rel = (g - r).abs() / denom;
        worst = worst.max(rel);
        if rel > 0.08 {
            bad += 1;
        }
    }
    let bad_frac = (bad as f32) / (N_ELEMS as f32);
    assert!(
        bad_frac < 0.02,
        "E4M3 roundtrip: {}/{} cells exceeded 8% relative error (worst {:.4}); \
         scale={}, max_repr=448.0",
        bad, N_ELEMS, worst, scale,
    );
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn e4m3_saturates_on_overflow() {
    // E4M3's max finite value is 448.0. Inputs larger than that, with
    // the initial identity scale (1.0), should clamp to 448.0 on the
    // cast and dequant back to 448.0 (no infinities).
    baracuda_driver::init().expect("driver init");
    let ctx = Context::new(&Device::get(0).expect("device")).expect("ctx");
    let stream = Stream::new(&ctx).expect("stream");

    let host_x: Vec<f32> = vec![1e6_f32; 128];
    let x: DeviceBuffer<f32> = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload");
    let mut x_fp8: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, 128).expect("alloc fp8");
    let mut y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 128).expect("alloc y");

    let mut recipe = Fp8Recipe::new(&ctx, &stream, Fp8Format::E4M3, 16).expect("recipe");
    // Single cast at the identity scale (no update yet).
    let cast_plan: Fp8CastPlan<f32> = Fp8CastPlan::select().expect("cast plan");
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

    for v in &got {
        assert!(
            v.is_finite(),
            "E4M3 saturating cast produced non-finite value {}", v,
        );
        assert!(
            v.abs() <= 448.0 + 1e-3,
            "E4M3 saturating cast did not clamp: got {} (expected <= 448.0)", v,
        );
    }
}
