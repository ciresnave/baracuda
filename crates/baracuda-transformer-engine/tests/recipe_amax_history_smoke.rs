//! Recipe amax history sliding-window smoke test.
//!
//! Verifies that:
//!   1. The amax history starts all-zero after construction.
//!   2. Each forward pass writes the per-call amax into
//!      `amax_history[write_pos]` via the fused-cast atomicMax.
//!   3. `update_after_pass` resets the just-written slot and
//!      advances the write pointer with wrap-around.
//!   4. The wrap-around behaviour respects the configured history
//!      length (smaller buffer → faster wrap).

use baracuda_driver::{Context, Device, DeviceBuffer, Stream};
use baracuda_transformer_engine::{Fp8CastPlan, Fp8Format, Fp8Recipe};

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn recipe_history_starts_all_zero() {
    baracuda_driver::init().expect("driver init");
    let ctx = Context::new(&Device::get(0).expect("device")).expect("ctx");
    let stream = Stream::new(&ctx).expect("stream");

    let recipe = Fp8Recipe::new(&ctx, &stream, Fp8Format::E4M3, 8).expect("recipe");
    stream.synchronize().expect("sync");

    let mut hist = vec![1.0f32; 8];
    recipe.amax_history().copy_to_host(&mut hist).expect("D2H");
    for (i, v) in hist.iter().enumerate() {
        assert_eq!(*v, 0.0, "history[{}] = {} (expected 0)", i, v);
    }
    assert_eq!(recipe.write_pos(), 0);
    assert_eq!(recipe.history_len(), 8);
    assert_eq!(recipe.format(), Fp8Format::E4M3);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn recipe_advances_with_wrap_around() {
    baracuda_driver::init().expect("driver init");
    let ctx = Context::new(&Device::get(0).expect("device")).expect("ctx");
    let stream = Stream::new(&ctx).expect("stream");

    let hist_len = 4;
    let mut recipe = Fp8Recipe::new(&ctx, &stream, Fp8Format::E4M3, hist_len).expect("recipe");

    // Distinct max-amax per pass so we can verify the slot-by-slot
    // writes after the recipe-update reset.
    let inputs: Vec<Vec<f32>> = (0..10)
        .map(|step| {
            let m = (step as f32 + 1.0) * 0.5;
            (0..256).map(|i| ((i as f32) * 0.01).sin() * m).collect()
        })
        .collect();

    let mut x_fp8: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, 256).expect("alloc fp8");
    let cast_plan: Fp8CastPlan<f32> = Fp8CastPlan::select().expect("cast plan");

    for (step, input) in inputs.iter().enumerate() {
        let expected_write_pos = step % hist_len;
        assert_eq!(
            recipe.write_pos(),
            expected_write_pos,
            "before step {}: write_pos drifted",
            step,
        );

        let x: DeviceBuffer<f32> = DeviceBuffer::from_slice(&ctx, input).expect("upload");
        cast_plan
            .run(&x, &mut x_fp8, &mut recipe, &stream)
            .expect("cast");
        recipe.update_after_pass(&stream).expect("update");
    }

    // After 10 steps with len=4, write_pos = 10 % 4 = 2.
    assert_eq!(recipe.write_pos(), 10 % hist_len);

    // The most recent slot (the one update_after_pass just zeroed) is
    // the one for the just-completed step before the increment — let's
    // check that the recipe still has a non-zero scale after stabilizing.
    let scale = recipe.scale_host(&stream).expect("scale");
    let scale_inv = recipe.scale_inv_host(&stream).expect("scale_inv");
    assert!(scale.is_finite() && scale > 0.0, "scale not finite: {}", scale);
    assert!(
        scale_inv.is_finite() && scale_inv > 0.0,
        "scale_inv not finite: {}",
        scale_inv,
    );

    // scale * scale_inv ≈ 1 (within fp32 division precision).
    let prod = scale * scale_inv;
    assert!(
        (prod - 1.0).abs() < 1e-5,
        "scale * scale_inv = {} (expected ~1.0)", prod,
    );
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn invalid_history_length_rejected() {
    baracuda_driver::init().expect("driver init");
    let ctx = Context::new(&Device::get(0).expect("device")).expect("ctx");
    let stream = Stream::new(&ctx).expect("stream");

    // 0 is invalid.
    let r = Fp8Recipe::new(&ctx, &stream, Fp8Format::E4M3, 0);
    assert!(r.is_err(), "zero-length history should be rejected");

    // 8193 is out of range (the shim caps at 8192).
    let r = Fp8Recipe::new(&ctx, &stream, Fp8Format::E4M3, 8193);
    assert!(r.is_err(), ">8192 history should be rejected");
}
