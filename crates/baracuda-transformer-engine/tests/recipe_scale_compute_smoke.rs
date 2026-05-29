//! Recipe `scale = max_representable / max_amax` formula correctness.
//!
//! Drives the recipe with a known max-amax input, runs update, and
//! verifies the published scale matches the analytic formula to f32
//! precision.

use baracuda_driver::{Context, Device, DeviceBuffer, Stream};
use baracuda_transformer_engine::{Fp8CastPlan, Fp8Format, Fp8Recipe};

const N: usize = 1024;

fn run_one(format: Fp8Format, max_input_abs: f32) -> (f32, f32, f32) {
    baracuda_driver::init().expect("driver init");
    let ctx = Context::new(&Device::get(0).expect("device")).expect("ctx");
    let stream = Stream::new(&ctx).expect("stream");

    // Build an input whose max(|x|) is exactly max_input_abs.
    // We put the spike in the middle to make sure the block-reduce
    // path picks it up.
    let mut host = vec![0.0f32; N];
    for (i, v) in host.iter_mut().enumerate() {
        // Small noise + the spike at index 512.
        *v = ((i as f32) * 0.01).sin() * (max_input_abs * 0.1);
    }
    host[512] = max_input_abs;
    host[768] = -max_input_abs; // signed amax test — abs should win.

    let x: DeviceBuffer<f32> = DeviceBuffer::from_slice(&ctx, &host).expect("upload");
    let mut x_fp8: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, N).expect("alloc fp8");
    let mut recipe = Fp8Recipe::new(&ctx, &stream, format, 8).expect("recipe");

    let plan: Fp8CastPlan<f32> = Fp8CastPlan::select().expect("plan");
    plan.run(&x, &mut x_fp8, &mut recipe, &stream).expect("cast");
    recipe.update_after_pass(&stream).expect("update");

    let scale = recipe.scale_host(&stream).expect("scale");
    let scale_inv = recipe.scale_inv_host(&stream).expect("scale_inv");

    let max_repr = format.max_representable();
    let expected_scale = max_repr / max_input_abs;
    (scale, scale_inv, expected_scale)
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn scale_formula_e4m3() {
    let (scale, scale_inv, expected) = run_one(Fp8Format::E4M3, 10.0);
    // E4M3 max_repr = 448, max_amax = 10, expected scale = 44.8.
    let rel = (scale - expected).abs() / expected;
    assert!(
        rel < 1e-5,
        "E4M3 scale = {} (expected {} = 448/10 = 44.8); rel err = {}",
        scale, expected, rel,
    );
    let prod = scale * scale_inv;
    assert!(
        (prod - 1.0).abs() < 1e-5,
        "scale * scale_inv = {} (expected ~1)", prod,
    );
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn scale_formula_e5m2() {
    let (scale, scale_inv, expected) = run_one(Fp8Format::E5M2, 100.0);
    // E5M2 max_repr = 57344, max_amax = 100, expected scale = 573.44.
    let rel = (scale - expected).abs() / expected;
    assert!(
        rel < 1e-5,
        "E5M2 scale = {} (expected {} = 57344/100 = 573.44); rel err = {}",
        scale, expected, rel,
    );
    let prod = scale * scale_inv;
    assert!((prod - 1.0).abs() < 1e-5);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn scale_floors_at_one_when_history_all_zero() {
    // Recipe just created (history all zero). `update_after_pass` with
    // no prior FW pass should floor the amax at 1.0 → scale = max_repr.
    baracuda_driver::init().expect("driver init");
    let ctx = Context::new(&Device::get(0).expect("device")).expect("ctx");
    let stream = Stream::new(&ctx).expect("stream");

    let mut recipe = Fp8Recipe::new(&ctx, &stream, Fp8Format::E4M3, 8).expect("recipe");
    // Update without ever having a cast — amax history is all zero.
    recipe.update_after_pass(&stream).expect("update");
    let scale = recipe.scale_host(&stream).expect("scale");

    // The shim floors amax at 1.0 in this case, so scale = 448 / 1 = 448.
    assert!(
        (scale - 448.0).abs() < 1e-3,
        "Expected scale to floor at 448 (E4M3 max_repr / 1.0); got {}",
        scale,
    );
}
