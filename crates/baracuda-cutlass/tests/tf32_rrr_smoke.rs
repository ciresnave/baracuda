//! Real-GPU smoke tests for TF32 RRR coverage (Phase 1.1).
//!
//! Coverage: f32 (TF32 path) × Rrr × {Identity, Bias, BiasRelu, BiasGelu,
//! BiasSilu}. 5 SKUs, one round-trip each, verified against a CPU reference
//! that applies the same activation in full f32. TF32 reduces inputs to
//! ~10-bit mantissa for the multiply-add, so the tolerance is on relative
//! error rather than absolute. `#[ignore]` by default — runs on real
//! hardware only.

use baracuda_cutlass::{
    ActivationKind, ElementKind, EpilogueKind, GemmArgs, GemmDescriptor, GemmPlan, LayoutSku,
    MathPrecision, MatrixMut, MatrixRef, PlanPreference, VectorRef, Workspace,
};
use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};

// ---- CPU references (f32 throughout) ----

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn erf_approx(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let p = 0.3275911_f32;
    let a1 = 0.254829592_f32;
    let a2 = -0.284496736_f32;
    let a3 = 1.421413741_f32;
    let a4 = -1.453152027_f32;
    let a5 = 1.061405429_f32;
    let t = 1.0 / (1.0 + p * ax);
    sign * (1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-ax * ax).exp())
}

fn gelu_exact(x: f32) -> f32 {
    0.5 * x * (1.0 + erf_approx(x / std::f32::consts::SQRT_2))
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[allow(clippy::too_many_arguments)]
fn cpu_bias_act_gemm_rrr(
    m: usize, n: usize, k: usize,
    a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    bias: Option<&[f32]>,
    alpha: f32,
    activation: Option<ActivationKind>,
    d: &mut [f32], ldd: usize,
) {
    let act_fn: fn(f32) -> f32 = match activation {
        Some(ActivationKind::Relu) => relu,
        Some(ActivationKind::Gelu) => gelu_exact,
        Some(ActivationKind::Silu) => silu,
        None => |x| x,
    };
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[i * lda + kk] * b[kk * ldb + j]; // row-major B
            }
            let val = alpha * acc + bias.map(|v| v[j]).unwrap_or(0.0);
            d[i * ldd + j] = act_fn(val);
        }
    }
}

fn run_tf32_rrr(m: i32, n: i32, k: i32, epilogue: EpilogueKind) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let host_a: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 0.01).sin()).collect();
    let host_b: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 0.013).cos()).collect();
    let host_bias: Vec<f32> = (0..n)
        .map(|j| -0.5 + 0.1 * (j as f32 % 7.0))
        .collect();

    let mut expected = vec![0.0f32; (m * n) as usize];
    cpu_bias_act_gemm_rrr(
        m as usize, n as usize, k as usize,
        &host_a, k as usize,
        &host_b, n as usize, // RRR: B's leading dim is N
        epilogue.requires_bias().then_some(host_bias.as_slice()),
        1.0,
        epilogue.activation(),
        &mut expected, n as usize,
    );

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload A");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload B");
    let dev_bias = DeviceBuffer::from_slice(&ctx, &host_bias).expect("upload bias");
    let mut dev_d: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (m * n) as usize).expect("alloc D");

    let desc = GemmDescriptor {
        m, n, k,
        layout: LayoutSku::Rrr,
        epilogue,
    };
    let plan =
        GemmPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("plan select");

    assert_eq!(plan.precision_guarantee().math_precision, MathPrecision::Tf32);
    assert_eq!(plan.precision_guarantee().accumulator, ElementKind::F32);
    assert_eq!(plan.sku().layout, LayoutSku::Rrr);
    assert_eq!(plan.sku().epilogue, epilogue);

    let bias_arg = epilogue
        .requires_bias()
        .then(|| VectorRef { data: dev_bias.as_slice(), len: n, stride: 1 });

    let args = GemmArgs::<f32> {
        a: MatrixRef { data: dev_a.as_slice(), rows: m, cols: k, ld: k as i64 },
        // RRR: B is row-major [K, N], so leading dim is N (not K as in RCR).
        b: MatrixRef { data: dev_b.as_slice(), rows: k, cols: n, ld: n as i64 },
        c: None,
        d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n as i64 },
        bias: bias_arg,
        alpha: 1.0, beta: 0.0,
    };
    plan.can_implement(&args).expect("can_implement");
    plan.run(&stream, Workspace::None, args).expect("run");

    let mut out = vec![0.0f32; (m * n) as usize];
    dev_d.copy_to_host(&mut out).expect("download D");

    let mut max_rel = 0.0f32;
    for (got, want) in out.iter().zip(expected.iter()) {
        let denom = want.abs().max(1.0);
        let rel = (got - want).abs() / denom;
        if rel > max_rel {
            max_rel = rel;
        }
    }
    let tol = (k as f32) * 5e-4;
    assert!(
        max_rel < tol,
        "TF32 RRR {:?} GEMM ({m}x{n}x{k}): max rel err {max_rel} > tol {tol}",
        epilogue
    );
    println!(
        "TF32 RRR sm80 {:?} GEMM ({m}x{n}x{k}): max rel err {max_rel} (tol {tol}) ✅",
        epilogue
    );
}

#[test]
#[ignore]
fn tf32_identity_rrr_sm80_64_64_32() {
    run_tf32_rrr(64, 64, 32, EpilogueKind::Identity);
}

#[test]
#[ignore]
fn tf32_identity_rrr_sm80_128_128_64() {
    run_tf32_rrr(128, 128, 64, EpilogueKind::Identity);
}

#[test]
#[ignore]
fn tf32_bias_rrr_sm80_64_64_32() {
    run_tf32_rrr(64, 64, 32, EpilogueKind::Bias);
}

#[test]
#[ignore]
fn tf32_bias_relu_rrr_sm80_64_64_32() {
    run_tf32_rrr(64, 64, 32, EpilogueKind::BiasRelu);
}

#[test]
#[ignore]
fn tf32_bias_gelu_rrr_sm80_64_64_32() {
    run_tf32_rrr(64, 64, 32, EpilogueKind::BiasGelu);
}

#[test]
#[ignore]
fn tf32_bias_silu_rrr_sm80_64_64_32() {
    run_tf32_rrr(64, 64, 32, EpilogueKind::BiasSilu);
}
