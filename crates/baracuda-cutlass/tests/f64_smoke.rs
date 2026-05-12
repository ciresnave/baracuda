//! Real-GPU smoke tests for f64 (DGEMM via Ampere FP64 tensor cores).
//!
//! Coverage: `f64 × {Rcr, Rrr} × {Identity, Bias, BiasRelu, BiasGelu,
//! BiasSilu}` = 10 SKUs. f64 GEMM uses the FP64 tensor cores (mma.sync
//! m8n8k4 in double) and accumulates into f64, so we can hold callers
//! to a much tighter tolerance than f32. `#[ignore]` by default.
//!
//! Also exercises the per-element scalar refactor: `GemmArgs::<f64>` has
//! `alpha: f64, beta: f64` (rather than the f32 used by every other
//! shipped dtype).

use baracuda_cutlass::{
    ActivationKind, ElementKind, EpilogueKind, GemmArgs, GemmDescriptor, GemmPlan, LayoutSku,
    MathPrecision, MatrixMut, MatrixRef, PlanPreference, VectorRef, Workspace,
};
use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};

fn relu(x: f64) -> f64 { x.max(0.0) }
fn erf_approx(x: f64) -> f64 {
    // A&S 7.1.26 — max |ε| ≈ 1.5e-7 over the full range.
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let p = 0.3275911_f64;
    let a1 = 0.254829592_f64;
    let a2 = -0.284496736_f64;
    let a3 = 1.421413741_f64;
    let a4 = -1.453152027_f64;
    let a5 = 1.061405429_f64;
    let t = 1.0 / (1.0 + p * ax);
    sign * (1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-ax * ax).exp())
}

fn gelu_exact(x: f64) -> f64 {
    // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2))). Matches CUTLASS's
    // GELU<double> (which calls libm `erf` on `x * sqrt(2)/2`).
    0.5 * x * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
}
fn silu(x: f64) -> f64 { x / (1.0 + (-x).exp()) }

#[allow(clippy::too_many_arguments)]
fn cpu_bias_act_gemm(
    layout: LayoutSku,
    m: usize, n: usize, k: usize,
    a: &[f64], lda: usize,
    b: &[f64], ldb: usize,
    bias: Option<&[f64]>,
    alpha: f64,
    activation: Option<ActivationKind>,
    d: &mut [f64], ldd: usize,
) {
    let act_fn: fn(f64) -> f64 = match activation {
        Some(ActivationKind::Relu) => relu,
        Some(ActivationKind::Gelu) => gelu_exact,
        Some(ActivationKind::Silu) => silu,
        None => |x| x,
    };
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for kk in 0..k {
                let b_val = match layout {
                    LayoutSku::Rcr => b[j * ldb + kk],
                    LayoutSku::Rrr => b[kk * ldb + j],
                };
                acc += a[i * lda + kk] * b_val;
            }
            let val = alpha * acc + bias.map(|v| v[j]).unwrap_or(0.0);
            d[i * ldd + j] = act_fn(val);
        }
    }
}

fn run_f64(layout: LayoutSku, epilogue: EpilogueKind, m: i32, n: i32, k: i32) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let host_a: Vec<f64> = (0..(m * k)).map(|i| ((i as f64) * 0.01).sin()).collect();
    let host_b: Vec<f64> = (0..(k * n)).map(|i| ((i as f64) * 0.013).cos()).collect();
    let host_bias: Vec<f64> = (0..n).map(|j| -0.5 + 0.1 * (j as f64 % 7.0)).collect();

    let ldb = match layout {
        LayoutSku::Rcr => k as usize,
        LayoutSku::Rrr => n as usize,
    };

    let mut expected = vec![0.0f64; (m * n) as usize];
    cpu_bias_act_gemm(
        layout,
        m as usize, n as usize, k as usize,
        &host_a, k as usize,
        &host_b, ldb,
        epilogue.requires_bias().then_some(host_bias.as_slice()),
        1.0,
        epilogue.activation(),
        &mut expected, n as usize,
    );

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload A");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload B");
    let dev_bias = DeviceBuffer::from_slice(&ctx, &host_bias).expect("upload bias");
    let mut dev_d: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (m * n) as usize).expect("alloc D");

    let desc = GemmDescriptor { m, n, k, layout, epilogue };
    let plan = GemmPlan::<f64>::select(&stream, &desc, PlanPreference::default())
        .expect("plan select");

    assert_eq!(plan.precision_guarantee().math_precision, MathPrecision::F64);
    assert_eq!(plan.precision_guarantee().accumulator, ElementKind::F64);
    assert_eq!(plan.sku().element, ElementKind::F64);
    assert_eq!(plan.sku().layout, layout);

    let bias_arg = epilogue
        .requires_bias()
        .then(|| VectorRef { data: dev_bias.as_slice(), len: n, stride: 1 });

    let args = GemmArgs::<f64> {
        a: MatrixRef { data: dev_a.as_slice(), rows: m, cols: k, ld: k as i64 },
        b: MatrixRef { data: dev_b.as_slice(), rows: k, cols: n, ld: ldb as i64 },
        c: None,
        d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n as i64 },
        bias: bias_arg,
        // f64 alpha/beta — this is the per-element scalar refactor in
        // action. f16/bf16/f32/F32Strict callers pass f32 literals here.
        alpha: 1.0_f64, beta: 0.0_f64,
    };
    plan.can_implement(&args).expect("can_implement");
    plan.run(&stream, Workspace::None, args).expect("run");

    let mut out = vec![0.0f64; (m * n) as usize];
    dev_d.copy_to_host(&mut out).expect("download D");

    // F64 GEMM accumulates into f64; tolerance is much tighter than f32.
    // Activation reference uses an A&S erf approximation, so the
    // GELU comparison floor is ~1e-7 abs rather than ULP-level.
    let mut max_abs = 0.0f64;
    for (got, want) in out.iter().zip(expected.iter()) {
        let diff = (got - want).abs();
        if diff > max_abs { max_abs = diff; }
    }
    let tol = match epilogue {
        EpilogueKind::BiasGelu => 1e-6,                   // erf approx floor
        _ => (k as f64) * 1e-12,                          // f64 accumulation ULP
    };
    assert!(
        max_abs < tol,
        "f64 {:?} {:?} GEMM ({m}x{n}x{k}): max abs err {max_abs} > tol {tol}",
        layout, epilogue
    );
    println!(
        "f64 {:?} sm80 {:?} GEMM ({m}x{n}x{k}): max abs err {max_abs} (tol {tol}) ✅",
        layout, epilogue
    );
}

// ---- RCR ----

#[test] #[ignore] fn f64_identity_rcr_64_64_32()   { run_f64(LayoutSku::Rcr, EpilogueKind::Identity,  64, 64, 32); }
#[test] #[ignore] fn f64_identity_rcr_128_128_64() { run_f64(LayoutSku::Rcr, EpilogueKind::Identity, 128, 128, 64); }
#[test] #[ignore] fn f64_bias_rcr_64_64_32()       { run_f64(LayoutSku::Rcr, EpilogueKind::Bias,      64, 64, 32); }
#[test] #[ignore] fn f64_bias_relu_rcr_64_64_32()  { run_f64(LayoutSku::Rcr, EpilogueKind::BiasRelu,  64, 64, 32); }
#[test] #[ignore] fn f64_bias_gelu_rcr_64_64_32()  { run_f64(LayoutSku::Rcr, EpilogueKind::BiasGelu,  64, 64, 32); }
#[test] #[ignore] fn f64_bias_silu_rcr_64_64_32()  { run_f64(LayoutSku::Rcr, EpilogueKind::BiasSilu,  64, 64, 32); }

// ---- RRR ----

#[test] #[ignore] fn f64_identity_rrr_64_64_32()   { run_f64(LayoutSku::Rrr, EpilogueKind::Identity,  64, 64, 32); }
#[test] #[ignore] fn f64_identity_rrr_128_128_64() { run_f64(LayoutSku::Rrr, EpilogueKind::Identity, 128, 128, 64); }
#[test] #[ignore] fn f64_bias_rrr_64_64_32()       { run_f64(LayoutSku::Rrr, EpilogueKind::Bias,      64, 64, 32); }
#[test] #[ignore] fn f64_bias_relu_rrr_64_64_32()  { run_f64(LayoutSku::Rrr, EpilogueKind::BiasRelu,  64, 64, 32); }
#[test] #[ignore] fn f64_bias_gelu_rrr_64_64_32()  { run_f64(LayoutSku::Rrr, EpilogueKind::BiasGelu,  64, 64, 32); }
#[test] #[ignore] fn f64_bias_silu_rrr_64_64_32()  { run_f64(LayoutSku::Rrr, EpilogueKind::BiasSilu,  64, 64, 32); }
