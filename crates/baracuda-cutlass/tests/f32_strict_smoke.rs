//! Real-GPU smoke tests for F32Strict (SIMT, full IEEE 754 binary32).
//!
//! Coverage: `F32Strict × {Rcr, Rrr} × {Identity, Bias, BiasRelu,
//! BiasGelu, BiasSilu}` = 10 SKUs. SIMT GEMM uses CUDA cores (no tensor
//! cores) and produces full-precision results, so the tolerance here is
//! much tighter than for the TF32 path: ~K * 1e-5 absolute relative
//! error vs the CPU reference. `#[ignore]` by default.

use baracuda_cutlass::{
    ActivationKind, ElementKind, EpilogueKind, F32Strict, GemmArgs, GemmDescriptor, GemmPlan,
    LayoutSku, MathPrecision, MatrixMut, MatrixRef, PlanPreference, VectorRef, Workspace,
};
use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};

// ---- CPU references ----

fn relu(x: f32) -> f32 { x.max(0.0) }
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
fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

#[allow(clippy::too_many_arguments)]
fn cpu_bias_act_gemm(
    layout: LayoutSku,
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
                let b_val = match layout {
                    // Rcr: B column-major, leading dim K, indexed [j*K + kk]
                    LayoutSku::Rcr => b[j * ldb + kk],
                    // Rrr: B row-major, leading dim N, indexed [kk*N + j]
                    LayoutSku::Rrr => b[kk * ldb + j],
                };
                acc += a[i * lda + kk] * b_val;
            }
            let val = alpha * acc + bias.map(|v| v[j]).unwrap_or(0.0);
            d[i * ldd + j] = act_fn(val);
        }
    }
}

fn run_f32_strict(layout: LayoutSku, epilogue: EpilogueKind, m: i32, n: i32, k: i32) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let host_a: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 0.01).sin()).collect();
    let host_b: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 0.013).cos()).collect();
    let host_bias: Vec<f32> = (0..n).map(|j| -0.5 + 0.1 * (j as f32 % 7.0)).collect();

    // RCR: B leading dim is K (column-major); RRR: B leading dim is N (row-major).
    let ldb = match layout {
        LayoutSku::Rcr => k as usize,
        LayoutSku::Rrr => n as usize,
    };

    let mut expected = vec![0.0f32; (m * n) as usize];
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

    // Allocate directly as F32Strict — same memory layout as f32 (the
    // wrapper is `#[repr(transparent)]`), but the typed `DeviceBuffer`
    // drives the plan to pick the SIMT kernels rather than the TF32 ones.
    let host_a_strict: Vec<F32Strict> = host_a.iter().copied().map(F32Strict).collect();
    let host_b_strict: Vec<F32Strict> = host_b.iter().copied().map(F32Strict).collect();
    let host_bias_strict: Vec<F32Strict> = host_bias.iter().copied().map(F32Strict).collect();

    let dev_a_buf = DeviceBuffer::from_slice(&ctx, &host_a_strict).expect("upload A");
    let dev_b_buf = DeviceBuffer::from_slice(&ctx, &host_b_strict).expect("upload B");
    let dev_bias_buf = DeviceBuffer::from_slice(&ctx, &host_bias_strict).expect("upload bias");
    let mut dev_d_buf: DeviceBuffer<F32Strict> =
        DeviceBuffer::zeros(&ctx, (m * n) as usize).expect("alloc D");

    let desc = GemmDescriptor { m, n, k, layout, epilogue };
    let plan = GemmPlan::<F32Strict>::select(&stream, &desc, PlanPreference::default())
        .expect("plan select");

    assert_eq!(plan.precision_guarantee().math_precision, MathPrecision::F32);
    assert_eq!(plan.precision_guarantee().accumulator, ElementKind::F32);
    assert!(plan.precision_guarantee().bit_stable_on_same_hardware);
    assert_eq!(plan.sku().element, ElementKind::F32Strict);
    assert_eq!(plan.sku().layout, layout);

    let bias_arg = epilogue
        .requires_bias()
        .then(|| VectorRef { data: dev_bias_buf.as_slice(), len: n, stride: 1 });

    let args = GemmArgs::<F32Strict> {
        a: MatrixRef { data: dev_a_buf.as_slice(), rows: m, cols: k, ld: k as i64 },
        b: MatrixRef { data: dev_b_buf.as_slice(), rows: k, cols: n, ld: ldb as i64 },
        c: None,
        d: MatrixMut { data: dev_d_buf.as_slice_mut(), rows: m, cols: n, ld: n as i64 },
        bias: bias_arg,
        alpha: 1.0, beta: 0.0,
    };
    plan.can_implement(&args).expect("can_implement");
    plan.run(&stream, Workspace::None, args).expect("run");

    let mut out_strict = vec![F32Strict(0.0); (m * n) as usize];
    dev_d_buf.copy_to_host(&mut out_strict).expect("download D");
    let out: Vec<f32> = out_strict.iter().map(|F32Strict(x)| *x).collect();

    // SIMT f32 GEMM is full-precision; tolerance is much tighter than TF32.
    // Use absolute tolerance since GELU/SiLU can produce values near 0.
    let mut max_abs = 0.0f32;
    for (got, want) in out.iter().zip(expected.iter()) {
        let diff = (got - want).abs();
        if diff > max_abs { max_abs = diff; }
    }
    let tol = (k as f32) * 1e-5;
    assert!(
        max_abs < tol,
        "F32Strict {:?} {:?} GEMM ({m}x{n}x{k}): max abs err {max_abs} > tol {tol}",
        layout, epilogue
    );
    println!(
        "F32Strict {:?} sm80 {:?} GEMM ({m}x{n}x{k}): max abs err {max_abs} (tol {tol}) ✅",
        layout, epilogue
    );
}

// ---- RCR ----

#[test] #[ignore] fn f32_strict_identity_rcr_64_64_32()   { run_f32_strict(LayoutSku::Rcr, EpilogueKind::Identity,  64, 64, 32); }
#[test] #[ignore] fn f32_strict_identity_rcr_128_128_64() { run_f32_strict(LayoutSku::Rcr, EpilogueKind::Identity, 128, 128, 64); }
#[test] #[ignore] fn f32_strict_bias_rcr_64_64_32()       { run_f32_strict(LayoutSku::Rcr, EpilogueKind::Bias,      64, 64, 32); }
#[test] #[ignore] fn f32_strict_bias_relu_rcr_64_64_32()  { run_f32_strict(LayoutSku::Rcr, EpilogueKind::BiasRelu,  64, 64, 32); }
#[test] #[ignore] fn f32_strict_bias_gelu_rcr_64_64_32()  { run_f32_strict(LayoutSku::Rcr, EpilogueKind::BiasGelu,  64, 64, 32); }
#[test] #[ignore] fn f32_strict_bias_silu_rcr_64_64_32()  { run_f32_strict(LayoutSku::Rcr, EpilogueKind::BiasSilu,  64, 64, 32); }

// ---- RRR ----

#[test] #[ignore] fn f32_strict_identity_rrr_64_64_32()   { run_f32_strict(LayoutSku::Rrr, EpilogueKind::Identity,  64, 64, 32); }
#[test] #[ignore] fn f32_strict_identity_rrr_128_128_64() { run_f32_strict(LayoutSku::Rrr, EpilogueKind::Identity, 128, 128, 64); }
#[test] #[ignore] fn f32_strict_bias_rrr_64_64_32()       { run_f32_strict(LayoutSku::Rrr, EpilogueKind::Bias,      64, 64, 32); }
#[test] #[ignore] fn f32_strict_bias_relu_rrr_64_64_32()  { run_f32_strict(LayoutSku::Rrr, EpilogueKind::BiasRelu,  64, 64, 32); }
#[test] #[ignore] fn f32_strict_bias_gelu_rrr_64_64_32()  { run_f32_strict(LayoutSku::Rrr, EpilogueKind::BiasGelu,  64, 64, 32); }
#[test] #[ignore] fn f32_strict_bias_silu_rrr_64_64_32()  { run_f32_strict(LayoutSku::Rrr, EpilogueKind::BiasSilu,  64, 64, 32); }
