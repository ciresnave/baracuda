//! Real-GPU smoke tests for the alpha.11 bias + activation SKUs:
//! `BiasRelu`, `BiasGelu`, `BiasSilu` over f16 + bf16 on Rcr × sm_80.
//!
//! All six combinations are exercised against a CPU reference that
//! applies the same activation to the matmul + bias accumulator. Each
//! test verifies (a) the activation reaches the output (e.g., negative
//! pre-activation values become 0 under ReLU) and (b) the precision
//! falls within the dtype's expected tolerance.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-cutlass --release -- --ignored`.

use baracuda_cutlass::{
    ActivationKind, EpilogueKind, GemmArgs, GemmDescriptor, GemmPlan, LayoutSku, MatrixMut,
    MatrixRef, PlanPreference, VectorRef, Workspace,
};
use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use half::{bf16, f16};

// ---- CPU references for each activation ----

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Abramowitz & Stegun 7.1.26 polynomial approximation to erf, max abs
/// error ~1.5e-7 — well below f16/bf16 precision so this is fine for
/// the CPU reference. Avoids pulling in `libm` as a dev-dep.
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
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-ax * ax).exp();
    sign * y
}

/// Exact (erf-based) GELU — matches CUTLASS's `epilogue::thread::GELU`
/// and PyTorch's default `nn.GELU()`. The closed form is
/// `gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`.
fn gelu_exact(x: f32) -> f32 {
    let inv_sqrt_2 = 1.0 / std::f32::consts::SQRT_2;
    0.5 * x * (1.0 + erf_approx(x * inv_sqrt_2))
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[allow(clippy::too_many_arguments)]
fn cpu_bias_act_gemm_rcr(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    bias: &[f32],
    alpha: f32,
    activation: ActivationKind,
    d: &mut [f32],
    ldd: usize,
) {
    let act_fn: fn(f32) -> f32 = match activation {
        ActivationKind::Relu => relu,
        ActivationKind::Gelu => gelu_exact,
        ActivationKind::Silu => silu,
    };
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let a_val = a[i * lda + kk];
                let b_val = b[j * ldb + kk]; // column-major B
                acc += a_val * b_val;
            }
            let pre = alpha * acc + bias[j];
            d[i * ldd + j] = act_fn(pre);
        }
    }
}

// ---- Generic helpers, one per element type ----

fn run_f16(m: i32, n: i32, k: i32, activation: ActivationKind, epilogue: EpilogueKind) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    // Bias values straddle 0 so ReLU has actual work to do (some
    // pre-activation values land negative and must clamp to 0).
    let host_a_f32: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 0.01).sin()).collect();
    let host_b_f32: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 0.013).cos()).collect();
    let host_bias_f32: Vec<f32> = (0..n)
        .map(|j| -0.5 + 0.1 * (j as f32 % 7.0))
        .collect();

    let mut host_d_ref = vec![0.0f32; (m * n) as usize];
    cpu_bias_act_gemm_rcr(
        m as usize, n as usize, k as usize,
        &host_a_f32, k as usize,
        &host_b_f32, k as usize,
        &host_bias_f32,
        1.0,
        activation,
        &mut host_d_ref, n as usize,
    );

    let host_a: Vec<f16> = host_a_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let host_b: Vec<f16> = host_b_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let host_bias: Vec<f16> = host_bias_f32.iter().map(|&x| f16::from_f32(x)).collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload A");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload B");
    let dev_bias = DeviceBuffer::from_slice(&ctx, &host_bias).expect("upload bias");
    let mut dev_d: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (m * n) as usize).expect("alloc D");

    let desc = GemmDescriptor {
        m, n, k,
        layout: LayoutSku::Rcr,
        epilogue,
    };
    let plan =
        GemmPlan::<f16>::select(&stream, &desc, PlanPreference::default()).expect("plan select");
    assert_eq!(plan.sku().epilogue, epilogue);
    assert_eq!(plan.sku().epilogue.activation(), Some(activation));

    let args = GemmArgs::<f16> {
        a: MatrixRef { data: dev_a.as_slice(), rows: m, cols: k, ld: k as i64 },
        b: MatrixRef { data: dev_b.as_slice(), rows: k, cols: n, ld: k as i64 },
        c: None,
        d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n as i64 },
        bias: Some(VectorRef { data: dev_bias.as_slice(), len: n, stride: 1 }),
        alpha: 1.0, beta: 0.0,
    };
    plan.can_implement(&args).expect("can_implement");
    plan.run(&stream, Workspace::None, args).expect("run");

    let mut host_d_out = vec![f16::ZERO; (m * n) as usize];
    dev_d.copy_to_host(&mut host_d_out).expect("download D");

    let mut max_err = 0.0f32;
    for (got, want) in host_d_out.iter().zip(host_d_ref.iter()) {
        let e = (got.to_f32() - want).abs();
        if e > max_err {
            max_err = e;
        }
    }
    let tol = (k as f32) * 5e-3;
    assert!(
        max_err < tol,
        "f16 bias+{:?} GEMM ({m}x{n}x{k}): max abs err {max_err} > tol {tol}",
        activation
    );
    println!(
        "f16 RCR sm80 bias+{:?} GEMM ({m}x{n}x{k}): max abs err {max_err} (tol {tol}) ✅",
        activation
    );
}

fn run_bf16(m: i32, n: i32, k: i32, activation: ActivationKind, epilogue: EpilogueKind) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let host_a_f32: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 0.01).sin()).collect();
    let host_b_f32: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 0.013).cos()).collect();
    let host_bias_f32: Vec<f32> = (0..n)
        .map(|j| -0.5 + 0.1 * (j as f32 % 7.0))
        .collect();

    let mut host_d_ref = vec![0.0f32; (m * n) as usize];
    cpu_bias_act_gemm_rcr(
        m as usize, n as usize, k as usize,
        &host_a_f32, k as usize,
        &host_b_f32, k as usize,
        &host_bias_f32,
        1.0,
        activation,
        &mut host_d_ref, n as usize,
    );

    let host_a: Vec<bf16> = host_a_f32.iter().map(|&x| bf16::from_f32(x)).collect();
    let host_b: Vec<bf16> = host_b_f32.iter().map(|&x| bf16::from_f32(x)).collect();
    let host_bias: Vec<bf16> = host_bias_f32.iter().map(|&x| bf16::from_f32(x)).collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload A");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload B");
    let dev_bias = DeviceBuffer::from_slice(&ctx, &host_bias).expect("upload bias");
    let mut dev_d: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, (m * n) as usize).expect("alloc D");

    let desc = GemmDescriptor {
        m, n, k,
        layout: LayoutSku::Rcr,
        epilogue,
    };
    let plan =
        GemmPlan::<bf16>::select(&stream, &desc, PlanPreference::default()).expect("plan select");
    assert_eq!(plan.sku().epilogue.activation(), Some(activation));

    let args = GemmArgs::<bf16> {
        a: MatrixRef { data: dev_a.as_slice(), rows: m, cols: k, ld: k as i64 },
        b: MatrixRef { data: dev_b.as_slice(), rows: k, cols: n, ld: k as i64 },
        c: None,
        d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n as i64 },
        bias: Some(VectorRef { data: dev_bias.as_slice(), len: n, stride: 1 }),
        alpha: 1.0, beta: 0.0,
    };
    plan.can_implement(&args).expect("can_implement");
    plan.run(&stream, Workspace::None, args).expect("run");

    let mut host_d_out = vec![bf16::ZERO; (m * n) as usize];
    dev_d.copy_to_host(&mut host_d_out).expect("download D");

    let mut max_err = 0.0f32;
    for (got, want) in host_d_out.iter().zip(host_d_ref.iter()) {
        let e = (got.to_f32() - want).abs();
        if e > max_err {
            max_err = e;
        }
    }
    let tol = (k as f32) * 5e-3;
    assert!(
        max_err < tol,
        "bf16 bias+{:?} GEMM ({m}x{n}x{k}): max abs err {max_err} > tol {tol}",
        activation
    );
    println!(
        "bf16 RCR sm80 bias+{:?} GEMM ({m}x{n}x{k}): max abs err {max_err} (tol {tol}) ✅",
        activation
    );
}

// ---- 6 SKU smoke tests ----

#[test]
#[ignore]
fn f16_bias_relu_rcr_sm80_64_64_32() {
    run_f16(64, 64, 32, ActivationKind::Relu, EpilogueKind::BiasRelu);
}

#[test]
#[ignore]
fn bf16_bias_relu_rcr_sm80_64_64_32() {
    run_bf16(64, 64, 32, ActivationKind::Relu, EpilogueKind::BiasRelu);
}

#[test]
#[ignore]
fn f16_bias_gelu_rcr_sm80_64_64_32() {
    run_f16(64, 64, 32, ActivationKind::Gelu, EpilogueKind::BiasGelu);
}

#[test]
#[ignore]
fn bf16_bias_gelu_rcr_sm80_64_64_32() {
    run_bf16(64, 64, 32, ActivationKind::Gelu, EpilogueKind::BiasGelu);
}

#[test]
#[ignore]
fn f16_bias_silu_rcr_sm80_64_64_32() {
    run_f16(64, 64, 32, ActivationKind::Silu, EpilogueKind::BiasSilu);
}

#[test]
#[ignore]
fn bf16_bias_silu_rcr_sm80_64_64_32() {
    run_bf16(64, 64, 32, ActivationKind::Silu, EpilogueKind::BiasSilu);
}

/// Sanity: the test setup must actually produce some negative
/// pre-activation values, otherwise the ReLU clamp branch is never
/// exercised and the test reduces to plain bias. This is a regression
/// guard on the test bias-distribution choice itself.
#[test]
fn test_inputs_actually_exercise_relu_clamp() {
    // Reproduce the CPU reference for the (m=64, n=64, k=32) f16 case.
    let m: usize = 64; let n: usize = 64; let k: usize = 32;
    let a: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 0.01).sin()).collect();
    let b: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 0.013).cos()).collect();
    let bias: Vec<f32> = (0..n).map(|j| -0.5 + 0.1 * (j as f32 % 7.0)).collect();

    let mut had_negative = false;
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[j * k + kk];
            }
            if acc + bias[j] < -0.05 {
                had_negative = true;
            }
        }
    }
    assert!(
        had_negative,
        "test inputs don't produce negative pre-activation values; \
         ReLU clamp branch isn't being tested"
    );
}
