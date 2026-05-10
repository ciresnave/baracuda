//! Real-GPU smoke test for bias-fused GEMM (alpha.10).
//!
//! Computes `D = alpha*AB + beta*C + bias_broadcast(N)` and compares
//! against a CPU reference. `#[ignore]` by default; run with
//! `cargo test -p baracuda-cutlass --release -- --ignored`.

use baracuda_cutlass::{
    EpilogueKind, GemmArgs, GemmDescriptor, GemmPlan, LayoutSku, MatrixMut, MatrixRef,
    PlanPreference, VectorRef, Workspace,
};
use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use half::{bf16, f16};

/// CPU reference for `D = alpha*AB + beta*C + bias_broadcast(N)`,
/// RCR layout (A row-major, B column-major, C/D row-major).
#[allow(clippy::too_many_arguments)]
fn cpu_bias_gemm_rcr(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    c: Option<&[f32]>,
    ldc: usize,
    bias: &[f32],
    alpha: f32,
    beta: f32,
    d: &mut [f32],
    ldd: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let a_val = a[i * lda + kk];
                let b_val = b[j * ldb + kk]; // column-major B
                acc += a_val * b_val;
            }
            let c_val = match c {
                Some(c_buf) => c_buf[i * ldc + j],
                None => 0.0,
            };
            d[i * ldd + j] = alpha * acc + beta * c_val + bias[j];
        }
    }
}

#[test]
#[ignore]
fn f16_bias_gemm_rcr_sm80_64_64_32() {
    bias_round_trip_f16(64, 64, 32);
}

#[test]
#[ignore]
fn f16_bias_gemm_rcr_sm80_128_128_64() {
    bias_round_trip_f16(128, 128, 64);
}

#[test]
#[ignore]
fn bf16_bias_gemm_rcr_sm80_64_64_32() {
    bias_round_trip_bf16(64, 64, 32);
}

fn bias_round_trip_f16(m: i32, n: i32, k: i32) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let host_a_f32: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 0.01).sin()).collect();
    let host_b_f32: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 0.013).cos()).collect();
    // Bias values are O(1) and distinct per column so a simple
    // broadcast bug would show up clearly in the error metric.
    let host_bias_f32: Vec<f32> = (0..n)
        .map(|j| 0.25 + 0.1 * (j as f32))
        .collect();

    let mut host_d_ref = vec![0.0f32; (m * n) as usize];
    cpu_bias_gemm_rcr(
        m as usize, n as usize, k as usize,
        &host_a_f32, k as usize,
        &host_b_f32, k as usize,
        None, n as usize,
        &host_bias_f32,
        1.0, 0.0,
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
        epilogue: EpilogueKind::Bias,
    };
    let plan =
        GemmPlan::<f16>::select(&stream, &desc, PlanPreference::default()).expect("plan select");

    // Sanity: epilogue must be Bias.
    assert_eq!(plan.sku().epilogue, EpilogueKind::Bias);

    let args = GemmArgs::<f16> {
        a: MatrixRef { data: dev_a.as_slice(), rows: m, cols: k, ld: k as i64 },
        b: MatrixRef { data: dev_b.as_slice(), rows: k, cols: n, ld: k as i64 },
        c: None,
        d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n as i64 },
        bias: Some(VectorRef { data: dev_bias.as_slice(), len: n, stride: 1 }),
        alpha: 1.0,
        beta: 0.0,
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
        "f16 bias GEMM ({m}x{n}x{k}): max abs err {max_err} > tol {tol}"
    );
    println!("f16 RCR sm80 bias GEMM ({m}x{n}x{k}): max abs err {max_err} (tol {tol}) ✅");
}

fn bias_round_trip_bf16(m: i32, n: i32, k: i32) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let host_a_f32: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 0.01).sin()).collect();
    let host_b_f32: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 0.013).cos()).collect();
    let host_bias_f32: Vec<f32> = (0..n)
        .map(|j| 0.25 + 0.1 * (j as f32))
        .collect();

    let mut host_d_ref = vec![0.0f32; (m * n) as usize];
    cpu_bias_gemm_rcr(
        m as usize, n as usize, k as usize,
        &host_a_f32, k as usize,
        &host_b_f32, k as usize,
        None, n as usize,
        &host_bias_f32,
        1.0, 0.0,
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
        epilogue: EpilogueKind::Bias,
    };
    let plan =
        GemmPlan::<bf16>::select(&stream, &desc, PlanPreference::default()).expect("plan select");
    let args = GemmArgs::<bf16> {
        a: MatrixRef { data: dev_a.as_slice(), rows: m, cols: k, ld: k as i64 },
        b: MatrixRef { data: dev_b.as_slice(), rows: k, cols: n, ld: k as i64 },
        c: None,
        d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n as i64 },
        bias: Some(VectorRef { data: dev_bias.as_slice(), len: n, stride: 1 }),
        alpha: 1.0,
        beta: 0.0,
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
        "bf16 bias GEMM ({m}x{n}x{k}): max abs err {max_err} > tol {tol}"
    );
    println!("bf16 RCR sm80 bias GEMM ({m}x{n}x{k}): max abs err {max_err} (tol {tol}) ✅");
}

/// Bias requires Some(bias) at construction; Identity rejects it.
/// These checks are host-only (rejected before any kernel launch),
/// but we need a Stream to construct the plan.
#[test]
#[ignore]
fn epilogue_bias_mismatch_rejected() {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let m = 64i32;
    let n = 64i32;
    let k = 32i32;
    let dev_a: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, (m * k) as usize).unwrap();
    let dev_b: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, (k * n) as usize).unwrap();
    let mut dev_d: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, (m * n) as usize).unwrap();
    let dev_bias: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n as usize).unwrap();

    // Case 1: epilogue = Bias but bias = None → rejected.
    let desc_bias = GemmDescriptor {
        m, n, k,
        layout: LayoutSku::Rcr,
        epilogue: EpilogueKind::Bias,
    };
    let plan_bias = GemmPlan::<f16>::select(&stream, &desc_bias, PlanPreference::default())
        .expect("plan select Bias");
    let args_missing_bias = GemmArgs::<f16> {
        a: MatrixRef { data: dev_a.as_slice(), rows: m, cols: k, ld: k as i64 },
        b: MatrixRef { data: dev_b.as_slice(), rows: k, cols: n, ld: k as i64 },
        c: None,
        d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n as i64 },
        bias: None,
        alpha: 1.0, beta: 0.0,
    };
    let r1 = plan_bias.can_implement(&args_missing_bias);
    assert!(
        matches!(r1, Err(baracuda_cutlass::Error::InvalidProblem(_))),
        "Bias epilogue with bias=None should be rejected; got {r1:?}"
    );

    // Case 2: epilogue = Identity but bias = Some(...) → rejected.
    let desc_identity = GemmDescriptor {
        m, n, k,
        layout: LayoutSku::Rcr,
        epilogue: EpilogueKind::Identity,
    };
    let plan_identity =
        GemmPlan::<f16>::select(&stream, &desc_identity, PlanPreference::default())
            .expect("plan select Identity");
    let args_extra_bias = GemmArgs::<f16> {
        a: MatrixRef { data: dev_a.as_slice(), rows: m, cols: k, ld: k as i64 },
        b: MatrixRef { data: dev_b.as_slice(), rows: k, cols: n, ld: k as i64 },
        c: None,
        d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n as i64 },
        bias: Some(VectorRef { data: dev_bias.as_slice(), len: n, stride: 1 }),
        alpha: 1.0, beta: 0.0,
    };
    let r2 = plan_identity.can_implement(&args_extra_bias);
    assert!(
        matches!(r2, Err(baracuda_cutlass::Error::InvalidProblem(_))),
        "Identity epilogue with bias=Some should be rejected; got {r2:?}"
    );

    println!("bias / epilogue mismatch rejection ✅");
}
