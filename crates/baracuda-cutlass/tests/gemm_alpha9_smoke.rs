//! Real-GPU smoke tests for the alpha.9 kernel SKUs:
//!
//! - Rrr layout (f16, bf16) — row-major × row-major matmul
//! - TF32 GEMM (f32 input via TF32 tensor cores)
//! - Batched GEMM with uniform shape (f16, bf16)
//!
//! `#[ignore]` so `cargo test --workspace` is GPU-free by default.
//! Run with `cargo test -p baracuda-cutlass --release -- --ignored`.

use baracuda_cutlass::{
    BatchedGemmArgs, BatchedGemmDescriptor, BatchedGemmPlan, ElementKind, EpilogueKind, GemmArgs,
    GemmDescriptor, GemmPlan, LayoutSku, MathPrecision, MatrixMut, MatrixRef, PlanPreference,
    Workspace,
};
use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use half::{bf16, f16};

// ---------- CPU references ----------

#[allow(clippy::too_many_arguments)]
fn cpu_gemm_rrr(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    c: Option<&[f32]>,
    ldc: usize,
    alpha: f32,
    beta: f32,
    d: &mut [f32],
    ldd: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let a_val = a[i * lda + kk]; // row-major A[i, kk]
                let b_val = b[kk * ldb + j]; // row-major B[kk, j]
                acc += a_val * b_val;
            }
            let c_val = match c {
                Some(c_buf) => c_buf[i * ldc + j],
                None => 0.0,
            };
            d[i * ldd + j] = alpha * acc + beta * c_val;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn cpu_gemm_rcr(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    c: Option<&[f32]>,
    ldc: usize,
    alpha: f32,
    beta: f32,
    d: &mut [f32],
    ldd: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let a_val = a[i * lda + kk]; // row-major A[i, kk]
                let b_val = b[j * ldb + kk]; // column-major B[kk, j]
                acc += a_val * b_val;
            }
            let c_val = match c {
                Some(c_buf) => c_buf[i * ldc + j],
                None => 0.0,
            };
            d[i * ldd + j] = alpha * acc + beta * c_val;
        }
    }
}

// ---------- Rrr layout: f16 + bf16 ----------

#[test]
#[ignore]
fn f16_gemm_rrr_sm80_64_64_32() {
    rrr_round_trip_f16(64, 64, 32);
}

#[test]
#[ignore]
fn bf16_gemm_rrr_sm80_64_64_32() {
    rrr_round_trip_bf16(64, 64, 32);
}

#[test]
#[ignore]
fn f16_gemm_rrr_sm80_128_128_64() {
    rrr_round_trip_f16(128, 128, 64);
}

fn rrr_round_trip_f16(m: i32, n: i32, k: i32) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let host_a_f32: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 0.01).sin()).collect();
    let host_b_f32: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 0.013).cos()).collect();
    let mut host_d_ref = vec![0.0f32; (m * n) as usize];
    cpu_gemm_rrr(
        m as usize, n as usize, k as usize,
        &host_a_f32, k as usize,
        &host_b_f32, n as usize,
        None, n as usize,
        1.0, 0.0,
        &mut host_d_ref, n as usize,
    );

    let host_a: Vec<f16> = host_a_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let host_b: Vec<f16> = host_b_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload A");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload B");
    let mut dev_d: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (m * n) as usize).expect("alloc D");

    let desc = GemmDescriptor {
        m, n, k,
        layout: LayoutSku::Rrr,
        epilogue: EpilogueKind::Identity,
    };
    let plan =
        GemmPlan::<f16>::select(&stream, &desc, PlanPreference::default()).expect("plan select");
    let args = GemmArgs::<f16> {
        a: MatrixRef { data: dev_a.as_slice(), rows: m, cols: k, ld: k as i64 },
        b: MatrixRef { data: dev_b.as_slice(), rows: k, cols: n, ld: n as i64 },
        c: None,
        d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n as i64 },
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
        "f16 Rrr GEMM ({m}x{n}x{k}): max abs err {max_err} > tol {tol}"
    );
    println!("f16 RRR sm80 GEMM ({m}x{n}x{k}): max abs err {max_err} (tol {tol}) ✅");
}

fn rrr_round_trip_bf16(m: i32, n: i32, k: i32) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let host_a_f32: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 0.01).sin()).collect();
    let host_b_f32: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 0.013).cos()).collect();
    let mut host_d_ref = vec![0.0f32; (m * n) as usize];
    cpu_gemm_rrr(
        m as usize, n as usize, k as usize,
        &host_a_f32, k as usize,
        &host_b_f32, n as usize,
        None, n as usize,
        1.0, 0.0,
        &mut host_d_ref, n as usize,
    );

    let host_a: Vec<bf16> = host_a_f32.iter().map(|&x| bf16::from_f32(x)).collect();
    let host_b: Vec<bf16> = host_b_f32.iter().map(|&x| bf16::from_f32(x)).collect();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload A");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload B");
    let mut dev_d: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, (m * n) as usize).expect("alloc D");

    let desc = GemmDescriptor {
        m, n, k,
        layout: LayoutSku::Rrr,
        epilogue: EpilogueKind::Identity,
    };
    let plan =
        GemmPlan::<bf16>::select(&stream, &desc, PlanPreference::default()).expect("plan select");
    let args = GemmArgs::<bf16> {
        a: MatrixRef { data: dev_a.as_slice(), rows: m, cols: k, ld: k as i64 },
        b: MatrixRef { data: dev_b.as_slice(), rows: k, cols: n, ld: n as i64 },
        c: None,
        d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n as i64 },
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
        "bf16 Rrr GEMM ({m}x{n}x{k}): max abs err {max_err} > tol {tol}"
    );
    println!("bf16 RRR sm80 GEMM ({m}x{n}x{k}): max abs err {max_err} (tol {tol}) ✅");
}

// ---------- TF32 (f32 inputs via TF32 tensor cores) ----------

#[test]
#[ignore]
fn tf32_gemm_rcr_sm80_64_64_32() {
    tf32_round_trip(64, 64, 32);
}

#[test]
#[ignore]
fn tf32_gemm_rcr_sm80_128_128_64() {
    tf32_round_trip(128, 128, 64);
}

fn tf32_round_trip(m: i32, n: i32, k: i32) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let host_a: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 0.01).sin()).collect();
    let host_b: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 0.013).cos()).collect();
    let mut host_d_ref = vec![0.0f32; (m * n) as usize];
    cpu_gemm_rcr(
        m as usize, n as usize, k as usize,
        &host_a, k as usize,
        &host_b, k as usize,
        None, n as usize,
        1.0, 0.0,
        &mut host_d_ref, n as usize,
    );

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload A");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload B");
    let mut dev_d: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (m * n) as usize).expect("alloc D");

    let desc = GemmDescriptor {
        m, n, k,
        layout: LayoutSku::Rcr,
        epilogue: EpilogueKind::Identity,
    };
    let plan =
        GemmPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("plan select");

    // Sanity check: f32 input → TF32 math (10-bit mantissa).
    assert_eq!(plan.precision_guarantee().math_precision, MathPrecision::Tf32);
    assert_eq!(plan.precision_guarantee().accumulator, ElementKind::F32);

    let args = GemmArgs::<f32> {
        a: MatrixRef { data: dev_a.as_slice(), rows: m, cols: k, ld: k as i64 },
        b: MatrixRef { data: dev_b.as_slice(), rows: k, cols: n, ld: k as i64 },
        c: None,
        d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n as i64 },
        alpha: 1.0, beta: 0.0,
    };
    plan.can_implement(&args).expect("can_implement");
    plan.run(&stream, Workspace::None, args).expect("run");

    let mut host_d_out = vec![0.0f32; (m * n) as usize];
    dev_d.copy_to_host(&mut host_d_out).expect("download D");

    let mut max_rel = 0.0f32;
    for (got, want) in host_d_out.iter().zip(host_d_ref.iter()) {
        let denom = want.abs().max(1.0);
        let rel = (got - want).abs() / denom;
        if rel > max_rel {
            max_rel = rel;
        }
    }
    // TF32 has ~10-bit mantissa (~1e-3 relative). Allow O(K * 1e-4) for
    // accumulation slack with our bounded inputs.
    let tol = (k as f32) * 5e-4;
    assert!(
        max_rel < tol,
        "TF32 GEMM ({m}x{n}x{k}): max rel err {max_rel} > tol {tol}"
    );
    println!("TF32 RCR sm80 GEMM ({m}x{n}x{k}): max rel err {max_rel} (tol {tol}) ✅");
}

// ---------- Batched GEMM (uniform shape) ----------

#[test]
#[ignore]
fn batched_f16_rcr_sm80_4_batches() {
    batched_round_trip_f16(64, 64, 32, 4);
}

#[test]
#[ignore]
fn batched_bf16_rcr_sm80_4_batches() {
    batched_round_trip_bf16(64, 64, 32, 4);
}

fn batched_round_trip_f16(m: i32, n: i32, k: i32, batch_count: i32) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    // Per-batch sin/cos seeds offset by batch index to give each batch
    // a distinct expected output.
    let elements_a = (m * k) as usize;
    let elements_b = (k * n) as usize;
    let elements_d = (m * n) as usize;
    let total_a = elements_a * batch_count as usize;
    let total_b = elements_b * batch_count as usize;
    let total_d = elements_d * batch_count as usize;

    let mut host_a_f32 = vec![0.0f32; total_a];
    let mut host_b_f32 = vec![0.0f32; total_b];
    for batch in 0..batch_count as usize {
        for i in 0..elements_a {
            host_a_f32[batch * elements_a + i] =
                (((batch * 1000 + i) as f32) * 0.01).sin();
        }
        for i in 0..elements_b {
            host_b_f32[batch * elements_b + i] =
                (((batch * 1000 + i) as f32) * 0.013).cos();
        }
    }

    let mut host_d_ref = vec![0.0f32; total_d];
    for batch in 0..batch_count as usize {
        let a_off = batch * elements_a;
        let b_off = batch * elements_b;
        let d_off = batch * elements_d;
        cpu_gemm_rcr(
            m as usize, n as usize, k as usize,
            &host_a_f32[a_off..a_off + elements_a], k as usize,
            &host_b_f32[b_off..b_off + elements_b], k as usize,
            None, n as usize,
            1.0, 0.0,
            &mut host_d_ref[d_off..d_off + elements_d], n as usize,
        );
    }

    let host_a: Vec<f16> = host_a_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let host_b: Vec<f16> = host_b_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload A");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload B");
    let mut dev_d: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, total_d).expect("alloc D");

    let desc = BatchedGemmDescriptor {
        m, n, k,
        batch_count,
        layout: LayoutSku::Rcr,
        epilogue: EpilogueKind::Identity,
    };
    let plan = BatchedGemmPlan::<f16>::select(&stream, &desc, PlanPreference::default())
        .expect("plan select");

    let args = BatchedGemmArgs::<f16> {
        a: MatrixRef { data: dev_a.as_slice(), rows: m, cols: k, ld: k as i64 },
        stride_a: elements_a as i64,
        b: MatrixRef { data: dev_b.as_slice(), rows: k, cols: n, ld: k as i64 },
        stride_b: elements_b as i64,
        c: None,
        stride_c: 0,
        d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n as i64 },
        stride_d: elements_d as i64,
        alpha: 1.0, beta: 0.0,
    };
    plan.can_implement(&args).expect("can_implement");
    plan.run(&stream, Workspace::None, args).expect("run");

    let mut host_d_out = vec![f16::ZERO; total_d];
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
        "f16 batched GEMM ({batch_count}x {m}x{n}x{k}): max abs err {max_err} > tol {tol}"
    );
    println!(
        "f16 batched RCR sm80 GEMM ({batch_count} x {m}x{n}x{k}): max abs err {max_err} (tol {tol}) ✅"
    );
}

fn batched_round_trip_bf16(m: i32, n: i32, k: i32, batch_count: i32) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let elements_a = (m * k) as usize;
    let elements_b = (k * n) as usize;
    let elements_d = (m * n) as usize;
    let total_a = elements_a * batch_count as usize;
    let total_b = elements_b * batch_count as usize;
    let total_d = elements_d * batch_count as usize;

    let mut host_a_f32 = vec![0.0f32; total_a];
    let mut host_b_f32 = vec![0.0f32; total_b];
    for batch in 0..batch_count as usize {
        for i in 0..elements_a {
            host_a_f32[batch * elements_a + i] =
                (((batch * 1000 + i) as f32) * 0.01).sin();
        }
        for i in 0..elements_b {
            host_b_f32[batch * elements_b + i] =
                (((batch * 1000 + i) as f32) * 0.013).cos();
        }
    }

    let mut host_d_ref = vec![0.0f32; total_d];
    for batch in 0..batch_count as usize {
        let a_off = batch * elements_a;
        let b_off = batch * elements_b;
        let d_off = batch * elements_d;
        cpu_gemm_rcr(
            m as usize, n as usize, k as usize,
            &host_a_f32[a_off..a_off + elements_a], k as usize,
            &host_b_f32[b_off..b_off + elements_b], k as usize,
            None, n as usize,
            1.0, 0.0,
            &mut host_d_ref[d_off..d_off + elements_d], n as usize,
        );
    }

    let host_a: Vec<bf16> = host_a_f32.iter().map(|&x| bf16::from_f32(x)).collect();
    let host_b: Vec<bf16> = host_b_f32.iter().map(|&x| bf16::from_f32(x)).collect();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload A");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload B");
    let mut dev_d: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, total_d).expect("alloc D");

    let desc = BatchedGemmDescriptor {
        m, n, k,
        batch_count,
        layout: LayoutSku::Rcr,
        epilogue: EpilogueKind::Identity,
    };
    let plan = BatchedGemmPlan::<bf16>::select(&stream, &desc, PlanPreference::default())
        .expect("plan select");
    let args = BatchedGemmArgs::<bf16> {
        a: MatrixRef { data: dev_a.as_slice(), rows: m, cols: k, ld: k as i64 },
        stride_a: elements_a as i64,
        b: MatrixRef { data: dev_b.as_slice(), rows: k, cols: n, ld: k as i64 },
        stride_b: elements_b as i64,
        c: None,
        stride_c: 0,
        d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n as i64 },
        stride_d: elements_d as i64,
        alpha: 1.0, beta: 0.0,
    };
    plan.can_implement(&args).expect("can_implement");
    plan.run(&stream, Workspace::None, args).expect("run");

    let mut host_d_out = vec![bf16::ZERO; total_d];
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
        "bf16 batched GEMM ({batch_count} x {m}x{n}x{k}): max abs err {max_err} > tol {tol}"
    );
    println!(
        "bf16 batched RCR sm80 GEMM ({batch_count} x {m}x{n}x{k}): max abs err {max_err} (tol {tol}) ✅"
    );
}
