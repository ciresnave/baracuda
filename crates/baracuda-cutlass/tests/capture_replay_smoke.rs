//! Stream-capture regression coverage for the alpha.9 / alpha.10 /
//! alpha.11 / alpha.12 plan surfaces.
//!
//! `GroupedGemmPlan` already has its own capture-replay test in
//! `grouped_gemm_smoke.rs` (the pinned-metadata H2D path was the
//! tricky one to get right). The plan types added since haven't been
//! exercised under capture — these tests close that gap, one per
//! dispatch family so a future kernel addition that breaks capture
//! safety (e.g. an inadvertent host-sync helper) gets caught here
//! rather than in Fuel's graph-replay path.
//!
//! Coverage: vanilla single GEMM (Rcr Identity, Rrr Identity, TF32),
//! bias-fused single GEMM, batched GEMM. `#[ignore]` by default; run
//! with `cargo test -p baracuda-cutlass --release -- --ignored`.

use baracuda_cutlass::{
    BatchedGemmArgs, BatchedGemmDescriptor, BatchedGemmPlan, EpilogueKind, GemmArgs,
    GemmDescriptor, GemmPlan, LayoutSku, MatrixMut, MatrixRef, PlanPreference, VectorRef,
    Workspace,
};
use baracuda_driver::{init, CaptureMode, Context, Device, DeviceBuffer, Stream};
use half::f16;

// ---- CPU references ----

fn cpu_gemm_rcr_f32(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], d: &mut [f32]) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[j * k + kk]; // column-major B
            }
            d[i * n + j] = acc;
        }
    }
}

fn cpu_gemm_rrr_f32(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], d: &mut [f32]) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j]; // row-major B
            }
            d[i * n + j] = acc;
        }
    }
}

// A&S 7.1.26 erf approximation (used by gelu_exact below).
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

#[allow(clippy::too_many_arguments)]
fn cpu_bias_gelu_rcr_f32(
    m: usize, n: usize, k: usize,
    a: &[f32], b: &[f32], bias: &[f32],
    d: &mut [f32],
) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[j * k + kk]; // column-major B
            }
            d[i * n + j] = gelu_exact(acc + bias[j]);
        }
    }
}

// ---- 1. Vanilla single GEMM: f16 Rcr Identity ----

#[test]
#[ignore]
fn gemm_capture_replay_f16_rcr_identity() {
    init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = Stream::new(&ctx).unwrap();

    let (m, n, k) = (64i32, 64i32, 32i32);
    let host_a_f32: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 0.01).sin()).collect();
    let host_b_f32: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 0.013).cos()).collect();
    let mut expected = vec![0.0f32; (m * n) as usize];
    cpu_gemm_rcr_f32(m as usize, n as usize, k as usize, &host_a_f32, &host_b_f32, &mut expected);

    let host_a: Vec<f16> = host_a_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let host_b: Vec<f16> = host_b_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).unwrap();
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).unwrap();
    let mut dev_d: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, (m * n) as usize).unwrap();

    let desc = GemmDescriptor { m, n, k, layout: LayoutSku::Rcr, epilogue: EpilogueKind::Identity };
    let plan = GemmPlan::<f16>::select(&stream, &desc, PlanPreference::default()).unwrap();

    let graph = stream
        .capture(CaptureMode::ThreadLocal, |s| {
            let args = GemmArgs::<f16> {
                a: MatrixRef { data: dev_a.as_slice(), rows: m, cols: k, ld: k as i64 },
                b: MatrixRef { data: dev_b.as_slice(), rows: k, cols: n, ld: k as i64 },
                c: None,
                d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n as i64 },
                bias: None,
                alpha: 1.0, beta: 0.0,
            };
            plan.run(s, Workspace::None, args).expect("run inside capture");
            Ok(())
        })
        .expect("capture failed");

    let exec = graph.instantiate().expect("instantiate");
    let tol = (k as f32) * 5e-3;
    for r in 0..2 {
        let zeros = vec![f16::ZERO; dev_d.len()];
        dev_d.copy_from_host(&zeros).unwrap();
        exec.launch(&stream).expect("graph launch");
        stream.synchronize().unwrap();
        let mut out = vec![f16::ZERO; dev_d.len()];
        dev_d.copy_to_host(&mut out).unwrap();
        let max_err = max_abs_err_f16(&out, &expected);
        assert!(max_err < tol, "f16 Rcr Identity replay #{r}: {max_err} > {tol}");
        println!("f16 Rcr Identity replay #{r}: max abs err {max_err} (tol {tol}) ✅");
    }
}

// ---- 2. Rrr layout ----

#[test]
#[ignore]
fn gemm_capture_replay_f16_rrr_identity() {
    init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = Stream::new(&ctx).unwrap();

    let (m, n, k) = (64i32, 64i32, 32i32);
    let host_a_f32: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 0.01).sin()).collect();
    let host_b_f32: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 0.013).cos()).collect();
    let mut expected = vec![0.0f32; (m * n) as usize];
    cpu_gemm_rrr_f32(m as usize, n as usize, k as usize, &host_a_f32, &host_b_f32, &mut expected);

    let host_a: Vec<f16> = host_a_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let host_b: Vec<f16> = host_b_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).unwrap();
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).unwrap();
    let mut dev_d: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, (m * n) as usize).unwrap();

    let desc = GemmDescriptor { m, n, k, layout: LayoutSku::Rrr, epilogue: EpilogueKind::Identity };
    let plan = GemmPlan::<f16>::select(&stream, &desc, PlanPreference::default()).unwrap();

    let graph = stream
        .capture(CaptureMode::ThreadLocal, |s| {
            let args = GemmArgs::<f16> {
                a: MatrixRef { data: dev_a.as_slice(), rows: m, cols: k, ld: k as i64 },
                b: MatrixRef { data: dev_b.as_slice(), rows: k, cols: n, ld: n as i64 },
                c: None,
                d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n as i64 },
                bias: None,
                alpha: 1.0, beta: 0.0,
            };
            plan.run(s, Workspace::None, args).expect("run inside capture");
            Ok(())
        })
        .expect("capture failed");

    let exec = graph.instantiate().expect("instantiate");
    let tol = (k as f32) * 5e-3;
    for r in 0..2 {
        let zeros = vec![f16::ZERO; dev_d.len()];
        dev_d.copy_from_host(&zeros).unwrap();
        exec.launch(&stream).expect("graph launch");
        stream.synchronize().unwrap();
        let mut out = vec![f16::ZERO; dev_d.len()];
        dev_d.copy_to_host(&mut out).unwrap();
        let max_err = max_abs_err_f16(&out, &expected);
        assert!(max_err < tol, "f16 Rrr Identity replay #{r}: {max_err} > {tol}");
        println!("f16 Rrr Identity replay #{r}: max abs err {max_err} (tol {tol}) ✅");
    }
}

// ---- 3. TF32 (f32 input via TF32 tensor cores) ----

#[test]
#[ignore]
fn gemm_capture_replay_tf32_rcr_identity() {
    init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = Stream::new(&ctx).unwrap();

    let (m, n, k) = (64i32, 64i32, 32i32);
    let host_a: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 0.01).sin()).collect();
    let host_b: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 0.013).cos()).collect();
    let mut expected = vec![0.0f32; (m * n) as usize];
    cpu_gemm_rcr_f32(m as usize, n as usize, k as usize, &host_a, &host_b, &mut expected);

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).unwrap();
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).unwrap();
    let mut dev_d: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, (m * n) as usize).unwrap();

    let desc = GemmDescriptor { m, n, k, layout: LayoutSku::Rcr, epilogue: EpilogueKind::Identity };
    let plan = GemmPlan::<f32>::select(&stream, &desc, PlanPreference::default()).unwrap();

    let graph = stream
        .capture(CaptureMode::ThreadLocal, |s| {
            let args = GemmArgs::<f32> {
                a: MatrixRef { data: dev_a.as_slice(), rows: m, cols: k, ld: k as i64 },
                b: MatrixRef { data: dev_b.as_slice(), rows: k, cols: n, ld: k as i64 },
                c: None,
                d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n as i64 },
                bias: None,
                alpha: 1.0, beta: 0.0,
            };
            plan.run(s, Workspace::None, args).expect("run inside capture");
            Ok(())
        })
        .expect("capture failed");

    let exec = graph.instantiate().expect("instantiate");
    // TF32 uses ~10-bit mantissa; relative error scaling is looser than f16.
    let tol = (k as f32) * 5e-4;
    for r in 0..2 {
        let zeros = vec![0.0f32; dev_d.len()];
        dev_d.copy_from_host(&zeros).unwrap();
        exec.launch(&stream).expect("graph launch");
        stream.synchronize().unwrap();
        let mut out = vec![0.0f32; dev_d.len()];
        dev_d.copy_to_host(&mut out).unwrap();
        let mut max_rel = 0.0f32;
        for (got, want) in out.iter().zip(expected.iter()) {
            let denom = want.abs().max(1.0);
            let rel = (got - want).abs() / denom;
            if rel > max_rel { max_rel = rel; }
        }
        assert!(max_rel < tol, "TF32 replay #{r}: max rel err {max_rel} > tol {tol}");
        println!("TF32 Rcr Identity replay #{r}: max rel err {max_rel} (tol {tol}) ✅");
    }
}

// ---- 4. Bias-fused single GEMM (covers the bias_ptr capture path) ----

#[test]
#[ignore]
fn gemm_capture_replay_f16_rcr_bias_gelu() {
    init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = Stream::new(&ctx).unwrap();

    let (m, n, k) = (64i32, 64i32, 32i32);
    let host_a_f32: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 0.01).sin()).collect();
    let host_b_f32: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 0.013).cos()).collect();
    let host_bias_f32: Vec<f32> = (0..n).map(|j| -0.5 + 0.1 * (j as f32 % 7.0)).collect();

    let mut expected = vec![0.0f32; (m * n) as usize];
    cpu_bias_gelu_rcr_f32(
        m as usize, n as usize, k as usize,
        &host_a_f32, &host_b_f32, &host_bias_f32,
        &mut expected,
    );

    let host_a: Vec<f16> = host_a_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let host_b: Vec<f16> = host_b_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let host_bias: Vec<f16> = host_bias_f32.iter().map(|&x| f16::from_f32(x)).collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).unwrap();
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).unwrap();
    let dev_bias = DeviceBuffer::from_slice(&ctx, &host_bias).unwrap();
    let mut dev_d: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, (m * n) as usize).unwrap();

    let desc = GemmDescriptor { m, n, k, layout: LayoutSku::Rcr, epilogue: EpilogueKind::BiasGelu };
    let plan = GemmPlan::<f16>::select(&stream, &desc, PlanPreference::default()).unwrap();

    let graph = stream
        .capture(CaptureMode::ThreadLocal, |s| {
            let args = GemmArgs::<f16> {
                a: MatrixRef { data: dev_a.as_slice(), rows: m, cols: k, ld: k as i64 },
                b: MatrixRef { data: dev_b.as_slice(), rows: k, cols: n, ld: k as i64 },
                c: None,
                d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n as i64 },
                bias: Some(VectorRef { data: dev_bias.as_slice(), len: n, stride: 1 }),
                alpha: 1.0, beta: 0.0,
            };
            plan.run(s, Workspace::None, args).expect("run inside capture");
            Ok(())
        })
        .expect("capture failed — likely the bias kernel did an implicit host sync");

    let exec = graph.instantiate().expect("instantiate");
    let tol = (k as f32) * 5e-3;
    for r in 0..2 {
        let zeros = vec![f16::ZERO; dev_d.len()];
        dev_d.copy_from_host(&zeros).unwrap();
        exec.launch(&stream).expect("graph launch");
        stream.synchronize().unwrap();
        let mut out = vec![f16::ZERO; dev_d.len()];
        dev_d.copy_to_host(&mut out).unwrap();
        let max_err = max_abs_err_f16(&out, &expected);
        assert!(max_err < tol, "f16 BiasGelu replay #{r}: {max_err} > {tol}");
        println!("f16 Rcr BiasGelu replay #{r}: max abs err {max_err} (tol {tol}) ✅");
    }
}

// ---- 5. Batched GEMM (uniform shape) ----

#[test]
#[ignore]
fn batched_gemm_capture_replay_f16_rcr() {
    init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = Stream::new(&ctx).unwrap();

    let (m, n, k) = (64i32, 64i32, 32i32);
    let batch_count: i32 = 4;
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
            host_a_f32[batch * elements_a + i] = (((batch * 1000 + i) as f32) * 0.01).sin();
        }
        for i in 0..elements_b {
            host_b_f32[batch * elements_b + i] = (((batch * 1000 + i) as f32) * 0.013).cos();
        }
    }
    let mut expected = vec![0.0f32; total_d];
    for batch in 0..batch_count as usize {
        let a_off = batch * elements_a;
        let b_off = batch * elements_b;
        let d_off = batch * elements_d;
        cpu_gemm_rcr_f32(
            m as usize, n as usize, k as usize,
            &host_a_f32[a_off..a_off + elements_a],
            &host_b_f32[b_off..b_off + elements_b],
            &mut expected[d_off..d_off + elements_d],
        );
    }

    let host_a: Vec<f16> = host_a_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let host_b: Vec<f16> = host_b_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).unwrap();
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).unwrap();
    let mut dev_d: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, total_d).unwrap();

    let desc = BatchedGemmDescriptor {
        m, n, k,
        batch_count,
        layout: LayoutSku::Rcr,
        epilogue: EpilogueKind::Identity,
    };
    let plan = BatchedGemmPlan::<f16>::select(&stream, &desc, PlanPreference::default()).unwrap();

    let graph = stream
        .capture(CaptureMode::ThreadLocal, |s| {
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
            plan.run(s, Workspace::None, args).expect("run inside capture");
            Ok(())
        })
        .expect("capture failed");

    let exec = graph.instantiate().expect("instantiate");
    let tol = (k as f32) * 5e-3;
    for r in 0..2 {
        let zeros = vec![f16::ZERO; dev_d.len()];
        dev_d.copy_from_host(&zeros).unwrap();
        exec.launch(&stream).expect("graph launch");
        stream.synchronize().unwrap();
        let mut out = vec![f16::ZERO; dev_d.len()];
        dev_d.copy_to_host(&mut out).unwrap();
        let max_err = max_abs_err_f16(&out, &expected);
        assert!(max_err < tol, "f16 batched replay #{r}: {max_err} > {tol}");
        println!("f16 batched Rcr Identity replay #{r}: max abs err {max_err} (tol {tol}) ✅");
    }
}

// ---- Helpers ----

fn max_abs_err_f16(out: &[f16], expected: &[f32]) -> f32 {
    let mut max_err = 0.0f32;
    for (got, want) in out.iter().zip(expected.iter()) {
        let err = (got.to_f32() - want).abs();
        if err > max_err {
            max_err = err;
        }
    }
    max_err
}
