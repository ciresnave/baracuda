//! Phase-30 cuBLAS backend smoke test for `GemmPlan`.
//!
//! Validates the cuBLAS-routing fast-path against a host CPU reference
//! for f16 / bf16 / f32, both via the auto-dispatch heuristic and via
//! the explicit `PlanPreference::prefer_backend = Some(Cublas)` override.
//!
//! Marked `#[ignore]` so `cargo test --workspace` is GPU-free by default.
//! Run on a CUDA + NVIDIA-GPU machine:
//!
//! ```text
//! cargo test -p baracuda-cutlass --release --test cublas_backend_smoke -- --ignored
//! ```

use baracuda_cutlass::{
    BackendKind, EpilogueKind, GemmArgs, GemmDescriptor, GemmPlan, LayoutSku, MatrixMut, MatrixRef,
    PlanPreference, Workspace,
};
use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use half::{bf16, f16};

/// Host CPU reference: D = alpha * (A @ B) + beta * C in f32.
/// A row-major [m, k], lda=k. B column-major [k, n], ldb=k (RCR).
fn cpu_gemm_rcr_f32(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    d: &mut [f32],
) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let a_val = a[i * k + kk];
                let b_val = b[j * k + kk]; // col-major B[kk, j] = b[j*k + kk]
                acc += a_val * b_val;
            }
            d[i * n + j] = acc;
        }
    }
}

fn run_one_f16_cublas(m: i32, n: i32, k: i32, force_cublas: bool) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let host_a_f32: Vec<f32> = (0..(m * k))
        .map(|i| ((i as f32) * 0.01).sin())
        .collect();
    let host_b_f32: Vec<f32> = (0..(k * n))
        .map(|i| ((i as f32) * 0.013).cos())
        .collect();
    let mut host_d_ref = vec![0.0f32; (m * n) as usize];
    cpu_gemm_rcr_f32(
        m as usize,
        n as usize,
        k as usize,
        &host_a_f32,
        &host_b_f32,
        &mut host_d_ref,
    );

    let host_a: Vec<f16> = host_a_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let host_b: Vec<f16> = host_b_f32.iter().map(|&x| f16::from_f32(x)).collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload A");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload B");
    let mut dev_d: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (m * n) as usize).expect("alloc D");

    let desc = GemmDescriptor {
        m,
        n,
        k,
        layout: LayoutSku::Rcr,
        epilogue: EpilogueKind::Identity,
    };
    let pref = if force_cublas {
        PlanPreference {
            prefer_backend: Some(BackendKind::Cublas),
            ..PlanPreference::default()
        }
    } else {
        PlanPreference::default()
    };
    let plan = GemmPlan::<f16>::select(&stream, &desc, pref).expect("plan select");

    let chosen = plan.backend();
    if force_cublas {
        assert_eq!(
            chosen,
            BackendKind::Cublas,
            "force-Cublas should pick cuBLAS backend"
        );
    }
    // At 2 <= m < 128 the auto heuristic picks cuBLAS for f16; assert.
    // M=1 stays on CUTLASS by the Phase-30 heuristic (cuBLAS's T·N
    // path is slow for pure GEMV at the bench's typical K=N).
    if !force_cublas && m >= 2 && m < 128 {
        assert_eq!(
            chosen,
            BackendKind::Cublas,
            "auto heuristic should pick cuBLAS for f16 2 <= M < 128"
        );
    }
    if !force_cublas && (m == 1 || m >= 128) {
        assert_eq!(
            chosen,
            BackendKind::Cutlass,
            "auto heuristic should pick CUTLASS at M=1 or M>=128"
        );
    }

    let args = GemmArgs::<f16> {
        a: MatrixRef {
            data: dev_a.as_slice(),
            rows: m,
            cols: k,
            ld: k as i64,
        },
        b: MatrixRef {
            data: dev_b.as_slice(),
            rows: k,
            cols: n,
            ld: k as i64,
        },
        c: None,
        d: MatrixMut {
            data: dev_d.as_slice_mut(),
            rows: m,
            cols: n,
            ld: n as i64,
        },
        bias: None,
        alpha: 1.0,
        beta: 0.0,
    };
    plan.can_implement(&args).expect("can_implement");
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut host_d_out = vec![f16::ZERO; (m * n) as usize];
    dev_d
        .copy_to_host(&mut host_d_out)
        .expect("download D");

    let mut max_err = 0.0f32;
    for (got, want) in host_d_out.iter().zip(host_d_ref.iter()) {
        let err = (got.to_f32() - want).abs();
        if err > max_err {
            max_err = err;
        }
    }
    let tol = (k as f32) * 5e-3;
    assert!(
        max_err < tol,
        "f16 cuBLAS GEMM ({m}x{n}x{k}, backend={chosen:?}): max abs err {max_err} exceeded tolerance {tol}"
    );
    println!(
        "f16 GEMM ({m}x{n}x{k}, backend={chosen:?}): max abs err {max_err} (tol {tol}) ✅"
    );
}

fn run_one_bf16_cublas(m: i32, n: i32, k: i32, force_cublas: bool) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let host_a_f32: Vec<f32> = (0..(m * k))
        .map(|i| ((i as f32) * 0.01).sin())
        .collect();
    let host_b_f32: Vec<f32> = (0..(k * n))
        .map(|i| ((i as f32) * 0.013).cos())
        .collect();
    let mut host_d_ref = vec![0.0f32; (m * n) as usize];
    cpu_gemm_rcr_f32(
        m as usize,
        n as usize,
        k as usize,
        &host_a_f32,
        &host_b_f32,
        &mut host_d_ref,
    );

    let host_a: Vec<bf16> = host_a_f32.iter().map(|&x| bf16::from_f32(x)).collect();
    let host_b: Vec<bf16> = host_b_f32.iter().map(|&x| bf16::from_f32(x)).collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload A");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload B");
    let mut dev_d: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, (m * n) as usize).expect("alloc D");

    let desc = GemmDescriptor {
        m,
        n,
        k,
        layout: LayoutSku::Rcr,
        epilogue: EpilogueKind::Identity,
    };
    let pref = if force_cublas {
        PlanPreference {
            prefer_backend: Some(BackendKind::Cublas),
            ..PlanPreference::default()
        }
    } else {
        PlanPreference::default()
    };
    let plan = GemmPlan::<bf16>::select(&stream, &desc, pref).expect("plan select");
    let chosen = plan.backend();
    if force_cublas {
        assert_eq!(chosen, BackendKind::Cublas);
    }

    let args = GemmArgs::<bf16> {
        a: MatrixRef {
            data: dev_a.as_slice(),
            rows: m,
            cols: k,
            ld: k as i64,
        },
        b: MatrixRef {
            data: dev_b.as_slice(),
            rows: k,
            cols: n,
            ld: k as i64,
        },
        c: None,
        d: MatrixMut {
            data: dev_d.as_slice_mut(),
            rows: m,
            cols: n,
            ld: n as i64,
        },
        bias: None,
        alpha: 1.0,
        beta: 0.0,
    };
    plan.can_implement(&args).expect("can_implement");
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut host_d_out = vec![bf16::ZERO; (m * n) as usize];
    dev_d
        .copy_to_host(&mut host_d_out)
        .expect("download D");

    let mut max_err = 0.0f32;
    for (got, want) in host_d_out.iter().zip(host_d_ref.iter()) {
        let err = (got.to_f32() - want).abs();
        if err > max_err {
            max_err = err;
        }
    }
    // bf16 has only 7 bits of mantissa — looser tolerance than f16.
    let tol = (k as f32) * 2e-2;
    assert!(
        max_err < tol,
        "bf16 cuBLAS GEMM ({m}x{n}x{k}, backend={chosen:?}): max abs err {max_err} exceeded tolerance {tol}"
    );
    println!(
        "bf16 GEMM ({m}x{n}x{k}, backend={chosen:?}): max abs err {max_err} (tol {tol}) ✅"
    );
}

fn run_one_f32_cublas_forced(m: i32, n: i32, k: i32) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let host_a: Vec<f32> = (0..(m * k))
        .map(|i| ((i as f32) * 0.01).sin())
        .collect();
    let host_b: Vec<f32> = (0..(k * n))
        .map(|i| ((i as f32) * 0.013).cos())
        .collect();
    let mut host_d_ref = vec![0.0f32; (m * n) as usize];
    cpu_gemm_rcr_f32(
        m as usize,
        n as usize,
        k as usize,
        &host_a,
        &host_b,
        &mut host_d_ref,
    );

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload A");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload B");
    let mut dev_d: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (m * n) as usize).expect("alloc D");

    let desc = GemmDescriptor {
        m,
        n,
        k,
        layout: LayoutSku::Rcr,
        epilogue: EpilogueKind::Identity,
    };
    let pref = PlanPreference {
        prefer_backend: Some(BackendKind::Cublas),
        ..PlanPreference::default()
    };
    let plan = GemmPlan::<f32>::select(&stream, &desc, pref).expect("plan select");
    assert_eq!(
        plan.backend(),
        BackendKind::Cublas,
        "f32 force-Cublas must pick cuBLAS"
    );

    let args = GemmArgs::<f32> {
        a: MatrixRef {
            data: dev_a.as_slice(),
            rows: m,
            cols: k,
            ld: k as i64,
        },
        b: MatrixRef {
            data: dev_b.as_slice(),
            rows: k,
            cols: n,
            ld: k as i64,
        },
        c: None,
        d: MatrixMut {
            data: dev_d.as_slice_mut(),
            rows: m,
            cols: n,
            ld: n as i64,
        },
        bias: None,
        alpha: 1.0,
        beta: 0.0,
    };
    plan.can_implement(&args).expect("can_implement");
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut host_d_out = vec![0.0f32; (m * n) as usize];
    dev_d
        .copy_to_host(&mut host_d_out)
        .expect("download D");

    let mut max_err = 0.0f32;
    for (got, want) in host_d_out.iter().zip(host_d_ref.iter()) {
        let err = (got - want).abs();
        if err > max_err {
            max_err = err;
        }
    }
    // f32 with f32 accumulator on tensor-core / GemmEx — tight tol.
    let tol = (k as f32) * 1e-4;
    assert!(
        max_err < tol,
        "f32 cuBLAS GEMM ({m}x{n}x{k}): max abs err {max_err} exceeded tolerance {tol}"
    );
    println!("f32 GEMM ({m}x{n}x{k}, backend=Cublas): max abs err {max_err} (tol {tol}) ✅");
}

// ============================================================================
// Tests
// ============================================================================

/// Auto-dispatch: f16 at M=1 stays on CUTLASS (the Phase-30 heuristic
/// excludes pure GEMV-shape because cuBLAS's `transa=T` path is slow
/// at K=N=4096 — see `should_use_cublas_for_fp` rustdoc).
#[test]
#[ignore]
fn f16_auto_dispatch_m1_picks_cutlass() {
    run_one_f16_cublas(1, 128, 256, /* force_cublas */ false);
}

/// Auto-dispatch: f16 at M=8 lands on cuBLAS (in the 2 <= M < 128
/// decode-batch window).
#[test]
#[ignore]
fn f16_auto_dispatch_m8_picks_cublas() {
    run_one_f16_cublas(8, 128, 256, /* force_cublas */ false);
}

/// Force cuBLAS at f16 / mid-M (still in the cuBLAS preference window).
#[test]
#[ignore]
fn f16_force_cublas_m32() {
    run_one_f16_cublas(32, 256, 256, /* force_cublas */ true);
}

/// Force cuBLAS at f16 / M=1 (decode GEMV). Verifies the path is
/// correct end-to-end — perf may be worse than CUTLASS at this shape
/// but the answer must match.
#[test]
#[ignore]
fn f16_force_cublas_m1() {
    run_one_f16_cublas(1, 128, 256, /* force_cublas */ true);
}

/// Force cuBLAS at f16 / large-M (heuristic would pick CUTLASS, override
/// wins).
#[test]
#[ignore]
fn f16_force_cublas_m256() {
    run_one_f16_cublas(256, 256, 256, /* force_cublas */ true);
}

/// Force cuBLAS at bf16.
#[test]
#[ignore]
fn bf16_force_cublas_m8() {
    run_one_bf16_cublas(8, 128, 256, /* force_cublas */ true);
}

/// Force cuBLAS at f32 (CUTLASS default, override picks cuBLAS).
#[test]
#[ignore]
fn f32_force_cublas_m16() {
    run_one_f32_cublas_forced(16, 128, 256);
}

/// Auto-dispatch at large M should stay on CUTLASS for f16 (the
/// CUTLASS path is preferred at prefill scale).
#[test]
#[ignore]
fn f16_auto_dispatch_large_m_picks_cutlass() {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let desc = GemmDescriptor {
        m: 256,
        n: 256,
        k: 256,
        layout: LayoutSku::Rcr,
        epilogue: EpilogueKind::Identity,
    };
    let plan = GemmPlan::<f16>::select(&stream, &desc, PlanPreference::default())
        .expect("plan select");
    assert_eq!(
        plan.backend(),
        BackendKind::Cutlass,
        "auto heuristic should pick CUTLASS at M=256"
    );
}

/// Force-Cublas with a Bias* epilogue must error — cuBLAS-classic has
/// no fused-bias-activation.
#[test]
#[ignore]
fn force_cublas_with_bias_epilogue_errors() {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let desc = GemmDescriptor {
        m: 32,
        n: 128,
        k: 256,
        layout: LayoutSku::Rcr,
        epilogue: EpilogueKind::BiasRelu,
    };
    let pref = PlanPreference {
        prefer_backend: Some(BackendKind::Cublas),
        ..PlanPreference::default()
    };
    let res = GemmPlan::<f16>::select(&stream, &desc, pref);
    assert!(
        res.is_err(),
        "force-Cublas + BiasRelu should error, got Ok"
    );
}
