//! Real-GPU smoke test: f16 + bf16 GEMM RCR sm80 against a CPU reference.
//!
//! Marked `#[ignore]` so `cargo test --workspace` is GPU-free by default.
//! Run on a machine with CUDA + an NVIDIA GPU:
//!
//! ```text
//! cargo test -p baracuda-cutlass --release -- --ignored
//! ```

use baracuda_cutlass::{
    EpilogueKind, GemmArgs, GemmDescriptor, GemmPlan, LayoutSku, MatrixMut, MatrixRef,
    PlanPreference, Workspace,
};
use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use half::{bf16, f16};

/// Compute D = alpha * (A @ B) + beta * C on the host in f32.
///
/// A: row-major [m, k], stride lda
/// B: column-major [k, n], stride ldb
/// C: optional row-major [m, n], stride ldc
/// D: row-major [m, n] output (caller-provided, length m*n)
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
                let a_val = a[i * lda + kk];           // row-major A[i, kk]
                let b_val = b[j * ldb + kk];           // column-major B[kk, j]
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

fn run_one_f16(m: i32, n: i32, k: i32) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    // Host reference data — small bounded values so f16 rounding stays sane.
    let host_a_f32: Vec<f32> = (0..(m * k))
        .map(|i| ((i as f32) * 0.01).sin())
        .collect();
    let host_b_f32: Vec<f32> = (0..(k * n))
        .map(|i| ((i as f32) * 0.013).cos())
        .collect();
    let mut host_d_ref = vec![0.0f32; (m * n) as usize];
    cpu_gemm_rcr(
        m as usize,
        n as usize,
        k as usize,
        &host_a_f32,
        k as usize,
        &host_b_f32,
        k as usize,
        None,
        n as usize,
        1.0,
        0.0,
        &mut host_d_ref,
        n as usize,
    );

    // Convert to f16 and upload.
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
    let plan =
        GemmPlan::<f16>::select(&stream, &desc, PlanPreference::default()).expect("plan select");
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
        alpha: 1.0,
        beta: 0.0,
    };

    plan.can_implement(&args).expect("can_implement");
    plan.run(&stream, Workspace::None, args).expect("run");

    let mut host_d_out_f16 = vec![f16::ZERO; (m * n) as usize];
    dev_d
        .copy_to_host(&mut host_d_out_f16)
        .expect("download D");

    let mut max_err = 0.0f32;
    for (got, want) in host_d_out_f16.iter().zip(host_d_ref.iter()) {
        let err = (got.to_f32() - want).abs();
        if err > max_err {
            max_err = err;
        }
    }
    // f16 has ~3 decimal digits of precision; our values are O(K)
    // accumulated from O(1)-ish products, so absolute error scales with K.
    let tol = (k as f32) * 5e-3;
    assert!(
        max_err < tol,
        "f16 GEMM ({m}x{n}x{k}): max abs err {max_err} exceeded tolerance {tol}"
    );
    println!("f16 RCR sm80 GEMM ({m}x{n}x{k}): max abs err {max_err} (tol {tol}) ✅");
}

fn run_one_bf16(m: i32, n: i32, k: i32) {
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
    cpu_gemm_rcr(
        m as usize,
        n as usize,
        k as usize,
        &host_a_f32,
        k as usize,
        &host_b_f32,
        k as usize,
        None,
        n as usize,
        1.0,
        0.0,
        &mut host_d_ref,
        n as usize,
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
    let plan =
        GemmPlan::<bf16>::select(&stream, &desc, PlanPreference::default()).expect("plan select");
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
        alpha: 1.0,
        beta: 0.0,
    };

    plan.can_implement(&args).expect("can_implement");
    plan.run(&stream, Workspace::None, args).expect("run");

    let mut host_d_out = vec![bf16::ZERO; (m * n) as usize];
    dev_d.copy_to_host(&mut host_d_out).expect("download D");

    let mut max_err = 0.0f32;
    for (got, want) in host_d_out.iter().zip(host_d_ref.iter()) {
        let err = (got.to_f32() - want).abs();
        if err > max_err {
            max_err = err;
        }
    }
    // bf16 has ~3 decimal digits; same scaling as f16.
    let tol = (k as f32) * 5e-3;
    assert!(
        max_err < tol,
        "bf16 GEMM ({m}x{n}x{k}): max abs err {max_err} exceeded tolerance {tol}"
    );
    println!("bf16 RCR sm80 GEMM ({m}x{n}x{k}): max abs err {max_err} (tol {tol}) ✅");
}

#[test]
#[ignore]
fn f16_gemm_rcr_sm80_128_128_64() {
    run_one_f16(128, 128, 64);
}

#[test]
#[ignore]
fn f16_gemm_rcr_sm80_64_64_32() {
    run_one_f16(64, 64, 32);
}

#[test]
#[ignore]
fn bf16_gemm_rcr_sm80_128_128_64() {
    run_one_bf16(128, 128, 64);
}

#[test]
#[ignore]
fn bf16_gemm_rcr_sm80_64_64_32() {
    run_one_bf16(64, 64, 32);
}

/// `can_implement` should accept a well-formed problem.
#[test]
#[ignore]
fn can_implement_accepts_aligned_problem() {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let m = 128i32;
    let n = 128i32;
    let k = 64i32;

    let dev_a: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (m * k) as usize).expect("alloc A");
    let dev_b: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (k * n) as usize).expect("alloc B");
    let mut dev_d: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (m * n) as usize).expect("alloc D");

    let desc = GemmDescriptor {
        m,
        n,
        k,
        layout: LayoutSku::Rcr,
        epilogue: EpilogueKind::Identity,
    };
    let plan =
        GemmPlan::<f16>::select(&stream, &desc, PlanPreference::default()).expect("plan select");
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
        alpha: 1.0,
        beta: 0.0,
    };
    plan.can_implement(&args)
        .expect("aligned 128x128x64 RCR f16 problem should be implementable");
    println!("can_implement accepts aligned 128x128x64 ✅");
}

/// `can_implement` should reject a problem whose `K` doesn't match the
/// kernel's element-per-access requirement.
///
/// The standard Ampere f16/bf16 tile uses 8-element-per-access loads,
/// so K must be a multiple of 8 for the inner loop's vectorized access
/// to land on aligned addresses. K = 7 should be rejected by CUTLASS's
/// `Gemm::can_implement` without launching a kernel.
#[test]
#[ignore]
fn can_implement_rejects_misaligned_k() {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let m = 128i32;
    let n = 128i32;
    let k = 7i32; // not a multiple of 8 — CUTLASS will refuse

    let dev_a: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (m * k) as usize).expect("alloc A");
    let dev_b: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (k * n) as usize).expect("alloc B");
    let mut dev_d: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (m * n) as usize).expect("alloc D");

    let desc = GemmDescriptor {
        m,
        n,
        k,
        layout: LayoutSku::Rcr,
        epilogue: EpilogueKind::Identity,
    };
    let plan =
        GemmPlan::<f16>::select(&stream, &desc, PlanPreference::default()).expect("plan select");
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
        alpha: 1.0,
        beta: 0.0,
    };

    let result = plan.can_implement(&args);
    assert!(
        matches!(
            result,
            Err(baracuda_cutlass::Error::MisalignedOperand)
                | Err(baracuda_cutlass::Error::Unsupported(_))
                | Err(baracuda_cutlass::Error::InvalidProblem(_))
        ),
        "K=7 should be rejected by CUTLASS can_implement; got {result:?}"
    );
    println!("can_implement rejects K=7 misalignment with: {result:?} ✅");
}

/// Regression test for the Fuel-team-flagged null-C contract bug.
///
/// Pre-fix, `args.c = None` with `args.beta != 0` silently produced
/// `D += alpha*AB` (because the kernel substitutes D for the C operand).
/// The fix forces `beta = 0` at the safe layer when `c` is `None`.
///
/// We pre-fill D with non-zero "garbage", run with `c = None` and
/// `beta = 7.0` (deliberately large and nonzero), and assert D contains
/// only `alpha * A @ B` — i.e., the prior contents were *overwritten*,
/// not accumulated.
#[test]
#[ignore]
fn null_c_with_nonzero_beta_overwrites_d() {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let m = 64i32;
    let n = 64i32;
    let k = 32i32;

    let host_a_f32: Vec<f32> = (0..(m * k))
        .map(|i| ((i as f32) * 0.01).sin())
        .collect();
    let host_b_f32: Vec<f32> = (0..(k * n))
        .map(|i| ((i as f32) * 0.013).cos())
        .collect();

    let mut host_d_ref = vec![0.0f32; (m * n) as usize];
    cpu_gemm_rcr(
        m as usize,
        n as usize,
        k as usize,
        &host_a_f32,
        k as usize,
        &host_b_f32,
        k as usize,
        None,
        n as usize,
        1.0,
        0.0, // CPU reference: alpha*AB only, no accumulation
        &mut host_d_ref,
        n as usize,
    );

    let host_a: Vec<f16> = host_a_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let host_b: Vec<f16> = host_b_f32.iter().map(|&x| f16::from_f32(x)).collect();
    // Pre-fill D with a large recognizable garbage value. If the bug
    // returned and beta=7.0 were honored, the result would be
    // alpha*AB + 7.0*42.0 = alpha*AB + 294.0 — way outside our tolerance.
    let host_d_garbage: Vec<f16> = vec![f16::from_f32(42.0); (m * n) as usize];

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload A");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload B");
    let mut dev_d = DeviceBuffer::from_slice(&ctx, &host_d_garbage).expect("upload D garbage");

    let desc = GemmDescriptor {
        m,
        n,
        k,
        layout: LayoutSku::Rcr,
        epilogue: EpilogueKind::Identity,
    };
    let plan =
        GemmPlan::<f16>::select(&stream, &desc, PlanPreference::default()).expect("plan select");
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
        alpha: 1.0,
        beta: 7.0, // intentionally nonzero — must be ignored when c = None
    };
    plan.run(&stream, Workspace::None, args).expect("run");

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
        "null-C regression: D was not overwritten; max abs err {max_err} exceeded tol {tol} \
         (likely the pre-existing 42.0 garbage was being accumulated via beta=7.0)"
    );
    println!(
        "null-C beta override regression: max abs err {max_err} (tol {tol}) ✅"
    );
}
