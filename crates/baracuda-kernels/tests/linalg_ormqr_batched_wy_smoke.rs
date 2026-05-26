//! Real-GPU smoke test for `BatchedOrmqrWyPlan` (WY-blocked bespoke
//! kernel + cuBLAS strided-batched GEMM, Milestone 6.17).
//!
//! Cross-checks against the **existing** reflector-by-reflector
//! [`BatchedOrmqrPlan`] (Milestone 6.14). The two plans implement the
//! same math (apply Householder-encoded Q from a `BatchedQrPlan` packed
//! output) at different rates — WY uses GEMM, reflector uses GEMV.
//!
//! Fixture: `B=2, M=64, N=32, K=64` so `num_blocks = (64 + 32 - 1) / 32 = 2`
//! and the WY iteration exercises the multi-block path (the per-block
//! iteration direction differs between op = N and op = T).
//!
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BatchedOrmqrArgs, BatchedOrmqrDescriptor, BatchedOrmqrOp, BatchedOrmqrPlan,
    BatchedOrmqrSide, BatchedOrmqrWyArgs, BatchedOrmqrWyDescriptor, BatchedOrmqrWyPlan,
    BatchedQrArgs, BatchedQrDescriptor, BatchedQrPlan, Complex32, Complex64, ElementKind,
    PlanPreference, TensorMut, TensorRef, Workspace,
};

#[allow(dead_code)]
fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// Deterministic pseudo-random matrix builder.
fn build_matrix_f32(b: usize, m: usize, n: usize, seed: u32) -> Vec<f32> {
    let mut a = vec![0f32; b * m * n];
    let mut s = seed.wrapping_mul(0x9E37_79B1);
    for cell in a.iter_mut() {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
        *cell = ((s >> 8) as f32 / (1u32 << 24) as f32) - 0.5;
    }
    a
}

fn build_matrix_f64(b: usize, m: usize, n: usize, seed: u32) -> Vec<f64> {
    build_matrix_f32(b, m, n, seed)
        .into_iter()
        .map(|v| v as f64)
        .collect()
}

fn fmt_op(op: BatchedOrmqrOp) -> &'static str {
    match op {
        BatchedOrmqrOp::N => "N",
        BatchedOrmqrOp::T => "T",
        BatchedOrmqrOp::C => "C",
    }
}

/// Run `BatchedQrPlan` on `[B, M, N_a]` f32 input. The post-`geqrf`
/// packed-A and tau are uploaded once and reused by both the WY and
/// reflector plans.
fn run_batched_qr_f32(
    ctx: &Context,
    stream: &Stream,
    a_host: &[f32],
    b: i32,
    m: i32,
    n: i32,
) -> (Vec<f32>, Vec<f32>) {
    let k = m.min(n);
    let mut dev_a = DeviceBuffer::from_slice(ctx, a_host).expect("upload a");
    let mut dev_tau: DeviceBuffer<f32> =
        DeviceBuffer::zeros(ctx, (b * k) as usize).expect("alloc tau");
    let desc = BatchedQrDescriptor {
        m,
        n,
        batch_size: b,
        element: ElementKind::F32,
    };
    let plan = BatchedQrPlan::<f32>::select(stream, &desc, PlanPreference::default())
        .expect("select BatchedQrPlan<f32>");
    let ws_bytes = plan.query_workspace_size(stream).expect("ws query");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(ctx, ws_bytes).expect("alloc ws");
    let a_shape = [b, m, n];
    let tau_shape = [b, k];
    let args = BatchedQrArgs::<f32> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: a_shape,
            stride: contiguous_stride(a_shape),
        },
        tau: TensorMut {
            data: dev_tau.as_slice_mut(),
            shape: tau_shape,
            stride: contiguous_stride(tau_shape),
        },
    };
    plan.run(stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run batched QR f32");
    stream.synchronize().expect("sync");
    let mut a_post = vec![0f32; (b * m * n) as usize];
    dev_a.copy_to_host(&mut a_post).expect("dl a-post");
    let mut tau_post = vec![0f32; (b * k) as usize];
    dev_tau.copy_to_host(&mut tau_post).expect("dl tau-post");
    (a_post, tau_post)
}

fn run_batched_qr_f64(
    ctx: &Context,
    stream: &Stream,
    a_host: &[f64],
    b: i32,
    m: i32,
    n: i32,
) -> (Vec<f64>, Vec<f64>) {
    let k = m.min(n);
    let mut dev_a = DeviceBuffer::from_slice(ctx, a_host).expect("upload a");
    let mut dev_tau: DeviceBuffer<f64> =
        DeviceBuffer::zeros(ctx, (b * k) as usize).expect("alloc tau");
    let desc = BatchedQrDescriptor {
        m,
        n,
        batch_size: b,
        element: ElementKind::F64,
    };
    let plan = BatchedQrPlan::<f64>::select(stream, &desc, PlanPreference::default())
        .expect("select BatchedQrPlan<f64>");
    let ws_bytes = plan.query_workspace_size(stream).expect("ws query");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(ctx, ws_bytes).expect("alloc ws");
    let args = BatchedQrArgs::<f64> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: [b, m, n],
            stride: contiguous_stride([b, m, n]),
        },
        tau: TensorMut {
            data: dev_tau.as_slice_mut(),
            shape: [b, k],
            stride: contiguous_stride([b, k]),
        },
    };
    plan.run(stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run batched QR f64");
    stream.synchronize().expect("sync");
    let mut a_post = vec![0f64; (b * m * n) as usize];
    dev_a.copy_to_host(&mut a_post).expect("dl a-post");
    let mut tau_post = vec![0f64; (b * k) as usize];
    dev_tau.copy_to_host(&mut tau_post).expect("dl tau-post");
    (a_post, tau_post)
}

/// Reference: reflector-by-reflector `BatchedOrmqrPlan` (Milestone 6.14).
fn ref_ormqr_f32(
    ctx: &Context,
    stream: &Stream,
    a_packed: &[f32],
    tau: &[f32],
    c_init: &[f32],
    b: i32,
    m: i32,
    n: i32,
    k: i32,
    op: BatchedOrmqrOp,
) -> Vec<f32> {
    let dev_a = DeviceBuffer::from_slice(ctx, a_packed).expect("upload a");
    let dev_tau = DeviceBuffer::from_slice(ctx, tau).expect("upload tau");
    let mut dev_c = DeviceBuffer::from_slice(ctx, c_init).expect("upload c");
    let desc = BatchedOrmqrDescriptor {
        m,
        n,
        k,
        batch_size: b,
        side: BatchedOrmqrSide::Left,
        op,
        element: ElementKind::F32,
    };
    let plan = BatchedOrmqrPlan::<f32>::select(stream, &desc, PlanPreference::default())
        .expect("select BatchedOrmqrPlan<f32>");
    let a_shape = [b, m, k];
    let tau_shape = [b, k];
    let c_shape = [b, m, n];
    let args = BatchedOrmqrArgs::<f32> {
        a_packed: TensorRef {
            data: dev_a.as_slice(),
            shape: a_shape,
            stride: contiguous_stride(a_shape),
        },
        tau: TensorRef {
            data: dev_tau.as_slice(),
            shape: tau_shape,
            stride: contiguous_stride(tau_shape),
        },
        c: TensorMut {
            data: dev_c.as_slice_mut(),
            shape: c_shape,
            stride: contiguous_stride(c_shape),
        },
    };
    plan.run(stream, Workspace::None, args)
        .expect("run reflector ormqr f32");
    stream.synchronize().expect("sync");
    let mut c_post = vec![0f32; (b * m * n) as usize];
    dev_c.copy_to_host(&mut c_post).expect("dl c-post");
    c_post
}

fn ref_ormqr_f64(
    ctx: &Context,
    stream: &Stream,
    a_packed: &[f64],
    tau: &[f64],
    c_init: &[f64],
    b: i32,
    m: i32,
    n: i32,
    k: i32,
    op: BatchedOrmqrOp,
) -> Vec<f64> {
    let dev_a = DeviceBuffer::from_slice(ctx, a_packed).expect("upload a");
    let dev_tau = DeviceBuffer::from_slice(ctx, tau).expect("upload tau");
    let mut dev_c = DeviceBuffer::from_slice(ctx, c_init).expect("upload c");
    let desc = BatchedOrmqrDescriptor {
        m,
        n,
        k,
        batch_size: b,
        side: BatchedOrmqrSide::Left,
        op,
        element: ElementKind::F64,
    };
    let plan = BatchedOrmqrPlan::<f64>::select(stream, &desc, PlanPreference::default())
        .expect("select BatchedOrmqrPlan<f64>");
    let a_shape = [b, m, k];
    let tau_shape = [b, k];
    let c_shape = [b, m, n];
    let args = BatchedOrmqrArgs::<f64> {
        a_packed: TensorRef {
            data: dev_a.as_slice(),
            shape: a_shape,
            stride: contiguous_stride(a_shape),
        },
        tau: TensorRef {
            data: dev_tau.as_slice(),
            shape: tau_shape,
            stride: contiguous_stride(tau_shape),
        },
        c: TensorMut {
            data: dev_c.as_slice_mut(),
            shape: c_shape,
            stride: contiguous_stride(c_shape),
        },
    };
    plan.run(stream, Workspace::None, args)
        .expect("run reflector ormqr f64");
    stream.synchronize().expect("sync");
    let mut c_post = vec![0f64; (b * m * n) as usize];
    dev_c.copy_to_host(&mut c_post).expect("dl c-post");
    c_post
}

/// WY-blocked plan under test.
fn wy_ormqr_f32(
    ctx: &Context,
    stream: &Stream,
    a_packed: &[f32],
    tau: &[f32],
    c_init: &[f32],
    b: i32,
    m: i32,
    n: i32,
    k: i32,
    op: BatchedOrmqrOp,
) -> Vec<f32> {
    let mut dev_a = DeviceBuffer::from_slice(ctx, a_packed).expect("upload a");
    let mut dev_tau = DeviceBuffer::from_slice(ctx, tau).expect("upload tau");
    let mut dev_c = DeviceBuffer::from_slice(ctx, c_init).expect("upload c");
    let desc = BatchedOrmqrWyDescriptor {
        m,
        n,
        k,
        batch_size: b,
        side: BatchedOrmqrSide::Left,
        op,
        element: ElementKind::F32,
    };
    let plan = BatchedOrmqrWyPlan::<f32>::select(stream, &desc, PlanPreference::default())
        .expect("select BatchedOrmqrWyPlan<f32>");
    let ws_bytes = plan.query_workspace_size(stream).expect("ws query");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(ctx, ws_bytes).expect("alloc ws");
    let a_shape = [b, m, k];
    let tau_shape = [b, k];
    let c_shape = [b, m, n];
    let args = BatchedOrmqrWyArgs::<f32> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: a_shape,
            stride: contiguous_stride(a_shape),
        },
        tau: TensorMut {
            data: dev_tau.as_slice_mut(),
            shape: tau_shape,
            stride: contiguous_stride(tau_shape),
        },
        c: TensorMut {
            data: dev_c.as_slice_mut(),
            shape: c_shape,
            stride: contiguous_stride(c_shape),
        },
    };
    plan.run(stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run WY ormqr f32");
    stream.synchronize().expect("sync");
    let mut c_post = vec![0f32; (b * m * n) as usize];
    dev_c.copy_to_host(&mut c_post).expect("dl c-post");
    c_post
}

fn wy_ormqr_f64(
    ctx: &Context,
    stream: &Stream,
    a_packed: &[f64],
    tau: &[f64],
    c_init: &[f64],
    b: i32,
    m: i32,
    n: i32,
    k: i32,
    op: BatchedOrmqrOp,
) -> Vec<f64> {
    let mut dev_a = DeviceBuffer::from_slice(ctx, a_packed).expect("upload a");
    let mut dev_tau = DeviceBuffer::from_slice(ctx, tau).expect("upload tau");
    let mut dev_c = DeviceBuffer::from_slice(ctx, c_init).expect("upload c");
    let desc = BatchedOrmqrWyDescriptor {
        m,
        n,
        k,
        batch_size: b,
        side: BatchedOrmqrSide::Left,
        op,
        element: ElementKind::F64,
    };
    let plan = BatchedOrmqrWyPlan::<f64>::select(stream, &desc, PlanPreference::default())
        .expect("select BatchedOrmqrWyPlan<f64>");
    let ws_bytes = plan.query_workspace_size(stream).expect("ws query");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(ctx, ws_bytes).expect("alloc ws");
    let a_shape = [b, m, k];
    let tau_shape = [b, k];
    let c_shape = [b, m, n];
    let args = BatchedOrmqrWyArgs::<f64> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: a_shape,
            stride: contiguous_stride(a_shape),
        },
        tau: TensorMut {
            data: dev_tau.as_slice_mut(),
            shape: tau_shape,
            stride: contiguous_stride(tau_shape),
        },
        c: TensorMut {
            data: dev_c.as_slice_mut(),
            shape: c_shape,
            stride: contiguous_stride(c_shape),
        },
    };
    plan.run(stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run WY ormqr f64");
    stream.synchronize().expect("sync");
    let mut c_post = vec![0f64; (b * m * n) as usize];
    dev_c.copy_to_host(&mut c_post).expect("dl c-post");
    c_post
}

fn check_f32(got: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        let t = tol * e.abs().max(1.0);
        assert!(
            diff <= t,
            "{label}: cell {i}: got={g}, expected={e}, diff={diff}, tol={t}",
        );
    }
}

fn check_f64(got: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        let t = tol * e.abs().max(1.0);
        assert!(
            diff <= t,
            "{label}: cell {i}: got={g}, expected={e}, diff={diff}, tol={t}",
        );
    }
}

// Fixture sized to exercise the multi-block WY path: with `WY_NB = 32`,
// `K = 64` gives `num_blocks = 2`, so iteration order matters and the
// per-block T-build runs twice.
const B: i32 = 2;
const M: i32 = 64;
const N: i32 = 32;
const K: i32 = 64;

#[test]
#[ignore]
fn ormqr_batched_wy_f32_left_n() {
    let (ctx, stream) = setup();
    let a_host = build_matrix_f32(B as usize, M as usize, K as usize, 0xA1A2_A3A4);
    let (a_packed, tau) = run_batched_qr_f32(&ctx, &stream, &a_host, B, M, K);
    let c_init = build_matrix_f32(B as usize, M as usize, N as usize, 0xB1B2_B3B4);

    let op = BatchedOrmqrOp::N;
    let wy = wy_ormqr_f32(&ctx, &stream, &a_packed, &tau, &c_init, B, M, N, K, op);
    let reference =
        ref_ormqr_f32(&ctx, &stream, &a_packed, &tau, &c_init, B, M, N, K, op);
    // WY uses GEMM accumulation — tighter bound than the reflector path.
    let tol = 16.0 * f32::EPSILON;
    check_f32(&wy, &reference, tol, &format!("f32 wy_ormqr op={}", fmt_op(op)));
}

#[test]
#[ignore]
fn ormqr_batched_wy_f32_left_t() {
    let (ctx, stream) = setup();
    let a_host = build_matrix_f32(B as usize, M as usize, K as usize, 0xC1C2_C3C4);
    let (a_packed, tau) = run_batched_qr_f32(&ctx, &stream, &a_host, B, M, K);
    let c_init = build_matrix_f32(B as usize, M as usize, N as usize, 0xD1D2_D3D4);

    let op = BatchedOrmqrOp::T;
    let wy = wy_ormqr_f32(&ctx, &stream, &a_packed, &tau, &c_init, B, M, N, K, op);
    let reference =
        ref_ormqr_f32(&ctx, &stream, &a_packed, &tau, &c_init, B, M, N, K, op);
    let tol = 16.0 * f32::EPSILON;
    check_f32(&wy, &reference, tol, &format!("f32 wy_ormqr op={}", fmt_op(op)));
}

#[test]
#[ignore]
fn ormqr_batched_wy_f64_left_n() {
    let (ctx, stream) = setup();
    let a_host = build_matrix_f64(B as usize, M as usize, K as usize, 0xE1E2_E3E4);
    let (a_packed, tau) = run_batched_qr_f64(&ctx, &stream, &a_host, B, M, K);
    let c_init = build_matrix_f64(B as usize, M as usize, N as usize, 0xF1F2_F3F4);

    let op = BatchedOrmqrOp::N;
    let wy = wy_ormqr_f64(&ctx, &stream, &a_packed, &tau, &c_init, B, M, N, K, op);
    let reference =
        ref_ormqr_f64(&ctx, &stream, &a_packed, &tau, &c_init, B, M, N, K, op);
    let tol = 32.0 * f64::EPSILON;
    check_f64(&wy, &reference, tol, &format!("f64 wy_ormqr op={}", fmt_op(op)));
}

#[test]
#[ignore]
fn ormqr_batched_wy_f64_left_t() {
    let (ctx, stream) = setup();
    let a_host = build_matrix_f64(B as usize, M as usize, K as usize, 0x1234_5678);
    let (a_packed, tau) = run_batched_qr_f64(&ctx, &stream, &a_host, B, M, K);
    let c_init = build_matrix_f64(B as usize, M as usize, N as usize, 0x8765_4321);

    let op = BatchedOrmqrOp::T;
    let wy = wy_ormqr_f64(&ctx, &stream, &a_packed, &tau, &c_init, B, M, N, K, op);
    let reference =
        ref_ormqr_f64(&ctx, &stream, &a_packed, &tau, &c_init, B, M, N, K, op);
    let tol = 32.0 * f64::EPSILON;
    check_f64(&wy, &reference, tol, &format!("f64 wy_ormqr op={}", fmt_op(op)));
}

// =============================================================================
// Phase 26 — Complex variants. Same fixture shape, complex Householder
// reflectors emitted by cuBLAS `C/ZgeqrfBatched`. The WY-blocked plan
// must agree with the reflector-by-reflector plan (which already
// supports complex via Milestone 6.18) at GEMM tolerances.
// =============================================================================

fn build_matrix_complex32(b: usize, m: usize, n: usize, seed: u32) -> Vec<Complex32> {
    let re = build_matrix_f32(b, m, n, seed);
    let im = build_matrix_f32(b, m, n, seed.wrapping_mul(0x6F4A_8B19));
    re.into_iter()
        .zip(im)
        .map(|(r, i)| Complex32::new(r, i))
        .collect()
}

fn build_matrix_complex64(b: usize, m: usize, n: usize, seed: u32) -> Vec<Complex64> {
    let re = build_matrix_f64(b, m, n, seed);
    let im = build_matrix_f64(b, m, n, seed.wrapping_mul(0x6F4A_8B19));
    re.into_iter()
        .zip(im)
        .map(|(r, i)| Complex64::new(r, i))
        .collect()
}

fn run_batched_qr_complex32(
    ctx: &Context,
    stream: &Stream,
    a_host: &[Complex32],
    b: i32,
    m: i32,
    n: i32,
) -> (Vec<Complex32>, Vec<Complex32>) {
    let k = m.min(n);
    let mut dev_a = DeviceBuffer::from_slice(ctx, a_host).expect("upload a");
    let mut dev_tau: DeviceBuffer<Complex32> =
        DeviceBuffer::zeros(ctx, (b * k) as usize).expect("alloc tau");
    let desc = BatchedQrDescriptor {
        m,
        n,
        batch_size: b,
        element: ElementKind::Complex32,
    };
    let plan = BatchedQrPlan::<Complex32>::select(stream, &desc, PlanPreference::default())
        .expect("select BatchedQrPlan<Complex32>");
    let ws_bytes = plan.query_workspace_size(stream).expect("ws query");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(ctx, ws_bytes).expect("alloc ws");
    let args = BatchedQrArgs::<Complex32> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: [b, m, n],
            stride: contiguous_stride([b, m, n]),
        },
        tau: TensorMut {
            data: dev_tau.as_slice_mut(),
            shape: [b, k],
            stride: contiguous_stride([b, k]),
        },
    };
    plan.run(stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run batched QR complex32");
    stream.synchronize().expect("sync");
    let mut a_post = vec![Complex32::new(0.0, 0.0); (b * m * n) as usize];
    dev_a.copy_to_host(&mut a_post).expect("dl a-post");
    let mut tau_post = vec![Complex32::new(0.0, 0.0); (b * k) as usize];
    dev_tau.copy_to_host(&mut tau_post).expect("dl tau-post");
    (a_post, tau_post)
}

fn run_batched_qr_complex64(
    ctx: &Context,
    stream: &Stream,
    a_host: &[Complex64],
    b: i32,
    m: i32,
    n: i32,
) -> (Vec<Complex64>, Vec<Complex64>) {
    let k = m.min(n);
    let mut dev_a = DeviceBuffer::from_slice(ctx, a_host).expect("upload a");
    let mut dev_tau: DeviceBuffer<Complex64> =
        DeviceBuffer::zeros(ctx, (b * k) as usize).expect("alloc tau");
    let desc = BatchedQrDescriptor {
        m,
        n,
        batch_size: b,
        element: ElementKind::Complex64,
    };
    let plan = BatchedQrPlan::<Complex64>::select(stream, &desc, PlanPreference::default())
        .expect("select BatchedQrPlan<Complex64>");
    let ws_bytes = plan.query_workspace_size(stream).expect("ws query");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(ctx, ws_bytes).expect("alloc ws");
    let args = BatchedQrArgs::<Complex64> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: [b, m, n],
            stride: contiguous_stride([b, m, n]),
        },
        tau: TensorMut {
            data: dev_tau.as_slice_mut(),
            shape: [b, k],
            stride: contiguous_stride([b, k]),
        },
    };
    plan.run(stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run batched QR complex64");
    stream.synchronize().expect("sync");
    let mut a_post = vec![Complex64::new(0.0, 0.0); (b * m * n) as usize];
    dev_a.copy_to_host(&mut a_post).expect("dl a-post");
    let mut tau_post = vec![Complex64::new(0.0, 0.0); (b * k) as usize];
    dev_tau.copy_to_host(&mut tau_post).expect("dl tau-post");
    (a_post, tau_post)
}

fn ref_ormqr_complex32(
    ctx: &Context,
    stream: &Stream,
    a_packed: &[Complex32],
    tau: &[Complex32],
    c_init: &[Complex32],
    b: i32,
    m: i32,
    n: i32,
    k: i32,
    op: BatchedOrmqrOp,
) -> Vec<Complex32> {
    let dev_a = DeviceBuffer::from_slice(ctx, a_packed).expect("upload a");
    let dev_tau = DeviceBuffer::from_slice(ctx, tau).expect("upload tau");
    let mut dev_c = DeviceBuffer::from_slice(ctx, c_init).expect("upload c");
    let desc = BatchedOrmqrDescriptor {
        m,
        n,
        k,
        batch_size: b,
        side: BatchedOrmqrSide::Left,
        op,
        element: ElementKind::Complex32,
    };
    let plan = BatchedOrmqrPlan::<Complex32>::select(stream, &desc, PlanPreference::default())
        .expect("select BatchedOrmqrPlan<Complex32>");
    let args = BatchedOrmqrArgs::<Complex32> {
        a_packed: TensorRef {
            data: dev_a.as_slice(),
            shape: [b, m, k],
            stride: contiguous_stride([b, m, k]),
        },
        tau: TensorRef {
            data: dev_tau.as_slice(),
            shape: [b, k],
            stride: contiguous_stride([b, k]),
        },
        c: TensorMut {
            data: dev_c.as_slice_mut(),
            shape: [b, m, n],
            stride: contiguous_stride([b, m, n]),
        },
    };
    plan.run(stream, Workspace::None, args)
        .expect("run reflector ormqr complex32");
    stream.synchronize().expect("sync");
    let mut c_post = vec![Complex32::new(0.0, 0.0); (b * m * n) as usize];
    dev_c.copy_to_host(&mut c_post).expect("dl c-post");
    c_post
}

fn ref_ormqr_complex64(
    ctx: &Context,
    stream: &Stream,
    a_packed: &[Complex64],
    tau: &[Complex64],
    c_init: &[Complex64],
    b: i32,
    m: i32,
    n: i32,
    k: i32,
    op: BatchedOrmqrOp,
) -> Vec<Complex64> {
    let dev_a = DeviceBuffer::from_slice(ctx, a_packed).expect("upload a");
    let dev_tau = DeviceBuffer::from_slice(ctx, tau).expect("upload tau");
    let mut dev_c = DeviceBuffer::from_slice(ctx, c_init).expect("upload c");
    let desc = BatchedOrmqrDescriptor {
        m,
        n,
        k,
        batch_size: b,
        side: BatchedOrmqrSide::Left,
        op,
        element: ElementKind::Complex64,
    };
    let plan = BatchedOrmqrPlan::<Complex64>::select(stream, &desc, PlanPreference::default())
        .expect("select BatchedOrmqrPlan<Complex64>");
    let args = BatchedOrmqrArgs::<Complex64> {
        a_packed: TensorRef {
            data: dev_a.as_slice(),
            shape: [b, m, k],
            stride: contiguous_stride([b, m, k]),
        },
        tau: TensorRef {
            data: dev_tau.as_slice(),
            shape: [b, k],
            stride: contiguous_stride([b, k]),
        },
        c: TensorMut {
            data: dev_c.as_slice_mut(),
            shape: [b, m, n],
            stride: contiguous_stride([b, m, n]),
        },
    };
    plan.run(stream, Workspace::None, args)
        .expect("run reflector ormqr complex64");
    stream.synchronize().expect("sync");
    let mut c_post = vec![Complex64::new(0.0, 0.0); (b * m * n) as usize];
    dev_c.copy_to_host(&mut c_post).expect("dl c-post");
    c_post
}

fn wy_ormqr_complex32(
    ctx: &Context,
    stream: &Stream,
    a_packed: &[Complex32],
    tau: &[Complex32],
    c_init: &[Complex32],
    b: i32,
    m: i32,
    n: i32,
    k: i32,
    op: BatchedOrmqrOp,
) -> Vec<Complex32> {
    let mut dev_a = DeviceBuffer::from_slice(ctx, a_packed).expect("upload a");
    let mut dev_tau = DeviceBuffer::from_slice(ctx, tau).expect("upload tau");
    let mut dev_c = DeviceBuffer::from_slice(ctx, c_init).expect("upload c");
    let desc = BatchedOrmqrWyDescriptor {
        m,
        n,
        k,
        batch_size: b,
        side: BatchedOrmqrSide::Left,
        op,
        element: ElementKind::Complex32,
    };
    let plan = BatchedOrmqrWyPlan::<Complex32>::select(stream, &desc, PlanPreference::default())
        .expect("select BatchedOrmqrWyPlan<Complex32>");
    let ws_bytes = plan.query_workspace_size(stream).expect("ws query");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(ctx, ws_bytes).expect("alloc ws");
    let args = BatchedOrmqrWyArgs::<Complex32> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: [b, m, k],
            stride: contiguous_stride([b, m, k]),
        },
        tau: TensorMut {
            data: dev_tau.as_slice_mut(),
            shape: [b, k],
            stride: contiguous_stride([b, k]),
        },
        c: TensorMut {
            data: dev_c.as_slice_mut(),
            shape: [b, m, n],
            stride: contiguous_stride([b, m, n]),
        },
    };
    plan.run(stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run WY ormqr complex32");
    stream.synchronize().expect("sync");
    let mut c_post = vec![Complex32::new(0.0, 0.0); (b * m * n) as usize];
    dev_c.copy_to_host(&mut c_post).expect("dl c-post");
    c_post
}

fn wy_ormqr_complex64(
    ctx: &Context,
    stream: &Stream,
    a_packed: &[Complex64],
    tau: &[Complex64],
    c_init: &[Complex64],
    b: i32,
    m: i32,
    n: i32,
    k: i32,
    op: BatchedOrmqrOp,
) -> Vec<Complex64> {
    let mut dev_a = DeviceBuffer::from_slice(ctx, a_packed).expect("upload a");
    let mut dev_tau = DeviceBuffer::from_slice(ctx, tau).expect("upload tau");
    let mut dev_c = DeviceBuffer::from_slice(ctx, c_init).expect("upload c");
    let desc = BatchedOrmqrWyDescriptor {
        m,
        n,
        k,
        batch_size: b,
        side: BatchedOrmqrSide::Left,
        op,
        element: ElementKind::Complex64,
    };
    let plan = BatchedOrmqrWyPlan::<Complex64>::select(stream, &desc, PlanPreference::default())
        .expect("select BatchedOrmqrWyPlan<Complex64>");
    let ws_bytes = plan.query_workspace_size(stream).expect("ws query");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(ctx, ws_bytes).expect("alloc ws");
    let args = BatchedOrmqrWyArgs::<Complex64> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: [b, m, k],
            stride: contiguous_stride([b, m, k]),
        },
        tau: TensorMut {
            data: dev_tau.as_slice_mut(),
            shape: [b, k],
            stride: contiguous_stride([b, k]),
        },
        c: TensorMut {
            data: dev_c.as_slice_mut(),
            shape: [b, m, n],
            stride: contiguous_stride([b, m, n]),
        },
    };
    plan.run(stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run WY ormqr complex64");
    stream.synchronize().expect("sync");
    let mut c_post = vec![Complex64::new(0.0, 0.0); (b * m * n) as usize];
    dev_c.copy_to_host(&mut c_post).expect("dl c-post");
    c_post
}

fn check_complex32(got: &[Complex32], expected: &[Complex32], tol: f32, label: &str) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let d_re = (g.re - e.re).abs();
        let d_im = (g.im - e.im).abs();
        let mag = (e.re * e.re + e.im * e.im).sqrt().max(1.0);
        let t = tol * mag;
        assert!(
            d_re <= t && d_im <= t,
            "{label}: cell {i}: got=({}, {}), expected=({}, {}), \
             d_re={d_re}, d_im={d_im}, tol={t}",
            g.re, g.im, e.re, e.im,
        );
    }
}

fn check_complex64(got: &[Complex64], expected: &[Complex64], tol: f64, label: &str) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let d_re = (g.re - e.re).abs();
        let d_im = (g.im - e.im).abs();
        let mag = (e.re * e.re + e.im * e.im).sqrt().max(1.0);
        let t = tol * mag;
        assert!(
            d_re <= t && d_im <= t,
            "{label}: cell {i}: got=({}, {}), expected=({}, {}), \
             d_re={d_re}, d_im={d_im}, tol={t}",
            g.re, g.im, e.re, e.im,
        );
    }
}

#[test]
#[ignore]
fn ormqr_batched_wy_complex32_left_n() {
    let (ctx, stream) = setup();
    let a_host = build_matrix_complex32(B as usize, M as usize, K as usize, 0x1357_2468);
    let (a_packed, tau) = run_batched_qr_complex32(&ctx, &stream, &a_host, B, M, K);
    let c_init = build_matrix_complex32(B as usize, M as usize, N as usize, 0x9ABC_DEF0);

    let op = BatchedOrmqrOp::N;
    let wy = wy_ormqr_complex32(&ctx, &stream, &a_packed, &tau, &c_init, B, M, N, K, op);
    let reference =
        ref_ormqr_complex32(&ctx, &stream, &a_packed, &tau, &c_init, B, M, N, K, op);
    // Complex GEMM accumulation is slightly noisier than real; pad the
    // tolerance to 32·ε (still tight by GEMM standards).
    let tol = 32.0 * f32::EPSILON;
    check_complex32(
        &wy,
        &reference,
        tol,
        &format!("complex32 wy_ormqr op={}", fmt_op(op)),
    );
}

#[test]
#[ignore]
fn ormqr_batched_wy_complex32_left_c() {
    let (ctx, stream) = setup();
    let a_host = build_matrix_complex32(B as usize, M as usize, K as usize, 0x2468_1357);
    let (a_packed, tau) = run_batched_qr_complex32(&ctx, &stream, &a_host, B, M, K);
    let c_init = build_matrix_complex32(B as usize, M as usize, N as usize, 0xDEAD_BEEF);

    let op = BatchedOrmqrOp::C;
    let wy = wy_ormqr_complex32(&ctx, &stream, &a_packed, &tau, &c_init, B, M, N, K, op);
    let reference =
        ref_ormqr_complex32(&ctx, &stream, &a_packed, &tau, &c_init, B, M, N, K, op);
    let tol = 32.0 * f32::EPSILON;
    check_complex32(
        &wy,
        &reference,
        tol,
        &format!("complex32 wy_ormqr op={}", fmt_op(op)),
    );
}

#[test]
#[ignore]
fn ormqr_batched_wy_complex64_left_n() {
    let (ctx, stream) = setup();
    let a_host = build_matrix_complex64(B as usize, M as usize, K as usize, 0xCAFE_F00D);
    let (a_packed, tau) = run_batched_qr_complex64(&ctx, &stream, &a_host, B, M, K);
    let c_init = build_matrix_complex64(B as usize, M as usize, N as usize, 0xFACE_FEED);

    let op = BatchedOrmqrOp::N;
    let wy = wy_ormqr_complex64(&ctx, &stream, &a_packed, &tau, &c_init, B, M, N, K, op);
    let reference =
        ref_ormqr_complex64(&ctx, &stream, &a_packed, &tau, &c_init, B, M, N, K, op);
    let tol = 64.0 * f64::EPSILON;
    check_complex64(
        &wy,
        &reference,
        tol,
        &format!("complex64 wy_ormqr op={}", fmt_op(op)),
    );
}

#[test]
#[ignore]
fn ormqr_batched_wy_complex64_left_c() {
    let (ctx, stream) = setup();
    let a_host = build_matrix_complex64(B as usize, M as usize, K as usize, 0xBEEF_C0DE);
    let (a_packed, tau) = run_batched_qr_complex64(&ctx, &stream, &a_host, B, M, K);
    let c_init = build_matrix_complex64(B as usize, M as usize, N as usize, 0xC0DE_F00D);

    let op = BatchedOrmqrOp::C;
    let wy = wy_ormqr_complex64(&ctx, &stream, &a_packed, &tau, &c_init, B, M, N, K, op);
    let reference =
        ref_ormqr_complex64(&ctx, &stream, &a_packed, &tau, &c_init, B, M, N, K, op);
    let tol = 64.0 * f64::EPSILON;
    check_complex64(
        &wy,
        &reference,
        tol,
        &format!("complex64 wy_ormqr op={}", fmt_op(op)),
    );
}
