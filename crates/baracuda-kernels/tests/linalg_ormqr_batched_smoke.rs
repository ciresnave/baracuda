//! Real-GPU smoke test for `BatchedOrmqrPlan` (bespoke kernel,
//! Milestone 6.14).
//!
//! Cross-checks the batched-`ormqr` kernel against cuSOLVER's
//! non-batched `cusolverDn{S,D}ormqr` looped over batch slots, for both
//! `op = N` (apply `Q`) and `op = T` (apply `Q^T`). Each input batch
//! is independently QR-factorized via [`BatchedQrPlan`] first; the
//! resulting packed (A_packed, tau) feeds both the bespoke kernel and
//! the per-slot cuSOLVER reference.
//!
//! `#[ignore]` by default — requires a real CUDA device.

use core::ffi::c_void;

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BatchedOrmqrArgs, BatchedOrmqrDescriptor, BatchedOrmqrOp, BatchedOrmqrPlan,
    BatchedOrmqrSide, BatchedQrArgs, BatchedQrDescriptor, BatchedQrPlan, Complex32, Complex64,
    ElementKind, PlanPreference, TensorMut, TensorRef, Workspace,
};
use baracuda_kernels_sys::{
    cuDoubleComplex, cuFloatComplex, cusolverDnCgeqrf, cusolverDnCgeqrf_bufferSize,
    cusolverDnCreate, cusolverDnCunmqr, cusolverDnCunmqr_bufferSize, cusolverDnDestroy,
    cusolverDnDormqr, cusolverDnDormqr_bufferSize, cusolverDnHandle_t, cusolverDnSetStream,
    cusolverDnSormqr, cusolverDnSormqr_bufferSize, cusolverDnZgeqrf, cusolverDnZgeqrf_bufferSize,
    cusolverDnZunmqr, cusolverDnZunmqr_bufferSize, CUBLAS_OP_C, CUBLAS_OP_N, CUBLAS_OP_T,
    CUBLAS_SIDE_LEFT, CUBLAS_SIDE_RIGHT,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// Deterministic pseudo-random matrix builder (LCG, host-side, used to
// seed the inputs so the test is reproducible).
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

fn op_to_cublas(op: BatchedOrmqrOp) -> i32 {
    match op {
        BatchedOrmqrOp::N => CUBLAS_OP_N,
        BatchedOrmqrOp::T => CUBLAS_OP_T,
        BatchedOrmqrOp::C => CUBLAS_OP_C,
    }
}

/// Run `BatchedQrPlan` over `[B, M, N]` `f32` input and return
/// (packed_A_host, tau_host).
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

/// Apply cuSOLVER's non-batched `ormqr` per slot and return the post-
/// ormqr C buffer (concatenated, `[B, M, N]` column-major). For Side =
/// Left the packed slot stride is `M·K` and `lda = M`; for Side =
/// Right the packed slot stride is `N·N` and `lda = N`.
fn cusolver_ormqr_per_slot_f32(
    ctx: &Context,
    stream: &Stream,
    a_packed: &[f32],
    tau: &[f32],
    c_init: &[f32],
    b: i32,
    m: i32,
    n: i32,
    k: i32,
    side: BatchedOrmqrSide,
    op: BatchedOrmqrOp,
) -> Vec<f32> {
    let stream_ptr = stream.as_raw() as *mut c_void;
    let mut handle: cusolverDnHandle_t = core::ptr::null_mut();
    let s = unsafe { cusolverDnCreate(&mut handle as *mut _) };
    assert_eq!(s, 0);
    let s = unsafe { cusolverDnSetStream(handle, stream_ptr) };
    assert_eq!(s, 0);

    // Upload per-slot copies and run ormqr in place.
    let mu = m as usize;
    let nu = n as usize;
    let ku = k as usize;
    let (side_flag, a_slot_elems, lda) = match side {
        BatchedOrmqrSide::Left => (CUBLAS_SIDE_LEFT, mu * ku, m),
        BatchedOrmqrSide::Right => (CUBLAS_SIDE_RIGHT, nu * nu, n),
    };
    let mut c_post = vec![0f32; (b * m * n) as usize];
    let trans = op_to_cublas(op);

    for bi in 0..b as usize {
        let dev_a_slot = DeviceBuffer::from_slice(
            ctx,
            &a_packed[bi * a_slot_elems..(bi + 1) * a_slot_elems],
        )
        .expect("upload slot a");
        let dev_tau_slot =
            DeviceBuffer::from_slice(ctx, &tau[bi * ku..(bi + 1) * ku]).expect("upload slot tau");
        let mut dev_c_slot = DeviceBuffer::from_slice(
            ctx,
            &c_init[bi * mu * nu..(bi + 1) * mu * nu],
        )
        .expect("upload slot c");

        let mut lwork: i32 = 0;
        let s = unsafe {
            cusolverDnSormqr_bufferSize(
                handle,
                side_flag,
                trans,
                m,
                n,
                k,
                dev_a_slot.as_slice().as_raw().0 as *const f32,
                lda,
                dev_tau_slot.as_slice().as_raw().0 as *const f32,
                dev_c_slot.as_slice().as_raw().0 as *const f32,
                m,
                &mut lwork as *mut _,
            )
        };
        assert_eq!(s, 0);
        let mut dev_work: DeviceBuffer<f32> =
            DeviceBuffer::zeros(ctx, lwork.max(1) as usize).expect("alloc work");
        let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(ctx, 1).expect("alloc info");

        let s = unsafe {
            cusolverDnSormqr(
                handle,
                side_flag,
                trans,
                m,
                n,
                k,
                dev_a_slot.as_slice().as_raw().0 as *const f32,
                lda,
                dev_tau_slot.as_slice().as_raw().0 as *const f32,
                dev_c_slot.as_slice_mut().as_raw().0 as *mut f32,
                m,
                dev_work.as_slice_mut().as_raw().0 as *mut f32,
                lwork,
                dev_info.as_slice_mut().as_raw().0 as *mut i32,
            )
        };
        assert_eq!(s, 0);
        stream.synchronize().expect("sync");
        let mut info_host = vec![0i32; 1];
        dev_info.copy_to_host(&mut info_host).expect("dl info");
        assert_eq!(info_host[0], 0);
        let mut slot_out = vec![0f32; mu * nu];
        dev_c_slot.copy_to_host(&mut slot_out).expect("dl slot c");
        c_post[bi * mu * nu..(bi + 1) * mu * nu].copy_from_slice(&slot_out);

        // Keep buffers around until after sync; allow drop here.
        drop(dev_a_slot);
        drop(dev_tau_slot);
        drop(dev_c_slot);
        drop(dev_work);
        drop(dev_info);
    }
    unsafe {
        let _ = cusolverDnDestroy(handle);
    }
    c_post
}

fn cusolver_ormqr_per_slot_f64(
    ctx: &Context,
    stream: &Stream,
    a_packed: &[f64],
    tau: &[f64],
    c_init: &[f64],
    b: i32,
    m: i32,
    n: i32,
    k: i32,
    side: BatchedOrmqrSide,
    op: BatchedOrmqrOp,
) -> Vec<f64> {
    let stream_ptr = stream.as_raw() as *mut c_void;
    let mut handle: cusolverDnHandle_t = core::ptr::null_mut();
    let s = unsafe { cusolverDnCreate(&mut handle as *mut _) };
    assert_eq!(s, 0);
    let s = unsafe { cusolverDnSetStream(handle, stream_ptr) };
    assert_eq!(s, 0);

    let mu = m as usize;
    let nu = n as usize;
    let ku = k as usize;
    let (side_flag, a_slot_elems, lda) = match side {
        BatchedOrmqrSide::Left => (CUBLAS_SIDE_LEFT, mu * ku, m),
        BatchedOrmqrSide::Right => (CUBLAS_SIDE_RIGHT, nu * nu, n),
    };
    let mut c_post = vec![0f64; (b * m * n) as usize];
    let trans = op_to_cublas(op);

    for bi in 0..b as usize {
        let dev_a_slot = DeviceBuffer::from_slice(
            ctx,
            &a_packed[bi * a_slot_elems..(bi + 1) * a_slot_elems],
        )
        .expect("upload slot a");
        let dev_tau_slot =
            DeviceBuffer::from_slice(ctx, &tau[bi * ku..(bi + 1) * ku]).expect("upload slot tau");
        let mut dev_c_slot = DeviceBuffer::from_slice(
            ctx,
            &c_init[bi * mu * nu..(bi + 1) * mu * nu],
        )
        .expect("upload slot c");

        let mut lwork: i32 = 0;
        let s = unsafe {
            cusolverDnDormqr_bufferSize(
                handle,
                side_flag,
                trans,
                m,
                n,
                k,
                dev_a_slot.as_slice().as_raw().0 as *const f64,
                lda,
                dev_tau_slot.as_slice().as_raw().0 as *const f64,
                dev_c_slot.as_slice().as_raw().0 as *const f64,
                m,
                &mut lwork as *mut _,
            )
        };
        assert_eq!(s, 0);
        let mut dev_work: DeviceBuffer<f64> =
            DeviceBuffer::zeros(ctx, lwork.max(1) as usize).expect("alloc work");
        let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(ctx, 1).expect("alloc info");

        let s = unsafe {
            cusolverDnDormqr(
                handle,
                side_flag,
                trans,
                m,
                n,
                k,
                dev_a_slot.as_slice().as_raw().0 as *const f64,
                lda,
                dev_tau_slot.as_slice().as_raw().0 as *const f64,
                dev_c_slot.as_slice_mut().as_raw().0 as *mut f64,
                m,
                dev_work.as_slice_mut().as_raw().0 as *mut f64,
                lwork,
                dev_info.as_slice_mut().as_raw().0 as *mut i32,
            )
        };
        assert_eq!(s, 0);
        stream.synchronize().expect("sync");
        let mut info_host = vec![0i32; 1];
        dev_info.copy_to_host(&mut info_host).expect("dl info");
        assert_eq!(info_host[0], 0);
        let mut slot_out = vec![0f64; mu * nu];
        dev_c_slot.copy_to_host(&mut slot_out).expect("dl slot c");
        c_post[bi * mu * nu..(bi + 1) * mu * nu].copy_from_slice(&slot_out);

        drop(dev_a_slot);
        drop(dev_tau_slot);
        drop(dev_c_slot);
        drop(dev_work);
        drop(dev_info);
    }
    unsafe {
        let _ = cusolverDnDestroy(handle);
    }
    c_post
}

/// Run the bespoke kernel and return the post-ormqr C buffer. The
/// `a_packed` shape semantics depend on Side: `[B, M, K]` for Left,
/// `[B, N, N]` for Right.
fn bespoke_ormqr_f32(
    ctx: &Context,
    stream: &Stream,
    a_packed: &[f32],
    tau: &[f32],
    c_init: &[f32],
    b: i32,
    m: i32,
    n: i32,
    k: i32,
    side: BatchedOrmqrSide,
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
        side,
        op,
        element: ElementKind::F32,
    };
    let plan = BatchedOrmqrPlan::<f32>::select(stream, &desc, PlanPreference::default())
        .expect("select BatchedOrmqrPlan<f32>");

    let a_shape = match side {
        BatchedOrmqrSide::Left => [b, m, k],
        BatchedOrmqrSide::Right => [b, n, n],
    };
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
        .expect("run bespoke batched ormqr f32");
    stream.synchronize().expect("sync");
    let mut c_post = vec![0f32; (b * m * n) as usize];
    dev_c.copy_to_host(&mut c_post).expect("dl c-post");
    c_post
}

fn bespoke_ormqr_f64(
    ctx: &Context,
    stream: &Stream,
    a_packed: &[f64],
    tau: &[f64],
    c_init: &[f64],
    b: i32,
    m: i32,
    n: i32,
    k: i32,
    side: BatchedOrmqrSide,
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
        side,
        op,
        element: ElementKind::F64,
    };
    let plan = BatchedOrmqrPlan::<f64>::select(stream, &desc, PlanPreference::default())
        .expect("select BatchedOrmqrPlan<f64>");

    let a_shape = match side {
        BatchedOrmqrSide::Left => [b, m, k],
        BatchedOrmqrSide::Right => [b, n, n],
    };
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
        .expect("run bespoke batched ormqr f64");
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

#[test]
#[ignore]
fn ormqr_batched_f32_left_n() {
    let (ctx, stream) = setup();
    let b: i32 = 2;
    let m: i32 = 4;
    let n: i32 = 3;          // columns of C
    let k = m.min(n);        // == 3 — number of reflectors
    let a_host = build_matrix_f32(b as usize, m as usize, n as usize, 0xA1A2_A3A4);
    let (a_packed, tau) = run_batched_qr_f32(&ctx, &stream, &a_host, b, m, n);
    let c_init = build_matrix_f32(b as usize, m as usize, n as usize, 0xB1B2_B3B4);

    let op = BatchedOrmqrOp::N;
    let side = BatchedOrmqrSide::Left;
    let bespoke = bespoke_ormqr_f32(
        &ctx, &stream, &a_packed, &tau, &c_init, b, m, n, k, side, op,
    );
    let reference = cusolver_ormqr_per_slot_f32(
        &ctx, &stream, &a_packed, &tau, &c_init, b, m, n, k, side, op,
    );
    let tol = 32.0 * f32::EPSILON;
    check_f32(&bespoke, &reference, tol, &format!("f32 ormqr_batched op={}", fmt_op(op)));
}

#[test]
#[ignore]
fn ormqr_batched_f32_left_t() {
    let (ctx, stream) = setup();
    let b: i32 = 2;
    let m: i32 = 4;
    let n: i32 = 3;
    let k = m.min(n);
    let a_host = build_matrix_f32(b as usize, m as usize, n as usize, 0xC1C2_C3C4);
    let (a_packed, tau) = run_batched_qr_f32(&ctx, &stream, &a_host, b, m, n);
    let c_init = build_matrix_f32(b as usize, m as usize, n as usize, 0xD1D2_D3D4);

    let op = BatchedOrmqrOp::T;
    let side = BatchedOrmqrSide::Left;
    let bespoke = bespoke_ormqr_f32(
        &ctx, &stream, &a_packed, &tau, &c_init, b, m, n, k, side, op,
    );
    let reference = cusolver_ormqr_per_slot_f32(
        &ctx, &stream, &a_packed, &tau, &c_init, b, m, n, k, side, op,
    );
    let tol = 32.0 * f32::EPSILON;
    check_f32(&bespoke, &reference, tol, &format!("f32 ormqr_batched op={}", fmt_op(op)));
}

#[test]
#[ignore]
fn ormqr_batched_f64_left_n() {
    let (ctx, stream) = setup();
    let b: i32 = 2;
    let m: i32 = 4;
    let n: i32 = 3;
    let k = m.min(n);
    let a_host = build_matrix_f64(b as usize, m as usize, n as usize, 0xE1E2_E3E4);
    let (a_packed, tau) = run_batched_qr_f64(&ctx, &stream, &a_host, b, m, n);
    let c_init = build_matrix_f64(b as usize, m as usize, n as usize, 0xF1F2_F3F4);

    let op = BatchedOrmqrOp::N;
    let side = BatchedOrmqrSide::Left;
    let bespoke = bespoke_ormqr_f64(
        &ctx, &stream, &a_packed, &tau, &c_init, b, m, n, k, side, op,
    );
    let reference = cusolver_ormqr_per_slot_f64(
        &ctx, &stream, &a_packed, &tau, &c_init, b, m, n, k, side, op,
    );
    let tol = 64.0 * f64::EPSILON;
    check_f64(&bespoke, &reference, tol, &format!("f64 ormqr_batched op={}", fmt_op(op)));
}

#[test]
#[ignore]
fn ormqr_batched_f64_left_t() {
    let (ctx, stream) = setup();
    let b: i32 = 2;
    let m: i32 = 4;
    let n: i32 = 3;
    let k = m.min(n);
    let a_host = build_matrix_f64(b as usize, m as usize, n as usize, 0x1234_5678);
    let (a_packed, tau) = run_batched_qr_f64(&ctx, &stream, &a_host, b, m, n);
    let c_init = build_matrix_f64(b as usize, m as usize, n as usize, 0x8765_4321);

    let op = BatchedOrmqrOp::T;
    let side = BatchedOrmqrSide::Left;
    let bespoke = bespoke_ormqr_f64(
        &ctx, &stream, &a_packed, &tau, &c_init, b, m, n, k, side, op,
    );
    let reference = cusolver_ormqr_per_slot_f64(
        &ctx, &stream, &a_packed, &tau, &c_init, b, m, n, k, side, op,
    );
    let tol = 64.0 * f64::EPSILON;
    check_f64(&bespoke, &reference, tol, &format!("f64 ormqr_batched op={}", fmt_op(op)));
}

// =============================================================================
// Milestone 6.18 — Right-side (real) and complex (Complex32 / Complex64)
// extensions. The Right-side path needs a square `[B, N, N]` packed input
// (Q is N × N); we obtain it by feeding BatchedQrPlan an `[B, N, N]`
// input matrix. The complex path can't use BatchedQrPlan (which is
// real-only), so we factor each slot with non-batched
// `cusolverDn{C,Z}geqrf` and concatenate the per-slot outputs.
// =============================================================================

// ----- Complex deterministic input builders ----------------------------------

fn build_matrix_complex32(b: usize, m: usize, n: usize, seed: u32) -> Vec<Complex32> {
    let mut out = Vec::with_capacity(b * m * n);
    // Seed real and imaginary halves with two LCG streams so the data is
    // genuinely complex (non-zero imaginary part).
    let real = build_matrix_f32(b, m, n, seed);
    let imag = build_matrix_f32(b, m, n, seed.wrapping_add(0x5151_5151));
    for (r, i) in real.into_iter().zip(imag.into_iter()) {
        out.push(Complex32::new(r, i));
    }
    out
}

fn build_matrix_complex64(b: usize, m: usize, n: usize, seed: u32) -> Vec<Complex64> {
    build_matrix_complex32(b, m, n, seed)
        .into_iter()
        .map(|z| Complex64::new(z.re as f64, z.im as f64))
        .collect()
}

// ----- Per-slot complex QR via cusolverDn{C,Z}geqrf --------------------------

fn run_cusolver_geqrf_per_slot_complex32(
    ctx: &Context,
    stream: &Stream,
    a_host: &[Complex32],
    b: i32,
    m: i32,
    n: i32,
) -> (Vec<Complex32>, Vec<Complex32>) {
    let stream_ptr = stream.as_raw() as *mut c_void;
    let mut handle: cusolverDnHandle_t = core::ptr::null_mut();
    let s = unsafe { cusolverDnCreate(&mut handle as *mut _) };
    assert_eq!(s, 0);
    let s = unsafe { cusolverDnSetStream(handle, stream_ptr) };
    assert_eq!(s, 0);

    let k = m.min(n);
    let mu = m as usize;
    let nu = n as usize;
    let ku = k as usize;
    let mut a_packed = vec![Complex32::new(0.0, 0.0); (b * m * n) as usize];
    let mut tau = vec![Complex32::new(0.0, 0.0); (b * k) as usize];

    for bi in 0..b as usize {
        let mut dev_a_slot = DeviceBuffer::from_slice(
            ctx,
            &a_host[bi * mu * nu..(bi + 1) * mu * nu],
        )
        .expect("upload slot a (cgeqrf)");
        let mut dev_tau_slot: DeviceBuffer<Complex32> =
            DeviceBuffer::zeros(ctx, ku).expect("alloc slot tau (cgeqrf)");
        let mut lwork: i32 = 0;
        let s = unsafe {
            cusolverDnCgeqrf_bufferSize(
                handle,
                m,
                n,
                dev_a_slot.as_slice_mut().as_raw().0 as *mut cuFloatComplex,
                m,
                &mut lwork as *mut _,
            )
        };
        assert_eq!(s, 0);
        let mut dev_work: DeviceBuffer<Complex32> =
            DeviceBuffer::zeros(ctx, lwork.max(1) as usize).expect("alloc work cgeqrf");
        let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(ctx, 1).expect("alloc info");
        let s = unsafe {
            cusolverDnCgeqrf(
                handle,
                m,
                n,
                dev_a_slot.as_slice_mut().as_raw().0 as *mut cuFloatComplex,
                m,
                dev_tau_slot.as_slice_mut().as_raw().0 as *mut cuFloatComplex,
                dev_work.as_slice_mut().as_raw().0 as *mut cuFloatComplex,
                lwork,
                dev_info.as_slice_mut().as_raw().0 as *mut i32,
            )
        };
        assert_eq!(s, 0);
        stream.synchronize().expect("sync cgeqrf");
        let mut info_host = vec![0i32; 1];
        dev_info.copy_to_host(&mut info_host).expect("dl info");
        assert_eq!(info_host[0], 0);
        let mut a_out = vec![Complex32::new(0.0, 0.0); mu * nu];
        dev_a_slot.copy_to_host(&mut a_out).expect("dl a");
        a_packed[bi * mu * nu..(bi + 1) * mu * nu].copy_from_slice(&a_out);
        let mut tau_out = vec![Complex32::new(0.0, 0.0); ku];
        dev_tau_slot.copy_to_host(&mut tau_out).expect("dl tau");
        tau[bi * ku..(bi + 1) * ku].copy_from_slice(&tau_out);
    }
    unsafe {
        let _ = cusolverDnDestroy(handle);
    }
    (a_packed, tau)
}

fn run_cusolver_geqrf_per_slot_complex64(
    ctx: &Context,
    stream: &Stream,
    a_host: &[Complex64],
    b: i32,
    m: i32,
    n: i32,
) -> (Vec<Complex64>, Vec<Complex64>) {
    let stream_ptr = stream.as_raw() as *mut c_void;
    let mut handle: cusolverDnHandle_t = core::ptr::null_mut();
    let s = unsafe { cusolverDnCreate(&mut handle as *mut _) };
    assert_eq!(s, 0);
    let s = unsafe { cusolverDnSetStream(handle, stream_ptr) };
    assert_eq!(s, 0);

    let k = m.min(n);
    let mu = m as usize;
    let nu = n as usize;
    let ku = k as usize;
    let mut a_packed = vec![Complex64::new(0.0, 0.0); (b * m * n) as usize];
    let mut tau = vec![Complex64::new(0.0, 0.0); (b * k) as usize];

    for bi in 0..b as usize {
        let mut dev_a_slot = DeviceBuffer::from_slice(
            ctx,
            &a_host[bi * mu * nu..(bi + 1) * mu * nu],
        )
        .expect("upload slot a (zgeqrf)");
        let mut dev_tau_slot: DeviceBuffer<Complex64> =
            DeviceBuffer::zeros(ctx, ku).expect("alloc slot tau (zgeqrf)");
        let mut lwork: i32 = 0;
        let s = unsafe {
            cusolverDnZgeqrf_bufferSize(
                handle,
                m,
                n,
                dev_a_slot.as_slice_mut().as_raw().0 as *mut cuDoubleComplex,
                m,
                &mut lwork as *mut _,
            )
        };
        assert_eq!(s, 0);
        let mut dev_work: DeviceBuffer<Complex64> =
            DeviceBuffer::zeros(ctx, lwork.max(1) as usize).expect("alloc work zgeqrf");
        let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(ctx, 1).expect("alloc info");
        let s = unsafe {
            cusolverDnZgeqrf(
                handle,
                m,
                n,
                dev_a_slot.as_slice_mut().as_raw().0 as *mut cuDoubleComplex,
                m,
                dev_tau_slot.as_slice_mut().as_raw().0 as *mut cuDoubleComplex,
                dev_work.as_slice_mut().as_raw().0 as *mut cuDoubleComplex,
                lwork,
                dev_info.as_slice_mut().as_raw().0 as *mut i32,
            )
        };
        assert_eq!(s, 0);
        stream.synchronize().expect("sync zgeqrf");
        let mut info_host = vec![0i32; 1];
        dev_info.copy_to_host(&mut info_host).expect("dl info");
        assert_eq!(info_host[0], 0);
        let mut a_out = vec![Complex64::new(0.0, 0.0); mu * nu];
        dev_a_slot.copy_to_host(&mut a_out).expect("dl a");
        a_packed[bi * mu * nu..(bi + 1) * mu * nu].copy_from_slice(&a_out);
        let mut tau_out = vec![Complex64::new(0.0, 0.0); ku];
        dev_tau_slot.copy_to_host(&mut tau_out).expect("dl tau");
        tau[bi * ku..(bi + 1) * ku].copy_from_slice(&tau_out);
    }
    unsafe {
        let _ = cusolverDnDestroy(handle);
    }
    (a_packed, tau)
}

// ----- Per-slot cuSOLVER unmqr reference (complex) ---------------------------

fn cusolver_unmqr_per_slot_complex32(
    ctx: &Context,
    stream: &Stream,
    a_packed: &[Complex32],
    tau: &[Complex32],
    c_init: &[Complex32],
    b: i32,
    m: i32,
    n: i32,
    k: i32,
    side: BatchedOrmqrSide,
    op: BatchedOrmqrOp,
) -> Vec<Complex32> {
    let stream_ptr = stream.as_raw() as *mut c_void;
    let mut handle: cusolverDnHandle_t = core::ptr::null_mut();
    let s = unsafe { cusolverDnCreate(&mut handle as *mut _) };
    assert_eq!(s, 0);
    let s = unsafe { cusolverDnSetStream(handle, stream_ptr) };
    assert_eq!(s, 0);

    let mu = m as usize;
    let nu = n as usize;
    let ku = k as usize;
    let (side_flag, a_slot_elems, lda) = match side {
        BatchedOrmqrSide::Left => (CUBLAS_SIDE_LEFT, mu * ku, m),
        BatchedOrmqrSide::Right => (CUBLAS_SIDE_RIGHT, nu * nu, n),
    };
    let trans = op_to_cublas(op);
    let mut c_post = vec![Complex32::new(0.0, 0.0); (b * m * n) as usize];

    for bi in 0..b as usize {
        let dev_a_slot = DeviceBuffer::from_slice(
            ctx,
            &a_packed[bi * a_slot_elems..(bi + 1) * a_slot_elems],
        )
        .expect("upload slot a (cunmqr)");
        let dev_tau_slot =
            DeviceBuffer::from_slice(ctx, &tau[bi * ku..(bi + 1) * ku]).expect("upload slot tau");
        let mut dev_c_slot = DeviceBuffer::from_slice(
            ctx,
            &c_init[bi * mu * nu..(bi + 1) * mu * nu],
        )
        .expect("upload slot c");
        let mut lwork: i32 = 0;
        let s = unsafe {
            cusolverDnCunmqr_bufferSize(
                handle,
                side_flag,
                trans,
                m,
                n,
                k,
                dev_a_slot.as_slice().as_raw().0 as *const cuFloatComplex,
                lda,
                dev_tau_slot.as_slice().as_raw().0 as *const cuFloatComplex,
                dev_c_slot.as_slice().as_raw().0 as *const cuFloatComplex,
                m,
                &mut lwork as *mut _,
            )
        };
        assert_eq!(s, 0);
        let mut dev_work: DeviceBuffer<Complex32> =
            DeviceBuffer::zeros(ctx, lwork.max(1) as usize).expect("alloc work cunmqr");
        let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(ctx, 1).expect("alloc info");
        let s = unsafe {
            cusolverDnCunmqr(
                handle,
                side_flag,
                trans,
                m,
                n,
                k,
                dev_a_slot.as_slice().as_raw().0 as *const cuFloatComplex,
                lda,
                dev_tau_slot.as_slice().as_raw().0 as *const cuFloatComplex,
                dev_c_slot.as_slice_mut().as_raw().0 as *mut cuFloatComplex,
                m,
                dev_work.as_slice_mut().as_raw().0 as *mut cuFloatComplex,
                lwork,
                dev_info.as_slice_mut().as_raw().0 as *mut i32,
            )
        };
        assert_eq!(s, 0);
        stream.synchronize().expect("sync cunmqr");
        let mut info_host = vec![0i32; 1];
        dev_info.copy_to_host(&mut info_host).expect("dl info");
        assert_eq!(info_host[0], 0);
        let mut slot_out = vec![Complex32::new(0.0, 0.0); mu * nu];
        dev_c_slot.copy_to_host(&mut slot_out).expect("dl slot c");
        c_post[bi * mu * nu..(bi + 1) * mu * nu].copy_from_slice(&slot_out);
    }
    unsafe {
        let _ = cusolverDnDestroy(handle);
    }
    c_post
}

fn cusolver_unmqr_per_slot_complex64(
    ctx: &Context,
    stream: &Stream,
    a_packed: &[Complex64],
    tau: &[Complex64],
    c_init: &[Complex64],
    b: i32,
    m: i32,
    n: i32,
    k: i32,
    side: BatchedOrmqrSide,
    op: BatchedOrmqrOp,
) -> Vec<Complex64> {
    let stream_ptr = stream.as_raw() as *mut c_void;
    let mut handle: cusolverDnHandle_t = core::ptr::null_mut();
    let s = unsafe { cusolverDnCreate(&mut handle as *mut _) };
    assert_eq!(s, 0);
    let s = unsafe { cusolverDnSetStream(handle, stream_ptr) };
    assert_eq!(s, 0);

    let mu = m as usize;
    let nu = n as usize;
    let ku = k as usize;
    let (side_flag, a_slot_elems, lda) = match side {
        BatchedOrmqrSide::Left => (CUBLAS_SIDE_LEFT, mu * ku, m),
        BatchedOrmqrSide::Right => (CUBLAS_SIDE_RIGHT, nu * nu, n),
    };
    let trans = op_to_cublas(op);
    let mut c_post = vec![Complex64::new(0.0, 0.0); (b * m * n) as usize];

    for bi in 0..b as usize {
        let dev_a_slot = DeviceBuffer::from_slice(
            ctx,
            &a_packed[bi * a_slot_elems..(bi + 1) * a_slot_elems],
        )
        .expect("upload slot a (zunmqr)");
        let dev_tau_slot =
            DeviceBuffer::from_slice(ctx, &tau[bi * ku..(bi + 1) * ku]).expect("upload slot tau");
        let mut dev_c_slot = DeviceBuffer::from_slice(
            ctx,
            &c_init[bi * mu * nu..(bi + 1) * mu * nu],
        )
        .expect("upload slot c");
        let mut lwork: i32 = 0;
        let s = unsafe {
            cusolverDnZunmqr_bufferSize(
                handle,
                side_flag,
                trans,
                m,
                n,
                k,
                dev_a_slot.as_slice().as_raw().0 as *const cuDoubleComplex,
                lda,
                dev_tau_slot.as_slice().as_raw().0 as *const cuDoubleComplex,
                dev_c_slot.as_slice().as_raw().0 as *const cuDoubleComplex,
                m,
                &mut lwork as *mut _,
            )
        };
        assert_eq!(s, 0);
        let mut dev_work: DeviceBuffer<Complex64> =
            DeviceBuffer::zeros(ctx, lwork.max(1) as usize).expect("alloc work zunmqr");
        let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(ctx, 1).expect("alloc info");
        let s = unsafe {
            cusolverDnZunmqr(
                handle,
                side_flag,
                trans,
                m,
                n,
                k,
                dev_a_slot.as_slice().as_raw().0 as *const cuDoubleComplex,
                lda,
                dev_tau_slot.as_slice().as_raw().0 as *const cuDoubleComplex,
                dev_c_slot.as_slice_mut().as_raw().0 as *mut cuDoubleComplex,
                m,
                dev_work.as_slice_mut().as_raw().0 as *mut cuDoubleComplex,
                lwork,
                dev_info.as_slice_mut().as_raw().0 as *mut i32,
            )
        };
        assert_eq!(s, 0);
        stream.synchronize().expect("sync zunmqr");
        let mut info_host = vec![0i32; 1];
        dev_info.copy_to_host(&mut info_host).expect("dl info");
        assert_eq!(info_host[0], 0);
        let mut slot_out = vec![Complex64::new(0.0, 0.0); mu * nu];
        dev_c_slot.copy_to_host(&mut slot_out).expect("dl slot c");
        c_post[bi * mu * nu..(bi + 1) * mu * nu].copy_from_slice(&slot_out);
    }
    unsafe {
        let _ = cusolverDnDestroy(handle);
    }
    c_post
}

// ----- Bespoke kernel wrappers (complex) -------------------------------------

fn bespoke_ormqr_complex32(
    ctx: &Context,
    stream: &Stream,
    a_packed: &[Complex32],
    tau: &[Complex32],
    c_init: &[Complex32],
    b: i32,
    m: i32,
    n: i32,
    k: i32,
    side: BatchedOrmqrSide,
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
        side,
        op,
        element: ElementKind::Complex32,
    };
    let plan = BatchedOrmqrPlan::<Complex32>::select(stream, &desc, PlanPreference::default())
        .expect("select BatchedOrmqrPlan<Complex32>");
    let a_shape = match side {
        BatchedOrmqrSide::Left => [b, m, k],
        BatchedOrmqrSide::Right => [b, n, n],
    };
    let args = BatchedOrmqrArgs::<Complex32> {
        a_packed: TensorRef {
            data: dev_a.as_slice(),
            shape: a_shape,
            stride: contiguous_stride(a_shape),
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
        .expect("run bespoke batched unmqr complex32");
    stream.synchronize().expect("sync");
    let mut c_post = vec![Complex32::new(0.0, 0.0); (b * m * n) as usize];
    dev_c.copy_to_host(&mut c_post).expect("dl c-post");
    c_post
}

fn bespoke_ormqr_complex64(
    ctx: &Context,
    stream: &Stream,
    a_packed: &[Complex64],
    tau: &[Complex64],
    c_init: &[Complex64],
    b: i32,
    m: i32,
    n: i32,
    k: i32,
    side: BatchedOrmqrSide,
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
        side,
        op,
        element: ElementKind::Complex64,
    };
    let plan = BatchedOrmqrPlan::<Complex64>::select(stream, &desc, PlanPreference::default())
        .expect("select BatchedOrmqrPlan<Complex64>");
    let a_shape = match side {
        BatchedOrmqrSide::Left => [b, m, k],
        BatchedOrmqrSide::Right => [b, n, n],
    };
    let args = BatchedOrmqrArgs::<Complex64> {
        a_packed: TensorRef {
            data: dev_a.as_slice(),
            shape: a_shape,
            stride: contiguous_stride(a_shape),
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
        .expect("run bespoke batched unmqr complex64");
    stream.synchronize().expect("sync");
    let mut c_post = vec![Complex64::new(0.0, 0.0); (b * m * n) as usize];
    dev_c.copy_to_host(&mut c_post).expect("dl c-post");
    c_post
}

fn check_complex32(got: &[Complex32], expected: &[Complex32], tol: f32, label: &str) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff_re = (g.re - e.re).abs();
        let diff_im = (g.im - e.im).abs();
        let scale = (e.re.abs().max(e.im.abs())).max(1.0);
        let t = tol * scale;
        assert!(
            diff_re <= t && diff_im <= t,
            "{label}: cell {i}: got={:?}, expected={:?}, diff=({diff_re}, {diff_im}), tol={t}",
            g, e,
        );
    }
}

fn check_complex64(got: &[Complex64], expected: &[Complex64], tol: f64, label: &str) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff_re = (g.re - e.re).abs();
        let diff_im = (g.im - e.im).abs();
        let scale = (e.re.abs().max(e.im.abs())).max(1.0);
        let t = tol * scale;
        assert!(
            diff_re <= t && diff_im <= t,
            "{label}: cell {i}: got={:?}, expected={:?}, diff=({diff_re}, {diff_im}), tol={t}",
            g, e,
        );
    }
}

// ============================================================================
// Right-side real tests (4)
// ============================================================================

// For Right-side, we need a square [B, N, N] packed input. We build a
// square [B, N, N] f32 matrix, factor it with BatchedQrPlan (M = N case,
// K = N), and feed the packed result to both kernels. C has its own
// shape [B, M, N] independent of the packed input.

#[test]
#[ignore]
fn ormqr_batched_f32_right_n() {
    let (ctx, stream) = setup();
    let b: i32 = 2;
    let m: i32 = 3;          // rows of C
    let n: i32 = 4;          // cols of C == size of Q
    let k = n;               // K = N for Right-side
    // Factor a square [B, N, N] to get the packed Householder Q.
    let a_host = build_matrix_f32(b as usize, n as usize, n as usize, 0x10_2030_40);
    let (a_packed, tau) = run_batched_qr_f32(&ctx, &stream, &a_host, b, n, n);
    let c_init = build_matrix_f32(b as usize, m as usize, n as usize, 0x50_6070_80);

    let op = BatchedOrmqrOp::N;
    let side = BatchedOrmqrSide::Right;
    let bespoke = bespoke_ormqr_f32(
        &ctx, &stream, &a_packed, &tau, &c_init, b, m, n, k, side, op,
    );
    let reference = cusolver_ormqr_per_slot_f32(
        &ctx, &stream, &a_packed, &tau, &c_init, b, m, n, k, side, op,
    );
    let tol = 32.0 * f32::EPSILON;
    check_f32(&bespoke, &reference, tol,
        &format!("f32 ormqr_batched side=Right op={}", fmt_op(op)));
}

#[test]
#[ignore]
fn ormqr_batched_f32_right_t() {
    let (ctx, stream) = setup();
    let b: i32 = 2;
    let m: i32 = 3;
    let n: i32 = 4;
    let k = n;
    let a_host = build_matrix_f32(b as usize, n as usize, n as usize, 0x11_1111_11);
    let (a_packed, tau) = run_batched_qr_f32(&ctx, &stream, &a_host, b, n, n);
    let c_init = build_matrix_f32(b as usize, m as usize, n as usize, 0x22_2222_22);

    let op = BatchedOrmqrOp::T;
    let side = BatchedOrmqrSide::Right;
    let bespoke = bespoke_ormqr_f32(
        &ctx, &stream, &a_packed, &tau, &c_init, b, m, n, k, side, op,
    );
    let reference = cusolver_ormqr_per_slot_f32(
        &ctx, &stream, &a_packed, &tau, &c_init, b, m, n, k, side, op,
    );
    let tol = 32.0 * f32::EPSILON;
    check_f32(&bespoke, &reference, tol,
        &format!("f32 ormqr_batched side=Right op={}", fmt_op(op)));
}

#[test]
#[ignore]
fn ormqr_batched_f64_right_n() {
    let (ctx, stream) = setup();
    let b: i32 = 2;
    let m: i32 = 3;
    let n: i32 = 4;
    let k = n;
    let a_host = build_matrix_f64(b as usize, n as usize, n as usize, 0x33_3333_33);
    let (a_packed, tau) = run_batched_qr_f64(&ctx, &stream, &a_host, b, n, n);
    let c_init = build_matrix_f64(b as usize, m as usize, n as usize, 0x44_4444_44);

    let op = BatchedOrmqrOp::N;
    let side = BatchedOrmqrSide::Right;
    let bespoke = bespoke_ormqr_f64(
        &ctx, &stream, &a_packed, &tau, &c_init, b, m, n, k, side, op,
    );
    let reference = cusolver_ormqr_per_slot_f64(
        &ctx, &stream, &a_packed, &tau, &c_init, b, m, n, k, side, op,
    );
    let tol = 64.0 * f64::EPSILON;
    check_f64(&bespoke, &reference, tol,
        &format!("f64 ormqr_batched side=Right op={}", fmt_op(op)));
}

#[test]
#[ignore]
fn ormqr_batched_f64_right_t() {
    let (ctx, stream) = setup();
    let b: i32 = 2;
    let m: i32 = 3;
    let n: i32 = 4;
    let k = n;
    let a_host = build_matrix_f64(b as usize, n as usize, n as usize, 0x55_5555_55);
    let (a_packed, tau) = run_batched_qr_f64(&ctx, &stream, &a_host, b, n, n);
    let c_init = build_matrix_f64(b as usize, m as usize, n as usize, 0x66_6666_66);

    let op = BatchedOrmqrOp::T;
    let side = BatchedOrmqrSide::Right;
    let bespoke = bespoke_ormqr_f64(
        &ctx, &stream, &a_packed, &tau, &c_init, b, m, n, k, side, op,
    );
    let reference = cusolver_ormqr_per_slot_f64(
        &ctx, &stream, &a_packed, &tau, &c_init, b, m, n, k, side, op,
    );
    let tol = 64.0 * f64::EPSILON;
    check_f64(&bespoke, &reference, tol,
        &format!("f64 ormqr_batched side=Right op={}", fmt_op(op)));
}

// ============================================================================
// Complex Left-side tests (4)
// ============================================================================

#[test]
#[ignore]
fn ormqr_batched_complex32_left_n() {
    let (ctx, stream) = setup();
    let b: i32 = 2;
    let m: i32 = 4;
    let n: i32 = 3;
    let k = m.min(n);
    let a_host = build_matrix_complex32(b as usize, m as usize, n as usize, 0x77_7777_77);
    let (a_packed, tau) =
        run_cusolver_geqrf_per_slot_complex32(&ctx, &stream, &a_host, b, m, n);
    let c_init = build_matrix_complex32(b as usize, m as usize, n as usize, 0x88_8888_88);

    let op = BatchedOrmqrOp::N;
    let side = BatchedOrmqrSide::Left;
    let bespoke = bespoke_ormqr_complex32(
        &ctx, &stream, &a_packed, &tau, &c_init, b, m, n, k, side, op,
    );
    let reference = cusolver_unmqr_per_slot_complex32(
        &ctx, &stream, &a_packed, &tau, &c_init, b, m, n, k, side, op,
    );
    let tol = 32.0 * f32::EPSILON;
    check_complex32(&bespoke, &reference, tol,
        &format!("Complex32 unmqr_batched side=Left op={}", fmt_op(op)));
}

#[test]
#[ignore]
fn ormqr_batched_complex32_left_c() {
    let (ctx, stream) = setup();
    let b: i32 = 2;
    let m: i32 = 4;
    let n: i32 = 3;
    let k = m.min(n);
    let a_host = build_matrix_complex32(b as usize, m as usize, n as usize, 0x99_9999_99);
    let (a_packed, tau) =
        run_cusolver_geqrf_per_slot_complex32(&ctx, &stream, &a_host, b, m, n);
    let c_init = build_matrix_complex32(b as usize, m as usize, n as usize, 0xAA_AAAA_AA);

    let op = BatchedOrmqrOp::C;
    let side = BatchedOrmqrSide::Left;
    let bespoke = bespoke_ormqr_complex32(
        &ctx, &stream, &a_packed, &tau, &c_init, b, m, n, k, side, op,
    );
    let reference = cusolver_unmqr_per_slot_complex32(
        &ctx, &stream, &a_packed, &tau, &c_init, b, m, n, k, side, op,
    );
    let tol = 32.0 * f32::EPSILON;
    check_complex32(&bespoke, &reference, tol,
        &format!("Complex32 unmqr_batched side=Left op={}", fmt_op(op)));
}

#[test]
#[ignore]
fn ormqr_batched_complex64_left_n() {
    let (ctx, stream) = setup();
    let b: i32 = 2;
    let m: i32 = 4;
    let n: i32 = 3;
    let k = m.min(n);
    let a_host = build_matrix_complex64(b as usize, m as usize, n as usize, 0xBB_BBBB_BB);
    let (a_packed, tau) =
        run_cusolver_geqrf_per_slot_complex64(&ctx, &stream, &a_host, b, m, n);
    let c_init = build_matrix_complex64(b as usize, m as usize, n as usize, 0xCC_CCCC_CC);

    let op = BatchedOrmqrOp::N;
    let side = BatchedOrmqrSide::Left;
    let bespoke = bespoke_ormqr_complex64(
        &ctx, &stream, &a_packed, &tau, &c_init, b, m, n, k, side, op,
    );
    let reference = cusolver_unmqr_per_slot_complex64(
        &ctx, &stream, &a_packed, &tau, &c_init, b, m, n, k, side, op,
    );
    let tol = 64.0 * f64::EPSILON;
    check_complex64(&bespoke, &reference, tol,
        &format!("Complex64 unmqr_batched side=Left op={}", fmt_op(op)));
}

#[test]
#[ignore]
fn ormqr_batched_complex64_left_c() {
    let (ctx, stream) = setup();
    let b: i32 = 2;
    let m: i32 = 4;
    let n: i32 = 3;
    let k = m.min(n);
    let a_host = build_matrix_complex64(b as usize, m as usize, n as usize, 0xDD_DDDD_DD);
    let (a_packed, tau) =
        run_cusolver_geqrf_per_slot_complex64(&ctx, &stream, &a_host, b, m, n);
    let c_init = build_matrix_complex64(b as usize, m as usize, n as usize, 0xEE_EEEE_EE);

    let op = BatchedOrmqrOp::C;
    let side = BatchedOrmqrSide::Left;
    let bespoke = bespoke_ormqr_complex64(
        &ctx, &stream, &a_packed, &tau, &c_init, b, m, n, k, side, op,
    );
    let reference = cusolver_unmqr_per_slot_complex64(
        &ctx, &stream, &a_packed, &tau, &c_init, b, m, n, k, side, op,
    );
    let tol = 64.0 * f64::EPSILON;
    check_complex64(&bespoke, &reference, tol,
        &format!("Complex64 unmqr_batched side=Left op={}", fmt_op(op)));
}

// ============================================================================
// Complex Right-side tests (4)
// ============================================================================

#[test]
#[ignore]
fn ormqr_batched_complex32_right_n() {
    let (ctx, stream) = setup();
    let b: i32 = 2;
    let m: i32 = 3;
    let n: i32 = 4;
    let k = n;
    let a_host = build_matrix_complex32(b as usize, n as usize, n as usize, 0x12_3456_78);
    let (a_packed, tau) =
        run_cusolver_geqrf_per_slot_complex32(&ctx, &stream, &a_host, b, n, n);
    let c_init = build_matrix_complex32(b as usize, m as usize, n as usize, 0x9A_BCDE_F0);

    let op = BatchedOrmqrOp::N;
    let side = BatchedOrmqrSide::Right;
    let bespoke = bespoke_ormqr_complex32(
        &ctx, &stream, &a_packed, &tau, &c_init, b, m, n, k, side, op,
    );
    let reference = cusolver_unmqr_per_slot_complex32(
        &ctx, &stream, &a_packed, &tau, &c_init, b, m, n, k, side, op,
    );
    let tol = 32.0 * f32::EPSILON;
    check_complex32(&bespoke, &reference, tol,
        &format!("Complex32 unmqr_batched side=Right op={}", fmt_op(op)));
}

#[test]
#[ignore]
fn ormqr_batched_complex32_right_c() {
    let (ctx, stream) = setup();
    let b: i32 = 2;
    let m: i32 = 3;
    let n: i32 = 4;
    let k = n;
    let a_host = build_matrix_complex32(b as usize, n as usize, n as usize, 0x21_4365_87);
    let (a_packed, tau) =
        run_cusolver_geqrf_per_slot_complex32(&ctx, &stream, &a_host, b, n, n);
    let c_init = build_matrix_complex32(b as usize, m as usize, n as usize, 0xA9_CBED_0F);

    let op = BatchedOrmqrOp::C;
    let side = BatchedOrmqrSide::Right;
    let bespoke = bespoke_ormqr_complex32(
        &ctx, &stream, &a_packed, &tau, &c_init, b, m, n, k, side, op,
    );
    let reference = cusolver_unmqr_per_slot_complex32(
        &ctx, &stream, &a_packed, &tau, &c_init, b, m, n, k, side, op,
    );
    let tol = 32.0 * f32::EPSILON;
    check_complex32(&bespoke, &reference, tol,
        &format!("Complex32 unmqr_batched side=Right op={}", fmt_op(op)));
}

#[test]
#[ignore]
fn ormqr_batched_complex64_right_n() {
    let (ctx, stream) = setup();
    let b: i32 = 2;
    let m: i32 = 3;
    let n: i32 = 4;
    let k = n;
    let a_host = build_matrix_complex64(b as usize, n as usize, n as usize, 0xFE_DCBA_98);
    let (a_packed, tau) =
        run_cusolver_geqrf_per_slot_complex64(&ctx, &stream, &a_host, b, n, n);
    let c_init = build_matrix_complex64(b as usize, m as usize, n as usize, 0x76_5432_10);

    let op = BatchedOrmqrOp::N;
    let side = BatchedOrmqrSide::Right;
    let bespoke = bespoke_ormqr_complex64(
        &ctx, &stream, &a_packed, &tau, &c_init, b, m, n, k, side, op,
    );
    let reference = cusolver_unmqr_per_slot_complex64(
        &ctx, &stream, &a_packed, &tau, &c_init, b, m, n, k, side, op,
    );
    let tol = 64.0 * f64::EPSILON;
    check_complex64(&bespoke, &reference, tol,
        &format!("Complex64 unmqr_batched side=Right op={}", fmt_op(op)));
}

#[test]
#[ignore]
fn ormqr_batched_complex64_right_c() {
    let (ctx, stream) = setup();
    let b: i32 = 2;
    let m: i32 = 3;
    let n: i32 = 4;
    let k = n;
    let a_host = build_matrix_complex64(b as usize, n as usize, n as usize, 0xEF_CDAB_89);
    let (a_packed, tau) =
        run_cusolver_geqrf_per_slot_complex64(&ctx, &stream, &a_host, b, n, n);
    let c_init = build_matrix_complex64(b as usize, m as usize, n as usize, 0x67_4523_01);

    let op = BatchedOrmqrOp::C;
    let side = BatchedOrmqrSide::Right;
    let bespoke = bespoke_ormqr_complex64(
        &ctx, &stream, &a_packed, &tau, &c_init, b, m, n, k, side, op,
    );
    let reference = cusolver_unmqr_per_slot_complex64(
        &ctx, &stream, &a_packed, &tau, &c_init, b, m, n, k, side, op,
    );
    let tol = 64.0 * f64::EPSILON;
    check_complex64(&bespoke, &reference, tol,
        &format!("Complex64 unmqr_batched side=Right op={}", fmt_op(op)));
}
