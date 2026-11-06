//! Concurrency stress for cuBLAS / cuSOLVER handle creation in the linalg
//! plans.
//!
//! `cublasCreate_v2` returned `NOT_INITIALIZED` when several threads built a
//! plan at once, so `BatchedQrPlan::run` failed intermittently with
//! `KernelInternal(-1)` (about 15% of 32-thread runs of the ormqr smoke
//! suite). Every thread here builds its own context, stream and plan and runs
//! it, released together by a barrier so the first handle creations overlap,
//! and the whole round is repeated in fresh child processes because the race
//! is in cold-start library initialisation. Any `Err` fails the test.
//!
//! Gated behind the `device-tests` feature — requires a real CUDA device.
#![cfg(feature = "device-tests")]

use std::sync::Barrier;

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    BatchedQrArgs, BatchedQrDescriptor, BatchedQrPlan, ElementKind, PlanPreference, QrArgs,
    QrDescriptor, QrPlan, TensorMut, Workspace, contiguous_stride,
};

const THREADS: usize = 32;
const PROCESSES: usize = 40;
const B: i32 = 4;
const M: i32 = 16;
const N: i32 = 8;

/// One thread's work: own context, stream, plan; one batched QR. Panics on any
/// error, which fails the worker (and, in a child process, the child).
fn build_and_run(device: &Device) {
    let ctx = Context::new(device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    let host: Vec<f32> = (0..(B * M * N) as usize)
        .map(|i| ((i * 31 % 17) as f32) * 0.1 - 0.8)
        .collect();
    let mut dev_a = DeviceBuffer::from_slice(&ctx, &host).expect("upload");
    let k = M.min(N);
    let mut dev_tau: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, (B * k) as usize).expect("tau");
    let desc = BatchedQrDescriptor {
        m: M,
        n: N,
        batch_size: B,
        element: ElementKind::F32,
    };
    let plan =
        BatchedQrPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let ws_bytes = plan.query_workspace_size(&stream).expect("workspace query");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("workspace");
    let a_shape = [B, M, N];
    let tau_shape = [B, k];
    let args = BatchedQrArgs::<f32> {
        a: TensorMut::new(dev_a.as_slice_mut(), a_shape, contiguous_stride(a_shape))
            .expect("a view"),
        tau: TensorMut::new(
            dev_tau.as_slice_mut(),
            tau_shape,
            contiguous_stride(tau_shape),
        )
        .expect("tau view"),
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run");
    stream.synchronize().expect("sync");
}

/// The cuSOLVER side: `QrPlan` creates a `cusolverDn` handle, which
/// initialises cuBLAS internally, so it races the cuBLAS-only plans above.
fn build_and_run_cusolver(device: &Device) {
    let ctx = Context::new(device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    let (m, n) = (6i32, 4i32);
    let host: Vec<f32> = (0..(m * n) as usize)
        .map(|i| ((i * 13 % 11) as f32) * 0.1 - 0.5)
        .collect();
    let mut dev_a = DeviceBuffer::from_slice(&ctx, &host).expect("upload");
    let mut dev_q: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, (m * m) as usize).expect("q");
    let mut dev_r: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, (m * n) as usize).expect("r");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("info");
    let desc = QrDescriptor {
        m,
        n,
        element: ElementKind::F32,
    };
    let plan =
        QrPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("cusolver select");
    let ws_bytes = plan
        .query_workspace_size(&stream)
        .expect("cusolver workspace query");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("workspace");
    let (a_shape, q_shape, r_shape) = ([m, n], [m, m], [m, n]);
    let args = QrArgs::<f32> {
        a: TensorMut::new(dev_a.as_slice_mut(), a_shape, contiguous_stride(a_shape))
            .expect("a view"),
        q: TensorMut::new(dev_q.as_slice_mut(), q_shape, contiguous_stride(q_shape))
            .expect("q view"),
        r: TensorMut::new(dev_r.as_slice_mut(), r_shape, contiguous_stride(r_shape))
            .expect("r view"),
        info: TensorMut::new(dev_info.as_slice_mut(), [1i32], [1]).expect("info view"),
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("cusolver run");
    stream.synchronize().expect("sync");
}

/// Set in the re-executed copies of this test: run one cold-start round.
const CHILD_ENV: &str = "BARACUDA_HANDLE_INIT_CHILD";
const TEST_NAME: &str = "concurrent_plan_creation_never_fails";

/// One round: `THREADS` threads released together by a barrier, each building
/// and running a plan. Returns how many workers panicked (their panic messages
/// go to stderr, which the parent captures).
fn one_round() -> usize {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let barrier = Barrier::new(THREADS);
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..THREADS)
            .map(|t| {
                let (barrier, device) = (&barrier, &device);
                s.spawn(move || {
                    barrier.wait();
                    if t % 2 == 0 {
                        build_and_run(device);
                    } else {
                        build_and_run_cusolver(device);
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join())
            .filter(Result::is_err)
            .count()
    })
}

/// Run one cold-start round in this process; used by the re-executed children.
fn child_round() {
    let panicked = one_round();
    assert_eq!(panicked, 0, "{panicked} of {THREADS} workers failed");
}

/// Start one fresh copy of this test binary running only this test, in child
/// mode. Returns `(succeeded, stdout, stderr)`.
fn run_child() -> (bool, String, String) {
    // The binary re-runs itself (a test of its own cold start): there is no
    // privilege or trust decision attached to this path.
    let exe = std::env::current_exe().expect("current_exe");
    let out = std::process::Command::new(exe)
        .args(["--exact", TEST_NAME, "--test-threads=1", "--nocapture"])
        .env(CHILD_ENV, "1")
        .output()
        .expect("spawn child");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// The race is in first-use library initialisation, so one process gives one
/// chance to hit it. The parent therefore re-executes this very test in
/// `PROCESSES` fresh child processes (each a cold start with `THREADS`
/// racing threads) and fails if any child saw an error.
#[test]
fn concurrent_plan_creation_never_fails() {
    if std::env::var_os(CHILD_ENV).is_some() {
        child_round();
        return;
    }
    let mut failed: Vec<String> = Vec::new();
    for i in 0..PROCESSES {
        let (ok, stdout, stderr) = run_child();
        // `--exact` matching nothing would exit 0 having run zero tests.
        assert!(
            !ok || stdout.contains("1 passed"),
            "child {i} ran no test:
{stdout}"
        );
        if !ok {
            failed.push(format!("child {i}: {}", stderr.trim()));
        }
    }
    assert!(
        failed.is_empty(),
        "{} of {PROCESSES} cold-start rounds ({THREADS} threads each) failed:
{}",
        failed.len(),
        failed.join(
            "
"
        )
    );
}
