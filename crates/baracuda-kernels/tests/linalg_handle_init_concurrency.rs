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

/// One thread's work: own context, stream, plan; one batched QR. Returns the
/// failure text instead of panicking so the test can report every failure.
fn build_and_run(device: &Device) -> Result<(), String> {
    let ctx = Context::new(device).map_err(|e| format!("context: {e:?}"))?;
    let stream = Stream::new(&ctx).map_err(|e| format!("stream: {e:?}"))?;
    let host: Vec<f32> = (0..(B * M * N) as usize)
        .map(|i| ((i * 31 % 17) as f32) * 0.1 - 0.8)
        .collect();
    let mut dev_a = DeviceBuffer::from_slice(&ctx, &host).map_err(|e| format!("upload: {e:?}"))?;
    let k = M.min(N);
    let mut dev_tau: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (B * k) as usize).map_err(|e| format!("tau: {e:?}"))?;
    let desc = BatchedQrDescriptor {
        m: M,
        n: N,
        batch_size: B,
        element: ElementKind::F32,
    };
    let plan = BatchedQrPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .map_err(|e| format!("select: {e:?}"))?;
    let ws_bytes = plan
        .query_workspace_size(&stream)
        .map_err(|e| format!("workspace query: {e:?}"))?;
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, ws_bytes).map_err(|e| format!("workspace: {e:?}"))?;
    let a_shape = [B, M, N];
    let tau_shape = [B, k];
    let args = BatchedQrArgs::<f32> {
        a: TensorMut::new(dev_a.as_slice_mut(), a_shape, contiguous_stride(a_shape))
            .map_err(|e| format!("a view: {e:?}"))?,
        tau: TensorMut::new(
            dev_tau.as_slice_mut(),
            tau_shape,
            contiguous_stride(tau_shape),
        )
        .map_err(|e| format!("tau view: {e:?}"))?,
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .map_err(|e| format!("run: {e:?}"))?;
    stream.synchronize().map_err(|e| format!("sync: {e:?}"))
}

/// The cuSOLVER side: `QrPlan` creates a `cusolverDn` handle, which
/// initialises cuBLAS internally, so it races the cuBLAS-only plans above.
fn build_and_run_cusolver(device: &Device) -> Result<(), String> {
    let ctx = Context::new(device).map_err(|e| format!("context: {e:?}"))?;
    let stream = Stream::new(&ctx).map_err(|e| format!("stream: {e:?}"))?;
    let (m, n) = (6i32, 4i32);
    let host: Vec<f32> = (0..(m * n) as usize)
        .map(|i| ((i * 13 % 11) as f32) * 0.1 - 0.5)
        .collect();
    let mut dev_a = DeviceBuffer::from_slice(&ctx, &host).map_err(|e| format!("upload: {e:?}"))?;
    let mut dev_q: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (m * m) as usize).map_err(|e| format!("q: {e:?}"))?;
    let mut dev_r: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (m * n) as usize).map_err(|e| format!("r: {e:?}"))?;
    let mut dev_info: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, 1).map_err(|e| format!("info: {e:?}"))?;
    let desc = QrDescriptor {
        m,
        n,
        element: ElementKind::F32,
    };
    let plan = QrPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .map_err(|e| format!("cusolver select: {e:?}"))?;
    let ws_bytes = plan
        .query_workspace_size(&stream)
        .map_err(|e| format!("cusolver workspace query: {e:?}"))?;
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, ws_bytes).map_err(|e| format!("workspace: {e:?}"))?;
    let (a_shape, q_shape, r_shape) = ([m, n], [m, m], [m, n]);
    let args = QrArgs::<f32> {
        a: TensorMut::new(dev_a.as_slice_mut(), a_shape, contiguous_stride(a_shape))
            .map_err(|e| format!("a view: {e:?}"))?,
        q: TensorMut::new(dev_q.as_slice_mut(), q_shape, contiguous_stride(q_shape))
            .map_err(|e| format!("q view: {e:?}"))?,
        r: TensorMut::new(dev_r.as_slice_mut(), r_shape, contiguous_stride(r_shape))
            .map_err(|e| format!("r view: {e:?}"))?,
        info: TensorMut::new(dev_info.as_slice_mut(), [1i32], [1])
            .map_err(|e| format!("info view: {e:?}"))?,
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .map_err(|e| format!("cusolver run: {e:?}"))?;
    stream.synchronize().map_err(|e| format!("sync: {e:?}"))
}

/// Set in the re-executed copies of this test: run one cold-start round.
const CHILD_ENV: &str = "BARACUDA_HANDLE_INIT_CHILD";
const TEST_NAME: &str = "concurrent_plan_creation_never_fails";

/// One round: `THREADS` threads released together by a barrier, each building
/// and running a plan. Returns every failure.
fn one_round() -> Vec<String> {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let barrier = Barrier::new(THREADS);
    let results: Vec<Result<(), String>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..THREADS)
            .map(|t| {
                let (barrier, device) = (&barrier, &device);
                s.spawn(move || {
                    barrier.wait();
                    if t % 2 == 0 {
                        build_and_run(device)
                    } else {
                        build_and_run_cusolver(device)
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("worker panicked"))
            .collect()
    });
    results.into_iter().filter_map(Result::err).collect()
}

/// The race is in first-use library initialisation, so one process gives one
/// chance to hit it. The parent therefore re-executes this very test in
/// `PROCESSES` fresh child processes (each a cold start with `THREADS`
/// racing threads) and fails if any child saw an error.
#[test]
fn concurrent_plan_creation_never_fails() {
    if std::env::var_os(CHILD_ENV).is_some() {
        let failures = one_round();
        assert!(
            failures.is_empty(),
            "{}",
            failures.join(
                "
"
            )
        );
        return;
    }
    let exe = std::env::current_exe().expect("current_exe");
    let mut failed: Vec<String> = Vec::new();
    for i in 0..PROCESSES {
        let out = std::process::Command::new(&exe)
            .args(["--exact", TEST_NAME, "--test-threads=1", "--nocapture"])
            .env(CHILD_ENV, "1")
            .output()
            .expect("spawn child");
        let text = String::from_utf8_lossy(&out.stderr);
        // `--exact` matching nothing would exit 0 having run zero tests.
        assert!(
            String::from_utf8_lossy(&out.stdout).contains("1 passed") || !out.status.success(),
            "child {i} ran no test:
{}",
            String::from_utf8_lossy(&out.stdout)
        );
        if !out.status.success() {
            failed.push(format!("child {i}: {}", text.trim()));
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
