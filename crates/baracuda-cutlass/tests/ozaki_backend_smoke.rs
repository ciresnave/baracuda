//! Real-GPU smoke test for the `BackendKind::Ozaki` FP64 `GemmPlan` path (vendored ozIMMU).
//!
//! Gated behind the `device-tests` feature and the `ozimmu` feature. It also pins the
//! thread-teardown contract of the per-thread ozIMMU handle cache: each iteration runs on a
//! fresh thread whose `Context` is dropped *before* the thread exits, so the cached handle's
//! destructor runs after the caller's context is gone (the access violation the cache's
//! `Context` clone prevents; see the matching cuBLAS cache note in `plan.rs`).
#![cfg(all(feature = "device-tests", feature = "ozimmu"))]

use baracuda_cutlass::{
    BackendKind, EpilogueKind, GemmArgs, GemmDescriptor, GemmPlan, LayoutSku, MatrixMut, MatrixRef,
    PlanPreference, Workspace,
};
use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};

const M: i32 = 64;
const N: i32 = 64;
const K: i32 = 64;

/// One Ozaki DGEMM on the calling thread; returns the max abs error vs a CPU DGEMM.
fn run_ozaki_once() -> f64 {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let (m, n, k) = (M as usize, N as usize, K as usize);
    let host_a: Vec<f64> = (0..m * k).map(|i| ((i as f64) * 0.01).sin()).collect();
    let host_b: Vec<f64> = (0..k * n).map(|i| ((i as f64) * 0.013).cos()).collect();
    // Rcr: A row-major [m,k], B column-major [k,n] stored as [n,k] row-major.
    let mut expected = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            expected[i * n + j] = (0..k)
                .map(|kk| host_a[i * k + kk] * host_b[j * k + kk])
                .sum();
        }
    }

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload A");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload B");
    let mut dev_d: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, m * n).expect("alloc D");

    let desc = GemmDescriptor {
        m: M,
        n: N,
        k: K,
        layout: LayoutSku::Rcr,
        epilogue: EpilogueKind::Identity,
    };
    let pref = PlanPreference {
        prefer_backend: Some(BackendKind::Ozaki { slices: 8 }),
        ..PlanPreference::default()
    };
    let plan = GemmPlan::<f64>::select(&stream, &desc, pref).expect("plan select");
    assert_eq!(plan.backend(), BackendKind::Ozaki { slices: 8 });

    let args = GemmArgs::<f64> {
        a: MatrixRef::new(dev_a.as_slice(), M, K, K as i64).unwrap(),
        b: MatrixRef::new(dev_b.as_slice(), K, N, K as i64).unwrap(),
        c: None,
        d: MatrixMut::new(dev_d.as_slice_mut(), M, N, N as i64).unwrap(),
        bias: None,
        alpha: 1.0,
        beta: 0.0,
    };
    plan.can_implement(&args).expect("can_implement");
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut out = vec![0.0f64; m * n];
    dev_d.copy_to_host(&mut out).expect("download D");
    out.iter()
        .zip(&expected)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0, f64::max)
}

#[test]
fn ozaki_s8_matches_cpu_dgemm() {
    let err = run_ozaki_once();
    // Ozaki at S=8 is "comparable to DGEMM on well-conditioned inputs", not bit-exact.
    assert!(err < 1e-9, "Ozaki S=8 max abs err {err}");
}

#[test]
fn ozaki_handle_cache_survives_thread_exit_after_context_drop() {
    // Fresh thread per iteration: its thread-local handle cache is destroyed at thread exit,
    // after `run_ozaki_once` has dropped its Context.
    for i in 0..8 {
        let err = std::thread::spawn(run_ozaki_once)
            .join()
            .unwrap_or_else(|_| panic!("iteration {i}: thread panicked or crashed"));
        assert!(err < 1e-9, "iteration {i}: max abs err {err}");
    }
}
