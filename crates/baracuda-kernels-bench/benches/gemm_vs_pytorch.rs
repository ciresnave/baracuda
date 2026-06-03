//! GEMM throughput — baracuda vs PyTorch (LibTorch via tch-rs).
//!
//! Phase 73 trailblazer for the PyTorch comparison arm of the
//! cross-implementation bench suite. Mirrors `gemm_vs_cublas.rs` but
//! the reference is PyTorch's `Tensor::matmul` on a CUDA tensor instead
//! of cuBLAS direct.
//!
//! Gated behind the `pytorch` cargo feature. Without it, this file
//! compiles to a no-op `main` so the bench binary still builds.
//!
//! ## Setup
//!
//! See `BENCHMARKS.md` and the `pytorch` feature comment in `Cargo.toml`.
//! Short version: `pip install torch --index-url https://download.pytorch.org/whl/cu126`
//! then set `LIBTORCH_USE_PYTORCH=1` + `LIBTORCH_BYPASS_VERSION_CHECK=1`.
//!
//! ## Running
//!
//! ```bash
//! cargo bench -p baracuda-kernels-bench --bench gemm_vs_pytorch \
//!   --features sm89,pytorch
//! ```
//!
//! ## Dtype coverage
//!
//! - `f32` — PyTorch `torch.matmul(a, b)` on `dtype=float32`.
//! - `f16` — `dtype=float16`, tensor-core path on Ada / Ampere.
//! - `bf16` — `dtype=bfloat16`, tensor-core path.

#[cfg(not(feature = "pytorch"))]
fn main() {
    eprintln!(
        "gemm_vs_pytorch: built without the `pytorch` feature; nothing to run.\n\
         Build with `--features sm89,pytorch` after installing LibTorch."
    );
}

#[cfg(feature = "pytorch")]
use baracuda_driver::DeviceBuffer;
#[cfg(feature = "pytorch")]
use baracuda_kernels::{
    EpilogueKind, GemmArgs, GemmDescriptor, GemmPlan, LayoutSku, MatrixMut, MatrixRef,
    PlanPreference, Workspace,
};
#[cfg(feature = "pytorch")]
use baracuda_kernels_bench::{
    append_csv_row, gemm_flops, measure_median_ns, setup_device, time_with_events, warmup,
    PhaseTwentyNineRow, CROSS_GEMM_KN_SWEEP, CROSS_GEMM_M_SWEEP,
};
#[cfg(feature = "pytorch")]
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
#[cfg(feature = "pytorch")]
use tch::{Cuda, Device, Kind, Tensor};

#[cfg(feature = "pytorch")]
const BENCH_NAME: &str = "gemm_vs_pytorch";

/// CUDA-event-equivalent timing for a tch tensor op. tch creates its
/// own internal CUDA streams; rather than try to wire baracuda's
/// stream events around tch's launches, we use wall-clock around an
/// explicit `Cuda::synchronize()` per inner iteration. This is the
/// same methodology PyTorch users use when benchmarking.
#[cfg(feature = "pytorch")]
fn time_tch_iter_custom<F>(iters: u64, mut launch: F) -> std::time::Duration
where
    F: FnMut(),
{
    // Warmup: matches the 10-launch warmup baracuda's `warmup()` does.
    for _ in 0..10 {
        launch();
    }
    Cuda::synchronize(0);
    let start = std::time::Instant::now();
    for _ in 0..iters {
        launch();
    }
    Cuda::synchronize(0);
    start.elapsed()
}

/// Measure PyTorch median wall-time per launch (ns) over `samples`
/// independent batches of `inner` launches. Mirrors
/// `measure_median_ns` for the tch path.
#[cfg(feature = "pytorch")]
fn measure_tch_median_ns<F>(samples: usize, inner: u64, mut launch: F) -> f64
where
    F: FnMut(),
{
    let mut measurements: Vec<f64> = Vec::with_capacity(samples);
    for _ in 0..samples {
        let dur = time_tch_iter_custom(inner, &mut launch);
        let ns = dur.as_secs_f64() * 1e9 / inner as f64;
        measurements.push(ns);
    }
    measurements.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    measurements[measurements.len() / 2]
}

/// Map `(m, k, n, Kind)` to a closure that runs one PyTorch matmul
/// on the GPU. The tensors are allocated once outside the timing
/// loop and reused — same as the baracuda side.
#[cfg(feature = "pytorch")]
fn build_tch_matmul(m: i32, k: i32, n: i32, kind: Kind) -> impl FnMut() {
    let device = Device::Cuda(0);
    let a = Tensor::ones(&[m as i64, k as i64], (kind, device));
    let b = Tensor::ones(&[k as i64, n as i64], (kind, device));
    move || {
        let _y = a.matmul(&b);
    }
}

/// PyTorch f32 matmul.
#[cfg(feature = "pytorch")]
fn bench_pytorch_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_vs_pytorch/pytorch/f32");
    for &kn in CROSS_GEMM_KN_SWEEP {
        for &m in CROSS_GEMM_M_SWEEP {
            let n = kn;
            let k = kn;
            let shape = format!("M{m}_N{n}_K{k}");
            group.throughput(Throughput::Bytes(gemm_flops(m, n, k) as u64));
            let mut launch = build_tch_matmul(m, k, n, Kind::Float);
            group.bench_with_input(BenchmarkId::from_parameter(&shape), &shape, |b, _| {
                b.iter_custom(|iters| time_tch_iter_custom(iters, &mut launch));
            });
            let median_ns = measure_tch_median_ns(11, 50, &mut launch);
            append_csv_row(
                BENCH_NAME,
                &PhaseTwentyNineRow {
                    op: "gemm",
                    shape: shape.clone(),
                    dtype: "f32",
                    baracuda_ns: f64::NAN,
                    reference_ns: Some(median_ns),
                    reference: "PyTorch",
                },
            );
        }
    }
    group.finish();
}

/// PyTorch f16 matmul (tensor cores).
#[cfg(feature = "pytorch")]
fn bench_pytorch_f16(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_vs_pytorch/pytorch/f16");
    for &kn in CROSS_GEMM_KN_SWEEP {
        for &m in CROSS_GEMM_M_SWEEP {
            let n = kn;
            let k = kn;
            let shape = format!("M{m}_N{n}_K{k}");
            group.throughput(Throughput::Bytes(gemm_flops(m, n, k) as u64));
            let mut launch = build_tch_matmul(m, k, n, Kind::Half);
            group.bench_with_input(BenchmarkId::from_parameter(&shape), &shape, |b, _| {
                b.iter_custom(|iters| time_tch_iter_custom(iters, &mut launch));
            });
            let median_ns = measure_tch_median_ns(11, 50, &mut launch);
            append_csv_row(
                BENCH_NAME,
                &PhaseTwentyNineRow {
                    op: "gemm",
                    shape: shape.clone(),
                    dtype: "f16",
                    baracuda_ns: f64::NAN,
                    reference_ns: Some(median_ns),
                    reference: "PyTorch",
                },
            );
        }
    }
    group.finish();
}

/// PyTorch bf16 matmul (tensor cores).
#[cfg(feature = "pytorch")]
fn bench_pytorch_bf16(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_vs_pytorch/pytorch/bf16");
    for &kn in CROSS_GEMM_KN_SWEEP {
        for &m in CROSS_GEMM_M_SWEEP {
            let n = kn;
            let k = kn;
            let shape = format!("M{m}_N{n}_K{k}");
            group.throughput(Throughput::Bytes(gemm_flops(m, n, k) as u64));
            let mut launch = build_tch_matmul(m, k, n, Kind::BFloat16);
            group.bench_with_input(BenchmarkId::from_parameter(&shape), &shape, |b, _| {
                b.iter_custom(|iters| time_tch_iter_custom(iters, &mut launch));
            });
            let median_ns = measure_tch_median_ns(11, 50, &mut launch);
            append_csv_row(
                BENCH_NAME,
                &PhaseTwentyNineRow {
                    op: "gemm",
                    shape: shape.clone(),
                    dtype: "bf16",
                    baracuda_ns: f64::NAN,
                    reference_ns: Some(median_ns),
                    reference: "PyTorch",
                },
            );
        }
    }
    group.finish();
}

/// baracuda f32 RCR GEMM — mirrors `gemm_vs_cublas::bench_baracuda_f32`
/// for direct side-by-side comparison in criterion's HTML report.
#[cfg(feature = "pytorch")]
fn bench_baracuda_f32(c: &mut Criterion) {
    let (ctx, stream) = setup_device();
    let mut group = c.benchmark_group("gemm_vs_pytorch/baracuda/f32");
    for &kn in CROSS_GEMM_KN_SWEEP {
        for &m in CROSS_GEMM_M_SWEEP {
            let n = kn;
            let k = kn;
            let shape = format!("M{m}_N{n}_K{k}");

            let host_a: Vec<f32> = vec![1.0; (m * k) as usize];
            let host_b: Vec<f32> = vec![1.0; (k * n) as usize];
            let dev_a = match DeviceBuffer::from_slice(&ctx, &host_a) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let dev_b = match DeviceBuffer::from_slice(&ctx, &host_b) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let mut dev_d: DeviceBuffer<f32> =
                match DeviceBuffer::zeros(&ctx, (m * n) as usize) {
                    Ok(b) => b,
                    Err(_) => continue,
                };

            let desc = GemmDescriptor {
                m,
                n,
                k,
                layout: LayoutSku::Rcr,
                epilogue: EpilogueKind::Identity,
            };
            let plan = match GemmPlan::<f32>::select(&stream, &desc, PlanPreference::default()) {
                Ok(p) => p,
                Err(_) => continue,
            };

            let run = || {
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
                plan.run(&stream, Workspace::None, args).expect("baracuda gemm");
            };
            warmup(&stream, run);

            group.throughput(Throughput::Bytes(gemm_flops(m, n, k) as u64));
            group.bench_with_input(BenchmarkId::from_parameter(&shape), &shape, |b, _| {
                b.iter_custom(|iters| time_with_events(&ctx, &stream, iters, run));
            });
            let median_ns = measure_median_ns(&ctx, &stream, 11, 50, run);
            append_csv_row(
                BENCH_NAME,
                &PhaseTwentyNineRow {
                    op: "gemm",
                    shape: shape.clone(),
                    dtype: "f32",
                    baracuda_ns: median_ns,
                    reference_ns: None,
                    reference: "baracuda-self",
                },
            );
        }
    }
    group.finish();
}

#[cfg(feature = "pytorch")]
criterion_group!(
    benches,
    bench_baracuda_f32,
    bench_pytorch_f32,
    bench_pytorch_f16,
    bench_pytorch_bf16
);
#[cfg(feature = "pytorch")]
criterion_main!(benches);
