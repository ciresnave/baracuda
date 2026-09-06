//! Dense Linalg throughput — baracuda self-bench + PyTorch reference.
//!
//! Wave-2 of the cross-implementation bench suite (Loss was Wave-1). cuSOLVER
//! backs this family and its dense API exposes **f32 + f64 only**, so unlike the
//! Loss wave there is no f16/bf16 axis — that is a property of the backend, not
//! a gap in the sweep.
//!
//! ⚠️ EVERY OP IN THIS FAMILY CONSUMES ITS INPUT IN PLACE. All eight Linalg
//! `*Args` structs take `a: TensorMut`, and `solve` additionally overwrites `b`.
//! A plain timing loop therefore measures the first iteration against the real
//! input and every later one against whatever the previous call left behind.
//!
//! Cholesky is the sharpest case and the reason this bench uses
//! `*_restored`: after call 1 the buffer holds `L`, which is not symmetric
//! positive-definite, so cuSOLVER halts at the failing minor and returns early.
//! Iterations 2..N measure a FAILURE PATH, the median lands far below the true
//! cost, and nothing raises — `info[i] != 0` records it and a timing loop never
//! reads `info`. The published number would simply be wrong.
//!
//! So each iteration restores the pristine input by device-to-device copy and
//! the restore is fenced out of the measurement by its own event pair.
//!
//! **Sample counts are deliberately lower than the elementwise benches.** A
//! factorization is milliseconds where an elementwise kernel is microseconds;
//! the usual `samples = 11, inner = 50` would be 550 launches per cell.
//!
//! Tranche 1 covers `cholesky` and `lu`. Deliberately excluded, so
//! the absence is a decision rather than an oversight:
//!
//! - `svd` / `eigh` — iterative, and an order of magnitude slower per call.
//!   They need their own shape sweep rather than sharing this one.
//! - `lstsq` — mixed-precision iterative refinement, so its cost is
//!   data-dependent in a way that needs a stated conditioning fixture first.
//! - `eig` (non-symmetric) — `torch.linalg.eig` returns complex even for real
//!   input, so the reference is not a like-for-like comparison.
//! - `qr` / `solve` — same in-place shape and same restore, but each needs its
//!   own output buffers (`q`/`r`; `b`/`pivot`); they follow once this tranche's
//!   restore discipline is validated on device.
//! - every batched variant — a separate tranche.

use baracuda_driver::DeviceBuffer;
use baracuda_kernels::{
    CholeskyArgs, CholeskyDescriptor, CholeskyPlan, ElementKind, LuArgs, LuDescriptor, LuPlan,
    PlanPreference, TensorMut, Workspace, contiguous_stride,
};
use baracuda_kernels_bench::{
    PhaseTwentyNineRow, PytorchBaseline, append_csv_row, measure_median_ns_restored, setup_device,
    time_with_events_restored, warmup,
};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

const BENCH_NAME: &str = "linalg";

/// Square sizes. `O(n^3)` work, so the sweep stays modest: at n=512 an f64
/// factorization is already milliseconds.
const N_SWEEP: &[i32] = &[256, 512];

/// Median over `SAMPLES` samples of `INNER` launches each — 15 launches per
/// cell, against 550 for the elementwise benches. See the module note.
const SAMPLES: usize = 5;
const INNER: u64 = 3;

fn leak_str(s: &str) -> &'static str {
    Box::leak(s.to_owned().into_boxed_str())
}

/// A symmetric, diagonally dominant `n x n` matrix — SPD by Gershgorin, so
/// Cholesky succeeds, and well-conditioned enough that LU / QR / solve are not
/// measuring a pathological case.
fn spd_host<T: From<f32> + Copy>(n: usize) -> Vec<T> {
    let mut a = vec![T::from(0.5_f32); n * n];
    for i in 0..n {
        a[i * n + i] = T::from(n as f32);
    }
    a
}

/// One `(op, dtype, n)` cell: time it, print, and append the CSV row that
/// reaches `BENCHMARKS.md`.
#[allow(clippy::too_many_arguments)]
fn record(
    c: &mut Criterion,
    ctx: &baracuda_driver::Context,
    stream: &baracuda_driver::Stream,
    op: &str,
    dtype_label: &str,
    n: i32,
    baseline: Option<&PytorchBaseline>,
    mut restore: impl FnMut(),
    mut launch: impl FnMut(),
) {
    let shape = format!("N{n}");
    warmup(stream, || {
        restore();
        launch();
    });

    let baracuda_ns =
        measure_median_ns_restored(ctx, stream, SAMPLES, INNER, &mut restore, &mut launch);

    let pytorch_ns = baseline.and_then(|b| b.lookup(op, &shape, dtype_label));
    let mut group = c.benchmark_group(format!("{op}/{dtype_label}"));
    group.bench_with_input(BenchmarkId::from_parameter(&shape), &(), |bb, _| {
        bb.iter_custom(|iters| {
            time_with_events_restored(ctx, stream, iters, &mut restore, &mut launch)
        });
    });
    group.finish();

    append_csv_row(
        BENCH_NAME,
        &PhaseTwentyNineRow {
            op: leak_str(op),
            shape: shape.clone(),
            dtype: leak_str(dtype_label),
            baracuda_ns,
            reference_ns: None,
            reference: "",
            pytorch_ns,
        },
    );
}

fn bench_f32(c: &mut Criterion, baseline: Option<&PytorchBaseline>) {
    let (ctx, stream) = setup_device();

    for &n in N_SWEEP {
        let nu = n as usize;
        let host = spd_host::<f32>(nu);

        // ---- cholesky ----
        {
            let pristine = match DeviceBuffer::from_slice(&ctx, &host) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let work = match DeviceBuffer::from_slice(&ctx, &host) {
                // RefCell because `restore` needs `&work` (copy destination) and
                // `launch` needs `&mut work`, and `record` holds both closures at
                // once. They never run concurrently, so the runtime check never
                // trips; the compiler simply cannot see the alternation.
                Ok(b) => std::cell::RefCell::new(b),
                Err(_) => continue,
            };
            let mut info: DeviceBuffer<i32> = match DeviceBuffer::zeros(&ctx, 1) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let desc = CholeskyDescriptor {
                matrix_size: n,
                batch_size: 1,
                lower: true,
                element: ElementKind::F32,
            };
            if let Ok(plan) = CholeskyPlan::<f32>::select(&stream, &desc, PlanPreference::default())
            {
                // `Workspace::None` is an ERROR, not an auto-allocate: cuSOLVER's
                // scratch is caller-supplied and `unpack_workspace` returns
                // `WorkspaceTooSmall { got: 0 }` whenever `needed > 0`. The size
                // is only known after the query, so ask, then allocate.
                let ws_bytes = plan.query_workspace_size(&stream).unwrap_or(0).max(1);
                let mut ws: DeviceBuffer<u8> = match DeviceBuffer::zeros(&ctx, ws_bytes) {
                    Ok(b) => b,
                    Err(_) => continue,
                };
                let sh = [1, n, n];
                let st = contiguous_stride(sh);
                record(
                    c,
                    &ctx,
                    &stream,
                    "cholesky",
                    "f32",
                    n,
                    baseline,
                    || {
                        pristine
                            .copy_to_device_async(&work.borrow(), &stream)
                            .expect("restore cholesky input");
                    },
                    || {
                        let mut w = work.borrow_mut();
                        let args = CholeskyArgs::<f32> {
                            a: TensorMut {
                                data: w.as_slice_mut(),
                                shape: sh,
                                stride: st,
                            },
                            info: TensorMut {
                                data: info.as_slice_mut(),
                                shape: [1],
                                stride: [1],
                            },
                        };
                        plan.run(&stream, Workspace::Borrowed(ws.as_slice_mut()), args)
                            .expect("cholesky");
                    },
                );
            }
        }

        // ---- lu ----
        {
            let pristine = match DeviceBuffer::from_slice(&ctx, &host) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let work = match DeviceBuffer::from_slice(&ctx, &host) {
                // RefCell because `restore` needs `&work` (copy destination) and
                // `launch` needs `&mut work`, and `record` holds both closures at
                // once. They never run concurrently, so the runtime check never
                // trips; the compiler simply cannot see the alternation.
                Ok(b) => std::cell::RefCell::new(b),
                Err(_) => continue,
            };
            let mut pivot: DeviceBuffer<i32> = match DeviceBuffer::zeros(&ctx, nu) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let mut info: DeviceBuffer<i32> = match DeviceBuffer::zeros(&ctx, 1) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let desc = LuDescriptor {
                m: n,
                n,
                batch_size: 1,
                element: ElementKind::F32,
            };
            if let Ok(plan) = LuPlan::<f32>::select(&stream, &desc, PlanPreference::default()) {
                // `Workspace::None` is an ERROR, not an auto-allocate: cuSOLVER's
                // scratch is caller-supplied and `unpack_workspace` returns
                // `WorkspaceTooSmall { got: 0 }` whenever `needed > 0`. The size
                // is only known after the query, so ask, then allocate.
                let ws_bytes = plan.query_workspace_size(&stream).unwrap_or(0).max(1);
                let mut ws: DeviceBuffer<u8> = match DeviceBuffer::zeros(&ctx, ws_bytes) {
                    Ok(b) => b,
                    Err(_) => continue,
                };
                let sh = [1, n, n];
                let st = contiguous_stride(sh);
                record(
                    c,
                    &ctx,
                    &stream,
                    "lu",
                    "f32",
                    n,
                    baseline,
                    || {
                        pristine
                            .copy_to_device_async(&work.borrow(), &stream)
                            .expect("restore lu input");
                    },
                    || {
                        let mut w = work.borrow_mut();
                        let args = LuArgs::<f32> {
                            a: TensorMut {
                                data: w.as_slice_mut(),
                                shape: sh,
                                stride: st,
                            },
                            pivot: TensorMut {
                                data: pivot.as_slice_mut(),
                                shape: [1, n],
                                stride: contiguous_stride([1, n]),
                            },
                            info: TensorMut {
                                data: info.as_slice_mut(),
                                shape: [1],
                                stride: [1],
                            },
                        };
                        plan.run(&stream, Workspace::Borrowed(ws.as_slice_mut()), args)
                            .expect("lu");
                    },
                );
            }
        }
    }
}

fn benches(c: &mut Criterion) {
    let baseline = PytorchBaseline::load_default();
    if baseline.is_none() {
        eprintln!("linalg: no PyTorch baseline found; the PyTorch column will be empty");
    }
    bench_f32(c, baseline.as_ref());
}

// `criterion_group!` expands into a `pub fn` whose signature is fixed by the
// macro - can't doc-comment it directly, so suppress the workspace
// `missing_docs = deny` lint on the generated fn. Same glue as `loss.rs`.
#[allow(missing_docs)]
mod criterion_glue {
    use super::*;
    criterion_group!(benches_grp, benches);
}
criterion_main!(criterion_glue::benches_grp);
