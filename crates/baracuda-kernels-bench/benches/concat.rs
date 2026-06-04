//! Concat (2-input `torch.cat`) throughput — baracuda self-bench + PyTorch reference.
//!
//! Phase 73.8a — a baracuda load-bearing shape op. The canonical use
//! case is the **KV-cache concatenation step** in autoregressive LLM
//! inference, where each generated token appends its new K and V rows
//! to the cumulative cache. The bench sweeps that decode-step pattern
//! at LLM-typical shapes:
//!
//! - Append-1 (decode step):  `[B*H, K_prev, D]  ⊕  [B*H, 1, D]`
//! - Mid-sequence joins:      `[B*H, K_a,  D]    ⊕  [B*H, K_b, D]`
//!
//! Compared against `torch.cat([a, b], dim=1)` via the frozen JSON
//! baseline. No NVIDIA-library reference (cuDNN/cuBLAS don't have a
//! direct concat).
//!
//! Bench scope: f32 + f16. concat_dim=1 (the sequence axis). Rank-3
//! tensors. Shape labels `BH{bh}_Ka{ka}_Kb{kb}_D{d}`.

use baracuda_driver::DeviceBuffer;
use baracuda_kernels::{
    contiguous_stride, ConcatArgs, ConcatDescriptor, ConcatPlan, ElementKind, PlanPreference,
    TensorMut, TensorRef, Workspace,
};
use baracuda_kernels_bench::{
    append_csv_row, measure_median_ns, setup_device, time_with_events, warmup,
    PhaseTwentyNineRow, PytorchBaseline,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use half::f16;

const BENCH_NAME: &str = "concat";

/// Concat shape sweep: `(B*H, K_a, K_b, D)`. KV-cache decode shape (Ka=2047, Kb=1)
/// + two mid-sequence joins.
const CONCAT_SWEEP: &[(i32, i32, i32, i32)] = &[
    // KV-cache append-1: append one new token to a 2047-long cache.
    (32, 2047, 1, 128),  // Llama-7B per-head layout.
    // Mid-sequence joins.
    (32, 1024, 1024, 128),
    (32, 512, 512, 128),
];

fn leak_str(s: &str) -> &'static str {
    Box::leak(s.to_owned().into_boxed_str())
}

fn bench<T>(
    c: &mut Criterion,
    dtype_label: &str,
    kind: ElementKind,
    fill: T,
    baseline: Option<&PytorchBaseline>,
) where
    T: baracuda_kernels::Element + Copy + 'static,
{
    let (ctx, stream) = setup_device();
    let mut group = c.benchmark_group(format!("concat/{dtype_label}"));

    for &(bh, ka, kb, d) in CONCAT_SWEEP {
        let label = format!("BH{bh}_Ka{ka}_Kb{kb}_D{d}");
        let a_numel = (bh * ka * d) as usize;
        let b_numel = (bh * kb * d) as usize;
        let y_numel = (bh * (ka + kb) * d) as usize;

        let host_a: Vec<T> = vec![fill; a_numel];
        let host_b: Vec<T> = vec![fill; b_numel];

        let dev_a = match DeviceBuffer::from_slice(&ctx, &host_a) {
            Ok(buf) => buf,
            Err(_) => continue,
        };
        let dev_b = match DeviceBuffer::from_slice(&ctx, &host_b) {
            Ok(buf) => buf,
            Err(_) => continue,
        };
        let mut dev_y: DeviceBuffer<T> = match DeviceBuffer::zeros(&ctx, y_numel) {
            Ok(buf) => buf,
            Err(_) => continue,
        };

        let a_shape = [bh, ka, d];
        let b_shape = [bh, kb, d];
        let y_shape = [bh, ka + kb, d];
        let sta = contiguous_stride(a_shape);
        let stb = contiguous_stride(b_shape);
        let sty = contiguous_stride(y_shape);

        let desc = ConcatDescriptor::<3> {
            a_shape,
            b_shape,
            concat_dim: 1,
            element: kind,
        };
        let plan =
            match ConcatPlan::<T, 3>::select(&stream, &desc, PlanPreference::default()) {
                Ok(p) => p,
                Err(_) => continue,
            };

        warmup(&stream, || {
            let args = ConcatArgs::<T, 3> {
                a: TensorRef {
                    data: dev_a.as_slice(),
                    shape: a_shape,
                    stride: sta,
                },
                b: TensorRef {
                    data: dev_b.as_slice(),
                    shape: b_shape,
                    stride: stb,
                },
                y: TensorMut {
                    data: dev_y.as_slice_mut(),
                    shape: y_shape,
                    stride: sty,
                },
            };
            plan.run(&stream, Workspace::None, args).expect("baracuda concat");
        });
        let baracuda_ns = measure_median_ns(&ctx, &stream, 11, 50, || {
            let args = ConcatArgs::<T, 3> {
                a: TensorRef {
                    data: dev_a.as_slice(),
                    shape: a_shape,
                    stride: sta,
                },
                b: TensorRef {
                    data: dev_b.as_slice(),
                    shape: b_shape,
                    stride: stb,
                },
                y: TensorMut {
                    data: dev_y.as_slice_mut(),
                    shape: y_shape,
                    stride: sty,
                },
            };
            plan.run(&stream, Workspace::None, args).expect("baracuda concat");
        });
        append_csv_row(
            BENCH_NAME,
            &PhaseTwentyNineRow {
                op: "concat",
                shape: label.clone(),
                dtype: leak_str(dtype_label),
                baracuda_ns,
                reference_ns: None,
                reference: "",
                pytorch_ns: baseline.and_then(|b| b.lookup("concat", &label, dtype_label)),
            },
        );
        group.bench_with_input(BenchmarkId::from_parameter(&label), &(), |bb, _| {
            bb.iter_custom(|iters| {
                time_with_events(&ctx, &stream, iters, || {
                    let args = ConcatArgs::<T, 3> {
                        a: TensorRef {
                            data: dev_a.as_slice(),
                            shape: a_shape,
                            stride: sta,
                        },
                        b: TensorRef {
                            data: dev_b.as_slice(),
                            shape: b_shape,
                            stride: stb,
                        },
                        y: TensorMut {
                            data: dev_y.as_slice_mut(),
                            shape: y_shape,
                            stride: sty,
                        },
                    };
                    plan.run(&stream, Workspace::None, args).expect("baracuda concat");
                })
            });
        });
    }
    group.finish();
}

/// Top-level criterion entry - invoked by criterion_main!.
fn benches(c: &mut Criterion) {
    let baseline = PytorchBaseline::load_default();
    let baseline_ref = baseline.as_ref();
    bench::<f32>(c, "f32", ElementKind::F32, 1.0_f32, baseline_ref);
    bench::<f16>(c, "f16", ElementKind::F16, f16::ONE, baseline_ref);
}

// `criterion_group!` expands into a `pub fn benches_grp` whose
// signature is fixed by the macro - can't doc-comment it directly, so
// suppress the workspace `missing_docs = deny` lint on the generated fn.
#[allow(missing_docs)]
mod criterion_glue {
    use super::*;
    criterion_group!(benches_grp, benches);
}

criterion_main!(criterion_glue::benches_grp);
