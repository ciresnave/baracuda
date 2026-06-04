//! Top-K throughput — baracuda self-bench + PyTorch reference.
//!
//! Phase 73.8e — `torch.topk(x, k, dim=-1, largest=True)` for sampling,
//! speculative decoding, MoE expert dispatch. baracuda's `TopkPlan`
//! trailblazer caps `row_len ≤ 1024` and `k ≤ 64` (block-bitonic
//! comparator network). Sweep stays inside those caps:
//!
//! - MoE-style: batch=32, row_len=128, k=4 (top-4 experts per token)
//! - Intermediate: batch=8, row_len=512, k=16
//! - Largest: batch=1, row_len=1024, k=64 (trailblazer cap)
//!
//! baracuda supports f32 / f64 only — no f16 in the trailblazer. Bench
//! is f32. No NVIDIA-library equivalent.

use baracuda_driver::DeviceBuffer;
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef, TopkArgs,
    TopkDescriptor, TopkPlan, Workspace,
};
use baracuda_kernels_bench::{
    append_csv_row, measure_median_ns, setup_device, time_with_events, warmup,
    PhaseTwentyNineRow, PytorchBaseline,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

const BENCH_NAME: &str = "topk";

/// TopK sweep: `(batch, row_len, k)`. All within baracuda's trailblazer
/// caps (`row_len ≤ 1024`, `k ≤ 64`).
const TOPK_SWEEP: &[(i32, i32, i32)] = &[
    (32, 128, 4),    // MoE expert dispatch — 32 tokens × 128 experts → top-4
    (8, 512, 16),    // Intermediate top-k
    (1, 1024, 64),   // Trailblazer-cap shape
];

fn leak_str(s: &str) -> &'static str {
    Box::leak(s.to_owned().into_boxed_str())
}

/// Deterministic pseudo-random fixture: `(i * 0.001 + b * 0.137)` →
/// non-constant values so top-k actually has work to do (constant
/// inputs make the comparator trivial). f32 only.
fn gen_fixture(batch: i32, row_len: i32) -> Vec<f32> {
    let mut out = Vec::with_capacity((batch * row_len) as usize);
    for b in 0..batch {
        for i in 0..row_len {
            let v = (i as f32) * 0.001 + (b as f32) * 0.137;
            out.push(v);
        }
    }
    out
}

fn bench_f32(c: &mut Criterion, baseline: Option<&PytorchBaseline>) {
    let (ctx, stream) = setup_device();
    let mut group = c.benchmark_group("topk/f32");

    for &(batch, row_len, k) in TOPK_SWEEP {
        let label = format!("B{batch}_L{row_len}_K{k}");
        let in_numel = (batch * row_len) as usize;
        let out_numel = (batch * k) as usize;

        let host_x = gen_fixture(batch, row_len);
        let dev_x = match DeviceBuffer::from_slice(&ctx, &host_x) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let mut dev_v: DeviceBuffer<f32> = match DeviceBuffer::zeros(&ctx, out_numel) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let mut dev_i: DeviceBuffer<i32> = match DeviceBuffer::zeros(&ctx, out_numel) {
            Ok(b) => b,
            Err(_) => continue,
        };

        let x_shape = [batch, row_len];
        let o_shape = [batch, k];
        let stx = contiguous_stride(x_shape);
        let sto = contiguous_stride(o_shape);

        let desc = TopkDescriptor {
            batch,
            row_len,
            k,
            largest: true,
            element: ElementKind::F32,
        };
        let plan = match TopkPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        {
            Ok(p) => p,
            Err(_) => continue,
        };

        warmup(&stream, || {
            let args = TopkArgs::<f32> {
                input: TensorRef {
                    data: dev_x.as_slice(),
                    shape: x_shape,
                    stride: stx,
                },
                values: TensorMut {
                    data: dev_v.as_slice_mut(),
                    shape: o_shape,
                    stride: sto,
                },
                indices: TensorMut {
                    data: dev_i.as_slice_mut(),
                    shape: o_shape,
                    stride: sto,
                },
            };
            plan.run(&stream, Workspace::None, args).expect("baracuda topk");
        });
        let baracuda_ns = measure_median_ns(&ctx, &stream, 11, 50, || {
            let args = TopkArgs::<f32> {
                input: TensorRef {
                    data: dev_x.as_slice(),
                    shape: x_shape,
                    stride: stx,
                },
                values: TensorMut {
                    data: dev_v.as_slice_mut(),
                    shape: o_shape,
                    stride: sto,
                },
                indices: TensorMut {
                    data: dev_i.as_slice_mut(),
                    shape: o_shape,
                    stride: sto,
                },
            };
            plan.run(&stream, Workspace::None, args).expect("baracuda topk");
        });
        append_csv_row(
            BENCH_NAME,
            &PhaseTwentyNineRow {
                op: "topk",
                shape: label.clone(),
                dtype: "f32",
                baracuda_ns,
                reference_ns: None,
                reference: "",
                pytorch_ns: baseline.and_then(|b| b.lookup("topk", &label, "f32")),
            },
        );
        group.bench_with_input(BenchmarkId::from_parameter(&label), &(), |bb, _| {
            bb.iter_custom(|iters| {
                time_with_events(&ctx, &stream, iters, || {
                    let args = TopkArgs::<f32> {
                        input: TensorRef {
                            data: dev_x.as_slice(),
                            shape: x_shape,
                            stride: stx,
                        },
                        values: TensorMut {
                            data: dev_v.as_slice_mut(),
                            shape: o_shape,
                            stride: sto,
                        },
                        indices: TensorMut {
                            data: dev_i.as_slice_mut(),
                            shape: o_shape,
                            stride: sto,
                        },
                    };
                    plan.run(&stream, Workspace::None, args).expect("baracuda topk");
                })
            });
        });
        // Make sure in_numel actually participates so `gen_fixture`'s
        // result-size isn't optimised away.
        debug_assert_eq!(host_x.len(), in_numel);
    }
    group.finish();
}

/// Top-level criterion entry - invoked by criterion_main!.
fn benches(c: &mut Criterion) {
    let baseline = PytorchBaseline::load_default();
    let baseline_ref = baseline.as_ref();
    bench_f32(c, baseline_ref);
}

#[allow(missing_docs)]
mod criterion_glue {
    use super::*;
    criterion_group!(benches_grp, benches);
}

criterion_main!(criterion_glue::benches_grp);
