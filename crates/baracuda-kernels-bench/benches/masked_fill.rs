//! `masked_fill` throughput — baracuda self-bench + PyTorch reference.
//!
//! Phase 73.8c — every LLM attention kernel applies a causal / padding
//! mask via this op (or fuses it into the attention kernel). Standalone,
//! it's an elementwise `out[i] = mask[i] ? value : src[i]`. Same shape
//! sweep as softmax / reductions (`R × H`) so the rows align in
//! BENCHMARKS.md.
//!
//! Compared against `tensor.masked_fill(mask, value)` via the frozen JSON
//! baseline. No NVIDIA-library reference. baracuda mask is `u8`; PyTorch
//! is `torch.bool` — same wire format (1 byte / cell), bit-equivalent
//! semantics. Fill value is `-inf` (the canonical attention-mask value)
//! for f32 / f16, `-1e9` for the integer path (not exercised here).

use baracuda_driver::DeviceBuffer;
use baracuda_kernels::{
    contiguous_stride, ElementKind, MaskedFillArgs, MaskedFillDescriptor, MaskedFillPlan,
    PlanPreference, TensorMut, TensorRef, Workspace,
};
use baracuda_kernels_bench::{
    append_csv_row, measure_median_ns, setup_device, time_with_events, warmup,
    PhaseTwentyNineRow, PytorchBaseline, CROSS_HIDDEN_SWEEP, CROSS_SEQLEN_SWEEP,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use half::f16;

const BENCH_NAME: &str = "masked_fill";

fn leak_str(s: &str) -> &'static str {
    Box::leak(s.to_owned().into_boxed_str())
}

fn bench<T>(
    c: &mut Criterion,
    dtype_label: &str,
    fill: T,
    desc_for_dtype: impl Fn([i32; 2]) -> MaskedFillDescriptor<2>,
    baseline: Option<&PytorchBaseline>,
) where
    T: baracuda_kernels::Element + Copy + 'static,
{
    let (ctx, stream) = setup_device();
    let mut group = c.benchmark_group(format!("masked_fill/{dtype_label}"));

    for &rows in CROSS_SEQLEN_SWEEP {
        for &hidden in CROSS_HIDDEN_SWEEP {
            let label = format!("R{rows}_H{hidden}");
            let numel = (rows * hidden) as usize;

            let host_src: Vec<T> = vec![fill; numel];
            // Half-and-half mask pattern: roughly 50% of cells masked.
            // Deterministic so reruns are stable.
            let host_mask: Vec<u8> = (0..numel).map(|i| (i & 1) as u8).collect();

            let dev_src = match DeviceBuffer::from_slice(&ctx, &host_src) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let dev_mask = match DeviceBuffer::from_slice(&ctx, &host_mask) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let mut dev_out: DeviceBuffer<T> = match DeviceBuffer::zeros(&ctx, numel) {
                Ok(b) => b,
                Err(_) => continue,
            };

            let xs = [rows, hidden];
            let stx = contiguous_stride(xs);

            let desc = desc_for_dtype(xs);
            let plan = match MaskedFillPlan::<T, 2>::select(
                &stream,
                &desc,
                PlanPreference::default(),
            ) {
                Ok(p) => p,
                Err(_) => continue,
            };

            warmup(&stream, || {
                let args = MaskedFillArgs::<T, 2> {
                    src: TensorRef {
                        data: dev_src.as_slice(),
                        shape: xs,
                        stride: stx,
                    },
                    mask: TensorRef {
                        data: dev_mask.as_slice(),
                        shape: xs,
                        stride: stx,
                    },
                    out: TensorMut {
                        data: dev_out.as_slice_mut(),
                        shape: xs,
                        stride: stx,
                    },
                };
                plan.run(&stream, Workspace::None, args).expect("baracuda masked_fill");
            });
            let baracuda_ns = measure_median_ns(&ctx, &stream, 11, 50, || {
                let args = MaskedFillArgs::<T, 2> {
                    src: TensorRef {
                        data: dev_src.as_slice(),
                        shape: xs,
                        stride: stx,
                    },
                    mask: TensorRef {
                        data: dev_mask.as_slice(),
                        shape: xs,
                        stride: stx,
                    },
                    out: TensorMut {
                        data: dev_out.as_slice_mut(),
                        shape: xs,
                        stride: stx,
                    },
                };
                plan.run(&stream, Workspace::None, args).expect("baracuda masked_fill");
            });
            append_csv_row(
                BENCH_NAME,
                &PhaseTwentyNineRow {
                    op: "masked_fill",
                    shape: label.clone(),
                    dtype: leak_str(dtype_label),
                    baracuda_ns,
                    reference_ns: None,
                    reference: "",
                    pytorch_ns: baseline
                        .and_then(|b| b.lookup("masked_fill", &label, dtype_label)),
                },
            );
            group.bench_with_input(BenchmarkId::from_parameter(&label), &(), |bb, _| {
                bb.iter_custom(|iters| {
                    time_with_events(&ctx, &stream, iters, || {
                        let args = MaskedFillArgs::<T, 2> {
                            src: TensorRef {
                                data: dev_src.as_slice(),
                                shape: xs,
                                stride: stx,
                            },
                            mask: TensorRef {
                                data: dev_mask.as_slice(),
                                shape: xs,
                                stride: stx,
                            },
                            out: TensorMut {
                                data: dev_out.as_slice_mut(),
                                shape: xs,
                                stride: stx,
                            },
                        };
                        plan.run(&stream, Workspace::None, args).expect("baracuda masked_fill");
                    })
                });
            });
        }
    }
    group.finish();
}

/// Top-level criterion entry - invoked by criterion_main!.
fn benches(c: &mut Criterion) {
    let baseline = PytorchBaseline::load_default();
    let baseline_ref = baseline.as_ref();
    // f16 path: baracuda's MaskedFillDescriptor only ships f32/f64/i32/bool
    // helpers in the trailblazer (Phase 11). f16 isn't wired at this layer.
    // f32 is the load-bearing attention-mask dtype anyway.
    bench::<f32>(
        c,
        "f32",
        1.0_f32,
        |shape| MaskedFillDescriptor::new_f32(shape, f32::NEG_INFINITY),
        baseline_ref,
    );
    let _ = ElementKind::F32;
    let _ = f16::ONE;  // suppress unused-import warning
}

#[allow(missing_docs)]
mod criterion_glue {
    use super::*;
    criterion_group!(benches_grp, benches);
}

criterion_main!(criterion_glue::benches_grp);
