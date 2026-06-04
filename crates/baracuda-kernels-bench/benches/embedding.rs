//! Embedding table-lookup throughput — baracuda self-bench + PyTorch reference.
//!
//! Phase 73.8b — LLM workhorse op. Every LLM forward pass starts with
//! `embedding[token_ids]` to project the input token sequence into the
//! model's hidden space. Sweep covers LLM-typical vocab × hidden +
//! input lengths matching both prefill and decode regimes.
//!
//! Compared against `F.embedding(input_ids, weight)` via the frozen JSON
//! baseline. No NVIDIA-library reference (cuBLAS / cuDNN don't have a
//! direct embedding op — it's a gather along axis 0).
//!
//! Shapes: `V{vocab}_D{hidden}_N{num_indices}`.
//!
//! Indices are random in `[0, V)`, generated with a fixed seed for
//! reproducibility. dtype f32 + f16. Indices are always `i32`.

use baracuda_driver::DeviceBuffer;
use baracuda_kernels::{
    contiguous_stride, ElementKind, EmbeddingArgs, EmbeddingDescriptor, EmbeddingPlan,
    PlanPreference, TensorMut, TensorRef, Workspace,
};
use baracuda_kernels_bench::{
    append_csv_row, measure_median_ns, setup_device, time_with_events, warmup,
    PhaseTwentyNineRow, PytorchBaseline,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use half::f16;

const BENCH_NAME: &str = "embedding";

/// Embedding shape sweep: `(vocab, hidden, num_indices)`.
/// - Llama-2 7B vocab×hidden = 32000×4096 — prefill 2048 + decode 1.
/// - Smaller dense lookup at vocab=8192.
const EMBEDDING_SWEEP: &[(i32, i32, i32)] = &[
    (32000, 4096, 1),     // Llama-2 7B decode-step
    (32000, 4096, 2048),  // Llama-2 7B prefill
    (8192, 1024, 512),    // Smaller / older-style model
];

fn leak_str(s: &str) -> &'static str {
    Box::leak(s.to_owned().into_boxed_str())
}

/// Deterministic pseudo-random `i32` indices in `[0, vocab)`. Same
/// fixture each run (no `rand` dep).
fn gen_indices(num: usize, vocab: i32) -> Vec<i32> {
    let mut out = Vec::with_capacity(num);
    let mut s: u64 = 0x9E3779B97F4A7C15;
    for _ in 0..num {
        // xorshift64*
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        let r = (s.wrapping_mul(0x2545F4914F6CDD1D) >> 33) as u32;
        out.push((r % (vocab as u32)) as i32);
    }
    out
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
    let mut group = c.benchmark_group(format!("embedding/{dtype_label}"));

    for &(vocab, hidden, num) in EMBEDDING_SWEEP {
        let label = format!("V{vocab}_D{hidden}_N{num}");
        let weight_numel = (vocab * hidden) as usize;
        let out_numel = (num * hidden) as usize;

        let host_weight: Vec<T> = vec![fill; weight_numel];
        let host_idx = gen_indices(num as usize, vocab);

        let dev_weight = match DeviceBuffer::from_slice(&ctx, &host_weight) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let dev_idx = match DeviceBuffer::from_slice(&ctx, &host_idx) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let mut dev_out: DeviceBuffer<T> = match DeviceBuffer::zeros(&ctx, out_numel) {
            Ok(b) => b,
            Err(_) => continue,
        };

        let w_shape = [vocab, hidden];
        let i_shape = [num];
        let o_shape = [num, hidden];
        let stw = contiguous_stride(w_shape);
        let sti = contiguous_stride(i_shape);
        let sto = contiguous_stride(o_shape);

        let desc = EmbeddingDescriptor {
            num_embeddings: vocab,
            embedding_dim: hidden,
            num_indices: num,
            padding_idx: None,
            element: kind,
        };
        let plan =
            match EmbeddingPlan::<T>::select(&stream, &desc, PlanPreference::default()) {
                Ok(p) => p,
                Err(_) => continue,
            };

        warmup(&stream, || {
            let args = EmbeddingArgs::<T, i32> {
                weight: TensorRef {
                    data: dev_weight.as_slice(),
                    shape: w_shape,
                    stride: stw,
                },
                indices: TensorRef {
                    data: dev_idx.as_slice(),
                    shape: i_shape,
                    stride: sti,
                },
                output: TensorMut {
                    data: dev_out.as_slice_mut(),
                    shape: o_shape,
                    stride: sto,
                },
            };
            plan.run(&stream, Workspace::None, args).expect("baracuda embedding");
        });
        let baracuda_ns = measure_median_ns(&ctx, &stream, 11, 50, || {
            let args = EmbeddingArgs::<T, i32> {
                weight: TensorRef {
                    data: dev_weight.as_slice(),
                    shape: w_shape,
                    stride: stw,
                },
                indices: TensorRef {
                    data: dev_idx.as_slice(),
                    shape: i_shape,
                    stride: sti,
                },
                output: TensorMut {
                    data: dev_out.as_slice_mut(),
                    shape: o_shape,
                    stride: sto,
                },
            };
            plan.run(&stream, Workspace::None, args).expect("baracuda embedding");
        });
        append_csv_row(
            BENCH_NAME,
            &PhaseTwentyNineRow {
                op: "embedding",
                shape: label.clone(),
                dtype: leak_str(dtype_label),
                baracuda_ns,
                reference_ns: None,
                reference: "",
                pytorch_ns: baseline.and_then(|b| b.lookup("embedding", &label, dtype_label)),
            },
        );
        group.bench_with_input(BenchmarkId::from_parameter(&label), &(), |bb, _| {
            bb.iter_custom(|iters| {
                time_with_events(&ctx, &stream, iters, || {
                    let args = EmbeddingArgs::<T, i32> {
                        weight: TensorRef {
                            data: dev_weight.as_slice(),
                            shape: w_shape,
                            stride: stw,
                        },
                        indices: TensorRef {
                            data: dev_idx.as_slice(),
                            shape: i_shape,
                            stride: sti,
                        },
                        output: TensorMut {
                            data: dev_out.as_slice_mut(),
                            shape: o_shape,
                            stride: sto,
                        },
                    };
                    plan.run(&stream, Workspace::None, args).expect("baracuda embedding");
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

#[allow(missing_docs)]
mod criterion_glue {
    use super::*;
    criterion_group!(benches_grp, benches);
}

criterion_main!(criterion_glue::benches_grp);
