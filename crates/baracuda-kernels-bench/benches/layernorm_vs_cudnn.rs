//! LayerNorm throughput — baracuda self-bench.
//!
//! Phase 29 cross-impl bench. cuDNN's classic API doesn't expose
//! LayerNorm directly — only BatchNorm + the new normalization API
//! (`cudnnNormalizationForward*`) which isn't yet wired in the safe
//! `baracuda-cudnn` Rust surface. We therefore bench baracuda
//! LayerNorm against itself across shapes and dtypes; the comparison
//! against the new norm API is left for a follow-up after the
//! `baracuda-cudnn` wrapping catches up.
//!
//! The bench name is `layernorm_vs_cudnn` for naming consistency with
//! the other cross-impl benches; the reference column in the CSV is
//! empty for now.
//!
//! Sweeps:
//! - `rows ∈ {512, 2048, 4096}`, `hidden ∈ {1024, 4096}`.
//! - Dtypes: `f32`, `f16`.
//! - `has_gamma = true`, `has_beta = true` (typical Llama / GPT shape).

use baracuda_driver::DeviceBuffer;
use baracuda_kernels::{
    contiguous_stride, LayerNormArgs, LayerNormDescriptor, LayerNormPlan, PlanPreference,
    TensorMut, TensorRef, Workspace,
};
use baracuda_kernels_bench::{
    append_csv_row, measure_median_ns, setup_device, time_with_events, warmup,
    PhaseTwentyNineRow, CROSS_HIDDEN_SWEEP, CROSS_SEQLEN_SWEEP,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use half::f16;

const BENCH_NAME: &str = "layernorm_vs_cudnn";

fn leak_str(s: &str) -> &'static str {
    Box::leak(s.to_owned().into_boxed_str())
}

fn bench_baracuda<T>(c: &mut Criterion, dtype_label: &str, fill: T)
where
    T: baracuda_kernels::Element + Copy + 'static,
{
    let (ctx, stream) = setup_device();
    let mut group = c.benchmark_group(format!("layernorm_vs_cudnn/baracuda/{dtype_label}"));

    for &rows in CROSS_SEQLEN_SWEEP {
        for &hidden in CROSS_HIDDEN_SWEEP {
            let shape = format!("R{rows}_H{hidden}");
            let numel = (rows * hidden) as usize;

            let host_x: Vec<T> = vec![fill; numel];
            let host_g: Vec<T> = vec![fill; hidden as usize];
            let host_b: Vec<T> = vec![fill; hidden as usize];

            let dev_x = match DeviceBuffer::from_slice(&ctx, &host_x) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let dev_g = match DeviceBuffer::from_slice(&ctx, &host_g) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let dev_b = match DeviceBuffer::from_slice(&ctx, &host_b) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let mut dev_y: DeviceBuffer<T> = match DeviceBuffer::zeros(&ctx, numel) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let mut dev_mean: DeviceBuffer<T> = match DeviceBuffer::zeros(&ctx, rows as usize) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let mut dev_inv: DeviceBuffer<T> = match DeviceBuffer::zeros(&ctx, rows as usize) {
                Ok(b) => b,
                Err(_) => continue,
            };

            // Normalize over the last axis (bit 1 set for rank-2 shape).
            let desc = LayerNormDescriptor::<2> {
                input_shape: [rows, hidden],
                norm_axes_mask: 0b10,
                eps: 1e-5,
                has_gamma: true,
                has_beta: true,
                element: T::KIND,
            };
            let plan = match LayerNormPlan::<T, 2>::select(
                &stream,
                &desc,
                PlanPreference::default(),
            ) {
                Ok(p) => p,
                Err(_) => continue,
            };

            let xs = [rows, hidden];
            let stx = contiguous_stride(xs);
            let save_shape = [rows, 1];
            let st_save = contiguous_stride(save_shape);
            let gs = [hidden];
            let stg = contiguous_stride(gs);

            warmup(&stream, || {
                let args = LayerNormArgs::<T, 2> {
                    x: TensorRef {
                        data: dev_x.as_slice(),
                        shape: xs,
                        stride: stx,
                    },
                    gamma: Some(TensorRef {
                        data: dev_g.as_slice(),
                        shape: gs,
                        stride: stg,
                    }),
                    beta: Some(TensorRef {
                        data: dev_b.as_slice(),
                        shape: gs,
                        stride: stg,
                    }),
                    y: TensorMut {
                        data: dev_y.as_slice_mut(),
                        shape: xs,
                        stride: stx,
                    },
                    mean: TensorMut {
                        data: dev_mean.as_slice_mut(),
                        shape: save_shape,
                        stride: st_save,
                    },
                    inv_std: TensorMut {
                        data: dev_inv.as_slice_mut(),
                        shape: save_shape,
                        stride: st_save,
                    },
                };
                plan.run(&stream, Workspace::None, args).expect("baracuda layernorm");
            });
            let baracuda_ns = measure_median_ns(&ctx, &stream, 11, 50, || {
                let args = LayerNormArgs::<T, 2> {
                    x: TensorRef {
                        data: dev_x.as_slice(),
                        shape: xs,
                        stride: stx,
                    },
                    gamma: Some(TensorRef {
                        data: dev_g.as_slice(),
                        shape: gs,
                        stride: stg,
                    }),
                    beta: Some(TensorRef {
                        data: dev_b.as_slice(),
                        shape: gs,
                        stride: stg,
                    }),
                    y: TensorMut {
                        data: dev_y.as_slice_mut(),
                        shape: xs,
                        stride: stx,
                    },
                    mean: TensorMut {
                        data: dev_mean.as_slice_mut(),
                        shape: save_shape,
                        stride: st_save,
                    },
                    inv_std: TensorMut {
                        data: dev_inv.as_slice_mut(),
                        shape: save_shape,
                        stride: st_save,
                    },
                };
                plan.run(&stream, Workspace::None, args).expect("baracuda layernorm");
            });
            append_csv_row(
                BENCH_NAME,
                &PhaseTwentyNineRow {
                    op: "layernorm",
                    shape: shape.clone(),
                    dtype: leak_str(dtype_label),
                    baracuda_ns,
                    reference_ns: None,
                    reference: "",
                    pytorch_ns: None,
                },
            );
            group.bench_with_input(BenchmarkId::from_parameter(&shape), &(), |bb, _| {
                bb.iter_custom(|iters| {
                    time_with_events(&ctx, &stream, iters, || {
                        let args = LayerNormArgs::<T, 2> {
                            x: TensorRef {
                                data: dev_x.as_slice(),
                                shape: xs,
                                stride: stx,
                            },
                            gamma: Some(TensorRef {
                                data: dev_g.as_slice(),
                                shape: gs,
                                stride: stg,
                            }),
                            beta: Some(TensorRef {
                                data: dev_b.as_slice(),
                                shape: gs,
                                stride: stg,
                            }),
                            y: TensorMut {
                                data: dev_y.as_slice_mut(),
                                shape: xs,
                                stride: stx,
                            },
                            mean: TensorMut {
                                data: dev_mean.as_slice_mut(),
                                shape: save_shape,
                                stride: st_save,
                            },
                            inv_std: TensorMut {
                                data: dev_inv.as_slice_mut(),
                                shape: save_shape,
                                stride: st_save,
                            },
                        };
                        plan.run(&stream, Workspace::None, args).expect("baracuda layernorm");
                    })
                });
            });
        }
    }
    group.finish();
}

fn ln_benches(c: &mut Criterion) {
    bench_baracuda::<f32>(c, "f32", 1.0_f32);
    bench_baracuda::<f16>(c, "f16", f16::ONE);
}

criterion_group!(benches_grp, ln_benches);
criterion_main!(benches_grp);
