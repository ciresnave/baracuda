//! RMSNorm throughput — baracuda self-bench.
//!
//! Phase 29 cross-impl bench. No cuDNN equivalent: cuDNN's
//! `cudnnNormalizationForward*` API doesn't expose RMSNorm (only
//! LayerNorm / BatchNorm / InstanceNorm / GroupNorm). PyTorch's
//! `torch.nn.functional.rms_norm` is a fused-kernel-of-the-month that
//! varies by backend; we bench baracuda against itself.
//!
//! Sweeps:
//! - `rows ∈ {512, 2048, 4096}`, `hidden ∈ {1024, 4096}`.
//! - Dtypes: `f32`, `f16`, `bf16`.
//! - `has_gamma = true` (typical Llama-family shape).

use baracuda_driver::DeviceBuffer;
use baracuda_kernels::{
    contiguous_stride, PlanPreference, RMSNormArgs, RMSNormDescriptor, RMSNormPlan, TensorMut,
    TensorRef, Workspace,
};
use baracuda_kernels_bench::{
    append_csv_row, measure_median_ns, setup_device, time_with_events, warmup,
    PhaseTwentyNineRow, CROSS_HIDDEN_SWEEP, CROSS_SEQLEN_SWEEP,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use half::{bf16, f16};

const BENCH_NAME: &str = "rmsnorm";

fn leak_str(s: &str) -> &'static str {
    Box::leak(s.to_owned().into_boxed_str())
}

fn bench<T>(c: &mut Criterion, dtype_label: &str, fill: T)
where
    T: baracuda_kernels::Element + Copy + 'static,
{
    let (ctx, stream) = setup_device();
    let mut group = c.benchmark_group(format!("rmsnorm/{dtype_label}"));

    for &rows in CROSS_SEQLEN_SWEEP {
        for &hidden in CROSS_HIDDEN_SWEEP {
            let shape = format!("R{rows}_H{hidden}");
            let numel = (rows * hidden) as usize;

            let host_x: Vec<T> = vec![fill; numel];
            let host_g: Vec<T> = vec![fill; hidden as usize];

            let dev_x = match DeviceBuffer::from_slice(&ctx, &host_x) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let dev_g = match DeviceBuffer::from_slice(&ctx, &host_g) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let mut dev_y: DeviceBuffer<T> = match DeviceBuffer::zeros(&ctx, numel) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let mut dev_rms: DeviceBuffer<T> = match DeviceBuffer::zeros(&ctx, rows as usize) {
                Ok(b) => b,
                Err(_) => continue,
            };

            let desc = RMSNormDescriptor::<2> {
                input_shape: [rows, hidden],
                norm_axes_mask: 0b10,
                eps: 1e-5,
                has_gamma: true,
                element: T::KIND,
            };
            let plan = match RMSNormPlan::<T, 2>::select(
                &stream,
                &desc,
                PlanPreference::default(),
            ) {
                Ok(p) => p,
                Err(_) => continue,
            };

            let xs = [rows, hidden];
            let stx = contiguous_stride(xs);
            let rs = [rows, 1];
            let st_rms = contiguous_stride(rs);
            let gs = [hidden];
            let stg = contiguous_stride(gs);

            warmup(&stream, || {
                let args = RMSNormArgs::<T, 2> {
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
                    y: TensorMut {
                        data: dev_y.as_slice_mut(),
                        shape: xs,
                        stride: stx,
                    },
                    rms: TensorMut {
                        data: dev_rms.as_slice_mut(),
                        shape: rs,
                        stride: st_rms,
                    },
                };
                plan.run(&stream, Workspace::None, args).expect("baracuda rmsnorm");
            });
            let baracuda_ns = measure_median_ns(&ctx, &stream, 11, 50, || {
                let args = RMSNormArgs::<T, 2> {
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
                    y: TensorMut {
                        data: dev_y.as_slice_mut(),
                        shape: xs,
                        stride: stx,
                    },
                    rms: TensorMut {
                        data: dev_rms.as_slice_mut(),
                        shape: rs,
                        stride: st_rms,
                    },
                };
                plan.run(&stream, Workspace::None, args).expect("baracuda rmsnorm");
            });
            append_csv_row(
                BENCH_NAME,
                &PhaseTwentyNineRow {
                    op: "rmsnorm",
                    shape: shape.clone(),
                    dtype: leak_str(dtype_label),
                    baracuda_ns,
                    reference_ns: None,
                    reference: "",
                },
            );
            group.bench_with_input(BenchmarkId::from_parameter(&shape), &(), |bb, _| {
                bb.iter_custom(|iters| {
                    time_with_events(&ctx, &stream, iters, || {
                        let args = RMSNormArgs::<T, 2> {
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
                            y: TensorMut {
                                data: dev_y.as_slice_mut(),
                                shape: xs,
                                stride: stx,
                            },
                            rms: TensorMut {
                                data: dev_rms.as_slice_mut(),
                                shape: rs,
                                stride: st_rms,
                            },
                        };
                        plan.run(&stream, Workspace::None, args).expect("baracuda rmsnorm");
                    })
                });
            });
        }
    }
    group.finish();
}

fn benches(c: &mut Criterion) {
    bench::<f32>(c, "f32", 1.0_f32);
    bench::<f16>(c, "f16", f16::ONE);
    bench::<bf16>(c, "bf16", bf16::ONE);
}

criterion_group!(benches_grp, benches);
criterion_main!(benches_grp);
