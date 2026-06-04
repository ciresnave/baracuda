//! BatchNorm forward throughput — baracuda self-bench + PyTorch reference.
//!
//! Phase 73.8d — vision-model workhorse. Sweep mirrors `pool_vs_cudnn.rs`
//! (ResNet-50 picks at 3 spatial scales) since BN and pool live in the
//! same forward graph.
//!
//! baracuda's `BatchNormPlan` (Phase 5d) is the bespoke trailblazer;
//! cuDNN does have `batch_normalization_forward_training` but it carries
//! a heavier API (handle + descriptor + activation fusion) than fits in
//! the simple bench pattern — left out here, baracuda self-bench only.
//!
//! PyTorch reference via the frozen JSON baseline using
//! `F.batch_norm` with training=False (eval mode — uses running mean/
//! var). For an apples-to-apples baracuda comparison, baracuda's
//! BatchNormPlan computes mean/rstd per-call (training mode); PyTorch's
//! training=False path is faster (no mean/var compute). Documented as
//! a methodology note.

use baracuda_driver::DeviceBuffer;
use baracuda_kernels::{
    contiguous_stride, BatchNormArgs, BatchNormDescriptor, BatchNormPlan, ElementKind,
    PlanPreference, TensorMut, TensorRef, Workspace,
};
use baracuda_kernels_bench::{
    append_csv_row, measure_median_ns, setup_device, time_with_events, warmup,
    PhaseTwentyNineRow, PoolShape, PytorchBaseline, POOL_SWEEP,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use half::f16;

const BENCH_NAME: &str = "batch_norm";

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
    let mut group = c.benchmark_group(format!("batch_norm/{dtype_label}"));

    for &shape in POOL_SWEEP {
        // Use the (N, C, H, W) part of the PoolShape; ignore the
        // window/stride/pad fields (BN doesn't pool).
        let n = shape.n;
        let c_in = shape.c;
        let h = shape.h;
        let w = shape.w;
        let label = format!("N{n}_C{c_in}_H{h}_W{w}");
        let numel = (n * c_in * h * w) as usize;
        let c_numel = c_in as usize;

        let host_x: Vec<T> = vec![fill; numel];
        let host_gamma: Vec<T> = vec![fill; c_numel];
        let host_beta: Vec<T> = vec![fill; c_numel];

        let dev_x = match DeviceBuffer::from_slice(&ctx, &host_x) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let dev_gamma = match DeviceBuffer::from_slice(&ctx, &host_gamma) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let dev_beta = match DeviceBuffer::from_slice(&ctx, &host_beta) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let mut dev_y: DeviceBuffer<T> = match DeviceBuffer::zeros(&ctx, numel) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let mut dev_mean: DeviceBuffer<T> = match DeviceBuffer::zeros(&ctx, c_numel) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let mut dev_rstd: DeviceBuffer<T> = match DeviceBuffer::zeros(&ctx, c_numel) {
            Ok(b) => b,
            Err(_) => continue,
        };

        let x_shape = [n, c_in, h, w];
        let c_shape = [c_in];
        let stx = contiguous_stride(x_shape);
        let stc = contiguous_stride(c_shape);

        let desc = BatchNormDescriptor::<4> {
            input_shape: x_shape,
            channel_axis: 1,
            eps: 1e-5,
            has_affine: true,
            element: kind,
        };
        let plan = match BatchNormPlan::<T, 4>::select(
            &stream,
            &desc,
            PlanPreference::default(),
        ) {
            Ok(p) => p,
            Err(_) => continue,
        };

        warmup(&stream, || {
            let args = BatchNormArgs::<T, 4> {
                x: TensorRef {
                    data: dev_x.as_slice(),
                    shape: x_shape,
                    stride: stx,
                },
                gamma: Some(TensorRef {
                    data: dev_gamma.as_slice(),
                    shape: c_shape,
                    stride: stc,
                }),
                beta: Some(TensorRef {
                    data: dev_beta.as_slice(),
                    shape: c_shape,
                    stride: stc,
                }),
                y: TensorMut {
                    data: dev_y.as_slice_mut(),
                    shape: x_shape,
                    stride: stx,
                },
                saved_mean: TensorMut {
                    data: dev_mean.as_slice_mut(),
                    shape: c_shape,
                    stride: stc,
                },
                saved_rstd: TensorMut {
                    data: dev_rstd.as_slice_mut(),
                    shape: c_shape,
                    stride: stc,
                },
            };
            plan.run(&stream, Workspace::None, args).expect("baracuda batch_norm");
        });
        let baracuda_ns = measure_median_ns(&ctx, &stream, 11, 50, || {
            let args = BatchNormArgs::<T, 4> {
                x: TensorRef {
                    data: dev_x.as_slice(),
                    shape: x_shape,
                    stride: stx,
                },
                gamma: Some(TensorRef {
                    data: dev_gamma.as_slice(),
                    shape: c_shape,
                    stride: stc,
                }),
                beta: Some(TensorRef {
                    data: dev_beta.as_slice(),
                    shape: c_shape,
                    stride: stc,
                }),
                y: TensorMut {
                    data: dev_y.as_slice_mut(),
                    shape: x_shape,
                    stride: stx,
                },
                saved_mean: TensorMut {
                    data: dev_mean.as_slice_mut(),
                    shape: c_shape,
                    stride: stc,
                },
                saved_rstd: TensorMut {
                    data: dev_rstd.as_slice_mut(),
                    shape: c_shape,
                    stride: stc,
                },
            };
            plan.run(&stream, Workspace::None, args).expect("baracuda batch_norm");
        });
        append_csv_row(
            BENCH_NAME,
            &PhaseTwentyNineRow {
                op: "batch_norm",
                shape: label.clone(),
                dtype: leak_str(dtype_label),
                baracuda_ns,
                reference_ns: None,
                reference: "",
                pytorch_ns: baseline.and_then(|b| b.lookup("batch_norm", &label, dtype_label)),
            },
        );
        group.bench_with_input(BenchmarkId::from_parameter(&label), &(), |bb, _| {
            bb.iter_custom(|iters| {
                time_with_events(&ctx, &stream, iters, || {
                    let args = BatchNormArgs::<T, 4> {
                        x: TensorRef {
                            data: dev_x.as_slice(),
                            shape: x_shape,
                            stride: stx,
                        },
                        gamma: Some(TensorRef {
                            data: dev_gamma.as_slice(),
                            shape: c_shape,
                            stride: stc,
                        }),
                        beta: Some(TensorRef {
                            data: dev_beta.as_slice(),
                            shape: c_shape,
                            stride: stc,
                        }),
                        y: TensorMut {
                            data: dev_y.as_slice_mut(),
                            shape: x_shape,
                            stride: stx,
                        },
                        saved_mean: TensorMut {
                            data: dev_mean.as_slice_mut(),
                            shape: c_shape,
                            stride: stc,
                        },
                        saved_rstd: TensorMut {
                            data: dev_rstd.as_slice_mut(),
                            shape: c_shape,
                            stride: stc,
                        },
                    };
                    plan.run(&stream, Workspace::None, args)
                        .expect("baracuda batch_norm");
                })
            });
        });
        // Suppress unused-import warning when PoolShape isn't otherwise used.
        let _ = PoolShape {
            n: 0,
            c: 0,
            h: 0,
            w: 0,
            k: 0,
            stride: 0,
            pad: 0,
        };
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
