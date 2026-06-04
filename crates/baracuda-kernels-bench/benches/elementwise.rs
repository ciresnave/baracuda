//! Elementwise throughput — baracuda self-bench.
//!
//! Bench scope: Add, Mul (binary), ReLU, GELU (unary). At one large
//! 1-D shape `numel ∈ {1M, 16M}`, dtype `f32` + `f16`. No cuBLAS /
//! cuDNN reference: elementwise ops are bandwidth-bound and the
//! reference would be "did we hit peak HBM throughput", which the
//! Phase 10 base bench already approximates. This bench's value is
//! tracking deltas across baracuda versions, not against a fixed
//! reference.

use baracuda_driver::DeviceBuffer;
use baracuda_kernels::{
    contiguous_stride, BinaryArgs, BinaryDescriptor, BinaryKind, BinaryPlan, PlanPreference,
    TensorMut, TensorRef, UnaryArgs, UnaryDescriptor, UnaryKind, UnaryPlan, Workspace,
};
use baracuda_kernels_bench::{
    append_csv_row, measure_median_ns, setup_device, time_with_events, warmup,
    PhaseTwentyNineRow,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use half::f16;

const BENCH_NAME: &str = "elementwise";

const ELT_SWEEP: &[i32] = &[1 << 20, 1 << 24];

fn leak_str(s: &str) -> &'static str {
    Box::leak(s.to_owned().into_boxed_str())
}

fn bench_binary<T>(c: &mut Criterion, kind: BinaryKind, kind_label: &str, dtype_label: &str, fill: T)
where
    T: baracuda_kernels::Element + Copy + 'static,
{
    let (ctx, stream) = setup_device();
    let mut group = c.benchmark_group(format!("elementwise/{kind_label}_{dtype_label}"));

    for &n in ELT_SWEEP {
        let shape_str = format!("N{n}");
        let host: Vec<T> = vec![fill; n as usize];
        let dev_a = match DeviceBuffer::from_slice(&ctx, &host) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let dev_b = match DeviceBuffer::from_slice(&ctx, &host) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let mut dev_y: DeviceBuffer<T> = match DeviceBuffer::zeros(&ctx, n as usize) {
            Ok(b) => b,
            Err(_) => continue,
        };

        let desc = BinaryDescriptor::<1> {
            kind,
            shape: [n],
            element: T::KIND,
        };
        let plan = match BinaryPlan::<T, 1>::select(&stream, &desc, PlanPreference::default()) {
            Ok(p) => p,
            Err(_) => continue,
        };

        let xs = [n];
        let stx = contiguous_stride(xs);

        warmup(&stream, || {
            let args = BinaryArgs::<T, 1> {
                a: TensorRef {
                    data: dev_a.as_slice(),
                    shape: xs,
                    stride: stx,
                },
                b: TensorRef {
                    data: dev_b.as_slice(),
                    shape: xs,
                    stride: stx,
                },
                y: TensorMut {
                    data: dev_y.as_slice_mut(),
                    shape: xs,
                    stride: stx,
                },
            };
            plan.run(&stream, Workspace::None, args).expect("binary ew");
        });
        let baracuda_ns = measure_median_ns(&ctx, &stream, 11, 100, || {
            let args = BinaryArgs::<T, 1> {
                a: TensorRef {
                    data: dev_a.as_slice(),
                    shape: xs,
                    stride: stx,
                },
                b: TensorRef {
                    data: dev_b.as_slice(),
                    shape: xs,
                    stride: stx,
                },
                y: TensorMut {
                    data: dev_y.as_slice_mut(),
                    shape: xs,
                    stride: stx,
                },
            };
            plan.run(&stream, Workspace::None, args).expect("binary ew");
        });
        let dtype_lbl = leak_str(dtype_label);
        let op_lbl = leak_str(kind_label);
        append_csv_row(
            BENCH_NAME,
            &PhaseTwentyNineRow {
                op: op_lbl,
                shape: shape_str.clone(),
                dtype: dtype_lbl,
                baracuda_ns,
                reference_ns: None,
                reference: "",
                pytorch_ns: None,
            },
        );

        group.throughput(Throughput::Bytes((n as u64) * (size_of::<T>() as u64) * 3));
        group.bench_with_input(BenchmarkId::from_parameter(&shape_str), &(), |bb, _| {
            bb.iter_custom(|iters| {
                time_with_events(&ctx, &stream, iters, || {
                    let args = BinaryArgs::<T, 1> {
                        a: TensorRef {
                            data: dev_a.as_slice(),
                            shape: xs,
                            stride: stx,
                        },
                        b: TensorRef {
                            data: dev_b.as_slice(),
                            shape: xs,
                            stride: stx,
                        },
                        y: TensorMut {
                            data: dev_y.as_slice_mut(),
                            shape: xs,
                            stride: stx,
                        },
                    };
                    plan.run(&stream, Workspace::None, args).expect("binary ew");
                })
            });
        });
    }
    group.finish();
}

fn bench_unary<T>(c: &mut Criterion, kind: UnaryKind, kind_label: &str, dtype_label: &str, fill: T)
where
    T: baracuda_kernels::Element + Copy + 'static,
{
    let (ctx, stream) = setup_device();
    let mut group = c.benchmark_group(format!("elementwise/{kind_label}_{dtype_label}"));

    for &n in ELT_SWEEP {
        let shape_str = format!("N{n}");
        let host: Vec<T> = vec![fill; n as usize];
        let dev_x = match DeviceBuffer::from_slice(&ctx, &host) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let mut dev_y: DeviceBuffer<T> = match DeviceBuffer::zeros(&ctx, n as usize) {
            Ok(b) => b,
            Err(_) => continue,
        };

        let desc = UnaryDescriptor::<1> {
            kind,
            shape: [n],
            element: T::KIND,
        };
        let plan = match UnaryPlan::<T, 1>::select(&stream, &desc, PlanPreference::default()) {
            Ok(p) => p,
            Err(_) => continue,
        };
        let xs = [n];
        let stx = contiguous_stride(xs);

        warmup(&stream, || {
            let args = UnaryArgs::<T, 1> {
                x: TensorRef {
                    data: dev_x.as_slice(),
                    shape: xs,
                    stride: stx,
                },
                y: TensorMut {
                    data: dev_y.as_slice_mut(),
                    shape: xs,
                    stride: stx,
                },
            };
            plan.run(&stream, Workspace::None, args).expect("unary ew");
        });
        let baracuda_ns = measure_median_ns(&ctx, &stream, 11, 100, || {
            let args = UnaryArgs::<T, 1> {
                x: TensorRef {
                    data: dev_x.as_slice(),
                    shape: xs,
                    stride: stx,
                },
                y: TensorMut {
                    data: dev_y.as_slice_mut(),
                    shape: xs,
                    stride: stx,
                },
            };
            plan.run(&stream, Workspace::None, args).expect("unary ew");
        });
        let dtype_lbl = leak_str(dtype_label);
        let op_lbl = leak_str(kind_label);
        append_csv_row(
            BENCH_NAME,
            &PhaseTwentyNineRow {
                op: op_lbl,
                shape: shape_str.clone(),
                dtype: dtype_lbl,
                baracuda_ns,
                reference_ns: None,
                reference: "",
                pytorch_ns: None,
            },
        );

        group.throughput(Throughput::Bytes((n as u64) * (size_of::<T>() as u64) * 2));
        group.bench_with_input(BenchmarkId::from_parameter(&shape_str), &(), |bb, _| {
            bb.iter_custom(|iters| {
                time_with_events(&ctx, &stream, iters, || {
                    let args = UnaryArgs::<T, 1> {
                        x: TensorRef {
                            data: dev_x.as_slice(),
                            shape: xs,
                            stride: stx,
                        },
                        y: TensorMut {
                            data: dev_y.as_slice_mut(),
                            shape: xs,
                            stride: stx,
                        },
                    };
                    plan.run(&stream, Workspace::None, args).expect("unary ew");
                })
            });
        });
    }
    group.finish();
}

/// Top-level criterion entry - invoked by criterion_main!.
fn benches(c: &mut Criterion) {
    bench_binary::<f32>(c, BinaryKind::Add, "add", "f32", 1.0_f32);
    bench_binary::<f16>(c, BinaryKind::Add, "add", "f16", f16::ONE);
    bench_binary::<f32>(c, BinaryKind::Mul, "mul", "f32", 1.0_f32);
    bench_binary::<f16>(c, BinaryKind::Mul, "mul", "f16", f16::ONE);
    bench_unary::<f32>(c, UnaryKind::Relu, "relu", "f32", 1.0_f32);
    bench_unary::<f16>(c, UnaryKind::Relu, "relu", "f16", f16::ONE);
    bench_unary::<f32>(c, UnaryKind::Gelu, "gelu", "f32", 1.0_f32);
    bench_unary::<f16>(c, UnaryKind::Gelu, "gelu", "f16", f16::ONE);
}

criterion_group!(benches_grp, benches);
criterion_main!(benches_grp);
