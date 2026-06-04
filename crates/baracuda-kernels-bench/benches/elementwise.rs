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
    PhaseTwentyNineRow, PytorchBaseline,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use half::f16;

const BENCH_NAME: &str = "elementwise";

const ELT_SWEEP: &[i32] = &[1 << 20, 1 << 24];

fn leak_str(s: &str) -> &'static str {
    Box::leak(s.to_owned().into_boxed_str())
}

fn bench_binary<T>(
    c: &mut Criterion,
    kind: BinaryKind,
    kind_label: &str,
    dtype_label: &str,
    fill: T,
    baseline: Option<&PytorchBaseline>,
)
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
                pytorch_ns: baseline.and_then(|b| b.lookup(kind_label, &shape_str, dtype_label)),
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

fn bench_unary<T>(
    c: &mut Criterion,
    kind: UnaryKind,
    kind_label: &str,
    dtype_label: &str,
    fill: T,
    baseline: Option<&PytorchBaseline>,
)
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
                pytorch_ns: baseline.and_then(|b| b.lookup(kind_label, &shape_str, dtype_label)),
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
    let baseline = PytorchBaseline::load_default();
    let baseline_ref = baseline.as_ref();
    bench_binary::<f32>(c, BinaryKind::Add, "add", "f32", 1.0_f32, baseline_ref);
    bench_binary::<f16>(c, BinaryKind::Add, "add", "f16", f16::ONE, baseline_ref);
    bench_binary::<f32>(c, BinaryKind::Mul, "mul", "f32", 1.0_f32, baseline_ref);
    bench_binary::<f16>(c, BinaryKind::Mul, "mul", "f16", f16::ONE, baseline_ref);
    bench_unary::<f32>(c, UnaryKind::Relu, "relu", "f32", 1.0_f32, baseline_ref);
    bench_unary::<f16>(c, UnaryKind::Relu, "relu", "f16", f16::ONE, baseline_ref);
    bench_unary::<f32>(c, UnaryKind::Gelu, "gelu", "f32", 1.0_f32, baseline_ref);
    bench_unary::<f16>(c, UnaryKind::Gelu, "gelu", "f16", f16::ONE, baseline_ref);
    // Phase 73.4: extend coverage with Llama-family Silu + classical Tanh/Sigmoid.
    bench_unary::<f32>(c, UnaryKind::Silu, "silu", "f32", 1.0_f32, baseline_ref);
    bench_unary::<f16>(c, UnaryKind::Silu, "silu", "f16", f16::ONE, baseline_ref);
    bench_unary::<f32>(c, UnaryKind::Tanh, "tanh", "f32", 1.0_f32, baseline_ref);
    bench_unary::<f16>(c, UnaryKind::Tanh, "tanh", "f16", f16::ONE, baseline_ref);
    bench_unary::<f32>(c, UnaryKind::Sigmoid, "sigmoid", "f32", 1.0_f32, baseline_ref);
    bench_unary::<f16>(c, UnaryKind::Sigmoid, "sigmoid", "f16", f16::ONE, baseline_ref);
    // Phase 73.5: comprehensive elementwise coverage. Activations,
    // basic math unaries, additional binaries — all via the existing
    // bench_unary / bench_binary harness.
    //
    // Additional activations.
    bench_unary::<f32>(c, UnaryKind::Mish, "mish", "f32", 1.0_f32, baseline_ref);
    bench_unary::<f16>(c, UnaryKind::Mish, "mish", "f16", f16::ONE, baseline_ref);
    bench_unary::<f32>(c, UnaryKind::Hardswish, "hardswish", "f32", 1.0_f32, baseline_ref);
    bench_unary::<f16>(c, UnaryKind::Hardswish, "hardswish", "f16", f16::ONE, baseline_ref);
    bench_unary::<f32>(c, UnaryKind::Hardsigmoid, "hardsigmoid", "f32", 1.0_f32, baseline_ref);
    bench_unary::<f16>(c, UnaryKind::Hardsigmoid, "hardsigmoid", "f16", f16::ONE, baseline_ref);
    bench_unary::<f32>(c, UnaryKind::Hardtanh, "hardtanh", "f32", 1.0_f32, baseline_ref);
    bench_unary::<f16>(c, UnaryKind::Hardtanh, "hardtanh", "f16", f16::ONE, baseline_ref);
    bench_unary::<f32>(c, UnaryKind::LeakyRelu, "leaky_relu", "f32", 1.0_f32, baseline_ref);
    bench_unary::<f16>(c, UnaryKind::LeakyRelu, "leaky_relu", "f16", f16::ONE, baseline_ref);
    bench_unary::<f32>(c, UnaryKind::Elu, "elu", "f32", 1.0_f32, baseline_ref);
    bench_unary::<f16>(c, UnaryKind::Elu, "elu", "f16", f16::ONE, baseline_ref);
    bench_unary::<f32>(c, UnaryKind::Selu, "selu", "f32", 1.0_f32, baseline_ref);
    bench_unary::<f16>(c, UnaryKind::Selu, "selu", "f16", f16::ONE, baseline_ref);
    bench_unary::<f32>(c, UnaryKind::Relu6, "relu6", "f32", 1.0_f32, baseline_ref);
    bench_unary::<f16>(c, UnaryKind::Relu6, "relu6", "f16", f16::ONE, baseline_ref);
    bench_unary::<f32>(c, UnaryKind::Softplus, "softplus", "f32", 1.0_f32, baseline_ref);
    bench_unary::<f16>(c, UnaryKind::Softplus, "softplus", "f16", f16::ONE, baseline_ref);
    bench_unary::<f32>(c, UnaryKind::Softsign, "softsign", "f32", 1.0_f32, baseline_ref);
    bench_unary::<f16>(c, UnaryKind::Softsign, "softsign", "f16", f16::ONE, baseline_ref);
    bench_unary::<f32>(c, UnaryKind::GeluTanh, "gelu_tanh", "f32", 1.0_f32, baseline_ref);
    bench_unary::<f16>(c, UnaryKind::GeluTanh, "gelu_tanh", "f16", f16::ONE, baseline_ref);
    // Basic math unaries.
    bench_unary::<f32>(c, UnaryKind::Abs, "abs", "f32", 1.0_f32, baseline_ref);
    bench_unary::<f16>(c, UnaryKind::Abs, "abs", "f16", f16::ONE, baseline_ref);
    bench_unary::<f32>(c, UnaryKind::Neg, "neg", "f32", 1.0_f32, baseline_ref);
    bench_unary::<f16>(c, UnaryKind::Neg, "neg", "f16", f16::ONE, baseline_ref);
    bench_unary::<f32>(c, UnaryKind::Sign, "sign", "f32", 1.0_f32, baseline_ref);
    bench_unary::<f16>(c, UnaryKind::Sign, "sign", "f16", f16::ONE, baseline_ref);
    bench_unary::<f32>(c, UnaryKind::Reciprocal, "reciprocal", "f32", 1.0_f32, baseline_ref);
    bench_unary::<f16>(c, UnaryKind::Reciprocal, "reciprocal", "f16", f16::ONE, baseline_ref);
    bench_unary::<f32>(c, UnaryKind::Sqrt, "sqrt", "f32", 1.0_f32, baseline_ref);
    bench_unary::<f16>(c, UnaryKind::Sqrt, "sqrt", "f16", f16::ONE, baseline_ref);
    bench_unary::<f32>(c, UnaryKind::Rsqrt, "rsqrt", "f32", 1.0_f32, baseline_ref);
    bench_unary::<f16>(c, UnaryKind::Rsqrt, "rsqrt", "f16", f16::ONE, baseline_ref);
    bench_unary::<f32>(c, UnaryKind::Square, "square", "f32", 1.0_f32, baseline_ref);
    bench_unary::<f16>(c, UnaryKind::Square, "square", "f16", f16::ONE, baseline_ref);
    bench_unary::<f32>(c, UnaryKind::Exp, "exp", "f32", 1.0_f32, baseline_ref);
    bench_unary::<f16>(c, UnaryKind::Exp, "exp", "f16", f16::ONE, baseline_ref);
    bench_unary::<f32>(c, UnaryKind::Log, "log", "f32", 1.0_f32, baseline_ref);
    bench_unary::<f16>(c, UnaryKind::Log, "log", "f16", f16::ONE, baseline_ref);
    bench_unary::<f32>(c, UnaryKind::Sin, "sin", "f32", 1.0_f32, baseline_ref);
    bench_unary::<f16>(c, UnaryKind::Sin, "sin", "f16", f16::ONE, baseline_ref);
    bench_unary::<f32>(c, UnaryKind::Cos, "cos", "f32", 1.0_f32, baseline_ref);
    bench_unary::<f16>(c, UnaryKind::Cos, "cos", "f16", f16::ONE, baseline_ref);
    bench_unary::<f32>(c, UnaryKind::Erf, "erf", "f32", 1.0_f32, baseline_ref);
    bench_unary::<f16>(c, UnaryKind::Erf, "erf", "f16", f16::ONE, baseline_ref);
    // Additional binaries.
    bench_binary::<f32>(c, BinaryKind::Sub, "sub", "f32", 1.0_f32, baseline_ref);
    bench_binary::<f16>(c, BinaryKind::Sub, "sub", "f16", f16::ONE, baseline_ref);
    bench_binary::<f32>(c, BinaryKind::Div, "div", "f32", 1.0_f32, baseline_ref);
    bench_binary::<f16>(c, BinaryKind::Div, "div", "f16", f16::ONE, baseline_ref);
    bench_binary::<f32>(c, BinaryKind::Maximum, "maximum", "f32", 1.0_f32, baseline_ref);
    bench_binary::<f16>(c, BinaryKind::Maximum, "maximum", "f16", f16::ONE, baseline_ref);
    bench_binary::<f32>(c, BinaryKind::Minimum, "minimum", "f32", 1.0_f32, baseline_ref);
    bench_binary::<f16>(c, BinaryKind::Minimum, "minimum", "f16", f16::ONE, baseline_ref);
    bench_binary::<f32>(c, BinaryKind::Pow, "pow", "f32", 1.0_f32, baseline_ref);
    bench_binary::<f16>(c, BinaryKind::Pow, "pow", "f16", f16::ONE, baseline_ref);
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
