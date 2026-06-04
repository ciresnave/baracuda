//! Reductions throughput — baracuda vs cuDNN reduce_tensor.
//!
//! Reduces along the last axis of a 2-D `(rows, hidden)` tensor. Both
//! baracuda and cuDNN are configured for the same `Sum` / `Max` /
//! `Mean` (cuDNN: `Avg`) reduction.
//!
//! cuDNN reduce shape: input `(rows, hidden, 1, 1)`, output
//! `(rows, 1, 1, 1)`. Reduction over the `C` axis (== `hidden`).
//!
//! Sweeps:
//! - `rows ∈ {512, 2048, 4096}`, `hidden ∈ {1024, 4096}`.
//! - Reduction kinds: `Sum`, `Max`, `Mean`.
//! - Dtype: `f32` only (cuDNN's reduce_tensor supports more, but the
//!   bench scope stays tight for cycle time).

#[cfg(not(feature = "cudnn"))]
fn main() {
    eprintln!(
        "reductions_vs_cudnn: the `cudnn` feature is disabled — \
         no work will run. Build with `--features cudnn`."
    );
}

#[cfg(feature = "cudnn")]
mod cudnn_impl {
    use baracuda_cudnn::{
        reduce_tensor, DType, Handle as CudnnHandle, ReduceOp, ReduceTensorDescriptor,
        TensorDescriptor, TensorFormat,
    };
    use baracuda_driver::DeviceBuffer;
    use baracuda_kernels::{
        contiguous_stride, ElementKind, PlanPreference, ReduceArgs, ReduceDescriptor, ReduceKind,
        ReducePlan, TensorMut, TensorRef, Workspace,
    };
    use baracuda_kernels_bench::{
        append_csv_row, measure_median_ns, setup_device, time_with_events, warmup,
        PhaseTwentyNineRow, PytorchBaseline, CROSS_HIDDEN_SWEEP, CROSS_SEQLEN_SWEEP,
    };
    use criterion::{BenchmarkId, Criterion};

    pub const BENCH_NAME: &str = "reductions_vs_cudnn";

    pub fn leak_str(s: &str) -> &'static str {
        Box::leak(s.to_owned().into_boxed_str())
    }

    fn bench_baracuda_one(
        c: &mut Criterion,
        kind: ReduceKind,
        kind_label: &str,
    ) {
        let (ctx, stream) = setup_device();
        let mut group =
            c.benchmark_group(format!("reductions_vs_cudnn/baracuda/{kind_label}_f32"));

        for &rows in CROSS_SEQLEN_SWEEP {
            for &hidden in CROSS_HIDDEN_SWEEP {
                let shape_str = format!("R{rows}_H{hidden}");
                let numel = (rows * hidden) as usize;

                let host: Vec<f32> = vec![1.0; numel];
                let dev_x = match DeviceBuffer::from_slice(&ctx, &host) {
                    Ok(b) => b,
                    Err(_) => continue,
                };
                let mut dev_y: DeviceBuffer<f32> =
                    match DeviceBuffer::zeros(&ctx, rows as usize) {
                        Ok(b) => b,
                        Err(_) => continue,
                    };

                let desc = ReduceDescriptor::<2> {
                    kind,
                    input_shape: [rows, hidden],
                    reduce_axis: 1,
                    element: ElementKind::F32,
                    correction: 1,
                };
                let plan = match ReducePlan::<f32, 2>::select(
                    &stream,
                    &desc,
                    PlanPreference::default(),
                ) {
                    Ok(p) => p,
                    Err(_) => continue,
                };
                let xs = [rows, hidden];
                let stx = contiguous_stride(xs);
                let ys = [rows, 1];
                let sty = contiguous_stride(ys);

                warmup(&stream, || {
                    let args = ReduceArgs::<f32, 2> {
                        x: TensorRef {
                            data: dev_x.as_slice(),
                            shape: xs,
                            stride: stx,
                        },
                        y: TensorMut {
                            data: dev_y.as_slice_mut(),
                            shape: ys,
                            stride: sty,
                        },
                    };
                    plan.run(&stream, Workspace::None, args).expect("baracuda reduce");
                });
                let baracuda_ns = measure_median_ns(&ctx, &stream, 11, 50, || {
                    let args = ReduceArgs::<f32, 2> {
                        x: TensorRef {
                            data: dev_x.as_slice(),
                            shape: xs,
                            stride: stx,
                        },
                        y: TensorMut {
                            data: dev_y.as_slice_mut(),
                            shape: ys,
                            stride: sty,
                        },
                    };
                    plan.run(&stream, Workspace::None, args).expect("baracuda reduce");
                });
                append_csv_row(
                    BENCH_NAME,
                    &PhaseTwentyNineRow {
                        op: leak_str(&format!("reduce_{kind_label}")),
                        shape: shape_str.clone(),
                        dtype: "f32",
                        baracuda_ns,
                        reference_ns: None,
                        reference: "baracuda",
                        pytorch_ns: None,
                    },
                );
                group.bench_with_input(BenchmarkId::from_parameter(&shape_str), &(), |bb, _| {
                    bb.iter_custom(|iters| {
                        time_with_events(&ctx, &stream, iters, || {
                            let args = ReduceArgs::<f32, 2> {
                                x: TensorRef {
                                    data: dev_x.as_slice(),
                                    shape: xs,
                                    stride: stx,
                                },
                                y: TensorMut {
                                    data: dev_y.as_slice_mut(),
                                    shape: ys,
                                    stride: sty,
                                },
                            };
                            plan.run(&stream, Workspace::None, args).expect("baracuda reduce");
                        })
                    });
                });
            }
        }
        group.finish();
    }

    fn bench_cudnn_one(
        c: &mut Criterion,
        op: ReduceOp,
        kind_label: &str,
        baseline: Option<&PytorchBaseline>,
    ) {
        let (ctx, stream) = setup_device();
        let cudnn = CudnnHandle::new().expect("cudnn handle");
        cudnn.set_stream(&stream).expect("cudnn set_stream");

        let mut group =
            c.benchmark_group(format!("reductions_vs_cudnn/cudnn/{kind_label}_f32"));

        for &rows in CROSS_SEQLEN_SWEEP {
            for &hidden in CROSS_HIDDEN_SWEEP {
                let shape_str = format!("R{rows}_H{hidden}");
                let numel = (rows * hidden) as usize;

                let host: Vec<f32> = vec![1.0; numel];
                let dev_x = match DeviceBuffer::from_slice(&ctx, &host) {
                    Ok(b) => b,
                    Err(_) => continue,
                };
                let mut dev_y: DeviceBuffer<f32> = match DeviceBuffer::zeros(&ctx, rows as usize)
                {
                    Ok(b) => b,
                    Err(_) => continue,
                };

                // Input: 4-D (rows, hidden, 1, 1), output (rows, 1, 1, 1).
                // cuDNN reduces axes where the output extent is 1.
                let x_desc = TensorDescriptor::new_4d(
                    TensorFormat::Nchw,
                    DType::F32,
                    rows,
                    hidden,
                    1,
                    1,
                )
                .expect("x_desc");
                let y_desc =
                    TensorDescriptor::new_4d(TensorFormat::Nchw, DType::F32, rows, 1, 1, 1)
                        .expect("y_desc");

                let reducer = ReduceTensorDescriptor::new(op, DType::F32).expect("reduce desc");
                let ws_bytes = reducer
                    .workspace_size(&cudnn, &x_desc, &y_desc)
                    .expect("ws size");
                let mut dev_ws: DeviceBuffer<u8> =
                    DeviceBuffer::zeros(&ctx, ws_bytes.max(1)).expect("alloc ws");

                warmup(&stream, || {
                    reduce_tensor(
                        &cudnn, &reducer, &mut dev_ws, 1.0, &x_desc, &dev_x, 0.0, &y_desc,
                        &mut dev_y,
                    )
                    .expect("cudnn reduce");
                });
                let cudnn_ns = measure_median_ns(&ctx, &stream, 11, 50, || {
                    reduce_tensor(
                        &cudnn, &reducer, &mut dev_ws, 1.0, &x_desc, &dev_x, 0.0, &y_desc,
                        &mut dev_y,
                    )
                    .expect("cudnn reduce");
                });
                let op_name = format!("reduce_{kind_label}");
                append_csv_row(
                    BENCH_NAME,
                    &PhaseTwentyNineRow {
                        op: leak_str(&op_name),
                        shape: shape_str.clone(),
                        dtype: "f32",
                        baracuda_ns: 0.0,
                        reference_ns: Some(cudnn_ns),
                        reference: "cuDNN",
                        pytorch_ns: baseline.and_then(|b| b.lookup(&op_name, &shape_str, "f32")),
                    },
                );
                group.bench_with_input(BenchmarkId::from_parameter(&shape_str), &(), |bb, _| {
                    bb.iter_custom(|iters| {
                        time_with_events(&ctx, &stream, iters, || {
                            reduce_tensor(
                                &cudnn, &reducer, &mut dev_ws, 1.0, &x_desc, &dev_x, 0.0,
                                &y_desc, &mut dev_y,
                            )
                            .expect("cudnn reduce");
                        })
                    });
                });
            }
        }
        group.finish();
    }

    /// Top-level criterion entry - invoked by criterion_main!.
    pub fn benches(c: &mut Criterion) {
        let baseline = PytorchBaseline::load_default();
        let baseline_ref = baseline.as_ref();
        bench_baracuda_one(c, ReduceKind::Sum, "sum");
        bench_cudnn_one(c, ReduceOp::Add, "sum", baseline_ref);
        bench_baracuda_one(c, ReduceKind::Max, "max");
        bench_cudnn_one(c, ReduceOp::Max, "max", baseline_ref);
        bench_baracuda_one(c, ReduceKind::Mean, "mean");
        bench_cudnn_one(c, ReduceOp::Avg, "mean", baseline_ref);
    }
}

#[cfg(feature = "cudnn")]
use criterion::{criterion_group, criterion_main};

// `criterion_group!` expands into a `pub fn benches_grp` whose
// signature is fixed by the macro - can't doc-comment it directly, so
// suppress the workspace `missing_docs = deny` lint on the generated fn.
#[cfg(feature = "cudnn")]
#[allow(missing_docs)]
mod criterion_glue {
    use super::*;
    criterion_group!(benches_grp, cudnn_impl::benches);
}
#[cfg(feature = "cudnn")]
criterion_main!(criterion_glue::benches_grp);
