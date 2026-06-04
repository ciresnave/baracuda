//! Softmax throughput — baracuda vs cuDNN.
//!
//! cuDNN's `cudnnSoftmaxForward` runs on a 4-D NCHW tensor and reduces
//! along the `C` axis (instance-mode) or the per-spatial-pixel axis
//! (channel-mode). For an apples-to-apples row-softmax comparison
//! against baracuda's `SoftmaxPlan` on a 2-D `(rows, cols)` tensor, we
//! shape the cuDNN tensor as `(rows, cols, 1, 1)` with `SoftmaxMode::Channel`
//! and reduce along `C` — which is the `cols` axis.
//!
//! Sweeps:
//! - `rows ∈ {512, 2048, 4096}`, `cols ∈ {1024, 4096}`.
//! - Dtypes: `f32`, `f16`.
//!
//! Reference: cuDNN `Accurate` algo with `SoftmaxMode::Channel`.

#[cfg(not(feature = "cudnn"))]
fn main() {
    eprintln!(
        "softmax_vs_cudnn: the `cudnn` feature is disabled — \
         no work will run. Build with `--features cudnn`."
    );
}

#[cfg(feature = "cudnn")]
mod cudnn_impl {
    use baracuda_cudnn::{
        softmax_forward, DType, Handle as CudnnHandle, SoftmaxAlgo, SoftmaxMode,
        TensorDescriptor, TensorFormat,
    };
    use baracuda_driver::DeviceBuffer;
    use baracuda_kernels::{
        contiguous_stride, ElementKind, PlanPreference, SoftmaxArgs, SoftmaxDescriptor,
        SoftmaxKind, SoftmaxPlan, TensorMut, TensorRef, Workspace,
    };
    use baracuda_kernels_bench::{
        append_csv_row, measure_median_ns, setup_device, time_with_events, warmup,
        PhaseTwentyNineRow, PytorchBaseline, CROSS_HIDDEN_SWEEP, CROSS_SEQLEN_SWEEP,
    };
    use criterion::{BenchmarkId, Criterion};
    use half::f16;

    pub const BENCH_NAME: &str = "softmax_vs_cudnn";

    pub fn leak_str(s: &str) -> &'static str {
        Box::leak(s.to_owned().into_boxed_str())
    }

    pub fn bench_baracuda<T>(c: &mut Criterion, dtype_label: &str, fill: T)
    where
        T: baracuda_kernels::Element + Copy + 'static,
    {
        let (ctx, stream) = setup_device();
        let mut group = c.benchmark_group(format!("softmax_vs_cudnn/baracuda/{dtype_label}"));

        for &rows in CROSS_SEQLEN_SWEEP {
            for &cols in CROSS_HIDDEN_SWEEP {
                let shape = format!("R{rows}_C{cols}");
                let numel = (rows * cols) as usize;

                let host: Vec<T> = vec![fill; numel];
                let dev_x = match DeviceBuffer::from_slice(&ctx, &host) {
                    Ok(b) => b,
                    Err(_) => continue,
                };
                let mut dev_y: DeviceBuffer<T> = match DeviceBuffer::zeros(&ctx, numel) {
                    Ok(b) => b,
                    Err(_) => continue,
                };

                let desc = SoftmaxDescriptor::<2> {
                    kind: SoftmaxKind::Softmax,
                    input_shape: [rows, cols],
                    softmax_axis: 1,
                    element: T::KIND,
                };
                let plan = match SoftmaxPlan::<T, 2>::select(
                    &stream,
                    &desc,
                    PlanPreference::default(),
                ) {
                    Ok(p) => p,
                    Err(_) => continue,
                };
                let xs = [rows, cols];
                let stx = contiguous_stride(xs);

                warmup(&stream, || {
                    let args = SoftmaxArgs::<T, 2> {
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
                    plan.run(&stream, Workspace::None, args).expect("baracuda softmax");
                });
                let baracuda_ns = measure_median_ns(&ctx, &stream, 11, 50, || {
                    let args = SoftmaxArgs::<T, 2> {
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
                    plan.run(&stream, Workspace::None, args).expect("baracuda softmax");
                });
                append_csv_row(
                    BENCH_NAME,
                    &PhaseTwentyNineRow {
                        op: "softmax",
                        shape: shape.clone(),
                        dtype: leak_str(dtype_label),
                        baracuda_ns,
                        reference_ns: None,
                        reference: "baracuda",
                        pytorch_ns: None,
                    },
                );
                group.bench_with_input(BenchmarkId::from_parameter(&shape), &(), |bb, _| {
                    bb.iter_custom(|iters| {
                        time_with_events(&ctx, &stream, iters, || {
                            let args = SoftmaxArgs::<T, 2> {
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
                            plan.run(&stream, Workspace::None, args).expect("baracuda softmax");
                        })
                    });
                });
                // Silence unused warning for the ElementKind import on
                // sm80-only configurations.
                let _ = ElementKind::F32;
            }
        }
        group.finish();
    }

    pub fn bench_cudnn<T: baracuda_cudnn::CudnnDataType + Default>(
        c: &mut Criterion,
        dtype_label: &str,
        dtype: DType,
        baseline: Option<&PytorchBaseline>,
    ) {
        let (ctx, stream) = setup_device();
        let cudnn = CudnnHandle::new().expect("cudnn handle");
        cudnn.set_stream(&stream).expect("cudnn set_stream");

        let mut group = c.benchmark_group(format!("softmax_vs_cudnn/cudnn/{dtype_label}"));

        for &rows in CROSS_SEQLEN_SWEEP {
            for &cols in CROSS_HIDDEN_SWEEP {
                let shape = format!("R{rows}_C{cols}");
                let numel = (rows * cols) as usize;

                let host: Vec<T> = vec![T::default(); numel];
                let dev_x = match DeviceBuffer::from_slice(&ctx, &host) {
                    Ok(b) => b,
                    Err(_) => continue,
                };
                let mut dev_y: DeviceBuffer<T> = match DeviceBuffer::zeros(&ctx, numel) {
                    Ok(b) => b,
                    Err(_) => continue,
                };

                // (N, C, H, W) = (rows, cols, 1, 1) — softmax along C.
                let x_desc =
                    TensorDescriptor::new_4d(TensorFormat::Nchw, dtype, rows, cols, 1, 1)
                        .expect("x_desc");
                let y_desc =
                    TensorDescriptor::new_4d(TensorFormat::Nchw, dtype, rows, cols, 1, 1)
                        .expect("y_desc");

                warmup(&stream, || {
                    softmax_forward(
                        &cudnn,
                        SoftmaxAlgo::Accurate,
                        SoftmaxMode::Channel,
                        1.0,
                        &x_desc,
                        &dev_x,
                        0.0,
                        &y_desc,
                        &mut dev_y,
                    )
                    .expect("cudnn softmax_forward");
                });
                let cudnn_ns = measure_median_ns(&ctx, &stream, 11, 50, || {
                    softmax_forward(
                        &cudnn,
                        SoftmaxAlgo::Accurate,
                        SoftmaxMode::Channel,
                        1.0,
                        &x_desc,
                        &dev_x,
                        0.0,
                        &y_desc,
                        &mut dev_y,
                    )
                    .expect("cudnn softmax_forward");
                });
                append_csv_row(
                    BENCH_NAME,
                    &PhaseTwentyNineRow {
                        op: "softmax",
                        shape: shape.clone(),
                        dtype: leak_str(dtype_label),
                        baracuda_ns: 0.0,
                        reference_ns: Some(cudnn_ns),
                        reference: "cuDNN",
                        pytorch_ns: baseline.and_then(|b| b.lookup("softmax", &shape, dtype_label)),
                    },
                );
                group.bench_with_input(BenchmarkId::from_parameter(&shape), &(), |bb, _| {
                    bb.iter_custom(|iters| {
                        time_with_events(&ctx, &stream, iters, || {
                            softmax_forward(
                                &cudnn,
                                SoftmaxAlgo::Accurate,
                                SoftmaxMode::Channel,
                                1.0,
                                &x_desc,
                                &dev_x,
                                0.0,
                                &y_desc,
                                &mut dev_y,
                            )
                            .expect("cudnn softmax_forward");
                        })
                    });
                });
            }
        }
        group.finish();
    }

    pub fn softmax_benches(c: &mut Criterion) {
        let baseline = PytorchBaseline::load_default();
        let baseline_ref = baseline.as_ref();
        bench_baracuda::<f32>(c, "f32", 1.0_f32);
        bench_cudnn::<f32>(c, "f32", DType::F32, baseline_ref);
        bench_baracuda::<f16>(c, "f16", f16::ONE);
        bench_cudnn::<f16>(c, "f16", DType::F16, baseline_ref);
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
    criterion_group!(benches_grp, cudnn_impl::softmax_benches);
}
#[cfg(feature = "cudnn")]
criterion_main!(criterion_glue::benches_grp);
