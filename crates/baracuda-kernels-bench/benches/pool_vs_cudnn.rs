//! Pool2d throughput — baracuda vs raw cuDNN (MaxPool2d / AvgPool2d FW).
//!
//! baracuda's `MaxPool2dPlan` / `AvgPool2dPlan` are cuDNN-backed, so
//! this bench measures wrapper overhead on the same shape grid.

#[cfg(not(feature = "cudnn"))]
fn main() {
    eprintln!(
        "pool_vs_cudnn: the `cudnn` feature is disabled — \
         no work will run. Build with `--features cudnn`."
    );
}

#[cfg(feature = "cudnn")]
mod cudnn_impl {
    use baracuda_cudnn::{
        pooling_forward, CudnnDataType, DType, Handle as CudnnHandle, PoolingDescriptor,
        PoolingMode as CudnnPoolMode, TensorDescriptor, TensorFormat,
    };
    use baracuda_driver::DeviceBuffer;
    use baracuda_kernels::{
        contiguous_stride, AvgPool2dPlan, ElementKind, MaxPool2dPlan, PlanPreference,
        Pool2dDescriptor, Pool2dFwArgs, PoolMode, TensorMut, TensorRef, Workspace,
    };
    use baracuda_kernels_bench::{
        append_csv_row, measure_median_ns, setup_device, time_with_events, warmup,
        PhaseTwentyNineRow, PoolShape, PytorchBaseline, POOL_SWEEP,
    };
    use criterion::{BenchmarkId, Criterion};
    use half::f16;

    pub const BENCH_NAME: &str = "pool_vs_cudnn";

    pub fn leak_str(s: &str) -> &'static str {
        Box::leak(s.to_owned().into_boxed_str())
    }

    fn out_dims(shape: PoolShape) -> (i32, i32) {
        let h_out = (shape.h + 2 * shape.pad - shape.k) / shape.stride + 1;
        let w_out = (shape.w + 2 * shape.pad - shape.k) / shape.stride + 1;
        (h_out, w_out)
    }

    pub fn bench_baracuda<T>(c: &mut Criterion, dtype_label: &str, kind: ElementKind, fill: T)
    where
        T: baracuda_kernels::Element + Copy + 'static,
    {
        let (ctx, stream) = setup_device();
        let mut group = c.benchmark_group(format!("pool_vs_cudnn/baracuda/{dtype_label}"));

        for &shape in POOL_SWEEP {
            let (h_out, w_out) = out_dims(shape);
            let label = format!(
                "N{}_C{}_H{}_W{}_K{}_S{}",
                shape.n, shape.c, shape.h, shape.w, shape.k, shape.stride
            );

            let x_numel = (shape.n * shape.c * shape.h * shape.w) as usize;
            let y_numel = (shape.n * shape.c * h_out * w_out) as usize;
            let host_x: Vec<T> = vec![fill; x_numel];
            let dev_x = match DeviceBuffer::from_slice(&ctx, &host_x) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let mut dev_y: DeviceBuffer<T> = match DeviceBuffer::zeros(&ctx, y_numel) {
                Ok(b) => b,
                Err(_) => continue,
            };

            let desc = Pool2dDescriptor::new(
                shape.n,
                shape.c,
                shape.h,
                shape.w,
                shape.k,
                shape.k,
                PoolMode::Max,
                kind,
            )
            .with_padding(shape.pad, shape.pad)
            .with_stride(shape.stride, shape.stride);
            let plan = match MaxPool2dPlan::<T>::select(&stream, &desc, PlanPreference::default())
            {
                Ok(p) => p,
                Err(_) => continue,
            };

            let x_shape = [shape.n, shape.c, shape.h, shape.w];
            let y_shape = [shape.n, shape.c, h_out, w_out];
            let stx = contiguous_stride(x_shape);
            let sty = contiguous_stride(y_shape);

            warmup(&stream, || {
                let args = Pool2dFwArgs::<T> {
                    x: TensorRef {
                        data: dev_x.as_slice(),
                        shape: x_shape,
                        stride: stx,
                    },
                    y: TensorMut {
                        data: dev_y.as_slice_mut(),
                        shape: y_shape,
                        stride: sty,
                    },
                };
                plan.run_fw(&stream, Workspace::None, args).expect("baracuda maxpool");
            });
            let baracuda_ns = measure_median_ns(&ctx, &stream, 11, 50, || {
                let args = Pool2dFwArgs::<T> {
                    x: TensorRef {
                        data: dev_x.as_slice(),
                        shape: x_shape,
                        stride: stx,
                    },
                    y: TensorMut {
                        data: dev_y.as_slice_mut(),
                        shape: y_shape,
                        stride: sty,
                    },
                };
                plan.run_fw(&stream, Workspace::None, args).expect("baracuda maxpool");
            });
            append_csv_row(
                BENCH_NAME,
                &PhaseTwentyNineRow {
                    op: "maxpool2d",
                    shape: label.clone(),
                    dtype: leak_str(dtype_label),
                    baracuda_ns,
                    reference_ns: None,
                    reference: "baracuda",
                    pytorch_ns: None,
                },
            );
            group.bench_with_input(BenchmarkId::from_parameter(&label), &(), |bb, _| {
                bb.iter_custom(|iters| {
                    time_with_events(&ctx, &stream, iters, || {
                        let args = Pool2dFwArgs::<T> {
                            x: TensorRef {
                                data: dev_x.as_slice(),
                                shape: x_shape,
                                stride: stx,
                            },
                            y: TensorMut {
                                data: dev_y.as_slice_mut(),
                                shape: y_shape,
                                stride: sty,
                            },
                        };
                        plan.run_fw(&stream, Workspace::None, args)
                            .expect("baracuda maxpool");
                    })
                });
            });
        }
        group.finish();
    }

    pub fn bench_cudnn<T: CudnnDataType + Default>(
        c: &mut Criterion,
        dtype_label: &str,
        dtype: DType,
        baseline: Option<&PytorchBaseline>,
    ) {
        let (ctx, stream) = setup_device();
        let cudnn = CudnnHandle::new().expect("cudnn handle");
        cudnn.set_stream(&stream).expect("cudnn set_stream");
        let mut group = c.benchmark_group(format!("pool_vs_cudnn/cudnn/{dtype_label}"));

        for &shape in POOL_SWEEP {
            let (h_out, w_out) = out_dims(shape);
            let label = format!(
                "N{}_C{}_H{}_W{}_K{}_S{}",
                shape.n, shape.c, shape.h, shape.w, shape.k, shape.stride
            );
            let x_numel = (shape.n * shape.c * shape.h * shape.w) as usize;
            let y_numel = (shape.n * shape.c * h_out * w_out) as usize;

            let host_x: Vec<T> = vec![T::default(); x_numel];
            let dev_x = match DeviceBuffer::from_slice(&ctx, &host_x) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let mut dev_y: DeviceBuffer<T> = match DeviceBuffer::zeros(&ctx, y_numel) {
                Ok(b) => b,
                Err(_) => continue,
            };

            let x_desc = TensorDescriptor::new_4d(
                TensorFormat::Nchw,
                dtype,
                shape.n,
                shape.c,
                shape.h,
                shape.w,
            )
            .expect("x_desc");
            let y_desc = TensorDescriptor::new_4d(
                TensorFormat::Nchw,
                dtype,
                shape.n,
                shape.c,
                h_out,
                w_out,
            )
            .expect("y_desc");
            let pool_desc = PoolingDescriptor::new_2d(
                CudnnPoolMode::Max,
                shape.k,
                shape.k,
                shape.pad,
                shape.pad,
                shape.stride,
                shape.stride,
            )
            .expect("pool_desc");

            warmup(&stream, || {
                pooling_forward(&cudnn, &pool_desc, 1.0, &x_desc, &dev_x, 0.0, &y_desc, &mut dev_y)
                    .expect("cudnn pool");
            });
            let cudnn_ns = measure_median_ns(&ctx, &stream, 11, 50, || {
                pooling_forward(&cudnn, &pool_desc, 1.0, &x_desc, &dev_x, 0.0, &y_desc, &mut dev_y)
                    .expect("cudnn pool");
            });
            append_csv_row(
                BENCH_NAME,
                &PhaseTwentyNineRow {
                    op: "maxpool2d",
                    shape: label.clone(),
                    dtype: leak_str(dtype_label),
                    baracuda_ns: 0.0,
                    reference_ns: Some(cudnn_ns),
                    reference: "cuDNN",
                    pytorch_ns: baseline.and_then(|b| b.lookup("maxpool2d", &label, dtype_label)),
                },
            );
            group.bench_with_input(BenchmarkId::from_parameter(&label), &(), |bb, _| {
                bb.iter_custom(|iters| {
                    time_with_events(&ctx, &stream, iters, || {
                        pooling_forward(
                            &cudnn, &pool_desc, 1.0, &x_desc, &dev_x, 0.0, &y_desc, &mut dev_y,
                        )
                        .expect("cudnn pool");
                    })
                });
            });
        }
        group.finish();
    }

    // Phase 73.7: AvgPool2d parallel benches. baracuda's `AvgPool2dPlan`
    // is a distinct type from `MaxPool2dPlan` (different kernels under
    // the hood), so the bench function is duplicated rather than
    // parameterized. Same `Pool2dDescriptor` / `Pool2dFwArgs` shape;
    // only the mode + plan type + op label change.

    pub fn bench_baracuda_avg<T>(
        c: &mut Criterion,
        dtype_label: &str,
        kind: ElementKind,
        fill: T,
    ) where
        T: baracuda_kernels::Element + Copy + 'static,
    {
        let (ctx, stream) = setup_device();
        let mut group = c.benchmark_group(format!("pool_vs_cudnn/baracuda/avg/{dtype_label}"));

        for &shape in POOL_SWEEP {
            let (h_out, w_out) = out_dims(shape);
            let label = format!(
                "N{}_C{}_H{}_W{}_K{}_S{}",
                shape.n, shape.c, shape.h, shape.w, shape.k, shape.stride
            );

            let x_numel = (shape.n * shape.c * shape.h * shape.w) as usize;
            let y_numel = (shape.n * shape.c * h_out * w_out) as usize;
            let host_x: Vec<T> = vec![fill; x_numel];
            let dev_x = match DeviceBuffer::from_slice(&ctx, &host_x) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let mut dev_y: DeviceBuffer<T> = match DeviceBuffer::zeros(&ctx, y_numel) {
                Ok(b) => b,
                Err(_) => continue,
            };

            let desc = Pool2dDescriptor::new(
                shape.n,
                shape.c,
                shape.h,
                shape.w,
                shape.k,
                shape.k,
                PoolMode::AvgIncludePad,
                kind,
            )
            .with_padding(shape.pad, shape.pad)
            .with_stride(shape.stride, shape.stride);
            let plan = match AvgPool2dPlan::<T>::select(&stream, &desc, PlanPreference::default())
            {
                Ok(p) => p,
                Err(_) => continue,
            };

            let x_shape = [shape.n, shape.c, shape.h, shape.w];
            let y_shape = [shape.n, shape.c, h_out, w_out];
            let stx = contiguous_stride(x_shape);
            let sty = contiguous_stride(y_shape);

            warmup(&stream, || {
                let args = Pool2dFwArgs::<T> {
                    x: TensorRef {
                        data: dev_x.as_slice(),
                        shape: x_shape,
                        stride: stx,
                    },
                    y: TensorMut {
                        data: dev_y.as_slice_mut(),
                        shape: y_shape,
                        stride: sty,
                    },
                };
                plan.run_fw(&stream, Workspace::None, args).expect("baracuda avgpool");
            });
            let baracuda_ns = measure_median_ns(&ctx, &stream, 11, 50, || {
                let args = Pool2dFwArgs::<T> {
                    x: TensorRef {
                        data: dev_x.as_slice(),
                        shape: x_shape,
                        stride: stx,
                    },
                    y: TensorMut {
                        data: dev_y.as_slice_mut(),
                        shape: y_shape,
                        stride: sty,
                    },
                };
                plan.run_fw(&stream, Workspace::None, args).expect("baracuda avgpool");
            });
            append_csv_row(
                BENCH_NAME,
                &PhaseTwentyNineRow {
                    op: "avgpool2d",
                    shape: label.clone(),
                    dtype: leak_str(dtype_label),
                    baracuda_ns,
                    reference_ns: None,
                    reference: "baracuda",
                    pytorch_ns: None,
                },
            );
            group.bench_with_input(BenchmarkId::from_parameter(&label), &(), |bb, _| {
                bb.iter_custom(|iters| {
                    time_with_events(&ctx, &stream, iters, || {
                        let args = Pool2dFwArgs::<T> {
                            x: TensorRef {
                                data: dev_x.as_slice(),
                                shape: x_shape,
                                stride: stx,
                            },
                            y: TensorMut {
                                data: dev_y.as_slice_mut(),
                                shape: y_shape,
                                stride: sty,
                            },
                        };
                        plan.run_fw(&stream, Workspace::None, args)
                            .expect("baracuda avgpool");
                    })
                });
            });
        }
        group.finish();
    }

    pub fn bench_cudnn_avg<T: CudnnDataType + Default>(
        c: &mut Criterion,
        dtype_label: &str,
        dtype: DType,
        baseline: Option<&PytorchBaseline>,
    ) {
        let (ctx, stream) = setup_device();
        let cudnn = CudnnHandle::new().expect("cudnn handle");
        cudnn.set_stream(&stream).expect("cudnn set_stream");
        let mut group = c.benchmark_group(format!("pool_vs_cudnn/cudnn/avg/{dtype_label}"));

        for &shape in POOL_SWEEP {
            let (h_out, w_out) = out_dims(shape);
            let label = format!(
                "N{}_C{}_H{}_W{}_K{}_S{}",
                shape.n, shape.c, shape.h, shape.w, shape.k, shape.stride
            );
            let x_numel = (shape.n * shape.c * shape.h * shape.w) as usize;
            let y_numel = (shape.n * shape.c * h_out * w_out) as usize;

            let host_x: Vec<T> = vec![T::default(); x_numel];
            let dev_x = match DeviceBuffer::from_slice(&ctx, &host_x) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let mut dev_y: DeviceBuffer<T> = match DeviceBuffer::zeros(&ctx, y_numel) {
                Ok(b) => b,
                Err(_) => continue,
            };

            let x_desc = TensorDescriptor::new_4d(
                TensorFormat::Nchw,
                dtype,
                shape.n,
                shape.c,
                shape.h,
                shape.w,
            )
            .expect("x_desc");
            let y_desc = TensorDescriptor::new_4d(
                TensorFormat::Nchw,
                dtype,
                shape.n,
                shape.c,
                h_out,
                w_out,
            )
            .expect("y_desc");
            // PyTorch's default `F.avg_pool2d` includes padded cells in the
            // denominator (`count_include_pad=True`); use the cuDNN matching
            // variant for a fair compare.
            let pool_desc = PoolingDescriptor::new_2d(
                CudnnPoolMode::AverageCountIncludePadding,
                shape.k,
                shape.k,
                shape.pad,
                shape.pad,
                shape.stride,
                shape.stride,
            )
            .expect("pool_desc");

            warmup(&stream, || {
                pooling_forward(&cudnn, &pool_desc, 1.0, &x_desc, &dev_x, 0.0, &y_desc, &mut dev_y)
                    .expect("cudnn pool");
            });
            let cudnn_ns = measure_median_ns(&ctx, &stream, 11, 50, || {
                pooling_forward(&cudnn, &pool_desc, 1.0, &x_desc, &dev_x, 0.0, &y_desc, &mut dev_y)
                    .expect("cudnn pool");
            });
            append_csv_row(
                BENCH_NAME,
                &PhaseTwentyNineRow {
                    op: "avgpool2d",
                    shape: label.clone(),
                    dtype: leak_str(dtype_label),
                    baracuda_ns: 0.0,
                    reference_ns: Some(cudnn_ns),
                    reference: "cuDNN",
                    pytorch_ns: baseline.and_then(|b| b.lookup("avgpool2d", &label, dtype_label)),
                },
            );
            group.bench_with_input(BenchmarkId::from_parameter(&label), &(), |bb, _| {
                bb.iter_custom(|iters| {
                    time_with_events(&ctx, &stream, iters, || {
                        pooling_forward(
                            &cudnn, &pool_desc, 1.0, &x_desc, &dev_x, 0.0, &y_desc, &mut dev_y,
                        )
                        .expect("cudnn pool");
                    })
                });
            });
        }
        group.finish();
    }

    /// Top-level criterion entry - invoked by criterion_main!.
    pub fn benches(c: &mut Criterion) {
        let baseline = PytorchBaseline::load_default();
        let baseline_ref = baseline.as_ref();
        bench_baracuda::<f32>(c, "f32", ElementKind::F32, 1.0_f32);
        bench_cudnn::<f32>(c, "f32", DType::F32, baseline_ref);
        bench_baracuda::<f16>(c, "f16", ElementKind::F16, f16::ONE);
        bench_cudnn::<f16>(c, "f16", DType::F16, baseline_ref);
        // Phase 73.7: AvgPool2d pass.
        bench_baracuda_avg::<f32>(c, "f32", ElementKind::F32, 1.0_f32);
        bench_cudnn_avg::<f32>(c, "f32", DType::F32, baseline_ref);
        bench_baracuda_avg::<f16>(c, "f16", ElementKind::F16, f16::ONE);
        bench_cudnn_avg::<f16>(c, "f16", DType::F16, baseline_ref);
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
