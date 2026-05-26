//! Conv2d throughput — baracuda vs raw cuDNN.
//!
//! baracuda's `Conv2dPlan` is **already** cuDNN-backed
//! (`BackendKind::Cudnn`), so this bench measures **wrapper overhead**
//! rather than a difference in the underlying kernel. A delta close to
//! 1.0 means the Rust facade adds negligible overhead; a meaningful
//! delta would indicate the Plan's host work (descriptor build, algo
//! selection, workspace query) is unexpectedly expensive.
//!
//! The bench reuses Phase 10's `CONV2D_SWEEP` (3 ResNet-50 picks).
//!
//! Dtypes: `f32`, `f16`.

#[cfg(not(feature = "cudnn"))]
fn main() {
    eprintln!(
        "conv2d_vs_cudnn: the `cudnn` feature is disabled — \
         no work will run. Build with `--features cudnn`."
    );
}

#[cfg(feature = "cudnn")]
mod cudnn_impl {
    use baracuda_cudnn::{
        convolution_forward, convolution_forward_workspace_size, ConvMode, ConvolutionDescriptor,
        CudnnDataType, DType, FilterDescriptor, FwdAlgo, Handle as CudnnHandle, TensorDescriptor,
        TensorFormat,
    };
    use baracuda_driver::DeviceBuffer;
    use baracuda_kernels::{
        contiguous_stride, Conv2dArgs, Conv2dDescriptor, Conv2dPlan, ElementKind, PlanPreference,
        TensorMut, TensorRef, Workspace,
    };
    use baracuda_kernels_bench::{
        append_csv_row, conv2d_flops, measure_median_ns, setup_device, time_with_events, warmup,
        Conv2dShape, PhaseTwentyNineRow, CONV2D_SWEEP,
    };
    use criterion::{BenchmarkId, Criterion, Throughput};
    use half::f16;

    pub const BENCH_NAME: &str = "conv2d_vs_cudnn";

    pub fn leak_str(s: &str) -> &'static str {
        Box::leak(s.to_owned().into_boxed_str())
    }

    pub fn bench_baracuda<T>(c: &mut Criterion, dtype_label: &str, kind: ElementKind, fill: T)
    where
        T: baracuda_kernels::Element + Copy + 'static,
    {
        let (ctx, stream) = setup_device();
        let mut group = c.benchmark_group(format!("conv2d_vs_cudnn/baracuda/{dtype_label}"));

        for &shape in CONV2D_SWEEP {
            let (h_out, w_out) = (shape.hw, shape.hw); // pad=k/2, stride=1.
            let label = format!(
                "N{}_Cin{}_Cout{}_HW{}_K{}",
                shape.n, shape.c_in, shape.c_out, shape.hw, shape.k
            );

            let x_numel = (shape.n * shape.c_in * shape.hw * shape.hw) as usize;
            let w_numel = (shape.c_out * shape.c_in * shape.k * shape.k) as usize;
            let y_numel = (shape.n * shape.c_out * h_out * w_out) as usize;
            let host_x: Vec<T> = vec![fill; x_numel];
            let host_w: Vec<T> = vec![fill; w_numel];
            let dev_x = match DeviceBuffer::from_slice(&ctx, &host_x) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let dev_w = match DeviceBuffer::from_slice(&ctx, &host_w) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let mut dev_y: DeviceBuffer<T> = match DeviceBuffer::zeros(&ctx, y_numel) {
                Ok(b) => b,
                Err(_) => continue,
            };

            let pad = shape.k / 2;
            let desc = Conv2dDescriptor {
                batch: shape.n,
                c_in: shape.c_in,
                h_in: shape.hw,
                w_in: shape.hw,
                c_out: shape.c_out,
                h_filt: shape.k,
                w_filt: shape.k,
                pad_h: pad,
                pad_w: pad,
                stride_h: 1,
                stride_w: 1,
                dilation_h: 1,
                dilation_w: 1,
                groups: 1,
                element: kind,
            };
            let plan = match Conv2dPlan::<T>::select(&stream, &desc, PlanPreference::default()) {
                Ok(p) => p,
                Err(_) => continue,
            };
            let ws_bytes = plan.query_fw_workspace_size(&stream).expect("ws query");
            let mut dev_ws: DeviceBuffer<u8> =
                DeviceBuffer::zeros(&ctx, ws_bytes.max(1)).expect("alloc ws");

            let x_shape = [shape.n, shape.c_in, shape.hw, shape.hw];
            let w_shape = [shape.c_out, shape.c_in, shape.k, shape.k];
            let y_shape = [shape.n, shape.c_out, h_out, w_out];
            let stx = contiguous_stride(x_shape);
            let stw = contiguous_stride(w_shape);
            let sty = contiguous_stride(y_shape);

            let mut run = || {
                let workspace = if ws_bytes == 0 {
                    Workspace::None
                } else {
                    Workspace::Borrowed(dev_ws.as_slice_mut())
                };
                let args = Conv2dArgs::<T> {
                    x: TensorRef {
                        data: dev_x.as_slice(),
                        shape: x_shape,
                        stride: stx,
                    },
                    w: TensorRef {
                        data: dev_w.as_slice(),
                        shape: w_shape,
                        stride: stw,
                    },
                    y: TensorMut {
                        data: dev_y.as_slice_mut(),
                        shape: y_shape,
                        stride: sty,
                    },
                };
                plan.run_fw(&stream, workspace, args).expect("baracuda conv2d");
            };
            warmup(&stream, &mut run);
            let baracuda_ns = measure_median_ns(&ctx, &stream, 11, 20, &mut run);
            append_csv_row(
                BENCH_NAME,
                &PhaseTwentyNineRow {
                    op: "conv2d",
                    shape: label.clone(),
                    dtype: leak_str(dtype_label),
                    baracuda_ns,
                    reference_ns: None,
                    reference: "baracuda",
                },
            );

            group.throughput(Throughput::Elements(conv2d_flops(shape)));
            group.bench_with_input(BenchmarkId::from_parameter(&label), &(), |bb, _| {
                bb.iter_custom(|iters| time_with_events(&ctx, &stream, iters, &mut run));
            });
        }
        group.finish();
    }

    pub fn bench_cudnn<T: CudnnDataType + Default>(
        c: &mut Criterion,
        dtype_label: &str,
        dtype: DType,
    ) {
        let (ctx, stream) = setup_device();
        let cudnn = CudnnHandle::new().expect("cudnn handle");
        cudnn.set_stream(&stream).expect("cudnn set_stream");
        let mut group = c.benchmark_group(format!("conv2d_vs_cudnn/cudnn/{dtype_label}"));

        for &shape in CONV2D_SWEEP {
            let label = format!(
                "N{}_Cin{}_Cout{}_HW{}_K{}",
                shape.n, shape.c_in, shape.c_out, shape.hw, shape.k
            );
            let pad = shape.k / 2;
            let (h_out, w_out) = (shape.hw, shape.hw);

            let x_numel = (shape.n * shape.c_in * shape.hw * shape.hw) as usize;
            let w_numel = (shape.c_out * shape.c_in * shape.k * shape.k) as usize;
            let y_numel = (shape.n * shape.c_out * h_out * w_out) as usize;
            let host_x: Vec<T> = vec![T::default(); x_numel];
            let host_w: Vec<T> = vec![T::default(); w_numel];
            let dev_x = match DeviceBuffer::from_slice(&ctx, &host_x) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let dev_w = match DeviceBuffer::from_slice(&ctx, &host_w) {
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
                shape.c_in,
                shape.hw,
                shape.hw,
            )
            .expect("x_desc");
            let w_desc = FilterDescriptor::new_4d(
                TensorFormat::Nchw,
                dtype,
                shape.c_out,
                shape.c_in,
                shape.k,
                shape.k,
            )
            .expect("w_desc");
            let y_desc = TensorDescriptor::new_4d(
                TensorFormat::Nchw,
                dtype,
                shape.n,
                shape.c_out,
                h_out,
                w_out,
            )
            .expect("y_desc");
            let conv_desc = ConvolutionDescriptor::new_2d(
                pad,
                pad,
                1,
                1,
                1,
                1,
                ConvMode::CrossCorrelation,
                dtype,
            )
            .expect("conv_desc");

            let algo = FwdAlgo::ImplicitGemm;
            let ws_bytes = convolution_forward_workspace_size(
                &cudnn, &x_desc, &w_desc, &conv_desc, &y_desc, algo,
            )
            .expect("ws query");
            let mut dev_ws: DeviceBuffer<u8> =
                DeviceBuffer::zeros(&ctx, ws_bytes.max(1)).expect("alloc ws");

            let mut run = || {
                convolution_forward(
                    &cudnn,
                    1.0,
                    &x_desc,
                    &dev_x,
                    &w_desc,
                    &dev_w,
                    &conv_desc,
                    algo,
                    &mut dev_ws,
                    0.0,
                    &y_desc,
                    &mut dev_y,
                )
                .expect("cudnn conv2d");
            };
            warmup(&stream, &mut run);
            let cudnn_ns = measure_median_ns(&ctx, &stream, 11, 20, &mut run);
            append_csv_row(
                BENCH_NAME,
                &PhaseTwentyNineRow {
                    op: "conv2d",
                    shape: label.clone(),
                    dtype: leak_str(dtype_label),
                    baracuda_ns: 0.0,
                    reference_ns: Some(cudnn_ns),
                    reference: "cuDNN",
                },
            );

            group.throughput(Throughput::Elements(conv2d_flops(shape)));
            group.bench_with_input(BenchmarkId::from_parameter(&label), &(), |bb, _| {
                bb.iter_custom(|iters| time_with_events(&ctx, &stream, iters, &mut run));
            });
        }
        group.finish();
    }

    pub fn benches(c: &mut Criterion) {
        bench_baracuda::<f32>(c, "f32", ElementKind::F32, 1.0_f32);
        bench_cudnn::<f32>(c, "f32", DType::F32);
        bench_baracuda::<f16>(c, "f16", ElementKind::F16, f16::ONE);
        bench_cudnn::<f16>(c, "f16", DType::F16);
        // Silence unused-binding lint on builds without sm89.
        let _ = Conv2dShape {
            n: 0,
            c_in: 0,
            c_out: 0,
            hw: 0,
            k: 0,
        };
    }
}

#[cfg(feature = "cudnn")]
use criterion::{criterion_group, criterion_main};

#[cfg(feature = "cudnn")]
criterion_group!(benches_grp, cudnn_impl::benches);
#[cfg(feature = "cudnn")]
criterion_main!(benches_grp);
