//! Conv2d throughput bench — GFLOPS across ResNet-50 block shapes ×
//! dtypes. Gated behind the `cudnn` feature.
//!
//! Sweep:
//! - 3 representative ResNet-50 stages (stem / mid / deep).
//! - Dtypes: `f32` / `f16`.
//!
//! Reports `Throughput::Elements(flops)` where `flops = 2·macs`.
//! Divide criterion's printed `elem/sec` by `1e9` for GFLOPS.

#[cfg(not(feature = "cudnn"))]
fn main() {
    // Conv2d requires cuDNN; bench is a no-op without it.
    eprintln!(
        "baracuda-kernels-bench/conv2d: the `cudnn` feature is disabled — \
         no Conv2d work will run. Build with `--features cudnn` to enable."
    );
}

#[cfg(feature = "cudnn")]
mod cudnn_impl {
    use baracuda_driver::DeviceBuffer;
    use baracuda_kernels::{
        contiguous_stride, Conv2dArgs, Conv2dDescriptor, Conv2dPlan, ElementKind, PlanPreference,
        TensorMut, TensorRef, Workspace,
    };
    use baracuda_kernels_bench::{
        conv2d_flops, setup_device, time_with_events, warmup, Conv2dShape, CONV2D_SWEEP,
    };
    use criterion::{BenchmarkId, Criterion, Throughput};
    use half::f16;

    fn out_dims(shape: Conv2dShape) -> (i32, i32) {
        // pad = k/2, stride = 1 → H_out = H_in.
        (shape.hw, shape.hw)
    }

    /// Generic Conv2d FW bench body.
    pub fn bench_conv<T>(c: &mut Criterion, dtype_label: &str, kind: ElementKind, fill: T)
    where
        T: baracuda_kernels::Element + Copy + 'static,
    {
        let (ctx, stream) = setup_device();
        let mut group = c.benchmark_group(format!("conv2d/{dtype_label}"));

        for &shape in CONV2D_SWEEP {
            let (h_out, w_out) = out_dims(shape);
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

            group.throughput(Throughput::Elements(conv2d_flops(shape)));
            group.bench_with_input(BenchmarkId::from_parameter(&label), &(), |bb, _| {
                warmup(&stream, || {
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
                    plan.run_fw(&stream, workspace, args).expect("conv2d warmup");
                });

                bb.iter_custom(|iters| {
                    time_with_events(&ctx, &stream, iters, || {
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
                        plan.run_fw(&stream, workspace, args).expect("conv2d run");
                    })
                });
            });
        }
        group.finish();
    }

    pub fn conv2d_benches(c: &mut Criterion) {
        bench_conv::<f32>(c, "f32", ElementKind::F32, 1.0_f32);
        bench_conv::<f16>(c, "f16", ElementKind::F16, f16::ONE);
    }
}

#[cfg(feature = "cudnn")]
use criterion::{criterion_group, criterion_main};

#[cfg(feature = "cudnn")]
criterion_group!(benches, cudnn_impl::conv2d_benches);
#[cfg(feature = "cudnn")]
criterion_main!(benches);
