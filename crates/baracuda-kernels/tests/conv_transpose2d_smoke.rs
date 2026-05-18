#![cfg(feature = "cudnn")]
//! Real-GPU smoke tests for `ConvTranspose2dPlan` (cuDNN wrap, Phase 11.7).
//!
//! Confirms the role-swap dispatch through `cudnnConvolutionBackwardData`
//! produces the correct output shape and a numerically reasonable result
//! against a naive CPU reference. Also confirms BW-data and BW-filter
//! dispatch (no NaN / no crash).
//!
//! All tests `#[ignore]` — need a real CUDA device + cuDNN at runtime.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ConvTranspose2dArgs, ConvTranspose2dBwArgs, ConvTranspose2dDescriptor,
    ConvTranspose2dDwArgs, ConvTranspose2dPlan, ElementKind, PlanPreference, TensorMut,
    TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// CPU reference for ConvTranspose2d forward (PyTorch convention).
///
/// `y[n,co,oh,ow] = Σ_ci,kh,kw x[n,ci,ih,iw] · w[ci,co,kh,kw]` where
/// `ih = (oh + pad_h - dilation_h*kh) / stride_h` (only the
/// integer-valued solutions contribute). Equivalently, transpose-conv
/// scatters each input cell's contribution into an output cell.
fn host_conv_transpose2d_fw_f64(
    n: i32,
    c_in: i32,
    h_in: i32,
    w_in: i32,
    c_out: i32,
    kh: i32,
    kw: i32,
    pad_h: i32,
    pad_w: i32,
    stride_h: i32,
    stride_w: i32,
    dilation_h: i32,
    dilation_w: i32,
    out_pad_h: i32,
    out_pad_w: i32,
    x: &[f64],
    w: &[f64],
) -> (Vec<f64>, i32, i32) {
    let h_out = (h_in - 1) * stride_h - 2 * pad_h + dilation_h * (kh - 1) + out_pad_h + 1;
    let w_out = (w_in - 1) * stride_w - 2 * pad_w + dilation_w * (kw - 1) + out_pad_w + 1;
    let mut y = vec![0f64; (n * c_out * h_out * w_out) as usize];
    for nn in 0..n {
        for ci in 0..c_in {
            for ih in 0..h_in {
                for iw in 0..w_in {
                    let xi = ((nn * c_in + ci) * h_in + ih) * w_in + iw;
                    let xv = x[xi as usize];
                    for co in 0..c_out {
                        for k_h in 0..kh {
                            for k_w in 0..kw {
                                let oh = ih * stride_h + k_h * dilation_h - pad_h;
                                let ow = iw * stride_w + k_w * dilation_w - pad_w;
                                if oh < 0 || oh >= h_out || ow < 0 || ow >= w_out {
                                    continue;
                                }
                                let wi = ((ci * c_out + co) * kh + k_h) * kw + k_w;
                                let yi = ((nn * c_out + co) * h_out + oh) * w_out + ow;
                                y[yi as usize] += xv * w[wi as usize];
                            }
                        }
                    }
                }
            }
        }
    }
    (y, h_out, w_out)
}

fn make_seq(n: usize, seed: u32, scale: f64) -> Vec<f64> {
    (0..n)
        .map(|i| ((seed.wrapping_add(i as u32) as f64) * scale).sin())
        .collect()
}

#[test]
#[ignore]
fn conv_transpose2d_f32_fw() {
    let (ctx, stream) = setup();
    let n = 1;
    let c_in = 2;
    let h_in = 3;
    let w_in = 3;
    let c_out = 2;
    let kh = 3;
    let kw = 3;
    let pad_h = 0;
    let pad_w = 0;
    let stride_h = 2;
    let stride_w = 2;
    let dilation_h = 1;
    let dilation_w = 1;
    let out_pad_h = 0;
    let out_pad_w = 0;

    let host_x = make_seq((n * c_in * h_in * w_in) as usize, 0x1234, 0.013);
    let host_w = make_seq((c_in * c_out * kh * kw) as usize, 0x5678, 0.027);
    let host_x_f32: Vec<f32> = host_x.iter().map(|&v| v as f32).collect();
    let host_w_f32: Vec<f32> = host_w.iter().map(|&v| v as f32).collect();

    let (exp_y, h_out, w_out) = host_conv_transpose2d_fw_f64(
        n, c_in, h_in, w_in, c_out, kh, kw, pad_h, pad_w, stride_h, stride_w, dilation_h,
        dilation_w, out_pad_h, out_pad_w, &host_x, &host_w,
    );
    let y_n = (n * c_out * h_out * w_out) as usize;

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x_f32).expect("up x");
    let dev_w = DeviceBuffer::from_slice(&ctx, &host_w_f32).expect("up w");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, y_n).expect("alloc y");

    let desc = ConvTranspose2dDescriptor {
        batch: n,
        c_in,
        h_in,
        w_in,
        c_out,
        h_filt: kh,
        w_filt: kw,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        output_pad_h: out_pad_h,
        output_pad_w: out_pad_w,
        groups: 1,
        element: ElementKind::F32,
    };
    let plan = ConvTranspose2dPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    assert_eq!(plan.output_dims(), (h_out, w_out));

    let ws = plan.query_fw_workspace_size(&stream).expect("ws fw");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws.max(1)).expect("alloc ws");
    let x_shape = [n, c_in, h_in, w_in];
    let w_shape = [c_in, c_out, kh, kw];
    let y_shape = [n, c_out, h_out, w_out];

    plan.run_fw(
        &stream,
        if ws == 0 {
            Workspace::None
        } else {
            Workspace::Borrowed(dev_ws.as_slice_mut())
        },
        ConvTranspose2dArgs::<f32> {
            x: TensorRef {
                data: dev_x.as_slice(),
                shape: x_shape,
                stride: contiguous_stride(x_shape),
            },
            w: TensorRef {
                data: dev_w.as_slice(),
                shape: w_shape,
                stride: contiguous_stride(w_shape),
            },
            y: TensorMut {
                data: dev_y.as_slice_mut(),
                shape: y_shape,
                stride: contiguous_stride(y_shape),
            },
        },
    )
    .expect("run_fw");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; y_n];
    dev_y.copy_to_host(&mut got).expect("dl y");
    let tol = 64.0 * (f32::EPSILON as f64);
    for i in 0..y_n {
        let diff = (got[i] as f64 - exp_y[i]).abs();
        let t = tol * exp_y[i].abs().max(1.0);
        assert!(
            diff <= t,
            "convT2d FW mismatch @ {i}: got={}, want={}, diff={diff}",
            got[i], exp_y[i]
        );
    }
}

#[test]
#[ignore]
fn conv_transpose2d_f32_bw_directions_run() {
    let (ctx, stream) = setup();
    let n = 1;
    let c_in = 2;
    let h_in = 3;
    let w_in = 3;
    let c_out = 2;
    let kh = 3;
    let kw = 3;
    let stride_h = 2;
    let stride_w = 2;
    let pad_h = 0;
    let pad_w = 0;

    // Output dims for these params.
    let h_out = (h_in - 1) * stride_h - 2 * pad_h + (kh - 1) + 1;
    let w_out = (w_in - 1) * stride_w - 2 * pad_w + (kw - 1) + 1;

    let x_n = (n * c_in * h_in * w_in) as usize;
    let w_n = (c_in * c_out * kh * kw) as usize;
    let y_n = (n * c_out * h_out * w_out) as usize;

    let dev_x: DeviceBuffer<f32> = DeviceBuffer::from_slice(&ctx, &vec![0f32; x_n]).expect("x");
    let dev_w: DeviceBuffer<f32> = DeviceBuffer::from_slice(&ctx, &vec![1f32; w_n]).expect("w");
    let dev_dy: DeviceBuffer<f32> = DeviceBuffer::from_slice(&ctx, &vec![1f32; y_n]).expect("dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, x_n).expect("alloc dx");
    let mut dev_dw: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, w_n).expect("alloc dw");

    let desc = ConvTranspose2dDescriptor {
        batch: n,
        c_in,
        h_in,
        w_in,
        c_out,
        h_filt: kh,
        w_filt: kw,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h: 1,
        dilation_w: 1,
        output_pad_h: 0,
        output_pad_w: 0,
        groups: 1,
        element: ElementKind::F32,
    };
    let plan = ConvTranspose2dPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");

    let x_shape = [n, c_in, h_in, w_in];
    let w_shape = [c_in, c_out, kh, kw];
    let y_shape = [n, c_out, h_out, w_out];

    // BW data
    let ws_bd = plan.query_bw_data_workspace_size(&stream).expect("ws bd");
    let mut dev_ws_bd: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, ws_bd.max(1)).expect("alloc ws bd");
    plan.run_bw_data(
        &stream,
        if ws_bd == 0 {
            Workspace::None
        } else {
            Workspace::Borrowed(dev_ws_bd.as_slice_mut())
        },
        ConvTranspose2dBwArgs::<f32> {
            w: TensorRef {
                data: dev_w.as_slice(),
                shape: w_shape,
                stride: contiguous_stride(w_shape),
            },
            dy: TensorRef {
                data: dev_dy.as_slice(),
                shape: y_shape,
                stride: contiguous_stride(y_shape),
            },
            dx: TensorMut {
                data: dev_dx.as_slice_mut(),
                shape: x_shape,
                stride: contiguous_stride(x_shape),
            },
        },
    )
    .expect("run_bw_data");
    stream.synchronize().expect("sync");

    // dy=1, w=1 → dx is a non-trivial positive sum (= count of contributing
    // taps per input cell). Just check no NaN, no error.
    let mut got_dx = vec![0f32; x_n];
    dev_dx.copy_to_host(&mut got_dx).expect("dl dx");
    for &v in &got_dx {
        assert!(v.is_finite(), "convT2d BW-data produced non-finite value");
    }

    // BW filter — x=0, dy=1 → dw = 0.
    let ws_bf = plan.query_bw_filter_workspace_size(&stream).expect("ws bf");
    let mut dev_ws_bf: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, ws_bf.max(1)).expect("alloc ws bf");
    plan.run_dw(
        &stream,
        if ws_bf == 0 {
            Workspace::None
        } else {
            Workspace::Borrowed(dev_ws_bf.as_slice_mut())
        },
        ConvTranspose2dDwArgs::<f32> {
            x: TensorRef {
                data: dev_x.as_slice(),
                shape: x_shape,
                stride: contiguous_stride(x_shape),
            },
            dy: TensorRef {
                data: dev_dy.as_slice(),
                shape: y_shape,
                stride: contiguous_stride(y_shape),
            },
            dw: TensorMut {
                data: dev_dw.as_slice_mut(),
                shape: w_shape,
                stride: contiguous_stride(w_shape),
            },
        },
    )
    .expect("run_dw");
    stream.synchronize().expect("sync");

    let mut got_dw = vec![0f32; w_n];
    dev_dw.copy_to_host(&mut got_dw).expect("dl dw");
    for (i, &v) in got_dw.iter().enumerate() {
        assert!(v.abs() < 1e-6, "convT2d BW-filter: dw[{i}] = {v}, expected 0");
    }
}
