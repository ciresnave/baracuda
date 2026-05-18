#![cfg(feature = "cudnn")]
//! Real-GPU smoke test for `Conv2dPlan` with `groups == c_in == c_out`
//! (depthwise convolution).
//!
//! Verifies that the Conv2dPlan handles `groups != 1` correctly via
//! cuDNN's `cudnnSetConvolutionGroupCount` — each input channel is
//! convolved with its own filter independently. Compares against a
//! naive depthwise CPU reference.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, Conv2dArgs, Conv2dDescriptor, Conv2dPlan, ElementKind, PlanPreference,
    TensorMut, TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Naive depthwise NCHW cross-correlation (one filter per channel,
/// groups == c_in == c_out).
fn host_depthwise_conv2d_fw_f64(
    n: i32,
    c: i32,
    h_in: i32,
    w_in: i32,
    kh: i32,
    kw: i32,
    pad_h: i32,
    pad_w: i32,
    x: &[f64],
    w: &[f64],
) -> (Vec<f64>, i32, i32) {
    let h_out = h_in + 2 * pad_h - (kh - 1);
    let w_out = w_in + 2 * pad_w - (kw - 1);
    let mut y = vec![0f64; (n * c * h_out * w_out) as usize];
    for nn in 0..n {
        for cc in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut acc = 0f64;
                    for k_h in 0..kh {
                        for k_w in 0..kw {
                            let ih = oh + k_h - pad_h;
                            let iw = ow + k_w - pad_w;
                            if ih < 0 || ih >= h_in || iw < 0 || iw >= w_in {
                                continue;
                            }
                            // Filter shape under groups=c: [c_out, c_in/groups=1, kh, kw].
                            // The "single in-channel-per-group" axis collapses, so
                            // the filter for channel cc lives at index
                            // ((cc * 1) * kh + k_h) * kw + k_w.
                            let xi = ((nn * c + cc) * h_in + ih) * w_in + iw;
                            let wi = (cc * kh + k_h) * kw + k_w;
                            acc += x[xi as usize] * w[wi as usize];
                        }
                    }
                    let yi = ((nn * c + cc) * h_out + oh) * w_out + ow;
                    y[yi as usize] = acc;
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
fn conv2d_depthwise_f32() {
    let (ctx, stream) = setup();
    let n = 2;
    let c = 4; // c_in == c_out == groups → depthwise.
    let h_in = 6;
    let w_in = 6;
    let kh = 3;
    let kw = 3;
    let pad_h = 1;
    let pad_w = 1;

    // For depthwise, filter shape is [c, 1, kh, kw]: c_in/groups = 1.
    let x_n = (n * c * h_in * w_in) as usize;
    let w_n = (c * 1 * kh * kw) as usize;

    let host_x = make_seq(x_n, 0xDEAD, 0.013);
    let host_w = make_seq(w_n, 0xBEEF, 0.027);

    let (exp_y, h_out, w_out) =
        host_depthwise_conv2d_fw_f64(n, c, h_in, w_in, kh, kw, pad_h, pad_w, &host_x, &host_w);
    let y_n = (n * c * h_out * w_out) as usize;

    let host_x_f32: Vec<f32> = host_x.iter().map(|&v| v as f32).collect();
    let host_w_f32: Vec<f32> = host_w.iter().map(|&v| v as f32).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x_f32).expect("up x");
    let dev_w = DeviceBuffer::from_slice(&ctx, &host_w_f32).expect("up w");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, y_n).expect("alloc y");

    let desc = Conv2dDescriptor {
        batch: n,
        c_in: c,
        h_in,
        w_in,
        c_out: c,
        h_filt: kh,
        w_filt: kw,
        pad_h,
        pad_w,
        stride_h: 1,
        stride_w: 1,
        dilation_h: 1,
        dilation_w: 1,
        groups: c, // depthwise: one filter per input channel.
        element: ElementKind::F32,
    };
    let plan =
        Conv2dPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("select");
    assert_eq!(plan.output_dims(), (h_out, w_out));

    let ws = plan.query_fw_workspace_size(&stream).expect("ws fw");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws.max(1)).expect("alloc ws");

    let x_shape = [n, c, h_in, w_in];
    // Depthwise filter: c_in/groups = 1.
    let w_shape = [c, 1, kh, kw];
    let y_shape = [n, c, h_out, w_out];

    plan.run_fw(
        &stream,
        if ws == 0 {
            Workspace::None
        } else {
            Workspace::Borrowed(dev_ws.as_slice_mut())
        },
        Conv2dArgs::<f32> {
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
            "depthwise conv2d FW mismatch @ {i}: got={}, want={}, diff={diff}",
            got[i], exp_y[i]
        );
    }
}
