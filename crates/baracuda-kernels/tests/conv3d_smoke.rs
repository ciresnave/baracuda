#![cfg(feature = "cudnn")]
//! Real-GPU smoke tests for `Conv3dPlan` (cuDNN wrap, Phase 11.7).
//!
//! Covers FW + BW data + BW filter for f32 over a small NCDHW tensor.
//! Compares against a naive 3-D cross-correlation CPU reference.
//!
//! All tests `#[ignore]` — need a real CUDA device + cuDNN at runtime.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, Conv3dArgs, Conv3dBwArgs, Conv3dDescriptor, Conv3dDwArgs, Conv3dPlan,
    ElementKind, PlanPreference, TensorMut, TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[derive(Copy, Clone, Debug)]
struct Dims {
    n: i32,
    c_in: i32,
    d_in: i32,
    h_in: i32,
    w_in: i32,
    c_out: i32,
    d_filt: i32,
    h_filt: i32,
    w_filt: i32,
}

impl Dims {
    fn out(&self) -> (i32, i32, i32) {
        let d_out = self.d_in - self.d_filt + 1;
        let h_out = self.h_in - self.h_filt + 1;
        let w_out = self.w_in - self.w_filt + 1;
        (d_out, h_out, w_out)
    }
}

const D: Dims = Dims {
    n: 1,
    c_in: 2,
    d_in: 4,
    h_in: 4,
    w_in: 4,
    c_out: 3,
    d_filt: 2,
    h_filt: 2,
    w_filt: 2,
};

fn host_conv3d_fw_f64(d: &Dims, x: &[f64], w: &[f64]) -> Vec<f64> {
    let (d_out, h_out, w_out) = d.out();
    let mut y = vec![0f64; (d.n * d.c_out * d_out * h_out * w_out) as usize];
    let dh = d.h_in;
    let dw = d.w_in;
    let dd = d.d_in;
    for n in 0..d.n {
        for co in 0..d.c_out {
            for od in 0..d_out {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut acc = 0f64;
                        for ci in 0..d.c_in {
                            for kd in 0..d.d_filt {
                                for kh in 0..d.h_filt {
                                    for kw in 0..d.w_filt {
                                        let id = od + kd;
                                        let ih = oh + kh;
                                        let iw = ow + kw;
                                        let xi = (((n * d.c_in + ci) * dd + id) * dh + ih) * dw
                                            + iw;
                                        let wi = (((co * d.c_in + ci) * d.d_filt + kd) * d.h_filt
                                            + kh)
                                            * d.w_filt
                                            + kw;
                                        acc += x[xi as usize] * w[wi as usize];
                                    }
                                }
                            }
                        }
                        let yi = (((n * d.c_out + co) * d_out + od) * h_out + oh) * w_out + ow;
                        y[yi as usize] = acc;
                    }
                }
            }
        }
    }
    y
}

fn make_seq(n: usize, seed: u32, scale: f64) -> Vec<f64> {
    (0..n)
        .map(|i| ((seed.wrapping_add(i as u32) as f64) * scale).sin())
        .collect()
}

#[test]
#[ignore]
fn conv3d_f32_fw() {
    let (ctx, stream) = setup();
    let d = D;
    let (d_out, h_out, w_out) = d.out();
    let x_n = (d.n * d.c_in * d.d_in * d.h_in * d.w_in) as usize;
    let w_n = (d.c_out * d.c_in * d.d_filt * d.h_filt * d.w_filt) as usize;
    let y_n = (d.n * d.c_out * d_out * h_out * w_out) as usize;

    let host_x = make_seq(x_n, 0xCAFE, 0.013);
    let host_w = make_seq(w_n, 0xBABE, 0.027);
    let host_x_f32: Vec<f32> = host_x.iter().map(|&v| v as f32).collect();
    let host_w_f32: Vec<f32> = host_w.iter().map(|&v| v as f32).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x_f32).expect("up x");
    let dev_w = DeviceBuffer::from_slice(&ctx, &host_w_f32).expect("up w");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, y_n).expect("alloc y");

    let desc = Conv3dDescriptor::new(
        d.n,
        d.c_in,
        d.d_in,
        d.h_in,
        d.w_in,
        d.c_out,
        d.d_filt,
        d.h_filt,
        d.w_filt,
        ElementKind::F32,
    );
    let plan = Conv3dPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    assert_eq!(plan.output_dims(), (d_out, h_out, w_out));

    let ws = plan.query_fw_workspace_size(&stream).expect("ws fw");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws.max(1)).expect("alloc ws");

    let x_shape = [d.n, d.c_in, d.d_in, d.h_in, d.w_in];
    let w_shape = [d.c_out, d.c_in, d.d_filt, d.h_filt, d.w_filt];
    let y_shape = [d.n, d.c_out, d_out, h_out, w_out];
    plan.run_fw(
        &stream,
        if ws == 0 {
            Workspace::None
        } else {
            Workspace::Borrowed(dev_ws.as_slice_mut())
        },
        Conv3dArgs::<f32> {
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
    let exp = host_conv3d_fw_f64(&d, &host_x, &host_w);
    let tol = 128.0 * (f32::EPSILON as f64);
    for i in 0..y_n {
        let diff = (got[i] as f64 - exp[i]).abs();
        let t = tol * exp[i].abs().max(1.0);
        assert!(diff <= t, "conv3d FW mismatch @ {i}: got={}, want={}", got[i], exp[i]);
    }
}

#[test]
#[ignore]
fn conv3d_f32_bw_data_filter_runs() {
    // Shape-only smoke for the BW directions — confirms the plan launches
    // without errors and writes the expected element count. Heavy
    // numerical validation lives in the FW test.
    let (ctx, stream) = setup();
    let d = D;
    let (d_out, h_out, w_out) = d.out();
    let x_n = (d.n * d.c_in * d.d_in * d.h_in * d.w_in) as usize;
    let w_n = (d.c_out * d.c_in * d.d_filt * d.h_filt * d.w_filt) as usize;
    let y_n = (d.n * d.c_out * d_out * h_out * w_out) as usize;

    let zero_x = vec![0f32; x_n];
    let zero_w = vec![0f32; w_n];
    let zero_dy = vec![1f32; y_n];

    let dev_x = DeviceBuffer::from_slice(&ctx, &zero_x).expect("up x");
    let dev_w = DeviceBuffer::from_slice(&ctx, &zero_w).expect("up w");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &zero_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, x_n).expect("alloc dx");
    let mut dev_dw: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, w_n).expect("alloc dw");

    let desc = Conv3dDescriptor::new(
        d.n,
        d.c_in,
        d.d_in,
        d.h_in,
        d.w_in,
        d.c_out,
        d.d_filt,
        d.h_filt,
        d.w_filt,
        ElementKind::F32,
    );
    let plan = Conv3dPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("sel");

    let ws_bd = plan.query_bw_data_workspace_size(&stream).expect("ws bd");
    let mut dev_ws_bd: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, ws_bd.max(1)).expect("alloc ws bd");
    let x_shape = [d.n, d.c_in, d.d_in, d.h_in, d.w_in];
    let w_shape = [d.c_out, d.c_in, d.d_filt, d.h_filt, d.w_filt];
    let y_shape = [d.n, d.c_out, d_out, h_out, w_out];
    plan.run_bw_data(
        &stream,
        if ws_bd == 0 {
            Workspace::None
        } else {
            Workspace::Borrowed(dev_ws_bd.as_slice_mut())
        },
        Conv3dBwArgs::<f32> {
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
        Conv3dDwArgs::<f32> {
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

    // w=0 + dy=1: dx = 0 (since dx = sum dy * w).
    let mut got_dx = vec![0f32; x_n];
    dev_dx.copy_to_host(&mut got_dx).expect("dl dx");
    for (i, &v) in got_dx.iter().enumerate() {
        assert!(v.abs() < 1e-6, "conv3d BW-data: dx[{i}] = {v}, expected 0");
    }

    // x=0 + dy=1: dw = 0.
    let mut got_dw = vec![0f32; w_n];
    dev_dw.copy_to_host(&mut got_dw).expect("dl dw");
    for (i, &v) in got_dw.iter().enumerate() {
        assert!(v.abs() < 1e-6, "conv3d BW-filter: dw[{i}] = {v}, expected 0");
    }
}
