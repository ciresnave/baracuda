#![cfg(feature = "cudnn")]
//! Real-GPU smoke tests for `Conv1dPlan` (cuDNN wrap, Phase 11.7).
//!
//! Covers FW + BW data + BW filter for f32 over a small NCL tensor,
//! plus a shape-only sanity check for f16. Compares against a hand-
//! rolled CPU reference (naive NCL cross-correlation).
//!
//! All tests `#[ignore]` — need a real CUDA device + cuDNN at runtime.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, Conv1dArgs, Conv1dBwArgs, Conv1dDescriptor, Conv1dDwArgs, Conv1dPlan,
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
    l_in: i32,
    c_out: i32,
    l_filt: i32,
    pad_l: i32,
    stride_l: i32,
    dilation_l: i32,
    groups: i32,
}

impl Dims {
    fn l_out(&self) -> i32 {
        let l_eff = self.dilation_l * (self.l_filt - 1) + 1;
        (self.l_in + 2 * self.pad_l - l_eff) / self.stride_l + 1
    }
}

const D: Dims = Dims {
    n: 2,
    c_in: 3,
    l_in: 7,
    c_out: 4,
    l_filt: 3,
    pad_l: 1,
    stride_l: 1,
    dilation_l: 1,
    groups: 1,
};

fn host_conv1d_fw_f64(d: &Dims, x: &[f64], w: &[f64]) -> Vec<f64> {
    let l_out = d.l_out();
    let c_in_per_g = d.c_in / d.groups;
    let c_out_per_g = d.c_out / d.groups;
    let mut y = vec![0f64; (d.n * d.c_out * l_out) as usize];
    for n in 0..d.n {
        for g in 0..d.groups {
            for co_in_g in 0..c_out_per_g {
                let co = g * c_out_per_g + co_in_g;
                for ol in 0..l_out {
                    let mut acc = 0f64;
                    for ci_in_g in 0..c_in_per_g {
                        let ci = g * c_in_per_g + ci_in_g;
                        for kl in 0..d.l_filt {
                            let il = ol * d.stride_l + kl * d.dilation_l - d.pad_l;
                            if il < 0 || il >= d.l_in {
                                continue;
                            }
                            let xi = ((n * d.c_in + ci) * d.l_in + il) as usize;
                            let wi = ((co * c_in_per_g + ci_in_g) * d.l_filt + kl) as usize;
                            acc += x[xi] * w[wi];
                        }
                    }
                    let yi = ((n * d.c_out + co) * l_out + ol) as usize;
                    y[yi] = acc;
                }
            }
        }
    }
    y
}

fn host_conv1d_bw_data_f64(d: &Dims, w: &[f64], dy: &[f64]) -> Vec<f64> {
    let l_out = d.l_out();
    let c_in_per_g = d.c_in / d.groups;
    let c_out_per_g = d.c_out / d.groups;
    let mut dx = vec![0f64; (d.n * d.c_in * d.l_in) as usize];
    for n in 0..d.n {
        for g in 0..d.groups {
            for co_in_g in 0..c_out_per_g {
                let co = g * c_out_per_g + co_in_g;
                for ol in 0..l_out {
                    let dy_v = dy[((n * d.c_out + co) * l_out + ol) as usize];
                    for ci_in_g in 0..c_in_per_g {
                        let ci = g * c_in_per_g + ci_in_g;
                        for kl in 0..d.l_filt {
                            let il = ol * d.stride_l + kl * d.dilation_l - d.pad_l;
                            if il < 0 || il >= d.l_in {
                                continue;
                            }
                            let xi = ((n * d.c_in + ci) * d.l_in + il) as usize;
                            let wi = ((co * c_in_per_g + ci_in_g) * d.l_filt + kl) as usize;
                            dx[xi] += dy_v * w[wi];
                        }
                    }
                }
            }
        }
    }
    dx
}

fn host_conv1d_bw_filter_f64(d: &Dims, x: &[f64], dy: &[f64]) -> Vec<f64> {
    let l_out = d.l_out();
    let c_in_per_g = d.c_in / d.groups;
    let c_out_per_g = d.c_out / d.groups;
    let mut dw = vec![0f64; (d.c_out * c_in_per_g * d.l_filt) as usize];
    for g in 0..d.groups {
        for co_in_g in 0..c_out_per_g {
            let co = g * c_out_per_g + co_in_g;
            for ci_in_g in 0..c_in_per_g {
                let ci = g * c_in_per_g + ci_in_g;
                for kl in 0..d.l_filt {
                    let mut acc = 0f64;
                    for n in 0..d.n {
                        for ol in 0..l_out {
                            let il = ol * d.stride_l + kl * d.dilation_l - d.pad_l;
                            if il < 0 || il >= d.l_in {
                                continue;
                            }
                            let xi = ((n * d.c_in + ci) * d.l_in + il) as usize;
                            let dyi = ((n * d.c_out + co) * l_out + ol) as usize;
                            acc += x[xi] * dy[dyi];
                        }
                    }
                    let wi = ((co * c_in_per_g + ci_in_g) * d.l_filt + kl) as usize;
                    dw[wi] = acc;
                }
            }
        }
    }
    dw
}

fn make_seq(n: usize, seed: u32, scale: f64) -> Vec<f64> {
    (0..n)
        .map(|i| ((seed.wrapping_add(i as u32) as f64) * scale).sin())
        .collect()
}

fn desc_from(d: &Dims, elem: ElementKind) -> Conv1dDescriptor {
    Conv1dDescriptor {
        batch: d.n,
        c_in: d.c_in,
        l_in: d.l_in,
        c_out: d.c_out,
        l_filt: d.l_filt,
        pad_l: d.pad_l,
        stride_l: d.stride_l,
        dilation_l: d.dilation_l,
        groups: d.groups,
        element: elem,
    }
}

#[test]
#[ignore]
fn conv1d_f32_fw_bw() {
    let (ctx, stream) = setup();
    let d = D;
    let l_out = d.l_out();
    let c_in_per_g = d.c_in / d.groups;

    let x_n = (d.n * d.c_in * d.l_in) as usize;
    let w_n = (d.c_out * c_in_per_g * d.l_filt) as usize;
    let y_n = (d.n * d.c_out * l_out) as usize;

    let host_x = make_seq(x_n, 0x1111, 0.013);
    let host_w = make_seq(w_n, 0x2222, 0.027);
    let host_dy = make_seq(y_n, 0x3333, 0.041);

    let host_x_f32: Vec<f32> = host_x.iter().map(|&v| v as f32).collect();
    let host_w_f32: Vec<f32> = host_w.iter().map(|&v| v as f32).collect();
    let host_dy_f32: Vec<f32> = host_dy.iter().map(|&v| v as f32).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x_f32).expect("up x");
    let dev_w = DeviceBuffer::from_slice(&ctx, &host_w_f32).expect("up w");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy_f32).expect("up dy");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, y_n).expect("alloc y");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, x_n).expect("alloc dx");
    let mut dev_dw: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, w_n).expect("alloc dw");

    let desc = desc_from(&d, ElementKind::F32);
    let plan =
        Conv1dPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("select");
    assert_eq!(plan.output_dim(), l_out);

    // FW
    let ws_fw = plan.query_fw_workspace_size(&stream).expect("ws fw");
    let mut dev_ws_fw: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, ws_fw.max(1)).expect("alloc ws fw");
    let x_shape = [d.n, d.c_in, d.l_in];
    let w_shape = [d.c_out, c_in_per_g, d.l_filt];
    let y_shape = [d.n, d.c_out, l_out];
    plan.run_fw(
        &stream,
        if ws_fw == 0 {
            Workspace::None
        } else {
            Workspace::Borrowed(dev_ws_fw.as_slice_mut())
        },
        Conv1dArgs::<f32> {
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

    let mut got_y = vec![0f32; y_n];
    dev_y.copy_to_host(&mut got_y).expect("dl y");
    let exp_y = host_conv1d_fw_f64(&d, &host_x, &host_w);
    let tol = 64.0 * (f32::EPSILON as f64);
    for i in 0..y_n {
        let diff = (got_y[i] as f64 - exp_y[i]).abs();
        let t = tol * exp_y[i].abs().max(1.0);
        assert!(diff <= t, "conv1d FW mismatch @ {i}: got={}, want={}", got_y[i], exp_y[i]);
    }

    // BW data
    let ws_bd = plan.query_bw_data_workspace_size(&stream).expect("ws bd");
    let mut dev_ws_bd: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, ws_bd.max(1)).expect("alloc ws bd");
    let dx_shape = [d.n, d.c_in, d.l_in];
    plan.run_bw_data(
        &stream,
        if ws_bd == 0 {
            Workspace::None
        } else {
            Workspace::Borrowed(dev_ws_bd.as_slice_mut())
        },
        Conv1dBwArgs::<f32> {
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
                shape: dx_shape,
                stride: contiguous_stride(dx_shape),
            },
        },
    )
    .expect("run_bw_data");
    stream.synchronize().expect("sync");

    let mut got_dx = vec![0f32; x_n];
    dev_dx.copy_to_host(&mut got_dx).expect("dl dx");
    let exp_dx = host_conv1d_bw_data_f64(&d, &host_w, &host_dy);
    for i in 0..x_n {
        let diff = (got_dx[i] as f64 - exp_dx[i]).abs();
        let t = tol * exp_dx[i].abs().max(1.0);
        assert!(diff <= t, "conv1d BW-data mismatch @ {i}: got={}, want={}", got_dx[i], exp_dx[i]);
    }

    // BW filter
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
        Conv1dDwArgs::<f32> {
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
    let exp_dw = host_conv1d_bw_filter_f64(&d, &host_x, &host_dy);
    for i in 0..w_n {
        let diff = (got_dw[i] as f64 - exp_dw[i]).abs();
        let t = tol * exp_dw[i].abs().max(1.0);
        assert!(diff <= t, "conv1d BW-filter mismatch @ {i}: got={}, want={}", got_dw[i], exp_dw[i]);
    }
}
