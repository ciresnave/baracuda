#![cfg(feature = "cudnn")]
//! Real-GPU smoke tests for `Conv2dPlan` (cuDNN wrap).
//!
//! Phase 7 Milestone 7.1 trailblazer. Covers FW + BW data + BW filter
//! across the four floating-point dtypes (f32, f64, f16, bf16), plus
//! padding / stride sanity for f32.
//!
//! All tests are `#[ignore]` — they need a real CUDA device + cuDNN at
//! runtime.
//!
//! ## CPU reference
//!
//! The convention is **cross-correlation** (kernel applied directly, no
//! flip) — matching PyTorch's `nn.Conv2d`. Output spatial extents follow
//! `H_out = floor((H_in + 2·pad - dilation·(H_filt - 1) - 1) / stride) + 1`.
//! For BW data the reference is the naive transposed-correlation
//! (`dx[n,c,i,j] = Σ_co,kh,kw dy[n,co,i',j'] · w[co,c,kh,kw]` where the
//! contributing `(i', j')` are the FW output cells whose receptive field
//! overlaps `(i, j)`). For BW filter the reference is naive
//! correlation over the upstream gradient: `dw[co,c,kh,kw] = Σ_n,oh,ow
//! dy[n,co,oh,ow] · x[n,c,oh·sh + kh·dh - ph, ow·sw + kw·dw - pw]`,
//! treating out-of-bound `x` cells as zero.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, Conv2dArgs, Conv2dBwArgs, Conv2dDescriptor, Conv2dDwArgs, Conv2dPlan,
    ElementKind, PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[derive(Copy, Clone, Debug)]
struct ConvDims {
    n: i32,
    c_in: i32,
    h_in: i32,
    w_in: i32,
    c_out: i32,
    h_filt: i32,
    w_filt: i32,
    pad_h: i32,
    pad_w: i32,
    stride_h: i32,
    stride_w: i32,
    dilation_h: i32,
    dilation_w: i32,
}

impl ConvDims {
    fn output_dims(&self) -> (i32, i32) {
        let h_eff = self.dilation_h * (self.h_filt - 1) + 1;
        let w_eff = self.dilation_w * (self.w_filt - 1) + 1;
        (
            (self.h_in + 2 * self.pad_h - h_eff) / self.stride_h + 1,
            (self.w_in + 2 * self.pad_w - w_eff) / self.stride_w + 1,
        )
    }
}

/// Naive NCHW cross-correlation forward (f64 reference).
fn host_conv2d_fw_f64(d: &ConvDims, x: &[f64], w: &[f64]) -> Vec<f64> {
    let (h_out, w_out) = d.output_dims();
    let mut y = vec![0f64; (d.n * d.c_out * h_out * w_out) as usize];
    let x_w = d.w_in as i64;
    let x_h = d.h_in as i64;
    for n in 0..d.n {
        for co in 0..d.c_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut acc = 0f64;
                    for ci in 0..d.c_in {
                        for kh in 0..d.h_filt {
                            for kw in 0..d.w_filt {
                                let ih = oh as i64 * d.stride_h as i64 + kh as i64 * d.dilation_h as i64
                                    - d.pad_h as i64;
                                let iw = ow as i64 * d.stride_w as i64 + kw as i64 * d.dilation_w as i64
                                    - d.pad_w as i64;
                                if ih < 0 || ih >= x_h || iw < 0 || iw >= x_w {
                                    continue;
                                }
                                let xi = ((n * d.c_in + ci) as i64 * x_h * x_w
                                    + ih * x_w
                                    + iw) as usize;
                                let wi = (((co * d.c_in + ci) * d.h_filt + kh) * d.w_filt + kw)
                                    as usize;
                                acc += x[xi] * w[wi];
                            }
                        }
                    }
                    let yi = (((n * d.c_out + co) * h_out + oh) * w_out + ow) as usize;
                    y[yi] = acc;
                }
            }
        }
    }
    y
}

/// Naive NCHW conv2d backward-data (f64 reference).
fn host_conv2d_bw_data_f64(d: &ConvDims, w: &[f64], dy: &[f64]) -> Vec<f64> {
    let (h_out, w_out) = d.output_dims();
    let mut dx = vec![0f64; (d.n * d.c_in * d.h_in * d.w_in) as usize];
    let x_w = d.w_in as i64;
    let x_h = d.h_in as i64;
    for n in 0..d.n {
        for co in 0..d.c_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let dy_v = dy[(((n * d.c_out + co) * h_out + oh) * w_out + ow) as usize];
                    for ci in 0..d.c_in {
                        for kh in 0..d.h_filt {
                            for kw in 0..d.w_filt {
                                let ih = oh as i64 * d.stride_h as i64 + kh as i64 * d.dilation_h as i64
                                    - d.pad_h as i64;
                                let iw = ow as i64 * d.stride_w as i64 + kw as i64 * d.dilation_w as i64
                                    - d.pad_w as i64;
                                if ih < 0 || ih >= x_h || iw < 0 || iw >= x_w {
                                    continue;
                                }
                                let xi = ((n * d.c_in + ci) as i64 * x_h * x_w
                                    + ih * x_w
                                    + iw) as usize;
                                let wi = (((co * d.c_in + ci) * d.h_filt + kh) * d.w_filt + kw)
                                    as usize;
                                dx[xi] += dy_v * w[wi];
                            }
                        }
                    }
                }
            }
        }
    }
    dx
}

/// Naive NCHW conv2d backward-filter (f64 reference).
fn host_conv2d_bw_filter_f64(d: &ConvDims, x: &[f64], dy: &[f64]) -> Vec<f64> {
    let (h_out, w_out) = d.output_dims();
    let mut dw = vec![0f64; (d.c_out * d.c_in * d.h_filt * d.w_filt) as usize];
    let x_w = d.w_in as i64;
    let x_h = d.h_in as i64;
    for co in 0..d.c_out {
        for ci in 0..d.c_in {
            for kh in 0..d.h_filt {
                for kw in 0..d.w_filt {
                    let mut acc = 0f64;
                    for n in 0..d.n {
                        for oh in 0..h_out {
                            for ow in 0..w_out {
                                let ih = oh as i64 * d.stride_h as i64 + kh as i64 * d.dilation_h as i64
                                    - d.pad_h as i64;
                                let iw = ow as i64 * d.stride_w as i64 + kw as i64 * d.dilation_w as i64
                                    - d.pad_w as i64;
                                if ih < 0 || ih >= x_h || iw < 0 || iw >= x_w {
                                    continue;
                                }
                                let xi = ((n * d.c_in + ci) as i64 * x_h * x_w
                                    + ih * x_w
                                    + iw) as usize;
                                let dyi =
                                    (((n * d.c_out + co) * h_out + oh) * w_out + ow) as usize;
                                acc += x[xi] * dy[dyi];
                            }
                        }
                    }
                    let wi = (((co * d.c_in + ci) * d.h_filt + kh) * d.w_filt + kw) as usize;
                    dw[wi] = acc;
                }
            }
        }
    }
    dw
}

/// Build deterministic input tensors for the trailblazer fixture.
fn make_x_f64(d: &ConvDims, seed: u32) -> Vec<f64> {
    let n = (d.n * d.c_in * d.h_in * d.w_in) as usize;
    (0..n)
        .map(|i| ((seed.wrapping_add(i as u32) as f64) * 0.013).sin())
        .collect()
}
fn make_w_f64(d: &ConvDims, seed: u32) -> Vec<f64> {
    let n = (d.c_out * d.c_in * d.h_filt * d.w_filt) as usize;
    (0..n)
        .map(|i| ((seed.wrapping_add(i as u32) as f64) * 0.027).cos() * 0.5)
        .collect()
}
fn make_dy_f64(d: &ConvDims, seed: u32) -> Vec<f64> {
    let (h_out, w_out) = d.output_dims();
    let n = (d.n * d.c_out * h_out * w_out) as usize;
    (0..n)
        .map(|i| ((seed.wrapping_add(i as u32) as f64) * 0.041).sin() * 0.3)
        .collect()
}

const TRAILBLAZER: ConvDims = ConvDims {
    n: 1,
    c_in: 3,
    h_in: 5,
    w_in: 5,
    c_out: 2,
    h_filt: 3,
    w_filt: 3,
    pad_h: 0,
    pad_w: 0,
    stride_h: 1,
    stride_w: 1,
    dilation_h: 1,
    dilation_w: 1,
};

/// Run a single FW pass and verify against the f64 reference within
/// the requested absolute tolerance.
fn run_fw_and_check<T, F, G>(
    ctx: &Context,
    stream: &Stream,
    d: &ConvDims,
    elem: ElementKind,
    host_x_f64: &[f64],
    host_w_f64: &[f64],
    tol: f64,
    to_t: F,
    from_t: G,
) where
    T: baracuda_kernels::Element + 'static,
    F: Fn(f64) -> T,
    G: Fn(T) -> f64,
{
    let host_x: Vec<T> = host_x_f64.iter().map(|&v| to_t(v)).collect();
    let host_w: Vec<T> = host_w_f64.iter().map(|&v| to_t(v)).collect();
    let (h_out, w_out) = d.output_dims();
    let y_numel = (d.n * d.c_out * h_out * w_out) as usize;

    let dev_x = DeviceBuffer::from_slice(ctx, &host_x).expect("up x");
    let dev_w = DeviceBuffer::from_slice(ctx, &host_w).expect("up w");
    let mut dev_y: DeviceBuffer<T> = DeviceBuffer::zeros(ctx, y_numel).expect("alloc y");

    let desc = Conv2dDescriptor::new(
        d.n, d.c_in, d.h_in, d.w_in, d.c_out, d.h_filt, d.w_filt, elem,
    )
    .with_padding(d.pad_h, d.pad_w)
    .with_stride(d.stride_h, d.stride_w)
    .with_dilation(d.dilation_h, d.dilation_w);
    let plan = Conv2dPlan::<T>::select(stream, &desc, PlanPreference::default())
        .expect("select Conv2dPlan");
    assert_eq!(plan.output_dims(), (h_out, w_out));

    let ws_bytes = plan.query_fw_workspace_size(stream).expect("ws query");
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(ctx, ws_bytes.max(1)).expect("alloc ws");

    let x_shape = [d.n, d.c_in, d.h_in, d.w_in];
    let w_shape = [d.c_out, d.c_in, d.h_filt, d.w_filt];
    let y_shape = [d.n, d.c_out, h_out, w_out];
    let workspace = if ws_bytes == 0 {
        Workspace::None
    } else {
        Workspace::Borrowed(dev_ws.as_slice_mut())
    };
    plan.run_fw(
        stream,
        workspace,
        Conv2dArgs::<T> {
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

    // Use a dummy initial value the wrapper type can construct from f32.
    let mut got: Vec<T> = (0..y_numel).map(|_| to_t(0.0)).collect();
    dev_y.copy_to_host(&mut got).expect("dl y");

    let expected = host_conv2d_fw_f64(d, host_x_f64, host_w_f64);
    for i in 0..y_numel {
        let g = from_t(got[i]);
        let diff = (g - expected[i]).abs();
        let t = tol * expected[i].abs().max(1.0);
        assert!(
            diff <= t,
            "conv2d FW {elem:?} mismatch @ {i}: got={g}, want={}, diff={diff}, tol={t}",
            expected[i],
        );
    }
}

#[test]
#[ignore]
fn conv2d_f32_no_padding() {
    let (ctx, stream) = setup();
    let d = TRAILBLAZER;
    let host_x = make_x_f64(&d, 0xA11C_E001);
    let host_w = make_w_f64(&d, 0xBEEF_0001);
    run_fw_and_check::<f32, _, _>(
        &ctx,
        &stream,
        &d,
        ElementKind::F32,
        &host_x,
        &host_w,
        32.0 * (f32::EPSILON as f64),
        |v| v as f32,
        |v| v as f64,
    );
}

#[test]
#[ignore]
fn conv2d_f64_no_padding() {
    let (ctx, stream) = setup();
    let d = TRAILBLAZER;
    let host_x = make_x_f64(&d, 0xC0DE_F00D);
    let host_w = make_w_f64(&d, 0xDEAD_BEEF);
    run_fw_and_check::<f64, _, _>(
        &ctx,
        &stream,
        &d,
        ElementKind::F64,
        &host_x,
        &host_w,
        32.0 * f64::EPSILON,
        |v| v,
        |v| v,
    );
}

#[test]
#[ignore]
fn conv2d_f16_no_padding() {
    let (ctx, stream) = setup();
    let d = TRAILBLAZER;
    let host_x = make_x_f64(&d, 0xF16F_16F1);
    let host_w = make_w_f64(&d, 0xF161_F161);
    run_fw_and_check::<f16, _, _>(
        &ctx,
        &stream,
        &d,
        ElementKind::F16,
        &host_x,
        &host_w,
        1.0e-3,
        |v| f16::from_f32(v as f32),
        |v| v.to_f32() as f64,
    );
}

#[test]
#[ignore]
fn conv2d_bf16_no_padding() {
    let (ctx, stream) = setup();
    let d = TRAILBLAZER;
    let host_x = make_x_f64(&d, 0xB16B_F16B);
    let host_w = make_w_f64(&d, 0xBF16_BF16);
    run_fw_and_check::<bf16, _, _>(
        &ctx,
        &stream,
        &d,
        ElementKind::Bf16,
        &host_x,
        &host_w,
        5.0e-3,
        |v| bf16::from_f32(v as f32),
        |v| v.to_f32() as f64,
    );
}

#[test]
#[ignore]
fn conv2d_f32_with_padding() {
    let (ctx, stream) = setup();
    let d = ConvDims {
        pad_h: 1,
        pad_w: 1,
        ..TRAILBLAZER
    };
    // With pad=1 over a 5x5 input + 3x3 filter + stride=1, output is 5x5.
    let (h_out, w_out) = d.output_dims();
    assert_eq!((h_out, w_out), (5, 5));

    let host_x = make_x_f64(&d, 0x1234_5678);
    let host_w = make_w_f64(&d, 0x9ABC_DEF0);
    run_fw_and_check::<f32, _, _>(
        &ctx,
        &stream,
        &d,
        ElementKind::F32,
        &host_x,
        &host_w,
        32.0 * (f32::EPSILON as f64),
        |v| v as f32,
        |v| v as f64,
    );
}

#[test]
#[ignore]
fn conv2d_f32_stride2() {
    let (ctx, stream) = setup();
    // 7x7 input, 3x3 filter, stride 2, pad 0 → output 3x3.
    let d = ConvDims {
        h_in: 7,
        w_in: 7,
        stride_h: 2,
        stride_w: 2,
        ..TRAILBLAZER
    };
    let (h_out, w_out) = d.output_dims();
    assert_eq!((h_out, w_out), (3, 3));

    let host_x = make_x_f64(&d, 0x5555_AAAA);
    let host_w = make_w_f64(&d, 0xAAAA_5555);
    run_fw_and_check::<f32, _, _>(
        &ctx,
        &stream,
        &d,
        ElementKind::F32,
        &host_x,
        &host_w,
        32.0 * (f32::EPSILON as f64),
        |v| v as f32,
        |v| v as f64,
    );
}

#[test]
#[ignore]
fn conv2d_f32_bw_data() {
    let (ctx, stream) = setup();
    let d = TRAILBLAZER;
    let host_w = make_w_f64(&d, 0xBEEF_0002);
    let host_dy = make_dy_f64(&d, 0xCAFE_BABE);

    let host_w_f32: Vec<f32> = host_w.iter().map(|&v| v as f32).collect();
    let host_dy_f32: Vec<f32> = host_dy.iter().map(|&v| v as f32).collect();

    let (h_out, w_out) = d.output_dims();
    let dx_numel = (d.n * d.c_in * d.h_in * d.w_in) as usize;

    let dev_w = DeviceBuffer::from_slice(&ctx, &host_w_f32).expect("up w");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy_f32).expect("up dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc dx");

    let desc = Conv2dDescriptor::new(
        d.n,
        d.c_in,
        d.h_in,
        d.w_in,
        d.c_out,
        d.h_filt,
        d.w_filt,
        ElementKind::F32,
    )
    .with_padding(d.pad_h, d.pad_w)
    .with_stride(d.stride_h, d.stride_w)
    .with_dilation(d.dilation_h, d.dilation_w);
    let plan = Conv2dPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let ws_bytes = plan
        .query_bw_data_workspace_size(&stream)
        .expect("ws bw data");
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, ws_bytes.max(1)).expect("alloc ws");

    let w_shape = [d.c_out, d.c_in, d.h_filt, d.w_filt];
    let dy_shape = [d.n, d.c_out, h_out, w_out];
    let dx_shape = [d.n, d.c_in, d.h_in, d.w_in];
    let workspace = if ws_bytes == 0 {
        Workspace::None
    } else {
        Workspace::Borrowed(dev_ws.as_slice_mut())
    };
    plan.run_bw_data(
        &stream,
        workspace,
        Conv2dBwArgs::<f32> {
            w: TensorRef {
                data: dev_w.as_slice(),
                shape: w_shape,
                stride: contiguous_stride(w_shape),
            },
            dy: TensorRef {
                data: dev_dy.as_slice(),
                shape: dy_shape,
                stride: contiguous_stride(dy_shape),
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

    let mut got = vec![0f32; dx_numel];
    dev_dx.copy_to_host(&mut got).expect("dl dx");

    let expected = host_conv2d_bw_data_f64(&d, &host_w, &host_dy);
    let tol = 32.0 * (f32::EPSILON as f64);
    for i in 0..dx_numel {
        let g = got[i] as f64;
        let diff = (g - expected[i]).abs();
        let t = tol * expected[i].abs().max(1.0);
        assert!(
            diff <= t,
            "conv2d BW-data f32 mismatch @ {i}: got={g}, want={}, diff={diff}",
            expected[i]
        );
    }
}

#[test]
#[ignore]
fn conv2d_f32_bw_filter() {
    let (ctx, stream) = setup();
    let d = TRAILBLAZER;
    let host_x = make_x_f64(&d, 0xABCD_1234);
    let host_dy = make_dy_f64(&d, 0x4321_DCBA);

    let host_x_f32: Vec<f32> = host_x.iter().map(|&v| v as f32).collect();
    let host_dy_f32: Vec<f32> = host_dy.iter().map(|&v| v as f32).collect();

    let (h_out, w_out) = d.output_dims();
    let dw_numel = (d.c_out * d.c_in * d.h_filt * d.w_filt) as usize;

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x_f32).expect("up x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy_f32).expect("up dy");
    let mut dev_dw: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, dw_numel).expect("alloc dw");

    let desc = Conv2dDescriptor::new(
        d.n,
        d.c_in,
        d.h_in,
        d.w_in,
        d.c_out,
        d.h_filt,
        d.w_filt,
        ElementKind::F32,
    )
    .with_padding(d.pad_h, d.pad_w)
    .with_stride(d.stride_h, d.stride_w)
    .with_dilation(d.dilation_h, d.dilation_w);
    let plan = Conv2dPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let ws_bytes = plan
        .query_bw_filter_workspace_size(&stream)
        .expect("ws bw filter");
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, ws_bytes.max(1)).expect("alloc ws");

    let x_shape = [d.n, d.c_in, d.h_in, d.w_in];
    let dy_shape = [d.n, d.c_out, h_out, w_out];
    let dw_shape = [d.c_out, d.c_in, d.h_filt, d.w_filt];
    let workspace = if ws_bytes == 0 {
        Workspace::None
    } else {
        Workspace::Borrowed(dev_ws.as_slice_mut())
    };
    plan.run_dw(
        &stream,
        workspace,
        Conv2dDwArgs::<f32> {
            x: TensorRef {
                data: dev_x.as_slice(),
                shape: x_shape,
                stride: contiguous_stride(x_shape),
            },
            dy: TensorRef {
                data: dev_dy.as_slice(),
                shape: dy_shape,
                stride: contiguous_stride(dy_shape),
            },
            dw: TensorMut {
                data: dev_dw.as_slice_mut(),
                shape: dw_shape,
                stride: contiguous_stride(dw_shape),
            },
        },
    )
    .expect("run_dw");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; dw_numel];
    dev_dw.copy_to_host(&mut got).expect("dl dw");

    let expected = host_conv2d_bw_filter_f64(&d, &host_x, &host_dy);
    let tol = 32.0 * (f32::EPSILON as f64);
    for i in 0..dw_numel {
        let g = got[i] as f64;
        let diff = (g - expected[i]).abs();
        let t = tol * expected[i].abs().max(1.0);
        assert!(
            diff <= t,
            "conv2d BW-filter f32 mismatch @ {i}: got={g}, want={}, diff={diff}",
            expected[i]
        );
    }
}
