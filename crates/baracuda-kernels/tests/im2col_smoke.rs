#![cfg(feature = "cudnn")]
//! Real-GPU smoke tests for `Im2ColPlan` / `Im2Col1dPlan` /
//! `Col2Im1dPlan` (Phase 19.3 — bespoke kernels).
//!
//! Covers basic stride/pad/dilation configurations against host
//! references plus the col2im round-trip identity.
//!
//! All tests are `#[ignore]` by default (require real CUDA device).

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, Col2Im1dArgs, Col2Im1dDescriptor, Col2Im1dPlan, ElementKind, Im2Col1dArgs,
    Im2Col1dDescriptor, Im2Col1dPlan, Im2ColArgs, Im2ColDescriptor, Im2ColPlan, PlanPreference,
    TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// =============================================================================
// Host references
// =============================================================================

/// 2-D im2col reference. PyTorch `nn.functional.unfold` equivalent.
/// Returns `(y, h_out, w_out)`. NCHW input, output layout
/// `[N, C·kh·kw, h_out·w_out]` row-major in `(c, ki, kj)` /
/// `(oh, ow)`.
fn host_im2col_2d(
    n: usize, c: usize, h_in: usize, w_in: usize, x: &[f32],
    kh: usize, kw: usize,
    sh: usize, sw: usize,
    pad_h: usize, pad_w: usize,
    dh: usize, dw: usize,
) -> (Vec<f32>, usize, usize) {
    let h_eff = dh * (kh - 1) + 1;
    let w_eff = dw * (kw - 1) + 1;
    let h_out = (h_in + 2 * pad_h - h_eff) / sh + 1;
    let w_out = (w_in + 2 * pad_w - w_eff) / sw + 1;
    let col_rows = c * kh * kw;
    let spatial = h_out * w_out;
    let mut y = vec![0f32; n * col_rows * spatial];
    for ni in 0..n {
        for ci in 0..c {
            for ki in 0..kh {
                for kj in 0..kw {
                    let row = ci * kh * kw + ki * kw + kj;
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let in_h = (oh * sh + ki * dh) as isize - pad_h as isize;
                            let in_w = (ow * sw + kj * dw) as isize - pad_w as isize;
                            let val = if in_h >= 0
                                && (in_h as usize) < h_in
                                && in_w >= 0
                                && (in_w as usize) < w_in
                            {
                                x[((ni * c + ci) * h_in + in_h as usize) * w_in + in_w as usize]
                            } else {
                                0.0
                            };
                            let col = oh * w_out + ow;
                            y[(ni * col_rows + row) * spatial + col] = val;
                        }
                    }
                }
            }
        }
    }
    (y, h_out, w_out)
}

/// 1-D im2col reference.
fn host_im2col_1d(
    n: usize, c: usize, l_in: usize, x: &[f32],
    kl: usize, sl: usize, pad_l: usize, dl: usize,
) -> (Vec<f32>, usize) {
    let l_eff = dl * (kl - 1) + 1;
    let l_out = (l_in + 2 * pad_l - l_eff) / sl + 1;
    let col_rows = c * kl;
    let mut y = vec![0f32; n * col_rows * l_out];
    for ni in 0..n {
        for ci in 0..c {
            for ki in 0..kl {
                let row = ci * kl + ki;
                for ol in 0..l_out {
                    let in_l = (ol * sl + ki * dl) as isize - pad_l as isize;
                    let val = if in_l >= 0 && (in_l as usize) < l_in {
                        x[(ni * c + ci) * l_in + in_l as usize]
                    } else {
                        0.0
                    };
                    y[(ni * col_rows + row) * l_out + ol] = val;
                }
            }
        }
    }
    (y, l_out)
}

/// 1-D col2im reference (scatter accumulate).
fn host_col2im_1d(
    n: usize, c: usize, l_in: usize, col: &[f32],
    kl: usize, sl: usize, pad_l: usize, dl: usize,
    l_out: usize,
) -> Vec<f32> {
    let col_rows = c * kl;
    let mut out = vec![0f32; n * c * l_in];
    for ni in 0..n {
        for ci in 0..c {
            for ki in 0..kl {
                let row = ci * kl + ki;
                for ol in 0..l_out {
                    let target_l = (ol * sl + ki * dl) as isize - pad_l as isize;
                    if target_l >= 0 && (target_l as usize) < l_in {
                        let val = col[(ni * col_rows + row) * l_out + ol];
                        out[(ni * c + ci) * l_in + target_l as usize] += val;
                    }
                }
            }
        }
    }
    out
}

// =============================================================================
// 2-D tests
// =============================================================================

#[test]
#[ignore]
fn im2col_2d_f32_3x3_stride1() {
    let (ctx, stream) = setup();
    let (n, c, h_in, w_in) = (1i32, 2i32, 4i32, 5i32);
    let host_x: Vec<f32> = (0..(n * c * h_in * w_in) as usize)
        .map(|k| k as f32 - 10.0)
        .collect();
    let (kh, kw, sh, sw, pad_h, pad_w, dh, dw) = (3i32, 3i32, 1i32, 1i32, 0i32, 0i32, 1i32, 1i32);
    let (exp_y, h_out, w_out) = host_im2col_2d(
        n as usize, c as usize, h_in as usize, w_in as usize, &host_x,
        kh as usize, kw as usize, sh as usize, sw as usize,
        pad_h as usize, pad_w as usize, dh as usize, dw as usize,
    );
    assert_eq!(h_out, 2);
    assert_eq!(w_out, 3);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, exp_y.len()).expect("y");

    let desc = Im2ColDescriptor {
        batch: n, channels: c, h_in, w_in,
        kh, kw, stride_h: sh, stride_w: sw,
        pad_h, pad_w, dilation_h: dh, dilation_w: dw,
        element: ElementKind::F32,
    };
    let plan = Im2ColPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    assert_eq!(plan.output_dims(), (h_out as i32, w_out as i32));

    let x_shape = [n, c, h_in, w_in];
    let y_shape = [n, c * kh * kw, h_out as i32 * w_out as i32];
    plan.run(
        &stream,
        Workspace::None,
        Im2ColArgs {
            input: TensorRef {
                data: dev_x.as_slice(),
                shape: x_shape,
                stride: contiguous_stride(x_shape),
            },
            output: TensorMut {
                data: dev_y.as_slice_mut(),
                shape: y_shape,
                stride: contiguous_stride(y_shape),
            },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got_y = vec![0f32; exp_y.len()];
    dev_y.copy_to_host(&mut got_y).expect("dl y");
    for (i, (g, e)) in got_y.iter().zip(exp_y.iter()).enumerate() {
        assert!((g - e).abs() < 1e-5, "im2col_2d mismatch @ {i}: got {g}, exp {e}");
    }
}

#[test]
#[ignore]
fn im2col_2d_f32_3x3_stride2_pad1() {
    let (ctx, stream) = setup();
    let (n, c, h_in, w_in) = (2i32, 3i32, 7i32, 7i32);
    let host_x: Vec<f32> = (0..(n * c * h_in * w_in) as usize)
        .map(|k| (k as f32) * 0.1 - 2.0)
        .collect();
    let (kh, kw, sh, sw, pad_h, pad_w, dh, dw) = (3i32, 3i32, 2i32, 2i32, 1i32, 1i32, 1i32, 1i32);
    let (exp_y, h_out, w_out) = host_im2col_2d(
        n as usize, c as usize, h_in as usize, w_in as usize, &host_x,
        kh as usize, kw as usize, sh as usize, sw as usize,
        pad_h as usize, pad_w as usize, dh as usize, dw as usize,
    );
    assert_eq!(h_out, 4);
    assert_eq!(w_out, 4);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, exp_y.len()).expect("y");

    let desc = Im2ColDescriptor {
        batch: n, channels: c, h_in, w_in,
        kh, kw, stride_h: sh, stride_w: sw,
        pad_h, pad_w, dilation_h: dh, dilation_w: dw,
        element: ElementKind::F32,
    };
    let plan = Im2ColPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let x_shape = [n, c, h_in, w_in];
    let y_shape = [n, c * kh * kw, h_out as i32 * w_out as i32];
    plan.run(
        &stream,
        Workspace::None,
        Im2ColArgs {
            input: TensorRef {
                data: dev_x.as_slice(),
                shape: x_shape,
                stride: contiguous_stride(x_shape),
            },
            output: TensorMut {
                data: dev_y.as_slice_mut(),
                shape: y_shape,
                stride: contiguous_stride(y_shape),
            },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got_y = vec![0f32; exp_y.len()];
    dev_y.copy_to_host(&mut got_y).expect("dl y");
    for (i, (g, e)) in got_y.iter().zip(exp_y.iter()).enumerate() {
        assert!((g - e).abs() < 1e-5, "im2col_2d strided/padded mismatch @ {i}: got {g}, exp {e}");
    }
}

#[test]
#[ignore]
fn im2col_2d_f32_with_dilation() {
    let (ctx, stream) = setup();
    let (n, c, h_in, w_in) = (1i32, 1i32, 7i32, 7i32);
    let host_x: Vec<f32> = (0..(n * c * h_in * w_in) as usize).map(|k| k as f32).collect();
    let (kh, kw, sh, sw, pad_h, pad_w, dh, dw) = (3i32, 3i32, 1i32, 1i32, 0i32, 0i32, 2i32, 2i32);
    let (exp_y, h_out, w_out) = host_im2col_2d(
        n as usize, c as usize, h_in as usize, w_in as usize, &host_x,
        kh as usize, kw as usize, sh as usize, sw as usize,
        pad_h as usize, pad_w as usize, dh as usize, dw as usize,
    );
    assert_eq!(h_out, 3); // (7 - 2*(3-1) - 1)/1 + 1 = 3
    assert_eq!(w_out, 3);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, exp_y.len()).expect("y");

    let desc = Im2ColDescriptor {
        batch: n, channels: c, h_in, w_in,
        kh, kw, stride_h: sh, stride_w: sw,
        pad_h, pad_w, dilation_h: dh, dilation_w: dw,
        element: ElementKind::F32,
    };
    let plan = Im2ColPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let x_shape = [n, c, h_in, w_in];
    let y_shape = [n, c * kh * kw, h_out as i32 * w_out as i32];
    plan.run(
        &stream,
        Workspace::None,
        Im2ColArgs {
            input: TensorRef {
                data: dev_x.as_slice(),
                shape: x_shape,
                stride: contiguous_stride(x_shape),
            },
            output: TensorMut {
                data: dev_y.as_slice_mut(),
                shape: y_shape,
                stride: contiguous_stride(y_shape),
            },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got_y = vec![0f32; exp_y.len()];
    dev_y.copy_to_host(&mut got_y).expect("dl y");
    for (i, (g, e)) in got_y.iter().zip(exp_y.iter()).enumerate() {
        assert!((g - e).abs() < 1e-5, "im2col_2d dilation mismatch @ {i}: got {g}, exp {e}");
    }
}

// =============================================================================
// 1-D tests
// =============================================================================

#[test]
#[ignore]
fn im2col_1d_f32_basic() {
    let (ctx, stream) = setup();
    let (n, c, l_in) = (1i32, 2i32, 8i32);
    let host_x: Vec<f32> = (0..(n * c * l_in) as usize).map(|k| k as f32).collect();
    let (kl, sl, pad_l, dl) = (3i32, 1i32, 1i32, 1i32);
    let (exp_y, l_out) = host_im2col_1d(
        n as usize, c as usize, l_in as usize, &host_x,
        kl as usize, sl as usize, pad_l as usize, dl as usize,
    );
    assert_eq!(l_out, 8); // (8 + 2 - 2 - 1)/1 + 1 = 8

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, exp_y.len()).expect("y");

    let desc = Im2Col1dDescriptor {
        batch: n, channels: c, l_in,
        kl, stride_l: sl, pad_l, dilation_l: dl,
        element: ElementKind::F32,
    };
    let plan =
        Im2Col1dPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    assert_eq!(plan.output_length(), l_out as i32);

    let x_shape = [n, c, l_in];
    let y_shape = [n, c * kl, l_out as i32];
    plan.run(
        &stream,
        Workspace::None,
        Im2Col1dArgs {
            input: TensorRef {
                data: dev_x.as_slice(),
                shape: x_shape,
                stride: contiguous_stride(x_shape),
            },
            output: TensorMut {
                data: dev_y.as_slice_mut(),
                shape: y_shape,
                stride: contiguous_stride(y_shape),
            },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got_y = vec![0f32; exp_y.len()];
    dev_y.copy_to_host(&mut got_y).expect("dl y");
    for (i, (g, e)) in got_y.iter().zip(exp_y.iter()).enumerate() {
        assert!((g - e).abs() < 1e-5, "im2col_1d mismatch @ {i}: got {g}, exp {e}");
    }
}

// =============================================================================
// Col2Im round-trip
// =============================================================================

#[test]
#[ignore]
fn col2im_1d_f32_roundtrip() {
    let (ctx, stream) = setup();
    // Use stride == kernel and no padding so col2im(im2col(x)) == x
    // (no overlap, no boundary loss). Overlap correctness is exercised
    // in the host-reference check below.
    let (n, c, l_in) = (1i32, 2i32, 9i32);
    let host_x: Vec<f32> = (0..(n * c * l_in) as usize).map(|k| (k + 1) as f32).collect();
    let (kl, sl, pad_l, dl) = (3i32, 3i32, 0i32, 1i32);
    let (col, l_out) = host_im2col_1d(
        n as usize, c as usize, l_in as usize, &host_x,
        kl as usize, sl as usize, pad_l as usize, dl as usize,
    );
    assert_eq!(l_out, 3);

    let dev_col = DeviceBuffer::from_slice(&ctx, &col).expect("up col");
    // Output must be pre-zeroed (atomicAdd scatter contract).
    let mut dev_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (n * c * l_in) as usize).expect("out");

    let desc = Col2Im1dDescriptor {
        batch: n, channels: c, l_in,
        kl, stride_l: sl, pad_l, dilation_l: dl,
        element: ElementKind::F32,
    };
    let plan =
        Col2Im1dPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    assert_eq!(plan.input_l_out(), l_out as i32);

    let in_shape = [n, c * kl, l_out as i32];
    let out_shape = [n, c, l_in];
    plan.run(
        &stream,
        Workspace::None,
        Col2Im1dArgs {
            input: TensorRef {
                data: dev_col.as_slice(),
                shape: in_shape,
                stride: contiguous_stride(in_shape),
            },
            output: TensorMut {
                data: dev_out.as_slice_mut(),
                shape: out_shape,
                stride: contiguous_stride(out_shape),
            },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got_out = vec![0f32; host_x.len()];
    dev_out.copy_to_host(&mut got_out).expect("dl out");

    // Stride == kernel + no pad → every input cell appears in exactly
    // one col slot, so col2im(im2col(x)) == x.
    for (i, (g, e)) in got_out.iter().zip(host_x.iter()).enumerate() {
        assert!((g - e).abs() < 1e-5, "col2im_1d roundtrip mismatch @ {i}: got {g}, exp {e}");
    }

    // Now exercise the overlap case (stride < kernel): col2im should
    // match the host reference accumulator.
    let (kl2, sl2, pad2, dl2) = (3i32, 1i32, 1i32, 1i32);
    let (col2, l_out2) = host_im2col_1d(
        n as usize, c as usize, l_in as usize, &host_x,
        kl2 as usize, sl2 as usize, pad2 as usize, dl2 as usize,
    );
    let exp_out2 = host_col2im_1d(
        n as usize, c as usize, l_in as usize, &col2,
        kl2 as usize, sl2 as usize, pad2 as usize, dl2 as usize, l_out2,
    );

    let dev_col2 = DeviceBuffer::from_slice(&ctx, &col2).expect("up col2");
    let mut dev_out2: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (n * c * l_in) as usize).expect("out2");
    let desc2 = Col2Im1dDescriptor {
        batch: n, channels: c, l_in,
        kl: kl2, stride_l: sl2, pad_l: pad2, dilation_l: dl2,
        element: ElementKind::F32,
    };
    let plan2 =
        Col2Im1dPlan::<f32>::select(&stream, &desc2, PlanPreference::default()).expect("sel2");
    let in_shape2 = [n, c * kl2, l_out2 as i32];
    plan2
        .run(
            &stream,
            Workspace::None,
            Col2Im1dArgs {
                input: TensorRef {
                    data: dev_col2.as_slice(),
                    shape: in_shape2,
                    stride: contiguous_stride(in_shape2),
                },
                output: TensorMut {
                    data: dev_out2.as_slice_mut(),
                    shape: out_shape,
                    stride: contiguous_stride(out_shape),
                },
            },
        )
        .expect("run2");
    stream.synchronize().expect("sync2");
    let mut got_out2 = vec![0f32; host_x.len()];
    dev_out2.copy_to_host(&mut got_out2).expect("dl out2");
    for (i, (g, e)) in got_out2.iter().zip(exp_out2.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-4,
            "col2im_1d overlap mismatch @ {i}: got {g}, exp {e}"
        );
    }
}

// =============================================================================
// f16 / bf16 quick correctness on 2-D
// =============================================================================

#[test]
#[ignore]
fn im2col_2d_f16_basic() {
    let (ctx, stream) = setup();
    let (n, c, h_in, w_in) = (1i32, 1i32, 4i32, 4i32);
    let host_x_f32: Vec<f32> = (0..(n * c * h_in * w_in) as usize).map(|k| k as f32).collect();
    let host_x: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let (kh, kw, sh, sw, pad_h, pad_w, dh, dw) = (2i32, 2i32, 1i32, 1i32, 0i32, 0i32, 1i32, 1i32);
    let (exp_y_f32, h_out, w_out) = host_im2col_2d(
        n as usize, c as usize, h_in as usize, w_in as usize, &host_x_f32,
        kh as usize, kw as usize, sh as usize, sw as usize,
        pad_h as usize, pad_w as usize, dh as usize, dw as usize,
    );

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, exp_y_f32.len()).expect("y");

    let desc = Im2ColDescriptor {
        batch: n, channels: c, h_in, w_in,
        kh, kw, stride_h: sh, stride_w: sw,
        pad_h, pad_w, dilation_h: dh, dilation_w: dw,
        element: ElementKind::F16,
    };
    let plan = Im2ColPlan::<f16>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let x_shape = [n, c, h_in, w_in];
    let y_shape = [n, c * kh * kw, h_out as i32 * w_out as i32];
    plan.run(
        &stream,
        Workspace::None,
        Im2ColArgs {
            input: TensorRef {
                data: dev_x.as_slice(),
                shape: x_shape,
                stride: contiguous_stride(x_shape),
            },
            output: TensorMut {
                data: dev_y.as_slice_mut(),
                shape: y_shape,
                stride: contiguous_stride(y_shape),
            },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got_y = vec![f16::from_f32(0.0); exp_y_f32.len()];
    dev_y.copy_to_host(&mut got_y).expect("dl y");
    for (i, (g, e)) in got_y.iter().zip(exp_y_f32.iter()).enumerate() {
        let gv = g.to_f32();
        assert!((gv - e).abs() < 1e-2, "im2col_2d f16 mismatch @ {i}: got {gv}, exp {e}");
    }
}

#[test]
#[ignore]
fn im2col_2d_bf16_basic() {
    let (ctx, stream) = setup();
    let (n, c, h_in, w_in) = (1i32, 1i32, 4i32, 4i32);
    let host_x_f32: Vec<f32> = (0..(n * c * h_in * w_in) as usize).map(|k| k as f32).collect();
    let host_x: Vec<bf16> = host_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let (kh, kw, sh, sw, pad_h, pad_w, dh, dw) = (2i32, 2i32, 1i32, 1i32, 0i32, 0i32, 1i32, 1i32);
    let (exp_y_f32, h_out, w_out) = host_im2col_2d(
        n as usize, c as usize, h_in as usize, w_in as usize, &host_x_f32,
        kh as usize, kw as usize, sh as usize, sw as usize,
        pad_h as usize, pad_w as usize, dh as usize, dw as usize,
    );

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, exp_y_f32.len()).expect("y");

    let desc = Im2ColDescriptor {
        batch: n, channels: c, h_in, w_in,
        kh, kw, stride_h: sh, stride_w: sw,
        pad_h, pad_w, dilation_h: dh, dilation_w: dw,
        element: ElementKind::Bf16,
    };
    let plan =
        Im2ColPlan::<bf16>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let x_shape = [n, c, h_in, w_in];
    let y_shape = [n, c * kh * kw, h_out as i32 * w_out as i32];
    plan.run(
        &stream,
        Workspace::None,
        Im2ColArgs {
            input: TensorRef {
                data: dev_x.as_slice(),
                shape: x_shape,
                stride: contiguous_stride(x_shape),
            },
            output: TensorMut {
                data: dev_y.as_slice_mut(),
                shape: y_shape,
                stride: contiguous_stride(y_shape),
            },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got_y = vec![bf16::from_f32(0.0); exp_y_f32.len()];
    dev_y.copy_to_host(&mut got_y).expect("dl y");
    for (i, (g, e)) in got_y.iter().zip(exp_y_f32.iter()).enumerate() {
        let gv = g.to_f32();
        // bf16 has only 7 mantissa bits — tolerate ~1% relative.
        let tol = (e.abs() * 1e-2).max(1e-1);
        assert!((gv - e).abs() < tol, "im2col_2d bf16 mismatch @ {i}: got {gv}, exp {e}");
    }
}
