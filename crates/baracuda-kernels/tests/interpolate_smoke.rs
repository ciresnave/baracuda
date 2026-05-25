//! Real-GPU smoke test for `InterpolatePlan<T>` (Phase 9 Category T;
//! Phase 21 align_corners + scale-factor overrides + f16/bf16 fanout).
//!
//! Bilinear-2D upsample of a 1×1×2×2 input to 1×1×4×4, compared
//! against an inline CPU reference using PyTorch's
//! `align_corners=false` mapping. `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, InterpolateArgs, InterpolateBackwardArgs,
    InterpolateBackwardDescriptor, InterpolateBackwardPlan, InterpolateDescriptor,
    InterpolateMode, InterpolatePlan, PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

/// Reference PyTorch-equivalent bilinear-2D forward with full
/// align_corners + scale_factor support. Output is f32 even when the
/// device dtype is f16/bf16 — caller compares within tolerance.
fn cpu_interp_bilinear_full_f32(
    x: &[f32],
    n: i32, c: i32, ih: i32, iw: i32, oh: i32, ow: i32,
    align_corners: bool,
    scale_h: Option<f64>,
    scale_w: Option<f64>,
) -> Vec<f32> {
    let mut out = vec![0f32; (n * c * oh * ow) as usize];
    let sh: f64 = match scale_h {
        Some(s) => 1.0 / s,
        None => {
            if align_corners {
                if oh > 1 { (ih - 1) as f64 / (oh - 1) as f64 } else { 0.0 }
            } else {
                ih as f64 / oh as f64
            }
        }
    };
    let sw: f64 = match scale_w {
        Some(s) => 1.0 / s,
        None => {
            if align_corners {
                if ow > 1 { (iw - 1) as f64 / (ow - 1) as f64 } else { 0.0 }
            } else {
                iw as f64 / ow as f64
            }
        }
    };
    for nn in 0..n {
        for cc in 0..c {
            for ohh in 0..oh {
                let sy = if align_corners {
                    (ohh as f64) * sh
                } else {
                    ((ohh as f64) + 0.5) * sh - 0.5
                } as f32;
                let y0 = sy.floor() as i32;
                let wy1 = sy - y0 as f32;
                let wy0 = 1.0 - wy1;
                let y1 = y0 + 1;
                let cy0 = y0.clamp(0, ih - 1);
                let cy1 = y1.clamp(0, ih - 1);
                for oww in 0..ow {
                    let sx = if align_corners {
                        (oww as f64) * sw
                    } else {
                        ((oww as f64) + 0.5) * sw - 0.5
                    } as f32;
                    let x0 = sx.floor() as i32;
                    let wx1 = sx - x0 as f32;
                    let wx0 = 1.0 - wx1;
                    let x1 = x0 + 1;
                    let cx0 = x0.clamp(0, iw - 1);
                    let cx1 = x1.clamp(0, iw - 1);
                    let plane = ((nn * c + cc) * ih) as usize * iw as usize;
                    let v00 = x[plane + cy0 as usize * iw as usize + cx0 as usize];
                    let v01 = x[plane + cy0 as usize * iw as usize + cx1 as usize];
                    let v10 = x[plane + cy1 as usize * iw as usize + cx0 as usize];
                    let v11 = x[plane + cy1 as usize * iw as usize + cx1 as usize];
                    let o = wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11);
                    let oi = (((nn * c + cc) * oh + ohh) * ow + oww) as usize;
                    out[oi] = o;
                }
            }
        }
    }
    out
}

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn cpu_interp_bilinear_f32(
    x: &[f32], n: i32, c: i32, ih: i32, iw: i32, oh: i32, ow: i32,
) -> Vec<f32> {
    let mut out = vec![0f32; (n * c * oh * ow) as usize];
    for nn in 0..n {
        for cc in 0..c {
            for ohh in 0..oh {
                let sy = (ohh as f32 + 0.5) * (ih as f32 / oh as f32) - 0.5;
                let y0 = sy.floor() as i32;
                let wy1 = sy - y0 as f32;
                let wy0 = 1.0 - wy1;
                let y1 = y0 + 1;
                let cy0 = y0.clamp(0, ih - 1);
                let cy1 = y1.clamp(0, ih - 1);
                for oww in 0..ow {
                    let sx = (oww as f32 + 0.5) * (iw as f32 / ow as f32) - 0.5;
                    let x0 = sx.floor() as i32;
                    let wx1 = sx - x0 as f32;
                    let wx0 = 1.0 - wx1;
                    let x1 = x0 + 1;
                    let cx0 = x0.clamp(0, iw - 1);
                    let cx1 = x1.clamp(0, iw - 1);
                    let plane = ((nn * c + cc) * ih) as usize * iw as usize;
                    let v00 = x[plane + cy0 as usize * iw as usize + cx0 as usize];
                    let v01 = x[plane + cy0 as usize * iw as usize + cx1 as usize];
                    let v10 = x[plane + cy1 as usize * iw as usize + cx0 as usize];
                    let v11 = x[plane + cy1 as usize * iw as usize + cx1 as usize];
                    let o = wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11);
                    let oi = (((nn * c + cc) * oh + ohh) * ow + oww) as usize;
                    out[oi] = o;
                }
            }
        }
    }
    out
}

#[test]
#[ignore]
fn interpolate_bilinear_2d_f32_upsample_2x() {
    let (ctx, stream) = setup();
    let (n, c, ih, iw) = (1, 1, 2, 2);
    let (oh, ow) = (4, 4);
    let host_in: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let expected = cpu_interp_bilinear_f32(&host_in, n, c, ih, iw, oh, ow);

    let dev_in = DeviceBuffer::from_slice(&ctx, &host_in).expect("up in");
    let mut dev_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (n * c * oh * ow) as usize).expect("alloc out");
    let desc = InterpolateDescriptor {
        n, c, ih, iw, oh, ow,
        mode: InterpolateMode::Bilinear2d,
        element: ElementKind::F32,
        align_corners: false,
        scale_h: None,
        scale_w: None,
    };
    let plan = InterpolatePlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = InterpolateArgs {
        input: TensorRef {
            data: dev_in.as_slice(),
            shape: [n, c, ih, iw],
            stride: contiguous_stride([n, c, ih, iw]),
        },
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [n, c, oh, ow],
            stride: contiguous_stride([n, c, oh, ow]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; expected.len()];
    dev_out.copy_to_host(&mut got).expect("dl");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        assert!(diff < 1e-5, "interpolate bilinear f32 mismatch @ {i}: got {g} vs {e}");
    }
}

#[test]
#[ignore]
fn interpolate_bilinear_2d_f64_upsample_2x() {
    let (ctx, stream) = setup();
    let (n, c, ih, iw) = (1, 1, 2, 2);
    let (oh, ow) = (4, 4);
    let host_in: Vec<f64> = vec![0.5, -1.0, 2.5, 3.25];
    // Same formula; just promote to f64.
    let mut expected = vec![0f64; (n * c * oh * ow) as usize];
    for ohh in 0..oh {
        let sy = (ohh as f64 + 0.5) * (ih as f64 / oh as f64) - 0.5;
        let y0 = sy.floor() as i32;
        let wy1 = sy - y0 as f64;
        let wy0 = 1.0 - wy1;
        let y1 = y0 + 1;
        let cy0 = y0.clamp(0, ih - 1);
        let cy1 = y1.clamp(0, ih - 1);
        for oww in 0..ow {
            let sx = (oww as f64 + 0.5) * (iw as f64 / ow as f64) - 0.5;
            let x0 = sx.floor() as i32;
            let wx1 = sx - x0 as f64;
            let wx0 = 1.0 - wx1;
            let x1 = x0 + 1;
            let cx0 = x0.clamp(0, iw - 1);
            let cx1 = x1.clamp(0, iw - 1);
            let v00 = host_in[cy0 as usize * iw as usize + cx0 as usize];
            let v01 = host_in[cy0 as usize * iw as usize + cx1 as usize];
            let v10 = host_in[cy1 as usize * iw as usize + cx0 as usize];
            let v11 = host_in[cy1 as usize * iw as usize + cx1 as usize];
            expected[ohh as usize * ow as usize + oww as usize] =
                wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11);
        }
    }

    let dev_in = DeviceBuffer::from_slice(&ctx, &host_in).expect("up in");
    let mut dev_out: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (n * c * oh * ow) as usize).expect("alloc out");
    let desc = InterpolateDescriptor {
        n, c, ih, iw, oh, ow,
        mode: InterpolateMode::Bilinear2d,
        element: ElementKind::F64,
        align_corners: false,
        scale_h: None,
        scale_w: None,
    };
    let plan = InterpolatePlan::<f64>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = InterpolateArgs {
        input: TensorRef {
            data: dev_in.as_slice(),
            shape: [n, c, ih, iw],
            stride: contiguous_stride([n, c, ih, iw]),
        },
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [n, c, oh, ow],
            stride: contiguous_stride([n, c, oh, ow]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; expected.len()];
    dev_out.copy_to_host(&mut got).expect("dl");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        assert!(diff < 1e-12, "interpolate bilinear f64 mismatch @ {i}: got {g} vs {e}");
    }
}

#[test]
#[ignore]
fn interpolate_bilinear_2d_backward_f32_smoke() {
    // Check that BW returns nonzero gradient at all 4 input cells.
    let (ctx, stream) = setup();
    let (n, c, ih, iw) = (1, 1, 2, 2);
    let (oh, ow) = (4, 4);
    let host_dout: Vec<f32> = vec![1.0; (n * c * oh * ow) as usize];
    let dev_dout = DeviceBuffer::from_slice(&ctx, &host_dout).expect("up");
    let mut dev_din: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (n * c * ih * iw) as usize).expect("alloc");
    let desc = InterpolateBackwardDescriptor {
        n, c, ih, iw, oh, ow,
        mode: InterpolateMode::Bilinear2d,
        element: ElementKind::F32,
        align_corners: false,
        scale_h: None,
        scale_w: None,
    };
    let plan = InterpolateBackwardPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = InterpolateBackwardArgs {
        dout: TensorRef {
            data: dev_dout.as_slice(),
            shape: [n, c, oh, ow],
            stride: contiguous_stride([n, c, oh, ow]),
        },
        dinput: TensorMut {
            data: dev_din.as_slice_mut(),
            shape: [n, c, ih, iw],
            stride: contiguous_stride([n, c, ih, iw]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; 4];
    dev_din.copy_to_host(&mut got).expect("dl");
    // Sum of input gradients should equal sum of output gradients (16).
    let s: f32 = got.iter().sum();
    let target: f32 = host_dout.iter().sum();
    assert!((s - target).abs() < 1e-4, "BW grad sum mismatch: {s} vs {target}");
    for (i, &v) in got.iter().enumerate() {
        assert!(v > 0.0, "interpolate BW grad at {i} not positive: {v}");
    }
}

// ============================================================================
// Phase 21 — align_corners + scale-factor overrides + f16/bf16 fanout.
// ============================================================================

/// Drives a single FW launch through the plan and returns the device
/// result alongside the CPU-equivalent reference. Used by the matrix
/// of align_corners / scale_factor / dtype permutations below.
fn run_fw_f32_case(
    ctx: &Context, stream: &Stream,
    host_in: &[f32],
    n: i32, c: i32, ih: i32, iw: i32, oh: i32, ow: i32,
    align_corners: bool,
    scale_h: Option<f64>,
    scale_w: Option<f64>,
) -> (Vec<f32>, Vec<f32>) {
    let expected = cpu_interp_bilinear_full_f32(
        host_in, n, c, ih, iw, oh, ow, align_corners, scale_h, scale_w,
    );
    let dev_in = DeviceBuffer::from_slice(ctx, host_in).expect("up");
    let mut dev_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(ctx, (n * c * oh * ow) as usize).expect("alloc out");
    let desc = InterpolateDescriptor {
        n, c, ih, iw, oh, ow,
        mode: InterpolateMode::Bilinear2d,
        element: ElementKind::F32,
        align_corners,
        scale_h,
        scale_w,
    };
    let plan = InterpolatePlan::<f32>::select(stream, &desc, PlanPreference::default())
        .expect("select");
    let args = InterpolateArgs {
        input: TensorRef {
            data: dev_in.as_slice(),
            shape: [n, c, ih, iw],
            stride: contiguous_stride([n, c, ih, iw]),
        },
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [n, c, oh, ow],
            stride: contiguous_stride([n, c, oh, ow]),
        },
    };
    plan.run(stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; expected.len()];
    dev_out.copy_to_host(&mut got).expect("dl");
    (got, expected)
}

#[test]
#[ignore]
fn interpolate_bilinear_2d_f32_align_corners_true_4x4_to_7x7() {
    // 4x4 → 7x7 with align_corners=true should differ from
    // align_corners=false: corners are exactly preserved (top-left,
    // top-right, bottom-left, bottom-right output cells match input
    // corners), and interior points sample at (i-1)/(o-1) ratio rather
    // than i/o ratio.
    let (ctx, stream) = setup();
    let host_in: Vec<f32> = (0..16).map(|i| i as f32 + 1.0).collect();

    let (got_ac, exp_ac) = run_fw_f32_case(
        &ctx, &stream, &host_in, 1, 1, 4, 4, 7, 7,
        true, None, None,
    );
    let (got_no, exp_no) = run_fw_f32_case(
        &ctx, &stream, &host_in, 1, 1, 4, 4, 7, 7,
        false, None, None,
    );
    // Each path must match its own reference.
    for (i, (g, e)) in got_ac.iter().zip(exp_ac.iter()).enumerate() {
        assert!((g - e).abs() < 5e-5, "ac=true f32 mismatch @ {i}: {g} vs {e}");
    }
    for (i, (g, e)) in got_no.iter().zip(exp_no.iter()).enumerate() {
        assert!((g - e).abs() < 5e-5, "ac=false f32 mismatch @ {i}: {g} vs {e}");
    }
    // align_corners=true must produce a different output than ac=false
    // for at least one cell (different mapping → different result).
    let max_diff = got_ac.iter().zip(got_no.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max_diff > 0.1,
        "ac=true vs ac=false produced near-identical output: max diff {max_diff}");
    // Corners should match input corners exactly under align_corners=true.
    assert!((got_ac[0] - host_in[0]).abs() < 1e-5);     // top-left
    assert!((got_ac[6] - host_in[3]).abs() < 1e-5);     // top-right
    assert!((got_ac[42] - host_in[12]).abs() < 1e-5);   // bottom-left
    assert!((got_ac[48] - host_in[15]).abs() < 1e-5);   // bottom-right
}

#[test]
#[ignore]
fn interpolate_bilinear_2d_f32_scale_factor_override_differs() {
    // 2x2 → 4x4, but with scale_h_factor=3.0 / scale_w_factor=3.0,
    // i.e. a SCALE override that doesn't match the (4/2)=2 ratio.
    // Per PyTorch's convention this changes the per-output step from
    // 1.0 / 2.0 = 0.5 to 1.0 / 3.0 ≈ 0.333, producing a substantively
    // different output.
    let (ctx, stream) = setup();
    let host_in: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let (got_override, exp_override) = run_fw_f32_case(
        &ctx, &stream, &host_in, 1, 1, 2, 2, 4, 4,
        false, Some(3.0), Some(3.0),
    );
    let (got_default, _) = run_fw_f32_case(
        &ctx, &stream, &host_in, 1, 1, 2, 2, 4, 4,
        false, None, None,
    );
    for (i, (g, e)) in got_override.iter().zip(exp_override.iter()).enumerate() {
        assert!((g - e).abs() < 5e-5,
            "scale-override f32 mismatch @ {i}: {g} vs {e}");
    }
    let max_diff = got_override.iter().zip(got_default.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max_diff > 0.1,
        "scale override vs derived produced near-identical output: max diff {max_diff}");
}

#[test]
#[ignore]
fn interpolate_bilinear_2d_f16_fw_matches_f32_ref() {
    // f16 path correctness within Q8_0-style tolerance against the
    // f32 CPU reference (computed at f32, then compared after the
    // device half→f32 round-trip).
    let (ctx, stream) = setup();
    let (n, c, ih, iw) = (1i32, 2i32, 4i32, 4i32);
    let (oh, ow) = (8i32, 8i32);
    let host_in_f32: Vec<f32> = (0..(n * c * ih * iw) as usize)
        .map(|i| (i as f32) * 0.1 - 1.0)
        .collect();
    let host_in_f16: Vec<f16> = host_in_f32.iter().map(|&v| f16::from_f32(v)).collect();
    // Use the f16-input-rounded values for the reference so we measure
    // kernel-only error rather than input-quantization error.
    let host_in_for_ref: Vec<f32> = host_in_f16.iter().map(|v| v.to_f32()).collect();
    let expected = cpu_interp_bilinear_full_f32(
        &host_in_for_ref, n, c, ih, iw, oh, ow, false, None, None,
    );

    let dev_in = DeviceBuffer::from_slice(&ctx, &host_in_f16).expect("up");
    let mut dev_out: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (n * c * oh * ow) as usize).expect("alloc out");
    let desc = InterpolateDescriptor {
        n, c, ih, iw, oh, ow,
        mode: InterpolateMode::Bilinear2d,
        element: ElementKind::F16,
        align_corners: false,
        scale_h: None,
        scale_w: None,
    };
    let plan = InterpolatePlan::<f16>::select(&stream, &desc, PlanPreference::default())
        .expect("select f16");
    let args = InterpolateArgs {
        input: TensorRef {
            data: dev_in.as_slice(),
            shape: [n, c, ih, iw],
            stride: contiguous_stride([n, c, ih, iw]),
        },
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [n, c, oh, ow],
            stride: contiguous_stride([n, c, oh, ow]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got_h: Vec<f16> = vec![f16::ZERO; expected.len()];
    dev_out.copy_to_host(&mut got_h).expect("dl");
    let got: Vec<f32> = got_h.iter().map(|v| v.to_f32()).collect();
    // f16 ulp ~ 9.77e-4; allow a few ulp of accumulated rounding.
    let tol = 5e-3_f32;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        assert!(diff <= tol * e.abs().max(1.0) + tol,
            "f16 FW mismatch @ {i}: got {g}, ref {e}, diff {diff}");
    }
}

#[test]
#[ignore]
fn interpolate_bilinear_2d_bf16_fw_matches_f32_ref() {
    let (ctx, stream) = setup();
    let (n, c, ih, iw) = (1i32, 1i32, 4i32, 4i32);
    let (oh, ow) = (8i32, 8i32);
    let host_in_f32: Vec<f32> = (0..(n * c * ih * iw) as usize)
        .map(|i| (i as f32) * 0.25 - 1.0)
        .collect();
    let host_in_bf16: Vec<bf16> = host_in_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_in_for_ref: Vec<f32> = host_in_bf16.iter().map(|v| v.to_f32()).collect();
    let expected = cpu_interp_bilinear_full_f32(
        &host_in_for_ref, n, c, ih, iw, oh, ow, true, None, None,
    );

    let dev_in = DeviceBuffer::from_slice(&ctx, &host_in_bf16).expect("up");
    let mut dev_out: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, (n * c * oh * ow) as usize).expect("alloc out");
    let desc = InterpolateDescriptor {
        n, c, ih, iw, oh, ow,
        mode: InterpolateMode::Bilinear2d,
        element: ElementKind::Bf16,
        align_corners: true,
        scale_h: None,
        scale_w: None,
    };
    let plan = InterpolatePlan::<bf16>::select(&stream, &desc, PlanPreference::default())
        .expect("select bf16");
    let args = InterpolateArgs {
        input: TensorRef {
            data: dev_in.as_slice(),
            shape: [n, c, ih, iw],
            stride: contiguous_stride([n, c, ih, iw]),
        },
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [n, c, oh, ow],
            stride: contiguous_stride([n, c, oh, ow]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got_h: Vec<bf16> = vec![bf16::ZERO; expected.len()];
    dev_out.copy_to_host(&mut got_h).expect("dl");
    let got: Vec<f32> = got_h.iter().map(|v| v.to_f32()).collect();
    // bf16 ulp ~ 7.81e-3; bilinear has up to 4-way sum so allow ~4 ulp.
    let tol = 4e-2_f32;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        assert!(diff <= tol * e.abs().max(1.0) + tol,
            "bf16 FW mismatch @ {i}: got {g}, ref {e}, diff {diff}");
    }
}

/// Host-side BW reference: scatter-adds dout into dinput using the
/// same bilinear weights as the FW. Mirrors the kernel formula.
fn cpu_interp_bilinear_bw_f32(
    dout: &[f32],
    n: i32, c: i32, ih: i32, iw: i32, oh: i32, ow: i32,
    align_corners: bool,
    scale_h: Option<f64>,
    scale_w: Option<f64>,
) -> Vec<f32> {
    let mut din = vec![0f32; (n * c * ih * iw) as usize];
    let sh: f64 = match scale_h {
        Some(s) => 1.0 / s,
        None => {
            if align_corners {
                if oh > 1 { (ih - 1) as f64 / (oh - 1) as f64 } else { 0.0 }
            } else { ih as f64 / oh as f64 }
        }
    };
    let sw: f64 = match scale_w {
        Some(s) => 1.0 / s,
        None => {
            if align_corners {
                if ow > 1 { (iw - 1) as f64 / (ow - 1) as f64 } else { 0.0 }
            } else { iw as f64 / ow as f64 }
        }
    };
    for nn in 0..n {
        for cc in 0..c {
            for ohh in 0..oh {
                let sy = if align_corners {
                    (ohh as f64) * sh
                } else { ((ohh as f64) + 0.5) * sh - 0.5 } as f32;
                let y0 = sy.floor() as i32;
                let wy1 = sy - y0 as f32;
                let wy0 = 1.0 - wy1;
                let y1 = y0 + 1;
                let cy0 = y0.clamp(0, ih - 1);
                let cy1 = y1.clamp(0, ih - 1);
                for oww in 0..ow {
                    let sx = if align_corners {
                        (oww as f64) * sw
                    } else { ((oww as f64) + 0.5) * sw - 0.5 } as f32;
                    let x0 = sx.floor() as i32;
                    let wx1 = sx - x0 as f32;
                    let wx0 = 1.0 - wx1;
                    let x1 = x0 + 1;
                    let cx0 = x0.clamp(0, iw - 1);
                    let cx1 = x1.clamp(0, iw - 1);
                    let g = dout[(((nn * c + cc) * oh + ohh) * ow + oww) as usize];
                    let plane = ((nn * c + cc) * ih) as usize * iw as usize;
                    din[plane + cy0 as usize * iw as usize + cx0 as usize] += g * wy0 * wx0;
                    din[plane + cy0 as usize * iw as usize + cx1 as usize] += g * wy0 * wx1;
                    din[plane + cy1 as usize * iw as usize + cx0 as usize] += g * wy1 * wx0;
                    din[plane + cy1 as usize * iw as usize + cx1 as usize] += g * wy1 * wx1;
                }
            }
        }
    }
    din
}

fn run_bw_f32_case(
    ctx: &Context, stream: &Stream,
    host_dout: &[f32],
    n: i32, c: i32, ih: i32, iw: i32, oh: i32, ow: i32,
    align_corners: bool,
    scale_h: Option<f64>,
    scale_w: Option<f64>,
) -> (Vec<f32>, Vec<f32>) {
    let expected = cpu_interp_bilinear_bw_f32(
        host_dout, n, c, ih, iw, oh, ow, align_corners, scale_h, scale_w,
    );
    let dev_dout = DeviceBuffer::from_slice(ctx, host_dout).expect("up");
    let mut dev_din: DeviceBuffer<f32> =
        DeviceBuffer::zeros(ctx, (n * c * ih * iw) as usize).expect("alloc din");
    let desc = InterpolateBackwardDescriptor {
        n, c, ih, iw, oh, ow,
        mode: InterpolateMode::Bilinear2d,
        element: ElementKind::F32,
        align_corners,
        scale_h,
        scale_w,
    };
    let plan = InterpolateBackwardPlan::<f32>::select(stream, &desc, PlanPreference::default())
        .expect("select");
    let args = InterpolateBackwardArgs {
        dout: TensorRef {
            data: dev_dout.as_slice(),
            shape: [n, c, oh, ow],
            stride: contiguous_stride([n, c, oh, ow]),
        },
        dinput: TensorMut {
            data: dev_din.as_slice_mut(),
            shape: [n, c, ih, iw],
            stride: contiguous_stride([n, c, ih, iw]),
        },
    };
    plan.run(stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; expected.len()];
    dev_din.copy_to_host(&mut got).expect("dl");
    (got, expected)
}

#[test]
#[ignore]
fn interpolate_bilinear_2d_bw_f32_align_corners_false_matches_cpu() {
    let (ctx, stream) = setup();
    let (n, c, ih, iw) = (1, 1, 3, 3);
    let (oh, ow) = (6, 6);
    let host_dout: Vec<f32> = (0..(n * c * oh * ow) as usize)
        .map(|i| (i as f32) * 0.1 + 0.01)
        .collect();
    let (got, expected) = run_bw_f32_case(
        &ctx, &stream, &host_dout, n, c, ih, iw, oh, ow,
        false, None, None,
    );
    // Grad sum should match dout sum (mass conservation under bilinear
    // weighting that sums to 1 per output cell).
    let gs: f32 = got.iter().sum();
    let es: f32 = expected.iter().sum();
    assert!((gs - es).abs() <= 1e-3 * es.abs().max(1.0),
        "BW grad mass mismatch ac=false: device {gs} vs cpu {es}");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        assert!(diff <= 1e-4 * e.abs().max(1.0) + 1e-4,
            "BW ac=false @ {i}: {g} vs {e}");
    }
}

#[test]
#[ignore]
fn interpolate_bilinear_2d_bw_f32_align_corners_true_matches_cpu() {
    let (ctx, stream) = setup();
    // 4x4 → 7x7 with align_corners=true exercises the corner-pinning
    // codepath of the BW (corner output cells map exactly to corner
    // input cells, distributing all weight to one input).
    let (n, c, ih, iw) = (1, 1, 4, 4);
    let (oh, ow) = (7, 7);
    let host_dout: Vec<f32> = (0..(n * c * oh * ow) as usize)
        .map(|i| 0.5 + (i as f32) * 0.05)
        .collect();
    let (got, expected) = run_bw_f32_case(
        &ctx, &stream, &host_dout, n, c, ih, iw, oh, ow,
        true, None, None,
    );
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        assert!(diff <= 1e-3 * e.abs().max(1.0) + 1e-3,
            "BW ac=true @ {i}: got {g}, cpu {e}, diff {diff}");
    }
}

#[test]
#[ignore]
fn interpolate_bilinear_2d_bw_f32_scale_factor_override_matches_cpu() {
    let (ctx, stream) = setup();
    let (n, c, ih, iw) = (1, 1, 2, 2);
    let (oh, ow) = (4, 4);
    let host_dout: Vec<f32> = (0..(n * c * oh * ow) as usize)
        .map(|i| (i as f32) * 0.1 + 0.01)
        .collect();
    let (got, expected) = run_bw_f32_case(
        &ctx, &stream, &host_dout, n, c, ih, iw, oh, ow,
        false, Some(3.0), Some(3.0),
    );
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        assert!(diff <= 1e-4 * e.abs().max(1.0) + 1e-4,
            "BW scale-override @ {i}: {g} vs {e}");
    }
}
