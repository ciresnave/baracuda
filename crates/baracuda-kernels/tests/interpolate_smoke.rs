//! Real-GPU smoke test for `InterpolatePlan<T>` (Phase 9 Category T).
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
