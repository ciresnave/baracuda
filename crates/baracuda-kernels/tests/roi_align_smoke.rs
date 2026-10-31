//! Real-GPU smoke for `RoiAlignPlan<T>` (Phase 9 Category T).
//! Single full-image RoI on a 1×1×4×4 input pooled to 2×2 →
//! the exact torchvision-convention bilinear average per bin. gated behind the `device-tests` feature.
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    ElementKind, PlanPreference, RoiAlignArgs, RoiAlignBackwardArgs, RoiAlignBackwardDescriptor,
    RoiAlignBackwardPlan, RoiAlignDescriptor, RoiAlignPlan, TensorMut, TensorRef, Workspace,
    contiguous_stride,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
fn roi_align_full_image_f32_smoke() {
    let (ctx, stream) = setup();
    let (n, c, h, w) = (1, 1, 4, 4);
    let host_in: Vec<f32> = (0..(n * c * h * w)).map(|i| i as f32 + 1.0).collect();
    // Full-image RoI covering [0, 4) × [0, 4): batch 0.
    let host_rois: Vec<f32> = vec![0.0, 0.0, 0.0, 4.0, 4.0];
    let num_rois = 1;
    let pooled = 2;
    let dev_in = DeviceBuffer::from_slice(&ctx, &host_in).expect("");
    let dev_rois = DeviceBuffer::from_slice(&ctx, &host_rois).expect("");
    let mut dev_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (num_rois * c * pooled * pooled) as usize).expect("");
    let desc = RoiAlignDescriptor {
        n,
        c,
        h,
        w,
        num_rois,
        pooled_h: pooled,
        pooled_w: pooled,
        spatial_scale: 1.0,
        sampling_ratio: 0,
        aligned: false,
        element: ElementKind::F32,
    };
    let plan = RoiAlignPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("");
    plan.run(
        &stream,
        Workspace::None,
        RoiAlignArgs {
            input: TensorRef::new(
                dev_in.as_slice(),
                [n, c, h, w],
                contiguous_stride([n, c, h, w]),
            )
            .unwrap(),
            rois: TensorRef::new(
                dev_rois.as_slice(),
                [num_rois, 5],
                contiguous_stride([num_rois, 5]),
            )
            .unwrap(),
            output: TensorMut::new(
                dev_out.as_slice_mut(),
                [num_rois, c, pooled, pooled],
                contiguous_stride([num_rois, c, pooled, pooled]),
            )
            .unwrap(),
        },
    )
    .expect("");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; 4];
    dev_out.copy_to_host(&mut got).expect("");
    // Exact reference (torchvision convention, `aligned = false`, `sampling_ratio = 0`).
    // Pixel (y, x) holds 4y + x + 1 and sits at the integer coordinate (y, x). The RoI
    // [0, 4)^2 pooled to 2x2 has 2x2-pixel bins, and adaptive sampling takes
    // ceil(2 / 1) = 2 samples per axis per bin, at bin_start + (k + 0.5) * 1. Sample
    // coordinates above H - 1 = 3 clamp to 3 (they are inside the image's [-1, H]
    // window, so they are not zeroed). The input is linear, so bilinear sampling is exact:
    //   y samples: top bin {0.5, 1.5}, bottom bin {2.5, 3.5 -> 3}; likewise for x.
    //   cell = 4 * mean(y) + mean(x) + 1 with mean(y|x) = 1.0 (first bin), 2.75 (second).
    // The old check compared against 2x2 quadrant means with a tolerance of 4.0, which
    // is larger than the 2.0 gap between horizontally adjacent cells, so a left/right
    // swapped output passed. The order and each value are pinned here.
    let expected = [6.0f32, 7.75, 13.0, 14.75];
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-4,
            "roi_align cell {i}: got {g}, expected {e}"
        );
    }
}

#[test]
fn roi_align_backward_smoke_f32() {
    let (ctx, stream) = setup();
    let (n, c, h, w) = (1, 1, 4, 4);
    let host_rois: Vec<f32> = vec![0.0, 0.0, 0.0, 4.0, 4.0];
    let num_rois = 1;
    let pooled = 2;
    let host_dout: Vec<f32> = vec![1.0; (num_rois * c * pooled * pooled) as usize];
    let dev_rois = DeviceBuffer::from_slice(&ctx, &host_rois).expect("");
    let dev_dout = DeviceBuffer::from_slice(&ctx, &host_dout).expect("");
    let mut dev_din: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (n * c * h * w) as usize).expect("");
    let desc = RoiAlignBackwardDescriptor {
        n,
        c,
        h,
        w,
        num_rois,
        pooled_h: pooled,
        pooled_w: pooled,
        spatial_scale: 1.0,
        sampling_ratio: 0,
        aligned: false,
        element: ElementKind::F32,
    };
    let plan =
        RoiAlignBackwardPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("");
    plan.run(
        &stream,
        Workspace::None,
        RoiAlignBackwardArgs {
            dout: TensorRef::new(
                dev_dout.as_slice(),
                [num_rois, c, pooled, pooled],
                contiguous_stride([num_rois, c, pooled, pooled]),
            )
            .unwrap(),
            rois: TensorRef::new(
                dev_rois.as_slice(),
                [num_rois, 5],
                contiguous_stride([num_rois, 5]),
            )
            .unwrap(),
            dinput: TensorMut::new(
                dev_din.as_slice_mut(),
                [n, c, h, w],
                contiguous_stride([n, c, h, w]),
            )
            .unwrap(),
        },
    )
    .expect("");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; (n * c * h * w) as usize];
    dev_din.copy_to_host(&mut got).expect("");
    let sum: f32 = got.iter().sum();
    let target: f32 = host_dout.iter().sum();
    // Each output cell distributes its gradient via 4 bilinear corners
    // weighted to sum to 1 over the sampling_ratio = ceil(bin_h)*ceil(bin_w)
    // grid → total grad = dout sum.
    assert!(
        (sum - target).abs() < 1e-3,
        "roi_align BW grad sum {sum} vs target {target}"
    );
}

#[test]
fn roi_align_f64_smoke() {
    let (ctx, stream) = setup();
    let (n, c, h, w) = (1, 1, 4, 4);
    let host_in: Vec<f64> = (0..(n * c * h * w)).map(|i| i as f64 + 1.0).collect();
    let host_rois: Vec<f64> = vec![0.0, 0.0, 0.0, 4.0, 4.0];
    let num_rois = 1;
    let pooled = 2;
    let dev_in = DeviceBuffer::from_slice(&ctx, &host_in).expect("");
    let dev_rois = DeviceBuffer::from_slice(&ctx, &host_rois).expect("");
    let mut dev_out: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, 4).expect("");
    let desc = RoiAlignDescriptor {
        n,
        c,
        h,
        w,
        num_rois,
        pooled_h: pooled,
        pooled_w: pooled,
        spatial_scale: 1.0,
        sampling_ratio: 0,
        aligned: false,
        element: ElementKind::F64,
    };
    let plan = RoiAlignPlan::<f64>::select(&stream, &desc, PlanPreference::default()).expect("");
    plan.run(
        &stream,
        Workspace::None,
        RoiAlignArgs {
            input: TensorRef::new(
                dev_in.as_slice(),
                [n, c, h, w],
                contiguous_stride([n, c, h, w]),
            )
            .unwrap(),
            rois: TensorRef::new(
                dev_rois.as_slice(),
                [num_rois, 5],
                contiguous_stride([num_rois, 5]),
            )
            .unwrap(),
            output: TensorMut::new(
                dev_out.as_slice_mut(),
                [num_rois, c, pooled, pooled],
                contiguous_stride([num_rois, c, pooled, pooled]),
            )
            .unwrap(),
        },
    )
    .expect("");
    stream.synchronize().expect("sync");
    let mut got = vec![0f64; 4];
    dev_out.copy_to_host(&mut got).expect("");
    // Sanity: outputs should be monotonically non-decreasing in the
    // raster order (row-major) given monotone input.
    assert!(
        got[0] < got[1] && got[1] < got[2] && got[2] < got[3],
        "roi_align f64 expected monotone, got {got:?}"
    );
}
