//! Real-GPU smoke for `RoiPoolPlan<T>` (Phase 9 Category T).

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, RoiPoolArgs, RoiPoolBackwardArgs,
    RoiPoolBackwardDescriptor, RoiPoolBackwardPlan, RoiPoolDescriptor, RoiPoolPlan, TensorMut,
    TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn roi_pool_full_image_f32_smoke() {
    let (ctx, stream) = setup();
    let (n, c, h, w) = (1, 1, 4, 4);
    let host_in: Vec<f32> = (0..(n * c * h * w)).map(|i| i as f32 + 1.0).collect();
    // Full-image RoI; pooled = 2 → 4 quadrant maxes.
    let host_rois: Vec<f32> = vec![0.0, 0.0, 0.0, 3.0, 3.0];
    let num_rois = 1;
    let pooled = 2;
    let dev_in = DeviceBuffer::from_slice(&ctx, &host_in).expect("");
    let dev_rois = DeviceBuffer::from_slice(&ctx, &host_rois).expect("");
    let mut dev_out: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 4).expect("");
    let mut dev_arg: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 4).expect("");
    let desc = RoiPoolDescriptor {
        n, c, h, w, num_rois, pooled_h: pooled, pooled_w: pooled,
        spatial_scale: 1.0,
        element: ElementKind::F32,
    };
    let plan = RoiPoolPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("");
    plan.run(&stream, Workspace::None, RoiPoolArgs {
        input: TensorRef {
            data: dev_in.as_slice(),
            shape: [n, c, h, w],
            stride: contiguous_stride([n, c, h, w]),
        },
        rois: TensorRef {
            data: dev_rois.as_slice(),
            shape: [num_rois, 5],
            stride: contiguous_stride([num_rois, 5]),
        },
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [num_rois, c, pooled, pooled],
            stride: contiguous_stride([num_rois, c, pooled, pooled]),
        },
        argmax: TensorMut {
            data: dev_arg.as_slice_mut(),
            shape: [num_rois, c, pooled, pooled],
            stride: contiguous_stride([num_rois, c, pooled, pooled]),
        },
    }).expect("");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; 4];
    let mut argmax = vec![0i32; 4];
    dev_out.copy_to_host(&mut got).expect("");
    dev_arg.copy_to_host(&mut argmax).expect("");
    // Each output cell should be the max of its quadrant.
    // Quadrant maxes: TL=6, TR=8, BL=14, BR=16.
    assert_eq!(got, vec![6.0, 8.0, 14.0, 16.0]);
    // argmax must all be in [0, 16).
    for &a in &argmax {
        assert!(a >= 0 && a < 16);
    }
}

#[test]
#[ignore]
fn roi_pool_backward_smoke_f32() {
    let (ctx, stream) = setup();
    let (n, c, h, w) = (1, 1, 4, 4);
    let host_rois: Vec<f32> = vec![0.0, 0.0, 0.0, 3.0, 3.0];
    let num_rois = 1;
    let pooled = 2;
    let host_dout: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let host_argmax: Vec<i32> = vec![5, 7, 13, 15];  // Indices of quadrant maxes.
    let dev_rois = DeviceBuffer::from_slice(&ctx, &host_rois).expect("");
    let dev_dout = DeviceBuffer::from_slice(&ctx, &host_dout).expect("");
    let dev_arg = DeviceBuffer::from_slice(&ctx, &host_argmax).expect("");
    let mut dev_din: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (n * c * h * w) as usize).expect("");
    let plan = RoiPoolBackwardPlan::<f32>::select(
        &stream,
        &RoiPoolBackwardDescriptor {
            n, c, h, w, num_rois, pooled_h: pooled, pooled_w: pooled,
            element: ElementKind::F32,
        },
        PlanPreference::default(),
    ).expect("");
    plan.run(&stream, Workspace::None, RoiPoolBackwardArgs {
        dout: TensorRef {
            data: dev_dout.as_slice(),
            shape: [num_rois, c, pooled, pooled],
            stride: contiguous_stride([num_rois, c, pooled, pooled]),
        },
        rois: TensorRef {
            data: dev_rois.as_slice(),
            shape: [num_rois, 5],
            stride: contiguous_stride([num_rois, 5]),
        },
        argmax: TensorRef {
            data: dev_arg.as_slice(),
            shape: [num_rois, c, pooled, pooled],
            stride: contiguous_stride([num_rois, c, pooled, pooled]),
        },
        dinput: TensorMut {
            data: dev_din.as_slice_mut(),
            shape: [n, c, h, w],
            stride: contiguous_stride([n, c, h, w]),
        },
    }).expect("");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; 16];
    dev_din.copy_to_host(&mut got).expect("");
    // Gradient should land only at saved argmax positions: 5→1, 7→2, 13→3, 15→4.
    assert_eq!(got[5], 1.0);
    assert_eq!(got[7], 2.0);
    assert_eq!(got[13], 3.0);
    assert_eq!(got[15], 4.0);
    // Everything else must be 0.
    for (i, &v) in got.iter().enumerate() {
        if !matches!(i, 5 | 7 | 13 | 15) {
            assert_eq!(v, 0.0, "BW grad leak at {i}: {v}");
        }
    }
}
