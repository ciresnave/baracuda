#![cfg(feature = "cudnn")]
//! Real-GPU smoke test for `MaxPool3dPlan` FW + BW.
//!
//! `[1, 1, 2, 2, 2]` input, window 2³, stride 2³ → `[1, 1, 1, 1, 1]`
//! output (a single max over the whole input). Cleanest 3-D fixture.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, MaxPool3dPlan, PlanPreference, Pool3dBwArgs,
    Pool3dDescriptor, Pool3dFwArgs, PoolMode, TensorMut, TensorRef, Workspace,
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
fn max_pool3d_f32() {
    let (ctx, stream) = setup();
    let (n, c, d_in, h_in, w_in) = (1i32, 1i32, 2i32, 2i32, 2i32);
    // 8 distinct values; max is 8.
    let host_x: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    let exp_y: Vec<f32> = vec![8.0];

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).expect("y");
    let desc = Pool3dDescriptor::new(
        n, c, d_in, h_in, w_in, 2, 2, 2, PoolMode::Max, ElementKind::F32,
    );
    let plan = MaxPool3dPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let x_shape = [n, c, d_in, h_in, w_in];
    let y_shape = [n, c, 1, 1, 1];
    plan.run_fw(&stream, Workspace::None, Pool3dFwArgs {
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: contiguous_stride(x_shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: y_shape, stride: contiguous_stride(y_shape) },
    }).expect("fw");
    stream.synchronize().expect("sync fw");

    let mut got_y = vec![0f32; 1];
    dev_y.copy_to_host(&mut got_y).expect("dl y");
    assert!((got_y[0] - exp_y[0]).abs() <= 16.0 * f32::EPSILON,
        "max_pool3d f32 y: got={} want={}", got_y[0], exp_y[0]);

    // BW: dy=1.0 → argmax-cell (the last element, value 8) gets 1.0,
    // rest are 0.
    let host_dy: Vec<f32> = vec![1.0];
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 8).expect("dx");
    plan.run_bw(&stream, Workspace::None, Pool3dBwArgs {
        y: TensorRef { data: dev_y.as_slice(), shape: y_shape, stride: contiguous_stride(y_shape) },
        dy: TensorRef { data: dev_dy.as_slice(), shape: y_shape, stride: contiguous_stride(y_shape) },
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: contiguous_stride(x_shape) },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape: x_shape, stride: contiguous_stride(x_shape) },
    }).expect("bw");
    stream.synchronize().expect("sync bw");

    let mut got_dx = vec![0f32; 8];
    dev_dx.copy_to_host(&mut got_dx).expect("dl dx");
    let mut exp_dx = vec![0f32; 8];
    exp_dx[7] = 1.0; // value 8 is at the last position
    for i in 0..8 {
        let t = (exp_dx[i].abs() * 16.0 * f32::EPSILON).max(16.0 * f32::EPSILON);
        assert!((got_dx[i] - exp_dx[i]).abs() <= t,
            "max_pool3d f32 dx @ {i}: got={} want={}", got_dx[i], exp_dx[i]);
    }
}
