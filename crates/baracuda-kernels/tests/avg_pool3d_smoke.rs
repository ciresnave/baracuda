//! Real-GPU smoke test for `AvgPool3dPlan` FW + BW.
#![cfg(feature = "cudnn")]
#![cfg(feature = "device-tests")]
#![allow(
    clippy::needless_range_loop,
    reason = "these host reference models index several parallel buffers by one i; iterator rewrites would hide the element-wise correspondence with the kernel under test."
)]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    AvgPool3dPlan, ElementKind, PlanPreference, Pool3dBwArgs, Pool3dDescriptor, Pool3dFwArgs,
    PoolMode, TensorMut, TensorRef, Workspace, contiguous_stride,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
fn avg_pool3d_f32() {
    let (ctx, stream) = setup();
    let (n, c, d_in, h_in, w_in) = (1i32, 1i32, 2i32, 2i32, 2i32);
    let host_x: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    // Average = (1+2+…+8)/8 = 36/8 = 4.5
    let exp_y: Vec<f32> = vec![4.5];

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).expect("y");
    let desc = Pool3dDescriptor::new(
        n,
        c,
        d_in,
        h_in,
        w_in,
        2,
        2,
        2,
        PoolMode::AvgExcludePad,
        ElementKind::F32,
    );
    let plan =
        AvgPool3dPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let x_shape = [n, c, d_in, h_in, w_in];
    let y_shape = [n, c, 1, 1, 1];
    plan.run_fw(
        &stream,
        Workspace::None,
        Pool3dFwArgs {
            x: TensorRef::new(dev_x.as_slice(), x_shape, contiguous_stride(x_shape)).unwrap(),
            y: TensorMut::new(dev_y.as_slice_mut(), y_shape, contiguous_stride(y_shape)).unwrap(),
        },
    )
    .expect("fw");
    stream.synchronize().expect("sync fw");

    let mut got_y = vec![0f32; 1];
    dev_y.copy_to_host(&mut got_y).expect("dl y");
    let tol = 32.0 * f32::EPSILON;
    assert!(
        (got_y[0] - exp_y[0]).abs() <= (exp_y[0].abs() * tol).max(tol),
        "avg_pool3d f32 y: got={} want={}",
        got_y[0],
        exp_y[0]
    );

    // BW: each input cell gets dy[0] / 8 = 1/8.
    let host_dy: Vec<f32> = vec![1.0];
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 8).expect("dx");
    plan.run_bw(
        &stream,
        Workspace::None,
        Pool3dBwArgs {
            y: TensorRef::new(dev_y.as_slice(), y_shape, contiguous_stride(y_shape)).unwrap(),
            dy: TensorRef::new(dev_dy.as_slice(), y_shape, contiguous_stride(y_shape)).unwrap(),
            x: TensorRef::new(dev_x.as_slice(), x_shape, contiguous_stride(x_shape)).unwrap(),
            dx: TensorMut::new(dev_dx.as_slice_mut(), x_shape, contiguous_stride(x_shape)).unwrap(),
        },
    )
    .expect("bw");
    stream.synchronize().expect("sync bw");

    let mut got_dx = vec![0f32; 8];
    dev_dx.copy_to_host(&mut got_dx).expect("dl dx");
    let want: f32 = 1.0 / 8.0;
    for i in 0..8 {
        let t = (want.abs() * tol).max(tol);
        assert!(
            (got_dx[i] - want).abs() <= t,
            "avg_pool3d f32 dx @ {i}: got={} want={}",
            got_dx[i],
            want
        );
    }
}
