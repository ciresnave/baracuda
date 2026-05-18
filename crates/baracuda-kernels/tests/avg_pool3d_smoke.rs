#![cfg(feature = "cudnn")]
//! Real-GPU smoke test for `AvgPool3dPlan` FW + BW.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, AvgPool3dPlan, ElementKind, PlanPreference, Pool3dBwArgs,
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
fn avg_pool3d_f32() {
    let (ctx, stream) = setup();
    let (n, c, d_in, h_in, w_in) = (1i32, 1i32, 2i32, 2i32, 2i32);
    let host_x: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    // Average = (1+2+…+8)/8 = 36/8 = 4.5
    let exp_y: Vec<f32> = vec![4.5];

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).expect("y");
    let desc = Pool3dDescriptor {
        batch: n, channels: c, d_in, h_in, w_in,
        window_d: 2, window_h: 2, window_w: 2,
        pad_d: 0, pad_h: 0, pad_w: 0,
        stride_d: 2, stride_h: 2, stride_w: 2,
        mode: PoolMode::AvgExcludePad,
        element: ElementKind::F32,
    };
    let plan = AvgPool3dPlan::<f32>::select(&stream, &desc, PlanPreference::default())
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
    let tol = 32.0 * f32::EPSILON;
    assert!((got_y[0] - exp_y[0]).abs() <= (exp_y[0].abs() * tol).max(tol),
        "avg_pool3d f32 y: got={} want={}", got_y[0], exp_y[0]);

    // BW: each input cell gets dy[0] / 8 = 1/8.
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
    let want: f32 = 1.0 / 8.0;
    for i in 0..8 {
        let t = (want.abs() * tol).max(tol);
        assert!((got_dx[i] - want).abs() <= t,
            "avg_pool3d f32 dx @ {i}: got={} want={}", got_dx[i], want);
    }
}
