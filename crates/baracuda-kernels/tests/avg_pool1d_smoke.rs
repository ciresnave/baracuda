#![cfg(feature = "cudnn")]
//! Real-GPU smoke test for `AvgPool1dPlan` FW + BW.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, AvgPool1dPlan, ElementKind, PlanPreference, Pool1dBwArgs,
    Pool1dDescriptor, Pool1dFwArgs, PoolMode, TensorMut, TensorRef, Workspace,
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
fn avg_pool1d_f32() {
    let (ctx, stream) = setup();
    let (n, c, l_in) = (1i32, 1i32, 8i32);
    let host_x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    // 2-wide avg-pool, stride 2:
    // (1+2)/2, (3+4)/2, (5+6)/2, (7+8)/2 = 1.5, 3.5, 5.5, 7.5
    let exp_y: Vec<f32> = vec![1.5, 3.5, 5.5, 7.5];
    let numel_x = host_x.len();
    let numel_y = exp_y.len();
    let l_out = 4i32;

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel_y).expect("y");

    let desc = Pool1dDescriptor::new(
        n, c, l_in, 2, PoolMode::AvgExcludePad, ElementKind::F32,
    );
    let plan = AvgPool1dPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let x_shape = [n, c, l_in];
    let y_shape = [n, c, l_out];
    plan.run_fw(&stream, Workspace::None, Pool1dFwArgs {
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: contiguous_stride(x_shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: y_shape, stride: contiguous_stride(y_shape) },
    }).expect("fw");
    stream.synchronize().expect("sync fw");

    let mut got_y = vec![0f32; numel_y];
    dev_y.copy_to_host(&mut got_y).expect("dl y");
    let tol = 32.0 * f32::EPSILON;
    for i in 0..numel_y {
        let t = (exp_y[i].abs() * tol).max(tol);
        assert!((got_y[i] - exp_y[i]).abs() <= t,
            "avg_pool1d f32 y @ {i}: got={} want={}", got_y[i], exp_y[i]);
    }

    // BW: each input cell receives dy[output_cell] / 2 (window=2,
    // exclude-pad, no padding so denom = 2 everywhere).
    let host_dy: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let mut exp_dx = vec![0f32; numel_x];
    for j in 0..numel_y {
        let g = host_dy[j] / 2.0;
        exp_dx[j * 2] += g;
        exp_dx[j * 2 + 1] += g;
    }
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel_x).expect("dx");
    plan.run_bw(&stream, Workspace::None, Pool1dBwArgs {
        y: TensorRef { data: dev_y.as_slice(), shape: y_shape, stride: contiguous_stride(y_shape) },
        dy: TensorRef { data: dev_dy.as_slice(), shape: y_shape, stride: contiguous_stride(y_shape) },
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: contiguous_stride(x_shape) },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape: x_shape, stride: contiguous_stride(x_shape) },
    }).expect("bw");
    stream.synchronize().expect("sync bw");

    let mut got_dx = vec![0f32; numel_x];
    dev_dx.copy_to_host(&mut got_dx).expect("dl dx");
    for i in 0..numel_x {
        let t = (exp_dx[i].abs() * tol).max(tol);
        assert!((got_dx[i] - exp_dx[i]).abs() <= t,
            "avg_pool1d f32 dx @ {i}: got={} want={}", got_dx[i], exp_dx[i]);
    }
}
