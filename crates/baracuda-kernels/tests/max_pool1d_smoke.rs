#![cfg(feature = "cudnn")]
//! Real-GPU smoke test for `MaxPool1dPlan` FW + BW.
//!
//! `[1, 2, 8]` input, window 2, stride 2, pad 0 → `[1, 2, 4]` output.
//! Each output cell is the max over a disjoint 2-element input tile.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, MaxPool1dPlan, PlanPreference, Pool1dBwArgs,
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
fn max_pool1d_f32() {
    let (ctx, stream) = setup();
    let (n, c, l_in) = (1i32, 2i32, 8i32);
    // C=0:  [1, 2, 3, 4, 5, 6, 7, 8]   → max-pool-2: [2, 4, 6, 8]
    // C=1:  [8, 7, 6, 5, 4, 3, 2, 1]   → max-pool-2: [8, 6, 4, 2]
    let host_x: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
    ];
    let exp_y: Vec<f32> = vec![2.0, 4.0, 6.0, 8.0, 8.0, 6.0, 4.0, 2.0];
    let numel_x = host_x.len();
    let numel_y = exp_y.len();
    let l_out = 4i32;

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel_y).expect("y");

    let desc =
        Pool1dDescriptor::new(n, c, l_in, 2, PoolMode::Max, ElementKind::F32);
    let plan = MaxPool1dPlan::<f32>::select(&stream, &desc, PlanPreference::default())
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
    let tol = 16.0 * f32::EPSILON;
    for i in 0..numel_y {
        let t = (exp_y[i].abs() * tol).max(tol);
        assert!((got_y[i] - exp_y[i]).abs() <= t,
            "max_pool1d f32 y @ {i}: got={} want={}", got_y[i], exp_y[i]);
    }

    // BW round-trip: pump random-ish dy and check that the argmax cells
    // each receive the corresponding dy and others are 0.
    let host_dy: Vec<f32> = (0..numel_y).map(|i| 0.5 + (i as f32) * 0.25).collect();
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
    // Compute the expected dx — argmax per window receives dy, others 0.
    // C=0 argmax cells: 1, 3, 5, 7 (the right element of each pair).
    // C=1 argmax cells: 8 (=offset 0), 10 (=2), 12 (=4), 14 (=6) — i.e.
    // 0+8, 2+8, 4+8, 6+8 = 8, 10, 12, 14.
    let mut exp_dx = vec![0f32; numel_x];
    // C=0:
    exp_dx[1] = host_dy[0];
    exp_dx[3] = host_dy[1];
    exp_dx[5] = host_dy[2];
    exp_dx[7] = host_dy[3];
    // C=1:
    exp_dx[8 + 0] = host_dy[4];
    exp_dx[8 + 2] = host_dy[5];
    exp_dx[8 + 4] = host_dy[6];
    exp_dx[8 + 6] = host_dy[7];
    for i in 0..numel_x {
        let t = (exp_dx[i].abs() * tol).max(tol);
        assert!((got_dx[i] - exp_dx[i]).abs() <= t,
            "max_pool1d f32 dx @ {i}: got={} want={}", got_dx[i], exp_dx[i]);
    }
}
