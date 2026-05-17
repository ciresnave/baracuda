//! Real-GPU smoke test for `PReluBackwardPlan`. BW × 4 dtypes × {per-channel, scalar}.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PReluBackwardArgs, PReluBackwardDescriptor,
    PReluBackwardPlan, PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

const F16_EPS: f32 = 9.77e-4;
const BF16_EPS: f32 = 7.81e-3;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn prelu_backward_f32_per_channel() {
    let (ctx, stream) = setup();
    let shape = [2i32, 3, 4];
    let n_total = (2 * 3 * 4) as usize;
    let c_stride = 4usize;
    let c_extent = 3usize;
    let h_x: Vec<f32> = (0..n_total).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let h_w: Vec<f32> = vec![0.25, 0.5, 0.75];
    let h_dy: Vec<f32> = (0..n_total).map(|i| 0.5 + (i as f32) * 0.01).collect();
    let exp_dx: Vec<f32> = (0..n_total)
        .map(|i| {
            let c = (i as usize / c_stride) % c_extent;
            if h_x[i] > 0.0 { h_dy[i] } else { h_w[c] * h_dy[i] }
        })
        .collect();
    let mut exp_dw = vec![0.0f32; c_extent];
    for i in 0..n_total {
        if h_x[i] < 0.0 {
            let c = (i as usize / c_stride) % c_extent;
            exp_dw[c] += h_dy[i] * h_x[i];
        }
    }
    let dev_x = DeviceBuffer::from_slice(&ctx, &h_x).unwrap();
    let dev_w = DeviceBuffer::from_slice(&ctx, &h_w).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &h_dy).unwrap();
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n_total).unwrap();
    let mut dev_dw: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, c_extent).unwrap();
    let desc = PReluBackwardDescriptor {
        input_shape: shape,
        channel_axis: 1,
        element: ElementKind::F32,
    };
    let plan =
        PReluBackwardPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        PReluBackwardArgs {
            dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
            x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            weight: TensorRef { data: dev_w.as_slice(), shape: [3], stride: [1] },
            dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
            dweight: TensorMut { data: dev_dw.as_slice_mut(), shape: [3], stride: [1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got_dx = vec![0f32; n_total];
    let mut got_dw = vec![0f32; c_extent];
    dev_dx.copy_to_host(&mut got_dx).unwrap();
    dev_dw.copy_to_host(&mut got_dw).unwrap();
    for i in 0..n_total {
        let tol = exp_dx[i].abs() * 4.0 * f32::EPSILON + 1e-7;
        assert!((got_dx[i] - exp_dx[i]).abs() <= tol, "PReLU BW dx @{i}: got={} want={}", got_dx[i], exp_dx[i]);
    }
    for c in 0..c_extent {
        let tol = exp_dw[c].abs() * 16.0 * f32::EPSILON + 1e-5;
        assert!((got_dw[c] - exp_dw[c]).abs() <= tol, "PReLU BW dweight @{c}: got={} want={}", got_dw[c], exp_dw[c]);
    }
}

#[test]
#[ignore]
fn prelu_backward_f32_scalar() {
    let (ctx, stream) = setup();
    let shape = [4i32, 5];
    let n_total = 4 * 5;
    let h_x: Vec<f32> = (0..n_total).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let h_w: Vec<f32> = vec![0.3];
    let h_dy: Vec<f32> = (0..n_total).map(|i| 0.5 + (i as f32) * 0.01).collect();
    let exp_dx: Vec<f32> = (0..n_total)
        .map(|i| if h_x[i] > 0.0 { h_dy[i] } else { h_w[0] * h_dy[i] })
        .collect();
    let mut exp_dw = 0.0f32;
    for i in 0..n_total {
        if h_x[i] < 0.0 {
            exp_dw += h_dy[i] * h_x[i];
        }
    }
    let dev_x = DeviceBuffer::from_slice(&ctx, &h_x).unwrap();
    let dev_w = DeviceBuffer::from_slice(&ctx, &h_w).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &h_dy).unwrap();
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n_total).unwrap();
    let mut dev_dw: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let desc = PReluBackwardDescriptor {
        input_shape: shape,
        channel_axis: -1,
        element: ElementKind::F32,
    };
    let plan =
        PReluBackwardPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        PReluBackwardArgs {
            dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
            x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            weight: TensorRef { data: dev_w.as_slice(), shape: [1], stride: [1] },
            dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
            dweight: TensorMut { data: dev_dw.as_slice_mut(), shape: [1], stride: [1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got_dx = vec![0f32; n_total];
    let mut got_dw = vec![0f32; 1];
    dev_dx.copy_to_host(&mut got_dx).unwrap();
    dev_dw.copy_to_host(&mut got_dw).unwrap();
    for i in 0..n_total {
        let tol = exp_dx[i].abs() * 4.0 * f32::EPSILON + 1e-7;
        assert!((got_dx[i] - exp_dx[i]).abs() <= tol);
    }
    let tol = exp_dw.abs() * 16.0 * f32::EPSILON + 1e-5;
    assert!((got_dw[0] - exp_dw).abs() <= tol);
}

#[test]
#[ignore]
fn prelu_backward_f64_per_channel() {
    let (ctx, stream) = setup();
    let shape = [2i32, 3, 4];
    let n_total = (2 * 3 * 4) as usize;
    let c_stride = 4usize;
    let c_extent = 3usize;
    let h_x: Vec<f64> = (0..n_total).map(|i| (i as f64) * 0.1 - 1.0).collect();
    let h_w: Vec<f64> = vec![0.25, 0.5, 0.75];
    let h_dy: Vec<f64> = (0..n_total).map(|i| 0.5 + (i as f64) * 0.01).collect();
    let exp_dx: Vec<f64> = (0..n_total)
        .map(|i| {
            let c = (i as usize / c_stride) % c_extent;
            if h_x[i] > 0.0 { h_dy[i] } else { h_w[c] * h_dy[i] }
        })
        .collect();
    let mut exp_dw = vec![0.0f64; c_extent];
    for i in 0..n_total {
        if h_x[i] < 0.0 {
            let c = (i as usize / c_stride) % c_extent;
            exp_dw[c] += h_dy[i] * h_x[i];
        }
    }
    let dev_x = DeviceBuffer::from_slice(&ctx, &h_x).unwrap();
    let dev_w = DeviceBuffer::from_slice(&ctx, &h_w).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &h_dy).unwrap();
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, n_total).unwrap();
    let mut dev_dw: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, c_extent).unwrap();
    let desc = PReluBackwardDescriptor {
        input_shape: shape,
        channel_axis: 1,
        element: ElementKind::F64,
    };
    let plan =
        PReluBackwardPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        PReluBackwardArgs {
            dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
            x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            weight: TensorRef { data: dev_w.as_slice(), shape: [3], stride: [1] },
            dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
            dweight: TensorMut { data: dev_dw.as_slice_mut(), shape: [3], stride: [1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got_dx = vec![0f64; n_total];
    let mut got_dw = vec![0f64; c_extent];
    dev_dx.copy_to_host(&mut got_dx).unwrap();
    dev_dw.copy_to_host(&mut got_dw).unwrap();
    for i in 0..n_total {
        let tol = exp_dx[i].abs() * 4.0 * f64::EPSILON + 1e-15;
        assert!((got_dx[i] - exp_dx[i]).abs() <= tol);
    }
    for c in 0..c_extent {
        let tol = exp_dw[c].abs() * 16.0 * f64::EPSILON + 1e-13;
        assert!((got_dw[c] - exp_dw[c]).abs() <= tol);
    }
}

#[test]
#[ignore]
fn prelu_backward_f64_scalar() {
    let (ctx, stream) = setup();
    let shape = [4i32, 5];
    let n_total = 4 * 5;
    let h_x: Vec<f64> = (0..n_total).map(|i| (i as f64) * 0.1 - 1.0).collect();
    let h_w: Vec<f64> = vec![0.3];
    let h_dy: Vec<f64> = (0..n_total).map(|i| 0.5 + (i as f64) * 0.01).collect();
    let exp_dx: Vec<f64> = (0..n_total)
        .map(|i| if h_x[i] > 0.0 { h_dy[i] } else { h_w[0] * h_dy[i] })
        .collect();
    let mut exp_dw = 0.0f64;
    for i in 0..n_total {
        if h_x[i] < 0.0 {
            exp_dw += h_dy[i] * h_x[i];
        }
    }
    let dev_x = DeviceBuffer::from_slice(&ctx, &h_x).unwrap();
    let dev_w = DeviceBuffer::from_slice(&ctx, &h_w).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &h_dy).unwrap();
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, n_total).unwrap();
    let mut dev_dw: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let desc = PReluBackwardDescriptor {
        input_shape: shape,
        channel_axis: -1,
        element: ElementKind::F64,
    };
    let plan =
        PReluBackwardPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        PReluBackwardArgs {
            dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
            x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            weight: TensorRef { data: dev_w.as_slice(), shape: [1], stride: [1] },
            dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
            dweight: TensorMut { data: dev_dw.as_slice_mut(), shape: [1], stride: [1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got_dx = vec![0f64; n_total];
    let mut got_dw = vec![0f64; 1];
    dev_dx.copy_to_host(&mut got_dx).unwrap();
    dev_dw.copy_to_host(&mut got_dw).unwrap();
    for i in 0..n_total {
        let tol = exp_dx[i].abs() * 4.0 * f64::EPSILON + 1e-15;
        assert!((got_dx[i] - exp_dx[i]).abs() <= tol);
    }
    let tol = exp_dw.abs() * 16.0 * f64::EPSILON + 1e-13;
    assert!((got_dw[0] - exp_dw).abs() <= tol);
}

#[test]
#[ignore]
fn prelu_backward_f16_per_channel() {
    let (ctx, stream) = setup();
    let shape = [2i32, 3, 4];
    let n_total = (2 * 3 * 4) as usize;
    let c_stride = 4usize;
    let c_extent = 3usize;
    let h_x_f32: Vec<f32> = (0..n_total).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let h_w_f32: Vec<f32> = vec![0.25, 0.5, 0.75];
    let h_dy_f32: Vec<f32> = (0..n_total).map(|i| 0.5 + (i as f32) * 0.01).collect();
    let h_x: Vec<f16> = h_x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let h_w: Vec<f16> = h_w_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let h_dy: Vec<f16> = h_dy_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let exp_dx: Vec<f32> = (0..n_total)
        .map(|i| {
            let c = (i as usize / c_stride) % c_extent;
            let xv = h_x[i].to_f32();
            let dyv = h_dy[i].to_f32();
            if xv > 0.0 { dyv } else { h_w[c].to_f32() * dyv }
        })
        .collect();
    let mut exp_dw = vec![0.0f32; c_extent];
    for i in 0..n_total {
        let xv = h_x[i].to_f32();
        if xv < 0.0 {
            let c = (i as usize / c_stride) % c_extent;
            exp_dw[c] += h_dy[i].to_f32() * xv;
        }
    }
    let dev_x = DeviceBuffer::from_slice(&ctx, &h_x).unwrap();
    let dev_w = DeviceBuffer::from_slice(&ctx, &h_w).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &h_dy).unwrap();
    let mut dev_dx: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n_total).unwrap();
    let mut dev_dw: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, c_extent).unwrap();
    let desc = PReluBackwardDescriptor {
        input_shape: shape,
        channel_axis: 1,
        element: ElementKind::F16,
    };
    let plan =
        PReluBackwardPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        PReluBackwardArgs {
            dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
            x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            weight: TensorRef { data: dev_w.as_slice(), shape: [3], stride: [1] },
            dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
            dweight: TensorMut { data: dev_dw.as_slice_mut(), shape: [3], stride: [1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got_dx = vec![f16::ZERO; n_total];
    let mut got_dw = vec![f16::ZERO; c_extent];
    dev_dx.copy_to_host(&mut got_dx).unwrap();
    dev_dw.copy_to_host(&mut got_dw).unwrap();
    for i in 0..n_total {
        let g = got_dx[i].to_f32();
        let e = exp_dx[i];
        let tol = e.abs().max(1.0) * 8.0 * F16_EPS + 1e-3;
        assert!((g - e).abs() <= tol, "f16 PReLU BW dx @{i}: got={} want={}", g, e);
    }
    for c in 0..c_extent {
        let g = got_dw[c].to_f32();
        let e = exp_dw[c];
        let tol = e.abs().max(1.0) * 16.0 * F16_EPS + 5e-2;
        assert!((g - e).abs() <= tol, "f16 PReLU BW dw @{c}: got={} want={}", g, e);
    }
}

#[test]
#[ignore]
fn prelu_backward_f16_scalar() {
    let (ctx, stream) = setup();
    let shape = [4i32, 5];
    let n_total = 4 * 5;
    let h_x_f32: Vec<f32> = (0..n_total).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let h_dy_f32: Vec<f32> = (0..n_total).map(|i| 0.5 + (i as f32) * 0.01).collect();
    let h_x: Vec<f16> = h_x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let h_w: Vec<f16> = vec![f16::from_f32(0.3)];
    let h_dy: Vec<f16> = h_dy_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let exp_dx: Vec<f32> = (0..n_total)
        .map(|i| {
            let xv = h_x[i].to_f32();
            let dyv = h_dy[i].to_f32();
            if xv > 0.0 { dyv } else { h_w[0].to_f32() * dyv }
        })
        .collect();
    let mut exp_dw = 0.0f32;
    for i in 0..n_total {
        let xv = h_x[i].to_f32();
        if xv < 0.0 {
            exp_dw += h_dy[i].to_f32() * xv;
        }
    }
    let dev_x = DeviceBuffer::from_slice(&ctx, &h_x).unwrap();
    let dev_w = DeviceBuffer::from_slice(&ctx, &h_w).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &h_dy).unwrap();
    let mut dev_dx: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n_total).unwrap();
    let mut dev_dw: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let desc = PReluBackwardDescriptor {
        input_shape: shape,
        channel_axis: -1,
        element: ElementKind::F16,
    };
    let plan =
        PReluBackwardPlan::<f16, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        PReluBackwardArgs {
            dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
            x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            weight: TensorRef { data: dev_w.as_slice(), shape: [1], stride: [1] },
            dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
            dweight: TensorMut { data: dev_dw.as_slice_mut(), shape: [1], stride: [1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got_dx = vec![f16::ZERO; n_total];
    let mut got_dw = vec![f16::ZERO; 1];
    dev_dx.copy_to_host(&mut got_dx).unwrap();
    dev_dw.copy_to_host(&mut got_dw).unwrap();
    for i in 0..n_total {
        let g = got_dx[i].to_f32();
        let e = exp_dx[i];
        let tol = e.abs().max(1.0) * 8.0 * F16_EPS + 1e-3;
        assert!((g - e).abs() <= tol);
    }
    let g = got_dw[0].to_f32();
    let tol = exp_dw.abs().max(1.0) * 16.0 * F16_EPS + 5e-2;
    assert!((g - exp_dw).abs() <= tol, "f16 PReLU BW dw scalar: got={} want={}", g, exp_dw);
}

#[test]
#[ignore]
fn prelu_backward_bf16_per_channel() {
    let (ctx, stream) = setup();
    let shape = [2i32, 3, 4];
    let n_total = (2 * 3 * 4) as usize;
    let c_stride = 4usize;
    let c_extent = 3usize;
    let h_x_f32: Vec<f32> = (0..n_total).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let h_w_f32: Vec<f32> = vec![0.25, 0.5, 0.75];
    let h_dy_f32: Vec<f32> = (0..n_total).map(|i| 0.5 + (i as f32) * 0.01).collect();
    let h_x: Vec<bf16> = h_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let h_w: Vec<bf16> = h_w_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let h_dy: Vec<bf16> = h_dy_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let exp_dx: Vec<f32> = (0..n_total)
        .map(|i| {
            let c = (i as usize / c_stride) % c_extent;
            let xv = h_x[i].to_f32();
            let dyv = h_dy[i].to_f32();
            if xv > 0.0 { dyv } else { h_w[c].to_f32() * dyv }
        })
        .collect();
    let mut exp_dw = vec![0.0f32; c_extent];
    for i in 0..n_total {
        let xv = h_x[i].to_f32();
        if xv < 0.0 {
            let c = (i as usize / c_stride) % c_extent;
            exp_dw[c] += h_dy[i].to_f32() * xv;
        }
    }
    let dev_x = DeviceBuffer::from_slice(&ctx, &h_x).unwrap();
    let dev_w = DeviceBuffer::from_slice(&ctx, &h_w).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &h_dy).unwrap();
    let mut dev_dx: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, n_total).unwrap();
    let mut dev_dw: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, c_extent).unwrap();
    let desc = PReluBackwardDescriptor {
        input_shape: shape,
        channel_axis: 1,
        element: ElementKind::Bf16,
    };
    let plan =
        PReluBackwardPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        PReluBackwardArgs {
            dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
            x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            weight: TensorRef { data: dev_w.as_slice(), shape: [3], stride: [1] },
            dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
            dweight: TensorMut { data: dev_dw.as_slice_mut(), shape: [3], stride: [1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got_dx = vec![bf16::ZERO; n_total];
    let mut got_dw = vec![bf16::ZERO; c_extent];
    dev_dx.copy_to_host(&mut got_dx).unwrap();
    dev_dw.copy_to_host(&mut got_dw).unwrap();
    for i in 0..n_total {
        let g = got_dx[i].to_f32();
        let e = exp_dx[i];
        let tol = e.abs().max(1.0) * 8.0 * BF16_EPS + 1e-2;
        assert!((g - e).abs() <= tol, "bf16 PReLU BW dx @{i}: got={} want={}", g, e);
    }
    for c in 0..c_extent {
        let g = got_dw[c].to_f32();
        let e = exp_dw[c];
        let tol = e.abs().max(1.0) * 16.0 * BF16_EPS + 1e-1;
        assert!((g - e).abs() <= tol, "bf16 PReLU BW dw @{c}: got={} want={}", g, e);
    }
}

#[test]
#[ignore]
fn prelu_backward_bf16_scalar() {
    let (ctx, stream) = setup();
    let shape = [4i32, 5];
    let n_total = 4 * 5;
    let h_x_f32: Vec<f32> = (0..n_total).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let h_dy_f32: Vec<f32> = (0..n_total).map(|i| 0.5 + (i as f32) * 0.01).collect();
    let h_x: Vec<bf16> = h_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let h_w: Vec<bf16> = vec![bf16::from_f32(0.3)];
    let h_dy: Vec<bf16> = h_dy_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let exp_dx: Vec<f32> = (0..n_total)
        .map(|i| {
            let xv = h_x[i].to_f32();
            let dyv = h_dy[i].to_f32();
            if xv > 0.0 { dyv } else { h_w[0].to_f32() * dyv }
        })
        .collect();
    let mut exp_dw = 0.0f32;
    for i in 0..n_total {
        let xv = h_x[i].to_f32();
        if xv < 0.0 {
            exp_dw += h_dy[i].to_f32() * xv;
        }
    }
    let dev_x = DeviceBuffer::from_slice(&ctx, &h_x).unwrap();
    let dev_w = DeviceBuffer::from_slice(&ctx, &h_w).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &h_dy).unwrap();
    let mut dev_dx: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, n_total).unwrap();
    let mut dev_dw: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let desc = PReluBackwardDescriptor {
        input_shape: shape,
        channel_axis: -1,
        element: ElementKind::Bf16,
    };
    let plan =
        PReluBackwardPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        PReluBackwardArgs {
            dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
            x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            weight: TensorRef { data: dev_w.as_slice(), shape: [1], stride: [1] },
            dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
            dweight: TensorMut { data: dev_dw.as_slice_mut(), shape: [1], stride: [1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got_dx = vec![bf16::ZERO; n_total];
    let mut got_dw = vec![bf16::ZERO; 1];
    dev_dx.copy_to_host(&mut got_dx).unwrap();
    dev_dw.copy_to_host(&mut got_dw).unwrap();
    for i in 0..n_total {
        let g = got_dx[i].to_f32();
        let e = exp_dx[i];
        let tol = e.abs().max(1.0) * 8.0 * BF16_EPS + 1e-2;
        assert!((g - e).abs() <= tol);
    }
    let g = got_dw[0].to_f32();
    let tol = exp_dw.abs().max(1.0) * 16.0 * BF16_EPS + 1e-1;
    assert!((g - exp_dw).abs() <= tol, "bf16 PReLU BW dw scalar: got={} want={}", g, exp_dw);
}
