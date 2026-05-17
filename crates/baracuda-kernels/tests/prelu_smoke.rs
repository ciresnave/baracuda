//! Real-GPU smoke test for `PReluPlan`. FW × 4 dtypes × {per-channel, scalar}.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PReluArgs, PReluDescriptor, PReluPlan, PlanPreference,
    TensorMut, TensorRef, Workspace,
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
fn prelu_f32_per_channel() {
    let (ctx, stream) = setup();
    // Shape [N, C, S] = [2, 3, 4], channel axis = 1.
    let shape = [2i32, 3, 4];
    let n_total = 2 * 3 * 4;
    let h_x: Vec<f32> = (0..n_total).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let h_w: Vec<f32> = vec![0.25, 0.5, 0.75];
    let c_stride = 4usize;
    let c_extent = 3usize;
    let expected: Vec<f32> = (0..n_total)
        .map(|i| {
            let c = (i as usize / c_stride) % c_extent;
            let x = h_x[i];
            if x > 0.0 { x } else { h_w[c] * x }
        })
        .collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &h_x).unwrap();
    let dev_w = DeviceBuffer::from_slice(&ctx, &h_w).unwrap();
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n_total).unwrap();
    let desc = PReluDescriptor {
        input_shape: shape,
        channel_axis: 1,
        element: ElementKind::F32,
    };
    let plan = PReluPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        PReluArgs {
            x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            weight: TensorRef { data: dev_w.as_slice(), shape: [3], stride: [1] },
            y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![0f32; n_total];
    dev_y.copy_to_host(&mut got).unwrap();
    for i in 0..n_total {
        let tol = expected[i].abs() * 4.0 * f32::EPSILON + 1e-7;
        assert!((got[i] - expected[i]).abs() <= tol, "PReLU @{i}: got={} want={}", got[i], expected[i]);
    }
}

#[test]
#[ignore]
fn prelu_f32_scalar() {
    let (ctx, stream) = setup();
    let shape = [4i32, 5];
    let n_total = 4 * 5;
    let h_x: Vec<f32> = (0..n_total).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let h_w: Vec<f32> = vec![0.3];
    let expected: Vec<f32> = h_x.iter().map(|&x| if x > 0.0 { x } else { h_w[0] * x }).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &h_x).unwrap();
    let dev_w = DeviceBuffer::from_slice(&ctx, &h_w).unwrap();
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n_total).unwrap();
    let desc = PReluDescriptor {
        input_shape: shape,
        channel_axis: -1,
        element: ElementKind::F32,
    };
    let plan = PReluPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        PReluArgs {
            x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            weight: TensorRef { data: dev_w.as_slice(), shape: [1], stride: [1] },
            y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![0f32; n_total];
    dev_y.copy_to_host(&mut got).unwrap();
    for i in 0..n_total {
        let tol = expected[i].abs() * 4.0 * f32::EPSILON + 1e-7;
        assert!((got[i] - expected[i]).abs() <= tol);
    }
}

#[test]
#[ignore]
fn prelu_f64_per_channel() {
    let (ctx, stream) = setup();
    let shape = [2i32, 3, 2];
    let n_total = 2 * 3 * 2;
    let h_x: Vec<f64> = (0..n_total).map(|i| (i as f64) * 0.1 - 0.7).collect();
    let h_w: Vec<f64> = vec![0.25, 0.5, 0.75];
    let c_stride = 2usize;
    let c_extent = 3usize;
    let expected: Vec<f64> = (0..n_total)
        .map(|i| {
            let c = (i as usize / c_stride) % c_extent;
            let x = h_x[i];
            if x > 0.0 { x } else { h_w[c] * x }
        })
        .collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &h_x).unwrap();
    let dev_w = DeviceBuffer::from_slice(&ctx, &h_w).unwrap();
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, n_total).unwrap();
    let desc = PReluDescriptor {
        input_shape: shape,
        channel_axis: 1,
        element: ElementKind::F64,
    };
    let plan = PReluPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        PReluArgs {
            x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            weight: TensorRef { data: dev_w.as_slice(), shape: [3], stride: [1] },
            y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![0f64; n_total];
    dev_y.copy_to_host(&mut got).unwrap();
    for i in 0..n_total {
        let tol = expected[i].abs() * 4.0 * f64::EPSILON + 1e-15;
        assert!((got[i] - expected[i]).abs() <= tol);
    }
}

#[test]
#[ignore]
fn prelu_f64_scalar() {
    let (ctx, stream) = setup();
    let shape = [4i32, 5];
    let n_total = 4 * 5;
    let h_x: Vec<f64> = (0..n_total).map(|i| (i as f64) * 0.1 - 1.0).collect();
    let h_w: Vec<f64> = vec![0.3];
    let expected: Vec<f64> = h_x.iter().map(|&x| if x > 0.0 { x } else { h_w[0] * x }).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &h_x).unwrap();
    let dev_w = DeviceBuffer::from_slice(&ctx, &h_w).unwrap();
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, n_total).unwrap();
    let desc = PReluDescriptor {
        input_shape: shape,
        channel_axis: -1,
        element: ElementKind::F64,
    };
    let plan = PReluPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        PReluArgs {
            x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            weight: TensorRef { data: dev_w.as_slice(), shape: [1], stride: [1] },
            y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![0f64; n_total];
    dev_y.copy_to_host(&mut got).unwrap();
    for i in 0..n_total {
        let tol = expected[i].abs() * 4.0 * f64::EPSILON + 1e-15;
        assert!((got[i] - expected[i]).abs() <= tol);
    }
}

#[test]
#[ignore]
fn prelu_f16_per_channel() {
    let (ctx, stream) = setup();
    let shape = [2i32, 3, 4];
    let n_total = 2 * 3 * 4;
    let h_x_f32: Vec<f32> = (0..n_total).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let h_w_f32: Vec<f32> = vec![0.25, 0.5, 0.75];
    let h_x: Vec<f16> = h_x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let h_w: Vec<f16> = h_w_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let c_stride = 4usize;
    let c_extent = 3usize;
    let expected: Vec<f32> = (0..n_total)
        .map(|i| {
            let c = (i as usize / c_stride) % c_extent;
            let x = h_x[i].to_f32();
            if x > 0.0 { x } else { h_w[c].to_f32() * x }
        })
        .collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &h_x).unwrap();
    let dev_w = DeviceBuffer::from_slice(&ctx, &h_w).unwrap();
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n_total).unwrap();
    let desc = PReluDescriptor {
        input_shape: shape,
        channel_axis: 1,
        element: ElementKind::F16,
    };
    let plan = PReluPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        PReluArgs {
            x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            weight: TensorRef { data: dev_w.as_slice(), shape: [3], stride: [1] },
            y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![f16::ZERO; n_total];
    dev_y.copy_to_host(&mut got).unwrap();
    for i in 0..n_total {
        let g = got[i].to_f32();
        let e = expected[i];
        let tol = e.abs().max(1.0) * 8.0 * F16_EPS + 1e-3;
        assert!((g - e).abs() <= tol, "PReLU f16 per-chan @{i}: got={} want={}", g, e);
    }
}

#[test]
#[ignore]
fn prelu_f16_scalar() {
    let (ctx, stream) = setup();
    let shape = [4i32, 5];
    let n_total = 4 * 5;
    let h_x_f32: Vec<f32> = (0..n_total).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let h_x: Vec<f16> = h_x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let h_w: Vec<f16> = vec![f16::from_f32(0.3)];
    let expected: Vec<f32> = h_x.iter()
        .map(|x| {
            let xv = x.to_f32();
            if xv > 0.0 { xv } else { h_w[0].to_f32() * xv }
        })
        .collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &h_x).unwrap();
    let dev_w = DeviceBuffer::from_slice(&ctx, &h_w).unwrap();
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n_total).unwrap();
    let desc = PReluDescriptor {
        input_shape: shape,
        channel_axis: -1,
        element: ElementKind::F16,
    };
    let plan = PReluPlan::<f16, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        PReluArgs {
            x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            weight: TensorRef { data: dev_w.as_slice(), shape: [1], stride: [1] },
            y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![f16::ZERO; n_total];
    dev_y.copy_to_host(&mut got).unwrap();
    for i in 0..n_total {
        let g = got[i].to_f32();
        let e = expected[i];
        let tol = e.abs().max(1.0) * 8.0 * F16_EPS + 1e-3;
        assert!((g - e).abs() <= tol, "PReLU f16 scalar @{i}: got={} want={}", g, e);
    }
}

#[test]
#[ignore]
fn prelu_bf16_per_channel() {
    let (ctx, stream) = setup();
    let shape = [2i32, 3, 4];
    let n_total = 2 * 3 * 4;
    let h_x_f32: Vec<f32> = (0..n_total).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let h_w_f32: Vec<f32> = vec![0.25, 0.5, 0.75];
    let h_x: Vec<bf16> = h_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let h_w: Vec<bf16> = h_w_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let c_stride = 4usize;
    let c_extent = 3usize;
    let expected: Vec<f32> = (0..n_total)
        .map(|i| {
            let c = (i as usize / c_stride) % c_extent;
            let x = h_x[i].to_f32();
            if x > 0.0 { x } else { h_w[c].to_f32() * x }
        })
        .collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &h_x).unwrap();
    let dev_w = DeviceBuffer::from_slice(&ctx, &h_w).unwrap();
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, n_total).unwrap();
    let desc = PReluDescriptor {
        input_shape: shape,
        channel_axis: 1,
        element: ElementKind::Bf16,
    };
    let plan = PReluPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        PReluArgs {
            x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            weight: TensorRef { data: dev_w.as_slice(), shape: [3], stride: [1] },
            y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![bf16::ZERO; n_total];
    dev_y.copy_to_host(&mut got).unwrap();
    for i in 0..n_total {
        let g = got[i].to_f32();
        let e = expected[i];
        let tol = e.abs().max(1.0) * 8.0 * BF16_EPS + 1e-2;
        assert!((g - e).abs() <= tol, "PReLU bf16 per-chan @{i}: got={} want={}", g, e);
    }
}

#[test]
#[ignore]
fn prelu_bf16_scalar() {
    let (ctx, stream) = setup();
    let shape = [4i32, 5];
    let n_total = 4 * 5;
    let h_x_f32: Vec<f32> = (0..n_total).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let h_x: Vec<bf16> = h_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let h_w: Vec<bf16> = vec![bf16::from_f32(0.3)];
    let expected: Vec<f32> = h_x.iter()
        .map(|x| {
            let xv = x.to_f32();
            if xv > 0.0 { xv } else { h_w[0].to_f32() * xv }
        })
        .collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &h_x).unwrap();
    let dev_w = DeviceBuffer::from_slice(&ctx, &h_w).unwrap();
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, n_total).unwrap();
    let desc = PReluDescriptor {
        input_shape: shape,
        channel_axis: -1,
        element: ElementKind::Bf16,
    };
    let plan = PReluPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        PReluArgs {
            x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            weight: TensorRef { data: dev_w.as_slice(), shape: [1], stride: [1] },
            y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![bf16::ZERO; n_total];
    dev_y.copy_to_host(&mut got).unwrap();
    for i in 0..n_total {
        let g = got[i].to_f32();
        let e = expected[i];
        let tol = e.abs().max(1.0) * 8.0 * BF16_EPS + 1e-2;
        assert!((g - e).abs() <= tol, "PReLU bf16 scalar @{i}: got={} want={}", g, e);
    }
}
