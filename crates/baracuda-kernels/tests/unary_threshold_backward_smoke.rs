//! Real-GPU smoke test for `UnaryParamBackwardPlan + UnaryKind::Threshold`.
//!
//! Forward: `y = (x > t) ? x : v`. Backward: `dx = (x > t) ? dy : 0`.
//! Saved-x. Like the forward, the BW is a pure compare + select, so the
//! result is bit-exact against the CPU reference.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef, UnaryKind,
    UnaryParamBackwardArgs, UnaryParamBackwardDescriptor, UnaryParamBackwardPlan, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

const T: f32 = 0.5;
const V: f32 = -1.25; // ignored by BW

#[test]
#[ignore]
fn threshold_backward_f32_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.01 - 5.0).collect();
    let host_dy: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.5 - 17.25).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");
    let stride = contiguous_stride(shape);
    let desc = UnaryParamBackwardDescriptor {
        kind: UnaryKind::Threshold,
        shape,
        element: ElementKind::F32,
        params: [T, V],
    };
    let plan = UnaryParamBackwardPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = UnaryParamBackwardArgs::<f32, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for i in 0..numel {
        let exp = if host_x[i] > T { host_dy[i] } else { 0.0 };
        assert_eq!(
            got[i].to_bits(), exp.to_bits(),
            "threshold bw f32 @ {i}: x={}, dy={}, got {}, exp {}",
            host_x[i], host_dy[i], got[i], exp
        );
    }
}

#[test]
#[ignore]
fn threshold_backward_f64_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.01 - 5.0).collect();
    let host_dy: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.5 - 17.25).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");
    let stride = contiguous_stride(shape);
    let desc = UnaryParamBackwardDescriptor {
        kind: UnaryKind::Threshold,
        shape,
        element: ElementKind::F64,
        params: [T, V],
    };
    let plan = UnaryParamBackwardPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = UnaryParamBackwardArgs::<f64, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f64; numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for i in 0..numel {
        let exp = if host_x[i] > T as f64 { host_dy[i] } else { 0.0 };
        assert_eq!(
            got[i].to_bits(), exp.to_bits(),
            "threshold bw f64 @ {i}: got {}, exp {}", got[i], exp
        );
    }
}

#[test]
#[ignore]
fn threshold_backward_f16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32((i as f32) * 0.01 - 5.0))
        .collect();
    let host_dy: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32(((i % 81) as f32) * 0.25 - 10.0))
        .collect();
    let zero_h = f16::from_f32(0.0);
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");
    let stride = contiguous_stride(shape);
    let desc = UnaryParamBackwardDescriptor {
        kind: UnaryKind::Threshold,
        shape,
        element: ElementKind::F16,
        params: [T, V],
    };
    let plan = UnaryParamBackwardPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = UnaryParamBackwardArgs::<f16, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![f16::from_f32(0.0); numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for i in 0..numel {
        let exp = if host_x[i].to_f32() > T { host_dy[i] } else { zero_h };
        assert_eq!(
            got[i].to_bits(), exp.to_bits(),
            "threshold bw f16 @ {i}: got {} exp {}", got[i].to_f32(), exp.to_f32()
        );
    }
}

#[test]
#[ignore]
fn threshold_backward_bf16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32((i as f32) * 0.01 - 5.0))
        .collect();
    let host_dy: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32(((i % 81) as f32) * 0.25 - 10.0))
        .collect();
    let zero_h = bf16::from_f32(0.0);
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");
    let stride = contiguous_stride(shape);
    let desc = UnaryParamBackwardDescriptor {
        kind: UnaryKind::Threshold,
        shape,
        element: ElementKind::Bf16,
        params: [T, V],
    };
    let plan = UnaryParamBackwardPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = UnaryParamBackwardArgs::<bf16, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![bf16::from_f32(0.0); numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for i in 0..numel {
        let exp = if host_x[i].to_f32() > T { host_dy[i] } else { zero_h };
        assert_eq!(
            got[i].to_bits(), exp.to_bits(),
            "threshold bw bf16 @ {i}: got {} exp {}", got[i].to_f32(), exp.to_f32()
        );
    }
}
