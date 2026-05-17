//! Real-GPU smoke test for `BinaryParamPlan + BinaryKind::Lerp`.
//!
//! Forward: `y = a + weight·(b - a)`. Tolerance: `4·eps` relative — the
//! kernel does one subtract, one multiply, one add (3 roundings); plus
//! f16/bf16 round once on the store.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BinaryKind, BinaryParamArgs, BinaryParamDescriptor, BinaryParamPlan,
    ElementKind, PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

const F16_EPS: f32 = 9.77e-4;
const BF16_EPS: f32 = 7.81e-3;

const W: f32 = 0.3;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn lerp_f32_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_a: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.5 - 17.25).collect();
    let host_b: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.125 + 3.75).collect();
    let host_expected: Vec<f32> = host_a
        .iter()
        .zip(host_b.iter())
        .map(|(a, b)| a + W * (b - a))
        .collect();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let desc = BinaryParamDescriptor {
        kind: BinaryKind::Lerp,
        shape,
        element: ElementKind::F32,
        param: W,
    };
    let plan = BinaryParamPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = BinaryParamArgs::<f32, 3> {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(host_expected.iter()).enumerate() {
        let diff = (g - e).abs();
        let tol = e.abs().max(1.0) * 4.0 * f32::EPSILON;
        assert!(
            diff <= tol,
            "lerp f32 @ {i}: got {g} exp {e} (diff {diff} > tol {tol})"
        );
    }
}

#[test]
#[ignore]
fn lerp_f64_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_a: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.5 - 17.25).collect();
    let host_b: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.125 + 3.75).collect();
    let host_expected: Vec<f64> = host_a
        .iter()
        .zip(host_b.iter())
        .map(|(a, b)| a + (W as f64) * (b - a))
        .collect();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let desc = BinaryParamDescriptor {
        kind: BinaryKind::Lerp,
        shape,
        element: ElementKind::F64,
        param: W,
    };
    let plan = BinaryParamPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = BinaryParamArgs::<f64, 3> {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f64; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(host_expected.iter()).enumerate() {
        let diff = (g - e).abs();
        let tol = e.abs().max(1.0) * 4.0 * f64::EPSILON;
        assert!(
            diff <= tol,
            "lerp f64 @ {i}: got {g} exp {e} (diff {diff} > tol {tol})"
        );
    }
}

#[test]
#[ignore]
fn lerp_f16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_a: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32(((i % 41) as f32) * 0.5 - 10.0))
        .collect();
    let host_b: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32(((i % 37) as f32) * 0.25 - 4.5))
        .collect();
    let host_expected: Vec<f16> = host_a
        .iter()
        .zip(host_b.iter())
        .map(|(a, b)| {
            let fa = a.to_f32();
            let fb = b.to_f32();
            f16::from_f32(fa + W * (fb - fa))
        })
        .collect();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let desc = BinaryParamDescriptor {
        kind: BinaryKind::Lerp,
        shape,
        element: ElementKind::F16,
        param: W,
    };
    let plan = BinaryParamPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = BinaryParamArgs::<f16, 3> {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![f16::from_f32(0.0); numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(host_expected.iter()).enumerate() {
        let gf = g.to_f32();
        let ef = e.to_f32();
        let diff = (gf - ef).abs();
        let tol = ef.abs().max(1.0) * 4.0 * F16_EPS;
        assert!(
            diff <= tol,
            "lerp f16 @ {i}: got {gf} exp {ef} (diff {diff} > tol {tol})"
        );
    }
}

#[test]
#[ignore]
fn lerp_bf16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_a: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32(((i % 41) as f32) * 0.5 - 10.0))
        .collect();
    let host_b: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32(((i % 37) as f32) * 0.25 - 4.5))
        .collect();
    let host_expected: Vec<bf16> = host_a
        .iter()
        .zip(host_b.iter())
        .map(|(a, b)| {
            let fa = a.to_f32();
            let fb = b.to_f32();
            bf16::from_f32(fa + W * (fb - fa))
        })
        .collect();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let desc = BinaryParamDescriptor {
        kind: BinaryKind::Lerp,
        shape,
        element: ElementKind::Bf16,
        param: W,
    };
    let plan = BinaryParamPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = BinaryParamArgs::<bf16, 3> {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![bf16::from_f32(0.0); numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(host_expected.iter()).enumerate() {
        let gf = g.to_f32();
        let ef = e.to_f32();
        let diff = (gf - ef).abs();
        let tol = ef.abs().max(1.0) * 4.0 * BF16_EPS;
        assert!(
            diff <= tol,
            "lerp bf16 @ {i}: got {gf} exp {ef} (diff {diff} > tol {tol})"
        );
    }
}
