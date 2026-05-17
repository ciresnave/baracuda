//! Real-GPU smoke test for `addcdiv` — `y = a + scale * (b / c)`.
//!
//! Covers all 4 FP dtypes contig. f32 / f64 bit-exact (unfused
//! `__fdiv_rn` matches Rust's `b / c`); f16 / bf16 with `4 * dtype_eps`
//! relative tolerance.
//!
//! Inputs keep `|c| >= 0.5` to avoid divide-by-zero / catastrophic-
//! cancellation territory.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef, TernaryArgs,
    TernaryDescriptor, TernaryKind, TernaryPlan, Workspace,
};
use half::{bf16, f16};

const F16_EPS: f32 = 9.77e-4_f32;
const BF16_EPS: f32 = 7.81e-3_f32;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn addcdiv_f32_contig() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let scale: f32 = 0.5;

    let host_a: Vec<f32> = (0..numel).map(|i| (i as f32 % 100.0) * 0.05 - 2.5).collect();
    let host_b: Vec<f32> = (0..numel).map(|i| (i as f32 % 100.0) * 0.05 - 1.0).collect();
    let host_c: Vec<f32> = (0..numel).map(|i| (i as f32 % 100.0) * 0.05 + 0.6).collect(); // ≥ 0.6
    let expected: Vec<f32> = (0..numel)
        .map(|i| {
            let t = host_b[i] / host_c[i];
            let t = scale * t;
            host_a[i] + t
        })
        .collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let dev_c = DeviceBuffer::from_slice(&ctx, &host_c).expect("upload c");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = TernaryDescriptor {
        kind: TernaryKind::Addcdiv,
        shape,
        element: ElementKind::F32,
        scale,
    };
    let plan = TernaryPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = TernaryArgs::<f32, 3> {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        c: TensorRef { data: dev_c.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "addcdiv f32 mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn addcdiv_f64_contig() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let scale: f32 = 0.5;

    let host_a: Vec<f64> = (0..numel).map(|i| (i as f64 % 100.0) * 0.05 - 2.5).collect();
    let host_b: Vec<f64> = (0..numel).map(|i| (i as f64 % 100.0) * 0.05 - 1.0).collect();
    let host_c: Vec<f64> = (0..numel).map(|i| (i as f64 % 100.0) * 0.05 + 0.6).collect();
    let scale_d = scale as f64;
    let expected: Vec<f64> = (0..numel)
        .map(|i| {
            let t = host_b[i] / host_c[i];
            let t = scale_d * t;
            host_a[i] + t
        })
        .collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let dev_c = DeviceBuffer::from_slice(&ctx, &host_c).expect("upload c");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = TernaryDescriptor {
        kind: TernaryKind::Addcdiv,
        shape,
        element: ElementKind::F64,
        scale,
    };
    let plan = TernaryPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = TernaryArgs::<f64, 3> {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        c: TensorRef { data: dev_c.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "addcdiv f64 mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn addcdiv_f16_contig() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let scale: f32 = 0.5;

    let host_a: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32((i as f32 % 80.0) * 0.05 - 2.0))
        .collect();
    let host_b: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32((i as f32 % 80.0) * 0.04 - 1.0))
        .collect();
    let host_c: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32((i as f32 % 80.0) * 0.04 + 0.75))
        .collect();
    let expected: Vec<f16> = (0..numel)
        .map(|i| {
            let a = host_a[i].to_f32();
            let b = host_b[i].to_f32();
            let c = host_c[i].to_f32();
            let t = b / c;
            let t = scale * t;
            f16::from_f32(a + t)
        })
        .collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let dev_c = DeviceBuffer::from_slice(&ctx, &host_c).expect("upload c");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = TernaryDescriptor {
        kind: TernaryKind::Addcdiv,
        shape,
        element: ElementKind::F16,
        scale,
    };
    let plan = TernaryPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = TernaryArgs::<f16, 3> {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        c: TensorRef { data: dev_c.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![f16::ZERO; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g.to_f32() - e.to_f32()).abs();
        let allow = e.to_f32().abs().max(1.0) * 4.0 * F16_EPS;
        assert!(
            diff <= allow,
            "addcdiv f16 mismatch @ {i}: got {g} expected {e} (diff {diff} > allow {allow})"
        );
    }
}

#[test]
#[ignore]
fn addcdiv_bf16_contig() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let scale: f32 = 0.5;

    let host_a: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32((i as f32 % 80.0) * 0.05 - 2.0))
        .collect();
    let host_b: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32((i as f32 % 80.0) * 0.04 - 1.0))
        .collect();
    let host_c: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32((i as f32 % 80.0) * 0.04 + 0.75))
        .collect();
    let expected: Vec<bf16> = (0..numel)
        .map(|i| {
            let a = host_a[i].to_f32();
            let b = host_b[i].to_f32();
            let c = host_c[i].to_f32();
            let t = b / c;
            let t = scale * t;
            bf16::from_f32(a + t)
        })
        .collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let dev_c = DeviceBuffer::from_slice(&ctx, &host_c).expect("upload c");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = TernaryDescriptor {
        kind: TernaryKind::Addcdiv,
        shape,
        element: ElementKind::Bf16,
        scale,
    };
    let plan = TernaryPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = TernaryArgs::<bf16, 3> {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        c: TensorRef { data: dev_c.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![bf16::ZERO; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g.to_f32() - e.to_f32()).abs();
        let allow = e.to_f32().abs().max(1.0) * 4.0 * BF16_EPS;
        assert!(
            diff <= allow,
            "addcdiv bf16 mismatch @ {i}: got {g} expected {e} (diff {diff} > allow {allow})"
        );
    }
}
