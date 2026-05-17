//! Real-GPU smoke test for `RandomPlan + RandomKind::Uniform` over
//! cuRAND. Verifies basic statistical sanity (mean, range) rather than
//! exact bit-equality, because cuRAND is intrinsically stochastic.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test random_uniform_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, RandomArgs, RandomDescriptor, RandomKind,
    RandomPlan, TensorMut, Workspace,
};

const N: usize = 1024 * 1024;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn uniform_f32_unit_interval() {
    let (ctx, stream) = setup();
    let shape = [N as i32];
    let stride = contiguous_stride(shape);

    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, N).expect("alloc y");

    let desc = RandomDescriptor {
        kind: RandomKind::Uniform,
        shape,
        element: ElementKind::F32,
        param1: 0.0,
        param2: 1.0,
        seed: 0xC0FF_EE12_3456_7890,
    };
    let plan = RandomPlan::<f32, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("select RandomPlan<f32>");
    let args = RandomArgs::<f32, 1> {
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape,
            stride,
        },
    };
    plan.run(&stream, Workspace::None, args).expect("uniform f32 run");
    stream.synchronize().expect("sync");

    let mut host = vec![0f32; N];
    dev_y.copy_to_host(&mut host).expect("download");

    // Range check — cuRAND returns (0, 1].
    let (mut min, mut max, mut sum) = (f32::INFINITY, f32::NEG_INFINITY, 0f64);
    for &v in &host {
        assert!(v.is_finite(), "uniform sample is not finite: {v}");
        assert!(v > 0.0 && v <= 1.0, "uniform sample out of range: {v}");
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
        sum += v as f64;
    }
    let mean = sum / (N as f64);
    // For Uniform(0, 1) the population mean is 0.5 and stddev = 1/sqrt(12) ≈ 0.289.
    // 3 stderr of the sample mean at N = 1M is 3 * 0.289 / sqrt(1M) ≈ 8.7e-4.
    // Generous tolerance of 5e-3 to absorb any cuRAND-implementation drift.
    assert!(
        (mean - 0.5).abs() < 5e-3,
        "uniform f32 sample mean = {mean}, expected ~0.5"
    );
    // We expect to span most of the unit interval with N = 1M samples.
    assert!(min < 0.01, "min = {min}, expected close to 0");
    assert!(max > 0.99, "max = {max}, expected close to 1");
}

#[test]
#[ignore]
fn uniform_f64_unit_interval() {
    let (ctx, stream) = setup();
    let shape = [N as i32];
    let stride = contiguous_stride(shape);

    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, N).expect("alloc y");

    let desc = RandomDescriptor {
        kind: RandomKind::Uniform,
        shape,
        element: ElementKind::F64,
        param1: 0.0,
        param2: 1.0,
        seed: 0x1234_5678_9ABC_DEF0,
    };
    let plan = RandomPlan::<f64, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("select RandomPlan<f64>");
    let args = RandomArgs::<f64, 1> {
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape,
            stride,
        },
    };
    plan.run(&stream, Workspace::None, args).expect("uniform f64 run");
    stream.synchronize().expect("sync");

    let mut host = vec![0f64; N];
    dev_y.copy_to_host(&mut host).expect("download");

    let (mut min, mut max, mut sum) = (f64::INFINITY, f64::NEG_INFINITY, 0f64);
    for &v in &host {
        assert!(v.is_finite());
        assert!(v > 0.0 && v <= 1.0);
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
        sum += v;
    }
    let mean = sum / (N as f64);
    assert!(
        (mean - 0.5).abs() < 5e-3,
        "uniform f64 sample mean = {mean}, expected ~0.5"
    );
    assert!(min < 0.01);
    assert!(max > 0.99);
}

#[test]
#[ignore]
fn uniform_f32_affine_range() {
    // Verify the in-place affine remap into Uniform(-3, 7].
    let (ctx, stream) = setup();
    let shape = [N as i32];
    let stride = contiguous_stride(shape);

    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, N).expect("alloc y");

    let desc = RandomDescriptor {
        kind: RandomKind::Uniform,
        shape,
        element: ElementKind::F32,
        param1: -3.0,
        param2: 7.0,
        seed: 0xDEAD_BEEF_0000_1111,
    };
    let plan = RandomPlan::<f32, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = RandomArgs::<f32, 1> {
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape,
            stride,
        },
    };
    plan.run(&stream, Workspace::None, args).expect("uniform affine run");
    stream.synchronize().expect("sync");

    let mut host = vec![0f32; N];
    dev_y.copy_to_host(&mut host).expect("download");

    let (mut min, mut max, mut sum) = (f32::INFINITY, f32::NEG_INFINITY, 0f64);
    for &v in &host {
        assert!(v.is_finite());
        assert!(v > -3.0 && v <= 7.0, "out of range: {v}");
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
        sum += v as f64;
    }
    let mean = sum / (N as f64);
    // Population mean of Uniform(-3, 7) is 2. stderr ≈ 3 * (10/sqrt(12))/sqrt(N) ≈ 8.7e-3.
    assert!(
        (mean - 2.0).abs() < 5e-2,
        "uniform-affine mean = {mean}, expected ~2.0"
    );
    assert!(min < -2.9);
    assert!(max > 6.9);
}
