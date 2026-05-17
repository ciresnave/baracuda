//! Real-GPU smoke test for `RandomPlan + RandomKind::Normal` over
//! cuRAND. Verifies sample mean / stddev match the requested parameters
//! within a generous statistical tolerance.

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
fn normal_f32_standard() {
    let (ctx, stream) = setup();
    let shape = [N as i32];
    let stride = contiguous_stride(shape);

    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, N).expect("alloc y");

    let desc = RandomDescriptor {
        kind: RandomKind::Normal,
        shape,
        element: ElementKind::F32,
        param1: 0.0, // mean
        param2: 1.0, // stddev
        seed: 0x4242_4242_DEAD_BEEF,
    };
    let plan = RandomPlan::<f32, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = RandomArgs::<f32, 1> {
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("normal run");
    stream.synchronize().expect("sync");

    let mut host = vec![0f32; N];
    dev_y.copy_to_host(&mut host).expect("download");

    let mut sum = 0f64;
    for &v in &host {
        assert!(v.is_finite(), "normal sample not finite: {v}");
        sum += v as f64;
    }
    let mean = sum / (N as f64);
    let mut var_acc = 0f64;
    for &v in &host {
        var_acc += ((v as f64) - mean).powi(2);
    }
    let var = var_acc / (N as f64);
    let std = var.sqrt();

    // For N(0, 1) with N = 1M samples:
    //   stderr(mean) = 1/sqrt(N) ≈ 1e-3  → 3-stderr ≈ 3e-3
    //   stderr(std)  ≈ 1/sqrt(2N) ≈ 7e-4 → 3-stderr ≈ 2e-3
    assert!(mean.abs() < 5e-3, "normal sample mean = {mean}");
    assert!((std - 1.0).abs() < 5e-3, "normal sample std = {std}");
}

#[test]
#[ignore]
fn normal_f32_shifted() {
    let (ctx, stream) = setup();
    let shape = [N as i32];
    let stride = contiguous_stride(shape);

    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, N).expect("alloc y");

    let desc = RandomDescriptor {
        kind: RandomKind::Normal,
        shape,
        element: ElementKind::F32,
        param1: 5.0,  // mean
        param2: 2.5,  // stddev
        seed: 0xABCD_1234_BEEF_9999,
    };
    let plan = RandomPlan::<f32, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = RandomArgs::<f32, 1> {
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("normal shifted run");
    stream.synchronize().expect("sync");

    let mut host = vec![0f32; N];
    dev_y.copy_to_host(&mut host).expect("download");

    let mut sum = 0f64;
    for &v in &host {
        sum += v as f64;
    }
    let mean = sum / (N as f64);
    let mut var_acc = 0f64;
    for &v in &host {
        var_acc += ((v as f64) - mean).powi(2);
    }
    let std = (var_acc / (N as f64)).sqrt();

    // For N(5, 2.5): stderr(mean) = 2.5/sqrt(N) ≈ 2.5e-3; 3-stderr ≈ 8e-3.
    assert!((mean - 5.0).abs() < 2e-2, "normal shifted mean = {mean}");
    assert!((std - 2.5).abs() < 2e-2, "normal shifted std = {std}");
}

#[test]
#[ignore]
fn normal_f64_standard() {
    let (ctx, stream) = setup();
    let shape = [N as i32];
    let stride = contiguous_stride(shape);

    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, N).expect("alloc y");

    let desc = RandomDescriptor {
        kind: RandomKind::Normal,
        shape,
        element: ElementKind::F64,
        param1: 0.0,
        param2: 1.0,
        seed: 0x9876_5432_1010_0101,
    };
    let plan = RandomPlan::<f64, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = RandomArgs::<f64, 1> {
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("normal f64 run");
    stream.synchronize().expect("sync");

    let mut host = vec![0f64; N];
    dev_y.copy_to_host(&mut host).expect("download");

    let mut sum = 0f64;
    for &v in &host {
        assert!(v.is_finite());
        sum += v;
    }
    let mean = sum / (N as f64);
    let mut var_acc = 0f64;
    for &v in &host {
        var_acc += (v - mean).powi(2);
    }
    let std = (var_acc / (N as f64)).sqrt();

    assert!(mean.abs() < 5e-3, "normal f64 mean = {mean}");
    assert!((std - 1.0).abs() < 5e-3, "normal f64 std = {std}");
}
