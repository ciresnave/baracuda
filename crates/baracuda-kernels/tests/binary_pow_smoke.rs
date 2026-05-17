//! Real-GPU smoke test for `BinaryPlan + BinaryKind::Pow` — `y = a^b`.
//! Transcendental — relative-eps compare per dtype.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test binary_pow_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BinaryArgs, BinaryDescriptor, BinaryKind, BinaryPlan, ElementKind,
    PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// Restrict inputs to `a > 0` so the formulas are unambiguous across all
// dtypes (no NaN / inf footguns from negative-base non-integer-exponent
// territory).

#[test]
#[ignore]
fn pow_f32_contig() {
    let (ctx, stream) = setup();
    let shape = [4i32, 64];
    let numel: usize = (shape[0] * shape[1]) as usize;
    let host_a: Vec<f32> = (0..numel).map(|i| 0.1 + (i as f32) * 0.05).collect();
    let host_b: Vec<f32> = (0..numel).map(|i| -2.0 + (i as f32) * 0.025).collect();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);

    let plan = BinaryPlan::<f32, 2>::select(
        &stream,
        &BinaryDescriptor { kind: BinaryKind::Pow, shape, element: ElementKind::F32 },
        PlanPreference::default(),
    ).expect("select");
    plan.run(&stream, Workspace::None, BinaryArgs {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    }).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    let eps = 4.0 * f32::EPSILON;
    for (i, ((&a, &b), &g)) in host_a.iter().zip(host_b.iter()).zip(got.iter()).enumerate() {
        let want = a.powf(b);
        let tol = (want.abs() * eps).max(eps);
        assert!((g - want).abs() <= tol, "f32 pow @ {i}: a={a} b={b} got={g} want={want}");
    }
}

#[test]
#[ignore]
fn pow_f64_contig() {
    let (ctx, stream) = setup();
    let shape = [3i32, 32];
    let numel: usize = (shape[0] * shape[1]) as usize;
    let host_a: Vec<f64> = (0..numel).map(|i| 0.1 + (i as f64) * 0.05).collect();
    let host_b: Vec<f64> = (0..numel).map(|i| -2.0 + (i as f64) * 0.025).collect();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let plan = BinaryPlan::<f64, 2>::select(
        &stream,
        &BinaryDescriptor { kind: BinaryKind::Pow, shape, element: ElementKind::F64 },
        PlanPreference::default(),
    ).expect("select");
    plan.run(&stream, Workspace::None, BinaryArgs {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    }).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f64; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    let eps = 4.0 * f64::EPSILON;
    for (i, ((&a, &b), &g)) in host_a.iter().zip(host_b.iter()).zip(got.iter()).enumerate() {
        let want = a.powf(b);
        let tol = (want.abs() * eps).max(eps);
        assert!((g - want).abs() <= tol, "f64 pow @ {i}: a={a} b={b} got={g} want={want}");
    }
}

#[test]
#[ignore]
fn pow_f16_contig() {
    let (ctx, stream) = setup();
    let shape = [4i32, 32];
    let numel: usize = (shape[0] * shape[1]) as usize;
    // f16 has very limited range — keep things modest.
    let host_a: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32(0.5 + (i as f32) * 0.02))
        .collect();
    let host_b: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32(-1.0 + (i as f32) * 0.01))
        .collect();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let plan = BinaryPlan::<f16, 2>::select(
        &stream,
        &BinaryDescriptor { kind: BinaryKind::Pow, shape, element: ElementKind::F16 },
        PlanPreference::default(),
    ).expect("select");
    plan.run(&stream, Workspace::None, BinaryArgs {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    }).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![f16::ZERO; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    let eps = 9.77e-4_f32; // f16 1 ULP
    for (i, ((&a, &b), &g)) in host_a.iter().zip(host_b.iter()).zip(got.iter()).enumerate() {
        let want = f16::from_f32(a.to_f32().powf(b.to_f32()));
        let diff = (g.to_f32() - want.to_f32()).abs();
        let tol = (want.to_f32().abs() * eps).max(eps);
        assert!(diff <= tol, "f16 pow @ {i}: got={g} want={want} diff={diff}");
    }
}

#[test]
#[ignore]
fn pow_bf16_contig() {
    let (ctx, stream) = setup();
    let shape = [3i32, 16];
    let numel: usize = (shape[0] * shape[1]) as usize;
    let host_a: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32(0.5 + (i as f32) * 0.05))
        .collect();
    let host_b: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32(-1.0 + (i as f32) * 0.025))
        .collect();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let plan = BinaryPlan::<bf16, 2>::select(
        &stream,
        &BinaryDescriptor { kind: BinaryKind::Pow, shape, element: ElementKind::Bf16 },
        PlanPreference::default(),
    ).expect("select");
    plan.run(&stream, Workspace::None, BinaryArgs {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    }).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![bf16::ZERO; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    let eps = 7.81e-3_f32; // bf16 1 ULP
    for (i, ((&a, &b), &g)) in host_a.iter().zip(host_b.iter()).zip(got.iter()).enumerate() {
        let want = bf16::from_f32(a.to_f32().powf(b.to_f32()));
        let diff = (g.to_f32() - want.to_f32()).abs();
        let tol = (want.to_f32().abs() * eps).max(eps);
        assert!(diff <= tol, "bf16 pow @ {i}: got={g} want={want} diff={diff}");
    }
}
