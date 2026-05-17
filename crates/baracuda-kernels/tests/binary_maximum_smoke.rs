//! Real-GPU smoke test for `BinaryPlan + BinaryKind::Maximum` —
//! `y = max(a, b)` with NaN-propagating semantics (matches
//! `torch.maximum`). Any NaN input → NaN output. Bit-exact across
//! all dtypes.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test binary_maximum_smoke -- --ignored`.

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

// NaN-propagating max reference: any NaN input produces NaN output.
fn maximum_ref_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() {
        a
    } else if b.is_nan() {
        b
    } else if a > b {
        a
    } else {
        b
    }
}

fn maximum_ref_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        a
    } else if b.is_nan() {
        b
    } else if a > b {
        a
    } else {
        b
    }
}

#[test]
#[ignore]
fn maximum_f32_contig() {
    let (ctx, stream) = setup();
    let shape = [4i32, 64];
    let numel: usize = (shape[0] * shape[1]) as usize;
    let mut host_a: Vec<f32> = (0..numel).map(|i| -1.5 + (i as f32) * 0.05).collect();
    let mut host_b: Vec<f32> = (0..numel).map(|i| 1.0 - (i as f32) * 0.04).collect();
    // NaN sprinkle: NaN in a → NaN out; NaN in b → NaN out; both NaN → NaN.
    host_a[5] = f32::NAN;
    host_b[10] = f32::NAN;
    host_a[20] = f32::NAN;
    host_b[20] = f32::NAN;
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);

    let plan = BinaryPlan::<f32, 2>::select(
        &stream,
        &BinaryDescriptor { kind: BinaryKind::Maximum, shape, element: ElementKind::F32 },
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
    for (i, ((&a, &b), &g)) in host_a.iter().zip(host_b.iter()).zip(got.iter()).enumerate() {
        let want = maximum_ref_f32(a, b);
        if want.is_nan() {
            assert!(g.is_nan(), "f32 maximum @ {i}: a={a} b={b} got={g} want=NaN");
        } else {
            assert_eq!(g.to_bits(), want.to_bits(), "f32 maximum @ {i}: a={a} b={b} got={g} want={want}");
        }
    }
}

#[test]
#[ignore]
fn maximum_f64_contig() {
    let (ctx, stream) = setup();
    let shape = [3i32, 32];
    let numel: usize = (shape[0] * shape[1]) as usize;
    let mut host_a: Vec<f64> = (0..numel).map(|i| -1.5 + (i as f64) * 0.05).collect();
    let mut host_b: Vec<f64> = (0..numel).map(|i| 1.0 - (i as f64) * 0.04).collect();
    host_a[3] = f64::NAN;
    host_b[7] = f64::NAN;
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let plan = BinaryPlan::<f64, 2>::select(
        &stream,
        &BinaryDescriptor { kind: BinaryKind::Maximum, shape, element: ElementKind::F64 },
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
    for (i, ((&a, &b), &g)) in host_a.iter().zip(host_b.iter()).zip(got.iter()).enumerate() {
        let want = maximum_ref_f64(a, b);
        if want.is_nan() {
            assert!(g.is_nan(), "f64 maximum @ {i}: a={a} b={b} got={g} want=NaN");
        } else {
            assert_eq!(g.to_bits(), want.to_bits(), "f64 maximum @ {i}: a={a} b={b} got={g} want={want}");
        }
    }
}

#[test]
#[ignore]
fn maximum_f16_contig() {
    let (ctx, stream) = setup();
    let shape = [4i32, 32];
    let numel: usize = (shape[0] * shape[1]) as usize;
    let mut host_a: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32(-1.0 + (i as f32) * 0.02))
        .collect();
    let mut host_b: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32(0.5 - (i as f32) * 0.015))
        .collect();
    host_a[5] = f16::NAN;
    host_b[11] = f16::NAN;
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let plan = BinaryPlan::<f16, 2>::select(
        &stream,
        &BinaryDescriptor { kind: BinaryKind::Maximum, shape, element: ElementKind::F16 },
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
    for (i, ((&a, &b), &g)) in host_a.iter().zip(host_b.iter()).zip(got.iter()).enumerate() {
        let want_f = maximum_ref_f32(a.to_f32(), b.to_f32());
        if want_f.is_nan() {
            assert!(g.to_f32().is_nan(), "f16 maximum @ {i}: a={a} b={b} got={g} want=NaN");
        } else {
            let want = f16::from_f32(want_f);
            assert_eq!(g.to_bits(), want.to_bits(), "f16 maximum @ {i}: a={a} b={b} got={g} want={want}");
        }
    }
}

#[test]
#[ignore]
fn maximum_bf16_contig() {
    let (ctx, stream) = setup();
    let shape = [3i32, 16];
    let numel: usize = (shape[0] * shape[1]) as usize;
    let mut host_a: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32(-1.0 + (i as f32) * 0.05))
        .collect();
    let mut host_b: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32(0.5 - (i as f32) * 0.025))
        .collect();
    host_a[3] = bf16::NAN;
    host_b[9] = bf16::NAN;
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let plan = BinaryPlan::<bf16, 2>::select(
        &stream,
        &BinaryDescriptor { kind: BinaryKind::Maximum, shape, element: ElementKind::Bf16 },
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
    for (i, ((&a, &b), &g)) in host_a.iter().zip(host_b.iter()).zip(got.iter()).enumerate() {
        let want_f = maximum_ref_f32(a.to_f32(), b.to_f32());
        if want_f.is_nan() {
            assert!(g.to_f32().is_nan(), "bf16 maximum @ {i}: a={a} b={b} got={g} want=NaN");
        } else {
            let want = bf16::from_f32(want_f);
            assert_eq!(g.to_bits(), want.to_bits(), "bf16 maximum @ {i}: a={a} b={b} got={g} want={want}");
        }
    }
}
