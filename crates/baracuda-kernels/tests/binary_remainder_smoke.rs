//! Real-GPU smoke test for `BinaryPlan + BinaryKind::Remainder` —
//! C-style remainder via `fmod`. Result sign matches `a`. Distinct from
//! `BinaryKind::Mod` (Python-style, sign of b). Inputs avoid `b == 0`.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test binary_remainder_smoke -- --ignored`.

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

#[test]
#[ignore]
fn remainder_f32_contig() {
    let (ctx, stream) = setup();
    let shape = [4i32, 64];
    let numel: usize = (shape[0] * shape[1]) as usize;
    let host_a: Vec<f32> = (0..numel).map(|i| -5.0 + (i as f32) * 0.13).collect();
    let host_b: Vec<f32> = (0..numel)
        .map(|i| {
            let v = if i % 3 == 0 { -0.7 } else { 0.6 } + (i as f32) * 0.011;
            if v.abs() < 0.05 { 0.5 } else { v }
        })
        .collect();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);

    let plan = BinaryPlan::<f32, 2>::select(
        &stream,
        &BinaryDescriptor { kind: BinaryKind::Remainder, shape, element: ElementKind::F32 },
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
    // Rust f32 `%` is C-style fmod, matching libdevice `fmodf` for finite
    // operands. Should be bit-exact.
    for (i, ((&a, &b), &g)) in host_a.iter().zip(host_b.iter()).zip(got.iter()).enumerate() {
        let want = a % b;
        assert_eq!(
            g.to_bits(),
            want.to_bits(),
            "f32 remainder @ {i}: a={a} b={b} got={g} want={want}"
        );
    }
}

#[test]
#[ignore]
fn remainder_f64_contig() {
    let (ctx, stream) = setup();
    let shape = [3i32, 32];
    let numel: usize = (shape[0] * shape[1]) as usize;
    let host_a: Vec<f64> = (0..numel).map(|i| -5.0 + (i as f64) * 0.17).collect();
    let host_b: Vec<f64> = (0..numel)
        .map(|i| {
            let v = if i % 3 == 0 { -0.7 } else { 0.6 } + (i as f64) * 0.013;
            if v.abs() < 0.05 { 0.5 } else { v }
        })
        .collect();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let plan = BinaryPlan::<f64, 2>::select(
        &stream,
        &BinaryDescriptor { kind: BinaryKind::Remainder, shape, element: ElementKind::F64 },
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
        let want = a % b;
        assert_eq!(
            g.to_bits(),
            want.to_bits(),
            "f64 remainder @ {i}: a={a} b={b} got={g} want={want}"
        );
    }
}

#[test]
#[ignore]
fn remainder_f16_contig() {
    let (ctx, stream) = setup();
    let shape = [4i32, 32];
    let numel: usize = (shape[0] * shape[1]) as usize;
    let host_a: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32(-2.0 + (i as f32) * 0.08))
        .collect();
    let host_b: Vec<f16> = (0..numel)
        .map(|i| {
            let v = if i % 3 == 0 { -0.7 } else { 0.6 } + (i as f32) * 0.011;
            f16::from_f32(if v.abs() < 0.05 { 0.5 } else { v })
        })
        .collect();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let plan = BinaryPlan::<f16, 2>::select(
        &stream,
        &BinaryDescriptor { kind: BinaryKind::Remainder, shape, element: ElementKind::F16 },
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
        // Replicate the f32-detour: compute in f32 then round to f16.
        let want = f16::from_f32(a.to_f32() % b.to_f32());
        assert_eq!(
            g.to_bits(),
            want.to_bits(),
            "f16 remainder @ {i}: a={a} b={b} got={g} want={want}"
        );
    }
}

#[test]
#[ignore]
fn remainder_bf16_contig() {
    let (ctx, stream) = setup();
    let shape = [3i32, 16];
    let numel: usize = (shape[0] * shape[1]) as usize;
    let host_a: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32(-2.0 + (i as f32) * 0.1))
        .collect();
    let host_b: Vec<bf16> = (0..numel)
        .map(|i| {
            let v = if i % 3 == 0 { -0.7 } else { 0.6 } + (i as f32) * 0.013;
            bf16::from_f32(if v.abs() < 0.05 { 0.5 } else { v })
        })
        .collect();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let plan = BinaryPlan::<bf16, 2>::select(
        &stream,
        &BinaryDescriptor { kind: BinaryKind::Remainder, shape, element: ElementKind::Bf16 },
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
        let want = bf16::from_f32(a.to_f32() % b.to_f32());
        assert_eq!(
            g.to_bits(),
            want.to_bits(),
            "bf16 remainder @ {i}: a={a} b={b} got={g} want={want}"
        );
    }
}
