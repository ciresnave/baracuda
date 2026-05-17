//! Real-GPU smoke test for `BinaryPlan + BinaryKind::Mod` — Python-style
//! modulo. Result sign matches `b`. Inputs avoid `b == 0`.
//!
//! Reference: implement via `fmod` then sign-fix, matching the kernel's
//! exact arithmetic so the result is bit-exact for f32 / f64.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test binary_mod_smoke -- --ignored`.

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

// Python-style mod reference: r = fmod(a, b); if r != 0 and sign(r) !=
// sign(b), add b. Matches the kernel's exact arithmetic.
fn python_mod_f32(a: f32, b: f32) -> f32 {
    let mut r = a % b; // Rust `%` is C-style fmod for f32.
    if r != 0.0 && (r < 0.0) != (b < 0.0) {
        r += b;
    }
    r
}

fn python_mod_f64(a: f64, b: f64) -> f64 {
    let mut r = a % b;
    if r != 0.0 && (r < 0.0) != (b < 0.0) {
        r += b;
    }
    r
}

#[test]
#[ignore]
fn mod_f32_contig() {
    let (ctx, stream) = setup();
    let shape = [4i32, 64];
    let numel: usize = (shape[0] * shape[1]) as usize;
    // Mix of signs so we exercise the sign-fix branch.
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
        &BinaryDescriptor { kind: BinaryKind::Mod, shape, element: ElementKind::F32 },
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
    let tol = 4.0 * f32::EPSILON;
    for (i, ((&a, &b), &g)) in host_a.iter().zip(host_b.iter()).zip(got.iter()).enumerate() {
        let want = python_mod_f32(a, b);
        let diff = (g - want).abs();
        let bound = tol * b.abs().max(1.0);
        assert!(
            diff <= bound,
            "f32 mod @ {i}: a={a} b={b} got={g} want={want} |diff|={diff} bound={bound}"
        );
    }
}

#[test]
#[ignore]
fn mod_f64_contig() {
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
        &BinaryDescriptor { kind: BinaryKind::Mod, shape, element: ElementKind::F64 },
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
    let tol = 4.0 * f64::EPSILON;
    for (i, ((&a, &b), &g)) in host_a.iter().zip(host_b.iter()).zip(got.iter()).enumerate() {
        let want = python_mod_f64(a, b);
        let diff = (g - want).abs();
        let bound = tol * b.abs().max(1.0);
        assert!(
            diff <= bound,
            "f64 mod @ {i}: a={a} b={b} got={g} want={want} |diff|={diff} bound={bound}"
        );
    }
}

#[test]
#[ignore]
fn mod_f16_contig() {
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
        &BinaryDescriptor { kind: BinaryKind::Mod, shape, element: ElementKind::F16 },
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
        // Replicate the f32-detour exactly: compute in f32 then round.
        let want = f16::from_f32(python_mod_f32(a.to_f32(), b.to_f32()));
        let diff = (g.to_f32() - want.to_f32()).abs();
        // f16 quantum is ~1e-3; allow 2 ulps in f16 (≈2 * b.abs() * 2^-10).
        let bound = 4.0 * (b.to_f32().abs().max(1.0)) / 1024.0;
        assert!(
            diff <= bound,
            "f16 mod @ {i}: a={a} b={b} got={g} want={want} |diff|={diff} bound={bound}"
        );
    }
}

#[test]
#[ignore]
fn mod_bf16_contig() {
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
        &BinaryDescriptor { kind: BinaryKind::Mod, shape, element: ElementKind::Bf16 },
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
        let want = bf16::from_f32(python_mod_f32(a.to_f32(), b.to_f32()));
        let diff = (g.to_f32() - want.to_f32()).abs();
        // bf16 quantum ≈ 2^-7; allow ~2 ulp.
        let bound = 4.0 * (b.to_f32().abs().max(1.0)) / 128.0;
        assert!(
            diff <= bound,
            "bf16 mod @ {i}: a={a} b={b} got={g} want={want} |diff|={diff} bound={bound}"
        );
    }
}
