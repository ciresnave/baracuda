//! Real-GPU smoke test for the Phase 3 fanout binary-sub kernel in
//! `baracuda-kernels-sys`.
//!
//! Covers `BinaryPlan<f32, N> + BinaryKind::Sub` over a contiguous
//! rank-3 shape. The kernel is a pure pointwise `a - b` SIMT kernel
//! (no tensor cores, no warp reduction), so the result is **bit-exact**
//! against the CPU reference — no tolerance.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BinaryArgs, BinaryDescriptor, BinaryKind, BinaryPlan, ElementKind,
    PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

// 1-ULP relative tolerance — see binary_add_smoke.rs for the rationale.
const F16_EPS: f32 = 9.77e-4;
const BF16_EPS: f32 = 7.81e-3;

fn run_case<const N: usize>(shape: [i32; N]) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let numel: usize = shape.iter().map(|&d| d as usize).product();
    assert!(numel > 0, "test shape must have non-zero element count");

    // Deterministic-but-non-pathological inputs. Mix in some half-
    // integer-ish values so a kernel that accidentally rounded would
    // show up.
    let host_a: Vec<f32> = (0..numel)
        .map(|i| (i as f32) * 0.5 - 17.25)
        .collect();
    let host_b: Vec<f32> = (0..numel)
        .map(|i| (i as f32) * 0.125 + 3.75)
        .collect();
    let host_expected: Vec<f32> = host_a
        .iter()
        .zip(host_b.iter())
        .map(|(a, b)| a - b)
        .collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload A");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload B");
    let mut dev_y: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, numel).expect("alloc Y");

    let stride = contiguous_stride(shape);

    let desc = BinaryDescriptor {
        kind: BinaryKind::Sub,
        shape,
        element: ElementKind::F32,
    };
    let plan = BinaryPlan::<f32, N>::select(&stream, &desc, PlanPreference::default())
        .expect("select BinaryPlan<f32, N>");

    let args = BinaryArgs::<f32, N> {
        a: TensorRef {
            data: dev_a.as_slice(),
            shape,
            stride,
        },
        b: TensorRef {
            data: dev_b.as_slice(),
            shape,
            stride,
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape,
            stride,
        },
    };
    plan.run(&stream, Workspace::None, args).expect("binary sub run");
    stream.synchronize().expect("stream sync");

    let mut host_got = vec![0f32; numel];
    dev_y.copy_to_host(&mut host_got).expect("download Y");

    // Bit-exact comparison — the kernel is a pure SIMT a - b sweep
    // with no warp reduction or fma fusion, so each cell of the
    // result equals the f32 sub of the corresponding cells in A and
    // B exactly.
    let mut mismatches = 0usize;
    let mut first: Option<(usize, f32, f32)> = None;
    for (i, (g, e)) in host_got.iter().zip(host_expected.iter()).enumerate() {
        if g.to_bits() != e.to_bits() {
            mismatches += 1;
            if first.is_none() {
                first = Some((i, *g, *e));
            }
        }
    }
    if mismatches > 0 {
        let (i, g, e) = first.unwrap();
        panic!(
            "binary sub f32: {mismatches} mismatches over {numel} cells \
             for shape {shape:?}; first @ {i}: got {g} (bits {:#x}) \
             expected {e} (bits {:#x})",
            g.to_bits(),
            e.to_bits()
        );
    }
}

#[test]
#[ignore]
fn sub_f32_3d() {
    run_case::<3>([8, 128, 128]);
}

// --- f16 / bf16 / f64 fanout ------------------------------------------------

fn run_case_f16_3d(shape: [i32; 3]) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

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
        .map(|(a, b)| *a - *b)
        .collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload A");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload B");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc Y");

    let stride = contiguous_stride(shape);
    let desc = BinaryDescriptor {
        kind: BinaryKind::Sub,
        shape,
        element: ElementKind::F16,
    };
    let plan = BinaryPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select BinaryPlan<f16, 3>");
    let args = BinaryArgs::<f16, 3> {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("binary sub f16 run");
    stream.synchronize().expect("stream sync");

    let mut host_got = vec![f16::from_f32(0.0); numel];
    dev_y.copy_to_host(&mut host_got).expect("download Y");

    let mut mismatches = 0usize;
    let mut first: Option<(usize, f16, f16)> = None;
    for (i, (g, e)) in host_got.iter().zip(host_expected.iter()).enumerate() {
        let gf = g.to_f32();
        let ef = e.to_f32();
        let tol = ef.abs().max(1.0) * F16_EPS;
        if !((gf - ef).abs() <= tol) {
            mismatches += 1;
            if first.is_none() { first = Some((i, *g, *e)); }
        }
    }
    if mismatches > 0 {
        let (i, g, e) = first.unwrap();
        panic!(
            "binary sub f16: {mismatches} mismatches over {numel} cells \
             for shape {shape:?}; first @ {i}: got {} (bits {:#x}) \
             expected {} (bits {:#x})",
            g.to_f32(), g.to_bits(), e.to_f32(), e.to_bits()
        );
    }
}

fn run_case_bf16_3d(shape: [i32; 3]) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

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
        .map(|(a, b)| *a - *b)
        .collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload A");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload B");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc Y");

    let stride = contiguous_stride(shape);
    let desc = BinaryDescriptor {
        kind: BinaryKind::Sub,
        shape,
        element: ElementKind::Bf16,
    };
    let plan = BinaryPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select BinaryPlan<bf16, 3>");
    let args = BinaryArgs::<bf16, 3> {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("binary sub bf16 run");
    stream.synchronize().expect("stream sync");

    let mut host_got = vec![bf16::from_f32(0.0); numel];
    dev_y.copy_to_host(&mut host_got).expect("download Y");

    let mut mismatches = 0usize;
    let mut first: Option<(usize, bf16, bf16)> = None;
    for (i, (g, e)) in host_got.iter().zip(host_expected.iter()).enumerate() {
        let gf = g.to_f32();
        let ef = e.to_f32();
        let tol = ef.abs().max(1.0) * BF16_EPS;
        if !((gf - ef).abs() <= tol) {
            mismatches += 1;
            if first.is_none() { first = Some((i, *g, *e)); }
        }
    }
    if mismatches > 0 {
        let (i, g, e) = first.unwrap();
        panic!(
            "binary sub bf16: {mismatches} mismatches over {numel} cells \
             for shape {shape:?}; first @ {i}: got {} (bits {:#x}) \
             expected {} (bits {:#x})",
            g.to_f32(), g.to_bits(), e.to_f32(), e.to_bits()
        );
    }
}

fn run_case_f64_3d(shape: [i32; 3]) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let numel: usize = shape.iter().map(|&d| d as usize).product();

    let host_a: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.5 - 17.25).collect();
    let host_b: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.125 + 3.75).collect();
    let host_expected: Vec<f64> = host_a.iter().zip(host_b.iter()).map(|(a, b)| a - b).collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload A");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload B");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc Y");

    let stride = contiguous_stride(shape);
    let desc = BinaryDescriptor {
        kind: BinaryKind::Sub,
        shape,
        element: ElementKind::F64,
    };
    let plan = BinaryPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select BinaryPlan<f64, 3>");
    let args = BinaryArgs::<f64, 3> {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("binary sub f64 run");
    stream.synchronize().expect("stream sync");

    let mut host_got = vec![0f64; numel];
    dev_y.copy_to_host(&mut host_got).expect("download Y");

    let mut mismatches = 0usize;
    let mut first: Option<(usize, f64, f64)> = None;
    for (i, (g, e)) in host_got.iter().zip(host_expected.iter()).enumerate() {
        if g.to_bits() != e.to_bits() {
            mismatches += 1;
            if first.is_none() { first = Some((i, *g, *e)); }
        }
    }
    if mismatches > 0 {
        let (i, g, e) = first.unwrap();
        panic!(
            "binary sub f64: {mismatches} mismatches over {numel} cells \
             for shape {shape:?}; first @ {i}: got {g} (bits {:#x}) \
             expected {e} (bits {:#x})",
            g.to_bits(), e.to_bits()
        );
    }
}

#[test]
#[ignore]
fn sub_f16_3d() {
    run_case_f16_3d([8, 128, 128]);
}

#[test]
#[ignore]
fn sub_bf16_3d() {
    run_case_bf16_3d([8, 128, 128]);
}

#[test]
#[ignore]
fn sub_f64_3d() {
    run_case_f64_3d([8, 128, 128]);
}
