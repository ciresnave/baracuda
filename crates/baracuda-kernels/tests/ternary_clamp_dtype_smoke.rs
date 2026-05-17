//! Dtype-fill smoke test for `TernaryPlan<T, N> + TernaryKind::Clamp`
//! on the 3 dtypes (f16 / bf16 / f64) that landed in the ternary
//! fanout. The f32 trailblazer cell stays in `ternary_clamp_smoke.rs`
//! (which also covers the scalar-broadcast and strided pattern).
//!
//! Each test runs a contig shape `[8, 128, 128]` with `x` ranging over
//! `[-20, +20]`, full per-cell `lo = -5` and `hi = +5`. Clamp on all
//! four FP dtypes is bit-exact between host and device:
//! - f64 uses `fmin(fmax(...))` directly — single rounding per step
//!   matches the host `f64::max/min` semantics.
//! - f16 / bf16 use the f32-detour pipeline (promote, clamp in f32,
//!   round back). Host reference does the same via `half`'s
//!   `to_f32()` / `from_f32(...)`, so the rounding paths agree
//!   bit-for-bit.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test ternary_clamp_dtype_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef, TernaryArgs,
    TernaryDescriptor, TernaryKind, TernaryPlan, Workspace,
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
fn clamp_f16_3d() {
    let (ctx, stream) = setup();
    let shape: [i32; 3] = [8, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    let host_x: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32((i as f32) * 0.05 - 20.0))
        .collect();
    let host_lo: Vec<f16> = (0..numel).map(|_| f16::from_f32(-5.0)).collect();
    let host_hi: Vec<f16> = (0..numel).map(|_| f16::from_f32(5.0)).collect();
    // Host reference: f32-detour clamp, matching the kernel's
    // ClampFunctor<__half> specialization (fminf(fmaxf(...)) on f32).
    let expected: Vec<f16> = host_x
        .iter()
        .zip(host_lo.iter())
        .zip(host_hi.iter())
        .map(|((x, lo), hi)| {
            f16::from_f32(x.to_f32().max(lo.to_f32()).min(hi.to_f32()))
        })
        .collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_lo = DeviceBuffer::from_slice(&ctx, &host_lo).expect("upload lo");
    let dev_hi = DeviceBuffer::from_slice(&ctx, &host_hi).expect("upload hi");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = TernaryDescriptor {
        kind: TernaryKind::Clamp,
        shape,
        element: ElementKind::F16,
        scale: 1.0,
    };
    let plan = TernaryPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select TernaryPlan<f16, 3>");
    let args = TernaryArgs::<f16, 3> {
        a: TensorRef { data: dev_x.as_slice(), shape, stride },
        b: TensorRef { data: dev_lo.as_slice(), shape, stride },
        c: TensorRef { data: dev_hi.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("clamp f16 run");
    stream.synchronize().expect("sync");

    let mut got = vec![f16::from_f32(0.0); numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "clamp f16 mismatch @ {i}: got bits 0x{:04x} expected 0x{:04x}",
            g.to_bits(),
            e.to_bits()
        );
    }
}

#[test]
#[ignore]
fn clamp_bf16_3d() {
    let (ctx, stream) = setup();
    let shape: [i32; 3] = [8, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    let host_x: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32((i as f32) * 0.05 - 20.0))
        .collect();
    let host_lo: Vec<bf16> = (0..numel).map(|_| bf16::from_f32(-5.0)).collect();
    let host_hi: Vec<bf16> = (0..numel).map(|_| bf16::from_f32(5.0)).collect();
    let expected: Vec<bf16> = host_x
        .iter()
        .zip(host_lo.iter())
        .zip(host_hi.iter())
        .map(|((x, lo), hi)| {
            bf16::from_f32(x.to_f32().max(lo.to_f32()).min(hi.to_f32()))
        })
        .collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_lo = DeviceBuffer::from_slice(&ctx, &host_lo).expect("upload lo");
    let dev_hi = DeviceBuffer::from_slice(&ctx, &host_hi).expect("upload hi");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = TernaryDescriptor {
        kind: TernaryKind::Clamp,
        shape,
        element: ElementKind::Bf16,
        scale: 1.0,
    };
    let plan = TernaryPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select TernaryPlan<bf16, 3>");
    let args = TernaryArgs::<bf16, 3> {
        a: TensorRef { data: dev_x.as_slice(), shape, stride },
        b: TensorRef { data: dev_lo.as_slice(), shape, stride },
        c: TensorRef { data: dev_hi.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("clamp bf16 run");
    stream.synchronize().expect("sync");

    let mut got = vec![bf16::from_f32(0.0); numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "clamp bf16 mismatch @ {i}: got bits 0x{:04x} expected 0x{:04x}",
            g.to_bits(),
            e.to_bits()
        );
    }
}

#[test]
#[ignore]
fn clamp_f64_3d() {
    let (ctx, stream) = setup();
    let shape: [i32; 3] = [8, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    let host_x: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.05 - 20.0).collect();
    let host_lo: Vec<f64> = (0..numel).map(|_| -5.0_f64).collect();
    let host_hi: Vec<f64> = (0..numel).map(|_| 5.0_f64).collect();
    let expected: Vec<f64> = host_x
        .iter()
        .zip(host_lo.iter())
        .zip(host_hi.iter())
        .map(|((&x, &lo), &hi)| x.max(lo).min(hi))
        .collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_lo = DeviceBuffer::from_slice(&ctx, &host_lo).expect("upload lo");
    let dev_hi = DeviceBuffer::from_slice(&ctx, &host_hi).expect("upload hi");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = TernaryDescriptor {
        kind: TernaryKind::Clamp,
        shape,
        element: ElementKind::F64,
        scale: 1.0,
    };
    let plan = TernaryPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select TernaryPlan<f64, 3>");
    let args = TernaryArgs::<f64, 3> {
        a: TensorRef { data: dev_x.as_slice(), shape, stride },
        b: TensorRef { data: dev_lo.as_slice(), shape, stride },
        c: TensorRef { data: dev_hi.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("clamp f64 run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "clamp f64 mismatch @ {i}: got {g} expected {e}"
        );
    }
}
