//! Real-GPU smoke test for `BinaryPlan + BinaryKind::Nextafter` —
//! `y = nextafter(a, b)`. Bit-exact across all dtypes — the operation
//! is a 1-ULP bit manipulation with no rounding.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test binary_nextafter_smoke -- --ignored`.

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

// CPU bit-pattern reference for f16 / bf16 nextafter. Mirrors the
// kernel's algorithm exactly.
fn nextafter_half_bits(a_bits: u16, b_bits: u16, a_f: f32, b_f: f32) -> u16 {
    if a_f.is_nan() || b_f.is_nan() {
        // Canonical NaN bit pattern with sign 0 (f16 / bf16 quiet NaN).
        // 0x7E00 is the standard f16 qNaN; 0x7FC0 is the bf16 qNaN. The
        // device path uses `__float2half(qnan_f32)` / `__float2bfloat16(qnan_f32)`
        // which produces these patterns. Since the test compares values
        // via NaN-check, we just signal "any NaN".
        return 0xFFFF; // sentinel — caller checks is_nan() instead
    }
    if a_f == b_f {
        // Returns b — matches the kernel which uses `__heq(a, b)` then
        // returns b directly.
        return b_bits;
    }
    if a_f == 0.0 {
        return if b_f > 0.0 { 0x0001 } else { 0x8001 };
    }
    let away = (a_f > 0.0) == (b_f > a_f);
    if away {
        a_bits.wrapping_add(1)
    } else {
        a_bits.wrapping_sub(1)
    }
}

#[test]
#[ignore]
fn nextafter_f32_contig() {
    let (ctx, stream) = setup();
    let shape = [4i32, 64];
    let numel: usize = (shape[0] * shape[1]) as usize;
    // Mix a < b, a > b, a == b across the buffer.
    let host_a: Vec<f32> = (0..numel)
        .map(|i| match i % 3 {
            0 => 1.0 + (i as f32) * 0.01,           // a < b
            1 => 2.0 - (i as f32) * 0.01,           // a > b
            _ => (i as f32) * 0.5,                  // a == b
        })
        .collect();
    let host_b: Vec<f32> = (0..numel)
        .map(|i| match i % 3 {
            0 => 100.0,                              // toward +∞
            1 => -100.0,                             // toward -∞
            _ => (i as f32) * 0.5,                   // == a
        })
        .collect();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);

    let plan = BinaryPlan::<f32, 2>::select(
        &stream,
        &BinaryDescriptor { kind: BinaryKind::Nextafter, shape, element: ElementKind::F32 },
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
        // Use libm `nextafter` semantics manually: it's bit-exact.
        // f32 has no stdlib nextafter; implement reference via bit math.
        let want = nextafter_f32_ref(a, b);
        assert_eq!(g.to_bits(), want.to_bits(),
            "f32 nextafter @ {i}: a={a:e} b={b:e} got_bits={:#x} want_bits={:#x}", g.to_bits(), want.to_bits());
    }
}

fn nextafter_f32_ref(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        return f32::NAN;
    }
    if a == b {
        return b;
    }
    if a == 0.0 {
        return if b > 0.0 { f32::from_bits(1) } else { f32::from_bits(1 | 0x8000_0000) };
    }
    let bits = a.to_bits();
    let away = (a > 0.0) == (b > a);
    if away {
        f32::from_bits(bits.wrapping_add(1))
    } else {
        f32::from_bits(bits.wrapping_sub(1))
    }
}

fn nextafter_f64_ref(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == b {
        return b;
    }
    if a == 0.0 {
        return if b > 0.0 { f64::from_bits(1) } else { f64::from_bits(1 | 0x8000_0000_0000_0000) };
    }
    let bits = a.to_bits();
    let away = (a > 0.0) == (b > a);
    if away {
        f64::from_bits(bits.wrapping_add(1))
    } else {
        f64::from_bits(bits.wrapping_sub(1))
    }
}

#[test]
#[ignore]
fn nextafter_f64_contig() {
    let (ctx, stream) = setup();
    let shape = [3i32, 32];
    let numel: usize = (shape[0] * shape[1]) as usize;
    let host_a: Vec<f64> = (0..numel)
        .map(|i| match i % 3 {
            0 => 1.0 + (i as f64) * 0.01,
            1 => 2.0 - (i as f64) * 0.01,
            _ => (i as f64) * 0.5,
        })
        .collect();
    let host_b: Vec<f64> = (0..numel)
        .map(|i| match i % 3 {
            0 => 100.0,
            1 => -100.0,
            _ => (i as f64) * 0.5,
        })
        .collect();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let plan = BinaryPlan::<f64, 2>::select(
        &stream,
        &BinaryDescriptor { kind: BinaryKind::Nextafter, shape, element: ElementKind::F64 },
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
        let want = nextafter_f64_ref(a, b);
        assert_eq!(g.to_bits(), want.to_bits(),
            "f64 nextafter @ {i}: a={a:e} b={b:e} got_bits={:#x} want_bits={:#x}", g.to_bits(), want.to_bits());
    }
}

#[test]
#[ignore]
fn nextafter_f16_contig() {
    let (ctx, stream) = setup();
    let shape = [4i32, 32];
    let numel: usize = (shape[0] * shape[1]) as usize;
    let host_a: Vec<f16> = (0..numel)
        .map(|i| match i % 3 {
            0 => f16::from_f32(0.5 + (i as f32) * 0.01),
            1 => f16::from_f32(-0.5 - (i as f32) * 0.01),
            _ => f16::from_f32((i as f32) * 0.1),
        })
        .collect();
    let host_b: Vec<f16> = (0..numel)
        .map(|i| match i % 3 {
            0 => f16::from_f32(10.0),
            1 => f16::from_f32(-10.0),
            _ => f16::from_f32((i as f32) * 0.1),
        })
        .collect();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let plan = BinaryPlan::<f16, 2>::select(
        &stream,
        &BinaryDescriptor { kind: BinaryKind::Nextafter, shape, element: ElementKind::F16 },
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
        let want_bits = nextafter_half_bits(a.to_bits(), b.to_bits(), a.to_f32(), b.to_f32());
        if want_bits == 0xFFFF {
            // NaN path — kernel returns a NaN. (Reference is computed via
            // f32::NAN → f16, which is also a canonical f16 NaN.)
            assert!(g.to_f32().is_nan(),
                "f16 nextafter @ {i}: a={a} b={b} got={g} expected NaN");
        } else {
            assert_eq!(g.to_bits(), want_bits,
                "f16 nextafter @ {i}: a_bits={:#06x} b_bits={:#06x} got={:#06x} want={:#06x}",
                a.to_bits(), b.to_bits(), g.to_bits(), want_bits);
        }
    }
}

#[test]
#[ignore]
fn nextafter_bf16_contig() {
    let (ctx, stream) = setup();
    let shape = [3i32, 16];
    let numel: usize = (shape[0] * shape[1]) as usize;
    let host_a: Vec<bf16> = (0..numel)
        .map(|i| match i % 3 {
            0 => bf16::from_f32(0.5 + (i as f32) * 0.01),
            1 => bf16::from_f32(-0.5 - (i as f32) * 0.01),
            _ => bf16::from_f32((i as f32) * 0.1),
        })
        .collect();
    let host_b: Vec<bf16> = (0..numel)
        .map(|i| match i % 3 {
            0 => bf16::from_f32(10.0),
            1 => bf16::from_f32(-10.0),
            _ => bf16::from_f32((i as f32) * 0.1),
        })
        .collect();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let plan = BinaryPlan::<bf16, 2>::select(
        &stream,
        &BinaryDescriptor { kind: BinaryKind::Nextafter, shape, element: ElementKind::Bf16 },
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
        let want_bits = nextafter_half_bits(a.to_bits(), b.to_bits(), a.to_f32(), b.to_f32());
        if want_bits == 0xFFFF {
            assert!(g.to_f32().is_nan(),
                "bf16 nextafter @ {i}: a={a} b={b} got={g} expected NaN");
        } else {
            assert_eq!(g.to_bits(), want_bits,
                "bf16 nextafter @ {i}: a_bits={:#06x} b_bits={:#06x} got={:#06x} want={:#06x}",
                a.to_bits(), b.to_bits(), g.to_bits(), want_bits);
        }
    }
}
