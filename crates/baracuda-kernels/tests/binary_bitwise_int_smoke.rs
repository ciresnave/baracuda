//! Real-GPU smoke test for the Phase 3.3 integer bitwise binary
//! kernels in `baracuda-kernels-sys`.
//!
//! Covers `BinaryPlan<T, N>` over the five bitwise ops
//! (`BitwiseAnd / BitwiseOr / BitwiseXor / BitwiseLeftShift /
//! BitwiseRightShift`) across the `{i32, i64}` integer family on
//! contiguous tensors. The kernels are pure integer-arithmetic SIMT
//! sweeps (no rounding, no warp reduction), so each cell of the
//! output must equal the CPU reference **bit-exactly** — no tolerance.
//!
//! Right-shift is **arithmetic** (sign-extending) on signed inputs,
//! matching PyTorch's contract. We exercise the sign-extension path
//! by including negative values in the input mix.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BinaryArgs, BinaryDescriptor, BinaryKind, BinaryPlan, ElementKind,
    PlanPreference, TensorMut, TensorRef, Workspace,
};

// =============================================================================
// i32 driver
// =============================================================================

fn run_case_i32(kind: BinaryKind, shape: [i32; 2]) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let numel: usize = shape.iter().map(|&d| d as usize).product();
    assert!(numel > 0);

    // Mix of positive / negative values to exercise sign extension on
    // arithmetic right shift and the high-bit handling on AND / OR / XOR.
    // Shift `b` is clamped to [0, 31] so we stay in the defined-behavior
    // range — out-of-range shifts inherit hardware behavior, which we
    // explicitly don't pin in the test.
    let host_a: Vec<i32> = (0..numel)
        .map(|i| {
            let v = ((i as i64).wrapping_mul(2654435761) & 0xFFFF_FFFF) as i32;
            // Force some negatives by subtracting a midrange offset.
            v.wrapping_sub(0x4000_0000)
        })
        .collect();
    let host_b: Vec<i32> = (0..numel)
        .map(|i| match kind {
            BinaryKind::BitwiseLeftShift | BinaryKind::BitwiseRightShift => {
                // Defined-behavior shift amount range for i32.
                (i % 32) as i32
            }
            _ => {
                let v = ((i as i64).wrapping_mul(40503) & 0xFFFF_FFFF) as i32;
                v.wrapping_sub(0x1234_5678)
            }
        })
        .collect();

    let host_expected: Vec<i32> = host_a
        .iter()
        .zip(host_b.iter())
        .map(|(a, b)| match kind {
            BinaryKind::BitwiseAnd => a & b,
            BinaryKind::BitwiseOr => a | b,
            BinaryKind::BitwiseXor => a ^ b,
            BinaryKind::BitwiseLeftShift => a.wrapping_shl(*b as u32),
            BinaryKind::BitwiseRightShift => a >> b, // arithmetic on signed
            _ => unreachable!(),
        })
        .collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload A");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload B");
    let mut dev_y: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, numel).expect("alloc Y");

    let stride = contiguous_stride(shape);
    let desc = BinaryDescriptor {
        kind,
        shape,
        element: ElementKind::I32,
    };
    let plan = BinaryPlan::<i32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select BinaryPlan<i32, 2>");
    let args = BinaryArgs::<i32, 2> {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("binary bitwise i32 run");
    stream.synchronize().expect("stream sync");

    let mut host_got = vec![0i32; numel];
    dev_y.copy_to_host(&mut host_got).expect("download Y");

    let mut mismatches = 0usize;
    let mut first: Option<(usize, i32, i32)> = None;
    for (i, (g, e)) in host_got.iter().zip(host_expected.iter()).enumerate() {
        if g != e {
            mismatches += 1;
            if first.is_none() {
                first = Some((i, *g, *e));
            }
        }
    }
    if mismatches > 0 {
        let (i, g, e) = first.unwrap();
        panic!(
            "binary {kind:?} i32: {mismatches} mismatches over {numel} cells \
             for shape {shape:?}; first @ {i}: got {g:#x} expected {e:#x}"
        );
    }
}

// =============================================================================
// i64 driver
// =============================================================================

fn run_case_i64(kind: BinaryKind, shape: [i32; 2]) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let numel: usize = shape.iter().map(|&d| d as usize).product();
    assert!(numel > 0);

    let host_a: Vec<i64> = (0..numel)
        .map(|i| {
            let v = (i as i64).wrapping_mul(11400714819323198485u64 as i64);
            // Mix of large positives, negatives, and small values.
            v.wrapping_sub(0x4000_0000_0000_0000)
        })
        .collect();
    let host_b: Vec<i64> = (0..numel)
        .map(|i| match kind {
            BinaryKind::BitwiseLeftShift | BinaryKind::BitwiseRightShift => {
                // Defined-behavior shift amount range for i64.
                (i % 64) as i64
            }
            _ => {
                let v = (i as i64).wrapping_mul(2862933555777941757u64 as i64);
                v.wrapping_sub(0x0123_4567_89AB_CDEF)
            }
        })
        .collect();

    let host_expected: Vec<i64> = host_a
        .iter()
        .zip(host_b.iter())
        .map(|(a, b)| match kind {
            BinaryKind::BitwiseAnd => a & b,
            BinaryKind::BitwiseOr => a | b,
            BinaryKind::BitwiseXor => a ^ b,
            BinaryKind::BitwiseLeftShift => a.wrapping_shl(*b as u32),
            BinaryKind::BitwiseRightShift => a >> b, // arithmetic on signed
            _ => unreachable!(),
        })
        .collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload A");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload B");
    let mut dev_y: DeviceBuffer<i64> =
        DeviceBuffer::zeros(&ctx, numel).expect("alloc Y");

    let stride = contiguous_stride(shape);
    let desc = BinaryDescriptor {
        kind,
        shape,
        element: ElementKind::I64,
    };
    let plan = BinaryPlan::<i64, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select BinaryPlan<i64, 2>");
    let args = BinaryArgs::<i64, 2> {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("binary bitwise i64 run");
    stream.synchronize().expect("stream sync");

    let mut host_got = vec![0i64; numel];
    dev_y.copy_to_host(&mut host_got).expect("download Y");

    let mut mismatches = 0usize;
    let mut first: Option<(usize, i64, i64)> = None;
    for (i, (g, e)) in host_got.iter().zip(host_expected.iter()).enumerate() {
        if g != e {
            mismatches += 1;
            if first.is_none() {
                first = Some((i, *g, *e));
            }
        }
    }
    if mismatches > 0 {
        let (i, g, e) = first.unwrap();
        panic!(
            "binary {kind:?} i64: {mismatches} mismatches over {numel} cells \
             for shape {shape:?}; first @ {i}: got {g:#x} expected {e:#x}"
        );
    }
}

// =============================================================================
// i32 test matrix — 5 ops × i32 = 5 tests
// =============================================================================

#[test]
#[ignore]
fn bitwise_and_i32() {
    run_case_i32(BinaryKind::BitwiseAnd, [64, 65]);
}

#[test]
#[ignore]
fn bitwise_or_i32() {
    run_case_i32(BinaryKind::BitwiseOr, [64, 65]);
}

#[test]
#[ignore]
fn bitwise_xor_i32() {
    run_case_i32(BinaryKind::BitwiseXor, [64, 65]);
}

#[test]
#[ignore]
fn bitwise_left_shift_i32() {
    run_case_i32(BinaryKind::BitwiseLeftShift, [64, 65]);
}

#[test]
#[ignore]
fn bitwise_right_shift_i32() {
    run_case_i32(BinaryKind::BitwiseRightShift, [64, 65]);
}

// =============================================================================
// i64 test matrix — 5 ops × i64 = 5 tests
// =============================================================================

#[test]
#[ignore]
fn bitwise_and_i64() {
    run_case_i64(BinaryKind::BitwiseAnd, [64, 65]);
}

#[test]
#[ignore]
fn bitwise_or_i64() {
    run_case_i64(BinaryKind::BitwiseOr, [64, 65]);
}

#[test]
#[ignore]
fn bitwise_xor_i64() {
    run_case_i64(BinaryKind::BitwiseXor, [64, 65]);
}

#[test]
#[ignore]
fn bitwise_left_shift_i64() {
    run_case_i64(BinaryKind::BitwiseLeftShift, [64, 65]);
}

#[test]
#[ignore]
fn bitwise_right_shift_i64() {
    run_case_i64(BinaryKind::BitwiseRightShift, [64, 65]);
}
