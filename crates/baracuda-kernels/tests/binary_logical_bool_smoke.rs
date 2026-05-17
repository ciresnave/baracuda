//! Real-GPU smoke test for the Phase 3.3 boolean logical binary
//! kernels in `baracuda-kernels-sys`.
//!
//! Covers `BinaryPlan<Bool, N>` over the three logical ops
//! (`LogicalAnd / LogicalOr / LogicalXor`) on contiguous tensors. The
//! kernels normalize each input byte to 0 / 1 before applying the
//! logical op, so the output is always canonical 0 or 1 even for
//! unnormalized input bytes (e.g. arbitrary non-zero values
//! representing "true").
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BinaryArgs, BinaryDescriptor, BinaryKind, BinaryPlan, Bool, ElementKind,
    PlanPreference, TensorMut, TensorRef, Workspace,
};

fn run_case_bool(kind: BinaryKind, shape: [i32; 2]) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let numel: usize = shape.iter().map(|&d| d as usize).product();
    assert!(numel > 0);

    // Use a mix of canonical {0, 1} bytes and unnormalized non-zero
    // bytes (e.g. 42, 255) for A; the kernel must treat all non-zero
    // values as truthy. B uses canonical {0, 1}.
    let host_a: Vec<Bool> = (0..numel)
        .map(|i| match i % 4 {
            0 => Bool(0),
            1 => Bool(1),
            2 => Bool(42),  // unnormalized "true"
            _ => Bool(255), // unnormalized "true"
        })
        .collect();
    let host_b: Vec<Bool> = (0..numel)
        .map(|i| if i % 3 == 0 { Bool(0) } else { Bool(1) })
        .collect();

    let host_expected: Vec<Bool> = host_a
        .iter()
        .zip(host_b.iter())
        .map(|(a, b)| {
            let av = a.0 != 0;
            let bv = b.0 != 0;
            let r = match kind {
                BinaryKind::LogicalAnd => av && bv,
                BinaryKind::LogicalOr => av || bv,
                BinaryKind::LogicalXor => av != bv,
                _ => unreachable!(),
            };
            Bool(if r { 1 } else { 0 })
        })
        .collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload A");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload B");
    let mut dev_y: DeviceBuffer<Bool> =
        DeviceBuffer::zeros(&ctx, numel).expect("alloc Y");

    let stride = contiguous_stride(shape);
    let desc = BinaryDescriptor {
        kind,
        shape,
        element: ElementKind::Bool,
    };
    let plan = BinaryPlan::<Bool, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select BinaryPlan<Bool, 2>");
    let args = BinaryArgs::<Bool, 2> {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("binary logical bool run");
    stream.synchronize().expect("stream sync");

    let mut host_got = vec![Bool(0); numel];
    dev_y.copy_to_host(&mut host_got).expect("download Y");

    let mut mismatches = 0usize;
    let mut first: Option<(usize, u8, u8)> = None;
    for (i, (g, e)) in host_got.iter().zip(host_expected.iter()).enumerate() {
        if g.0 != e.0 {
            mismatches += 1;
            if first.is_none() {
                first = Some((i, g.0, e.0));
            }
        }
    }
    if mismatches > 0 {
        let (i, g, e) = first.unwrap();
        panic!(
            "binary {kind:?} bool: {mismatches} mismatches over {numel} cells \
             for shape {shape:?}; first @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn logical_and_bool() {
    run_case_bool(BinaryKind::LogicalAnd, [64, 65]);
}

#[test]
#[ignore]
fn logical_or_bool() {
    run_case_bool(BinaryKind::LogicalOr, [64, 65]);
}

#[test]
#[ignore]
fn logical_xor_bool() {
    run_case_bool(BinaryKind::LogicalXor, [64, 65]);
}
