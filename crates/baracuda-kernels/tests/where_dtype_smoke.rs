//! Dtype-fanout smoke test for `WherePlan<T, N>` on the 3 dtypes
//! (f16 / bf16 / f64) that landed in the heterogeneous-ternary fanout.
//! The f32 trailblazer cell stays in `where_smoke.rs` (which also
//! covers the per-row and scalar-cond broadcast patterns).
//!
//! `where` is pure element selection — no arithmetic — so output is
//! bit-exact against host reference `cond ? a : b` regardless of
//! dtype. Each test compares `.to_bits()` directly.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test where_dtype_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef, WhereArgs,
    WhereDescriptor, WherePlan, Workspace,
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
fn where_f16_3d() {
    let (ctx, stream) = setup();
    let shape: [i32; 3] = [8, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    // Cond alternates every cell so we exercise both branches.
    let host_cond: Vec<u8> = (0..numel).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();
    // Small magnitudes in [-10, 10] — well inside f16's representable
    // range and avoiding subnormal corner cases.
    let host_a: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32(((i as f32) * 0.0125) - 10.0))
        .collect();
    let host_b: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32(((i as f32) * 0.00625) - 5.0))
        .collect();
    let expected: Vec<f16> = host_cond
        .iter()
        .zip(host_a.iter())
        .zip(host_b.iter())
        .map(|((&c, &a), &b)| if c != 0 { a } else { b })
        .collect();

    let dev_cond = DeviceBuffer::from_slice(&ctx, &host_cond).expect("upload cond");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = WhereDescriptor {
        shape,
        element: ElementKind::F16,
    };
    let plan = WherePlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select WherePlan<f16, 3>");

    let args = WhereArgs::<f16, 3> {
        cond: TensorRef {
            data: dev_cond.as_slice(),
            shape,
            stride,
        },
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
    plan.run(&stream, Workspace::None, args).expect("where f16 run");
    stream.synchronize().expect("sync");

    let mut got = vec![f16::from_f32(0.0); numel];
    dev_y.copy_to_host(&mut got).expect("download");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "where f16 mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn where_bf16_3d() {
    let (ctx, stream) = setup();
    let shape: [i32; 3] = [8, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    let host_cond: Vec<u8> = (0..numel).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();
    let host_a: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32(((i as f32) * 0.0125) - 10.0))
        .collect();
    let host_b: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32(((i as f32) * 0.00625) - 5.0))
        .collect();
    let expected: Vec<bf16> = host_cond
        .iter()
        .zip(host_a.iter())
        .zip(host_b.iter())
        .map(|((&c, &a), &b)| if c != 0 { a } else { b })
        .collect();

    let dev_cond = DeviceBuffer::from_slice(&ctx, &host_cond).expect("upload cond");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = WhereDescriptor {
        shape,
        element: ElementKind::Bf16,
    };
    let plan = WherePlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select WherePlan<bf16, 3>");

    let args = WhereArgs::<bf16, 3> {
        cond: TensorRef {
            data: dev_cond.as_slice(),
            shape,
            stride,
        },
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
    plan.run(&stream, Workspace::None, args).expect("where bf16 run");
    stream.synchronize().expect("sync");

    let mut got = vec![bf16::from_f32(0.0); numel];
    dev_y.copy_to_host(&mut got).expect("download");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "where bf16 mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn where_f64_3d() {
    let (ctx, stream) = setup();
    let shape: [i32; 3] = [8, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    let host_cond: Vec<u8> = (0..numel).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();
    let host_a: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.5 - 17.25).collect();
    let host_b: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.25 + 100.0).collect();
    let expected: Vec<f64> = host_cond
        .iter()
        .zip(host_a.iter())
        .zip(host_b.iter())
        .map(|((&c, &a), &b)| if c != 0 { a } else { b })
        .collect();

    let dev_cond = DeviceBuffer::from_slice(&ctx, &host_cond).expect("upload cond");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = WhereDescriptor {
        shape,
        element: ElementKind::F64,
    };
    let plan = WherePlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select WherePlan<f64, 3>");

    let args = WhereArgs::<f64, 3> {
        cond: TensorRef {
            data: dev_cond.as_slice(),
            shape,
            stride,
        },
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
    plan.run(&stream, Workspace::None, args).expect("where f64 run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; numel];
    dev_y.copy_to_host(&mut got).expect("download");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "where f64 mismatch @ {i}: got {g} expected {e}"
        );
    }
}
