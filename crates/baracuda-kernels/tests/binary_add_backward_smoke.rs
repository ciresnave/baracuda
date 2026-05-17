//! Real-GPU smoke test for the binary-add backward kernel
//! (`BinaryBackwardPlan<T, N> + BinaryKind::Add`).
//!
//! Forward: `y = a + b`. Backward: `(da, db) = (dy, dy)`. Bit-exact —
//! no math, pure copy of upstream gradient to both gradient outputs.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test binary_add_backward_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BinaryBackwardArgs, BinaryBackwardDescriptor, BinaryBackwardPlan,
    BinaryKind, ElementKind, PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn run_case_f32<const N: usize>(shape: [i32; N]) {
    let (ctx, stream) = setup();
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    let host_dy: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.5 - 17.25).collect();
    let expected_da = host_dy.clone();
    let expected_db = host_dy.clone();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_da: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, numel).expect("alloc da");
    let mut dev_db: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, numel).expect("alloc db");

    let stride = contiguous_stride(shape);
    let desc = BinaryBackwardDescriptor {
        kind: BinaryKind::Add,
        shape,
        element: ElementKind::F32,
    };
    let plan = BinaryBackwardPlan::<f32, N>::select(&stream, &desc, PlanPreference::default())
        .expect("select");

    let args = BinaryBackwardArgs::<f32, N> {
        dy: TensorRef {
            data: dev_dy.as_slice(),
            shape,
            stride,
        },
        a: None,
        b: None,
        da: TensorMut {
            data: dev_da.as_slice_mut(),
            shape,
            stride,
        },
        db: TensorMut {
            data: dev_db.as_slice_mut(),
            shape,
            stride,
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got_da = vec![0f32; numel];
    let mut got_db = vec![0f32; numel];
    dev_da.copy_to_host(&mut got_da).expect("download da");
    dev_db.copy_to_host(&mut got_db).expect("download db");

    for (i, (g, e)) in got_da.iter().zip(expected_da.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "add backward da mismatch @ {i}");
    }
    for (i, (g, e)) in got_db.iter().zip(expected_db.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "add backward db mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn add_backward_f32_1d() {
    run_case_f32::<1>([2048]);
}

#[test]
#[ignore]
fn add_backward_f32_2d() {
    run_case_f32::<2>([64, 64]);
}

#[test]
#[ignore]
fn add_backward_f32_3d() {
    run_case_f32::<3>([8, 128, 128]);
}

// --- f16 / bf16 / f64 fanout (Add backward is a pure copy — bit-exact) ----

#[test]
#[ignore]
fn add_backward_f16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32((i % 41) as f32 * 0.5 - 10.0))
        .collect();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_da: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc da");
    let mut dev_db: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc db");
    let stride = contiguous_stride(shape);
    let desc = BinaryBackwardDescriptor {
        kind: BinaryKind::Add,
        shape,
        element: ElementKind::F16,
    };
    let plan = BinaryBackwardPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = BinaryBackwardArgs::<f16, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        a: None,
        b: None,
        da: TensorMut { data: dev_da.as_slice_mut(), shape, stride },
        db: TensorMut { data: dev_db.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got_da = vec![f16::from_f32(0.0); numel];
    let mut got_db = vec![f16::from_f32(0.0); numel];
    dev_da.copy_to_host(&mut got_da).expect("download da");
    dev_db.copy_to_host(&mut got_db).expect("download db");
    for (i, (g, e)) in got_da.iter().zip(host_dy.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "add backward f16 da mismatch @ {i}");
    }
    for (i, (g, e)) in got_db.iter().zip(host_dy.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "add backward f16 db mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn add_backward_bf16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32((i % 41) as f32 * 0.5 - 10.0))
        .collect();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_da: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc da");
    let mut dev_db: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc db");
    let stride = contiguous_stride(shape);
    let desc = BinaryBackwardDescriptor {
        kind: BinaryKind::Add,
        shape,
        element: ElementKind::Bf16,
    };
    let plan = BinaryBackwardPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = BinaryBackwardArgs::<bf16, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        a: None,
        b: None,
        da: TensorMut { data: dev_da.as_slice_mut(), shape, stride },
        db: TensorMut { data: dev_db.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got_da = vec![bf16::from_f32(0.0); numel];
    let mut got_db = vec![bf16::from_f32(0.0); numel];
    dev_da.copy_to_host(&mut got_da).expect("download da");
    dev_db.copy_to_host(&mut got_db).expect("download db");
    for (i, (g, e)) in got_da.iter().zip(host_dy.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "add backward bf16 da mismatch @ {i}");
    }
    for (i, (g, e)) in got_db.iter().zip(host_dy.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "add backward bf16 db mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn add_backward_f64_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.5 - 17.25).collect();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_da: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc da");
    let mut dev_db: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc db");
    let stride = contiguous_stride(shape);
    let desc = BinaryBackwardDescriptor {
        kind: BinaryKind::Add,
        shape,
        element: ElementKind::F64,
    };
    let plan = BinaryBackwardPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = BinaryBackwardArgs::<f64, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        a: None,
        b: None,
        da: TensorMut { data: dev_da.as_slice_mut(), shape, stride },
        db: TensorMut { data: dev_db.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got_da = vec![0f64; numel];
    let mut got_db = vec![0f64; numel];
    dev_da.copy_to_host(&mut got_da).expect("download da");
    dev_db.copy_to_host(&mut got_db).expect("download db");
    for (i, (g, e)) in got_da.iter().zip(host_dy.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "add backward f64 da mismatch @ {i}");
    }
    for (i, (g, e)) in got_db.iter().zip(host_dy.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "add backward f64 db mismatch @ {i}");
    }
}

/// `can_implement` rejects Mul/Div when the caller forgets to supply
/// the saved forward inputs `a` and `b` (gradient formula requires them).
/// This is a host-side test — no GPU work needed.
#[test]
fn mul_backward_requires_saves() {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    let desc = BinaryBackwardDescriptor {
        kind: BinaryKind::Mul,
        shape: [4],
        element: ElementKind::F32,
    };
    let plan = BinaryBackwardPlan::<f32, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("Mul backward × f32 is wired in select");
    let dev_dy: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 4).expect("alloc dy");
    let mut dev_da: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 4).expect("alloc da");
    let mut dev_db: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 4).expect("alloc db");
    let stride = contiguous_stride([4]);
    let args = BinaryBackwardArgs::<f32, 1> {
        dy: TensorRef { data: dev_dy.as_slice(), shape: [4], stride },
        a: None,
        b: None,
        da: TensorMut { data: dev_da.as_slice_mut(), shape: [4], stride },
        db: TensorMut { data: dev_db.as_slice_mut(), shape: [4], stride },
    };
    let err = plan.can_implement(&args);
    assert!(
        err.is_err(),
        "Mul backward must reject missing saved inputs at can_implement time"
    );
}
