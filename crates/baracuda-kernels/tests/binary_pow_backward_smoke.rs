//! Real-GPU smoke test for `BinaryBackwardPlan + BinaryKind::Pow`.
//!
//! Forward: `y = a^b`. Backward:
//!   da = dy * b * a^(b-1)
//!   db = dy * a^b * ln(a)
//! Needs saved `a`, `b`. Test inputs restricted to `a > 0` so all paths
//! are well-defined.

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

#[test]
#[ignore]
fn pow_backward_f32() {
    let (ctx, stream) = setup();
    let shape = [4i32, 32];
    let numel: usize = (shape[0] * shape[1]) as usize;
    let host_dy: Vec<f32> = (0..numel).map(|i| 0.25 + (i as f32) * 0.03).collect();
    let host_a: Vec<f32> = (0..numel).map(|i| 0.2 + (i as f32) * 0.05).collect();
    let host_b: Vec<f32> = (0..numel).map(|i| -1.0 + (i as f32) * 0.02).collect();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload");
    let mut dev_da: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let mut dev_db: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let stride = contiguous_stride(shape);
    let plan = BinaryBackwardPlan::<f32, 2>::select(
        &stream,
        &BinaryBackwardDescriptor { kind: BinaryKind::Pow, shape, element: ElementKind::F32 },
        PlanPreference::default(),
    ).expect("select");
    plan.run(&stream, Workspace::None, BinaryBackwardArgs {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        a: Some(TensorRef { data: dev_a.as_slice(), shape, stride }),
        b: Some(TensorRef { data: dev_b.as_slice(), shape, stride }),
        da: TensorMut { data: dev_da.as_slice_mut(), shape, stride },
        db: TensorMut { data: dev_db.as_slice_mut(), shape, stride },
    }).expect("run");
    stream.synchronize().expect("sync");
    let mut got_da = vec![0f32; numel];
    let mut got_db = vec![0f32; numel];
    dev_da.copy_to_host(&mut got_da).expect("dl");
    dev_db.copy_to_host(&mut got_db).expect("dl");
    let eps = 8.0 * f32::EPSILON;
    for i in 0..numel {
        let (dy, a, b) = (host_dy[i], host_a[i], host_b[i]);
        let want_da = dy * b * a.powf(b - 1.0);
        let want_db = dy * a.powf(b) * a.ln();
        let tol_da = (want_da.abs() * eps).max(eps);
        let tol_db = (want_db.abs() * eps).max(eps);
        assert!((got_da[i] - want_da).abs() <= tol_da,
            "f32 pow BW da @ {i}: got={} want={}", got_da[i], want_da);
        assert!((got_db[i] - want_db).abs() <= tol_db,
            "f32 pow BW db @ {i}: got={} want={}", got_db[i], want_db);
    }
}

#[test]
#[ignore]
fn pow_backward_f64() {
    let (ctx, stream) = setup();
    let shape = [3i32, 16];
    let numel: usize = (shape[0] * shape[1]) as usize;
    let host_dy: Vec<f64> = (0..numel).map(|i| 0.25 + (i as f64) * 0.03).collect();
    let host_a: Vec<f64> = (0..numel).map(|i| 0.2 + (i as f64) * 0.05).collect();
    let host_b: Vec<f64> = (0..numel).map(|i| -1.0 + (i as f64) * 0.02).collect();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload");
    let mut dev_da: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let mut dev_db: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let stride = contiguous_stride(shape);
    let plan = BinaryBackwardPlan::<f64, 2>::select(
        &stream,
        &BinaryBackwardDescriptor { kind: BinaryKind::Pow, shape, element: ElementKind::F64 },
        PlanPreference::default(),
    ).expect("select");
    plan.run(&stream, Workspace::None, BinaryBackwardArgs {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        a: Some(TensorRef { data: dev_a.as_slice(), shape, stride }),
        b: Some(TensorRef { data: dev_b.as_slice(), shape, stride }),
        da: TensorMut { data: dev_da.as_slice_mut(), shape, stride },
        db: TensorMut { data: dev_db.as_slice_mut(), shape, stride },
    }).expect("run");
    stream.synchronize().expect("sync");
    let mut got_da = vec![0f64; numel];
    let mut got_db = vec![0f64; numel];
    dev_da.copy_to_host(&mut got_da).expect("dl");
    dev_db.copy_to_host(&mut got_db).expect("dl");
    let eps = 8.0 * f64::EPSILON;
    for i in 0..numel {
        let (dy, a, b) = (host_dy[i], host_a[i], host_b[i]);
        let want_da = dy * b * a.powf(b - 1.0);
        let want_db = dy * a.powf(b) * a.ln();
        let tol_da = (want_da.abs() * eps).max(eps);
        let tol_db = (want_db.abs() * eps).max(eps);
        assert!((got_da[i] - want_da).abs() <= tol_da, "f64 pow BW da @ {i}");
        assert!((got_db[i] - want_db).abs() <= tol_db, "f64 pow BW db @ {i}");
    }
}

#[test]
#[ignore]
fn pow_backward_f16() {
    let (ctx, stream) = setup();
    let shape = [4i32, 16];
    let numel: usize = (shape[0] * shape[1]) as usize;
    let host_dy: Vec<f16> = (0..numel).map(|i| f16::from_f32(0.25 + (i as f32) * 0.02)).collect();
    let host_a: Vec<f16> = (0..numel).map(|i| f16::from_f32(0.5 + (i as f32) * 0.03)).collect();
    let host_b: Vec<f16> = (0..numel).map(|i| f16::from_f32(-0.5 + (i as f32) * 0.015)).collect();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload");
    let mut dev_da: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let mut dev_db: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let stride = contiguous_stride(shape);
    let plan = BinaryBackwardPlan::<f16, 2>::select(
        &stream,
        &BinaryBackwardDescriptor { kind: BinaryKind::Pow, shape, element: ElementKind::F16 },
        PlanPreference::default(),
    ).expect("select");
    plan.run(&stream, Workspace::None, BinaryBackwardArgs {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        a: Some(TensorRef { data: dev_a.as_slice(), shape, stride }),
        b: Some(TensorRef { data: dev_b.as_slice(), shape, stride }),
        da: TensorMut { data: dev_da.as_slice_mut(), shape, stride },
        db: TensorMut { data: dev_db.as_slice_mut(), shape, stride },
    }).expect("run");
    stream.synchronize().expect("sync");
    let mut got_da = vec![f16::ZERO; numel];
    let mut got_db = vec![f16::ZERO; numel];
    dev_da.copy_to_host(&mut got_da).expect("dl");
    dev_db.copy_to_host(&mut got_db).expect("dl");
    let eps = 2.0 * 9.77e-4_f32; // 2 ULP at f16
    for i in 0..numel {
        let (dy, a, b) = (host_dy[i].to_f32(), host_a[i].to_f32(), host_b[i].to_f32());
        let want_da = dy * b * a.powf(b - 1.0);
        let want_db = dy * a.powf(b) * a.ln();
        let tol_da = (want_da.abs() * eps).max(eps);
        let tol_db = (want_db.abs() * eps).max(eps);
        let diff_da = (got_da[i].to_f32() - want_da).abs();
        let diff_db = (got_db[i].to_f32() - want_db).abs();
        assert!(diff_da <= tol_da,
            "f16 pow BW da @ {i}: got={} want={} diff={}", got_da[i].to_f32(), want_da, diff_da);
        assert!(diff_db <= tol_db,
            "f16 pow BW db @ {i}: got={} want={} diff={}", got_db[i].to_f32(), want_db, diff_db);
    }
}

#[test]
#[ignore]
fn pow_backward_bf16() {
    let (ctx, stream) = setup();
    let shape = [3i32, 16];
    let numel: usize = (shape[0] * shape[1]) as usize;
    let host_dy: Vec<bf16> = (0..numel).map(|i| bf16::from_f32(0.25 + (i as f32) * 0.03)).collect();
    let host_a: Vec<bf16> = (0..numel).map(|i| bf16::from_f32(0.5 + (i as f32) * 0.04)).collect();
    let host_b: Vec<bf16> = (0..numel).map(|i| bf16::from_f32(-0.5 + (i as f32) * 0.02)).collect();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload");
    let mut dev_da: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let mut dev_db: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let stride = contiguous_stride(shape);
    let plan = BinaryBackwardPlan::<bf16, 2>::select(
        &stream,
        &BinaryBackwardDescriptor { kind: BinaryKind::Pow, shape, element: ElementKind::Bf16 },
        PlanPreference::default(),
    ).expect("select");
    plan.run(&stream, Workspace::None, BinaryBackwardArgs {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        a: Some(TensorRef { data: dev_a.as_slice(), shape, stride }),
        b: Some(TensorRef { data: dev_b.as_slice(), shape, stride }),
        da: TensorMut { data: dev_da.as_slice_mut(), shape, stride },
        db: TensorMut { data: dev_db.as_slice_mut(), shape, stride },
    }).expect("run");
    stream.synchronize().expect("sync");
    let mut got_da = vec![bf16::ZERO; numel];
    let mut got_db = vec![bf16::ZERO; numel];
    dev_da.copy_to_host(&mut got_da).expect("dl");
    dev_db.copy_to_host(&mut got_db).expect("dl");
    let eps = 2.0 * 7.81e-3_f32; // 2 ULP at bf16
    for i in 0..numel {
        let (dy, a, b) = (host_dy[i].to_f32(), host_a[i].to_f32(), host_b[i].to_f32());
        let want_da = dy * b * a.powf(b - 1.0);
        let want_db = dy * a.powf(b) * a.ln();
        let tol_da = (want_da.abs() * eps).max(eps);
        let tol_db = (want_db.abs() * eps).max(eps);
        let diff_da = (got_da[i].to_f32() - want_da).abs();
        let diff_db = (got_db[i].to_f32() - want_db).abs();
        assert!(diff_da <= tol_da, "bf16 pow BW da @ {i}: diff={diff_da}");
        assert!(diff_db <= tol_db, "bf16 pow BW db @ {i}: diff={diff_db}");
    }
}
