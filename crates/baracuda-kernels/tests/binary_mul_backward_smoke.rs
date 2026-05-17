//! Real-GPU smoke test for the binary-mul backward kernel
//! (`BinaryBackwardPlan<T, N> + BinaryKind::Mul`).
//!
//! Forward: `y = a * b`. Backward: `(da, db) = (dy * b, dy * a)`. Needs
//! saved forward inputs `a` and `b`. Bit-exact compare for f32 / f64
//! (each output is a single multiplication, no FMA opportunity).
//! Tolerance compare for f16 / bf16 to match the forward Mul fanout
//! pattern.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test binary_mul_backward_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BinaryBackwardArgs, BinaryBackwardDescriptor, BinaryBackwardPlan,
    BinaryKind, ElementKind, PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

// 1-ULP relative tolerance (matches binary_mul_smoke.rs).
const F16_EPS: f32 = 9.77e-4;
const BF16_EPS: f32 = 7.81e-3;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn mul_backward_f32_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.5 - 17.25).collect();
    let host_a: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.125 + 1.0).collect();
    let host_b: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.0625 - 3.0).collect();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_da: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc da");
    let mut dev_db: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc db");
    let stride = contiguous_stride(shape);
    let desc = BinaryBackwardDescriptor {
        kind: BinaryKind::Mul,
        shape,
        element: ElementKind::F32,
    };
    let plan = BinaryBackwardPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = BinaryBackwardArgs::<f32, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        a: Some(TensorRef { data: dev_a.as_slice(), shape, stride }),
        b: Some(TensorRef { data: dev_b.as_slice(), shape, stride }),
        da: TensorMut { data: dev_da.as_slice_mut(), shape, stride },
        db: TensorMut { data: dev_db.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got_da = vec![0f32; numel];
    let mut got_db = vec![0f32; numel];
    dev_da.copy_to_host(&mut got_da).expect("download da");
    dev_db.copy_to_host(&mut got_db).expect("download db");
    for i in 0..numel {
        let exp_da = host_dy[i] * host_b[i];
        let exp_db = host_dy[i] * host_a[i];
        assert_eq!(got_da[i].to_bits(), exp_da.to_bits(), "mul backward f32 da @ {i}");
        assert_eq!(got_db[i].to_bits(), exp_db.to_bits(), "mul backward f32 db @ {i}");
    }
}

#[test]
#[ignore]
fn mul_backward_f16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<f16> = (0..numel).map(|i| f16::from_f32((i % 41) as f32 * 0.25 - 5.0)).collect();
    let host_a:  Vec<f16> = (0..numel).map(|i| f16::from_f32((i % 37) as f32 * 0.125 + 1.0)).collect();
    let host_b:  Vec<f16> = (0..numel).map(|i| f16::from_f32((i % 29) as f32 * 0.0625 - 1.5)).collect();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_da: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc da");
    let mut dev_db: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc db");
    let stride = contiguous_stride(shape);
    let desc = BinaryBackwardDescriptor {
        kind: BinaryKind::Mul,
        shape,
        element: ElementKind::F16,
    };
    let plan = BinaryBackwardPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = BinaryBackwardArgs::<f16, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        a: Some(TensorRef { data: dev_a.as_slice(), shape, stride }),
        b: Some(TensorRef { data: dev_b.as_slice(), shape, stride }),
        da: TensorMut { data: dev_da.as_slice_mut(), shape, stride },
        db: TensorMut { data: dev_db.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got_da = vec![f16::from_f32(0.0); numel];
    let mut got_db = vec![f16::from_f32(0.0); numel];
    dev_da.copy_to_host(&mut got_da).expect("download da");
    dev_db.copy_to_host(&mut got_db).expect("download db");
    for i in 0..numel {
        let exp_da = host_dy[i] * host_b[i];
        let exp_db = host_dy[i] * host_a[i];
        let gd = got_da[i].to_f32();
        let ed = exp_da.to_f32();
        let tol = ed.abs().max(1.0) * F16_EPS;
        assert!((gd - ed).abs() <= tol, "mul backward f16 da @ {i}: got {gd}, exp {ed}");
        let gdb = got_db[i].to_f32();
        let edb = exp_db.to_f32();
        let tol = edb.abs().max(1.0) * F16_EPS;
        assert!((gdb - edb).abs() <= tol, "mul backward f16 db @ {i}: got {gdb}, exp {edb}");
    }
}

#[test]
#[ignore]
fn mul_backward_bf16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<bf16> = (0..numel).map(|i| bf16::from_f32((i % 41) as f32 * 0.25 - 5.0)).collect();
    let host_a:  Vec<bf16> = (0..numel).map(|i| bf16::from_f32((i % 37) as f32 * 0.125 + 1.0)).collect();
    let host_b:  Vec<bf16> = (0..numel).map(|i| bf16::from_f32((i % 29) as f32 * 0.0625 - 1.5)).collect();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_da: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc da");
    let mut dev_db: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc db");
    let stride = contiguous_stride(shape);
    let desc = BinaryBackwardDescriptor {
        kind: BinaryKind::Mul,
        shape,
        element: ElementKind::Bf16,
    };
    let plan = BinaryBackwardPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = BinaryBackwardArgs::<bf16, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        a: Some(TensorRef { data: dev_a.as_slice(), shape, stride }),
        b: Some(TensorRef { data: dev_b.as_slice(), shape, stride }),
        da: TensorMut { data: dev_da.as_slice_mut(), shape, stride },
        db: TensorMut { data: dev_db.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got_da = vec![bf16::from_f32(0.0); numel];
    let mut got_db = vec![bf16::from_f32(0.0); numel];
    dev_da.copy_to_host(&mut got_da).expect("download da");
    dev_db.copy_to_host(&mut got_db).expect("download db");
    for i in 0..numel {
        let exp_da = host_dy[i] * host_b[i];
        let exp_db = host_dy[i] * host_a[i];
        let gd = got_da[i].to_f32();
        let ed = exp_da.to_f32();
        let tol = ed.abs().max(1.0) * BF16_EPS;
        assert!((gd - ed).abs() <= tol, "mul backward bf16 da @ {i}: got {gd}, exp {ed}");
        let gdb = got_db[i].to_f32();
        let edb = exp_db.to_f32();
        let tol = edb.abs().max(1.0) * BF16_EPS;
        assert!((gdb - edb).abs() <= tol, "mul backward bf16 db @ {i}: got {gdb}, exp {edb}");
    }
}

#[test]
#[ignore]
fn mul_backward_f64_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.5 - 17.25).collect();
    let host_a:  Vec<f64> = (0..numel).map(|i| (i as f64) * 0.125 + 1.0).collect();
    let host_b:  Vec<f64> = (0..numel).map(|i| (i as f64) * 0.0625 - 3.0).collect();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_da: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc da");
    let mut dev_db: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc db");
    let stride = contiguous_stride(shape);
    let desc = BinaryBackwardDescriptor {
        kind: BinaryKind::Mul,
        shape,
        element: ElementKind::F64,
    };
    let plan = BinaryBackwardPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = BinaryBackwardArgs::<f64, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        a: Some(TensorRef { data: dev_a.as_slice(), shape, stride }),
        b: Some(TensorRef { data: dev_b.as_slice(), shape, stride }),
        da: TensorMut { data: dev_da.as_slice_mut(), shape, stride },
        db: TensorMut { data: dev_db.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got_da = vec![0f64; numel];
    let mut got_db = vec![0f64; numel];
    dev_da.copy_to_host(&mut got_da).expect("download da");
    dev_db.copy_to_host(&mut got_db).expect("download db");
    for i in 0..numel {
        let exp_da = host_dy[i] * host_b[i];
        let exp_db = host_dy[i] * host_a[i];
        assert_eq!(got_da[i].to_bits(), exp_da.to_bits(), "mul backward f64 da @ {i}");
        assert_eq!(got_db[i].to_bits(), exp_db.to_bits(), "mul backward f64 db @ {i}");
    }
}
