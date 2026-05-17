//! Real-GPU smoke test for `TernaryBackwardPlan<T, N> + TernaryKind::Addcmul`
//! across f32 / f16 / bf16 / f64.
//!
//! Forward: `y = a + scale * b * c`. Backward:
//!   da = dy
//!   db = dy * (scale * c)
//!   dc = dy * (scale * b)
//!
//! Tolerances:
//! - f32 / f64: `4 * dtype_eps` relative — two unfused mul rounds; tiny
//!   denormal-region drift possible against the host reference.
//! - f16 / bf16: 2-ULP relative (matches the FW `4 * dtype_eps` band for
//!   uniformity).
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test ternary_addcmul_backward_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef, TernaryBackwardArgs,
    TernaryBackwardDescriptor, TernaryBackwardPlan, TernaryKind, Workspace,
};
use half::{bf16, f16};

const F32_EPS: f32 = f32::EPSILON;
const F64_EPS: f64 = f64::EPSILON;
const F16_EPS: f32 = 9.77e-4_f32; // 2^-10
const BF16_EPS: f32 = 7.81e-3_f32; // 2^-7

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn addcmul_backward_f32_3d() {
    let (ctx, stream) = setup();
    let shape = [4i32, 64, 64];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let scale: f32 = 0.125;
    let host_dy: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.25 - 7.5).collect();
    let host_a: Vec<f32> = (0..numel).map(|i| ((i % 31) as f32) * 0.125 - 1.0).collect();
    let host_b: Vec<f32> = (0..numel).map(|i| ((i % 29) as f32) * 0.0625 + 0.5).collect();
    let host_c: Vec<f32> = (0..numel).map(|i| ((i % 23) as f32) * 0.125 - 1.0).collect();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).unwrap();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).unwrap();
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).unwrap();
    let dev_c = DeviceBuffer::from_slice(&ctx, &host_c).unwrap();
    let mut dev_da: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let mut dev_db: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let mut dev_dc: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let stride = contiguous_stride(shape);
    let desc = TernaryBackwardDescriptor {
        kind: TernaryKind::Addcmul,
        shape,
        element: ElementKind::F32,
        scale,
    };
    let plan = TernaryBackwardPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = TernaryBackwardArgs::<f32, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        c: TensorRef { data: dev_c.as_slice(), shape, stride },
        da: TensorMut { data: dev_da.as_slice_mut(), shape, stride },
        db: TensorMut { data: dev_db.as_slice_mut(), shape, stride },
        dc: TensorMut { data: dev_dc.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got_da = vec![0f32; numel];
    let mut got_db = vec![0f32; numel];
    let mut got_dc = vec![0f32; numel];
    dev_da.copy_to_host(&mut got_da).unwrap();
    dev_db.copy_to_host(&mut got_db).unwrap();
    dev_dc.copy_to_host(&mut got_dc).unwrap();
    for i in 0..numel {
        // Match kernel ordering: t = scale*c, db = dy*t.
        let t1 = scale * host_c[i];
        let ed_b = host_dy[i] * t1;
        let t2 = scale * host_b[i];
        let ed_c = host_dy[i] * t2;
        let ed_a = host_dy[i];
        let tol_b = ed_b.abs().max(1.0) * 4.0 * F32_EPS;
        let tol_c = ed_c.abs().max(1.0) * 4.0 * F32_EPS;
        assert_eq!(got_da[i].to_bits(), ed_a.to_bits(), "addcmul BW f32 da @ {i}");
        assert!((got_db[i] - ed_b).abs() <= tol_b,
            "addcmul BW f32 db @ {i}: got {} exp {}", got_db[i], ed_b);
        assert!((got_dc[i] - ed_c).abs() <= tol_c,
            "addcmul BW f32 dc @ {i}: got {} exp {}", got_dc[i], ed_c);
    }
}

#[test]
#[ignore]
fn addcmul_backward_f16_3d() {
    let (ctx, stream) = setup();
    let shape = [4i32, 32, 64];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let scale: f32 = 0.5;
    let host_dy: Vec<f16> = (0..numel).map(|i| f16::from_f32(((i % 41) as f32) * 0.125 - 2.5)).collect();
    let host_a: Vec<f16> = (0..numel).map(|i| f16::from_f32(((i % 37) as f32) * 0.0625 + 0.25)).collect();
    let host_b: Vec<f16> = (0..numel).map(|i| f16::from_f32(((i % 29) as f32) * 0.0625 - 1.0)).collect();
    let host_c: Vec<f16> = (0..numel).map(|i| f16::from_f32(((i % 23) as f32) * 0.125 - 1.0)).collect();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).unwrap();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).unwrap();
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).unwrap();
    let dev_c = DeviceBuffer::from_slice(&ctx, &host_c).unwrap();
    let mut dev_da: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let mut dev_db: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let mut dev_dc: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let stride = contiguous_stride(shape);
    let desc = TernaryBackwardDescriptor {
        kind: TernaryKind::Addcmul,
        shape,
        element: ElementKind::F16,
        scale,
    };
    let plan = TernaryBackwardPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = TernaryBackwardArgs::<f16, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        c: TensorRef { data: dev_c.as_slice(), shape, stride },
        da: TensorMut { data: dev_da.as_slice_mut(), shape, stride },
        db: TensorMut { data: dev_db.as_slice_mut(), shape, stride },
        dc: TensorMut { data: dev_dc.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got_da = vec![f16::from_f32(0.0); numel];
    let mut got_db = vec![f16::from_f32(0.0); numel];
    let mut got_dc = vec![f16::from_f32(0.0); numel];
    dev_da.copy_to_host(&mut got_da).unwrap();
    dev_db.copy_to_host(&mut got_db).unwrap();
    dev_dc.copy_to_host(&mut got_dc).unwrap();
    for i in 0..numel {
        let dy = host_dy[i].to_f32();
        let b = host_b[i].to_f32();
        let c = host_c[i].to_f32();
        let ed_b = f16::from_f32(dy * (scale * c)).to_f32();
        let ed_c = f16::from_f32(dy * (scale * b)).to_f32();
        let gd_a = got_da[i].to_f32();
        let gd_b = got_db[i].to_f32();
        let gd_c = got_dc[i].to_f32();
        assert_eq!(got_da[i].to_bits(), host_dy[i].to_bits(), "addcmul BW f16 da @ {i}");
        let tol_b = ed_b.abs().max(1.0) * 2.0 * F16_EPS;
        let tol_c = ed_c.abs().max(1.0) * 2.0 * F16_EPS;
        assert!((gd_b - ed_b).abs() <= tol_b,
            "addcmul BW f16 db @ {i}: got {gd_b} exp {ed_b}");
        assert!((gd_c - ed_c).abs() <= tol_c,
            "addcmul BW f16 dc @ {i}: got {gd_c} exp {ed_c}");
        let _ = gd_a;
    }
}

#[test]
#[ignore]
fn addcmul_backward_bf16_3d() {
    let (ctx, stream) = setup();
    let shape = [4i32, 32, 64];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let scale: f32 = 0.5;
    let host_dy: Vec<bf16> = (0..numel).map(|i| bf16::from_f32(((i % 41) as f32) * 0.125 - 2.5)).collect();
    let host_a: Vec<bf16> = (0..numel).map(|i| bf16::from_f32(((i % 37) as f32) * 0.0625 + 0.25)).collect();
    let host_b: Vec<bf16> = (0..numel).map(|i| bf16::from_f32(((i % 29) as f32) * 0.0625 - 1.0)).collect();
    let host_c: Vec<bf16> = (0..numel).map(|i| bf16::from_f32(((i % 23) as f32) * 0.125 - 1.0)).collect();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).unwrap();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).unwrap();
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).unwrap();
    let dev_c = DeviceBuffer::from_slice(&ctx, &host_c).unwrap();
    let mut dev_da: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let mut dev_db: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let mut dev_dc: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let stride = contiguous_stride(shape);
    let desc = TernaryBackwardDescriptor {
        kind: TernaryKind::Addcmul,
        shape,
        element: ElementKind::Bf16,
        scale,
    };
    let plan = TernaryBackwardPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = TernaryBackwardArgs::<bf16, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        c: TensorRef { data: dev_c.as_slice(), shape, stride },
        da: TensorMut { data: dev_da.as_slice_mut(), shape, stride },
        db: TensorMut { data: dev_db.as_slice_mut(), shape, stride },
        dc: TensorMut { data: dev_dc.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got_da = vec![bf16::from_f32(0.0); numel];
    let mut got_db = vec![bf16::from_f32(0.0); numel];
    let mut got_dc = vec![bf16::from_f32(0.0); numel];
    dev_da.copy_to_host(&mut got_da).unwrap();
    dev_db.copy_to_host(&mut got_db).unwrap();
    dev_dc.copy_to_host(&mut got_dc).unwrap();
    for i in 0..numel {
        let dy = host_dy[i].to_f32();
        let b = host_b[i].to_f32();
        let c = host_c[i].to_f32();
        let ed_b = bf16::from_f32(dy * (scale * c)).to_f32();
        let ed_c = bf16::from_f32(dy * (scale * b)).to_f32();
        let gd_b = got_db[i].to_f32();
        let gd_c = got_dc[i].to_f32();
        assert_eq!(got_da[i].to_bits(), host_dy[i].to_bits(), "addcmul BW bf16 da @ {i}");
        let tol_b = ed_b.abs().max(1.0) * 2.0 * BF16_EPS;
        let tol_c = ed_c.abs().max(1.0) * 2.0 * BF16_EPS;
        assert!((gd_b - ed_b).abs() <= tol_b,
            "addcmul BW bf16 db @ {i}: got {gd_b} exp {ed_b}");
        assert!((gd_c - ed_c).abs() <= tol_c,
            "addcmul BW bf16 dc @ {i}: got {gd_c} exp {ed_c}");
    }
}

#[test]
#[ignore]
fn addcmul_backward_f64_3d() {
    let (ctx, stream) = setup();
    let shape = [4i32, 64, 64];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let scale: f32 = 0.125;
    let scale_d: f64 = scale as f64;
    let host_dy: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.25 - 7.5).collect();
    let host_a: Vec<f64> = (0..numel).map(|i| ((i % 31) as f64) * 0.125 - 1.0).collect();
    let host_b: Vec<f64> = (0..numel).map(|i| ((i % 29) as f64) * 0.0625 + 0.5).collect();
    let host_c: Vec<f64> = (0..numel).map(|i| ((i % 23) as f64) * 0.125 - 1.0).collect();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).unwrap();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).unwrap();
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).unwrap();
    let dev_c = DeviceBuffer::from_slice(&ctx, &host_c).unwrap();
    let mut dev_da: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let mut dev_db: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let mut dev_dc: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let stride = contiguous_stride(shape);
    let desc = TernaryBackwardDescriptor {
        kind: TernaryKind::Addcmul,
        shape,
        element: ElementKind::F64,
        scale,
    };
    let plan = TernaryBackwardPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = TernaryBackwardArgs::<f64, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        c: TensorRef { data: dev_c.as_slice(), shape, stride },
        da: TensorMut { data: dev_da.as_slice_mut(), shape, stride },
        db: TensorMut { data: dev_db.as_slice_mut(), shape, stride },
        dc: TensorMut { data: dev_dc.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got_da = vec![0f64; numel];
    let mut got_db = vec![0f64; numel];
    let mut got_dc = vec![0f64; numel];
    dev_da.copy_to_host(&mut got_da).unwrap();
    dev_db.copy_to_host(&mut got_db).unwrap();
    dev_dc.copy_to_host(&mut got_dc).unwrap();
    for i in 0..numel {
        let ed_b = host_dy[i] * (scale_d * host_c[i]);
        let ed_c = host_dy[i] * (scale_d * host_b[i]);
        let ed_a = host_dy[i];
        let tol_b = ed_b.abs().max(1.0) * 4.0 * F64_EPS;
        let tol_c = ed_c.abs().max(1.0) * 4.0 * F64_EPS;
        assert_eq!(got_da[i].to_bits(), ed_a.to_bits(), "addcmul BW f64 da @ {i}");
        assert!((got_db[i] - ed_b).abs() <= tol_b,
            "addcmul BW f64 db @ {i}: got {} exp {}", got_db[i], ed_b);
        assert!((got_dc[i] - ed_c).abs() <= tol_c,
            "addcmul BW f64 dc @ {i}: got {} exp {}", got_dc[i], ed_c);
    }
}
