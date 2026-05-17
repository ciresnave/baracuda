//! Real-GPU smoke test for `BinaryParamBackwardPlan + BinaryKind::Lerp`.
//!
//! Forward: `y = a + weight·(b - a)`. Backward: `da = (1 - weight)·dy`,
//! `db = weight·dy`. No saves. Tolerance: `4·eps` relative — each output
//! is one mul (the `(1 - weight)` factor for `da` is computed in f32 at
//! kernel-build time on the f64 path, contributing negligible error).

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BinaryKind, BinaryParamBackwardArgs, BinaryParamBackwardDescriptor,
    BinaryParamBackwardPlan, ElementKind, PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

const F16_EPS: f32 = 9.77e-4;
const BF16_EPS: f32 = 7.81e-3;

const W: f32 = 0.3;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn lerp_backward_f32_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.5 - 17.25).collect();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_da: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc da");
    let mut dev_db: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc db");
    let stride = contiguous_stride(shape);
    let desc = BinaryParamBackwardDescriptor {
        kind: BinaryKind::Lerp,
        shape,
        element: ElementKind::F32,
        param: W,
    };
    let plan = BinaryParamBackwardPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = BinaryParamBackwardArgs::<f32, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
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
        let exp_da = (1.0 - W) * host_dy[i];
        let exp_db = W * host_dy[i];
        let tol_a = exp_da.abs().max(1.0) * 4.0 * f32::EPSILON;
        let tol_b = exp_db.abs().max(1.0) * 4.0 * f32::EPSILON;
        assert!(
            (got_da[i] - exp_da).abs() <= tol_a,
            "lerp bw da f32 @ {i}: got {} exp {} (diff {})", got_da[i], exp_da, (got_da[i] - exp_da).abs()
        );
        assert!(
            (got_db[i] - exp_db).abs() <= tol_b,
            "lerp bw db f32 @ {i}: got {} exp {} (diff {})", got_db[i], exp_db, (got_db[i] - exp_db).abs()
        );
    }
}

#[test]
#[ignore]
fn lerp_backward_f64_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.5 - 17.25).collect();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_da: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc da");
    let mut dev_db: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc db");
    let stride = contiguous_stride(shape);
    let desc = BinaryParamBackwardDescriptor {
        kind: BinaryKind::Lerp,
        shape,
        element: ElementKind::F64,
        param: W,
    };
    let plan = BinaryParamBackwardPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = BinaryParamBackwardArgs::<f64, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        da: TensorMut { data: dev_da.as_slice_mut(), shape, stride },
        db: TensorMut { data: dev_db.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got_da = vec![0f64; numel];
    let mut got_db = vec![0f64; numel];
    dev_da.copy_to_host(&mut got_da).expect("download da");
    dev_db.copy_to_host(&mut got_db).expect("download db");
    let dw = W as f64;
    for i in 0..numel {
        let exp_da = (1.0 - dw) * host_dy[i];
        let exp_db = dw * host_dy[i];
        let tol_a = exp_da.abs().max(1.0) * 4.0 * f64::EPSILON;
        let tol_b = exp_db.abs().max(1.0) * 4.0 * f64::EPSILON;
        assert!(
            (got_da[i] - exp_da).abs() <= tol_a,
            "lerp bw da f64 @ {i}: got {} exp {}", got_da[i], exp_da
        );
        assert!(
            (got_db[i] - exp_db).abs() <= tol_b,
            "lerp bw db f64 @ {i}: got {} exp {}", got_db[i], exp_db
        );
    }
}

#[test]
#[ignore]
fn lerp_backward_f16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32(((i % 81) as f32) * 0.25 - 10.0))
        .collect();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_da: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc da");
    let mut dev_db: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc db");
    let stride = contiguous_stride(shape);
    let desc = BinaryParamBackwardDescriptor {
        kind: BinaryKind::Lerp,
        shape,
        element: ElementKind::F16,
        param: W,
    };
    let plan = BinaryParamBackwardPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = BinaryParamBackwardArgs::<f16, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
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
        let dy_f = host_dy[i].to_f32();
        let exp_da = (1.0 - W) * dy_f;
        let exp_db = W * dy_f;
        let tol_a = exp_da.abs().max(1.0) * 4.0 * F16_EPS;
        let tol_b = exp_db.abs().max(1.0) * 4.0 * F16_EPS;
        let g_a = got_da[i].to_f32();
        let g_b = got_db[i].to_f32();
        assert!(
            (g_a - exp_da).abs() <= tol_a,
            "lerp bw da f16 @ {i}: got {g_a} exp {exp_da}"
        );
        assert!(
            (g_b - exp_db).abs() <= tol_b,
            "lerp bw db f16 @ {i}: got {g_b} exp {exp_db}"
        );
    }
}

#[test]
#[ignore]
fn lerp_backward_bf16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32(((i % 81) as f32) * 0.25 - 10.0))
        .collect();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_da: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc da");
    let mut dev_db: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc db");
    let stride = contiguous_stride(shape);
    let desc = BinaryParamBackwardDescriptor {
        kind: BinaryKind::Lerp,
        shape,
        element: ElementKind::Bf16,
        param: W,
    };
    let plan = BinaryParamBackwardPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = BinaryParamBackwardArgs::<bf16, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
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
        let dy_f = host_dy[i].to_f32();
        let exp_da = (1.0 - W) * dy_f;
        let exp_db = W * dy_f;
        let tol_a = exp_da.abs().max(1.0) * 4.0 * BF16_EPS;
        let tol_b = exp_db.abs().max(1.0) * 4.0 * BF16_EPS;
        let g_a = got_da[i].to_f32();
        let g_b = got_db[i].to_f32();
        assert!(
            (g_a - exp_da).abs() <= tol_a,
            "lerp bw da bf16 @ {i}: got {g_a} exp {exp_da}"
        );
        assert!(
            (g_b - exp_db).abs() <= tol_b,
            "lerp bw db bf16 @ {i}: got {g_b} exp {exp_db}"
        );
    }
}
