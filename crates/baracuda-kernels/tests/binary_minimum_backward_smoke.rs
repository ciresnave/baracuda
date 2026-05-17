//! Real-GPU smoke test for `BinaryBackwardPlan<T, N> + BinaryKind::Minimum`.
//!
//! Mirror of `binary_maximum_backward_smoke.rs` with the gradient flipped
//! to the `<` operand. Same tie / NaN convention.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test binary_minimum_backward_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BinaryBackwardArgs, BinaryBackwardDescriptor, BinaryBackwardPlan,
    BinaryKind, ElementKind, PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

const F16_EPS: f32 = 9.77e-4;
const BF16_EPS: f32 = 7.81e-3;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn build_inputs_f32(numel: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let host_dy: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.5 - 17.25).collect();
    let mut host_a: Vec<f32> = Vec::with_capacity(numel);
    let mut host_b: Vec<f32> = Vec::with_capacity(numel);
    for i in 0..numel {
        let m = i % 8;
        let base = (i as f32) * 0.125 - 1.0;
        match m {
            0 | 1 | 2 | 3 => {
                host_a.push(base + 1.0);
                host_b.push(base);
            }
            4 | 5 => {
                host_a.push(base);
                host_b.push(base + 1.0);
            }
            6 => {
                host_a.push(base);
                host_b.push(base);
            }
            _ => {
                host_a.push(f32::NAN);
                host_b.push(base);
            }
        }
    }
    (host_dy, host_a, host_b)
}

fn ref_minimum_backward(dy: f32, a: f32, b: f32) -> (f32, f32) {
    // PyTorch derivatives.yaml `minimum`: flip < / >.
    if a == b {
        let half = dy * 0.5;
        (half, half)
    } else {
        let da = if a > b { 0.0 } else { dy };
        let db = if b > a { 0.0 } else { dy };
        (da, db)
    }
}

#[test]
#[ignore]
fn minimum_backward_f32_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let (host_dy, host_a, host_b) = build_inputs_f32(numel);
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_da: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc da");
    let mut dev_db: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc db");
    let stride = contiguous_stride(shape);
    let desc = BinaryBackwardDescriptor {
        kind: BinaryKind::Minimum,
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
        let (exp_da, exp_db) = ref_minimum_backward(host_dy[i], host_a[i], host_b[i]);
        assert_eq!(got_da[i].to_bits(), exp_da.to_bits(),
            "minimum bw f32 da @ {i}: a={}, b={}, dy={}", host_a[i], host_b[i], host_dy[i]);
        assert_eq!(got_db[i].to_bits(), exp_db.to_bits(),
            "minimum bw f32 db @ {i}: a={}, b={}, dy={}", host_a[i], host_b[i], host_dy[i]);
    }
}

#[test]
#[ignore]
fn minimum_backward_f64_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let (host_dy_f32, host_a_f32, host_b_f32) = build_inputs_f32(numel);
    let host_dy: Vec<f64> = host_dy_f32.iter().map(|&x| x as f64).collect();
    let host_a: Vec<f64> = host_a_f32.iter().map(|&x| x as f64).collect();
    let host_b: Vec<f64> = host_b_f32.iter().map(|&x| x as f64).collect();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_da: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc da");
    let mut dev_db: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc db");
    let stride = contiguous_stride(shape);
    let desc = BinaryBackwardDescriptor {
        kind: BinaryKind::Minimum,
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
        let (exp_da_f32, exp_db_f32) =
            ref_minimum_backward(host_dy_f32[i], host_a_f32[i], host_b_f32[i]);
        let exp_da = exp_da_f32 as f64;
        let exp_db = exp_db_f32 as f64;
        assert_eq!(got_da[i].to_bits(), exp_da.to_bits(), "minimum bw f64 da @ {i}");
        assert_eq!(got_db[i].to_bits(), exp_db.to_bits(), "minimum bw f64 db @ {i}");
    }
}

#[test]
#[ignore]
fn minimum_backward_f16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let (host_dy_f32, host_a_f32, host_b_f32) = build_inputs_f32(numel);
    let host_dy: Vec<f16> = host_dy_f32.iter().map(|&x| f16::from_f32(x.clamp(-32.0, 32.0))).collect();
    let host_a: Vec<f16> = host_a_f32.iter().map(|&x| {
        if x.is_nan() { f16::from_f32(0.0) } else { f16::from_f32(x.clamp(-32.0, 32.0)) }
    }).collect();
    let host_b: Vec<f16> = host_b_f32.iter().map(|&x| f16::from_f32(x.clamp(-32.0, 32.0))).collect();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_da: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc da");
    let mut dev_db: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc db");
    let stride = contiguous_stride(shape);
    let desc = BinaryBackwardDescriptor {
        kind: BinaryKind::Minimum,
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
        let dy = host_dy[i].to_f32();
        let a = host_a[i].to_f32();
        let b = host_b[i].to_f32();
        let (exp_da, exp_db) = ref_minimum_backward(dy, a, b);
        let gd = got_da[i].to_f32();
        let tol = exp_da.abs().max(1.0) * F16_EPS;
        assert!((gd - exp_da).abs() <= tol, "minimum bw f16 da @ {i}: got {gd}, exp {exp_da}");
        let gdb = got_db[i].to_f32();
        let tol = exp_db.abs().max(1.0) * F16_EPS;
        assert!((gdb - exp_db).abs() <= tol, "minimum bw f16 db @ {i}: got {gdb}, exp {exp_db}");
    }
}

#[test]
#[ignore]
fn minimum_backward_bf16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let (host_dy_f32, host_a_f32, host_b_f32) = build_inputs_f32(numel);
    let host_dy: Vec<bf16> = host_dy_f32.iter().map(|&x| bf16::from_f32(x)).collect();
    let host_a: Vec<bf16> = host_a_f32.iter().map(|&x| {
        if x.is_nan() { bf16::from_f32(0.0) } else { bf16::from_f32(x) }
    }).collect();
    let host_b: Vec<bf16> = host_b_f32.iter().map(|&x| bf16::from_f32(x)).collect();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_da: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc da");
    let mut dev_db: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc db");
    let stride = contiguous_stride(shape);
    let desc = BinaryBackwardDescriptor {
        kind: BinaryKind::Minimum,
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
        let dy = host_dy[i].to_f32();
        let a = host_a[i].to_f32();
        let b = host_b[i].to_f32();
        let (exp_da, exp_db) = ref_minimum_backward(dy, a, b);
        let gd = got_da[i].to_f32();
        let tol = exp_da.abs().max(1.0) * BF16_EPS;
        assert!((gd - exp_da).abs() <= tol, "minimum bw bf16 da @ {i}: got {gd}, exp {exp_da}");
        let gdb = got_db[i].to_f32();
        let tol = exp_db.abs().max(1.0) * BF16_EPS;
        assert!((gdb - exp_db).abs() <= tol, "minimum bw bf16 db @ {i}: got {gdb}, exp {exp_db}");
    }
}
