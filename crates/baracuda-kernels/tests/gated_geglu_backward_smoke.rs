//! Real-GPU smoke test for the gated GeGLU backward (exact, erf-based).
//!
//! Forward: `y = a · gelu(b)`. Backward (saved x):
//!   da = dy · gelu(b)
//!   db = dy · a · (Φ(b) + b·φ(b))
//! Tolerance: 8·eps (f32/f64), 8·F16_EPS / 8·BF16_EPS (f16/bf16).

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, GatedActivationBackwardArgs, GatedActivationBackwardDescriptor,
    GatedActivationBackwardPlan, GatedActivationKind, PlanPreference, TensorMut, TensorRef,
    Workspace,
};
use half::{bf16, f16};

const F16_EPS: f32 = 9.77e-4;
const BF16_EPS: f32 = 7.81e-3;
const INV_SQRT2_F: f32 = 0.70710678118654752440;
const INV_SQRT2PI_F: f32 = 0.39894228040143267794;
const INV_SQRT2_D: f64 = 0.70710678118654752440;
const INV_SQRT2PI_D: f64 = 0.39894228040143267794;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn geglu_bw_ref_f32(dy: f32, a: f32, b: f32) -> (f32, f32) {
    let cdf = if b >= 0.0 {
        0.5 * (1.0 + libm::erff(b * INV_SQRT2_F))
    } else {
        0.5 * libm::erfcf(-b * INV_SQRT2_F)
    };
    let pdf = INV_SQRT2PI_F * (-0.5 * b * b).exp();
    let gelu_b = b * cdf;
    let gelu_prime_b = cdf + b * pdf;
    (dy * gelu_b, dy * a * gelu_prime_b)
}
fn geglu_bw_ref_f64(dy: f64, a: f64, b: f64) -> (f64, f64) {
    let cdf = if b >= 0.0 {
        0.5 * (1.0 + libm::erf(b * INV_SQRT2_D))
    } else {
        0.5 * libm::erfc(-b * INV_SQRT2_D)
    };
    let pdf = INV_SQRT2PI_D * (-0.5 * b * b).exp();
    let gelu_b = b * cdf;
    let gelu_prime_b = cdf + b * pdf;
    (dy * gelu_b, dy * a * gelu_prime_b)
}

fn cpu_ref_bw_f32(x: &[f32], dy: &[f32], input_shape: &[i32], split_dim: usize) -> Vec<f32> {
    let half = input_shape[split_dim] as usize / 2;
    let mut out_shape: Vec<i32> = input_shape.to_vec();
    out_shape[split_dim] = half as i32;
    let in_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let out_numel: usize = out_shape.iter().map(|&d| d as usize).product();
    let n = input_shape.len();
    let mut in_strides = vec![0i64; n];
    in_strides[n - 1] = 1;
    for d in (0..n - 1).rev() { in_strides[d] = in_strides[d + 1] * input_shape[d + 1] as i64; }
    let mut out_strides = vec![0i64; n];
    out_strides[n - 1] = 1;
    for d in (0..n - 1).rev() { out_strides[d] = out_strides[d + 1] * out_shape[d + 1] as i64; }
    let x_half_offset = (half as i64) * in_strides[split_dim];
    let mut dx = vec![0f32; in_numel];
    for i in 0..out_numel {
        let mut linear = i as i64;
        let mut off_x = 0i64; let mut off_dy = 0i64;
        for d in (0..n).rev() {
            let s = out_shape[d] as i64;
            let coord = linear % s; linear /= s;
            off_x += coord * in_strides[d];
            off_dy += coord * out_strides[d];
        }
        let a = x[off_x as usize];
        let b = x[(off_x + x_half_offset) as usize];
        let dyv = dy[off_dy as usize];
        let (da, db) = geglu_bw_ref_f32(dyv, a, b);
        dx[off_x as usize] = da;
        dx[(off_x + x_half_offset) as usize] = db;
    }
    dx
}
fn cpu_ref_bw_f64(x: &[f64], dy: &[f64], input_shape: &[i32], split_dim: usize) -> Vec<f64> {
    let half = input_shape[split_dim] as usize / 2;
    let mut out_shape: Vec<i32> = input_shape.to_vec();
    out_shape[split_dim] = half as i32;
    let in_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let out_numel: usize = out_shape.iter().map(|&d| d as usize).product();
    let n = input_shape.len();
    let mut in_strides = vec![0i64; n];
    in_strides[n - 1] = 1;
    for d in (0..n - 1).rev() { in_strides[d] = in_strides[d + 1] * input_shape[d + 1] as i64; }
    let mut out_strides = vec![0i64; n];
    out_strides[n - 1] = 1;
    for d in (0..n - 1).rev() { out_strides[d] = out_strides[d + 1] * out_shape[d + 1] as i64; }
    let x_half_offset = (half as i64) * in_strides[split_dim];
    let mut dx = vec![0f64; in_numel];
    for i in 0..out_numel {
        let mut linear = i as i64;
        let mut off_x = 0i64; let mut off_dy = 0i64;
        for d in (0..n).rev() {
            let s = out_shape[d] as i64;
            let coord = linear % s; linear /= s;
            off_x += coord * in_strides[d];
            off_dy += coord * out_strides[d];
        }
        let a = x[off_x as usize];
        let b = x[(off_x + x_half_offset) as usize];
        let dyv = dy[off_dy as usize];
        let (da, db) = geglu_bw_ref_f64(dyv, a, b);
        dx[off_x as usize] = da;
        dx[(off_x + x_half_offset) as usize] = db;
    }
    dx
}

#[test]
#[ignore]
fn geglu_bw_f32() {
    let (ctx, stream) = setup();
    let input_shape = [2i32, 4, 8];
    let split_dim: u8 = 1;
    let in_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f32> = (0..in_numel).map(|i| ((i % 200) as f32) * 0.05 - 5.0).collect();
    let desc = GatedActivationBackwardDescriptor { kind: GatedActivationKind::GeGlu, input_shape, split_dim, element: ElementKind::F32 };
    let output_shape = desc.output_shape();
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<f32> = (0..out_numel).map(|i| (i as f32) * 0.5 - 8.0).collect();
    let expected = cpu_ref_bw_f32(&host_x, &host_dy, &input_shape, split_dim as usize);
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, in_numel).expect("alloc dx");
    let plan = GatedActivationBackwardPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = GatedActivationBackwardArgs::<f32, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape: output_shape, stride: contiguous_stride(output_shape) },
        x:  TensorRef { data: dev_x.as_slice(),  shape: input_shape,  stride: contiguous_stride(input_shape) },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape: input_shape, stride: contiguous_stride(input_shape) },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; in_numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for i in 0..in_numel {
        let exp = expected[i];
        let tol = exp.abs().max(1.0) * 8.0 * f32::EPSILON;
        assert!((got[i] - exp).abs() <= tol, "geglu bw f32 @ {i}: got {} exp {}", got[i], exp);
    }
}

#[test]
#[ignore]
fn geglu_bw_f64() {
    let (ctx, stream) = setup();
    let input_shape = [3i32, 32];
    let split_dim: u8 = 1;
    let in_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f64> = (0..in_numel).map(|i| ((i % 200) as f64) * 0.05 - 5.0).collect();
    let desc = GatedActivationBackwardDescriptor { kind: GatedActivationKind::GeGlu, input_shape, split_dim, element: ElementKind::F64 };
    let output_shape = desc.output_shape();
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<f64> = (0..out_numel).map(|i| (i as f64) * 0.5 - 8.0).collect();
    let expected = cpu_ref_bw_f64(&host_x, &host_dy, &input_shape, split_dim as usize);
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, in_numel).expect("alloc dx");
    let plan = GatedActivationBackwardPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = GatedActivationBackwardArgs::<f64, 2> {
        dy: TensorRef { data: dev_dy.as_slice(), shape: output_shape, stride: contiguous_stride(output_shape) },
        x:  TensorRef { data: dev_x.as_slice(),  shape: input_shape,  stride: contiguous_stride(input_shape) },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape: input_shape, stride: contiguous_stride(input_shape) },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f64; in_numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for i in 0..in_numel {
        let exp = expected[i];
        let tol = exp.abs().max(1.0) * 8.0 * f64::EPSILON;
        assert!((got[i] - exp).abs() <= tol, "geglu bw f64 @ {i}: got {} exp {}", got[i], exp);
    }
}

#[test]
#[ignore]
fn geglu_bw_f16() {
    let (ctx, stream) = setup();
    let input_shape = [2i32, 4, 8];
    let split_dim: u8 = 1;
    let in_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f16> = (0..in_numel).map(|i| f16::from_f32(((i % 200) as f32) * 0.05 - 5.0)).collect();
    let desc = GatedActivationBackwardDescriptor { kind: GatedActivationKind::GeGlu, input_shape, split_dim, element: ElementKind::F16 };
    let output_shape = desc.output_shape();
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<f16> = (0..out_numel).map(|i| f16::from_f32(((i % 41) as f32) * 0.25 - 5.0)).collect();
    let x_f32: Vec<f32> = host_x.iter().map(|v| v.to_f32()).collect();
    let dy_f32: Vec<f32> = host_dy.iter().map(|v| v.to_f32()).collect();
    let expected = cpu_ref_bw_f32(&x_f32, &dy_f32, &input_shape, split_dim as usize);
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, in_numel).expect("alloc dx");
    let plan = GatedActivationBackwardPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = GatedActivationBackwardArgs::<f16, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape: output_shape, stride: contiguous_stride(output_shape) },
        x:  TensorRef { data: dev_x.as_slice(),  shape: input_shape,  stride: contiguous_stride(input_shape) },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape: input_shape, stride: contiguous_stride(input_shape) },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![f16::from_f32(0.0); in_numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for i in 0..in_numel {
        let exp = expected[i];
        let g = got[i].to_f32();
        let tol = exp.abs().max(1.0) * 8.0 * F16_EPS;
        assert!((g - exp).abs() <= tol, "geglu bw f16 @ {i}: got {g} exp {exp}");
    }
}

#[test]
#[ignore]
fn geglu_bw_bf16() {
    let (ctx, stream) = setup();
    let input_shape = [2i32, 4, 8];
    let split_dim: u8 = 1;
    let in_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<bf16> = (0..in_numel).map(|i| bf16::from_f32(((i % 200) as f32) * 0.05 - 5.0)).collect();
    let desc = GatedActivationBackwardDescriptor { kind: GatedActivationKind::GeGlu, input_shape, split_dim, element: ElementKind::Bf16 };
    let output_shape = desc.output_shape();
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<bf16> = (0..out_numel).map(|i| bf16::from_f32(((i % 41) as f32) * 0.25 - 5.0)).collect();
    let x_f32: Vec<f32> = host_x.iter().map(|v| v.to_f32()).collect();
    let dy_f32: Vec<f32> = host_dy.iter().map(|v| v.to_f32()).collect();
    let expected = cpu_ref_bw_f32(&x_f32, &dy_f32, &input_shape, split_dim as usize);
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, in_numel).expect("alloc dx");
    let plan = GatedActivationBackwardPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = GatedActivationBackwardArgs::<bf16, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape: output_shape, stride: contiguous_stride(output_shape) },
        x:  TensorRef { data: dev_x.as_slice(),  shape: input_shape,  stride: contiguous_stride(input_shape) },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape: input_shape, stride: contiguous_stride(input_shape) },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![bf16::from_f32(0.0); in_numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for i in 0..in_numel {
        let exp = expected[i];
        let g = got[i].to_f32();
        let tol = exp.abs().max(1.0) * 8.0 * BF16_EPS;
        assert!((g - exp).abs() <= tol, "geglu bw bf16 @ {i}: got {g} exp {exp}");
    }
}
