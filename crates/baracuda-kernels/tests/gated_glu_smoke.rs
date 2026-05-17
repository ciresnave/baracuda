//! Real-GPU smoke test for the gated GLU forward
//! (`GatedActivationPlan<T, N> + GatedActivationKind::Glu`).
//!
//! Forward: `y = a · sigmoid(b)`. Tolerances: 4·eps (f32/f64),
//! 4·F16_EPS / 4·BF16_EPS (f16/bf16) — sigmoid+exp inside.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, GatedActivationArgs, GatedActivationDescriptor,
    GatedActivationKind, GatedActivationPlan, PlanPreference, TensorMut, TensorRef, Workspace,
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

fn sigmoid_f32(b: f32) -> f32 {
    if b >= 0.0 { 1.0 / (1.0 + (-b).exp()) } else { let e = b.exp(); e / (1.0 + e) }
}
fn sigmoid_f64(b: f64) -> f64 {
    if b >= 0.0 { 1.0 / (1.0 + (-b).exp()) } else { let e = b.exp(); e / (1.0 + e) }
}

fn cpu_ref_f32(x: &[f32], input_shape: &[i32], split_dim: usize) -> Vec<f32> {
    let half = input_shape[split_dim] as usize / 2;
    let mut out_shape: Vec<i32> = input_shape.to_vec();
    out_shape[split_dim] = half as i32;
    let out_numel: usize = out_shape.iter().map(|&d| d as usize).product();
    let n = input_shape.len();
    let mut in_strides = vec![0i64; n];
    in_strides[n - 1] = 1;
    for d in (0..n - 1).rev() { in_strides[d] = in_strides[d + 1] * input_shape[d + 1] as i64; }
    let mut out_strides = vec![0i64; n];
    out_strides[n - 1] = 1;
    for d in (0..n - 1).rev() { out_strides[d] = out_strides[d + 1] * out_shape[d + 1] as i64; }
    let x_half_offset = (half as i64) * in_strides[split_dim];
    let mut y = vec![0f32; out_numel];
    for i in 0..out_numel {
        let mut linear = i as i64;
        let mut off_x = 0i64; let mut off_y = 0i64;
        for d in (0..n).rev() {
            let s = out_shape[d] as i64;
            let coord = linear % s; linear /= s;
            off_x += coord * in_strides[d];
            off_y += coord * out_strides[d];
        }
        let a = x[off_x as usize];
        let b = x[(off_x + x_half_offset) as usize];
        y[off_y as usize] = a * sigmoid_f32(b);
    }
    y
}
fn cpu_ref_f64(x: &[f64], input_shape: &[i32], split_dim: usize) -> Vec<f64> {
    let half = input_shape[split_dim] as usize / 2;
    let mut out_shape: Vec<i32> = input_shape.to_vec();
    out_shape[split_dim] = half as i32;
    let out_numel: usize = out_shape.iter().map(|&d| d as usize).product();
    let n = input_shape.len();
    let mut in_strides = vec![0i64; n];
    in_strides[n - 1] = 1;
    for d in (0..n - 1).rev() { in_strides[d] = in_strides[d + 1] * input_shape[d + 1] as i64; }
    let mut out_strides = vec![0i64; n];
    out_strides[n - 1] = 1;
    for d in (0..n - 1).rev() { out_strides[d] = out_strides[d + 1] * out_shape[d + 1] as i64; }
    let x_half_offset = (half as i64) * in_strides[split_dim];
    let mut y = vec![0f64; out_numel];
    for i in 0..out_numel {
        let mut linear = i as i64;
        let mut off_x = 0i64; let mut off_y = 0i64;
        for d in (0..n).rev() {
            let s = out_shape[d] as i64;
            let coord = linear % s; linear /= s;
            off_x += coord * in_strides[d];
            off_y += coord * out_strides[d];
        }
        let a = x[off_x as usize];
        let b = x[(off_x + x_half_offset) as usize];
        y[off_y as usize] = a * sigmoid_f64(b);
    }
    y
}

#[test]
#[ignore]
fn glu_fw_f32() {
    let (ctx, stream) = setup();
    let input_shape = [2i32, 4, 8];
    let split_dim: u8 = 1;
    let in_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f32> = (0..in_numel).map(|i| ((i % 200) as f32) * 0.05 - 5.0).collect();
    let expected = cpu_ref_f32(&host_x, &input_shape, split_dim as usize);
    let desc = GatedActivationDescriptor { kind: GatedActivationKind::Glu, input_shape, split_dim, element: ElementKind::F32 };
    let output_shape = desc.output_shape();
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc y");
    let plan = GatedActivationPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = GatedActivationArgs::<f32, 3> {
        x: TensorRef { data: dev_x.as_slice(), shape: input_shape, stride: contiguous_stride(input_shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: output_shape, stride: contiguous_stride(output_shape) },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; out_numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let allow = e.abs().max(1.0) * 4.0 * f32::EPSILON;
        assert!((g - e).abs() <= allow, "glu f32 @ {i}: got {g} exp {e}");
    }
}

#[test]
#[ignore]
fn glu_fw_f64() {
    let (ctx, stream) = setup();
    let input_shape = [3i32, 32];
    let split_dim: u8 = 1;
    let in_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f64> = (0..in_numel).map(|i| ((i % 200) as f64) * 0.05 - 5.0).collect();
    let expected = cpu_ref_f64(&host_x, &input_shape, split_dim as usize);
    let desc = GatedActivationDescriptor { kind: GatedActivationKind::Glu, input_shape, split_dim, element: ElementKind::F64 };
    let output_shape = desc.output_shape();
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc y");
    let plan = GatedActivationPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = GatedActivationArgs::<f64, 2> {
        x: TensorRef { data: dev_x.as_slice(), shape: input_shape, stride: contiguous_stride(input_shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: output_shape, stride: contiguous_stride(output_shape) },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f64; out_numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let allow = e.abs().max(1.0) * 4.0 * f64::EPSILON;
        assert!((g - e).abs() <= allow, "glu f64 @ {i}: got {g} exp {e}");
    }
}

#[test]
#[ignore]
fn glu_fw_f16() {
    let (ctx, stream) = setup();
    let input_shape = [2i32, 4, 8];
    let split_dim: u8 = 1;
    let in_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f16> = (0..in_numel).map(|i| f16::from_f32(((i % 200) as f32) * 0.05 - 5.0)).collect();
    let x_f32: Vec<f32> = host_x.iter().map(|v| v.to_f32()).collect();
    let expected = cpu_ref_f32(&x_f32, &input_shape, split_dim as usize);
    let desc = GatedActivationDescriptor { kind: GatedActivationKind::Glu, input_shape, split_dim, element: ElementKind::F16 };
    let output_shape = desc.output_shape();
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc y");
    let plan = GatedActivationPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = GatedActivationArgs::<f16, 3> {
        x: TensorRef { data: dev_x.as_slice(), shape: input_shape, stride: contiguous_stride(input_shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: output_shape, stride: contiguous_stride(output_shape) },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![f16::from_f32(0.0); out_numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let g_f = g.to_f32();
        let allow = e.abs().max(1.0) * 4.0 * F16_EPS;
        assert!((g_f - e).abs() <= allow, "glu f16 @ {i}: got {g_f} exp {e}");
    }
}

#[test]
#[ignore]
fn glu_fw_bf16() {
    let (ctx, stream) = setup();
    let input_shape = [2i32, 4, 8];
    let split_dim: u8 = 1;
    let in_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<bf16> = (0..in_numel).map(|i| bf16::from_f32(((i % 200) as f32) * 0.05 - 5.0)).collect();
    let x_f32: Vec<f32> = host_x.iter().map(|v| v.to_f32()).collect();
    let expected = cpu_ref_f32(&x_f32, &input_shape, split_dim as usize);
    let desc = GatedActivationDescriptor { kind: GatedActivationKind::Glu, input_shape, split_dim, element: ElementKind::Bf16 };
    let output_shape = desc.output_shape();
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc y");
    let plan = GatedActivationPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = GatedActivationArgs::<bf16, 3> {
        x: TensorRef { data: dev_x.as_slice(), shape: input_shape, stride: contiguous_stride(input_shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: output_shape, stride: contiguous_stride(output_shape) },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![bf16::from_f32(0.0); out_numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let g_f = g.to_f32();
        let allow = e.abs().max(1.0) * 4.0 * BF16_EPS;
        assert!((g_f - e).abs() <= allow, "glu bf16 @ {i}: got {g_f} exp {e}");
    }
}
