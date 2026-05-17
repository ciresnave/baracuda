//! Real-GPU smoke test for unary GELU (tanh approximation) backward
//! (`UnaryBackwardPlan<T, 3> + UnaryKind::GeluTanh`).
//!
//! Forward: `y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))`.
//! Backward — using the numerically stable sigmoid form (see kernel
//! header for the derivation; `0.5*(1+tanh(u)) = sigmoid(2u)` avoids
//! the cancellation in `(1+t)` as `t → -1`):
//!   c  = sqrt(2/pi)
//!   u  = c * (x + 0.044715*x^3)
//!   u' = c * (1 + 0.134145*x^2)
//!   s  = sigmoid(2u) = 1 / (1 + exp(-2u))
//!   dx = dy * s * (1 + 2*x*(1-s)*u')
//! Saved-x; one `exp` plus a well-conditioned multiplicative chain.
//!
//! Tolerance model: **cancellation-weighted absolute**. The sigmoid
//! reformulation eliminates the `(1+t)` cancellation as `t → -1` (the
//! original showstopper), but a residual cancellation of magnitude
//! `(1 + 2x(1-s)u')` near x ≈ -0.8 persists (the bracket can be ~7×
//! smaller than its larger summand). We bound the absolute error by
//! the cancellation magnitude `|dy| · s · (1 + |2x(1-s)u'|)` so the
//! tolerance scales with what's actually being added/subtracted; K = 16
//! absorbs the 1-2 ULP libdevice-vs-libm gap × the multiplicative chain.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef, UnaryBackwardArgs,
    UnaryBackwardDescriptor, UnaryBackwardPlan, UnaryKind, Workspace,
};
use half::{bf16, f16};

const F16_EPS: f32 = 9.77e-4;
const BF16_EPS: f32 = 7.81e-3;
const C_F32: f32 = 0.79788456080286535588;
const C_F64: f64 = 0.79788456080286535588;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Returns `(result, cancellation_magnitude)`. The cancellation magnitude
/// is `|dy| · s · (1 + |2·x·(1-s)·u'|)` — bound on the absolute error
/// from the residual `1 + 2x(1-s)u'` cancellation near x ≈ -0.8.
fn gelu_tanh_bw_f32(dy: f32, x: f32) -> (f32, f32) {
    let xx = x * x;
    let u = C_F32 * (x + 0.044715 * x * xx);
    let s = 1.0 / (1.0 + (-2.0 * u).exp());
    let u_prime = C_F32 * (1.0 + 0.134145 * xx);
    let inner = 1.0 + 2.0 * x * (1.0 - s) * u_prime;
    let result = dy * s * inner;
    let cancel_mag = dy.abs() * s * (1.0 + (2.0 * x * (1.0 - s) * u_prime).abs());
    (result, cancel_mag)
}
fn gelu_tanh_bw_f64(dy: f64, x: f64) -> (f64, f64) {
    let xx = x * x;
    let u = C_F64 * (x + 0.044715 * x * xx);
    let s = 1.0 / (1.0 + (-2.0 * u).exp());
    let u_prime = C_F64 * (1.0 + 0.134145 * xx);
    let inner = 1.0 + 2.0 * x * (1.0 - s) * u_prime;
    let result = dy * s * inner;
    let cancel_mag = dy.abs() * s * (1.0 + (2.0 * x * (1.0 - s) * u_prime).abs());
    (result, cancel_mag)
}

#[test]
#[ignore]
fn gelu_tanh_backward_f32_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f32> = (0..numel).map(|i| ((i % 600) as f32) * 0.01 - 3.0).collect();
    let host_dy: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.5 - 17.25).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");
    let stride = contiguous_stride(shape);
    let desc = UnaryBackwardDescriptor { kind: UnaryKind::GeluTanh, shape, element: ElementKind::F32 };
    let plan = UnaryBackwardPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = UnaryBackwardArgs::<f32, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        x: Some(TensorRef { data: dev_x.as_slice(), shape, stride }),
        y: None,
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for i in 0..numel {
        let (exp, cancel_mag) = gelu_tanh_bw_f32(host_dy[i], host_x[i]);
        let tol = (cancel_mag.max(exp.abs())) * 16.0 * f32::EPSILON;
        assert!((got[i] - exp).abs() <= tol,
            "gelu_tanh bw f32 @ {i}: x={} dy={} got {} exp {} diff {} tol {}",
            host_x[i], host_dy[i], got[i], exp, (got[i] - exp).abs(), tol);
    }
}

#[test]
#[ignore]
fn gelu_tanh_backward_f64_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f64> = (0..numel).map(|i| ((i % 600) as f64) * 0.01 - 3.0).collect();
    let host_dy: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.5 - 17.25).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");
    let stride = contiguous_stride(shape);
    let desc = UnaryBackwardDescriptor { kind: UnaryKind::GeluTanh, shape, element: ElementKind::F64 };
    let plan = UnaryBackwardPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = UnaryBackwardArgs::<f64, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        x: Some(TensorRef { data: dev_x.as_slice(), shape, stride }),
        y: None,
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f64; numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for i in 0..numel {
        let (exp, cancel_mag) = gelu_tanh_bw_f64(host_dy[i], host_x[i]);
        let tol = (cancel_mag.max(exp.abs())) * 16.0 * f64::EPSILON;
        assert!((got[i] - exp).abs() <= tol, "gelu_tanh bw f64 @ {i}");
    }
}

#[test]
#[ignore]
fn gelu_tanh_backward_f16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f16> = (0..numel).map(|i| f16::from_f32(((i % 600) as f32) * 0.01 - 3.0)).collect();
    let host_dy: Vec<f16> = (0..numel).map(|i| f16::from_f32(((i % 41) as f32) * 0.25 - 5.0)).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");
    let stride = contiguous_stride(shape);
    let desc = UnaryBackwardDescriptor { kind: UnaryKind::GeluTanh, shape, element: ElementKind::F16 };
    let plan = UnaryBackwardPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = UnaryBackwardArgs::<f16, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        x: Some(TensorRef { data: dev_x.as_slice(), shape, stride }),
        y: None,
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![f16::from_f32(0.0); numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for i in 0..numel {
        let fx = host_x[i].to_f32();
        let dy = host_dy[i].to_f32();
        let (exp, cancel_mag) = gelu_tanh_bw_f32(dy, fx);
        let g = got[i].to_f32();
        let tol = (cancel_mag.max(exp.abs())) * 16.0 * F16_EPS;
        assert!((g - exp).abs() <= tol, "gelu_tanh bw f16 @ {i}: got {g} exp {exp}");
    }
}

#[test]
#[ignore]
fn gelu_tanh_backward_bf16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<bf16> = (0..numel).map(|i| bf16::from_f32(((i % 600) as f32) * 0.01 - 3.0)).collect();
    let host_dy: Vec<bf16> = (0..numel).map(|i| bf16::from_f32(((i % 41) as f32) * 0.25 - 5.0)).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");
    let stride = contiguous_stride(shape);
    let desc = UnaryBackwardDescriptor { kind: UnaryKind::GeluTanh, shape, element: ElementKind::Bf16 };
    let plan = UnaryBackwardPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = UnaryBackwardArgs::<bf16, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        x: Some(TensorRef { data: dev_x.as_slice(), shape, stride }),
        y: None,
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![bf16::from_f32(0.0); numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for i in 0..numel {
        let fx = host_x[i].to_f32();
        let dy = host_dy[i].to_f32();
        let (exp, cancel_mag) = gelu_tanh_bw_f32(dy, fx);
        let g = got[i].to_f32();
        let tol = (cancel_mag.max(exp.abs())) * 16.0 * BF16_EPS;
        assert!((g - exp).abs() <= tol, "gelu_tanh bw bf16 @ {i}: got {g} exp {exp}");
    }
}
