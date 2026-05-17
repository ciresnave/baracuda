//! Real-GPU smoke test for unary GELU (exact, erf-based) backward
//! (`UnaryBackwardPlan<T, 3> + UnaryKind::Gelu`).
//!
//! Backward:
//!   dx = dy * (0.5 * (1 + erf(x/sqrt(2))) + x * (1/sqrt(2*pi)) * exp(-x*x/2))
//! Saved-x; uses `erf` and `exp`.
//!
//! Numerical-stability note: the kernel computes `Φ(x)` via the
//! `erfc(-x/sqrt(2))` form when `x < 0` to dodge the catastrophic
//! cancellation in `(1 + erf(x))` for `erf(x) ≈ -1`. The host reference
//! matches that piecewise form so we compare apples-to-apples.
//!
//! Tolerance model: **cancellation-weighted absolute**, not flat ULP.
//! GELU' has an intrinsic cancellation in the bracket `Φ(x) + x·φ(x)`
//! across the band `x ∈ [-1.5, -0.3]`: the two summands are roughly
//! equal in magnitude and opposite in sign (e.g. at x = -0.76, Φ ≈
//! 0.224, x·φ ≈ -0.227, bracket ≈ -0.0035 — a 64× cancellation of two
//! ~0.22-magnitude summands). Each summand carries ~few ULPs of
//! libdevice-vs-libm divergence (`erf`/`erfc`/`exp`; NVIDIA documents
//! libdevice f64 `erf`/`erfc` max ULP errors of 5.6 / 4.8, and Rust
//! libm `erfc` is similarly loose). After the cancellation, the bracket
//! relative error is `~few ULPs × cancellation_factor`, propagated to
//! the final result by `dy`.
//!
//! Rather than burn a flat ULP factor large enough for the worst x, we
//! bound the error by the *cancellation magnitude* directly:
//!   abs_err  ≤  K · eps · |dy| · (|Φ| + |x·φ|)
//! This is the textbook bound for `dy · (a + b)` with `|a|, |b|`
//! independently rounded; K absorbs the libdevice-vs-libm gap (K = 16
//! is principled — 8 ULPs of erf gap × 2 for compounding multiplies).
//! Outside the cancellation band the bound reduces to a normal
//! relative-tolerance compare (|Φ| ≈ |bracket| there).
//!
//! This cancellation is mathematical, not implementation-specific:
//! PyTorch and JAX use the same formula. No equivalent reformulation
//! avoids it (Mills'-ratio-based forms simply move the cancellation to
//! a different `x` band).

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef, UnaryBackwardArgs,
    UnaryBackwardDescriptor, UnaryBackwardPlan, UnaryKind, Workspace,
};
use half::{bf16, f16};

const F16_EPS: f32 = 9.77e-4;
const BF16_EPS: f32 = 7.81e-3;
const INV_SQRT_2_F32: f32 = 0.70710678118654752440;
const INV_SQRT_2PI_F32: f32 = 0.39894228040143267794;
const INV_SQRT_2_F64: f64 = 0.70710678118654752440;
const INV_SQRT_2PI_F64: f64 = 0.39894228040143267794;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Returns `(result, cancellation_magnitude)`. The cancellation magnitude
/// is `|dy| * (|Φ(x)| + |x·φ(x)|)` — the textbook upper bound on the
/// absolute error when the two summands inside the bracket are each
/// rounded independently.
fn gelu_bw_f32(dy: f32, x: f32) -> (f32, f32) {
    let cdf = if x >= 0.0 {
        0.5 * (1.0 + libm::erff(x * INV_SQRT_2_F32))
    } else {
        0.5 * libm::erfcf(-x * INV_SQRT_2_F32)
    };
    let pdf = INV_SQRT_2PI_F32 * (-0.5 * x * x).exp();
    let xpdf = x * pdf;
    let result = dy * (cdf + xpdf);
    let cancel_mag = dy.abs() * (cdf.abs() + xpdf.abs());
    (result, cancel_mag)
}
fn gelu_bw_f64(dy: f64, x: f64) -> (f64, f64) {
    let cdf = if x >= 0.0 {
        0.5 * (1.0 + libm::erf(x * INV_SQRT_2_F64))
    } else {
        0.5 * libm::erfc(-x * INV_SQRT_2_F64)
    };
    let pdf = INV_SQRT_2PI_F64 * (-0.5 * x * x).exp();
    let xpdf = x * pdf;
    let result = dy * (cdf + xpdf);
    let cancel_mag = dy.abs() * (cdf.abs() + xpdf.abs());
    (result, cancel_mag)
}

#[test]
#[ignore]
fn gelu_backward_f32_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f32> = (0..numel).map(|i| ((i % 600) as f32) * 0.01 - 3.0).collect();
    let host_dy: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.5 - 17.25).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");
    let stride = contiguous_stride(shape);
    let desc = UnaryBackwardDescriptor { kind: UnaryKind::Gelu, shape, element: ElementKind::F32 };
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
        let (exp, cancel_mag) = gelu_bw_f32(host_dy[i], host_x[i]);
        // Cancellation-weighted absolute tolerance: K · eps · |dy| · (|cdf|+|x·pdf|).
        // K = 16 absorbs ~8 ULPs of `erf`/`erfc`/`exp` libdevice-vs-libm gap × 2 for
        // the multiplicative chain. Outside the cancellation band this reduces to
        // a normal relative-tolerance compare (~16 ULPs).
        let tol = (cancel_mag.max(exp.abs())) * 16.0 * f32::EPSILON;
        assert!((got[i] - exp).abs() <= tol,
            "gelu bw f32 @ {i}: x={} dy={} got {} exp {} diff {} tol {}",
            host_x[i], host_dy[i], got[i], exp, (got[i] - exp).abs(), tol);
    }
}

#[test]
#[ignore]
fn gelu_backward_f64_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f64> = (0..numel).map(|i| ((i % 600) as f64) * 0.01 - 3.0).collect();
    let host_dy: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.5 - 17.25).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");
    let stride = contiguous_stride(shape);
    let desc = UnaryBackwardDescriptor { kind: UnaryKind::Gelu, shape, element: ElementKind::F64 };
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
        let (exp, cancel_mag) = gelu_bw_f64(host_dy[i], host_x[i]);
        // Cancellation-weighted absolute tolerance — see f32 test for derivation.
        // K = 16 (~7 ULP erf libdevice + ~2 ULP erfc libm + multiplicative slack).
        let tol = (cancel_mag.max(exp.abs())) * 16.0 * f64::EPSILON;
        assert!((got[i] - exp).abs() <= tol,
            "gelu bw f64 @ {i}: x={} dy={} got {} exp {} diff {} tol {}",
            host_x[i], host_dy[i], got[i], exp, (got[i] - exp).abs(), tol);
    }
}

#[test]
#[ignore]
fn gelu_backward_f16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f16> = (0..numel).map(|i| f16::from_f32(((i % 600) as f32) * 0.01 - 3.0)).collect();
    let host_dy: Vec<f16> = (0..numel).map(|i| f16::from_f32(((i % 41) as f32) * 0.25 - 5.0)).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");
    let stride = contiguous_stride(shape);
    let desc = UnaryBackwardDescriptor { kind: UnaryKind::Gelu, shape, element: ElementKind::F16 };
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
        let (exp, cancel_mag) = gelu_bw_f32(dy, fx);
        let g = got[i].to_f32();
        let tol = (cancel_mag.max(exp.abs())) * 16.0 * F16_EPS;
        assert!((g - exp).abs() <= tol, "gelu bw f16 @ {i}: got {g} exp {exp}");
    }
}

#[test]
#[ignore]
fn gelu_backward_bf16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<bf16> = (0..numel).map(|i| bf16::from_f32(((i % 600) as f32) * 0.01 - 3.0)).collect();
    let host_dy: Vec<bf16> = (0..numel).map(|i| bf16::from_f32(((i % 41) as f32) * 0.25 - 5.0)).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");
    let stride = contiguous_stride(shape);
    let desc = UnaryBackwardDescriptor { kind: UnaryKind::Gelu, shape, element: ElementKind::Bf16 };
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
        let (exp, cancel_mag) = gelu_bw_f32(dy, fx);
        let g = got[i].to_f32();
        let tol = (cancel_mag.max(exp.abs())) * 16.0 * BF16_EPS;
        assert!((g - exp).abs() <= tol, "gelu bw bf16 @ {i}: got {g} exp {exp}");
    }
}
