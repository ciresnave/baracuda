//! Real-GPU smoke test for the unary Exp backward trailblazer
//! (`UnaryBackwardPlan<f32, N> + UnaryKind::Exp`).
//!
//! Forward: `y = exp(x)`. Backward: `dx = dy * y`. Requires the saved
//! forward *output* `y` — proves the *saved-y* shape of the unary
//! backward family.
//!
//! The kernel is a single multiplication; no FMA opportunity in
//! `dy * y` alone. Bit-exact compare against the host `f32 * f32`.
//! The test feeds an arbitrary `y` (which doesn't need to actually be
//! `exp(x)` of anything — the kernel just multiplies); this isolates
//! the backward kernel from forward Exp's transcendental precision.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test unary_exp_backward_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef, UnaryBackwardArgs,
    UnaryBackwardDescriptor, UnaryBackwardPlan, UnaryKind, Workspace,
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

fn run_case<const N: usize>(shape: [i32; N]) {
    let (ctx, stream) = setup();
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_y: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.125 + 0.5).collect();
    let host_dy: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.5 - 17.25).collect();
    let expected: Vec<f32> = host_y
        .iter()
        .zip(host_dy.iter())
        .map(|(y, dy)| *dy * *y)
        .collect();

    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("upload y");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");

    let stride = contiguous_stride(shape);
    let desc = UnaryBackwardDescriptor {
        kind: UnaryKind::Exp,
        shape,
        element: ElementKind::F32,
    };
    let plan = UnaryBackwardPlan::<f32, N>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = UnaryBackwardArgs::<f32, N> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        x: None,
        y: Some(TensorRef { data: dev_y.as_slice(), shape, stride }),
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "exp backward f32 mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn exp_backward_f32_1d() {
    run_case::<1>([2048]);
}

#[test]
#[ignore]
fn exp_backward_f32_2d() {
    run_case::<2>([64, 64]);
}

#[test]
#[ignore]
fn exp_backward_f32_3d() {
    run_case::<3>([8, 128, 128]);
}

/// `can_implement` rejects missing saved-y for Exp backward.
#[test]
fn exp_backward_requires_y() {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    let desc = UnaryBackwardDescriptor {
        kind: UnaryKind::Exp,
        shape: [4],
        element: ElementKind::F32,
    };
    let plan = UnaryBackwardPlan::<f32, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("Exp × f32 wired");
    let dev_dy: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 4).expect("alloc");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 4).expect("alloc");
    let stride = contiguous_stride([4]);
    let args = UnaryBackwardArgs::<f32, 1> {
        dy: TensorRef { data: dev_dy.as_slice(), shape: [4], stride },
        x: None,
        y: None,
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape: [4], stride },
    };
    let err = plan.can_implement(&args);
    assert!(err.is_err(), "Exp backward must reject missing saved-y");
}

/// `select` rejects unwired ops. `Erfinv` is saved-x but its backward
/// (`dy * (√π/2) * exp(erfinv(x)²)` — needs `expf` on top of saved-y)
/// hasn't been wired yet.
#[test]
fn select_rejects_unwired_unary_today() {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    let desc = UnaryBackwardDescriptor {
        kind: UnaryKind::Erfinv,
        shape: [4],
        element: ElementKind::F32,
    };
    let err = UnaryBackwardPlan::<f32, 1>::select(&stream, &desc, PlanPreference::default());
    assert!(err.is_err(), "Erfinv BW must be unwired in this wave");
}

// --- f16 / bf16 / f64 fanout ------------------------------------------------
// Exp BW is just `dy * y` — bit-exact on f32 / f64. f16 / bf16 may
// differ by 1 ULP between the host (half-crate Mul detours through f32)
// and the GPU (native half ops), so we tolerance-compare those.

#[test]
#[ignore]
fn exp_backward_f16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_y: Vec<f16> = (0..numel).map(|i| f16::from_f32((i % 41) as f32 * 0.125 + 0.5)).collect();
    let host_dy: Vec<f16> = (0..numel).map(|i| f16::from_f32((i % 37) as f32 * 0.25 - 5.0)).collect();
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("upload y");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");
    let stride = contiguous_stride(shape);
    let desc = UnaryBackwardDescriptor { kind: UnaryKind::Exp, shape, element: ElementKind::F16 };
    let plan = UnaryBackwardPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = UnaryBackwardArgs::<f16, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        x: None,
        y: Some(TensorRef { data: dev_y.as_slice(), shape, stride }),
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![f16::from_f32(0.0); numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for i in 0..numel {
        let y = host_y[i].to_f32();
        let dy = host_dy[i].to_f32();
        let exp = dy * y;
        let g = got[i].to_f32();
        let tol = exp.abs().max(1.0) * 4.0 * F16_EPS;
        assert!((g - exp).abs() <= tol, "exp bw f16 @ {i}: got {g} exp {exp}");
    }
}

#[test]
#[ignore]
fn exp_backward_bf16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_y: Vec<bf16> = (0..numel).map(|i| bf16::from_f32((i % 41) as f32 * 0.125 + 0.5)).collect();
    let host_dy: Vec<bf16> = (0..numel).map(|i| bf16::from_f32((i % 37) as f32 * 0.25 - 5.0)).collect();
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("upload y");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");
    let stride = contiguous_stride(shape);
    let desc = UnaryBackwardDescriptor { kind: UnaryKind::Exp, shape, element: ElementKind::Bf16 };
    let plan = UnaryBackwardPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = UnaryBackwardArgs::<bf16, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        x: None,
        y: Some(TensorRef { data: dev_y.as_slice(), shape, stride }),
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![bf16::from_f32(0.0); numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for i in 0..numel {
        let y = host_y[i].to_f32();
        let dy = host_dy[i].to_f32();
        let exp = dy * y;
        let g = got[i].to_f32();
        let tol = exp.abs().max(1.0) * 4.0 * BF16_EPS;
        assert!((g - exp).abs() <= tol, "exp bw bf16 @ {i}: got {g} exp {exp}");
    }
}

#[test]
#[ignore]
fn exp_backward_f64_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_y: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.125 + 0.5).collect();
    let host_dy: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.5 - 17.25).collect();
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("upload y");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");
    let stride = contiguous_stride(shape);
    let desc = UnaryBackwardDescriptor { kind: UnaryKind::Exp, shape, element: ElementKind::F64 };
    let plan = UnaryBackwardPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = UnaryBackwardArgs::<f64, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        x: None,
        y: Some(TensorRef { data: dev_y.as_slice(), shape, stride }),
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f64; numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for i in 0..numel {
        let exp = host_dy[i] * host_y[i];
        assert_eq!(got[i].to_bits(), exp.to_bits(), "exp bw f64 @ {i}");
    }
}
