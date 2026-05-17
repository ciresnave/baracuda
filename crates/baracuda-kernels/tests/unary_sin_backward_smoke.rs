//! Real-GPU smoke test for the unary Sin backward trailblazer
//! (`UnaryBackwardPlan<f32, N> + UnaryKind::Sin`).
//!
//! Forward: `y = sin(x)`. Backward: `dx = dy * cos(x)`. Requires the
//! saved forward input `x` — proves the *saved-x* shape of the unary
//! backward family.
//!
//! `cosf` is transcendental and not bit-exact between Rust's libm and
//! CUDA's libdevice, so we use the same 4×eps relative tolerance as the
//! forward Sin test.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test unary_sin_backward_smoke -- --ignored`.

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

fn assert_close(got: f32, expected: f32, idx: usize) {
    let diff = (got - expected).abs();
    let allow = expected.abs().max(1.0) * 4.0 * f32::EPSILON;
    assert!(
        diff <= allow,
        "sin backward f32 mismatch @ {idx}: got {got} expected {expected} \
         (diff {diff} > allow {allow})"
    );
}

fn run_case<const N: usize>(shape: [i32; N]) {
    let (ctx, stream) = setup();
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f32> = (0..numel).map(|i| ((i % 200) as f32) * 0.01 - 1.0).collect();
    let host_dy: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.5 - 17.25).collect();
    let expected: Vec<f32> = host_x
        .iter()
        .zip(host_dy.iter())
        .map(|(x, dy)| *dy * x.cos())
        .collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");

    let stride = contiguous_stride(shape);
    let desc = UnaryBackwardDescriptor {
        kind: UnaryKind::Sin,
        shape,
        element: ElementKind::F32,
    };
    let plan = UnaryBackwardPlan::<f32, N>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = UnaryBackwardArgs::<f32, N> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        x: Some(TensorRef { data: dev_x.as_slice(), shape, stride }),
        y: None,
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_close(*g, *e, i);
    }
}

#[test]
#[ignore]
fn sin_backward_f32_1d() {
    run_case::<1>([2048]);
}

#[test]
#[ignore]
fn sin_backward_f32_2d() {
    run_case::<2>([64, 64]);
}

#[test]
#[ignore]
fn sin_backward_f32_3d() {
    run_case::<3>([8, 128, 128]);
}

// --- f16 / bf16 / f64 fanout ------------------------------------------------

#[test]
#[ignore]
fn sin_backward_f16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f16> = (0..numel).map(|i| f16::from_f32(((i % 200) as f32) * 0.01 - 1.0)).collect();
    let host_dy: Vec<f16> = (0..numel).map(|i| f16::from_f32((i % 41) as f32 * 0.25 - 5.0)).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");
    let stride = contiguous_stride(shape);
    let desc = UnaryBackwardDescriptor { kind: UnaryKind::Sin, shape, element: ElementKind::F16 };
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
        let x = host_x[i].to_f32();
        let dy = host_dy[i].to_f32();
        let exp = dy * x.cos();
        let g = got[i].to_f32();
        let tol = exp.abs().max(1.0) * 4.0 * F16_EPS;
        assert!((g - exp).abs() <= tol, "sin bw f16 @ {i}: got {g} exp {exp}");
    }
}

#[test]
#[ignore]
fn sin_backward_bf16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<bf16> = (0..numel).map(|i| bf16::from_f32(((i % 200) as f32) * 0.01 - 1.0)).collect();
    let host_dy: Vec<bf16> = (0..numel).map(|i| bf16::from_f32((i % 41) as f32 * 0.25 - 5.0)).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");
    let stride = contiguous_stride(shape);
    let desc = UnaryBackwardDescriptor { kind: UnaryKind::Sin, shape, element: ElementKind::Bf16 };
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
        let x = host_x[i].to_f32();
        let dy = host_dy[i].to_f32();
        let exp = dy * x.cos();
        let g = got[i].to_f32();
        let tol = exp.abs().max(1.0) * 4.0 * BF16_EPS;
        assert!((g - exp).abs() <= tol, "sin bw bf16 @ {i}: got {g} exp {exp}");
    }
}

#[test]
#[ignore]
fn sin_backward_f64_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f64> = (0..numel).map(|i| ((i % 200) as f64) * 0.01 - 1.0).collect();
    let host_dy: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.5 - 17.25).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");
    let stride = contiguous_stride(shape);
    let desc = UnaryBackwardDescriptor { kind: UnaryKind::Sin, shape, element: ElementKind::F64 };
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
        let exp = host_dy[i] * host_x[i].cos();
        let tol = exp.abs().max(1.0) * 4.0 * f64::EPSILON;
        assert!((got[i] - exp).abs() <= tol, "sin bw f64 @ {i}");
    }
}

/// `can_implement` rejects missing saved-x for Sin backward.
#[test]
fn sin_backward_requires_x() {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    let desc = UnaryBackwardDescriptor {
        kind: UnaryKind::Sin,
        shape: [4],
        element: ElementKind::F32,
    };
    let plan = UnaryBackwardPlan::<f32, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("Sin × f32 wired");
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
    assert!(err.is_err(), "Sin backward must reject missing saved-x");
}
