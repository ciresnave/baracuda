//! Real-GPU smoke test for the Phase 3 affine kernel family
//! (`AffinePlan<T>`).
//!
//! Validates `y = a * x + b` against a CPU reference for f32 / f64.
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test affine_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, AffineArgs, AffineDescriptor, AffinePlan, ElementKind, PlanPreference,
    TensorMut, TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn affine_f32_bit_exact_when_no_fma_fusion() {
    let (ctx, stream) = setup();
    let numel = 1024usize;
    let a: f32 = 2.5;
    let b: f32 = -1.25;
    let host_x: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.5 - 17.25).collect();
    // CPU reference does the same `a*x + b` sequence the kernel
    // computes — bit-exact for representable values when no FMA
    // contraction kicks in.
    let expected: Vec<f32> = host_x.iter().map(|&x| a * x + b).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");

    let desc = AffineDescriptor {
        numel: numel as i32,
        a,
        b,
        element: ElementKind::F32,
    };
    let plan = AffinePlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = AffineArgs {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape: [numel as i32],
            stride: contiguous_stride([numel as i32]),
        },
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: [numel as i32],
            stride: contiguous_stride([numel as i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("download");

    // GPU may emit FFMA (fused multiply-add) which differs from the
    // CPU's separate mul-then-add by 0 or 1 ulp. Accept a 2-ulp
    // relative tolerance.
    let tol_eps = 2.0f32 * f32::EPSILON;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let err = (g - e).abs();
        let tol = e.abs().max(1.0) * tol_eps;
        assert!(err <= tol, "affine_f32 @ {i}: got {g} expected {e}");
    }
}

#[test]
#[ignore]
fn affine_f64_close_to_reference() {
    let (ctx, stream) = setup();
    let numel = 1024usize;
    let a: f64 = -3.0;
    let b: f64 = 0.75;
    let host_x: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.125 + 2.0).collect();
    let expected: Vec<f64> = host_x.iter().map(|&x| a * x + b).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");

    let desc = AffineDescriptor {
        numel: numel as i32,
        a,
        b,
        element: ElementKind::F64,
    };
    let plan = AffinePlan::<f64>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = AffineArgs {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape: [numel as i32],
            stride: contiguous_stride([numel as i32]),
        },
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: [numel as i32],
            stride: contiguous_stride([numel as i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; numel];
    dev_y.copy_to_host(&mut got).expect("download");

    // 2-ulp f64 relative tolerance (FFMA vs mul-then-add).
    let tol_eps = 2.0f64 * f64::EPSILON;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let err = (g - e).abs();
        let tol = e.abs().max(1.0) * tol_eps;
        assert!(err <= tol, "affine_f64 @ {i}: got {g} expected {e}");
    }
}

#[test]
fn select_rejects_mismatched_dtype() {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let desc = AffineDescriptor {
        numel: 16,
        a: 0.0f32,
        b: 0.0f32,
        element: ElementKind::F64, // mismatched with f32 below
    };
    let err = AffinePlan::<f32>::select(&stream, &desc, PlanPreference::default());
    assert!(err.is_err(), "mismatched dtype must be rejected");
    let _ = ctx;
}
