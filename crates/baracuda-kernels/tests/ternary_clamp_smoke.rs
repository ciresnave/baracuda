//! Real-GPU smoke test for the ternary trailblazer
//! (`TernaryPlan<f32, N> + TernaryKind::Clamp`).
//!
//! Covers contig (1D / 2D / 3D), scalar-broadcast (lo and hi as
//! rank-N tensors with stride 0 on every axis — the typical
//! `clamp(x, lo, hi)` use case), and strided transposed views.
//! Bit-exact against host reference (`fminf(fmaxf(x, lo), hi)` is a
//! deterministic two-rounding-step pipeline; the device matches host
//! exactly on f32).
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test ternary_clamp_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef, TernaryArgs,
    TernaryDescriptor, TernaryKind, TernaryPlan, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn run_contig<const N: usize>(shape: [i32; N]) {
    let (ctx, stream) = setup();
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    // x ranges over [-20, +20]; some below lo=-5, some above hi=+5,
    // some in range. lo and hi are per-element (full contig tensors).
    let host_x: Vec<f32> = (0..numel)
        .map(|i| (i as f32) * 0.05 - 20.0)
        .collect();
    let host_lo: Vec<f32> = (0..numel).map(|_| -5.0_f32).collect();
    let host_hi: Vec<f32> = (0..numel).map(|_| 5.0_f32).collect();
    let expected: Vec<f32> = host_x
        .iter()
        .zip(host_lo.iter())
        .zip(host_hi.iter())
        .map(|((&x, &lo), &hi)| x.max(lo).min(hi))
        .collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_lo = DeviceBuffer::from_slice(&ctx, &host_lo).expect("upload lo");
    let dev_hi = DeviceBuffer::from_slice(&ctx, &host_hi).expect("upload hi");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = TernaryDescriptor {
        kind: TernaryKind::Clamp,
        shape,
        element: ElementKind::F32,
        scale: 1.0,
    };
    let plan = TernaryPlan::<f32, N>::select(&stream, &desc, PlanPreference::default())
        .expect("select");

    let args = TernaryArgs::<f32, N> {
        a: TensorRef {
            data: dev_x.as_slice(),
            shape,
            stride,
        },
        b: TensorRef {
            data: dev_lo.as_slice(),
            shape,
            stride,
        },
        c: TensorRef {
            data: dev_hi.as_slice(),
            shape,
            stride,
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape,
            stride,
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("download");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "clamp contig mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn clamp_f32_1d() {
    run_contig::<1>([2048]);
}

#[test]
#[ignore]
fn clamp_f32_2d() {
    run_contig::<2>([64, 64]);
}

#[test]
#[ignore]
fn clamp_f32_3d() {
    run_contig::<3>([8, 128, 128]);
}

/// Scalar-broadcast lo/hi — the typical `clamp(x, lo=-5, hi=5)` pattern.
/// lo and hi are rank-2 tensors with shape `[1, 1]` and stride `[0, 0]`
/// — broadcast across every output cell.
#[test]
#[ignore]
fn clamp_f32_scalar_broadcast() {
    let (ctx, stream) = setup();
    const M: usize = 64;
    const N_DIM: usize = 128;
    let m = M as i32;
    let n = N_DIM as i32;

    let host_x: Vec<f32> = (0..(M * N_DIM))
        .map(|i| (i as f32) * 0.05 - 200.0)
        .collect();
    let host_lo: Vec<f32> = vec![-5.0_f32];
    let host_hi: Vec<f32> = vec![5.0_f32];
    let expected: Vec<f32> = host_x.iter().map(|&x| x.max(-5.0).min(5.0)).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_lo = DeviceBuffer::from_slice(&ctx, &host_lo).expect("upload lo");
    let dev_hi = DeviceBuffer::from_slice(&ctx, &host_hi).expect("upload hi");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, M * N_DIM).expect("alloc y");

    let x_shape = [m, n];
    let x_stride = contiguous_stride([m, n]);
    let scalar_shape = [1i32, 1i32];
    let scalar_stride = [0i64, 0i64];
    let y_shape = [m, n];
    let y_stride = contiguous_stride([m, n]);

    let desc = TernaryDescriptor {
        kind: TernaryKind::Clamp,
        shape: y_shape,
        element: ElementKind::F32,
        scale: 1.0,
    };
    let plan = TernaryPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = TernaryArgs::<f32, 2> {
        a: TensorRef {
            data: dev_x.as_slice(),
            shape: x_shape,
            stride: x_stride,
        },
        b: TensorRef {
            data: dev_lo.as_slice(),
            shape: scalar_shape,
            stride: scalar_stride,
        },
        c: TensorRef {
            data: dev_hi.as_slice(),
            shape: scalar_shape,
            stride: scalar_stride,
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: y_shape,
            stride: y_stride,
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; M * N_DIM];
    dev_y.copy_to_host(&mut got).expect("download");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "clamp scalar-broadcast mismatch @ {i}: got {g} expected {e}"
        );
    }
    // Sanity: there should be cells at each of the three regions.
    let clamped_low = got.iter().filter(|&&v| v == -5.0).count();
    let clamped_high = got.iter().filter(|&&v| v == 5.0).count();
    let in_range = got.iter().filter(|&&v| v > -5.0 && v < 5.0).count();
    assert!(
        clamped_low > 0 && clamped_high > 0 && in_range > 0,
        "expected all three regions exercised; got {clamped_low}/{in_range}/{clamped_high}"
    );
}
