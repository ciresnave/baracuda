//! Real-GPU smoke test for the heterogeneous-dtype ternary trailblazer
//! (`WherePlan<f32, N>`).
//!
//! Covers contig (1D / 2D / 3D), per-row cond broadcast (`[M, 1]` cond
//! vs `[M, N]` values — common pattern for row-masked selection), and
//! scalar-cond broadcast.
//!
//! Output is bit-exact against host reference `cond ? a : b` —
//! `where` does no arithmetic, just element selection, so host and
//! device produce identical bit patterns regardless of dtype.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test where_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef, WhereArgs,
    WhereDescriptor, WherePlan, Workspace,
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

    // Cond alternates every cell so we exercise both branches.
    let host_cond: Vec<u8> = (0..numel).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();
    let host_a: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.5 - 17.25).collect();
    let host_b: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.25 + 100.0).collect();
    let expected: Vec<f32> = host_cond
        .iter()
        .zip(host_a.iter())
        .zip(host_b.iter())
        .map(|((&c, &a), &b)| if c != 0 { a } else { b })
        .collect();

    let dev_cond = DeviceBuffer::from_slice(&ctx, &host_cond).expect("upload cond");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = WhereDescriptor {
        shape,
        element: ElementKind::F32,
    };
    let plan = WherePlan::<f32, N>::select(&stream, &desc, PlanPreference::default())
        .expect("select");

    let args = WhereArgs::<f32, N> {
        cond: TensorRef {
            data: dev_cond.as_slice(),
            shape,
            stride,
        },
        a: TensorRef {
            data: dev_a.as_slice(),
            shape,
            stride,
        },
        b: TensorRef {
            data: dev_b.as_slice(),
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
            "where contig mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn where_f32_1d() {
    run_contig::<1>([2048]);
}

#[test]
#[ignore]
fn where_f32_2d() {
    run_contig::<2>([64, 64]);
}

#[test]
#[ignore]
fn where_f32_3d() {
    run_contig::<3>([8, 128, 128]);
}

/// Per-row cond broadcast: `cond: [M, 1]` vs `a, b: [M, N]`.
/// Common pattern — a row mask is broadcast across the column dim to
/// select between two full matrices row-by-row.
#[test]
#[ignore]
fn where_f32_broadcast_row_cond() {
    let (ctx, stream) = setup();
    const M: usize = 32;
    const N_DIM: usize = 64;
    let m = M as i32;
    let n = N_DIM as i32;

    // Cond is per-row: alternating 1 / 0 by row index.
    let host_cond: Vec<u8> = (0..M).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();
    let host_a: Vec<f32> = (0..(M * N_DIM))
        .map(|i| (i as f32) * 0.25 - 3.0)
        .collect();
    let host_b: Vec<f32> = (0..(M * N_DIM))
        .map(|i| (i as f32) * 0.0625 + 50.0)
        .collect();

    let mut expected = vec![0f32; M * N_DIM];
    for i in 0..M {
        for j in 0..N_DIM {
            let c = host_cond[i];
            let idx = i * N_DIM + j;
            expected[idx] = if c != 0 { host_a[idx] } else { host_b[idx] };
        }
    }

    let dev_cond = DeviceBuffer::from_slice(&ctx, &host_cond).expect("upload cond");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, M * N_DIM).expect("alloc y");

    let cond_shape = [m, 1i32];
    let cond_stride = [1i64, 0]; // varies along rows, broadcasts along cols
    let val_shape = [m, n];
    let val_stride = contiguous_stride([m, n]);
    let y_shape = [m, n];
    let y_stride = contiguous_stride([m, n]);

    let desc = WhereDescriptor {
        shape: y_shape,
        element: ElementKind::F32,
    };
    let plan = WherePlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = WhereArgs::<f32, 2> {
        cond: TensorRef {
            data: dev_cond.as_slice(),
            shape: cond_shape,
            stride: cond_stride,
        },
        a: TensorRef {
            data: dev_a.as_slice(),
            shape: val_shape,
            stride: val_stride,
        },
        b: TensorRef {
            data: dev_b.as_slice(),
            shape: val_shape,
            stride: val_stride,
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
            "where row-cond broadcast mismatch @ {i}: got {g} expected {e}"
        );
    }
}

/// Scalar cond broadcast: single-element cond drives every output cell
/// to a or b. Degenerate but verifies the stride-0-on-every-axis path
/// on the cond input.
#[test]
#[ignore]
fn where_f32_scalar_cond() {
    let (ctx, stream) = setup();
    const M: usize = 64;
    const N_DIM: usize = 128;
    let m = M as i32;
    let n = N_DIM as i32;

    let host_cond: Vec<u8> = vec![1u8]; // single true → all-a
    let host_a: Vec<f32> = (0..(M * N_DIM)).map(|i| (i as f32) * 0.1 - 5.0).collect();
    let host_b: Vec<f32> = (0..(M * N_DIM))
        .map(|i| (i as f32) * 0.05 + 1000.0)
        .collect();
    let expected: Vec<f32> = host_a.clone(); // cond=1 everywhere

    let dev_cond = DeviceBuffer::from_slice(&ctx, &host_cond).expect("upload cond");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, M * N_DIM).expect("alloc y");

    let cond_shape = [1i32, 1i32];
    let cond_stride = [0i64, 0i64];
    let val_shape = [m, n];
    let val_stride = contiguous_stride([m, n]);
    let y_shape = [m, n];
    let y_stride = contiguous_stride([m, n]);

    let desc = WhereDescriptor {
        shape: y_shape,
        element: ElementKind::F32,
    };
    let plan = WherePlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = WhereArgs::<f32, 2> {
        cond: TensorRef {
            data: dev_cond.as_slice(),
            shape: cond_shape,
            stride: cond_stride,
        },
        a: TensorRef {
            data: dev_a.as_slice(),
            shape: val_shape,
            stride: val_stride,
        },
        b: TensorRef {
            data: dev_b.as_slice(),
            shape: val_shape,
            stride: val_stride,
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
            "where scalar-cond mismatch @ {i}: got {g} expected {e}"
        );
    }
}
