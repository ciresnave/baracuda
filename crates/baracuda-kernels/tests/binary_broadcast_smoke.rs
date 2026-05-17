//! Real-GPU smoke test for the strided / broadcast path of
//! `BinaryPlan<f32, N> + BinaryKind::Add`.
//!
//! Exercises the four canonical broadcast patterns plus a non-broadcast
//! strided (transposed) view to confirm the same kernel handles every
//! non-contig case. Each test builds inputs / strides on the host, runs
//! the GPU kernel via the unified BinaryPlan, and compares **bit-exact**
//! against a CPU reference that walks the same strides — f32 add is one
//! IEEE rounding step, so the GPU and host agree exactly.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test binary_broadcast_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BinaryArgs, BinaryDescriptor, BinaryKind, BinaryPlan, ElementKind,
    PlanPreference, TensorMut, TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// CPU reference that walks the same strides the kernel does. Iterates
/// every output coord, follows each operand's stride to compute the
/// gmem offset, and writes the f32 add into the output buffer at the
/// corresponding contig output offset (the test always uses a contig
/// output, so y is indexed linearly).
fn cpu_strided_add<const N: usize>(
    a_host: &[f32],
    a_shape: [i32; N],
    a_stride: [i64; N],
    b_host: &[f32],
    _b_shape: [i32; N],
    b_stride: [i64; N],
    y_shape: [i32; N],
) -> Vec<f32> {
    let numel: usize = y_shape.iter().map(|&d| d as usize).product();
    let mut y_host = vec![0f32; numel];
    let _ = a_shape; // unused — kernel reads via stride only
    for i in 0..numel {
        // Unravel i into coord using row-major (rightmost = innermost).
        let mut linear = i as i64;
        let mut off_a: i64 = 0;
        let mut off_b: i64 = 0;
        for d in (0..N).rev() {
            let s = y_shape[d] as i64;
            let coord = if s == 0 { 0 } else { linear % s };
            if s != 0 {
                linear /= s;
            }
            off_a += coord * a_stride[d];
            off_b += coord * b_stride[d];
        }
        y_host[i] = a_host[off_a as usize] + b_host[off_b as usize];
    }
    y_host
}

// ============================================================================
// Test: broadcast across rows — `a: [1, N]` + `b: [M, N]` → `y: [M, N]`
// ============================================================================
//
// This is the bias-add pattern: a single row broadcast across every row
// of b. a is stored as N elements but logically [1, N] with stride [0, 1];
// b is stored contig as [M, N] with stride [N, 1].
#[test]
#[ignore]
fn broadcast_add_row_bias() {
    let (ctx, stream) = setup();
    const M: usize = 64;
    const N_DIM: usize = 128;
    let m = M as i32;
    let n = N_DIM as i32;

    let a_host: Vec<f32> = (0..N_DIM).map(|i| (i as f32) * 0.5 - 7.25).collect();
    let b_host: Vec<f32> = (0..(M * N_DIM))
        .map(|i| (i as f32) * 0.125 + 1.0)
        .collect();

    let a_shape = [1i32, n];
    let a_stride = [0i64, 1];
    let b_shape = [m, n];
    let b_stride = [n as i64, 1];
    let y_shape = [m, n];
    let y_stride = [n as i64, 1];

    let expected = cpu_strided_add::<2>(
        &a_host, a_shape, a_stride,
        &b_host, b_shape, b_stride,
        y_shape,
    );

    let dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &b_host).expect("upload b");
    let mut dev_y: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, M * N_DIM).expect("alloc y");

    let desc = BinaryDescriptor {
        kind: BinaryKind::Add,
        shape: y_shape,
        element: ElementKind::F32,
    };
    let plan = BinaryPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");

    let args = BinaryArgs::<f32, 2> {
        a: TensorRef {
            data: dev_a.as_slice(),
            shape: a_shape,
            stride: a_stride,
        },
        b: TensorRef {
            data: dev_b.as_slice(),
            shape: b_shape,
            stride: b_stride,
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
            "row-bias broadcast mismatch @ {i}: got {g} expected {e}"
        );
    }
}

// ============================================================================
// Test: broadcast across cols — `a: [M, 1]` + `b: [M, N]`
// ============================================================================
//
// Per-row scalar broadcast — common pattern for layer-norm scale, weight
// per output row. a has stride [1, 0]: along axis 0 (rows) it advances
// one element; along axis 1 (cols) it doesn't move.
#[test]
#[ignore]
fn broadcast_add_col_bias() {
    let (ctx, stream) = setup();
    const M: usize = 64;
    const N_DIM: usize = 128;
    let m = M as i32;
    let n = N_DIM as i32;

    let a_host: Vec<f32> = (0..M).map(|i| (i as f32) * 0.25 - 3.0).collect();
    let b_host: Vec<f32> = (0..(M * N_DIM))
        .map(|i| (i as f32) * 0.0625 + 0.5)
        .collect();

    let a_shape = [m, 1i32];
    let a_stride = [1i64, 0];
    let b_shape = [m, n];
    let b_stride = [n as i64, 1];
    let y_shape = [m, n];
    let y_stride = [n as i64, 1];

    let expected = cpu_strided_add::<2>(
        &a_host, a_shape, a_stride,
        &b_host, b_shape, b_stride,
        y_shape,
    );

    let dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &b_host).expect("upload b");
    let mut dev_y: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, M * N_DIM).expect("alloc y");

    let desc = BinaryDescriptor {
        kind: BinaryKind::Add,
        shape: y_shape,
        element: ElementKind::F32,
    };
    let plan = BinaryPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = BinaryArgs::<f32, 2> {
        a: TensorRef {
            data: dev_a.as_slice(),
            shape: a_shape,
            stride: a_stride,
        },
        b: TensorRef {
            data: dev_b.as_slice(),
            shape: b_shape,
            stride: b_stride,
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
            "col-bias broadcast mismatch @ {i}: got {g} expected {e}"
        );
    }
}

// ============================================================================
// Test: broadcast both dims — `a: [1, 1]` + `b: [M, N]`
// ============================================================================
//
// Scalar-broadcast pattern — single value added to every cell. Even more
// degenerate than row-bias; both axes have stride 0 on `a`.
#[test]
#[ignore]
fn broadcast_add_scalar() {
    let (ctx, stream) = setup();
    const M: usize = 32;
    const N_DIM: usize = 64;
    let m = M as i32;
    let n = N_DIM as i32;

    let a_host: Vec<f32> = vec![3.14159];
    let b_host: Vec<f32> = (0..(M * N_DIM)).map(|i| (i as f32) * 0.5 - 5.0).collect();

    let a_shape = [1i32, 1i32];
    let a_stride = [0i64, 0i64];
    let b_shape = [m, n];
    let b_stride = [n as i64, 1];
    let y_shape = [m, n];
    let y_stride = [n as i64, 1];

    let expected = cpu_strided_add::<2>(
        &a_host, a_shape, a_stride,
        &b_host, b_shape, b_stride,
        y_shape,
    );

    let dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &b_host).expect("upload b");
    let mut dev_y: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, M * N_DIM).expect("alloc y");

    let desc = BinaryDescriptor {
        kind: BinaryKind::Add,
        shape: y_shape,
        element: ElementKind::F32,
    };
    let plan = BinaryPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = BinaryArgs::<f32, 2> {
        a: TensorRef {
            data: dev_a.as_slice(),
            shape: a_shape,
            stride: a_stride,
        },
        b: TensorRef {
            data: dev_b.as_slice(),
            shape: b_shape,
            stride: b_stride,
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
            "scalar broadcast mismatch @ {i}: got {g} expected {e}"
        );
    }
}

// ============================================================================
// Test: rank-3 broadcast — `a: [1, S, 1]` + `b: [B, S, D]`
// ============================================================================
//
// Realistic transformer pattern: positional embedding broadcast across
// batch and feature dims. `a` has stride [0, D_dim, 0] — only varies
// along the sequence axis.
#[test]
#[ignore]
fn broadcast_add_rank3_seq_only() {
    let (ctx, stream) = setup();
    const B: usize = 4;
    const S: usize = 16;
    const D: usize = 32;
    let bb = B as i32;
    let ss = S as i32;
    let dd = D as i32;

    // a is logically [1, S, 1] — values along the sequence axis only.
    // Stored as S elements with stride [0, 1, 0] (where stride[1] = 1
    // because that's the only varying axis; we lay it out as a single
    // contig chunk of S elements).
    let a_host: Vec<f32> = (0..S).map(|i| (i as f32) * 0.1 - 0.5).collect();
    let b_host: Vec<f32> = (0..(B * S * D)).map(|i| (i as f32) * 0.01).collect();

    let a_shape = [1i32, ss, 1i32];
    let a_stride = [0i64, 1, 0];
    let b_shape = [bb, ss, dd];
    let b_stride = [(S * D) as i64, D as i64, 1];
    let y_shape = [bb, ss, dd];
    let y_stride = [(S * D) as i64, D as i64, 1];

    let expected = cpu_strided_add::<3>(
        &a_host, a_shape, a_stride,
        &b_host, b_shape, b_stride,
        y_shape,
    );

    let dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &b_host).expect("upload b");
    let mut dev_y: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, B * S * D).expect("alloc y");

    let desc = BinaryDescriptor {
        kind: BinaryKind::Add,
        shape: y_shape,
        element: ElementKind::F32,
    };
    let plan = BinaryPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = BinaryArgs::<f32, 3> {
        a: TensorRef {
            data: dev_a.as_slice(),
            shape: a_shape,
            stride: a_stride,
        },
        b: TensorRef {
            data: dev_b.as_slice(),
            shape: b_shape,
            stride: b_stride,
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: y_shape,
            stride: y_stride,
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; B * S * D];
    dev_y.copy_to_host(&mut got).expect("download");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "rank-3 seq-only broadcast mismatch @ {i}: got {g} expected {e}"
        );
    }
}

// ============================================================================
// Test: non-broadcast strided view (transposed B) + contig A → contig Y
// ============================================================================
//
// Confirms the strided kernel handles non-stride-0 cases too: a is a
// normal contig [M, N] tensor, b is logically [N, M]^T (a transposed
// view backed by an [M, N] buffer with swapped strides), output is
// contig [M, N]. The kernel reads b[i, j] = b_buf[j * M + i] via the
// swapped stride.
#[test]
#[ignore]
fn strided_add_transposed_b() {
    let (ctx, stream) = setup();
    const M: usize = 48;
    const N_DIM: usize = 32;
    let m = M as i32;
    let n = N_DIM as i32;

    let a_host: Vec<f32> = (0..(M * N_DIM))
        .map(|i| (i as f32) * 0.25 - 1.5)
        .collect();
    // b_buf is [N, M] contig. We view it as [M, N] by transposing
    // (stride [1, M] instead of contig [N, 1]). So b_logical[i, j] =
    // b_buf[j, i] = b_buf[j * M + i].
    let b_buf: Vec<f32> = (0..(N_DIM * M))
        .map(|i| (i as f32) * 0.1 + 2.0)
        .collect();

    let a_shape = [m, n];
    let a_stride = contiguous_stride([m, n]);
    let b_shape = [m, n];
    let b_stride = [1i64, M as i64]; // transposed
    let y_shape = [m, n];
    let y_stride = contiguous_stride([m, n]);

    let expected = cpu_strided_add::<2>(
        &a_host, a_shape, a_stride,
        &b_buf, b_shape, b_stride,
        y_shape,
    );

    let dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &b_buf).expect("upload b");
    let mut dev_y: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, M * N_DIM).expect("alloc y");

    let desc = BinaryDescriptor {
        kind: BinaryKind::Add,
        shape: y_shape,
        element: ElementKind::F32,
    };
    let plan = BinaryPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = BinaryArgs::<f32, 2> {
        a: TensorRef {
            data: dev_a.as_slice(),
            shape: a_shape,
            stride: a_stride,
        },
        b: TensorRef {
            data: dev_b.as_slice(),
            shape: b_shape,
            stride: b_stride,
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
            "transposed-B strided add mismatch @ {i}: got {g} expected {e}"
        );
    }
}
