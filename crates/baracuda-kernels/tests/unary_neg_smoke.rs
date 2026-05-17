//! Real-GPU smoke test for the Phase 3 unary trailblazer kernel
//! (`UnaryPlan<f32, N> + UnaryKind::Neg`).
//!
//! Covers both the contig fast path (1D / 2D / 3D / 4D) and the
//! strided path (transposed view), with bit-exact compare in both
//! cases — `-x` on f32 is exact (one PTX `neg.f32` instruction per
//! element).
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test unary_neg_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef, UnaryArgs,
    UnaryDescriptor, UnaryKind, UnaryPlan, Workspace,
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
    assert!(numel > 0, "test shape must have non-zero element count");

    let host_x: Vec<f32> = (0..numel)
        .map(|i| (i as f32) * 0.5 - 17.25)
        .collect();
    let expected: Vec<f32> = host_x.iter().map(|x| -x).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = UnaryDescriptor {
        kind: UnaryKind::Neg,
        shape,
        element: ElementKind::F32,
    };
    let plan = UnaryPlan::<f32, N>::select(&stream, &desc, PlanPreference::default())
        .expect("select UnaryPlan<f32, N>");

    let args = UnaryArgs::<f32, N> {
        x: TensorRef {
            data: dev_x.as_slice(),
            shape,
            stride,
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape,
            stride,
        },
    };
    plan.run(&stream, Workspace::None, args).expect("unary neg run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("download");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "unary neg f32 contig mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn neg_f32_1d() {
    run_contig::<1>([2048]);
}

#[test]
#[ignore]
fn neg_f32_1d_huge() {
    // Exercises the grid-cap loop.
    run_contig::<1>([1 << 20]);
}

#[test]
#[ignore]
fn neg_f32_2d() {
    run_contig::<2>([64, 64]);
}

#[test]
#[ignore]
fn neg_f32_3d() {
    run_contig::<3>([8, 128, 128]);
}

#[test]
#[ignore]
fn neg_f32_4d() {
    run_contig::<4>([2, 32, 8, 64]);
}

#[test]
#[ignore]
fn neg_f32_ragged_1d() {
    run_contig::<1>([2049]);
}

// ============================================================================
// Strided path: transposed view
// ============================================================================
//
// Input is stored as `[N, M]` row-major but viewed logically as `[M, N]`
// with swapped strides (stride `[1, M]` instead of contig `[N, 1]`).
// Output is plain contig `[M, N]`. The strided kernel reads
// `x_logical[i, j] = x_buf[j * M + i]` via the swapped stride.
#[test]
#[ignore]
fn neg_f32_strided_transposed() {
    let (ctx, stream) = setup();
    const M: usize = 48;
    const N_DIM: usize = 32;
    let m = M as i32;
    let n = N_DIM as i32;

    // x_buf is N_DIM rows of M cols, contig. We view it as M×N_DIM with
    // swapped strides — so x_logical[i, j] = x_buf[j, i] = x_buf[j*M + i].
    let x_buf: Vec<f32> = (0..(N_DIM * M))
        .map(|i| (i as f32) * 0.25 - 1.5)
        .collect();

    let x_shape = [m, n];
    let x_stride = [1i64, M as i64]; // transposed
    let y_shape = [m, n];
    let y_stride = contiguous_stride([m, n]);

    // CPU reference walks the same strides.
    let numel = (M * N_DIM) as usize;
    let mut expected = vec![0f32; numel];
    for i in 0..M {
        for j in 0..N_DIM {
            // x_logical[i, j] follows x_stride [1, M] → byte offset
            // i*1 + j*M = j*M + i.
            let x_lin = j * M + i;
            let y_lin = i * N_DIM + j; // contig
            expected[y_lin] = -x_buf[x_lin];
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &x_buf).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let desc = UnaryDescriptor {
        kind: UnaryKind::Neg,
        shape: y_shape,
        element: ElementKind::F32,
    };
    let plan = UnaryPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = UnaryArgs::<f32, 2> {
        x: TensorRef {
            data: dev_x.as_slice(),
            shape: x_shape,
            stride: x_stride,
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: y_shape,
            stride: y_stride,
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
            "unary neg strided transposed mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
fn select_rejects_non_neg_today() {
    // Confirm select rejects reserved-but-unimplemented variants. After
    // the full unary FP fanout (trivial math + transcendentals + math /
    // rounding + activations across {f32, f16, bf16, f64}), BitwiseNot
    // (int-only) stands in as a still-reserved discriminant — it sits
    // outside the FP-only `kind_in_scope` and is unlikely to land soon.
    let desc = UnaryDescriptor {
        kind: UnaryKind::BitwiseNot,
        shape: [4],
        element: ElementKind::F32,
    };
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    let err = UnaryPlan::<f32, 1>::select(&stream, &desc, PlanPreference::default());
    assert!(
        err.is_err(),
        "BitwiseNot is reserved but not implemented today; select must reject"
    );
}
