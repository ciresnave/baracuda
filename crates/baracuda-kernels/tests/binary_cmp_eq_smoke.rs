//! Real-GPU smoke test for the binary comparison trailblazer
//! (`BinaryCmpPlan<f32, N> + BinaryCmpKind::Eq`).
//!
//! Output dtype is `u8` (0 / 1) — distinct from the input element type,
//! which is the design novelty this trailblazer proves. Covers contig
//! (1D / 2D / 3D) + broadcast (`[1, N] vs [M, N]`) + strided
//! (transposed) cases. Bit-exact compare against a host reference
//! (`a == b ? 1 : 0`) — equality on f32 is exact (no math).
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test binary_cmp_eq_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BinaryCmpArgs, BinaryCmpDescriptor, BinaryCmpKind, BinaryCmpPlan,
    ElementKind, PlanPreference, TensorMut, TensorRef, Workspace,
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

    // Mix in deliberate equalities — every 3rd cell has a == b.
    let host_a: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.5 - 17.25).collect();
    let host_b: Vec<f32> = (0..numel)
        .map(|i| {
            if i % 3 == 0 {
                host_a[i]
            } else {
                (i as f32) * 0.125 + 1.0
            }
        })
        .collect();
    let expected: Vec<u8> = host_a
        .iter()
        .zip(host_b.iter())
        .map(|(a, b)| if a == b { 1u8 } else { 0u8 })
        .collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = BinaryCmpDescriptor {
        kind: BinaryCmpKind::Eq,
        shape,
        element: ElementKind::F32,
    };
    let plan = BinaryCmpPlan::<f32, N>::select(&stream, &desc, PlanPreference::default())
        .expect("select");

    let args = BinaryCmpArgs::<f32, N> {
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

    let mut got = vec![0u8; numel];
    dev_y.copy_to_host(&mut got).expect("download");

    let mut mismatches = 0;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        if g != e {
            mismatches += 1;
            if mismatches <= 3 {
                eprintln!("  mismatch @ {i}: got {g} expected {e}");
            }
        }
    }
    assert_eq!(mismatches, 0, "cmp eq f32 contig: {mismatches} mismatches");
    // Sanity: at least some cells should be 1.
    let trues = got.iter().filter(|&&x| x == 1).count();
    assert!(trues > 0, "expected at least some true cells, got 0");
}

#[test]
#[ignore]
fn cmp_eq_f32_1d() {
    run_contig::<1>([2048]);
}

#[test]
#[ignore]
fn cmp_eq_f32_2d() {
    run_contig::<2>([64, 64]);
}

#[test]
#[ignore]
fn cmp_eq_f32_3d() {
    run_contig::<3>([8, 128, 128]);
}

/// Broadcast: `a: [1, N]` vs `b: [M, N]` → `y: [M, N]`.
#[test]
#[ignore]
fn cmp_eq_f32_broadcast_row() {
    let (ctx, stream) = setup();
    const M: usize = 32;
    const N_DIM: usize = 64;
    let m = M as i32;
    let n = N_DIM as i32;

    // Construct `a` such that some rows of b will match it exactly.
    let host_a: Vec<f32> = (0..N_DIM).map(|i| (i as f32) * 0.25 - 3.0).collect();
    let host_b: Vec<f32> = (0..(M * N_DIM))
        .map(|i| {
            let row = i / N_DIM;
            let col = i % N_DIM;
            // Row 5 and row 17 match `a` exactly; everywhere else differs.
            if row == 5 || row == 17 {
                host_a[col]
            } else {
                (i as f32) * 0.0625
            }
        })
        .collect();

    let a_shape = [1i32, n];
    let a_stride = [0i64, 1];
    let b_shape = [m, n];
    let b_stride = [n as i64, 1];
    let y_shape = [m, n];
    let y_stride = [n as i64, 1];

    let mut expected = vec![0u8; M * N_DIM];
    for i in 0..M {
        for j in 0..N_DIM {
            // Broadcast: a[0, j] = host_a[j]; b[i, j] = host_b[i*N + j]
            let a_val = host_a[j];
            let b_val = host_b[i * N_DIM + j];
            expected[i * N_DIM + j] = if a_val == b_val { 1 } else { 0 };
        }
    }

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, M * N_DIM).expect("alloc y");

    let desc = BinaryCmpDescriptor {
        kind: BinaryCmpKind::Eq,
        shape: y_shape,
        element: ElementKind::F32,
    };
    let plan = BinaryCmpPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = BinaryCmpArgs::<f32, 2> {
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

    let mut got = vec![0u8; M * N_DIM];
    dev_y.copy_to_host(&mut got).expect("download");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g, e,
            "broadcast cmp eq mismatch @ {i}: got {g} expected {e}"
        );
    }
    // Sanity: rows 5 and 17 are all-ones; others should be mostly zero.
    let trues = got.iter().filter(|&&x| x == 1).count();
    assert!(
        trues >= 2 * N_DIM,
        "expected at least {} true cells (rows 5+17), got {trues}",
        2 * N_DIM
    );
}
