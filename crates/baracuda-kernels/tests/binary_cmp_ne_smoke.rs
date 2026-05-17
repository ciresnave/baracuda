//! Real-GPU smoke test for `BinaryCmpPlan<T, N> + BinaryCmpKind::Ne`
//! across the 4 FP dtypes (f32 / f16 / bf16 / f64).
//!
//! Output dtype is `u8` (0 / 1). Comparisons produce exact u8 results
//! with no rounding step, so bit-exact compare against a host
//! reference applies for ALL dtypes. Each test mixes deliberate
//! equalities (every 3rd cell → 0) and non-equalities (other cells →
//! 1) so the kernel exercises both outcomes.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test binary_cmp_ne_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BinaryCmpArgs, BinaryCmpDescriptor, BinaryCmpKind, BinaryCmpPlan,
    ElementKind, PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// =========================================================================
// Contig: f32
// =========================================================================
#[test]
#[ignore]
fn cmp_ne_f32_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    // Inputs in [-10, 10]. Every 3rd cell is deliberately equal.
    let host_a: Vec<f32> = (0..numel)
        .map(|i| ((i % 41) as f32) * 0.5 - 10.0)
        .collect();
    let host_b: Vec<f32> = (0..numel)
        .map(|i| {
            if i % 3 == 0 {
                host_a[i]
            } else {
                ((i % 37) as f32) * 0.25 - 4.5
            }
        })
        .collect();
    let expected: Vec<u8> = host_a
        .iter()
        .zip(host_b.iter())
        .map(|(a, b)| if a != b { 1 } else { 0 })
        .collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = BinaryCmpDescriptor {
        kind: BinaryCmpKind::Ne,
        shape,
        element: ElementKind::F32,
    };
    let plan = BinaryCmpPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = BinaryCmpArgs::<f32, 3> {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0u8; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "cmp ne f32 mismatch @ {i}");
    }
    let zeros = got.iter().filter(|&&x| x == 0).count();
    let ones = got.iter().filter(|&&x| x == 1).count();
    assert!(zeros > 0 && ones > 0, "expected mixed 0s and 1s, got zeros={zeros} ones={ones}");
}

// =========================================================================
// Contig: f16
// =========================================================================
#[test]
#[ignore]
fn cmp_ne_f16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    let host_a: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32(((i % 41) as f32) * 0.5 - 10.0))
        .collect();
    let host_b: Vec<f16> = (0..numel)
        .map(|i| {
            if i % 3 == 0 {
                host_a[i]
            } else {
                f16::from_f32(((i % 37) as f32) * 0.25 - 4.5)
            }
        })
        .collect();
    let expected: Vec<u8> = host_a
        .iter()
        .zip(host_b.iter())
        .map(|(a, b)| if a != b { 1 } else { 0 })
        .collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = BinaryCmpDescriptor {
        kind: BinaryCmpKind::Ne,
        shape,
        element: ElementKind::F16,
    };
    let plan = BinaryCmpPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = BinaryCmpArgs::<f16, 3> {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0u8; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "cmp ne f16 mismatch @ {i}");
    }
    let zeros = got.iter().filter(|&&x| x == 0).count();
    let ones = got.iter().filter(|&&x| x == 1).count();
    assert!(zeros > 0 && ones > 0, "expected mixed 0s and 1s, got zeros={zeros} ones={ones}");
}

// =========================================================================
// Contig: bf16
// =========================================================================
#[test]
#[ignore]
fn cmp_ne_bf16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    let host_a: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32(((i % 41) as f32) * 0.5 - 10.0))
        .collect();
    let host_b: Vec<bf16> = (0..numel)
        .map(|i| {
            if i % 3 == 0 {
                host_a[i]
            } else {
                bf16::from_f32(((i % 37) as f32) * 0.25 - 4.5)
            }
        })
        .collect();
    let expected: Vec<u8> = host_a
        .iter()
        .zip(host_b.iter())
        .map(|(a, b)| if a != b { 1 } else { 0 })
        .collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = BinaryCmpDescriptor {
        kind: BinaryCmpKind::Ne,
        shape,
        element: ElementKind::Bf16,
    };
    let plan = BinaryCmpPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = BinaryCmpArgs::<bf16, 3> {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0u8; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "cmp ne bf16 mismatch @ {i}");
    }
    let zeros = got.iter().filter(|&&x| x == 0).count();
    let ones = got.iter().filter(|&&x| x == 1).count();
    assert!(zeros > 0 && ones > 0, "expected mixed 0s and 1s, got zeros={zeros} ones={ones}");
}

// =========================================================================
// Contig: f64
// =========================================================================
#[test]
#[ignore]
fn cmp_ne_f64_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    let host_a: Vec<f64> = (0..numel)
        .map(|i| ((i % 41) as f64) * 0.5 - 10.0)
        .collect();
    let host_b: Vec<f64> = (0..numel)
        .map(|i| {
            if i % 3 == 0 {
                host_a[i]
            } else {
                ((i % 37) as f64) * 0.25 - 4.5
            }
        })
        .collect();
    let expected: Vec<u8> = host_a
        .iter()
        .zip(host_b.iter())
        .map(|(a, b)| if a != b { 1 } else { 0 })
        .collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = BinaryCmpDescriptor {
        kind: BinaryCmpKind::Ne,
        shape,
        element: ElementKind::F64,
    };
    let plan = BinaryCmpPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = BinaryCmpArgs::<f64, 3> {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0u8; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "cmp ne f64 mismatch @ {i}");
    }
    let zeros = got.iter().filter(|&&x| x == 0).count();
    let ones = got.iter().filter(|&&x| x == 1).count();
    assert!(zeros > 0 && ones > 0, "expected mixed 0s and 1s, got zeros={zeros} ones={ones}");
}

// =========================================================================
// Strided: f32 transposed view
// =========================================================================
//
// `a` is stored as `[N, M]` row-major but viewed as `[M, N]` with
// swapped strides (`[1, M]` instead of contig `[N, 1]`). `b` is plain
// contig `[M, N]`. The strided kernel reads
// `a_logical[i, j] = a_buf[j * M + i]` via the swapped stride.
#[test]
#[ignore]
fn cmp_ne_f32_strided_transposed() {
    let (ctx, stream) = setup();
    const M: usize = 48;
    const N_DIM: usize = 32;
    let m = M as i32;
    let n = N_DIM as i32;

    // `a_buf` is N_DIM rows × M cols, contig.
    let a_buf: Vec<f32> = (0..(N_DIM * M))
        .map(|i| ((i % 41) as f32) * 0.5 - 10.0)
        .collect();
    // `b` is contig [M, N_DIM]. Mix deliberate equalities every 3rd cell.
    let b_buf: Vec<f32> = (0..(M * N_DIM))
        .map(|i| {
            let row = i / N_DIM;
            let col = i % N_DIM;
            if i % 3 == 0 {
                // Make a_logical[row, col] == b[row, col].
                // a_logical[row, col] = a_buf[col * M + row].
                a_buf[col * M + row]
            } else {
                ((i % 37) as f32) * 0.25 - 4.5
            }
        })
        .collect();

    let a_shape = [m, n];
    let a_stride = [1i64, M as i64]; // transposed
    let b_shape = [m, n];
    let b_stride = contiguous_stride([m, n]);
    let y_shape = [m, n];
    let y_stride = contiguous_stride([m, n]);

    let numel = M * N_DIM;
    let mut expected = vec![0u8; numel];
    for i in 0..M {
        for j in 0..N_DIM {
            let a_val = a_buf[j * M + i];
            let b_val = b_buf[i * N_DIM + j];
            expected[i * N_DIM + j] = if a_val != b_val { 1 } else { 0 };
        }
    }

    let dev_a = DeviceBuffer::from_slice(&ctx, &a_buf).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &b_buf).expect("upload b");
    let mut dev_y: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let desc = BinaryCmpDescriptor {
        kind: BinaryCmpKind::Ne,
        shape: y_shape,
        element: ElementKind::F32,
    };
    let plan = BinaryCmpPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = BinaryCmpArgs::<f32, 2> {
        a: TensorRef { data: dev_a.as_slice(), shape: a_shape, stride: a_stride },
        b: TensorRef { data: dev_b.as_slice(), shape: b_shape, stride: b_stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: y_shape, stride: y_stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0u8; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "cmp ne f32 strided mismatch @ {i}");
    }
    let zeros = got.iter().filter(|&&x| x == 0).count();
    let ones = got.iter().filter(|&&x| x == 1).count();
    assert!(zeros > 0 && ones > 0, "expected mixed 0s and 1s, got zeros={zeros} ones={ones}");
}
