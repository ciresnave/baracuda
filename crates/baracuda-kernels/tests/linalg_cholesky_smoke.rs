//! Real-GPU smoke test for `CholeskyPlan` (cuSOLVER wrap).
//!
//! Verifies `L · L^T ≈ A` for an SPD matrix `A`. Storage convention is
//! column-major end-to-end (cuSOLVER native) — the SPD matrix is
//! symmetric so row-major and column-major views coincide bit-for-bit,
//! which keeps the test grounded.
//!
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, CholeskyArgs, CholeskyDescriptor, CholeskyPlan, ElementKind,
    PlanPreference, TensorMut, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Build a deterministic `n × n` SPD matrix `A = M · M^T + n · I`. The
/// shifted identity guarantees `A` is positive-definite for any seed.
/// Storage is column-major (which equals row-major because A is
/// symmetric).
fn spd_matrix_f32(n: usize, seed: u32) -> Vec<f32> {
    // Build M with a simple deterministic pattern.
    let mut m = vec![0f32; n * n];
    let mut s = seed.wrapping_mul(0x9E37_79B1);
    for v in m.iter_mut() {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
        let f = ((s >> 8) as f32 / (1u32 << 24) as f32) - 0.5;
        *v = f;
    }
    // A = M · M^T + n · I
    let mut a = vec![0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0f32;
            for kk in 0..n {
                acc += m[i * n + kk] * m[j * n + kk];
            }
            a[i * n + j] = acc;
        }
        a[i * n + i] += n as f32;
    }
    a
}

fn spd_matrix_f64(n: usize, seed: u32) -> Vec<f64> {
    let a32 = spd_matrix_f32(n, seed);
    a32.into_iter().map(|v| v as f64).collect()
}

#[test]
#[ignore]
fn cholesky_f32_basic() {
    let (ctx, stream) = setup();
    let n: i32 = 6;
    let a_host = spd_matrix_f32(n as usize, 0xC0DE_F00D);

    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc info");

    let desc = CholeskyDescriptor {
        matrix_size: n,
        batch_size: 1,
        lower: true,
        element: ElementKind::F32,
    };
    let plan = CholeskyPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select CholeskyPlan<f32>");

    // Query workspace size.
    let ws_bytes = plan.query_workspace_size(&stream).expect("ws query");
    assert!(ws_bytes > 0, "cuSOLVER reported zero workspace for potrf");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    let shape = [1i32, n, n];
    let stride = contiguous_stride(shape);
    let args = CholeskyArgs::<f32> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape,
            stride,
        },
        info: TensorMut {
            data: dev_info.as_slice_mut(),
            shape: [1i32],
            stride: [1],
        },
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run cholesky f32");
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; 1];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    assert_eq!(info_host[0], 0, "cholesky info != 0 (matrix not SPD?)");

    let mut l_host = vec![0f32; (n * n) as usize];
    dev_a.copy_to_host(&mut l_host).expect("dl L");

    // The matrix A is symmetric so row-major == column-major for A.
    //
    // The plan was told `lower: true` (row-major lower-L convention)
    // and internally flipped this to `uplo = UPPER` for cuSOLVER
    // (column-major). cuSOLVER's `potrf(uplo=UPPER)` factors
    // `A = U^T · U` where `U` is **upper-triangular in column-major**.
    // In a column-major buffer, cell (k, i) lives at offset `i*n + k`;
    // an upper-triangular column-major matrix has non-zero entries
    // only where `k <= i` (row index <= col index).
    //
    // Zero the strict lower triangle (column-major: row > col) of the
    // returned buffer — cuSOLVER leaves those cells with their pre-
    // factorization contents.
    let n_usize = n as usize;
    for j in 0..n_usize {
        for i in 0..n_usize {
            if i > j {
                l_host[j * n_usize + i] = 0.0;
            }
        }
    }
    // Reconstruct `A = U^T · U` in column-major:
    //   A[i, j] = Σ_k U[k, i] · U[k, j]
    // where `U[k, i] = l_host[i*n + k]` (column-major indexing).
    let mut reconstructed = vec![0f32; n_usize * n_usize];
    for i in 0..n_usize {
        for j in 0..n_usize {
            let mut acc = 0f32;
            for kk in 0..n_usize {
                acc += l_host[i * n_usize + kk] * l_host[j * n_usize + kk];
            }
            reconstructed[j * n_usize + i] = acc;
        }
    }

    let tol = 1e-4f32;
    for i in 0..n_usize {
        for j in 0..n_usize {
            let got = reconstructed[j * n_usize + i];
            let expected = a_host[j * n_usize + i];
            let diff = (got - expected).abs();
            let t = tol * expected.abs().max(1.0);
            assert!(
                diff <= t,
                "f32 cholesky reconstruct ({i},{j}): got={got}, expected={expected}, diff={diff}",
            );
        }
    }
}

#[test]
#[ignore]
fn cholesky_f64_basic() {
    let (ctx, stream) = setup();
    let n: i32 = 6;
    let a_host = spd_matrix_f64(n as usize, 0xDEAD_BEEF);

    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc info");

    let desc = CholeskyDescriptor {
        matrix_size: n,
        batch_size: 1,
        lower: true,
        element: ElementKind::F64,
    };
    let plan = CholeskyPlan::<f64>::select(&stream, &desc, PlanPreference::default())
        .expect("select CholeskyPlan<f64>");
    let ws_bytes = plan.query_workspace_size(&stream).expect("ws query");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    let shape = [1i32, n, n];
    let stride = contiguous_stride(shape);
    let args = CholeskyArgs::<f64> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape,
            stride,
        },
        info: TensorMut {
            data: dev_info.as_slice_mut(),
            shape: [1i32],
            stride: [1],
        },
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run cholesky f64");
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; 1];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    assert_eq!(info_host[0], 0);

    let n_usize = n as usize;
    let mut l_host = vec![0f64; n_usize * n_usize];
    dev_a.copy_to_host(&mut l_host).expect("dl L");
    // See f32 test for the column-major upper-triangular U convention.
    for j in 0..n_usize {
        for i in 0..n_usize {
            if i > j {
                l_host[j * n_usize + i] = 0.0;
            }
        }
    }
    let mut reconstructed = vec![0f64; n_usize * n_usize];
    for i in 0..n_usize {
        for j in 0..n_usize {
            let mut acc = 0f64;
            for kk in 0..n_usize {
                acc += l_host[i * n_usize + kk] * l_host[j * n_usize + kk];
            }
            reconstructed[j * n_usize + i] = acc;
        }
    }
    let tol = 1e-10f64;
    for i in 0..n_usize {
        for j in 0..n_usize {
            let got = reconstructed[j * n_usize + i];
            let expected = a_host[j * n_usize + i];
            let diff = (got - expected).abs();
            let t = tol * expected.abs().max(1.0);
            assert!(
                diff <= t,
                "f64 cholesky reconstruct ({i},{j}): got={got}, expected={expected}, diff={diff}",
            );
        }
    }
}
