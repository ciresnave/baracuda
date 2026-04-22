//! GPU-gated integration tests for cuBLAS Sgemm + Saxpy.

use baracuda_cublas::{axpy, gemm, Handle, Op};
use baracuda_driver::{Context, Device, DeviceBuffer};

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn sgemm_2x3_times_3x2() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let handle = Handle::new().unwrap();

    // Column-major 2×3:   [[1, 2, 3],
    //                      [4, 5, 6]]  → stored as [1, 4, 2, 5, 3, 6]
    let a_host: [f32; 6] = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];

    // Column-major 3×2:   [[ 7,  8],
    //                      [ 9, 10],
    //                      [11, 12]]   → stored as [7, 9, 11, 8, 10, 12]
    let b_host: [f32; 6] = [7.0, 9.0, 11.0, 8.0, 10.0, 12.0];

    // Expected (2×2):  A × B = [[58, 64], [139, 154]]
    //                          column-major: [58, 139, 64, 154]
    let expected: [f32; 4] = [58.0, 139.0, 64.0, 154.0];

    let a = DeviceBuffer::from_slice(&ctx, &a_host).unwrap();
    let b = DeviceBuffer::from_slice(&ctx, &b_host).unwrap();
    let mut c: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 4).unwrap();

    gemm(
        &handle,
        Op::N,
        Op::N,
        2,
        2,
        3,
        1.0,
        &a,
        2,
        &b,
        3,
        0.0,
        &mut c,
        2,
    )
    .expect("Sgemm");

    // cuBLAS is stream-ordered; default stream synchronizes via the implicit
    // sync on cuMemcpy D2H below.
    let mut got = [0.0f32; 4];
    c.copy_to_host(&mut got).unwrap();

    for (g, e) in got.iter().zip(&expected) {
        assert!(
            (g - e).abs() < 1e-3,
            "Sgemm mismatch: got {got:?}, expected {expected:?}"
        );
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn saxpy_basic() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let handle = Handle::new().unwrap();

    let n = 1024;
    let x_host: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let y_host: Vec<f32> = (0..n).map(|i| i as f32 * 2.0).collect();
    let alpha = 3.0f32;
    let expected: Vec<f32> = x_host
        .iter()
        .zip(&y_host)
        .map(|(x, y)| alpha * x + y)
        .collect();

    let x = DeviceBuffer::from_slice(&ctx, &x_host).unwrap();
    let mut y = DeviceBuffer::from_slice(&ctx, &y_host).unwrap();
    let x_slice = x.as_slice();
    axpy(&handle, n as i32, alpha, &x_slice, 1, &mut y, 1).expect("Saxpy");

    let mut got = vec![0.0f32; n];
    y.copy_to_host(&mut got).unwrap();
    for (g, e) in got.iter().zip(&expected) {
        assert!((g - e).abs() < 1e-3);
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn strided_batched_sgemm_small() {
    use baracuda_cublas::gemm_strided_batched;

    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let handle = Handle::new().unwrap();

    // 4 parallel 2×2 × 2×2 products, column-major. B_i = identity, so C_i = A_i.
    let batch = 4;
    let m: i32 = 2;
    let k: i32 = 2;
    let n: i32 = 2;
    let per_matrix: usize = 4;

    let mut a_host: Vec<f32> = Vec::with_capacity(per_matrix * batch);
    let mut b_host: Vec<f32> = Vec::with_capacity(per_matrix * batch);
    for i in 0..batch {
        let f = i as f32;
        a_host.extend_from_slice(&[1.0 + f, 3.0 + f, 2.0 + f, 4.0 + f]);
        b_host.extend_from_slice(&[1.0, 0.0, 0.0, 1.0]);
    }

    let d_a = DeviceBuffer::from_slice(&ctx, &a_host).unwrap();
    let d_b = DeviceBuffer::from_slice(&ctx, &b_host).unwrap();
    let mut d_c: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, per_matrix * batch).unwrap();

    gemm_strided_batched(
        &handle,
        Op::N,
        Op::N,
        m,
        n,
        k,
        1.0f32,
        &d_a,
        m,
        per_matrix as i64,
        &d_b,
        k,
        per_matrix as i64,
        0.0f32,
        &mut d_c,
        m,
        per_matrix as i64,
        batch as i32,
    )
    .expect("cublasSgemmStridedBatched");

    let mut got = vec![0.0f32; per_matrix * batch];
    d_c.copy_to_host(&mut got).unwrap();
    for (g, e) in got.iter().zip(&a_host) {
        assert!(
            (g - e).abs() < 1e-3,
            "batched Sgemm mismatch: got {got:?}, expected {a_host:?}"
        );
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn version_query_works() {
    baracuda_driver::init().unwrap();
    let _device = Device::get(0).unwrap();
    let _ctx = Context::new(&_device).unwrap();
    let handle = Handle::new().unwrap();
    let v = handle.version().unwrap();
    eprintln!("cuBLAS version: {v}");
    assert!(v >= 11040);
}
