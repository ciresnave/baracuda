//! Minimal cuBLAS `Sgemm` demo.
//!
//! Multiplies two small column-major `f32` matrices on the GPU and
//! verifies the result element-by-element against a CPU reference.
//!
//! Run with:
//!
//! ```text
//! cargo run --example gemm -p baracuda-cublas
//! ```

use baracuda_cublas::{gemm, Handle, Op};
use baracuda_driver::{Context, Device, DeviceBuffer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    baracuda_driver::init()?;
    let device = Device::get(0)?;
    println!("device: {}", device.name()?);
    let ctx = Context::new(&device)?;
    let handle = Handle::new()?;

    // C = A × B with A: m×k, B: k×n, C: m×n, column-major.
    // Use a small 2×3 × 3×2 case so we can spell out the expected matrix.
    let (m, k, n) = (2usize, 3usize, 2usize);
    let lda = m;
    let ldb = k;
    let ldc = m;

    // Column-major: A is stored column-by-column.
    //   A = [[1, 2, 3],
    //        [4, 5, 6]]  →  [1, 4, 2, 5, 3, 6]
    //   B = [[ 7,  8],
    //        [ 9, 10],
    //        [11, 12]]   →  [7, 9, 11, 8, 10, 12]
    let a_host: Vec<f32> = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let b_host: Vec<f32> = vec![7.0, 9.0, 11.0, 8.0, 10.0, 12.0];

    let d_a = DeviceBuffer::from_slice(&ctx, &a_host)?;
    let d_b = DeviceBuffer::from_slice(&ctx, &b_host)?;
    let mut d_c: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, m * n)?;

    gemm(
        &handle,
        Op::N,
        Op::N,
        m as i32,
        n as i32,
        k as i32,
        1.0,
        &d_a,
        lda as i32,
        &d_b,
        ldb as i32,
        0.0,
        &mut d_c,
        ldc as i32,
    )?;

    let mut c_host = vec![0.0f32; m * n];
    d_c.copy_to_host(&mut c_host)?;

    // Expected (row-major presentation):
    //   C = [[ 58,  64],
    //        [139, 154]]
    // Column-major storage: [58, 139, 64, 154]
    let expected: [f32; 4] = [58.0, 139.0, 64.0, 154.0];
    for j in 0..n {
        for i in 0..m {
            let got = c_host[j * ldc + i];
            let want = expected[j * ldc + i];
            assert!(
                (got - want).abs() < 1e-3,
                "mismatch at ({i},{j}): {got} vs {want}"
            );
        }
    }
    println!("cuBLAS {m}×{k} · {k}×{n} Sgemm OK; C (col-major) = {c_host:?}");
    Ok(())
}
