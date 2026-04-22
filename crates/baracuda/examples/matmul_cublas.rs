//! Minimal cuBLAS matmul demo: multiply two random column-major float
//! matrices via `cublasSgemm` and verify the result against a CPU reference.
//!
//! ```text
//! cargo run --example matmul_cublas --features "cublas curand"
//! ```

use baracuda::cublas::{gemm, Handle, Op};
use baracuda::curand::{Generator, RngKind};
use baracuda::driver::{Context, Device, DeviceBuffer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    baracuda::driver::init()?;
    let device = Device::get(0)?;
    println!("device: {}", device.name()?);
    let ctx = Context::new(&device)?;
    let handle = Handle::new()?;

    // C = A × B with A: m×k, B: k×n, C: m×n, column-major.
    let (m, k, n) = (256, 128, 192);
    let lda = m;
    let ldb = k;
    let ldc = m;

    // Fill A and B with uniform random [0, 1) values on-device.
    let mut d_a: DeviceBuffer<f32> = DeviceBuffer::new(&ctx, m * k)?;
    let mut d_b: DeviceBuffer<f32> = DeviceBuffer::new(&ctx, k * n)?;
    let mut d_c: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, m * n)?;

    let gen = Generator::new(RngKind::Philox4_32_10)?;
    gen.seed(0xDEAD_BEEF)?;
    gen.uniform(&mut d_a)?;
    gen.uniform(&mut d_b)?;

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

    // Pull everything back and CPU-verify on a random sample.
    let mut a_host = vec![0.0f32; m * k];
    let mut b_host = vec![0.0f32; k * n];
    let mut c_host = vec![0.0f32; m * n];
    d_a.copy_to_host(&mut a_host)?;
    d_b.copy_to_host(&mut b_host)?;
    d_c.copy_to_host(&mut c_host)?;

    let mut max_err = 0.0f32;
    for j in (0..n).step_by(7) {
        for i in (0..m).step_by(5) {
            let mut expected = 0.0f32;
            for p in 0..k {
                expected += a_host[p * lda + i] * b_host[j * ldb + p];
            }
            let got = c_host[j * ldc + i];
            let err = (got - expected).abs();
            max_err = max_err.max(err);
        }
    }

    println!("cuBLAS {m}×{k} × {k}×{n} Sgemm — max abs error vs CPU: {max_err}");
    if max_err > 1e-2 {
        return Err(format!("excessive error: {max_err}").into());
    }
    println!("OK");
    Ok(())
}
