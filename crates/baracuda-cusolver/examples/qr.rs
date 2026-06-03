//! Minimal cuSOLVER dense QR factorization demo.
//!
//! Factorizes a small column-major `f32` matrix into `A = Q · R` via
//! `cusolverDnSgeqrf` and prints the upper-triangular `R` plus the
//! Householder reflector scalars `tau`. After `geqrf` the input buffer
//! holds `R` in the upper triangle and the implicit reflectors in the
//! lower triangle.
//!
//! Run with:
//!
//! ```text
//! cargo run --example qr -p baracuda-cusolver
//! ```

use baracuda_cusolver::{geqrf, DnHandle};
use baracuda_driver::{Context, Device, DeviceBuffer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    baracuda_driver::init()?;
    let device = Device::get(0)?;
    println!("device: {}", device.name()?);
    let ctx = Context::new(&device)?;
    let handle = DnHandle::new()?;

    // 3×3 column-major:
    //   A = [[12, -51,   4],
    //        [ 6, 167, -68],
    //        [-4,  24, -41]]
    // Stored as columns: [12, 6, -4,  -51, 167, 24,  4, -68, -41]
    let m: i32 = 3;
    let n: i32 = 3;
    let lda: i32 = m;
    let a_host: [f32; 9] = [12.0, 6.0, -4.0, -51.0, 167.0, 24.0, 4.0, -68.0, -41.0];

    let mut a = DeviceBuffer::from_slice(&ctx, &a_host)?;
    let mut tau: DeviceBuffer<f32> = DeviceBuffer::new(&ctx, m.min(n) as usize)?;
    let mut info: DeviceBuffer<i32> = DeviceBuffer::new(&ctx, 1)?;

    geqrf::<f32>(&handle, m, n, &mut a, lda, &mut tau, &mut info)?;

    let mut info_host = [42i32];
    info.copy_to_host(&mut info_host)?;
    assert_eq!(info_host[0], 0, "geqrf info = {}", info_host[0]);

    let mut a_out = [0.0f32; 9];
    a.copy_to_host(&mut a_out)?;
    let mut tau_out = vec![0.0f32; tau.len()];
    tau.copy_to_host(&mut tau_out)?;

    // Print R (the upper triangle of `a` after geqrf, column-major).
    println!("R (upper triangle, in column-major positions):");
    for i in 0..m as usize {
        for j in 0..n as usize {
            if i <= j {
                print!("{:10.4} ", a_out[j * lda as usize + i]);
            } else {
                print!("{:>10} ", "·");
            }
        }
        println!();
    }
    println!("tau = {tau_out:?}");

    // For this fixture the magnitude of R[0,0] is sqrt(12^2 + 6^2 + 4^2) = 14.
    let r00 = a_out[0].abs();
    assert!((r00 - 14.0).abs() < 1e-3, "R[0,0]={r00}, expected 14");
    println!("OK (|R[0,0]| = {r00})");
    Ok(())
}
