//! Ill-conditioned input smoke test for the Phase 44b baracuda-owned
//! ozIMMU. Proves the slice-count knob actually moves the needle:
//! a Hilbert-like ill-conditioned matrix at S=6 should miss the
//! 1e-4 tolerance bar; the same matrix at S=18 should clear 1e-8.
//!
//! Not a strict accuracy claim — just a "did the algorithm respond
//! to the precision dial?" sanity check.

use baracuda_cublas::{gemm as cublas_gemm, Handle as CublasHandle, Op as CublasOp};
use baracuda_driver::{Context, Device, DeviceBuffer, Stream};
use baracuda_ozimmu::{Handle as OzimmuHandle, Op as OzimmuOp, OzakiSlices};

fn hilbert_like_matrix(n: usize, offset: f64) -> Vec<f64> {
    // Hilbert matrix H_{ij} = 1 / (i + j + offset + 1). The classic
    // ill-conditioned-by-design family.
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            // Column-major layout: cell (i, j) → out[j * n + i]
            out[j * n + i] = 1.0 / ((i + j) as f64 + offset + 1.0);
        }
    }
    out
}

fn relative_fro_error(got: &[f64], ref_: &[f64]) -> f64 {
    assert_eq!(got.len(), ref_.len());
    let mut num = 0.0;
    let mut den = 0.0;
    for (g, r) in got.iter().zip(ref_) {
        let d = g - r;
        num += d * d;
        den += r * r;
    }
    if den == 0.0 {
        return num.sqrt();
    }
    (num / den).sqrt()
}

fn run_hilbert(n: usize, slices: OzakiSlices) -> f64 {
    baracuda_driver::init().expect("driver init");
    let device = Device::get(0).expect("device");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let a_host = hilbert_like_matrix(n, 0.0);
    let b_host = hilbert_like_matrix(n, 1.0);

    let a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload A");
    let b = DeviceBuffer::from_slice(&ctx, &b_host).expect("upload B");

    let cublas_handle = CublasHandle::new().expect("cublas handle");
    let mut c_ref: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, n * n).expect("alloc c_ref");
    cublas_gemm(
        &cublas_handle,
        CublasOp::N, CublasOp::N,
        n as i32, n as i32, n as i32,
        1.0,
        &a, n as i32,
        &b, n as i32,
        0.0,
        &mut c_ref, n as i32,
    )
    .expect("cublas dgemm");

    let oz_handle = OzimmuHandle::new().expect("ozimmu handle");
    oz_handle.set_stream(&stream);
    let c_oz: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, n * n).expect("alloc c_oz");
    unsafe {
        oz_handle
            .dgemm(
                OzimmuOp::N, OzimmuOp::N,
                n, n, n,
                1.0,
                a.as_raw().0 as *const f64, n,
                b.as_raw().0 as *const f64, n,
                0.0,
                c_oz.as_raw().0 as *mut f64, n,
                slices,
            )
            .expect("ozimmu dgemm");
    }
    stream.synchronize().expect("stream sync");

    let mut got = vec![0.0f64; n * n];
    let mut ref_ = vec![0.0f64; n * n];
    c_oz.copy_to_host(&mut got).expect("D2H got");
    c_ref.copy_to_host(&mut ref_).expect("D2H ref");

    relative_fro_error(&got, &ref_)
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn slice_count_dial_responds_to_ill_conditioned_input() {
    let n = 128;
    let err_s6  = run_hilbert(n, OzakiSlices::S6);
    let err_s18 = run_hilbert(n, OzakiSlices::S18);

    // We don't pin absolute numbers (Hilbert is famously
    // hardware-noisy). The minimum claim is: S=18 must beat S=6 by
    // at least 10× on this ill-conditioned input.
    assert!(
        err_s18 < err_s6 / 10.0,
        "slice-count dial inactive: S=6 err={:e}, S=18 err={:e} (expected S=18 < S=6/10)",
        err_s6, err_s18
    );
}
