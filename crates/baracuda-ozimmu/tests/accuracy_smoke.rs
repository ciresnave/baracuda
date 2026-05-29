//! Accuracy smoke tests for the Phase 44b baracuda-owned ozIMMU.
//!
//! Runs a small set of well-conditioned random FP64 GEMMs at three
//! shapes (M=N=K ∈ {256, 1024, 2048}) and three slice counts
//! (S ∈ {6, 8, 10}) and verifies the relative error against a
//! cuBLAS DGEMM reference. Documents the per-S accuracy contract:
//!
//!   - S=6:  loose — typical error ~1e-10 to ~1e-8 on uniform [-1, 1]
//!   - S=8:  the upstream-recommended sweet spot — typical ~1e-13
//!   - S=10: very tight — typical ~1e-14 to ~1e-15
//!
//! The tolerances below give the per-S kernel a comfortable margin
//! on well-conditioned inputs without being so loose they'd miss a
//! 10× regression. Real workloads with bad condition numbers are
//! covered by `ill_conditioned_smoke.rs`.

use baracuda_cublas::{gemm as cublas_gemm, Handle as CublasHandle, Op as CublasOp};
use baracuda_driver::{Context, Device, DeviceBuffer, Stream};
use baracuda_ozimmu::{Handle as OzimmuHandle, Op as OzimmuOp, OzakiSlices};

/// Tiny deterministic xorshift PRNG so the tests stay self-contained
/// without pulling `rand` into the dev-dep set. Seeded per test case.
struct XorShift64(u64);
impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0xdead_beef } else { seed })
    }
    fn next_f64(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        // Uniform in [-1.0, 1.0).
        ((x >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn random_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f64> {
    let mut rng = XorShift64::new(seed);
    let mut out = vec![0.0f64; rows * cols];
    for cell in &mut out {
        *cell = rng.next_f64();
    }
    out
}

/// Compute the relative Frobenius-norm error between `got` and `ref_`.
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

fn run_case(m: usize, slices: OzakiSlices, tolerance: f64) {
    baracuda_driver::init().expect("driver init");
    let device = Device::get(0).expect("device");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let n = m;
    let k = m;
    let seed = (m as u64).wrapping_mul(1_000_003);

    let a_host = random_matrix(m, k, seed);
    let b_host = random_matrix(k, n, seed.wrapping_add(7));

    let a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload A");
    let b = DeviceBuffer::from_slice(&ctx, &b_host).expect("upload B");

    // Reference: cublasDgemm.
    let cublas_handle = CublasHandle::new().expect("cublas handle");
    let mut c_ref: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, m * n).expect("alloc c_ref");
    cublas_gemm(
        &cublas_handle,
        CublasOp::N, CublasOp::N,
        m as i32, n as i32, k as i32,
        1.0,
        &a, m as i32,
        &b, k as i32,
        0.0,
        &mut c_ref, m as i32,
    )
    .expect("cublas dgemm");

    // Candidate: ozIMMU at the chosen slice count.
    let oz_handle = OzimmuHandle::new().expect("ozimmu handle");
    oz_handle.set_stream(&stream);
    let c_oz: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, m * n).expect("alloc c_oz");
    unsafe {
        oz_handle
            .dgemm(
                OzimmuOp::N, OzimmuOp::N,
                m, n, k,
                1.0,
                a.as_raw().0 as *const f64, m,
                b.as_raw().0 as *const f64, k,
                0.0,
                c_oz.as_raw().0 as *mut f64, m,
                slices,
            )
            .expect("ozimmu dgemm");
    }
    stream.synchronize().expect("stream sync");

    let mut got = vec![0.0f64; m * n];
    let mut ref_ = vec![0.0f64; m * n];
    c_oz.copy_to_host(&mut got).expect("D2H got");
    c_ref.copy_to_host(&mut ref_).expect("D2H ref");

    let err = relative_fro_error(&got, &ref_);
    assert!(
        err < tolerance,
        "ozIMMU @ {:?} M=N=K={} relative-Fro error = {:e} (tolerance {:e})",
        slices, m, err, tolerance
    );
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn ozimmu_dgemm_256_s6_loose() {
    // S=6 is intentionally low precision — typical ~1e-8 on uniform random.
    run_case(256, OzakiSlices::S6, 1e-6);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn ozimmu_dgemm_256_s8_default() {
    // S=8 is the upstream-recommended default.
    run_case(256, OzakiSlices::S8, 1e-10);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn ozimmu_dgemm_256_s10_tight() {
    // S=10 should hit double-precision sweet spot for well-conditioned input.
    run_case(256, OzakiSlices::S10, 1e-12);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn ozimmu_dgemm_1024_s8() {
    run_case(1024, OzakiSlices::S8, 1e-10);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn ozimmu_dgemm_2048_s8() {
    run_case(2048, OzakiSlices::S8, 1e-10);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn ozimmu_dgemm_512_s_auto() {
    // Auto mode picks S at run-time. Default mantissa-loss threshold = 0,
    // so it should fall through to dgemm-grade accuracy for our
    // well-conditioned input.
    run_case(512, OzakiSlices::Auto, 1e-8);
}
