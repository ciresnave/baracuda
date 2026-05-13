//! Real-GPU smoke test for the bespoke FP8 E4M3 RCR Identity kernel in
//! `baracuda-kernels-sys` (the Phase 2 trailblazer).
//!
//! Compares the kernel's output to a CPU reference that does the
//! reduction in f32 — converting E4M3 inputs to f32 via the `float8`
//! crate (matches NVIDIA's `__nv_cvt_fp8_to_halfraw` lookup table),
//! accumulating, then quantizing back to E4M3 with `from_f32` (matches
//! NVIDIA's `__nv_cvt_float_to_fp8(_, __NV_SATFINITE, __NV_E4M3)`).
//!
//! Tensor-core MMA reduction order isn't bit-stable per the PTX spec,
//! so we allow ±1 step on the E4M3 grid (≈ ½ ULP at the value's
//! exponent bucket); the integer-like value patterns we feed minimize
//! how often rounding can disagree, so 0 mismatches is the expected
//! outcome.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --no-default-features --features sm89 \
//!  --release --test fp8_e4m3_rcr_smoke -- --ignored`.

#![cfg(feature = "sm89")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    EpilogueKind, Fp8E4M3, Fp8GemmArgs, Fp8GemmDescriptor, Fp8GemmPlan, LayoutSku, MatrixMut,
    MatrixRef, PlanPreference, Workspace,
};
use float8::F8E4M3;

// ============================================================================
// E4M3 grid spacing (1 ULP) — used as the per-cell tolerance.
// ============================================================================

/// Spacing between adjacent E4M3 grid values at magnitude `|v|`.
///
/// Subnormal range (|v| < 2^-6 = 1/64): uniform spacing = 2^-9 = 1/512.
/// Normal range: spacing = 2^(E-10) where the biased exponent
/// `E = floor(log2(|v|)) + 7` (E4M3 bias). Equivalently, spacing
/// scales by 2 across each octave; at |v|=1 spacing is 0.125, at
/// |v|=448 (max-finite) it's 32.
fn e4m3_grid_spacing(v: f32) -> f32 {
    let a = v.abs();
    if a == 0.0 || a < 1.0 / 64.0 {
        return 1.0 / 512.0;
    }
    let e_unb = a.log2().floor() as i32; // 2^e_unb <= |v| < 2^(e_unb+1)
    2f32.powi(e_unb - 3) // mantissa has 3 bits → 8 grid steps per octave
}

/// Bit-stable E4M3 cast — matches NVIDIA's
/// `__nv_cvt_float_to_fp8(_, __NV_SATFINITE, __NV_E4M3)`.
fn quantize_e4m3(x: f32) -> u8 {
    F8E4M3::from_f32(x).to_bits()
}

/// Decode an E4M3 byte to f32 — matches the GPU's
/// `__nv_cvt_fp8_to_halfraw(_, __NV_E4M3)` → `__half2float` round-trip.
fn dequantize_e4m3(bits: u8) -> f32 {
    F8E4M3::from_bits(bits).to_f32()
}

// ============================================================================
// CPU reference — RCR layout (col-major B, indexed b[j * ldb + k]).
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn cpu_fp8_e4m3_gemm_rcr(
    m: usize,
    n: usize,
    k: usize,
    a_bits: &[u8],
    lda: usize, // row stride of A (>= K)
    b_bits: &[u8],
    ldb: usize, // column stride of B (>= K)
    alpha: f32,
    expected_bits: &mut [u8], // M*N, row-major
    expected_f32: &mut [f32], // same shape, pre-quantization
    ldd: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut acc: f32 = 0.0;
            for kk in 0..k {
                let a_val = dequantize_e4m3(a_bits[i * lda + kk]);
                // Col-major B: B[k, j] = b_bits[j * ldb + k]
                let b_val = dequantize_e4m3(b_bits[j * ldb + kk]);
                acc += a_val * b_val;
            }
            let z = alpha * acc;
            expected_f32[i * ldd + j] = z;
            expected_bits[i * ldd + j] = quantize_e4m3(z);
        }
    }
}

// ============================================================================
// Test harness
// ============================================================================

fn run_fp8_e4m3_rcr_identity(m: i32, n: i32, k: i32) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let mu = m as usize;
    let nu = n as usize;
    let ku = k as usize;

    // Inputs on the E4M3 grid, magnitudes bounded so the f32
    // accumulator (max ≈ K · 1.0 · 1.0 = K) stays well below the
    // E4M3 max-finite (448). Patterns are deterministic and span
    // both signs so the saturating-cast path is exercised.
    let host_a_bits: Vec<u8> = (0..(mu * ku))
        .map(|i| {
            let v = (((i as i32 * 5) % 13) as f32 - 6.0) * 0.125; // ∈ [-0.75, +0.75]
            quantize_e4m3(v)
        })
        .collect();
    let host_b_bits: Vec<u8> = (0..(ku * nu))
        .map(|i| {
            let v = (((i as i32 * 7) % 11) as f32 - 5.0) * 0.125; // ∈ [-0.625, +0.625]
            quantize_e4m3(v)
        })
        .collect();

    let alpha: f32 = 0.25;
    let beta: f32 = 0.0;

    let mut expected_bits = vec![0u8; mu * nu];
    let mut expected_f32 = vec![0f32; mu * nu];
    cpu_fp8_e4m3_gemm_rcr(
        mu, nu, ku,
        &host_a_bits, ku,
        &host_b_bits, ku, // B is col-major [K, N] with column stride = K
        alpha,
        &mut expected_bits,
        &mut expected_f32,
        nu,
    );

    let dev_a_bytes = DeviceBuffer::from_slice(&ctx, &host_a_bits).expect("upload A");
    let dev_b_bytes = DeviceBuffer::from_slice(&ctx, &host_b_bits).expect("upload B");
    let dev_a = dev_a_bytes.view_as::<Fp8E4M3>();
    let dev_b = dev_b_bytes.view_as::<Fp8E4M3>();
    let mut dev_d: DeviceBuffer<Fp8E4M3> =
        DeviceBuffer::zeros(&ctx, mu * nu).expect("alloc D");

    let desc = Fp8GemmDescriptor {
        m, n, k,
        layout: LayoutSku::Rcr,
        epilogue: EpilogueKind::Identity,
    };
    let plan = Fp8GemmPlan::<Fp8E4M3>::select(&stream, &desc, PlanPreference::default())
        .expect("select FP8 E4M3 RCR plan");

    let args = Fp8GemmArgs::<Fp8E4M3> {
        a: MatrixRef { data: dev_a, rows: m, cols: k, ld: k as i64 },
        // RCR: B is col-major [K, N] with column stride = K
        b: MatrixRef { data: dev_b, rows: k, cols: n, ld: k as i64 },
        c: None,
        d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n as i64 },
        bias: None,
        alpha,
        beta,
    };
    plan.run(&stream, Workspace::None, args)
        .expect("FP8 E4M3 RCR GEMM run");
    stream.synchronize().expect("stream sync");

    let mut host_d_bits = vec![Fp8E4M3(0); mu * nu];
    dev_d.copy_to_host(&mut host_d_bits).expect("download D");

    let mut mismatches = 0usize;
    let mut first_mismatch: Option<(usize, u8, u8, f32, f32, f32)> = None;
    for (idx, (got, &expected)) in host_d_bits.iter().zip(expected_bits.iter()).enumerate() {
        let got_f = dequantize_e4m3(got.0);
        let exp_f = dequantize_e4m3(expected);
        // Tolerance: 1 grid step at the larger of |got| or |expected|.
        let spacing = e4m3_grid_spacing(got_f.abs().max(exp_f.abs()));
        let delta = (got_f - exp_f).abs();
        if delta > spacing + 1e-7 {
            mismatches += 1;
            if first_mismatch.is_none() {
                first_mismatch = Some((idx, got.0, expected, got_f, exp_f, expected_f32[idx]));
            }
        }
    }
    if mismatches > 0 {
        let (idx, gb, eb, gf, ef, raw) = first_mismatch.unwrap();
        panic!(
            "{mismatches} mismatches across {} cells \
             (M={m} N={n} K={k}); first @ idx {idx}: \
             got bits=0x{gb:02x} ({gf}) expected bits=0x{eb:02x} ({ef}); \
             pre-quant f32 ref = {raw}",
            host_d_bits.len(),
        );
    }
}

// ============================================================================
// Tests — Identity-only at four shapes that span tile-aligned and
// ragged cases (mirrors the int8 RRR trailblazer's coverage).
// ============================================================================

#[test] #[ignore]
fn fp8_e4m3_rcr_identity_64_64_32() {
    run_fp8_e4m3_rcr_identity(64, 64, 32);
}

#[test] #[ignore]
fn fp8_e4m3_rcr_identity_128_128_128() {
    run_fp8_e4m3_rcr_identity(128, 128, 128);
}

#[test] #[ignore]
fn fp8_e4m3_rcr_identity_256_128_64() {
    run_fp8_e4m3_rcr_identity(256, 128, 64);
}

#[test] #[ignore]
fn fp8_e4m3_rcr_identity_100_70_50() {
    run_fp8_e4m3_rcr_identity(100, 70, 50);
}
