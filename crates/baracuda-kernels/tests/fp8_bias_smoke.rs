//! Real-GPU smoke test for the FP8 bias / activation epilogue family:
//! 16 SKUs spanning `{E4M3, E5M2} × {RCR, RRR} × {Bias, BiasRelu,
//! BiasGelu, BiasSilu}`. One shape (128×128×128) per SKU.
//!
//! Reference activation chain matches the int8 RRR bias smoke:
//!   * relu = `max(x, 0)`
//!   * gelu = `0.5 * x * (1 + erf(x / sqrt(2)))` via Abramowitz-Stegun
//!   * silu = `x / (1 + exp(-x))`
//!
//! Tolerance: 1 grid step for Bias / BiasRelu (the activation is
//! linear / piecewise-linear so the f32 path is exact); 2 grid steps
//! for BiasGelu / BiasSilu — libdevice's `erff` / `__expf` diverge
//! from libm by ~1.5e-7 relative, which can shift the quantized
//! output by one extra step at the right magnitude bucket.

#![cfg(feature = "sm89")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    EpilogueKind, Fp8E4M3, Fp8E5M2, Fp8GemmArgs, Fp8GemmDescriptor, Fp8GemmPlan, FpElement,
    LayoutSku, MatrixMut, MatrixRef, PlanPreference, VectorRef, Workspace,
};
use float8::{F8E4M3, F8E5M2};

// ============================================================================
// Activation reference (mirrors int8_rrr_smoke.rs).
// ============================================================================

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn erf_approx(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let p = 0.3275911_f32;
    let a1 = 0.254829592_f32;
    let a2 = -0.284496736_f32;
    let a3 = 1.421413741_f32;
    let a4 = -1.453152027_f32;
    let a5 = 1.061405429_f32;
    let t = 1.0 / (1.0 + p * ax);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-ax * ax).exp();
    sign * y
}

fn gelu_exact(x: f32) -> f32 {
    let inv_sqrt_2 = 1.0 / std::f32::consts::SQRT_2;
    0.5 * x * (1.0 + erf_approx(x * inv_sqrt_2))
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn apply_activation(ep: EpilogueKind, x: f32) -> f32 {
    match ep {
        EpilogueKind::Identity | EpilogueKind::Bias => x,
        EpilogueKind::BiasRelu => relu(x),
        EpilogueKind::BiasGelu => gelu_exact(x),
        EpilogueKind::BiasSilu => silu(x),
    }
}

// ============================================================================
// Per-encoding shims.
// ============================================================================

/// Bundles the encoding-specific bits of an FP8 element: host
/// quantize/dequantize, the per-magnitude grid-step tolerance, and
/// a zero-bit constructor for download buffers.
trait Fp8Shim: FpElement + Default {
    fn quantize(x: f32) -> u8;
    fn dequantize(bits: u8) -> f32;
    fn grid_spacing(v: f32) -> f32;
    /// Extract the underlying `u8` storage. Both wrappers are
    /// `#[repr(transparent)]` over u8, so this is a no-op cast.
    fn to_bits(self) -> u8;
}

impl Fp8Shim for Fp8E4M3 {
    fn quantize(x: f32) -> u8 { F8E4M3::from_f32(x).to_bits() }
    fn dequantize(bits: u8) -> f32 { F8E4M3::from_bits(bits).to_f32() }
    fn grid_spacing(v: f32) -> f32 {
        let a = v.abs();
        if a == 0.0 || a < 1.0 / 64.0 {
            return 1.0 / 512.0;
        }
        let e_unb = a.log2().floor() as i32;
        2f32.powi(e_unb - 3)
    }
    fn to_bits(self) -> u8 { self.0 }
}

impl Fp8Shim for Fp8E5M2 {
    fn quantize(x: f32) -> u8 { F8E5M2::from_f32(x).to_bits() }
    fn dequantize(bits: u8) -> f32 { F8E5M2::from_bits(bits).to_f32() }
    fn grid_spacing(v: f32) -> f32 {
        let a = v.abs();
        if a == 0.0 || a < (1.0_f32 / (1u32 << 14) as f32) {
            return 1.0_f32 / (1u32 << 16) as f32;
        }
        let e_unb = a.log2().floor() as i32;
        2f32.powi(e_unb - 2)
    }
    fn to_bits(self) -> u8 { self.0 }
}

// ============================================================================
// CPU reference (per-layout indexing).
// ============================================================================

fn cpu_fp8_gemm<T: Fp8Shim>(
    m: usize,
    n: usize,
    k: usize,
    a_bits: &[u8],
    lda: usize,
    b_bits: &[u8],
    ldb: usize,
    layout: LayoutSku,
    bias: &[f32],
    alpha: f32,
    ep: EpilogueKind,
    expected_bits: &mut [u8],
    expected_f32: &mut [f32],
    ldd: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut acc: f32 = 0.0;
            for kk in 0..k {
                let a_val = T::dequantize(a_bits[i * lda + kk]);
                let b_val = match layout {
                    // RCR: col-major B → b[j * ldb + k]
                    LayoutSku::Rcr => T::dequantize(b_bits[j * ldb + kk]),
                    // RRR: row-major B → b[k * ldb + j]
                    LayoutSku::Rrr => T::dequantize(b_bits[kk * ldb + j]),
                };
                acc += a_val * b_val;
            }
            let mut z = alpha * acc + bias[j];
            z = apply_activation(ep, z);
            expected_f32[i * ldd + j] = z;
            expected_bits[i * ldd + j] = T::quantize(z);
        }
    }
}

// ============================================================================
// Bias generator (same shape as int8_rrr_smoke::mk_bias_f32).
// ============================================================================

fn mk_bias_f32(n: usize) -> Vec<f32> {
    (0..n).map(|j| ((j as f32 % 5.0) - 2.0) * 0.5).collect()
}

// ============================================================================
// Generic harness.
// ============================================================================

fn run_fp8_bias<T: Fp8Shim>(
    m: i32,
    n: i32,
    k: i32,
    layout: LayoutSku,
    ep: EpilogueKind,
) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let mu = m as usize;
    let nu = n as usize;
    let ku = k as usize;

    // Small-magnitude inputs keep the accumulator and the post-bias
    // / post-activation result well inside both E4M3 (±448) and E5M2
    // (±57344) range.
    let host_a_bits: Vec<u8> = (0..(mu * ku))
        .map(|i| {
            let v = (((i as i32 * 5) % 13) as f32 - 6.0) * 0.125;
            T::quantize(v)
        })
        .collect();
    let host_b_bits: Vec<u8> = (0..(ku * nu))
        .map(|i| {
            let v = (((i as i32 * 7) % 11) as f32 - 5.0) * 0.125;
            T::quantize(v)
        })
        .collect();
    let host_bias = mk_bias_f32(nu);

    let alpha: f32 = 0.25;
    let beta: f32 = 0.0;

    let (ldb, _ldb_purpose) = match layout {
        LayoutSku::Rcr => (ku, "col-stride (>=K)"),
        LayoutSku::Rrr => (nu, "row-stride (>=N)"),
    };

    let mut expected_bits = vec![0u8; mu * nu];
    let mut expected_f32 = vec![0f32; mu * nu];
    cpu_fp8_gemm::<T>(
        mu, nu, ku,
        &host_a_bits, ku,
        &host_b_bits, ldb,
        layout,
        &host_bias,
        alpha,
        ep,
        &mut expected_bits,
        &mut expected_f32,
        nu,
    );

    let dev_a_bytes = DeviceBuffer::from_slice(&ctx, &host_a_bits).expect("upload A");
    let dev_b_bytes = DeviceBuffer::from_slice(&ctx, &host_b_bits).expect("upload B");
    let dev_bias = DeviceBuffer::from_slice(&ctx, &host_bias).expect("upload bias");
    let dev_a = dev_a_bytes.view_as::<T>();
    let dev_b = dev_b_bytes.view_as::<T>();
    let mut dev_d: DeviceBuffer<T> = DeviceBuffer::zeros(&ctx, mu * nu).expect("alloc D");

    let desc = Fp8GemmDescriptor { m, n, k, layout, epilogue: ep };
    let plan = Fp8GemmPlan::<T>::select(&stream, &desc, PlanPreference::default())
        .expect("select FP8 plan");

    let args = Fp8GemmArgs::<T> {
        a: MatrixRef { data: dev_a, rows: m, cols: k, ld: k as i64 },
        b: MatrixRef { data: dev_b, rows: k, cols: n, ld: ldb as i64 },
        c: None,
        d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n as i64 },
        bias: Some(VectorRef {
            data: dev_bias.as_slice(),
            len: n,
            stride: 1,
        }),
        alpha,
        beta,
    };
    plan.run(&stream, Workspace::None, args)
        .expect("FP8 bias GEMM run");
    stream.synchronize().expect("stream sync");

    let mut host_d: Vec<T> = vec![T::default(); mu * nu];
    dev_d.copy_to_host(&mut host_d).expect("download D");
    let host_d_bits: Vec<u8> = host_d.into_iter().map(T::to_bits).collect();

    let allowed_steps: f32 = match ep {
        EpilogueKind::BiasGelu | EpilogueKind::BiasSilu => 2.0,
        _ => 1.0,
    };

    let mut mismatches = 0usize;
    let mut first_mismatch: Option<(usize, u8, u8, f32, f32, f32)> = None;
    for (idx, (got, &expected)) in host_d_bits.iter().zip(expected_bits.iter()).enumerate() {
        let got_f = T::dequantize(*got);
        let exp_f = T::dequantize(expected);
        let spacing = T::grid_spacing(got_f.abs().max(exp_f.abs()));
        let delta = (got_f - exp_f).abs();
        if delta > allowed_steps * spacing + 1e-7 {
            mismatches += 1;
            if first_mismatch.is_none() {
                first_mismatch = Some((idx, *got, expected, got_f, exp_f, expected_f32[idx]));
            }
        }
    }
    if mismatches > 0 {
        let (idx, gb, eb, gf, ef, raw) = first_mismatch.unwrap();
        panic!(
            "{mismatches} mismatches across {} cells \
             ({:?} {:?} M={m} N={n} K={k}); first @ idx {idx}: \
             got bits=0x{gb:02x} ({gf}) expected bits=0x{eb:02x} ({ef}); \
             pre-quant f32 ref = {raw}",
            host_d_bits.len(),
            layout, ep,
        );
    }
}

// ============================================================================
// Tests — 16 SKUs at one shape each (128³).
// ============================================================================

const M: i32 = 128;
const N: i32 = 128;
const K: i32 = 128;

// ---- E4M3 × RCR ----

#[test] #[ignore]
fn fp8_e4m3_rcr_bias()      { run_fp8_bias::<Fp8E4M3>(M, N, K, LayoutSku::Rcr, EpilogueKind::Bias); }
#[test] #[ignore]
fn fp8_e4m3_rcr_bias_relu() { run_fp8_bias::<Fp8E4M3>(M, N, K, LayoutSku::Rcr, EpilogueKind::BiasRelu); }
#[test] #[ignore]
fn fp8_e4m3_rcr_bias_gelu() { run_fp8_bias::<Fp8E4M3>(M, N, K, LayoutSku::Rcr, EpilogueKind::BiasGelu); }
#[test] #[ignore]
fn fp8_e4m3_rcr_bias_silu() { run_fp8_bias::<Fp8E4M3>(M, N, K, LayoutSku::Rcr, EpilogueKind::BiasSilu); }

// ---- E4M3 × RRR ----

#[test] #[ignore]
fn fp8_e4m3_rrr_bias()      { run_fp8_bias::<Fp8E4M3>(M, N, K, LayoutSku::Rrr, EpilogueKind::Bias); }
#[test] #[ignore]
fn fp8_e4m3_rrr_bias_relu() { run_fp8_bias::<Fp8E4M3>(M, N, K, LayoutSku::Rrr, EpilogueKind::BiasRelu); }
#[test] #[ignore]
fn fp8_e4m3_rrr_bias_gelu() { run_fp8_bias::<Fp8E4M3>(M, N, K, LayoutSku::Rrr, EpilogueKind::BiasGelu); }
#[test] #[ignore]
fn fp8_e4m3_rrr_bias_silu() { run_fp8_bias::<Fp8E4M3>(M, N, K, LayoutSku::Rrr, EpilogueKind::BiasSilu); }

// ---- E5M2 × RCR ----

#[test] #[ignore]
fn fp8_e5m2_rcr_bias()      { run_fp8_bias::<Fp8E5M2>(M, N, K, LayoutSku::Rcr, EpilogueKind::Bias); }
#[test] #[ignore]
fn fp8_e5m2_rcr_bias_relu() { run_fp8_bias::<Fp8E5M2>(M, N, K, LayoutSku::Rcr, EpilogueKind::BiasRelu); }
#[test] #[ignore]
fn fp8_e5m2_rcr_bias_gelu() { run_fp8_bias::<Fp8E5M2>(M, N, K, LayoutSku::Rcr, EpilogueKind::BiasGelu); }
#[test] #[ignore]
fn fp8_e5m2_rcr_bias_silu() { run_fp8_bias::<Fp8E5M2>(M, N, K, LayoutSku::Rcr, EpilogueKind::BiasSilu); }

// ---- E5M2 × RRR ----

#[test] #[ignore]
fn fp8_e5m2_rrr_bias()      { run_fp8_bias::<Fp8E5M2>(M, N, K, LayoutSku::Rrr, EpilogueKind::Bias); }
#[test] #[ignore]
fn fp8_e5m2_rrr_bias_relu() { run_fp8_bias::<Fp8E5M2>(M, N, K, LayoutSku::Rrr, EpilogueKind::BiasRelu); }
#[test] #[ignore]
fn fp8_e5m2_rrr_bias_gelu() { run_fp8_bias::<Fp8E5M2>(M, N, K, LayoutSku::Rrr, EpilogueKind::BiasGelu); }
#[test] #[ignore]
fn fp8_e5m2_rrr_bias_silu() { run_fp8_bias::<Fp8E5M2>(M, N, K, LayoutSku::Rrr, EpilogueKind::BiasSilu); }
