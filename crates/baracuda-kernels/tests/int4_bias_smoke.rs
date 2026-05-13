//! Real-GPU smoke tests for the bespoke int4 bias-family kernels in
//! `baracuda-kernels-sys` — 32 SKUs total:
//!
//!   `{S4, U4} × {RCR, RRR} × {Bias, BiasRelu, BiasGelu, BiasSilu} ×
//!    {f32 bias, i32 bias}`
//!
//! Each test verifies that the kernel matches a CPU reference for the
//! int32-accumulate → f32-compute → bias-add → activation →
//! saturating-cast-to-int4 chain. Comparisons are tolerated to ±1 on
//! the int4 cast output: the f32-domain activation chain (gelu, silu)
//! is not bit-stable between libdevice (CUDA) and Rust's libm
//! equivalents.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --no-default-features --features sm89 \
//!  --release --test int4_bias_smoke -- --ignored`.

#![cfg(feature = "sm89")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    ActivationKind, BiasElement, EpilogueKind, Int4GemmArgs, Int4GemmDescriptor, Int4GemmPlan,
    IntElement, LayoutSku, MatrixMut, MatrixRef, PlanPreference, S4, U4, VectorRef, Workspace,
};

// ============================================================================
// Nibble pack / unpack helpers — host-side mirror of `baracuda_dtype.cuh`.
// ============================================================================

#[inline]
fn s4_decode(nibble: u8) -> i32 {
    ((nibble & 0x0F) << 4) as i8 as i32 >> 4
}

#[inline]
fn u4_decode(nibble: u8) -> i32 {
    (nibble & 0x0F) as i32
}

#[inline]
fn pack_pair(lo: i32, hi: i32) -> u8 {
    (lo as u8 & 0x0F) | ((hi as u8 & 0x0F) << 4)
}

#[inline]
fn unpack_s4(byte: u8) -> [i32; 2] {
    [s4_decode(byte & 0x0F), s4_decode((byte >> 4) & 0x0F)]
}

#[inline]
fn unpack_u4(byte: u8) -> [i32; 2] {
    [u4_decode(byte & 0x0F), u4_decode((byte >> 4) & 0x0F)]
}

#[inline]
fn sat_cast_s4(x: f32) -> u8 {
    (((x.round_ties_even() as i32).clamp(-8, 7)) as u8) & 0x0F
}

#[inline]
fn sat_cast_u4(x: f32) -> u8 {
    (((x.round_ties_even() as i32).clamp(0, 15)) as u8) & 0x0F
}

// ============================================================================
// CPU-side activation chain (mirrors GPU `apply_activation_f32<Act>`).
// ============================================================================

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn erf_approx(x: f32) -> f32 {
    // Abramowitz-Stegun 7.1.26 — 5-term polynomial. Max error 1.5e-7.
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

fn apply_activation(act: Option<ActivationKind>, x: f32) -> f32 {
    match act {
        None => x,
        Some(ActivationKind::Relu) => relu(x),
        Some(ActivationKind::Gelu) => gelu_exact(x),
        Some(ActivationKind::Silu) => silu(x),
    }
}

// ============================================================================
// CPU reference — int4 GEMM with bias + activation, RCR or RRR layout.
//
// A: row-major [M, K], pair-packed along K.
// B: col-major [K, N] (RCR) — pair-packed along K (packed-pair byte at
//    `j * ldb + kk_byte` holds `(B[2*kk_byte, j], B[2*kk_byte+1, j])`),
//    or row-major [K, N] (RRR) — pair-packed along N (byte at
//    `k * ldb + j_pair` holds `(B[k, 2*j_pair], B[k, 2*j_pair+1])`).
// D: row-major [M, N], pair-packed along N.
// bias: [N], f32 or i32.
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn cpu_int4_gemm_bias(
    m: usize,
    n: usize,
    k: usize,
    a_bytes: &[u8],
    lda_bytes: usize,
    b_bytes: &[u8],
    ldb_bytes: usize,
    layout: LayoutSku,
    bias_f32: &[f32], // already coerced to f32 by the test harness
    alpha: f32,
    activation: Option<ActivationKind>,
    is_signed: bool,
    expected_bytes: &mut [u8],
    ldd_bytes: usize,
) {
    assert_eq!(k % 2, 0);
    assert_eq!(n % 2, 0);
    let k_bytes = k / 2;
    let n_bytes = n / 2;
    let unpack = if is_signed { unpack_s4 } else { unpack_u4 };
    let sat_cast = if is_signed { sat_cast_s4 } else { sat_cast_u4 };

    for i in 0..m {
        let mut acc_row = vec![0i32; n];
        match layout {
            LayoutSku::Rcr => {
                // B col-major: byte at (kk_byte, j) lives at `j * ldb + kk_byte`.
                for kk_byte in 0..k_bytes {
                    let [a_lo, a_hi] = unpack(a_bytes[i * lda_bytes + kk_byte]);
                    for j in 0..n {
                        let [b_lo, b_hi] = unpack(b_bytes[j * ldb_bytes + kk_byte]);
                        acc_row[j] += a_lo * b_lo + a_hi * b_hi;
                    }
                }
            }
            LayoutSku::Rrr => {
                // B row-major pair-packed along N: byte at (k, j_pair)
                // lives at `k * ldb + j_pair` and holds (B[k, 2j], B[k, 2j+1]).
                for kk_byte in 0..k_bytes {
                    let [a_lo, a_hi] = unpack(a_bytes[i * lda_bytes + kk_byte]);
                    let k_lo = 2 * kk_byte;
                    let k_hi = k_lo + 1;
                    for j_pair in 0..n_bytes {
                        let j0 = 2 * j_pair;
                        let j1 = j0 + 1;
                        let [b_lo_lo, b_lo_hi] = unpack(b_bytes[k_lo * ldb_bytes + j_pair]);
                        let [b_hi_lo, b_hi_hi] = unpack(b_bytes[k_hi * ldb_bytes + j_pair]);
                        acc_row[j0] += a_lo * b_lo_lo + a_hi * b_hi_lo;
                        acc_row[j1] += a_lo * b_lo_hi + a_hi * b_hi_hi;
                    }
                }
            }
        }
        // Apply alpha + bias + activation + sat-cast back to int4.
        for j_pair in 0..n_bytes {
            let j0 = 2 * j_pair;
            let j1 = j0 + 1;
            let mut v0 = alpha * acc_row[j0] as f32;
            let mut v1 = alpha * acc_row[j1] as f32;
            v0 += bias_f32[j0];
            v1 += bias_f32[j1];
            v0 = apply_activation(activation, v0);
            v1 = apply_activation(activation, v1);
            // `sat_cast` already returns the low-nibble representation
            // (s4: two's-complement in `[0, 15]`; u4: `[0, 15]`), so
            // assemble the packed byte directly from the raw nibbles.
            let q0 = sat_cast(v0);
            let q1 = sat_cast(v1);
            expected_bytes[i * ldd_bytes + j_pair] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }
    }
}

// ============================================================================
// Per-SKU harness — generic over T ∈ {S4, U4} and BT ∈ {f32, i32}.
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn run_int4_bias_smoke<T, BT>(
    m: i32,
    n: i32,
    k: i32,
    layout: LayoutSku,
    epilogue: EpilogueKind,
    is_signed: bool,
    mk_a: fn(usize, usize) -> i32,
    mk_b: fn(usize, usize) -> i32,
    mk_bias: fn(usize) -> Vec<BT>,
    bias_to_f32: fn(BT) -> f32,
    abs_max_a: i32,
    abs_max_b: i32,
    out_max_abs: f32,
) where
    T: IntElement + Default + 'static,
    BT: BiasElement + Copy + Default + 'static,
{
    assert!(epilogue.requires_bias(), "this harness is bias-only");

    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let mu = m as usize;
    let nu = n as usize;
    let ku = k as usize;
    let k_bytes = ku / 2;
    let n_bytes = nu / 2;

    // Pack A row-major [M, K] along K.
    let mut host_a_bytes = vec![0u8; mu * k_bytes];
    for i in 0..mu {
        for kk_byte in 0..k_bytes {
            host_a_bytes[i * k_bytes + kk_byte] =
                pack_pair(mk_a(i, 2 * kk_byte), mk_a(i, 2 * kk_byte + 1));
        }
    }
    // Pack B according to layout.
    //   RCR: col-major [K, N] along K — byte at (kk_byte, j) at offset
    //        `j * k_bytes + kk_byte` holds (B[2*kk_byte, j], B[2*kk_byte+1, j]).
    //   RRR: row-major [K, N] along N — byte at (k, j_pair) at offset
    //        `k * n_bytes + j_pair` holds (B[k, 2*j_pair], B[k, 2*j_pair+1]).
    let (host_b_bytes, ldb_bytes_host) = match layout {
        LayoutSku::Rcr => {
            let mut buf = vec![0u8; nu * k_bytes];
            for j in 0..nu {
                for kk_byte in 0..k_bytes {
                    buf[j * k_bytes + kk_byte] =
                        pack_pair(mk_b(2 * kk_byte, j), mk_b(2 * kk_byte + 1, j));
                }
            }
            (buf, k_bytes)
        }
        LayoutSku::Rrr => {
            let mut buf = vec![0u8; ku * n_bytes];
            for kk in 0..ku {
                for j_pair in 0..n_bytes {
                    buf[kk * n_bytes + j_pair] =
                        pack_pair(mk_b(kk, 2 * j_pair), mk_b(kk, 2 * j_pair + 1));
                }
            }
            (buf, n_bytes)
        }
    };

    // Bias vector (typed) and an f32 mirror for the CPU reference.
    let host_bias: Vec<BT> = mk_bias(nu);
    let host_bias_f32: Vec<f32> = host_bias.iter().copied().map(bias_to_f32).collect();

    // Scale alpha so `alpha * acc` lands in the int4 saturating range
    // before bias-add (bias values are O(2), well within the int4 cast
    // headroom). `|acc| <= K * abs_max_a * abs_max_b`.
    let alpha: f32 = out_max_abs / ((abs_max_a * abs_max_b) as f32 * k as f32);
    let beta: f32 = 0.0;

    let mut expected_bytes = vec![0u8; mu * n_bytes];
    cpu_int4_gemm_bias(
        mu, nu, ku,
        &host_a_bytes, k_bytes,
        &host_b_bytes, ldb_bytes_host,
        layout,
        &host_bias_f32,
        alpha,
        epilogue.activation(),
        is_signed,
        &mut expected_bytes,
        n_bytes,
    );

    let dev_a_bytes = DeviceBuffer::from_slice(&ctx, &host_a_bytes).expect("upload A");
    let dev_b_bytes = DeviceBuffer::from_slice(&ctx, &host_b_bytes).expect("upload B");
    let dev_a = dev_a_bytes.view_as::<T>();
    let dev_b = dev_b_bytes.view_as::<T>();
    let dev_bias = DeviceBuffer::from_slice(&ctx, &host_bias).expect("upload bias");
    let mut dev_d: DeviceBuffer<T> =
        DeviceBuffer::zeros(&ctx, mu * n_bytes).expect("alloc D");

    let desc = Int4GemmDescriptor { m, n, k, layout, epilogue };
    let plan = Int4GemmPlan::<T, BT>::select(&stream, &desc, PlanPreference::default())
        .expect("select int4 bias plan");

    let args = Int4GemmArgs::<T, BT> {
        a: MatrixRef { data: dev_a, rows: m, cols: k, ld: k_bytes as i64 },
        b: MatrixRef {
            data: dev_b,
            rows: k,
            cols: n,
            ld: ldb_bytes_host as i64,
        },
        c: None,
        d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n_bytes as i64 },
        bias: Some(VectorRef {
            data: dev_bias.as_slice(),
            len: n,
            stride: 1,
        }),
        alpha,
        beta,
    };
    plan.run(&stream, Workspace::None, args)
        .expect("int4 bias GEMM run");
    stream.synchronize().expect("stream sync");

    let mut host_d_bytes = vec![0u8; mu * n_bytes];
    {
        let mut tmp_t: Vec<T> = vec![T::default(); mu * n_bytes];
        dev_d.copy_to_host(&mut tmp_t).expect("download D");
        // SAFETY: S4 and U4 are `#[repr(transparent)]` wrappers around u8.
        let src: &[u8] = unsafe {
            core::slice::from_raw_parts(tmp_t.as_ptr() as *const u8, tmp_t.len())
        };
        host_d_bytes.copy_from_slice(src);
    }

    // Compare nibble-by-nibble; ±1 tolerance per nibble (f32 path
    // divergence on gelu / silu can produce off-by-one differences).
    let unpack = if is_signed { unpack_s4 } else { unpack_u4 };
    let mut mismatches = 0usize;
    let mut first_mismatch: Option<(usize, usize, i32, i32)> = None;
    for i in 0..mu {
        for j_pair in 0..n_bytes {
            let got = host_d_bytes[i * n_bytes + j_pair];
            let exp = expected_bytes[i * n_bytes + j_pair];
            if got == exp {
                continue;
            }
            let [g_lo, g_hi] = unpack(got);
            let [e_lo, e_hi] = unpack(exp);
            if (g_lo - e_lo).abs() > 1 {
                mismatches += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some((i, 2 * j_pair, g_lo, e_lo));
                }
            }
            if (g_hi - e_hi).abs() > 1 {
                mismatches += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some((i, 2 * j_pair + 1, g_hi, e_hi));
                }
            }
        }
    }
    if mismatches > 0 {
        let (i, j, got, expected) = first_mismatch.unwrap();
        panic!(
            "{mismatches} element mismatches > 1 across {} cells \
             (layout={:?} epilogue={:?} signed={is_signed}, \
              M={m} N={n} K={k}, alpha={alpha}); \
             first @ (i={i}, j={j}): got={got} expected={expected}",
            mu * nu,
            layout,
            epilogue,
        );
    }
}

// ============================================================================
// Test inputs — s4 in [-2, +2], u4 in [0, 4]. Bias values small.
// ============================================================================

fn mk_a_s4(i: usize, kk: usize) -> i32 {
    (((i as i32 * 7 + kk as i32 * 3) % 5) - 2).clamp(-7, 7)
}
fn mk_b_s4(kk: usize, j: usize) -> i32 {
    (((j as i32 * 11 + kk as i32 * 5) % 5) - 2).clamp(-7, 7)
}

fn mk_a_u4(i: usize, kk: usize) -> i32 {
    ((i as i32 * 7 + kk as i32 * 3) % 5).clamp(0, 15)
}
fn mk_b_u4(kk: usize, j: usize) -> i32 {
    ((j as i32 * 11 + kk as i32 * 5) % 5).clamp(0, 15)
}

fn mk_bias_f32(n: usize) -> Vec<f32> {
    // Small constants in [-2, +2] so the post-bias value stays well
    // within int4 sat-cast range.
    (0..n).map(|j| ((j as f32 % 5.0) - 2.0) * 0.5).collect()
}
fn bias_to_f32_f32(b: f32) -> f32 { b }

fn mk_bias_i32(n: usize) -> Vec<i32> {
    // Small ints in [-2, +2].
    (0..n).map(|j| (j as i32 % 5) - 2).collect()
}
fn bias_to_f32_i32(b: i32) -> f32 { b as f32 }

// ============================================================================
// Shape — one tile-aligned 128 × 128 × 128 shape per SKU (32 tests).
// K_TILE = N_TILE = M_TILE = 64 → 2×2 grid with 2 K-tiles.
// ============================================================================

const M: i32 = 128;
const N: i32 = 128;
const K: i32 = 128;

// Output-magnitude target for the alpha scaling: keep `|alpha * acc|`
// small enough that bias-add doesn't always saturate. Values:
//   s4 element magnitudes <= 2 → max |acc| <= K * 4 → alpha = 5/(4K)
//                                pushes |alpha*acc| ≤ 5, plus |bias| ≤ 2.
//   u4 element magnitudes <= 4 → max |acc| <= K * 16 → alpha = 12/(16K)
//                                pushes |alpha*acc| ≤ 12, plus |bias| ≤ 2.
const S4_OUT_MAX: f32 = 5.0;
const U4_OUT_MAX: f32 = 12.0;

// ----------------------------------------------------------------------------
// S4 × RCR × {Bias, BiasRelu, BiasGelu, BiasSilu} × {f32, i32}
// ----------------------------------------------------------------------------

#[test] #[ignore]
fn s4_rcr_bias_f32() {
    run_int4_bias_smoke::<S4, f32>(M, N, K, LayoutSku::Rcr, EpilogueKind::Bias, true,
        mk_a_s4, mk_b_s4, mk_bias_f32, bias_to_f32_f32, 2, 2, S4_OUT_MAX);
}
#[test] #[ignore]
fn s4_rcr_bias_relu_f32() {
    run_int4_bias_smoke::<S4, f32>(M, N, K, LayoutSku::Rcr, EpilogueKind::BiasRelu, true,
        mk_a_s4, mk_b_s4, mk_bias_f32, bias_to_f32_f32, 2, 2, S4_OUT_MAX);
}
#[test] #[ignore]
fn s4_rcr_bias_gelu_f32() {
    run_int4_bias_smoke::<S4, f32>(M, N, K, LayoutSku::Rcr, EpilogueKind::BiasGelu, true,
        mk_a_s4, mk_b_s4, mk_bias_f32, bias_to_f32_f32, 2, 2, S4_OUT_MAX);
}
#[test] #[ignore]
fn s4_rcr_bias_silu_f32() {
    run_int4_bias_smoke::<S4, f32>(M, N, K, LayoutSku::Rcr, EpilogueKind::BiasSilu, true,
        mk_a_s4, mk_b_s4, mk_bias_f32, bias_to_f32_f32, 2, 2, S4_OUT_MAX);
}
#[test] #[ignore]
fn s4_rcr_bias_i32() {
    run_int4_bias_smoke::<S4, i32>(M, N, K, LayoutSku::Rcr, EpilogueKind::Bias, true,
        mk_a_s4, mk_b_s4, mk_bias_i32, bias_to_f32_i32, 2, 2, S4_OUT_MAX);
}
#[test] #[ignore]
fn s4_rcr_bias_relu_i32() {
    run_int4_bias_smoke::<S4, i32>(M, N, K, LayoutSku::Rcr, EpilogueKind::BiasRelu, true,
        mk_a_s4, mk_b_s4, mk_bias_i32, bias_to_f32_i32, 2, 2, S4_OUT_MAX);
}
#[test] #[ignore]
fn s4_rcr_bias_gelu_i32() {
    run_int4_bias_smoke::<S4, i32>(M, N, K, LayoutSku::Rcr, EpilogueKind::BiasGelu, true,
        mk_a_s4, mk_b_s4, mk_bias_i32, bias_to_f32_i32, 2, 2, S4_OUT_MAX);
}
#[test] #[ignore]
fn s4_rcr_bias_silu_i32() {
    run_int4_bias_smoke::<S4, i32>(M, N, K, LayoutSku::Rcr, EpilogueKind::BiasSilu, true,
        mk_a_s4, mk_b_s4, mk_bias_i32, bias_to_f32_i32, 2, 2, S4_OUT_MAX);
}

// ----------------------------------------------------------------------------
// U4 × RCR × {Bias, BiasRelu, BiasGelu, BiasSilu} × {f32, i32}
// ----------------------------------------------------------------------------

#[test] #[ignore]
fn u4_rcr_bias_f32() {
    run_int4_bias_smoke::<U4, f32>(M, N, K, LayoutSku::Rcr, EpilogueKind::Bias, false,
        mk_a_u4, mk_b_u4, mk_bias_f32, bias_to_f32_f32, 4, 4, U4_OUT_MAX);
}
#[test] #[ignore]
fn u4_rcr_bias_relu_f32() {
    run_int4_bias_smoke::<U4, f32>(M, N, K, LayoutSku::Rcr, EpilogueKind::BiasRelu, false,
        mk_a_u4, mk_b_u4, mk_bias_f32, bias_to_f32_f32, 4, 4, U4_OUT_MAX);
}
#[test] #[ignore]
fn u4_rcr_bias_gelu_f32() {
    run_int4_bias_smoke::<U4, f32>(M, N, K, LayoutSku::Rcr, EpilogueKind::BiasGelu, false,
        mk_a_u4, mk_b_u4, mk_bias_f32, bias_to_f32_f32, 4, 4, U4_OUT_MAX);
}
#[test] #[ignore]
fn u4_rcr_bias_silu_f32() {
    run_int4_bias_smoke::<U4, f32>(M, N, K, LayoutSku::Rcr, EpilogueKind::BiasSilu, false,
        mk_a_u4, mk_b_u4, mk_bias_f32, bias_to_f32_f32, 4, 4, U4_OUT_MAX);
}
#[test] #[ignore]
fn u4_rcr_bias_i32() {
    run_int4_bias_smoke::<U4, i32>(M, N, K, LayoutSku::Rcr, EpilogueKind::Bias, false,
        mk_a_u4, mk_b_u4, mk_bias_i32, bias_to_f32_i32, 4, 4, U4_OUT_MAX);
}
#[test] #[ignore]
fn u4_rcr_bias_relu_i32() {
    run_int4_bias_smoke::<U4, i32>(M, N, K, LayoutSku::Rcr, EpilogueKind::BiasRelu, false,
        mk_a_u4, mk_b_u4, mk_bias_i32, bias_to_f32_i32, 4, 4, U4_OUT_MAX);
}
#[test] #[ignore]
fn u4_rcr_bias_gelu_i32() {
    run_int4_bias_smoke::<U4, i32>(M, N, K, LayoutSku::Rcr, EpilogueKind::BiasGelu, false,
        mk_a_u4, mk_b_u4, mk_bias_i32, bias_to_f32_i32, 4, 4, U4_OUT_MAX);
}
#[test] #[ignore]
fn u4_rcr_bias_silu_i32() {
    run_int4_bias_smoke::<U4, i32>(M, N, K, LayoutSku::Rcr, EpilogueKind::BiasSilu, false,
        mk_a_u4, mk_b_u4, mk_bias_i32, bias_to_f32_i32, 4, 4, U4_OUT_MAX);
}

// ----------------------------------------------------------------------------
// S4 × RRR × {Bias, BiasRelu, BiasGelu, BiasSilu} × {f32, i32}
// ----------------------------------------------------------------------------

#[test] #[ignore]
fn s4_rrr_bias_f32() {
    run_int4_bias_smoke::<S4, f32>(M, N, K, LayoutSku::Rrr, EpilogueKind::Bias, true,
        mk_a_s4, mk_b_s4, mk_bias_f32, bias_to_f32_f32, 2, 2, S4_OUT_MAX);
}
#[test] #[ignore]
fn s4_rrr_bias_relu_f32() {
    run_int4_bias_smoke::<S4, f32>(M, N, K, LayoutSku::Rrr, EpilogueKind::BiasRelu, true,
        mk_a_s4, mk_b_s4, mk_bias_f32, bias_to_f32_f32, 2, 2, S4_OUT_MAX);
}
#[test] #[ignore]
fn s4_rrr_bias_gelu_f32() {
    run_int4_bias_smoke::<S4, f32>(M, N, K, LayoutSku::Rrr, EpilogueKind::BiasGelu, true,
        mk_a_s4, mk_b_s4, mk_bias_f32, bias_to_f32_f32, 2, 2, S4_OUT_MAX);
}
#[test] #[ignore]
fn s4_rrr_bias_silu_f32() {
    run_int4_bias_smoke::<S4, f32>(M, N, K, LayoutSku::Rrr, EpilogueKind::BiasSilu, true,
        mk_a_s4, mk_b_s4, mk_bias_f32, bias_to_f32_f32, 2, 2, S4_OUT_MAX);
}
#[test] #[ignore]
fn s4_rrr_bias_i32() {
    run_int4_bias_smoke::<S4, i32>(M, N, K, LayoutSku::Rrr, EpilogueKind::Bias, true,
        mk_a_s4, mk_b_s4, mk_bias_i32, bias_to_f32_i32, 2, 2, S4_OUT_MAX);
}
#[test] #[ignore]
fn s4_rrr_bias_relu_i32() {
    run_int4_bias_smoke::<S4, i32>(M, N, K, LayoutSku::Rrr, EpilogueKind::BiasRelu, true,
        mk_a_s4, mk_b_s4, mk_bias_i32, bias_to_f32_i32, 2, 2, S4_OUT_MAX);
}
#[test] #[ignore]
fn s4_rrr_bias_gelu_i32() {
    run_int4_bias_smoke::<S4, i32>(M, N, K, LayoutSku::Rrr, EpilogueKind::BiasGelu, true,
        mk_a_s4, mk_b_s4, mk_bias_i32, bias_to_f32_i32, 2, 2, S4_OUT_MAX);
}
#[test] #[ignore]
fn s4_rrr_bias_silu_i32() {
    run_int4_bias_smoke::<S4, i32>(M, N, K, LayoutSku::Rrr, EpilogueKind::BiasSilu, true,
        mk_a_s4, mk_b_s4, mk_bias_i32, bias_to_f32_i32, 2, 2, S4_OUT_MAX);
}

// ----------------------------------------------------------------------------
// U4 × RRR × {Bias, BiasRelu, BiasGelu, BiasSilu} × {f32, i32}
// ----------------------------------------------------------------------------

#[test] #[ignore]
fn u4_rrr_bias_f32() {
    run_int4_bias_smoke::<U4, f32>(M, N, K, LayoutSku::Rrr, EpilogueKind::Bias, false,
        mk_a_u4, mk_b_u4, mk_bias_f32, bias_to_f32_f32, 4, 4, U4_OUT_MAX);
}
#[test] #[ignore]
fn u4_rrr_bias_relu_f32() {
    run_int4_bias_smoke::<U4, f32>(M, N, K, LayoutSku::Rrr, EpilogueKind::BiasRelu, false,
        mk_a_u4, mk_b_u4, mk_bias_f32, bias_to_f32_f32, 4, 4, U4_OUT_MAX);
}
#[test] #[ignore]
fn u4_rrr_bias_gelu_f32() {
    run_int4_bias_smoke::<U4, f32>(M, N, K, LayoutSku::Rrr, EpilogueKind::BiasGelu, false,
        mk_a_u4, mk_b_u4, mk_bias_f32, bias_to_f32_f32, 4, 4, U4_OUT_MAX);
}
#[test] #[ignore]
fn u4_rrr_bias_silu_f32() {
    run_int4_bias_smoke::<U4, f32>(M, N, K, LayoutSku::Rrr, EpilogueKind::BiasSilu, false,
        mk_a_u4, mk_b_u4, mk_bias_f32, bias_to_f32_f32, 4, 4, U4_OUT_MAX);
}
#[test] #[ignore]
fn u4_rrr_bias_i32() {
    run_int4_bias_smoke::<U4, i32>(M, N, K, LayoutSku::Rrr, EpilogueKind::Bias, false,
        mk_a_u4, mk_b_u4, mk_bias_i32, bias_to_f32_i32, 4, 4, U4_OUT_MAX);
}
#[test] #[ignore]
fn u4_rrr_bias_relu_i32() {
    run_int4_bias_smoke::<U4, i32>(M, N, K, LayoutSku::Rrr, EpilogueKind::BiasRelu, false,
        mk_a_u4, mk_b_u4, mk_bias_i32, bias_to_f32_i32, 4, 4, U4_OUT_MAX);
}
#[test] #[ignore]
fn u4_rrr_bias_gelu_i32() {
    run_int4_bias_smoke::<U4, i32>(M, N, K, LayoutSku::Rrr, EpilogueKind::BiasGelu, false,
        mk_a_u4, mk_b_u4, mk_bias_i32, bias_to_f32_i32, 4, 4, U4_OUT_MAX);
}
#[test] #[ignore]
fn u4_rrr_bias_silu_i32() {
    run_int4_bias_smoke::<U4, i32>(M, N, K, LayoutSku::Rrr, EpilogueKind::BiasSilu, false,
        mk_a_u4, mk_b_u4, mk_bias_i32, bias_to_f32_i32, 4, 4, U4_OUT_MAX);
}
