//! Real-GPU smoke test for the bespoke S4 RCR Identity kernel in
//! `baracuda-kernels-sys` (the Phase 2 int4 trailblazer).
//!
//! Verifies the kernel matches a CPU reference for the int32-accumulate
//! → f32-compute → saturating-cast chain on packed-pair int4 storage
//! (two int4 per byte; low nibble = even index, high nibble = odd
//! index along the K axis for A/B and along the N axis for D).
//!
//! Identity epilogue with `beta = 0` is **bit-exact** against the
//! reference (no f32 activation chain, integer MMA has no warp-
//! reduction nondeterminism, satfinite-clamp matches host `clamp`).
//! Mismatch tolerance is therefore zero.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --no-default-features --features sm89 \
//!  --release --test s4_rcr_smoke -- --ignored`.

#![cfg(feature = "sm89")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    EpilogueKind, Int4GemmArgs, Int4GemmDescriptor, Int4GemmPlan, LayoutSku, MatrixMut, MatrixRef,
    PlanPreference, S4, Workspace,
};

// ============================================================================
// s4 packing / unpacking helpers — host-side mirror of `baracuda_dtype.cuh`.
// ============================================================================

/// Decode the low 4 bits of `nibble` as a signed 4-bit value in `[-8, +7]`.
#[inline]
fn s4_decode(nibble: u8) -> i32 {
    // Place the nibble in the high 4 bits of an i8, then arithmetic-
    // shift back down — matches the GPU's `((s8)(nibble << 4)) >> 4`.
    ((nibble & 0x0F) << 4) as i8 as i32 >> 4
}

/// Pack two s4 values `(lo, hi)` (each in `[-8, +7]`) into one byte.
#[inline]
fn s4_pack(lo: i32, hi: i32) -> u8 {
    ((lo as u8) & 0x0F) | (((hi as u8) & 0x0F) << 4)
}

/// Unpack a packed-pair byte into `[even_index_value, odd_index_value]`.
#[inline]
fn s4_unpack(byte: u8) -> [i32; 2] {
    [s4_decode(byte & 0x0F), s4_decode((byte >> 4) & 0x0F)]
}

/// Saturating round-to-nearest-even cast from f32 to s4. Matches the
/// GPU's `__float2int_rn` + `clamp(-8, +7)`. Returns the nibble in
/// `[0, 15]` (sign-extend yourself if needed).
#[inline]
fn sat_cast_s4(x: f32) -> u8 {
    let r = (x.round_ties_even() as i32).clamp(-8, 7);
    (r as u8) & 0x0F
}

// ============================================================================
// CPU reference — S4 RCR Identity.
// ============================================================================
//
// `A` row-major `[M, K]`, packed-pair bytes along K: byte at
// `(i, k/2)` holds the pair `(A[i, 2*(k/2)], A[i, 2*(k/2)+1])` =
// `(low_nibble, high_nibble)`.
//
// `B` col-major `[K, N]`, packed-pair bytes along K: byte at
// `(k/2, j)` (i.e. byte index `j * ldb + k/2` from the start of B's
// gmem) holds the pair `(B[2*(k/2), j], B[2*(k/2)+1, j])`.
//
// `D` row-major `[M, N]`, packed-pair bytes along N: byte at
// `(i, j/2)` holds the pair `(D[i, 2*(j/2)], D[i, 2*(j/2)+1])`.

#[allow(clippy::too_many_arguments)]
fn cpu_s4_gemm_rcr_identity(
    m: usize,
    n: usize,
    k: usize,
    a_bytes: &[u8],
    lda_bytes: usize, // row stride of A in bytes (>= K/2)
    b_bytes: &[u8],
    ldb_bytes: usize, // column stride of B in bytes (>= K/2)
    alpha: f32,
    expected_bytes: &mut [u8], // M * N / 2, row-major packed
    expected_i32: &mut [i32],  // M * N, pre-quant int32 accum (debugging aid)
    ldd_bytes: usize,
) {
    assert_eq!(k % 2, 0, "K must be even");
    assert_eq!(n % 2, 0, "N must be even");
    let k_bytes = k / 2;
    let n_bytes = n / 2;
    let _ = n_bytes;

    for i in 0..m {
        // First compute the int32 accumulator for every (i, j) in this row.
        let mut acc_row = vec![0i32; n];
        for kk_byte in 0..k_bytes {
            let a_byte = a_bytes[i * lda_bytes + kk_byte];
            let [a_lo, a_hi] = s4_unpack(a_byte);
            for j in 0..n {
                // B is col-major: byte at (kk_byte, j) lives at
                // index `j * ldb_bytes + kk_byte`.
                let b_byte = b_bytes[j * ldb_bytes + kk_byte];
                let [b_lo, b_hi] = s4_unpack(b_byte);
                // Pair contribution: A[i, 2*kk_byte] * B[2*kk_byte, j]
                //                  + A[i, 2*kk_byte+1] * B[2*kk_byte+1, j]
                acc_row[j] += a_lo * b_lo + a_hi * b_hi;
            }
        }
        // Apply alpha + sat-cast back to s4, pack pairs of adjacent
        // columns into one output byte.
        for j_pair in 0..(n / 2) {
            let j0 = 2 * j_pair;
            let j1 = j0 + 1;
            let v0 = alpha * acc_row[j0] as f32;
            let v1 = alpha * acc_row[j1] as f32;
            let q0 = sat_cast_s4(v0);
            let q1 = sat_cast_s4(v1);
            expected_bytes[i * ldd_bytes + j_pair] = s4_pack(s4_decode(q0), s4_decode(q1));
            expected_i32[i * n + j0] = acc_row[j0];
            expected_i32[i * n + j1] = acc_row[j1];
        }
    }
}

// ============================================================================
// Test harness
// ============================================================================

fn run_s4_rcr_identity(m: i32, n: i32, k: i32) {
    assert_eq!(k % 2, 0, "K must be even for int4 packed storage");
    assert_eq!(n % 2, 0, "N must be even for int4 packed output");

    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let mu = m as usize;
    let nu = n as usize;
    let ku = k as usize;
    let k_bytes = ku / 2;
    let n_bytes = nu / 2;

    // Build A/B as logical i32 vectors in [-2, +2] (well within the
    // s4 range [-8, +7]), then pack pairs along K into bytes. Patterns
    // are deterministic and span both signs.
    let mk_a = |i: usize, kk: usize| -> i32 {
        (((i as i32 * 7 + kk as i32 * 3) % 5) - 2).clamp(-7, 7)
    };
    let mk_b = |kk: usize, j: usize| -> i32 {
        (((j as i32 * 11 + kk as i32 * 5) % 5) - 2).clamp(-7, 7)
    };

    // Pack A row-major: byte at (i, kk_byte) = (A[i, 2*kk_byte], A[i, 2*kk_byte+1]).
    let mut host_a_bytes = vec![0u8; mu * k_bytes];
    for i in 0..mu {
        for kk_byte in 0..k_bytes {
            let lo = mk_a(i, 2 * kk_byte);
            let hi = mk_a(i, 2 * kk_byte + 1);
            host_a_bytes[i * k_bytes + kk_byte] = s4_pack(lo, hi);
        }
    }
    // Pack B col-major: byte at (kk_byte, j) = (B[2*kk_byte, j], B[2*kk_byte+1, j]).
    // Storage layout: for column j, bytes are at offset (j * ldb_bytes + kk_byte).
    let mut host_b_bytes = vec![0u8; nu * k_bytes];
    for j in 0..nu {
        for kk_byte in 0..k_bytes {
            let lo = mk_b(2 * kk_byte, j);
            let hi = mk_b(2 * kk_byte + 1, j);
            host_b_bytes[j * k_bytes + kk_byte] = s4_pack(lo, hi);
        }
    }

    // Choose alpha so the output spans most of `[-7, +7]` without
    // hard-saturating most cells. With element magnitudes capped at 2,
    // `|acc|` ≤ K * 4. Pick alpha = 5 / (K * 4) → max |out| ≈ 5.
    let alpha: f32 = 5.0 / (4.0 * k as f32);
    let beta: f32 = 0.0;

    let mut expected_bytes = vec![0u8; mu * n_bytes];
    let mut expected_i32 = vec![0i32; mu * nu];
    cpu_s4_gemm_rcr_identity(
        mu, nu, ku,
        &host_a_bytes, k_bytes,
        &host_b_bytes, k_bytes,
        alpha,
        &mut expected_bytes,
        &mut expected_i32,
        n_bytes,
    );

    let dev_a_bytes = DeviceBuffer::from_slice(&ctx, &host_a_bytes).expect("upload A");
    let dev_b_bytes = DeviceBuffer::from_slice(&ctx, &host_b_bytes).expect("upload B");
    let dev_a = dev_a_bytes.view_as::<S4>();
    let dev_b = dev_b_bytes.view_as::<S4>();
    let mut dev_d: DeviceBuffer<S4> =
        DeviceBuffer::zeros(&ctx, mu * n_bytes).expect("alloc D");

    let desc = Int4GemmDescriptor {
        m, n, k,
        layout: LayoutSku::Rcr,
        epilogue: EpilogueKind::Identity,
    };
    let plan = Int4GemmPlan::<S4>::select(&stream, &desc, PlanPreference::default())
        .expect("select S4 RCR plan");

    let args = Int4GemmArgs::<S4> {
        // A: row-major [M, K]; ld in BYTES = K/2.
        a: MatrixRef { data: dev_a, rows: m, cols: k, ld: k_bytes as i64 },
        // B: col-major [K, N] (RCR); ld in BYTES = K/2.
        b: MatrixRef { data: dev_b, rows: k, cols: n, ld: k_bytes as i64 },
        c: None,
        // D: row-major [M, N]; ld in BYTES = N/2.
        d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n_bytes as i64 },
        bias: None,
        alpha,
        beta,
    };
    plan.run(&stream, Workspace::None, args)
        .expect("S4 RCR GEMM run");
    stream.synchronize().expect("stream sync");

    let mut host_d_bytes = vec![S4(0); mu * n_bytes];
    dev_d.copy_to_host(&mut host_d_bytes).expect("download D");

    let mut mismatches = 0usize;
    let mut first_mismatch: Option<(usize, usize, i32, i32, i32)> = None;
    for i in 0..mu {
        for j_pair in 0..n_bytes {
            let got_byte = host_d_bytes[i * n_bytes + j_pair].0;
            let exp_byte = expected_bytes[i * n_bytes + j_pair];
            if got_byte == exp_byte {
                continue;
            }
            let [g_lo, g_hi] = s4_unpack(got_byte);
            let [e_lo, e_hi] = s4_unpack(exp_byte);
            if g_lo != e_lo {
                mismatches += 1;
                if first_mismatch.is_none() {
                    let raw = expected_i32[i * nu + 2 * j_pair];
                    first_mismatch = Some((i, 2 * j_pair, g_lo, e_lo, raw));
                }
            }
            if g_hi != e_hi {
                mismatches += 1;
                if first_mismatch.is_none() {
                    let raw = expected_i32[i * nu + 2 * j_pair + 1];
                    first_mismatch = Some((i, 2 * j_pair + 1, g_hi, e_hi, raw));
                }
            }
        }
    }
    if mismatches > 0 {
        let (i, j, got, expected, raw) = first_mismatch.unwrap();
        panic!(
            "{mismatches} element mismatches across {} cells \
             (M={m} N={n} K={k}, alpha={alpha}); \
             first @ (i={i}, j={j}): got s4={got} expected s4={expected}; \
             pre-quant int32 acc = {raw}, alpha*acc = {}",
            mu * nu,
            alpha * raw as f32,
        );
    }
}

// ============================================================================
// Tests — Identity at four shapes that exercise tile-aligned and ragged
// cases (mirrors the FP8 / int8 trailblazer coverage).
// ============================================================================

#[test] #[ignore]
fn s4_rcr_identity_64_64_64() {
    // Smallest tile-aligned shape: 1 block, 1 K-tile. Pure smoke.
    run_s4_rcr_identity(64, 64, 64);
}

#[test] #[ignore]
fn s4_rcr_identity_128_128_128() {
    // 2x2 grid, 2 K-tiles. Exercises smem reuse across K iters.
    run_s4_rcr_identity(128, 128, 128);
}

#[test] #[ignore]
fn s4_rcr_identity_256_128_64() {
    // 4x2 grid, 1 K-tile. Larger M, asymmetric tile count.
    run_s4_rcr_identity(256, 128, 64);
}

#[test] #[ignore]
fn s4_rcr_identity_100_70_64() {
    // Ragged in M (100 not a multiple of 64) and N (70 not a multiple
    // of 64; still even so packing is well-defined). K = 64 = 1 K-tile.
    // Exercises edge-of-tile padding in both the gmem-load and the
    // epilogue-store paths.
    run_s4_rcr_identity(100, 70, 64);
}
