//! Real-GPU smoke test for the bespoke int4 RRR Identity kernels in
//! `baracuda-kernels-sys` — covers both `S4` and `U4` element types.
//!
//! Proves the novel nibble-gather B-load: B in gmem is row-major
//! `[K, N]` pair-packed along N, but the kernel's MMA fragment wants
//! B in smem pair-packed along K within each output column. The
//! kernel gathers two nibbles from two K-row bytes per output column
//! and re-packs them into one K-pair smem byte.
//!
//! Identity epilogue with `beta = 0` is bit-exact against the int32
//! CPU reference; mismatch tolerance is zero.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --no-default-features --features sm89 \
//!  --release --test int4_rrr_smoke -- --ignored`.

#![cfg(feature = "sm89")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    EpilogueKind, Int4GemmArgs, Int4GemmDescriptor, Int4GemmPlan, IntElement, LayoutSku,
    MatrixMut, MatrixRef, PlanPreference, S4, U4, Workspace,
};

// ============================================================================
// s4 / u4 nibble helpers — host-side mirror of `baracuda_dtype.cuh`.
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
// CPU reference — int4 RRR Identity.
//
// `A` row-major `[M, K]`, pair-packed along K (byte at `(i, k/2)` holds
// `[A[i, 2*(k/2)], A[i, 2*(k/2)+1]]`).
// `B` row-major `[K, N]`, pair-packed along N (byte at `(k, j/2)` holds
// `[B[k, 2*(j/2)], B[k, 2*(j/2)+1]]`).
// `D` row-major `[M, N]`, pair-packed along N.
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn cpu_int4_gemm_rrr_identity(
    m: usize,
    n: usize,
    k: usize,
    a_bytes: &[u8],
    lda_bytes: usize,
    b_bytes: &[u8],
    ldb_bytes: usize,
    alpha: f32,
    expected_bytes: &mut [u8],
    expected_i32: &mut [i32],
    ldd_bytes: usize,
    is_signed: bool,
) {
    assert_eq!(k % 2, 0);
    assert_eq!(n % 2, 0);
    let k_bytes = k / 2;
    let n_bytes = n / 2;
    let unpack = if is_signed { unpack_s4 } else { unpack_u4 };
    let sat_cast = if is_signed { sat_cast_s4 } else { sat_cast_u4 };

    for i in 0..m {
        let mut acc_row = vec![0i32; n];
        for kk_byte in 0..k_bytes {
            // A: byte at (i, kk_byte) is pair [A[i, 2*kk_byte], A[i, 2*kk_byte+1]].
            let [a_lo, a_hi] = unpack(a_bytes[i * lda_bytes + kk_byte]);
            // Accumulate K_lo = 2*kk_byte contribution (a_lo * B[K_lo, :])
            // and K_hi = 2*kk_byte + 1 contribution (a_hi * B[K_hi, :]).
            for j_pair in 0..n_bytes {
                let j0 = 2 * j_pair;
                let j1 = j0 + 1;
                // B row K_lo, pair (j0, j1):
                let [b_lo_lo, b_lo_hi] =
                    unpack(b_bytes[(2 * kk_byte) * ldb_bytes + j_pair]);
                // B row K_hi, pair (j0, j1):
                let [b_hi_lo, b_hi_hi] =
                    unpack(b_bytes[(2 * kk_byte + 1) * ldb_bytes + j_pair]);
                acc_row[j0] += a_lo * b_lo_lo + a_hi * b_hi_lo;
                acc_row[j1] += a_lo * b_lo_hi + a_hi * b_hi_hi;
            }
        }
        for j_pair in 0..n_bytes {
            let j0 = 2 * j_pair;
            let j1 = j0 + 1;
            let q0 = sat_cast(alpha * acc_row[j0] as f32);
            let q1 = sat_cast(alpha * acc_row[j1] as f32);
            expected_bytes[i * ldd_bytes + j_pair] =
                pack_pair(q0 as i8 as i32, q1 as i8 as i32);
            expected_i32[i * n + j0] = acc_row[j0];
            expected_i32[i * n + j1] = acc_row[j1];
        }
    }
}

// ============================================================================
// Test harness — generic over T ∈ {S4, U4}
// ============================================================================

fn run_int4_rrr_identity<T: IntElement + Default>(
    m: i32,
    n: i32,
    k: i32,
    is_signed: bool,
    mk_a: fn(usize, usize) -> i32,
    mk_b: fn(usize, usize) -> i32,
    abs_max_a: i32,
    abs_max_b: i32,
    out_max_abs: f32,
) where
    T: 'static,
{
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let mu = m as usize;
    let nu = n as usize;
    let ku = k as usize;
    let k_bytes = ku / 2;
    let n_bytes = nu / 2;

    // A: row-major [M, K], pair-packed along K.
    let mut host_a_bytes = vec![0u8; mu * k_bytes];
    for i in 0..mu {
        for kk_byte in 0..k_bytes {
            host_a_bytes[i * k_bytes + kk_byte] =
                pack_pair(mk_a(i, 2 * kk_byte), mk_a(i, 2 * kk_byte + 1));
        }
    }
    // B: row-major [K, N], pair-packed along N.
    let mut host_b_bytes = vec![0u8; ku * n_bytes];
    for kk in 0..ku {
        for j_pair in 0..n_bytes {
            host_b_bytes[kk * n_bytes + j_pair] =
                pack_pair(mk_b(kk, 2 * j_pair), mk_b(kk, 2 * j_pair + 1));
        }
    }

    // Scale alpha so output stays in range after sat-cast.
    let alpha: f32 = out_max_abs / ((abs_max_a * abs_max_b) as f32 * k as f32);
    let beta: f32 = 0.0;

    let mut expected_bytes = vec![0u8; mu * n_bytes];
    let mut expected_i32 = vec![0i32; mu * nu];
    cpu_int4_gemm_rrr_identity(
        mu, nu, ku,
        &host_a_bytes, k_bytes,
        &host_b_bytes, n_bytes,
        alpha,
        &mut expected_bytes,
        &mut expected_i32,
        n_bytes,
        is_signed,
    );

    let dev_a_bytes = DeviceBuffer::from_slice(&ctx, &host_a_bytes).expect("upload A");
    let dev_b_bytes = DeviceBuffer::from_slice(&ctx, &host_b_bytes).expect("upload B");
    let dev_a = dev_a_bytes.view_as::<T>();
    let dev_b = dev_b_bytes.view_as::<T>();
    let mut dev_d: DeviceBuffer<T> =
        DeviceBuffer::zeros(&ctx, mu * n_bytes).expect("alloc D");

    let desc = Int4GemmDescriptor {
        m, n, k,
        layout: LayoutSku::Rrr,
        epilogue: EpilogueKind::Identity,
    };
    let plan = Int4GemmPlan::<T>::select(&stream, &desc, PlanPreference::default())
        .expect("select int4 RRR plan");

    let args = Int4GemmArgs::<T> {
        a: MatrixRef { data: dev_a, rows: m, cols: k, ld: k_bytes as i64 },
        // RRR: B row-major [K, N], row stride = N/2 bytes.
        b: MatrixRef { data: dev_b, rows: k, cols: n, ld: n_bytes as i64 },
        c: None,
        d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n_bytes as i64 },
        bias: None,
        alpha,
        beta,
    };
    plan.run(&stream, Workspace::None, args)
        .expect("int4 RRR GEMM run");
    stream.synchronize().expect("stream sync");

    // Download as bytes (T is 1 byte regardless of S4 vs U4).
    let mut host_d_bytes = vec![0u8; mu * n_bytes];
    {
        // The buffer-as-T view is fine but we need byte readback for
        // comparison. Re-view as u8 by copying through a Vec<T> then
        // taking bytes. (`view_as` only goes the other way today; we
        // reinterpret here on the host.)
        let mut tmp_t: Vec<T> = vec![T::default(); mu * n_bytes];
        dev_d.copy_to_host(&mut tmp_t).expect("download D");
        // SAFETY: T is `#[repr(transparent)]` around `u8` for both S4
        // and U4 (single packed-pair byte per element).
        let src: &[u8] = unsafe {
            core::slice::from_raw_parts(tmp_t.as_ptr() as *const u8, tmp_t.len())
        };
        host_d_bytes.copy_from_slice(src);
    }

    let mut mismatches = 0usize;
    let mut first_mismatch: Option<(usize, usize, i32, i32, i32)> = None;
    let unpack = if is_signed { unpack_s4 } else { unpack_u4 };
    for i in 0..mu {
        for j_pair in 0..n_bytes {
            let got_byte = host_d_bytes[i * n_bytes + j_pair];
            let exp_byte = expected_bytes[i * n_bytes + j_pair];
            if got_byte == exp_byte {
                continue;
            }
            let [g_lo, g_hi] = unpack(got_byte);
            let [e_lo, e_hi] = unpack(exp_byte);
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
             first @ (i={i}, j={j}): got={got} expected={expected}; \
             pre-quant int32 acc = {raw}, alpha*acc = {}",
            mu * nu,
            alpha * raw as f32,
        );
    }
}

fn run_s4(m: i32, n: i32, k: i32) {
    fn mk_a(i: usize, kk: usize) -> i32 {
        (((i as i32 * 7 + kk as i32 * 3) % 5) - 2).clamp(-7, 7)
    }
    fn mk_b(kk: usize, j: usize) -> i32 {
        (((j as i32 * 11 + kk as i32 * 5) % 5) - 2).clamp(-7, 7)
    }
    run_int4_rrr_identity::<S4>(
        m, n, k,
        /*is_signed=*/true,
        mk_a, mk_b,
        /*abs_max_a=*/2, /*abs_max_b=*/2,
        /*out_max_abs=*/5.0,
    );
}

fn run_u4(m: i32, n: i32, k: i32) {
    fn mk_a(i: usize, kk: usize) -> i32 {
        ((i as i32 * 7 + kk as i32 * 3) % 5).clamp(0, 15)
    }
    fn mk_b(kk: usize, j: usize) -> i32 {
        ((j as i32 * 11 + kk as i32 * 5) % 5).clamp(0, 15)
    }
    run_int4_rrr_identity::<U4>(
        m, n, k,
        /*is_signed=*/false,
        mk_a, mk_b,
        /*abs_max_a=*/4, /*abs_max_b=*/4,
        /*out_max_abs=*/12.0,
    );
}

// ============================================================================
// Tests — Identity at four shapes that exercise tile-alignment edges.
// ============================================================================

#[test] #[ignore]
fn s4_rrr_identity_64_64_64() {
    run_s4(64, 64, 64);
}
#[test] #[ignore]
fn s4_rrr_identity_128_128_128() {
    run_s4(128, 128, 128);
}
#[test] #[ignore]
fn s4_rrr_identity_256_128_64() {
    run_s4(256, 128, 64);
}
#[test] #[ignore]
fn s4_rrr_identity_100_70_64() {
    run_s4(100, 70, 64);
}

#[test] #[ignore]
fn u4_rrr_identity_64_64_64() {
    run_u4(64, 64, 64);
}
#[test] #[ignore]
fn u4_rrr_identity_128_128_128() {
    run_u4(128, 128, 128);
}
#[test] #[ignore]
fn u4_rrr_identity_256_128_64() {
    run_u4(256, 128, 64);
}
#[test] #[ignore]
fn u4_rrr_identity_100_70_64() {
    run_u4(100, 70, 64);
}
