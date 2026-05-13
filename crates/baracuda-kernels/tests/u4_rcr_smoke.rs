//! Real-GPU smoke test for the bespoke U4 RCR Identity kernel in
//! `baracuda-kernels-sys`.
//!
//! Identical shape to the S4 trailblazer smoke; differs only in the
//! MMA operand encoding (`.u4.u4`) and the saturating cast (clamp to
//! `[0, 15]` instead of `[-8, +7]`). Bit-exact against the int32 CPU
//! reference.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --no-default-features --features sm89 \
//!  --release --test u4_rcr_smoke -- --ignored`.

#![cfg(feature = "sm89")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    EpilogueKind, Int4GemmArgs, Int4GemmDescriptor, Int4GemmPlan, LayoutSku, MatrixMut, MatrixRef,
    PlanPreference, U4, Workspace,
};

// ============================================================================
// u4 packing / unpacking helpers — host-side mirror of `baracuda_dtype.cuh`.
// ============================================================================

#[inline]
fn u4_decode(nibble: u8) -> i32 {
    (nibble & 0x0F) as i32
}

#[inline]
fn u4_pack(lo: i32, hi: i32) -> u8 {
    (lo as u8 & 0x0F) | ((hi as u8 & 0x0F) << 4)
}

#[inline]
fn u4_unpack(byte: u8) -> [i32; 2] {
    [u4_decode(byte & 0x0F), u4_decode((byte >> 4) & 0x0F)]
}

#[inline]
fn sat_cast_u4(x: f32) -> u8 {
    let r = (x.round_ties_even() as i32).clamp(0, 15);
    (r as u8) & 0x0F
}

// ============================================================================
// CPU reference — U4 RCR Identity.
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn cpu_u4_gemm_rcr_identity(
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
) {
    assert_eq!(k % 2, 0);
    assert_eq!(n % 2, 0);
    let k_bytes = k / 2;
    for i in 0..m {
        let mut acc_row = vec![0i32; n];
        for kk_byte in 0..k_bytes {
            let [a_lo, a_hi] = u4_unpack(a_bytes[i * lda_bytes + kk_byte]);
            for j in 0..n {
                let [b_lo, b_hi] = u4_unpack(b_bytes[j * ldb_bytes + kk_byte]);
                acc_row[j] += a_lo * b_lo + a_hi * b_hi;
            }
        }
        for j_pair in 0..(n / 2) {
            let j0 = 2 * j_pair;
            let j1 = j0 + 1;
            let q0 = sat_cast_u4(alpha * acc_row[j0] as f32);
            let q1 = sat_cast_u4(alpha * acc_row[j1] as f32);
            expected_bytes[i * ldd_bytes + j_pair] = u4_pack(q0 as i32, q1 as i32);
            expected_i32[i * n + j0] = acc_row[j0];
            expected_i32[i * n + j1] = acc_row[j1];
        }
    }
}

// ============================================================================
// Test harness
// ============================================================================

fn run_u4_rcr_identity(m: i32, n: i32, k: i32) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let mu = m as usize;
    let nu = n as usize;
    let ku = k as usize;
    let k_bytes = ku / 2;
    let n_bytes = nu / 2;

    // Logical u4 values in [0, 4] — well within u4 range [0, 15] and
    // small enough that |acc| stays modest. alpha is tuned so the
    // output spans most of [0, 15] without hard-saturating most cells.
    let mk_a = |i: usize, kk: usize| -> i32 {
        (((i as i32 * 7 + kk as i32 * 3) % 5)).clamp(0, 15)
    };
    let mk_b = |kk: usize, j: usize| -> i32 {
        (((j as i32 * 11 + kk as i32 * 5) % 5)).clamp(0, 15)
    };

    let mut host_a_bytes = vec![0u8; mu * k_bytes];
    for i in 0..mu {
        for kk_byte in 0..k_bytes {
            host_a_bytes[i * k_bytes + kk_byte] =
                u4_pack(mk_a(i, 2 * kk_byte), mk_a(i, 2 * kk_byte + 1));
        }
    }
    let mut host_b_bytes = vec![0u8; nu * k_bytes];
    for j in 0..nu {
        for kk_byte in 0..k_bytes {
            host_b_bytes[j * k_bytes + kk_byte] =
                u4_pack(mk_b(2 * kk_byte, j), mk_b(2 * kk_byte + 1, j));
        }
    }

    // |a| ≤ 4, |b| ≤ 4 → max acc ≤ K * 16. Pick alpha so max |out| ≈
    // 12 (close to 15 but with margin).
    let alpha: f32 = 12.0 / (16.0 * k as f32);
    let beta: f32 = 0.0;

    let mut expected_bytes = vec![0u8; mu * n_bytes];
    let mut expected_i32 = vec![0i32; mu * nu];
    cpu_u4_gemm_rcr_identity(
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
    let dev_a = dev_a_bytes.view_as::<U4>();
    let dev_b = dev_b_bytes.view_as::<U4>();
    let mut dev_d: DeviceBuffer<U4> =
        DeviceBuffer::zeros(&ctx, mu * n_bytes).expect("alloc D");

    let desc = Int4GemmDescriptor {
        m, n, k,
        layout: LayoutSku::Rcr,
        epilogue: EpilogueKind::Identity,
    };
    let plan = Int4GemmPlan::<U4>::select(&stream, &desc, PlanPreference::default())
        .expect("select U4 RCR plan");

    let args = Int4GemmArgs::<U4> {
        a: MatrixRef { data: dev_a, rows: m, cols: k, ld: k_bytes as i64 },
        b: MatrixRef { data: dev_b, rows: k, cols: n, ld: k_bytes as i64 },
        c: None,
        d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n_bytes as i64 },
        bias: None,
        alpha,
        beta,
    };
    plan.run(&stream, Workspace::None, args)
        .expect("U4 RCR GEMM run");
    stream.synchronize().expect("stream sync");

    let mut host_d_bytes = vec![U4(0); mu * n_bytes];
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
            let [g_lo, g_hi] = u4_unpack(got_byte);
            let [e_lo, e_hi] = u4_unpack(exp_byte);
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
             first @ (i={i}, j={j}): got u4={got} expected u4={expected}; \
             pre-quant int32 acc = {raw}, alpha*acc = {}",
            mu * nu,
            alpha * raw as f32,
        );
    }
}

#[test] #[ignore]
fn u4_rcr_identity_64_64_64() {
    run_u4_rcr_identity(64, 64, 64);
}

#[test] #[ignore]
fn u4_rcr_identity_128_128_128() {
    run_u4_rcr_identity(128, 128, 128);
}

#[test] #[ignore]
fn u4_rcr_identity_256_128_64() {
    run_u4_rcr_identity(256, 128, 64);
}

#[test] #[ignore]
fn u4_rcr_identity_100_70_64() {
    run_u4_rcr_identity(100, 70, 64);
}
