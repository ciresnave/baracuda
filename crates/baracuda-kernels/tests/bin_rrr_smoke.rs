//! Real-GPU smoke test for the bespoke binary (B1) RRR GEMM kernel in
//! `baracuda-kernels-sys`.
//!
//! Distinct from `bin_smoke.rs` (RCR) in B's gmem layout: B is row-
//! major `[K, N]` and **bit-packed along N** (the byte at gmem
//! `k * ldb + j/8` holds 8 N-cols of K-row `k`, bit `j%8`). The kernel
//! re-packs into K-bit-packed smem via a bit-gather load (8 gmem byte
//! reads per smem byte) before invoking the same
//! `mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc` path.
//!
//! Integer popcount has no warp-reduction nondeterminism, so the
//! tolerance is **zero** — any mismatch is a kernel bug.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --no-default-features --features sm89 \
//!  --release --test bin_rrr_smoke -- --ignored`.

#![cfg(feature = "sm89")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    Bin, BinGemmArgs, BinGemmDescriptor, BinGemmPlan, LayoutSku, MatrixMut, MatrixRef,
    PlanPreference, Workspace,
};

// ============================================================================
// CPU reference — bin RRR.
//
// `A` row-major `[M, K bits]`, bit-packed along K. Byte at `(i, k_byte)`
// holds 8 K-bits, LSB-first.
//
// `B` row-major `[K bits, N]`, bit-packed along N. Byte at `(k, j_byte)`
// holds 8 N-bits of row k, LSB-first: `B[k, 8*j_byte + b]` is bit `b`
// of byte at `k * ldb + j_byte`.
//
// `D[i, j] = sum over k_byte of popcount(A[i, k_byte] XOR
//                                       column_byte_at_K_row(k_byte, j))`
// where `column_byte_at_K_row(k_byte, j)` is the 8-bit value formed
// by gathering bit `j%8` from each of 8 K-row bytes of B at K rows
// `8*k_byte..8*k_byte+7`. That's the matrix multiplication on packed
// b1 expressed in terms of XOR-popcount over K-rows.
//
// Equivalently (and simpler to reference): per (i, j) iterate over
// individual K bits and count `A[i, k] != B[k, j]`.
// ============================================================================

fn cpu_bin_gemm_rrr(
    m: usize,
    n: usize,
    k: usize,
    a_bytes: &[u8],
    lda_bytes: usize, // row stride of A in bytes (>= K/8)
    b_bytes: &[u8],
    ldb_bytes: usize, // row stride of B in bytes (>= N/8)
    d: &mut [i32],
    ldd_elements: usize,
) {
    assert_eq!(k % 8, 0, "K must be a multiple of 8");
    assert_eq!(n % 8, 0, "N must be a multiple of 8");
    let k_bytes = k / 8;
    for i in 0..m {
        for j in 0..n {
            let j_byte = j / 8;
            let j_bit = j & 7;
            let mut acc = 0u32;
            for kk in 0..k_bytes {
                let a = a_bytes[i * lda_bytes + kk];
                // Gather 8 bits from 8 K-rows of B for output column j.
                let mut b_col_byte = 0u8;
                for bb in 0..8 {
                    let k_elem = 8 * kk + bb;
                    let b = b_bytes[k_elem * ldb_bytes + j_byte];
                    b_col_byte |= ((b >> j_bit) & 1) << bb;
                }
                acc += (a ^ b_col_byte).count_ones();
            }
            d[i * ldd_elements + j] = acc as i32;
        }
    }
}

// ============================================================================
// Test harness
// ============================================================================

fn run_bin_rrr(m: i32, n: i32, k: i32) {
    assert_eq!(k % 8, 0, "K must be a multiple of 8 for packed-bit storage");
    assert_eq!(n % 8, 0, "N must be a multiple of 8 for bin RRR (B is N-bit-packed)");

    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let mu = m as usize;
    let nu = n as usize;
    let ku = k as usize;
    let k_bytes = ku / 8;
    let n_bytes = nu / 8;

    let host_a_bytes: Vec<u8> = (0..(mu * k_bytes))
        .map(|i| ((i as u32 * 73 + 11).wrapping_mul(2654435761) >> 24) as u8)
        .collect();
    // B is row-major [K, N] with row stride = N/8 bytes (bit-packed along N).
    let host_b_bytes: Vec<u8> = (0..(ku * n_bytes))
        .map(|i| ((i as u32 * 41 + 7).wrapping_mul(2246822519) >> 24) as u8)
        .collect();

    let mut expected = vec![0i32; mu * nu];
    cpu_bin_gemm_rrr(
        mu, nu, ku,
        &host_a_bytes, k_bytes,
        &host_b_bytes, n_bytes,
        &mut expected, nu,
    );

    let dev_a_bytes = DeviceBuffer::from_slice(&ctx, &host_a_bytes).expect("upload A");
    let dev_b_bytes = DeviceBuffer::from_slice(&ctx, &host_b_bytes).expect("upload B");
    let dev_a = dev_a_bytes.view_as::<Bin>();
    let dev_b = dev_b_bytes.view_as::<Bin>();
    let mut dev_d: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, mu * nu).expect("alloc D");

    let desc = BinGemmDescriptor {
        m, n, k,
        layout: LayoutSku::Rrr,
    };
    let plan = BinGemmPlan::select(&stream, &desc, PlanPreference::default())
        .expect("select bin RRR plan");

    let args = BinGemmArgs {
        // A: row-major [M, K]; ld in BYTES = K/8.
        a: MatrixRef { data: dev_a, rows: m, cols: k, ld: k_bytes as i64 },
        // B: row-major [K, N] (RRR); ld in BYTES = N/8.
        b: MatrixRef { data: dev_b, rows: k, cols: n, ld: n_bytes as i64 },
        // D: row-major [M, N] int32; ld in ELEMENTS = N.
        d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n as i64 },
    };
    plan.run(&stream, Workspace::None, args)
        .expect("bin RRR GEMM run");
    stream.synchronize().expect("stream sync");

    let mut host_d = vec![0i32; mu * nu];
    dev_d.copy_to_host(&mut host_d).expect("download D");

    let mut mismatches = 0usize;
    let mut first_mismatch: Option<(usize, usize, i32, i32)> = None;
    for i in 0..mu {
        for j in 0..nu {
            let got = host_d[i * nu + j];
            let exp = expected[i * nu + j];
            if got != exp {
                mismatches += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some((i, j, got, exp));
                }
            }
        }
    }
    if mismatches > 0 {
        let (i, j, got, exp) = first_mismatch.unwrap();
        panic!(
            "{mismatches} mismatches across {} cells (M={m} N={n} K={k}); \
             first @ (i={i}, j={j}): got={got} expected={exp}",
            mu * nu,
        );
    }
}

// ============================================================================
// Tests — shapes that exercise tile-alignment edges. K must be a
// multiple of 8 (bit-packing along K); N must be a multiple of 8 for
// the RRR variant (B is bit-packed along N in gmem).
// ============================================================================

#[test] #[ignore]
fn bin_rrr_64_64_256() {
    run_bin_rrr(64, 64, 256);
}

#[test] #[ignore]
fn bin_rrr_128_128_512() {
    run_bin_rrr(128, 128, 512);
}

#[test] #[ignore]
fn bin_rrr_256_128_256() {
    run_bin_rrr(256, 128, 256);
}

#[test] #[ignore]
fn bin_rrr_104_72_128() {
    // Ragged in M and N; K=128 is below K_TILE=256 (exercises K-tile
    // padding). N=72 is divisible by 8 (RRR requires this).
    run_bin_rrr(104, 72, 128);
}

#[test] #[ignore]
fn bin_rrr_72_72_264() {
    // Tile-edge ragged: M, N each just past tile boundary; K=264 is
    // one K-tile + 8 bits (exercises K-tile padding when the second
    // tile is mostly zero).
    run_bin_rrr(72, 72, 264);
}
