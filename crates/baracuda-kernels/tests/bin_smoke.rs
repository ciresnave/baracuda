//! Real-GPU smoke test for the bespoke binary (B1) RCR GEMM kernel in
//! `baracuda-kernels-sys`.
//!
//! Proves the `mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc`
//! path on RTX 4070 (sm_89) — Hamming-distance per (i, j) cell against
//! a bit-exact CPU reference using `u8::count_ones`.
//!
//! Integer popcount has no warp-reduction nondeterminism, so the
//! tolerance is **zero** — any mismatch is a kernel bug.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --no-default-features --features sm89 \
//!  --release --test bin_smoke -- --ignored`.

#![cfg(feature = "sm89")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    Bin, BinGemmArgs, BinGemmDescriptor, BinGemmPlan, LayoutSku, MatrixMut, MatrixRef,
    PlanPreference, Workspace,
};

// ============================================================================
// CPU reference — bin RCR.
//
// `A` row-major `[M, K bits]`: byte at `(i, k_byte)` holds bits
// `[A[i, 8*k_byte], A[i, 8*k_byte+1], ..., A[i, 8*k_byte+7]]` with bit
// `j` at position `j` of the byte (LSB-first).
//
// `B` col-major `[K bits, N]`: byte at `(k_byte, j)` (= gmem offset
// `j * ldb + k_byte`) holds 8 K-bits of column `j` (LSB-first).
//
// `D[i, j] = sum over k_byte of popcount(A[i, k_byte] XOR B[k_byte, j])`.
// ============================================================================

fn cpu_bin_gemm_rcr(
    m: usize,
    n: usize,
    k: usize,
    a_bytes: &[u8],
    lda_bytes: usize, // row stride of A in bytes (>= K/8)
    b_bytes: &[u8],
    ldb_bytes: usize, // column stride of B in bytes (>= K/8)
    d: &mut [i32],
    ldd_elements: usize,
) {
    assert_eq!(k % 8, 0, "K must be a multiple of 8");
    let k_bytes = k / 8;
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0u32;
            for kk in 0..k_bytes {
                let a = a_bytes[i * lda_bytes + kk];
                let b = b_bytes[j * ldb_bytes + kk];
                acc += (a ^ b).count_ones();
            }
            d[i * ldd_elements + j] = acc as i32;
        }
    }
}

// ============================================================================
// Test harness
// ============================================================================

fn run_bin_rcr(m: i32, n: i32, k: i32) {
    assert_eq!(k % 8, 0, "K must be a multiple of 8 for packed-bit storage");

    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let mu = m as usize;
    let nu = n as usize;
    let ku = k as usize;
    let k_bytes = ku / 8;

    // Build A / B as deterministic byte patterns spanning the full
    // range of nibble combinations (each byte takes 256 distinct
    // values across the buffer).
    let host_a_bytes: Vec<u8> = (0..(mu * k_bytes))
        .map(|i| ((i as u32 * 73 + 11).wrapping_mul(2654435761) >> 24) as u8)
        .collect();
    let host_b_bytes: Vec<u8> = (0..(nu * k_bytes))
        .map(|i| ((i as u32 * 41 + 7).wrapping_mul(2246822519) >> 24) as u8)
        .collect();

    let mut expected = vec![0i32; mu * nu];
    cpu_bin_gemm_rcr(
        mu, nu, ku,
        &host_a_bytes, k_bytes,
        &host_b_bytes, k_bytes,
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
        layout: LayoutSku::Rcr,
    };
    let plan = BinGemmPlan::select(&stream, &desc, PlanPreference::default())
        .expect("select bin RCR plan");

    let args = BinGemmArgs {
        // A: row-major [M, K]; ld in BYTES = K/8.
        a: MatrixRef { data: dev_a, rows: m, cols: k, ld: k_bytes as i64 },
        // B: col-major [K, N] (RCR); ld in BYTES = K/8.
        b: MatrixRef { data: dev_b, rows: k, cols: n, ld: k_bytes as i64 },
        // D: row-major [M, N] int32; ld in ELEMENTS = N.
        d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n as i64 },
    };
    plan.run(&stream, Workspace::None, args)
        .expect("bin RCR GEMM run");
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
// multiple of 8 (bit-packing). The kernel's K_TILE is 256 elements
// (= 32 bytes), so 256/512 hits aligned tiles cleanly; 128 is below
// one K-tile and exercises padding.
// ============================================================================

#[test] #[ignore]
fn bin_rcr_64_64_256() {
    // Smallest tile-aligned shape (M_TILE=N_TILE=K_TILE).
    run_bin_rcr(64, 64, 256);
}

#[test] #[ignore]
fn bin_rcr_128_128_512() {
    // 2x2 grid, 2 K-tiles.
    run_bin_rcr(128, 128, 512);
}

#[test] #[ignore]
fn bin_rcr_256_128_256() {
    // 4x2 grid, 1 K-tile.
    run_bin_rcr(256, 128, 256);
}

#[test] #[ignore]
fn bin_rcr_100_70_128() {
    // Ragged in M and N; K=128 is below K_TILE=256 (exercises K-tile
    // padding at the boundary).
    run_bin_rcr(100, 70, 128);
}

#[test] #[ignore]
fn bin_rcr_65_65_264() {
    // Tile-edge ragged: M, N each 1 past tile boundary; K=264 is one
    // K-tile + 8 bits (exercises K-tile padding when the second tile
    // is mostly zero).
    run_bin_rcr(65, 65, 264);
}
