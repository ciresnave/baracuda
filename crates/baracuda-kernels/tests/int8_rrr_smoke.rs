//! Real-GPU smoke tests for the bespoke int8 RRR kernels in
//! `baracuda-kernels-sys` (the SKUs that aren't expressible through
//! CUTLASS 4.2.0 — see
//! ~/.claude/plans/baracuda-kernels-comprehensive.md §5).
//!
//! Phase 1 coverage: `S8 × Rrr × Identity` (this test file). The
//! remaining 17 RRR SKUs (`U8`, bias family across `f32` / `i32` bias)
//! follow in subsequent commits.
//!
//! Each test verifies that the kernel produces bit-identical output to
//! a CPU reference that replicates the kernel's
//! int32-accumulate → float-alpha-scale → saturating-cast-to-int8
//! pipeline. Integer GEMM is deterministic (no warp-reduction
//! nondeterminism), so the assertion is bit-exact — any mismatch is a
//! real bug.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    EpilogueKind, IntGemmArgs, IntGemmDescriptor, IntGemmPlan, LayoutSku, MatrixMut, MatrixRef,
    PlanPreference, S8, Workspace,
};

// ============================================================================
// CPU reference — RRR layout (row-major B).
// ============================================================================

fn sat_cast_s8(x: f32) -> i8 {
    let r = x.round() as i32;
    r.clamp(i8::MIN as i32, i8::MAX as i32) as i8
}

/// Reference int8 GEMM, RRR layout, Identity epilogue.
///
/// Mirrors the bespoke kernel's compute order:
///   acc = Σ_k (i32) A[i, k] * (i32) B[k, j]     // int32 accumulator
///   z   = alpha * (f32)acc + beta * (f32)C[i, j]
///   D[i, j] = sat_cast_s8(z)
///
/// `A` row-major [M, K] with stride `lda` along K.
/// `B` row-major [K, N] with stride `ldb` along N — indexed
///     `b[k * ldb + j]` (the gmem layout the bespoke kernel consumes
///     directly without transposing).
#[allow(clippy::too_many_arguments)]
fn cpu_int_gemm_rrr_identity(
    m: usize,
    n: usize,
    k: usize,
    a: &[i8],
    lda: usize,
    b: &[i8],
    ldb: usize,
    c: Option<(&[i8], usize)>,
    alpha: f32,
    beta: f32,
    d: &mut [i8],
    ldd: usize,
) {
    for i in 0..m {
        for j in 0..n {
            // int32 accumulator — saturating to match the kernel's
            // `mma.sync.…satfinite` semantics.
            let mut acc: i32 = 0;
            for kk in 0..k {
                let a_val = a[i * lda + kk] as i32;
                let b_val = b[kk * ldb + j] as i32;
                acc = acc.saturating_add(a_val.saturating_mul(b_val));
            }
            let mut z = alpha * (acc as f32);
            if let Some((c_buf, ldc)) = c {
                let c_val = c_buf[i * ldc + j] as i32;
                z += beta * (c_val as f32);
            }
            d[i * ldd + j] = sat_cast_s8(z);
        }
    }
}

// ============================================================================
// Test harness
// ============================================================================

fn run_s8_rrr_identity_smoke(m: i32, n: i32, k: i32) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let mu = m as usize;
    let nu = n as usize;
    let ku = k as usize;

    // Bounded inputs so the int32 accumulator stays meaningful for the
    // s8 output (no constant saturation).
    let host_a: Vec<i8> = (0..(mu * ku))
        .map(|i| (((i as i32 * 7) % 15) - 7) as i8)
        .collect();
    let host_b: Vec<i8> = (0..(ku * nu))
        .map(|i| (((i as i32 * 11) % 13) - 6) as i8)
        .collect();

    let alpha: f32 = 0.125; // small alpha keeps the post-dequant scalars in s8 range
    let beta: f32 = 0.0;

    // CPU reference
    let mut host_d_ref = vec![0i8; mu * nu];
    cpu_int_gemm_rrr_identity(
        mu, nu, ku,
        &host_a, ku,
        &host_b, nu,
        None,
        alpha, beta,
        &mut host_d_ref, nu,
    );

    // Upload inputs via `DeviceBuffer<u8>` + `view_as::<S8>()` — the
    // same pattern the int8 RCR smoke test uses (S8 is
    // `#[repr(transparent)]` over `i8`, so the byte buffer is a valid
    // S8 buffer by construction).
    let host_a_u: &[u8] =
        unsafe { core::slice::from_raw_parts(host_a.as_ptr() as *const u8, host_a.len()) };
    let host_b_u: &[u8] =
        unsafe { core::slice::from_raw_parts(host_b.as_ptr() as *const u8, host_b.len()) };
    let dev_a_bytes = DeviceBuffer::from_slice(&ctx, host_a_u).expect("upload A");
    let dev_b_bytes = DeviceBuffer::from_slice(&ctx, host_b_u).expect("upload B");
    let dev_a = dev_a_bytes.view_as::<S8>();
    let dev_b = dev_b_bytes.view_as::<S8>();
    let mut dev_d: DeviceBuffer<S8> =
        DeviceBuffer::zeros(&ctx, mu * nu).expect("alloc D");

    let desc = IntGemmDescriptor {
        m,
        n,
        k,
        layout: LayoutSku::Rrr,
        epilogue: EpilogueKind::Identity,
    };

    // Note: `IntGemmPlan<S8>` defaults `BT = f32`; for Identity the bias
    // type is irrelevant (no bias supplied).
    let plan = IntGemmPlan::<S8>::select(&stream, &desc, PlanPreference::default())
        .expect("select s8 RRR Identity plan");

    let args = IntGemmArgs::<S8, f32> {
        a: MatrixRef { data: dev_a, rows: m, cols: k, ld: k as i64 },
        b: MatrixRef { data: dev_b, rows: k, cols: n, ld: n as i64 },
        c: None,
        d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n as i64 },
        bias: None,
        alpha,
        beta,
    };
    plan.run(&stream, Workspace::None, args).expect("run s8 RRR Identity kernel");
    stream.synchronize().expect("stream sync");

    let mut host_d_s8 = vec![S8(0); mu * nu];
    dev_d.copy_to_host(&mut host_d_s8).expect("download D");

    let mut mismatches = 0usize;
    let mut first_mismatch: Option<(usize, i8, i8)> = None;
    for (idx, (got, &expected)) in host_d_s8.iter().zip(host_d_ref.iter()).enumerate() {
        if got.0 != expected {
            mismatches += 1;
            if first_mismatch.is_none() {
                first_mismatch = Some((idx, got.0, expected));
            }
        }
    }
    if mismatches > 0 {
        let (idx, got, expected) = first_mismatch.unwrap();
        let row = idx / nu;
        let col = idx % nu;
        panic!(
            "{mismatches} mismatches across {} cells (M={m} N={n} K={k}); \
             first @ idx={idx} (row={row}, col={col}): got s8={got} expected s8={expected}",
            host_d_s8.len(),
        );
    }
}

// ============================================================================
// Tests
// ============================================================================

/// Tile-exact shape — matches the kernel's 64×64×32 threadblock tile,
/// so all per-thread bounds-guards are in their happy path. If this
/// fails the bug is in core MMA load / register layout / store, not in
/// edge handling.
#[test]
#[ignore]
fn s8_rrr_identity_64_64_32() {
    run_s8_rrr_identity_smoke(64, 64, 32);
}

/// Bigger square — exercises the K-tile inner loop (K = 4 K-tiles) and
/// the M × N grid (2 × 2 threadblocks).
#[test]
#[ignore]
fn s8_rrr_identity_128_128_128() {
    run_s8_rrr_identity_smoke(128, 128, 128);
}

/// Non-square problem typical of attention QKV projections.
#[test]
#[ignore]
fn s8_rrr_identity_256_128_64() {
    run_s8_rrr_identity_smoke(256, 128, 64);
}

/// Non-tile-aligned shape — exercises the bounds-guard paths in both
/// the gmem→smem load and the store epilogue.
#[test]
#[ignore]
fn s8_rrr_identity_100_70_50() {
    run_s8_rrr_identity_smoke(100, 70, 50);
}
