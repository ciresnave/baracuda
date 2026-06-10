//! Real-GPU smoke tests for the Phase 74 dense FP GEMM family —
//! `DenseGemmPlan` (cuBLAS-backed) and its flat C twin
//! `baracuda_kernels_gemm_dense_*`.
//!
//! Coverage:
//! - all three layouts (RRR / RCR / CRR) on f32 vs an f64 CPU reference;
//! - padded leading dims (`lda > K` etc.) — the row-slice-view case
//!   that motivated the family (Fuel's BERT / SD-CLIP / Qwen2-MoE
//!   non-contiguous matmul operands);
//! - `β ≠ 0` read-modify-write accumulation into `D`;
//! - strided-batch (`batch = 3`) including the `stride_a = 0`
//!   broadcast case;
//! - f64 / f16 / bf16 dtypes (half dtypes accumulate in f32);
//! - one direct-FFI launch shaped exactly like Fuel's binding-table
//!   call (raw pointers, no plan layer);
//! - host-side rejection cases on the FFI `_can_implement`.
//!
//! Tolerances: cuBLAS's summation order differs from the naive CPU
//! loop, so f32 compares against an f64 reference at `1e-5` relative;
//! f16 / bf16 accumulate in f32 and round once to half storage —
//! tolerance is a few storage ULPs (`4e-3` / `1.6e-2` relative).
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --test dense_gemm_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    DenseGemmArgs, DenseGemmDescriptor, DenseGemmLayout, DenseGemmPlan, MatrixMut, MatrixRef,
    PlanPreference, Workspace,
};
use half::{bf16, f16};

// ============================================================================
// CPU reference — f64 accumulation over the row-major logical problem,
// with per-layout storage indexing and padded leading dims.
// ============================================================================

/// `D[g] = α · A[g] · B[g] + β · D[g]` over `f64`-converted inputs.
/// `a`/`b`/`d_init` are storage slices covering all batch slots.
#[allow(clippy::too_many_arguments)]
fn cpu_dense_gemm(
    m: usize,
    n: usize,
    k: usize,
    batch: usize,
    layout: DenseGemmLayout,
    a: &[f64],
    lda: usize,
    stride_a: usize,
    b: &[f64],
    ldb: usize,
    stride_b: usize,
    d_init: &[f64],
    ldd: usize,
    stride_d: usize,
    alpha: f64,
    beta: f64,
) -> Vec<f64> {
    let mut out = d_init.to_vec();
    for g in 0..batch {
        let a0 = g * stride_a;
        let b0 = g * stride_b;
        let d0 = g * stride_d;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for kk in 0..k {
                    let av = match layout {
                        // A row-major [M, K]
                        DenseGemmLayout::Rrr | DenseGemmLayout::Rcr => a[a0 + i * lda + kk],
                        // A col-major [M, K]: element (i, kk) at i + kk·lda
                        DenseGemmLayout::Crr => a[a0 + kk * lda + i],
                    };
                    let bv = match layout {
                        // B row-major [K, N]
                        DenseGemmLayout::Rrr | DenseGemmLayout::Crr => b[b0 + kk * ldb + j],
                        // B col-major [K, N]: element (kk, j) at kk + j·ldb
                        DenseGemmLayout::Rcr => b[b0 + j * ldb + kk],
                    };
                    acc += av * bv;
                }
                let idx = d0 + i * ldd + j;
                out[idx] = alpha * acc + beta * d_init[idx];
            }
        }
    }
    out
}

/// Deterministic, sign-spanning fill pattern.
fn pattern(i: usize, scale: f64, modulus: i32, offset: f64) -> f64 {
    (((i as i32 * 7 + 3) % modulus) as f64 + offset) * scale
}

fn gpu_context() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// ============================================================================
// f32 harness — parameterized over layout, lds, batch, strides, α/β.
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn run_dense_f32(
    m: i32,
    n: i32,
    k: i32,
    batch: i32,
    layout: DenseGemmLayout,
    lda: usize,
    ldb: usize,
    ldd: usize,
    stride_a: usize,
    stride_b: usize,
    stride_d: usize,
    alpha: f32,
    beta: f32,
) {
    let (ctx, stream) = gpu_context();
    let (mu, nu, ku, bu) = (m as usize, n as usize, k as usize, batch as usize);

    // Storage extents: enough for the last batch slot's full matrix at
    // its padded leading dim.
    let a_rows = match layout {
        DenseGemmLayout::Rrr | DenseGemmLayout::Rcr => mu,
        DenseGemmLayout::Crr => ku,
    };
    let b_rows = match layout {
        DenseGemmLayout::Rrr | DenseGemmLayout::Crr => ku,
        DenseGemmLayout::Rcr => nu,
    };
    let a_len = (bu - 1) * stride_a + a_rows * lda;
    let b_len = (bu - 1) * stride_b + b_rows * ldb;
    let d_len = (bu - 1) * stride_d + mu * ldd;

    let host_a: Vec<f64> = (0..a_len).map(|i| pattern(i, 0.25, 13, -6.0)).collect();
    let host_b: Vec<f64> = (0..b_len).map(|i| pattern(i, 0.125, 11, -5.0)).collect();
    let host_d: Vec<f64> = (0..d_len).map(|i| pattern(i, 0.5, 7, -3.0)).collect();

    let expected = cpu_dense_gemm(
        mu, nu, ku, bu, layout,
        &host_a, lda, stride_a,
        &host_b, ldb, stride_b,
        &host_d, ldd, stride_d,
        alpha as f64, beta as f64,
    );

    let host_a32: Vec<f32> = host_a.iter().map(|&v| v as f32).collect();
    let host_b32: Vec<f32> = host_b.iter().map(|&v| v as f32).collect();
    let host_d32: Vec<f32> = host_d.iter().map(|&v| v as f32).collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a32).expect("upload A");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b32).expect("upload B");
    let mut dev_d = DeviceBuffer::from_slice(&ctx, &host_d32).expect("upload D");

    let desc = DenseGemmDescriptor { m, n, k, batch, layout };
    let plan = DenseGemmPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select dense f32 plan");
    assert_eq!(plan.workspace_size(), 0);

    let args = DenseGemmArgs::<f32> {
        a: MatrixRef { data: dev_a.as_slice(), rows: m, cols: k, ld: lda as i64 },
        stride_a: stride_a as i64,
        b: MatrixRef { data: dev_b.as_slice(), rows: k, cols: n, ld: ldb as i64 },
        stride_b: stride_b as i64,
        d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: ldd as i64 },
        stride_d: stride_d as i64,
        alpha,
        beta,
    };
    plan.run(&stream, Workspace::None, args).expect("dense f32 run");
    stream.synchronize().expect("stream sync");

    let mut got = vec![0f32; d_len];
    dev_d.copy_to_host(&mut got).expect("download D");

    // Compare every cell inside a batch slot's [M, N] window; the
    // ld-padding columns within each visited row must hold their
    // original values. (Inter-slot gap ROWS, when stride_d > m·ldd,
    // are not visited by this loop.)
    let mut checked = 0usize;
    for g in 0..bu {
        for i in 0..mu {
            for j in 0..ldd {
                let idx = g * stride_d + i * ldd + j;
                if j < nu {
                    let e = expected[idx];
                    let tol = e.abs().max(1.0) * 1e-5;
                    assert!(
                        (got[idx] as f64 - e).abs() <= tol,
                        "mismatch @ batch {g}, ({i}, {j}): got {} expected {e} \
                         (layout {layout:?}, M={m} N={n} K={k})",
                        got[idx],
                    );
                    checked += 1;
                } else {
                    // Padding column — must hold its original value.
                    assert_eq!(
                        got[idx], host_d32[idx],
                        "ld-padding clobbered @ batch {g}, ({i}, {j})"
                    );
                }
            }
        }
    }
    assert_eq!(checked, bu * mu * nu);
}

// ============================================================================
// f32 — layouts, padded lds, β-accumulate, batch
// ============================================================================

#[test]
#[ignore]
fn dense_f32_rrr_basic() {
    // Tight lds, single batch, β = 0, ragged shape.
    run_dense_f32(33, 29, 17, 1, DenseGemmLayout::Rrr, 17, 29, 29, 0, 0, 0, 1.25, 0.0);
}

#[test]
#[ignore]
fn dense_f32_rcr_basic() {
    // RCR: B col-major [K, N] → ldb ≥ K.
    run_dense_f32(33, 29, 17, 1, DenseGemmLayout::Rcr, 17, 17, 29, 0, 0, 0, 1.0, 0.0);
}

#[test]
#[ignore]
fn dense_f32_crr_basic() {
    // CRR: A col-major [M, K] → lda ≥ M. The grad-weight shape.
    run_dense_f32(33, 29, 17, 1, DenseGemmLayout::Crr, 33, 29, 29, 0, 0, 0, 1.0, 0.0);
}

#[test]
#[ignore]
fn dense_f32_rrr_padded_lds_beta() {
    // Padded leading dims (row-slice views) + β-accumulate in one go.
    run_dense_f32(32, 24, 16, 1, DenseGemmLayout::Rrr, 16 + 3, 24 + 5, 24 + 2, 0, 0, 0, 0.75, 0.7);
}

#[test]
#[ignore]
fn dense_f32_rcr_padded_lds() {
    run_dense_f32(20, 31, 12, 1, DenseGemmLayout::Rcr, 12 + 4, 12 + 6, 31 + 1, 0, 0, 0, 1.5, 0.0);
}

#[test]
#[ignore]
fn dense_f32_crr_padded_lds() {
    run_dense_f32(21, 18, 11, 1, DenseGemmLayout::Crr, 21 + 2, 18 + 3, 18 + 4, 0, 0, 0, 1.0, 0.25);
}

#[test]
#[ignore]
fn dense_f32_rrr_strided_batch() {
    // batch = 3, disjoint slots, padded D ld.
    let (m, n, k) = (16usize, 13usize, 9usize);
    run_dense_f32(
        m as i32, n as i32, k as i32, 3, DenseGemmLayout::Rrr,
        k, n, n + 2,
        m * k, k * n, m * (n + 2),
        1.0, 0.0,
    );
}

#[test]
#[ignore]
fn dense_f32_rrr_batch_broadcast_a() {
    // stride_a = 0: one A shared across all 3 slots (the GQA-ish case).
    let (m, n, k) = (8usize, 12usize, 10usize);
    run_dense_f32(
        m as i32, n as i32, k as i32, 3, DenseGemmLayout::Rrr,
        k, n, n,
        0, k * n, m * n,
        1.0, 0.0,
    );
}

// ============================================================================
// f64
// ============================================================================

#[test]
#[ignore]
fn dense_f64_rrr_basic() {
    let (ctx, stream) = gpu_context();
    let (m, n, k) = (24i32, 19i32, 15i32);
    let (mu, nu, ku) = (24usize, 19usize, 15usize);

    let host_a: Vec<f64> = (0..mu * ku).map(|i| pattern(i, 0.25, 13, -6.0)).collect();
    let host_b: Vec<f64> = (0..ku * nu).map(|i| pattern(i, 0.125, 11, -5.0)).collect();
    let host_d = vec![0f64; mu * nu];
    let expected = cpu_dense_gemm(
        mu, nu, ku, 1, DenseGemmLayout::Rrr,
        &host_a, ku, 0, &host_b, nu, 0, &host_d, nu, 0,
        1.0, 0.0,
    );

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload A");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload B");
    let mut dev_d: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, mu * nu).expect("alloc D");

    let desc = DenseGemmDescriptor { m, n, k, batch: 1, layout: DenseGemmLayout::Rrr };
    let plan = DenseGemmPlan::<f64>::select(&stream, &desc, PlanPreference::default())
        .expect("select dense f64 plan");
    let args = DenseGemmArgs::<f64> {
        a: MatrixRef { data: dev_a.as_slice(), rows: m, cols: k, ld: ku as i64 },
        stride_a: 0,
        b: MatrixRef { data: dev_b.as_slice(), rows: k, cols: n, ld: nu as i64 },
        stride_b: 0,
        d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: nu as i64 },
        stride_d: 0,
        alpha: 1.0f64,
        beta: 0.0f64,
    };
    plan.run(&stream, Workspace::None, args).expect("dense f64 run");
    stream.synchronize().expect("stream sync");

    let mut got = vec![0f64; mu * nu];
    dev_d.copy_to_host(&mut got).expect("download D");
    for (idx, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let tol = e.abs().max(1.0) * 1e-12;
        assert!((g - e).abs() <= tol, "f64 mismatch @ {idx}: got {g} expected {e}");
    }
}

// ============================================================================
// f16 / bf16 — accumulate in f32, round once to half storage.
// ============================================================================

/// Shared half-precision harness. `to_half` / `from_half` round-trip
/// through the storage dtype; `rel_tol` covers storage ULPs + cuBLAS
/// summation-order drift.
fn run_dense_half<T, FI, FO>(
    layout: DenseGemmLayout,
    lda: usize,
    ldb: usize,
    to_half: FI,
    from_half: FO,
    rel_tol: f64,
) where
    T: baracuda_kernels::Element + Copy + Default + 'static,
    FI: Fn(f32) -> T,
    FO: Fn(T) -> f32,
{
    use baracuda_kernels::ScalarType;

    let (ctx, stream) = gpu_context();
    let (m, n, k) = (32i32, 28i32, 24i32);
    let (mu, nu, ku) = (32usize, 28usize, 24usize);

    let a_rows = mu; // both layouts used here keep A row-major
    let b_rows = match layout {
        DenseGemmLayout::Rrr | DenseGemmLayout::Crr => ku,
        DenseGemmLayout::Rcr => nu,
    };

    // Fill with exactly-representable half values (multiples of 1/8 in
    // [-0.75, 0.75]) so the only rounding is the final store.
    let host_a_h: Vec<T> = (0..a_rows * lda)
        .map(|i| to_half(pattern(i, 0.125, 13, -6.0) as f32))
        .collect();
    let host_b_h: Vec<T> = (0..b_rows * ldb)
        .map(|i| to_half(pattern(i, 0.125, 11, -5.0) as f32))
        .collect();

    let host_a: Vec<f64> = host_a_h.iter().map(|&v| from_half(v) as f64).collect();
    let host_b: Vec<f64> = host_b_h.iter().map(|&v| from_half(v) as f64).collect();
    let host_d = vec![0f64; mu * nu];
    let expected = cpu_dense_gemm(
        mu, nu, ku, 1, layout,
        &host_a, lda, 0, &host_b, ldb, 0, &host_d, nu, 0,
        1.0, 0.0,
    );

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a_h).expect("upload A");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b_h).expect("upload B");
    let mut dev_d: DeviceBuffer<T> = DeviceBuffer::zeros(&ctx, mu * nu).expect("alloc D");

    let desc = DenseGemmDescriptor { m, n, k, batch: 1, layout };
    let plan = DenseGemmPlan::<T>::select(&stream, &desc, PlanPreference::default())
        .expect("select dense half plan");
    let args = DenseGemmArgs::<T> {
        a: MatrixRef { data: dev_a.as_slice(), rows: m, cols: k, ld: lda as i64 },
        stride_a: 0,
        b: MatrixRef { data: dev_b.as_slice(), rows: k, cols: n, ld: ldb as i64 },
        stride_b: 0,
        d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: nu as i64 },
        stride_d: 0,
        alpha: <T::Scalar as ScalarType>::ONE,
        beta: <T::Scalar as ScalarType>::ZERO,
    };
    plan.run(&stream, Workspace::None, args).expect("dense half run");
    stream.synchronize().expect("stream sync");

    let mut got = vec![T::default(); mu * nu];
    dev_d.copy_to_host(&mut got).expect("download D");
    for (idx, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let gf = from_half(g) as f64;
        let tol = e.abs().max(1.0) * rel_tol;
        assert!(
            (gf - e).abs() <= tol,
            "half mismatch @ {idx}: got {gf} expected {e} (layout {layout:?})"
        );
    }
}

#[test]
#[ignore]
fn dense_f16_rrr_basic() {
    run_dense_half::<f16, _, _>(
        DenseGemmLayout::Rrr, 24, 28,
        f16::from_f32, |v: f16| v.to_f32(),
        4e-3,
    );
}

#[test]
#[ignore]
fn dense_bf16_rcr_basic() {
    run_dense_half::<bf16, _, _>(
        DenseGemmLayout::Rcr, 24, 24,
        bf16::from_f32, |v: bf16| v.to_f32(),
        1.6e-2,
    );
}

// ============================================================================
// Direct FFI — the exact shape Fuel's binding table calls.
// ============================================================================

#[test]
#[ignore]
fn dense_f32_direct_ffi() {
    use core::ffi::c_void;

    let (ctx, stream) = gpu_context();
    let (m, n, k) = (8usize, 6usize, 4usize);

    let host_a: Vec<f32> = (0..m * k).map(|i| pattern(i, 0.5, 7, -3.0) as f32).collect();
    let host_b: Vec<f32> = (0..k * n).map(|i| pattern(i, 0.25, 5, -2.0) as f32).collect();
    let host_d64: Vec<f64> = vec![0.0; m * n];
    let host_a64: Vec<f64> = host_a.iter().map(|&v| v as f64).collect();
    let host_b64: Vec<f64> = host_b.iter().map(|&v| v as f64).collect();
    let expected = cpu_dense_gemm(
        m, n, k, 1, DenseGemmLayout::Rrr,
        &host_a64, k, 0, &host_b64, n, 0, &host_d64, n, 0,
        1.0, 0.0,
    );

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload A");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload B");
    let mut dev_d: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, m * n).expect("alloc D");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_gemm_dense_f32_run(
            m as i32, n as i32, k as i32, 1, /* layout RRR */ 0,
            1.0f32, 0.0f32,
            dev_a.as_slice().as_raw().0 as *const c_void, k as i64, 0,
            dev_b.as_slice().as_raw().0 as *const c_void, n as i64, 0,
            dev_d.as_slice_mut().as_raw().0 as *mut c_void, n as i64, 0,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "direct FFI dense f32 run failed");
    stream.synchronize().expect("stream sync");

    let mut got = vec![0f32; m * n];
    dev_d.copy_to_host(&mut got).expect("download D");
    for (idx, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g as f64 - e).abs() <= e.abs().max(1.0) * 1e-5,
            "direct-FFI mismatch @ {idx}: got {g} expected {e}"
        );
    }
}

// ============================================================================
// Host-side rejection — FFI `_can_implement` (no device work).
// ============================================================================

#[test]
#[ignore]
fn dense_can_implement_rejections() {
    // Valid baseline.
    assert_eq!(
        baracuda_kernels_sys::baracuda_kernels_gemm_dense_f32_can_implement(
            8, 6, 4, 1, 0, 4, 6, 6, 0, 0, 0,
        ),
        0
    );
    // Bad layout tag.
    assert_eq!(
        baracuda_kernels_sys::baracuda_kernels_gemm_dense_f32_can_implement(
            8, 6, 4, 1, 3, 4, 6, 6, 0, 0, 0,
        ),
        2
    );
    // lda below the RRR minimum (K = 4).
    assert_eq!(
        baracuda_kernels_sys::baracuda_kernels_gemm_dense_f32_can_implement(
            8, 6, 4, 1, 0, 3, 6, 6, 0, 0, 0,
        ),
        2
    );
    // RCR needs ldb ≥ K, not ≥ N: ldb = 4 is legal there...
    assert_eq!(
        baracuda_kernels_sys::baracuda_kernels_gemm_dense_f32_can_implement(
            8, 6, 4, 1, 1, 4, 4, 6, 0, 0, 0,
        ),
        0
    );
    // ...but illegal under RRR (needs ldb ≥ N = 6).
    assert_eq!(
        baracuda_kernels_sys::baracuda_kernels_gemm_dense_f32_can_implement(
            8, 6, 4, 1, 0, 4, 4, 6, 0, 0, 0,
        ),
        2
    );
    // CRR needs lda ≥ M.
    assert_eq!(
        baracuda_kernels_sys::baracuda_kernels_gemm_dense_f32_can_implement(
            8, 6, 4, 1, 2, 8, 6, 6, 0, 0, 0,
        ),
        0
    );
    assert_eq!(
        baracuda_kernels_sys::baracuda_kernels_gemm_dense_f32_can_implement(
            8, 6, 4, 1, 2, 7, 6, 6, 0, 0, 0,
        ),
        2
    );
    // batch > 1 with stride_d = 0 races.
    assert_eq!(
        baracuda_kernels_sys::baracuda_kernels_gemm_dense_f32_can_implement(
            8, 6, 4, 2, 0, 4, 6, 6, 0, 0, 0,
        ),
        2
    );
    // Negative extent.
    assert_eq!(
        baracuda_kernels_sys::baracuda_kernels_gemm_dense_f32_can_implement(
            -1, 6, 4, 1, 0, 4, 6, 6, 0, 0, 0,
        ),
        2
    );
    // Workspace query is always 0.
    assert_eq!(
        baracuda_kernels_sys::baracuda_kernels_gemm_dense_f32_workspace_size(8, 6, 4, 1, 0),
        0
    );
}

// ============================================================================
// Plan-layer buffer-bounds rejection — the safe layer must refuse to
// launch when the device slices can't cover the batch-stride reach
// (the FFI layer has no length information; this typed check is the
// soundness boundary).
// ============================================================================

#[test]
#[ignore]
fn dense_plan_rejects_undersized_and_negative_stride_buffers() {
    let (ctx, stream) = gpu_context();
    let (m, n, k) = (8i32, 6i32, 4i32);
    let (mu, nu, ku) = (8usize, 6usize, 4usize);

    let dev_a: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, mu * ku * 3).expect("alloc A");
    let dev_b: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, ku * nu * 3).expect("alloc B");
    // D sized for ONE slot only — batch = 3 must be rejected.
    let mut dev_d: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, mu * nu).expect("alloc D");

    let desc = DenseGemmDescriptor { m, n, k, batch: 3, layout: DenseGemmLayout::Rrr };
    let plan = DenseGemmPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");

    let args = DenseGemmArgs::<f32> {
        a: MatrixRef { data: dev_a.as_slice(), rows: m, cols: k, ld: ku as i64 },
        stride_a: (mu * ku) as i64,
        b: MatrixRef { data: dev_b.as_slice(), rows: k, cols: n, ld: nu as i64 },
        stride_b: (ku * nu) as i64,
        d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: nu as i64 },
        stride_d: (mu * nu) as i64,
        alpha: 1.0,
        beta: 0.0,
    };
    let err = plan.can_implement(&args).expect_err("single-slot D must be rejected");
    assert!(
        matches!(err, baracuda_kernels::Error::BufferTooSmall { .. }),
        "expected BufferTooSmall, got {err:?}"
    );

    // Negative batch stride: never in-bounds from the slice base.
    let mut dev_d3: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, mu * nu * 3).expect("alloc D3");
    let args = DenseGemmArgs::<f32> {
        a: MatrixRef { data: dev_a.as_slice(), rows: m, cols: k, ld: ku as i64 },
        stride_a: -((mu * ku) as i64),
        b: MatrixRef { data: dev_b.as_slice(), rows: k, cols: n, ld: nu as i64 },
        stride_b: (ku * nu) as i64,
        d: MatrixMut { data: dev_d3.as_slice_mut(), rows: m, cols: n, ld: nu as i64 },
        stride_d: (mu * nu) as i64,
        alpha: 1.0,
        beta: 0.0,
    };
    let err = plan.can_implement(&args).expect_err("negative stride_a must be rejected");
    assert!(
        matches!(err, baracuda_kernels::Error::InvalidProblem(_)),
        "expected InvalidProblem, got {err:?}"
    );
}

// ============================================================================
// Handle-pool concurrency — N threads (> POOL_SLOTS) each running a
// loop of small GEMMs on their own context exercises take/put racing,
// slot claiming across contexts, and the overflow-destroy fallback.
// Correctness check per thread keeps the test meaningful beyond
// "didn't crash".
// ============================================================================

#[test]
#[ignore]
fn dense_f32_concurrent_handle_pool() {
    init().expect("driver init");
    let threads: Vec<_> = (0..12)
        .map(|t| {
            std::thread::spawn(move || {
                let device = Device::get(0).expect("device 0");
                let ctx = Context::new(&device).expect("context");
                let stream = Stream::new(&ctx).expect("stream");
                let (m, n, k) = (16i32, 12i32, 8i32);
                let (mu, nu, ku) = (16usize, 12usize, 8usize);

                let host_a: Vec<f32> =
                    (0..mu * ku).map(|i| pattern(i + t, 0.25, 13, -6.0) as f32).collect();
                let host_b: Vec<f32> =
                    (0..ku * nu).map(|i| pattern(i + t, 0.125, 11, -5.0) as f32).collect();
                let host_a64: Vec<f64> = host_a.iter().map(|&v| v as f64).collect();
                let host_b64: Vec<f64> = host_b.iter().map(|&v| v as f64).collect();
                let host_d = vec![0f64; mu * nu];
                let expected = cpu_dense_gemm(
                    mu, nu, ku, 1, DenseGemmLayout::Rrr,
                    &host_a64, ku, 0, &host_b64, nu, 0, &host_d, nu, 0,
                    1.0, 0.0,
                );

                let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload A");
                let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload B");
                let mut dev_d: DeviceBuffer<f32> =
                    DeviceBuffer::zeros(&ctx, mu * nu).expect("alloc D");

                let desc = DenseGemmDescriptor {
                    m, n, k, batch: 1, layout: DenseGemmLayout::Rrr,
                };
                let plan =
                    DenseGemmPlan::<f32>::select(&stream, &desc, PlanPreference::default())
                        .expect("select");
                // Several iterations per thread: first takes/creates,
                // the rest re-take from the pool (or transiently
                // create under contention).
                for _ in 0..8 {
                    let args = DenseGemmArgs::<f32> {
                        a: MatrixRef { data: dev_a.as_slice(), rows: m, cols: k, ld: ku as i64 },
                        stride_a: 0,
                        b: MatrixRef { data: dev_b.as_slice(), rows: k, cols: n, ld: nu as i64 },
                        stride_b: 0,
                        d: MatrixMut {
                            data: dev_d.as_slice_mut(),
                            rows: m, cols: n, ld: nu as i64,
                        },
                        stride_d: 0,
                        alpha: 1.0,
                        beta: 0.0,
                    };
                    plan.run(&stream, Workspace::None, args).expect("run");
                }
                stream.synchronize().expect("sync");

                let mut got = vec![0f32; mu * nu];
                dev_d.copy_to_host(&mut got).expect("download D");
                for (idx, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
                    assert!(
                        (g as f64 - e).abs() <= e.abs().max(1.0) * 1e-5,
                        "thread {t} mismatch @ {idx}: got {g} expected {e}"
                    );
                }
            })
        })
        .collect();
    for th in threads {
        th.join().expect("worker thread panicked");
    }
}
