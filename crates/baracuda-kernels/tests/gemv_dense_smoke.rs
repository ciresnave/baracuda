//! Real-GPU smoke tests for the capture-safe dense m=1 GEMV family —
//! `GemvDensePlan` (bespoke kernel) and its flat C twin
//! `baracuda_kernels_gemv_dense_m1_*`.
//!
//! Coverage:
//! - a plain batch = 1 GEMV (f32) vs an f64 CPU matmul reference;
//! - a batched **stride_b == 0 GQA broadcast** case (batch = n_heads,
//!   distinct A/D per head, one shared B) — the valuable axis Fuel's
//!   capture envelope needs;
//! - a bf16 case (widened to f32 inside the kernel, rounded once on
//!   store);
//! - padded leading dims (`ldb > n`, `ldd > n`) — the padding columns
//!   must survive untouched;
//! - one direct-FFI launch shaped exactly like Fuel's binding-table
//!   call (raw pointers, no plan layer);
//! - host-side rejection cases on the FFI `_can_implement`.
//!
//! The kernel accumulates in f32 (true IEEE, NOT TF32) with a fixed
//! serial K-loop, so f32 compares against an f64 reference at a tight
//! `1e-4` relative tolerance; bf16 rounds once to storage → a few
//! storage ULPs (`1.6e-2` relative).
//!
//! These tests exercise warm CORRECTNESS. The load-bearing capture→replay
//! byte-identity guarantee (the reason this kernel exists — cuBLAS
//! `gemm_dense` cannot promise it) is verified on-device in the sibling
//! `capture_replay_gather_gemv.rs` (4 graph replays byte-identical to the
//! warm reference, confirmed on sm_89 / CUDA 13.3).
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --test gemv_dense_smoke -- --ignored`.

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    GemvDenseArgs, GemvDenseDescriptor, GemvDensePlan, MatrixMut, MatrixRef, PlanPreference,
    Workspace,
};
use half::bf16;

// ============================================================================
// CPU reference — f64 accumulation over the RRR m=1 problem.
// `a`/`b`/`d_init` are storage slices covering all batch slots.
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn cpu_gemv(
    n: usize,
    k: usize,
    batch: usize,
    a: &[f64],
    stride_a: usize,
    b: &[f64],
    ldb: usize,
    stride_b: usize,
    d_init: &[f64],
    stride_d: usize,
    alpha: f64,
    beta: f64,
) -> Vec<f64> {
    let mut out = d_init.to_vec();
    for g in 0..batch {
        let a0 = g * stride_a;
        let b0 = g * stride_b;
        let d0 = g * stride_d;
        for j in 0..n {
            let mut acc = 0.0f64;
            // A is a single row (m == 1): element kk at a0 + kk.
            for kk in 0..k {
                acc += a[a0 + kk] * b[b0 + kk * ldb + j];
            }
            out[d0 + j] = alpha * acc + beta * d_init[d0 + j];
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
// f32 harness — parameterized over n, k, batch, lds, strides, α/β.
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn run_gemv_f32(
    n: i32,
    k: i32,
    batch: i32,
    ldb: usize,
    ldd: usize,
    stride_a: usize,
    stride_b: usize,
    stride_d: usize,
    alpha: f32,
    beta: f32,
) {
    let (ctx, stream) = gpu_context();
    let (nu, ku, bu) = (n as usize, k as usize, batch as usize);
    let lda = ku; // RRR minimum; unused at m == 1 but honored.

    // Storage extents covering the last batch slot at its padded ld.
    let a_len = (bu - 1) * stride_a + ku;
    let b_len = (bu - 1) * stride_b + (ku - 1) * ldb + nu;
    let d_len = (bu - 1) * stride_d + nu;

    let host_a: Vec<f64> = (0..a_len).map(|i| pattern(i, 0.25, 13, -6.0)).collect();
    let host_b: Vec<f64> = (0..b_len).map(|i| pattern(i, 0.125, 11, -5.0)).collect();
    let host_d: Vec<f64> = (0..d_len).map(|i| pattern(i, 0.5, 7, -3.0)).collect();

    let expected = cpu_gemv(
        nu,
        ku,
        bu,
        &host_a,
        stride_a,
        &host_b,
        ldb,
        stride_b,
        &host_d,
        stride_d,
        alpha as f64,
        beta as f64,
    );

    let host_a32: Vec<f32> = host_a.iter().map(|&v| v as f32).collect();
    let host_b32: Vec<f32> = host_b.iter().map(|&v| v as f32).collect();
    let host_d32: Vec<f32> = host_d.iter().map(|&v| v as f32).collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a32).expect("upload A");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b32).expect("upload B");
    let mut dev_d = DeviceBuffer::from_slice(&ctx, &host_d32).expect("upload D");

    let desc = GemvDenseDescriptor { m: 1, n, k, batch };
    let plan = GemvDensePlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select gemv f32 plan");
    assert_eq!(plan.workspace_size(), 0);

    let args = GemvDenseArgs::<f32> {
        a: MatrixRef {
            data: dev_a.as_slice(),
            rows: 1,
            cols: k,
            ld: lda as i64,
        },
        stride_a: stride_a as i64,
        b: MatrixRef {
            data: dev_b.as_slice(),
            rows: k,
            cols: n,
            ld: ldb as i64,
        },
        stride_b: stride_b as i64,
        d: MatrixMut {
            data: dev_d.as_slice_mut(),
            rows: 1,
            cols: n,
            ld: ldd as i64,
        },
        stride_d: stride_d as i64,
        alpha,
        beta,
    };
    plan.run(&stream, Workspace::None, args)
        .expect("gemv f32 run");
    stream.synchronize().expect("stream sync");

    let mut got = vec![0f32; d_len];
    dev_d.copy_to_host(&mut got).expect("download D");

    // At m == 1 each slot's D is a single row of `n` contiguous columns;
    // `ldd` never creates intra-row padding (there is only one row). The
    // only untouched cells are INTER-slot gaps when `stride_d > n` — those
    // must hold their original value (guards against an out-of-window
    // write). `expected` already carries the original at gap cells.
    let is_output = |idx: usize| -> bool {
        (0..bu).any(|g| {
            let base = g * stride_d;
            idx >= base && idx < base + nu
        })
    };
    let mut checked = 0usize;
    for idx in 0..d_len {
        if is_output(idx) {
            let e = expected[idx];
            let tol = e.abs().max(1.0) * 1e-4;
            assert!(
                (got[idx] as f64 - e).abs() <= tol,
                "mismatch @ col {idx}: got {} expected {e} (N={n} K={k})",
                got[idx],
            );
            checked += 1;
        } else {
            assert_eq!(got[idx], host_d32[idx], "inter-slot gap clobbered @ {idx}");
        }
    }
    assert_eq!(checked, bu * nu);
}

// ============================================================================
// f32 — batch = 1, GQA broadcast, padded lds, β-accumulate.
// ============================================================================

#[test]
#[ignore]
fn gemv_f32_batch1_basic() {
    // Tight lds, single batch, β = 0.
    run_gemv_f32(29, 40, 1, 29, 29, 0, 0, 0, 1.25, 0.0);
}

#[test]
#[ignore]
fn gemv_f32_batch1_padded_beta() {
    // Padded ldb / ldd + β-accumulate into D in one launch.
    run_gemv_f32(24, 33, 1, 24 + 5, 24 + 2, 0, 0, 0, 0.75, 0.7);
}

#[test]
#[ignore]
fn gemv_f32_gqa_broadcast_b() {
    // The GQA scores·v shape: batch = n_heads, distinct A/D per head,
    // ONE shared B (stride_b = 0). This is the axis cuBLAS can't capture
    // and the reason this kernel exists.
    let (n, k, batch) = (17i32, 40i32, 4i32);
    let (nu, ku) = (n as usize, k as usize);
    run_gemv_f32(
        n, k, batch, nu, // ldb tight
        nu, // ldd tight
        ku, // stride_a: distinct A row per head
        0,  // stride_b == 0: shared B (GQA broadcast)
        nu, // stride_d: distinct D per head
        1.0, 0.0,
    );
}

#[test]
#[ignore]
fn gemv_f32_strided_batch_distinct_b() {
    // batch = 3 with distinct B per slot (stride_b != 0) + padded D ld.
    let (n, k, batch) = (13i32, 20i32, 3i32);
    let (nu, ku) = (n as usize, k as usize);
    run_gemv_f32(n, k, batch, nu, nu + 2, ku, ku * nu, nu + 2, 1.0, 0.0);
}

// ============================================================================
// bf16 — accumulate in f32, round once to storage.
// ============================================================================

#[test]
#[ignore]
fn gemv_bf16_batch1_basic() {
    let (ctx, stream) = gpu_context();
    let (n, k) = (28i32, 24i32);
    let (nu, ku) = (n as usize, k as usize);

    // Exactly-representable bf16 values (multiples of 1/8 in a small
    // range) so the only rounding is the final store.
    let host_a_h: Vec<bf16> = (0..ku)
        .map(|i| bf16::from_f32(pattern(i, 0.125, 13, -6.0) as f32))
        .collect();
    let host_b_h: Vec<bf16> = (0..ku * nu)
        .map(|i| bf16::from_f32(pattern(i, 0.125, 11, -5.0) as f32))
        .collect();

    let host_a: Vec<f64> = host_a_h.iter().map(|&v| v.to_f32() as f64).collect();
    let host_b: Vec<f64> = host_b_h.iter().map(|&v| v.to_f32() as f64).collect();
    let host_d = vec![0f64; nu];
    let expected = cpu_gemv(nu, ku, 1, &host_a, 0, &host_b, nu, 0, &host_d, 0, 1.0, 0.0);

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a_h).expect("upload A");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b_h).expect("upload B");
    let mut dev_d: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, nu).expect("alloc D");

    let desc = GemvDenseDescriptor {
        m: 1,
        n,
        k,
        batch: 1,
    };
    let plan = GemvDensePlan::<bf16>::select(&stream, &desc, PlanPreference::default())
        .expect("select gemv bf16 plan");

    let args = GemvDenseArgs::<bf16> {
        a: MatrixRef {
            data: dev_a.as_slice(),
            rows: 1,
            cols: k,
            ld: ku as i64,
        },
        stride_a: 0,
        b: MatrixRef {
            data: dev_b.as_slice(),
            rows: k,
            cols: n,
            ld: nu as i64,
        },
        stride_b: 0,
        d: MatrixMut {
            data: dev_d.as_slice_mut(),
            rows: 1,
            cols: n,
            ld: nu as i64,
        },
        stride_d: 0,
        alpha: 1.0f32,
        beta: 0.0f32,
    };
    plan.run(&stream, Workspace::None, args)
        .expect("gemv bf16 run");
    stream.synchronize().expect("stream sync");

    let mut got = vec![bf16::ZERO; nu];
    dev_d.copy_to_host(&mut got).expect("download D");
    for (idx, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let gf = g.to_f32() as f64;
        let tol = e.abs().max(1.0) * 1.6e-2;
        assert!(
            (gf - e).abs() <= tol,
            "bf16 mismatch @ {idx}: got {gf} expected {e}"
        );
    }
}

// ============================================================================
// Direct FFI — the exact shape Fuel's binding table calls.
// ============================================================================

#[test]
#[ignore]
fn gemv_f32_direct_ffi() {
    use core::ffi::c_void;

    let (ctx, stream) = gpu_context();
    let (n, k) = (6usize, 8usize);

    let host_a: Vec<f32> = (0..k).map(|i| pattern(i, 0.5, 7, -3.0) as f32).collect();
    let host_b: Vec<f32> = (0..k * n)
        .map(|i| pattern(i, 0.25, 5, -2.0) as f32)
        .collect();
    let host_a64: Vec<f64> = host_a.iter().map(|&v| v as f64).collect();
    let host_b64: Vec<f64> = host_b.iter().map(|&v| v as f64).collect();
    let host_d64 = vec![0.0f64; n];
    let expected = cpu_gemv(
        n, k, 1, &host_a64, 0, &host_b64, n, 0, &host_d64, 0, 1.0, 0.0,
    );

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload A");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload B");
    let mut dev_d: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n).expect("alloc D");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_gemv_dense_m1_f32_run(
            1,
            n as i32,
            k as i32,
            1,
            /* layout RRR */ 0,
            1.0f32,
            0.0f32,
            dev_a.as_slice().as_raw().0 as *const c_void,
            k as i64,
            0,
            dev_b.as_slice().as_raw().0 as *const c_void,
            n as i64,
            0,
            dev_d.as_slice_mut().as_raw().0 as *mut c_void,
            n as i64,
            0,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "direct-FFI gemv f32 run failed");
    stream.synchronize().expect("stream sync");

    let mut got = vec![0f32; n];
    dev_d.copy_to_host(&mut got).expect("download D");
    for (idx, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g as f64 - e).abs() <= e.abs().max(1.0) * 1e-4,
            "direct-FFI mismatch @ {idx}: got {g} expected {e}"
        );
    }
}

// ============================================================================
// Host-side rejection — FFI `_can_implement` (no device work).
// ============================================================================

#[test]
#[ignore]
fn gemv_can_implement_rejections() {
    use baracuda_kernels_sys::baracuda_kernels_gemv_dense_m1_f32_can_implement as can;
    // args: (m, n, k, batch, layout, lda, ldb, ldd, stride_a, stride_b, stride_d)
    // SAFETY: can_implement / workspace_size are pure host-side validation (no
    // device work, no pointer deref); the `unsafe` is only Rust 2024's extern-fn
    // call obligation.
    unsafe {
        // Valid baseline (m = 1, RRR, tight lds).
        assert_eq!(can(1, 6, 4, 1, 0, 4, 6, 6, 0, 0, 0), 0);
        // m > 1 is rejected (belongs on the batched gemm_dense path).
        assert_eq!(can(2, 6, 4, 1, 0, 4, 6, 6, 0, 0, 0), 2);
        // Bad layout tag (only RRR = 0 supported).
        assert_eq!(can(1, 6, 4, 1, 1, 4, 6, 6, 0, 0, 0), 2);
        // lda below K.
        assert_eq!(can(1, 6, 4, 1, 0, 3, 6, 6, 0, 0, 0), 2);
        // ldb below N.
        assert_eq!(can(1, 6, 4, 1, 0, 4, 5, 6, 0, 0, 0), 2);
        // ldd below N.
        assert_eq!(can(1, 6, 4, 1, 0, 4, 6, 5, 0, 0, 0), 2);
        // batch > 1 with stride_d = 0 races.
        assert_eq!(can(1, 6, 4, 2, 0, 4, 6, 6, 0, 0, 0), 2);
        // batch > 1 with stride_b = 0 (GQA broadcast) is fine.
        assert_eq!(can(1, 6, 4, 2, 0, 4, 6, 6, 4, 0, 6), 0);
        // Negative extent.
        assert_eq!(can(1, -1, 4, 1, 0, 4, 6, 6, 0, 0, 0), 2);
        // Workspace query is always 0.
        assert_eq!(
            baracuda_kernels_sys::baracuda_kernels_gemv_dense_m1_f32_workspace_size(1, 6, 4, 1, 0),
            0
        );
    }
}
