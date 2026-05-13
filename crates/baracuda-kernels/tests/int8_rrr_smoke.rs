//! Real-GPU smoke tests for the bespoke int8 RRR kernels in
//! `baracuda-kernels-sys`. Eighteen SKUs total:
//!
//!   `{S8, U8} × Rrr × {Identity, Bias, BiasRelu, BiasGelu, BiasSilu}`
//!
//! with the bias variants exercised against both `f32` and `i32` bias
//! element types. The kernels themselves don't exist in CUTLASS 4.2.0
//! (see ~/.claude/plans/baracuda-kernels-comprehensive.md §5); this
//! smoke test is the only correctness guarantee for them today.
//!
//! Each test verifies that the kernel matches a CPU reference for the
//! int32-accumulate → f32-compute → activation → saturating-cast
//! chain, indexing B as `b[k * ldb + j]` (row-major B, the RRR case).
//! Comparisons are tolerated to ±1 on the cast output: the f32-domain
//! activation chain (gelu, silu) is not bit-stable between libdevice
//! (CUDA) and Rust's libm equivalents.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    ActivationKind, BiasElement, EpilogueKind, IntGemmArgs, IntGemmDescriptor, IntGemmPlan,
    LayoutSku, MatrixMut, MatrixRef, PlanPreference, S8, U8, VectorRef, Workspace,
};

// ============================================================================
// CPU reference — RRR layout (row-major B).
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

// Saturating round-to-nearest-int casts matching the GPU's
// `__float2int_rn` (round-half-to-even, IEEE banker's). Rust's
// `f32::round` rounds half-away-from-zero — a different mode that
// disagrees with the GPU at half-integers whose floor is even (e.g.
// 12.5 → 13 vs 12). Use `round_ties_even` so the CPU reference is
// bit-exact against the kernel for Identity, and within ±1 for the
// bias / activation variants where f32 path divergence allows the rare
// off-by-one.
fn sat_cast_s8(x: f32) -> i8 {
    (x.round_ties_even() as i32).clamp(i8::MIN as i32, i8::MAX as i32) as i8
}

fn sat_cast_u8(x: f32) -> u8 {
    (x.round_ties_even() as i32).clamp(u8::MIN as i32, u8::MAX as i32) as u8
}

/// Reference int GEMM, RRR layout, with optional bias + activation.
///
/// Mirrors the kernel's compute order:
///   acc = Σ_k (i32) A[i, k] * (i32) B[k, j]
///   z   = alpha * (f32)acc + beta * (f32)C[i, j] + bias[j]
///   z'  = activation(z)
///
/// `A` row-major `[M, K]`. `B` row-major `[K, N]` — indexed
/// `b[k * ldb + j]`, the gmem layout the kernel consumes without
/// transposing.
#[allow(clippy::too_many_arguments)]
fn cpu_int_gemm_rrr<TIn: Copy + Into<i32>, TBias: Copy>(
    m: usize,
    n: usize,
    k: usize,
    a: &[TIn],
    lda: usize,
    b: &[TIn],
    ldb: usize,
    c: Option<(&[TIn], usize)>,
    bias: Option<&[TBias]>,
    bias_to_f32: fn(TBias) -> f32,
    alpha: f32,
    beta: f32,
    activation: Option<ActivationKind>,
    d_f32: &mut [f32],
    ldd: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut acc: i32 = 0;
            for kk in 0..k {
                let a_val: i32 = a[i * lda + kk].into();
                let b_val: i32 = b[kk * ldb + j].into();
                acc = acc.saturating_add(a_val.saturating_mul(b_val));
            }
            let mut z = alpha * (acc as f32);
            if let Some((c_buf, ldc)) = c {
                let c_val: i32 = c_buf[i * ldc + j].into();
                z += beta * (c_val as f32);
            }
            if let Some(bias_buf) = bias {
                z += bias_to_f32(bias_buf[j]);
            }
            d_f32[i * ldd + j] = apply_activation(activation, z);
        }
    }
}

// ============================================================================
// Per-SKU smoke harness
// ============================================================================

#[derive(Copy, Clone, Debug)]
struct IntSmokeCase {
    epilogue: EpilogueKind,
}

fn run_s8_smoke<BT: BiasElement + Copy + Default>(
    m: i32,
    n: i32,
    k: i32,
    case: IntSmokeCase,
    mk_bias: fn(usize) -> Vec<BT>,
    bias_to_f32: fn(BT) -> f32,
) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let mu = m as usize;
    let nu = n as usize;
    let ku = k as usize;

    let host_a: Vec<i8> = (0..(mu * ku))
        .map(|i| (((i as i32 * 7) % 15) - 7) as i8)
        .collect();
    let host_b: Vec<i8> = (0..(ku * nu))
        .map(|i| (((i as i32 * 11) % 13) - 6) as i8)
        .collect();
    let host_bias: Vec<BT> = if case.epilogue.requires_bias() {
        mk_bias(nu)
    } else {
        Vec::new()
    };

    let alpha: f32 = 0.125;
    let beta: f32 = 0.0;

    let mut host_d_ref = vec![0f32; mu * nu];
    cpu_int_gemm_rrr::<i8, BT>(
        mu, nu, ku,
        &host_a, ku,
        &host_b, nu,
        None,
        if case.epilogue.requires_bias() { Some(&host_bias) } else { None },
        bias_to_f32,
        alpha,
        beta,
        case.epilogue.activation(),
        &mut host_d_ref, nu,
    );

    let host_a_u: &[u8] = unsafe {
        core::slice::from_raw_parts(host_a.as_ptr() as *const u8, host_a.len())
    };
    let host_b_u: &[u8] = unsafe {
        core::slice::from_raw_parts(host_b.as_ptr() as *const u8, host_b.len())
    };
    let dev_a_bytes = DeviceBuffer::from_slice(&ctx, host_a_u).expect("upload A");
    let dev_b_bytes = DeviceBuffer::from_slice(&ctx, host_b_u).expect("upload B");
    let dev_a = dev_a_bytes.view_as::<S8>();
    let dev_b = dev_b_bytes.view_as::<S8>();
    let dev_bias_opt = if case.epilogue.requires_bias() {
        Some(DeviceBuffer::from_slice(&ctx, &host_bias).expect("upload bias"))
    } else {
        None
    };
    let mut dev_d: DeviceBuffer<S8> =
        DeviceBuffer::zeros(&ctx, mu * nu).expect("alloc D");

    let desc = IntGemmDescriptor {
        m, n, k,
        layout: LayoutSku::Rrr,
        epilogue: case.epilogue,
    };
    let plan = IntGemmPlan::<S8, BT>::select(&stream, &desc, PlanPreference::default())
        .expect("select s8 RRR plan");

    let args = IntGemmArgs::<S8, BT> {
        a: MatrixRef { data: dev_a, rows: m, cols: k, ld: k as i64 },
        b: MatrixRef { data: dev_b, rows: k, cols: n, ld: n as i64 },
        c: None,
        d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n as i64 },
        bias: dev_bias_opt.as_ref().map(|buf| VectorRef {
            data: buf.as_slice(),
            len: n,
            stride: 1,
        }),
        alpha,
        beta,
    };
    plan.run(&stream, Workspace::None, args).expect("s8 RRR GEMM run");
    stream.synchronize().expect("stream sync");

    let mut host_d_s8 = vec![S8(0); mu * nu];
    dev_d.copy_to_host(&mut host_d_s8).expect("download D");

    let mut mismatches = 0usize;
    let mut first_mismatch: Option<(usize, i8, i8, f32)> = None;
    for (idx, (got, &expected_f32)) in host_d_s8.iter().zip(host_d_ref.iter()).enumerate() {
        let expected = sat_cast_s8(expected_f32);
        if got.0 != expected && (got.0 as i32 - expected as i32).abs() > 1 {
            mismatches += 1;
            if first_mismatch.is_none() {
                first_mismatch = Some((idx, got.0, expected, expected_f32));
            }
        }
    }
    if mismatches > 0 {
        let (idx, got, expected, expected_f32) = first_mismatch.unwrap();
        panic!(
            "{mismatches} mismatches across {} cells for case {:?} (M={m} N={n} K={k}); \
             first @ {idx}: got s8={got} expected s8={expected} (f32 ref={expected_f32})",
            host_d_s8.len(),
            case,
        );
    }
}

fn run_u8_smoke<BT: BiasElement + Copy + Default>(
    m: i32,
    n: i32,
    k: i32,
    case: IntSmokeCase,
    mk_bias: fn(usize) -> Vec<BT>,
    bias_to_f32: fn(BT) -> f32,
) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let mu = m as usize;
    let nu = n as usize;
    let ku = k as usize;

    let host_a: Vec<u8> = (0..(mu * ku))
        .map(|i| ((i as u32 * 7) % 16) as u8)
        .collect();
    let host_b: Vec<u8> = (0..(ku * nu))
        .map(|i| ((i as u32 * 11) % 14) as u8)
        .collect();
    let host_bias: Vec<BT> = if case.epilogue.requires_bias() {
        mk_bias(nu)
    } else {
        Vec::new()
    };

    let alpha: f32 = 0.125;
    let beta: f32 = 0.0;

    let mut host_d_ref = vec![0f32; mu * nu];
    cpu_int_gemm_rrr::<u8, BT>(
        mu, nu, ku,
        &host_a, ku,
        &host_b, nu,
        None,
        if case.epilogue.requires_bias() { Some(&host_bias) } else { None },
        bias_to_f32,
        alpha,
        beta,
        case.epilogue.activation(),
        &mut host_d_ref, nu,
    );

    let dev_a_bytes = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload A");
    let dev_b_bytes = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload B");
    let dev_a = dev_a_bytes.view_as::<U8>();
    let dev_b = dev_b_bytes.view_as::<U8>();
    let dev_bias_opt = if case.epilogue.requires_bias() {
        Some(DeviceBuffer::from_slice(&ctx, &host_bias).expect("upload bias"))
    } else {
        None
    };
    let mut dev_d: DeviceBuffer<U8> =
        DeviceBuffer::zeros(&ctx, mu * nu).expect("alloc D");

    let desc = IntGemmDescriptor {
        m, n, k,
        layout: LayoutSku::Rrr,
        epilogue: case.epilogue,
    };
    let plan = IntGemmPlan::<U8, BT>::select(&stream, &desc, PlanPreference::default())
        .expect("select u8 RRR plan");

    let args = IntGemmArgs::<U8, BT> {
        a: MatrixRef { data: dev_a, rows: m, cols: k, ld: k as i64 },
        b: MatrixRef { data: dev_b, rows: k, cols: n, ld: n as i64 },
        c: None,
        d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n as i64 },
        bias: dev_bias_opt.as_ref().map(|buf| VectorRef {
            data: buf.as_slice(),
            len: n,
            stride: 1,
        }),
        alpha,
        beta,
    };
    plan.run(&stream, Workspace::None, args).expect("u8 RRR GEMM run");
    stream.synchronize().expect("stream sync");

    let mut host_d_u8 = vec![U8(0); mu * nu];
    dev_d.copy_to_host(&mut host_d_u8).expect("download D");

    let mut mismatches = 0usize;
    let mut first_mismatch: Option<(usize, u8, u8, f32)> = None;
    for (idx, (got, &expected_f32)) in host_d_u8.iter().zip(host_d_ref.iter()).enumerate() {
        let expected = sat_cast_u8(expected_f32);
        if got.0 != expected && (got.0 as i32 - expected as i32).abs() > 1 {
            mismatches += 1;
            if first_mismatch.is_none() {
                first_mismatch = Some((idx, got.0, expected, expected_f32));
            }
        }
    }
    if mismatches > 0 {
        let (idx, got, expected, expected_f32) = first_mismatch.unwrap();
        panic!(
            "{mismatches} mismatches across {} cells for case {:?} (M={m} N={n} K={k}); \
             first @ {idx}: got u8={got} expected u8={expected} (f32 ref={expected_f32})",
            host_d_u8.len(),
            case,
        );
    }
}

// ============================================================================
// Bias generators
// ============================================================================

fn mk_bias_f32(n: usize) -> Vec<f32> {
    (0..n).map(|j| ((j as f32 % 5.0) - 2.0) * 0.5).collect()
}

fn bias_to_f32_f32(b: f32) -> f32 {
    b
}

fn mk_bias_i32(n: usize) -> Vec<i32> {
    (0..n).map(|j| ((j as i32 % 5) - 2) * 2).collect()
}

fn bias_to_f32_i32(b: i32) -> f32 {
    b as f32
}

// ============================================================================
// Tests — Identity covers the four shapes that exercise tile-alignment
// edges (the smoking gun of the b2230a7 debug round); bias variants
// pin a single 128³ shape per SKU since the kernel body is the same.
// ============================================================================

// ---- S8 Identity — 4 problem shapes ----

#[test] #[ignore]
fn s8_rrr_identity_64_64_32() {
    run_s8_smoke::<f32>(
        64, 64, 32,
        IntSmokeCase { epilogue: EpilogueKind::Identity },
        mk_bias_f32, bias_to_f32_f32,
    );
}

#[test] #[ignore]
fn s8_rrr_identity_128_128_128() {
    run_s8_smoke::<f32>(
        128, 128, 128,
        IntSmokeCase { epilogue: EpilogueKind::Identity },
        mk_bias_f32, bias_to_f32_f32,
    );
}

#[test] #[ignore]
fn s8_rrr_identity_256_128_64() {
    run_s8_smoke::<f32>(
        256, 128, 64,
        IntSmokeCase { epilogue: EpilogueKind::Identity },
        mk_bias_f32, bias_to_f32_f32,
    );
}

#[test] #[ignore]
fn s8_rrr_identity_100_70_50() {
    run_s8_smoke::<f32>(
        100, 70, 50,
        IntSmokeCase { epilogue: EpilogueKind::Identity },
        mk_bias_f32, bias_to_f32_f32,
    );
}

// ---- S8 × f32 bias ----

const M: i32 = 128;
const N: i32 = 128;
const K: i32 = 128;

#[test] #[ignore]
fn s8_rrr_bias_f32() {
    run_s8_smoke::<f32>(M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::Bias },
        mk_bias_f32, bias_to_f32_f32);
}
#[test] #[ignore]
fn s8_rrr_bias_relu_f32() {
    run_s8_smoke::<f32>(M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::BiasRelu },
        mk_bias_f32, bias_to_f32_f32);
}
#[test] #[ignore]
fn s8_rrr_bias_gelu_f32() {
    run_s8_smoke::<f32>(M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::BiasGelu },
        mk_bias_f32, bias_to_f32_f32);
}
#[test] #[ignore]
fn s8_rrr_bias_silu_f32() {
    run_s8_smoke::<f32>(M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::BiasSilu },
        mk_bias_f32, bias_to_f32_f32);
}

// ---- S8 × i32 bias ----

#[test] #[ignore]
fn s8_rrr_bias_i32() {
    run_s8_smoke::<i32>(M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::Bias },
        mk_bias_i32, bias_to_f32_i32);
}
#[test] #[ignore]
fn s8_rrr_bias_relu_i32() {
    run_s8_smoke::<i32>(M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::BiasRelu },
        mk_bias_i32, bias_to_f32_i32);
}
#[test] #[ignore]
fn s8_rrr_bias_gelu_i32() {
    run_s8_smoke::<i32>(M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::BiasGelu },
        mk_bias_i32, bias_to_f32_i32);
}
#[test] #[ignore]
fn s8_rrr_bias_silu_i32() {
    run_s8_smoke::<i32>(M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::BiasSilu },
        mk_bias_i32, bias_to_f32_i32);
}

// ---- U8 Identity ----

#[test] #[ignore]
fn u8_rrr_identity_128_128_128() {
    run_u8_smoke::<f32>(M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::Identity },
        mk_bias_f32, bias_to_f32_f32);
}

// ---- U8 × f32 bias ----

#[test] #[ignore]
fn u8_rrr_bias_f32() {
    run_u8_smoke::<f32>(M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::Bias },
        mk_bias_f32, bias_to_f32_f32);
}
#[test] #[ignore]
fn u8_rrr_bias_relu_f32() {
    run_u8_smoke::<f32>(M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::BiasRelu },
        mk_bias_f32, bias_to_f32_f32);
}
#[test] #[ignore]
fn u8_rrr_bias_gelu_f32() {
    run_u8_smoke::<f32>(M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::BiasGelu },
        mk_bias_f32, bias_to_f32_f32);
}
#[test] #[ignore]
fn u8_rrr_bias_silu_f32() {
    run_u8_smoke::<f32>(M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::BiasSilu },
        mk_bias_f32, bias_to_f32_f32);
}

// ---- U8 × i32 bias ----

#[test] #[ignore]
fn u8_rrr_bias_i32() {
    run_u8_smoke::<i32>(M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::Bias },
        mk_bias_i32, bias_to_f32_i32);
}
#[test] #[ignore]
fn u8_rrr_bias_relu_i32() {
    run_u8_smoke::<i32>(M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::BiasRelu },
        mk_bias_i32, bias_to_f32_i32);
}
#[test] #[ignore]
fn u8_rrr_bias_gelu_i32() {
    run_u8_smoke::<i32>(M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::BiasGelu },
        mk_bias_i32, bias_to_f32_i32);
}
#[test] #[ignore]
fn u8_rrr_bias_silu_i32() {
    run_u8_smoke::<i32>(M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::BiasSilu },
        mk_bias_i32, bias_to_f32_i32);
}
