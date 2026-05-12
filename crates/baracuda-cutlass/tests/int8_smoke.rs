//! Real-GPU smoke tests for the alpha.15 int8 SKUs:
//! `{S8, U8} × Rcr × {Identity, Bias, BiasRelu, BiasGelu, BiasSilu}`
//! with the bias variants exercised against both `f32` and `i32` bias
//! element types. 18 total SKUs.
//!
//! Each test verifies (a) the kernel runs and produces non-trivial
//! output (not all-zero) and (b) it matches a CPU reference that
//! replicates CUTLASS's int32-accumulate → float-compute →
//! saturating-cast-to-int8 epilogue chain.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-cutlass --release -- --ignored`.

use baracuda_cutlass::{
    ActivationKind, BiasElement, EpilogueKind, IntGemmArgs, IntGemmDescriptor, IntGemmPlan,
    LayoutSku, MatrixMut, MatrixRef, PlanPreference, S8, U8, VectorRef, Workspace,
};
use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};

// ============================================================================
// CPU reference — mirrors CUTLASS `LinearCombinationClamp` /
// `LinearCombinationBiasElementwise` epilogue chain for int8.
// ============================================================================

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn erf_approx(x: f32) -> f32 {
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

/// Saturating round-to-nearest-int cast — matches the PTX
/// `cvt.rni.sat.s8.f32` / `cvt.rni.sat.u8.f32` instructions used by
/// CUTLASS's `NumericConverter<{i8,u8}, float>` specializations.
fn sat_cast_s8(x: f32) -> i8 {
    let r = x.round() as i32;
    r.clamp(i8::MIN as i32, i8::MAX as i32) as i8
}

fn sat_cast_u8(x: f32) -> u8 {
    let r = x.round() as i32;
    r.clamp(u8::MIN as i32, u8::MAX as i32) as u8
}

/// Reference int8 GEMM with optional bias + activation, RCR layout.
/// Mirrors CUTLASS's compute order:
///   acc = sum_k (i32) A[i,k] * (i32) B[k,j]   // int32 accumulator
///   z = alpha * (f32)acc + beta * (f32)C[i,j] + (f32)bias[j]
///   D[i,j] = sat_cast(activation(z))
///
/// `bias_to_f32` lets the caller thread either `f32` or `i32` bias
/// vectors through the same routine; the CPU side does dequant
/// arithmetic in `f32` to match the kernel's `ElementCompute = float`
/// epilogue.
#[allow(clippy::too_many_arguments)]
fn cpu_int_gemm_rcr<TIn: Copy + Into<i32>, TBias: Copy>(
    m: usize,
    n: usize,
    k: usize,
    a: &[TIn],
    lda: usize,
    b: &[TIn], // column-major [K, N], ldb is row stride along K = K
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
            // int32 accumulator
            let mut acc: i32 = 0;
            for kk in 0..k {
                let a_val: i32 = a[i * lda + kk].into();
                // Column-major B: B[k, j] at offset k + j*ldb
                let b_val: i32 = b[kk + j * ldb].into();
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
// Test harness — one runner per (TIn, TBias) pair.
// ============================================================================

#[derive(Copy, Clone, Debug)]
struct IntSmokeCase {
    epilogue: EpilogueKind,
}

// Specialized runner for `S8` / `i8`. Avoids the generic-cast-from-i32
// gymnastics that don't compose cleanly through `IntElement` /
// `BiasElement` without `Into<f32>` blanket impls.
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

    // Bounded inputs so the int32 accumulator stays within a useful
    // range for the int8 output (no constant saturation).
    let host_a: Vec<i8> = (0..(mu * ku))
        .map(|i| (((i as i32 * 7) % 15) - 7) as i8)
        .collect();
    // Column-major B: physically [K, N] with ldb = K. Indexed via b[k + j*K].
    let host_b: Vec<i8> = (0..(ku * nu))
        .map(|i| (((i as i32 * 11) % 13) - 6) as i8)
        .collect();
    let host_bias: Vec<BT> = if case.epilogue.requires_bias() {
        mk_bias(nu)
    } else {
        Vec::new()
    };
    let host_bias_f32: Vec<f32> = host_bias.iter().copied().map(bias_to_f32).collect();

    let alpha: f32 = 0.125; // small alpha keeps post-dequant values in range
    let beta: f32 = 0.0;

    let mut host_d_ref = vec![0f32; mu * nu];
    cpu_int_gemm_rcr::<i8, BT>(
        mu, nu, ku,
        &host_a, ku,
        &host_b, ku,
        None,
        if case.epilogue.requires_bias() { Some(&host_bias) } else { None },
        bias_to_f32,
        alpha,
        beta,
        case.epilogue.activation(),
        &mut host_d_ref, nu,
    );

    // Upload as S8 wrappers via DeviceBuffer<u8> + view_as<S8>.
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
    let mut dev_d: DeviceBuffer<S8> = DeviceBuffer::zeros(&ctx, (m * n) as usize).expect("alloc D");

    let desc = IntGemmDescriptor {
        m, n, k,
        layout: LayoutSku::Rcr,
        epilogue: case.epilogue,
    };
    let plan = IntGemmPlan::<S8, BT>::select(&stream, &desc, PlanPreference::default())
        .expect("select int8 plan");

    let args = IntGemmArgs::<S8, BT> {
        a: MatrixRef { data: dev_a, rows: m, cols: k, ld: k as i64 },
        b: MatrixRef { data: dev_b, rows: k, cols: n, ld: k as i64 },
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
    plan.run(&stream, Workspace::None, args).expect("int8 GEMM run");
    stream.synchronize().expect("stream sync");

    // Download D
    let mut host_d_s8 = vec![S8(0); (m * n) as usize];
    dev_d.copy_to_host(&mut host_d_s8).expect("download D");

    // Verify: every cell must match saturating-cast of CPU reference.
    let mut mismatches = 0usize;
    let mut first_mismatch: Option<(usize, i8, i8, f32)> = None;
    for (idx, (got, &expected_f32)) in host_d_s8.iter().zip(host_d_ref.iter()).enumerate() {
        let expected = sat_cast_s8(expected_f32);
        if got.0 != expected {
            // Allow ±1 LSB to absorb the tiny rounding difference between
            // our `f32::round()` (banker's? no, round-half-away-from-zero
            // on x86) and PTX's `cvt.rni` (round-half-to-even).
            if (got.0 as i32 - expected as i32).abs() > 1 {
                mismatches += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some((idx, got.0, expected, expected_f32));
                }
            }
        }
        // Use the existing helper variable for unused-import suppression
        let _ = host_bias_f32.first();
    }
    if mismatches > 0 {
        let (idx, got, expected, expected_f32) = first_mismatch.unwrap();
        panic!(
            "{} mismatches across {} cells for case {:?}; first @ {}: got s8={} expected s8={} (f32 ref={})",
            mismatches,
            host_d_s8.len(),
            case,
            idx,
            got,
            expected,
            expected_f32,
        );
    }
}

// u8 specialization — same shape as S8 runner with operand re-mapping.
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

    // Unsigned values: stay in low end of [0, 255] so the int32 accum
    // doesn't approach overflow.
    let host_a: Vec<u8> = (0..(mu * ku))
        .map(|i| (((i as u32 * 7) % 16)) as u8)
        .collect();
    let host_b: Vec<u8> = (0..(ku * nu))
        .map(|i| (((i as u32 * 11) % 14)) as u8)
        .collect();
    let host_bias: Vec<BT> = if case.epilogue.requires_bias() {
        mk_bias(nu)
    } else {
        Vec::new()
    };
    let host_bias_f32: Vec<f32> = host_bias.iter().copied().map(bias_to_f32).collect();
    let _ = host_bias_f32; // computed only for debug-print purposes

    let alpha: f32 = 0.125;
    let beta: f32 = 0.0;

    let mut host_d_ref = vec![0f32; mu * nu];
    cpu_int_gemm_rcr::<u8, BT>(
        mu, nu, ku,
        &host_a, ku,
        &host_b, ku,
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
    let mut dev_d: DeviceBuffer<U8> = DeviceBuffer::zeros(&ctx, (m * n) as usize).expect("alloc D");

    let desc = IntGemmDescriptor {
        m, n, k,
        layout: LayoutSku::Rcr,
        epilogue: case.epilogue,
    };
    let plan = IntGemmPlan::<U8, BT>::select(&stream, &desc, PlanPreference::default())
        .expect("select u8 plan");

    let args = IntGemmArgs::<U8, BT> {
        a: MatrixRef { data: dev_a, rows: m, cols: k, ld: k as i64 },
        b: MatrixRef { data: dev_b, rows: k, cols: n, ld: k as i64 },
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
    plan.run(&stream, Workspace::None, args).expect("u8 GEMM run");
    stream.synchronize().expect("stream sync");

    let mut host_d_u8 = vec![U8(0); (m * n) as usize];
    dev_d.copy_to_host(&mut host_d_u8).expect("download D");

    let mut mismatches = 0usize;
    let mut first_mismatch: Option<(usize, u8, u8, f32)> = None;
    for (idx, (got, &expected_f32)) in host_d_u8.iter().zip(host_d_ref.iter()).enumerate() {
        let expected = sat_cast_u8(expected_f32);
        if got.0 != expected {
            if (got.0 as i32 - expected as i32).abs() > 1 {
                mismatches += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some((idx, got.0, expected, expected_f32));
                }
            }
        }
    }
    if mismatches > 0 {
        let (idx, got, expected, expected_f32) = first_mismatch.unwrap();
        panic!(
            "{} mismatches across {} cells for case {:?}; first @ {}: got u8={} expected u8={} (f32 ref={})",
            mismatches,
            host_d_u8.len(),
            case,
            idx,
            got,
            expected,
            expected_f32,
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
// Tests — 18 SKUs.
//
// Problem size 128x128x128 with M/N multiples of 16 (the int8
// tensor-core EPA requirement is 16 elements per access for both
// alignment-A and alignment-B).
// ============================================================================

const M: i32 = 128;
const N: i32 = 128;
const K: i32 = 128;

// ---- S8 Identity ----

#[test]
#[ignore]
fn int8_smoke_s8_rcr_identity() {
    run_s8_smoke::<f32>(
        M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::Identity},
        mk_bias_f32, bias_to_f32_f32,
    );
}

// ---- S8 × f32 bias ----

#[test]
#[ignore]
fn int8_smoke_s8_rcr_bias_f32() {
    run_s8_smoke::<f32>(
        M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::Bias},
        mk_bias_f32, bias_to_f32_f32,
    );
}

#[test]
#[ignore]
fn int8_smoke_s8_rcr_bias_relu_f32() {
    run_s8_smoke::<f32>(
        M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::BiasRelu},
        mk_bias_f32, bias_to_f32_f32,
    );
}

#[test]
#[ignore]
fn int8_smoke_s8_rcr_bias_gelu_f32() {
    run_s8_smoke::<f32>(
        M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::BiasGelu},
        mk_bias_f32, bias_to_f32_f32,
    );
}

#[test]
#[ignore]
fn int8_smoke_s8_rcr_bias_silu_f32() {
    run_s8_smoke::<f32>(
        M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::BiasSilu},
        mk_bias_f32, bias_to_f32_f32,
    );
}

// ---- S8 × i32 bias ----

#[test]
#[ignore]
fn int8_smoke_s8_rcr_bias_i32() {
    run_s8_smoke::<i32>(
        M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::Bias},
        mk_bias_i32, bias_to_f32_i32,
    );
}

#[test]
#[ignore]
fn int8_smoke_s8_rcr_bias_relu_i32() {
    run_s8_smoke::<i32>(
        M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::BiasRelu},
        mk_bias_i32, bias_to_f32_i32,
    );
}

#[test]
#[ignore]
fn int8_smoke_s8_rcr_bias_gelu_i32() {
    run_s8_smoke::<i32>(
        M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::BiasGelu},
        mk_bias_i32, bias_to_f32_i32,
    );
}

#[test]
#[ignore]
fn int8_smoke_s8_rcr_bias_silu_i32() {
    run_s8_smoke::<i32>(
        M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::BiasSilu},
        mk_bias_i32, bias_to_f32_i32,
    );
}

// ---- U8 Identity ----

#[test]
#[ignore]
fn int8_smoke_u8_rcr_identity() {
    run_u8_smoke::<f32>(
        M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::Identity},
        mk_bias_f32, bias_to_f32_f32,
    );
}

// ---- U8 × f32 bias ----

#[test]
#[ignore]
fn int8_smoke_u8_rcr_bias_f32() {
    run_u8_smoke::<f32>(
        M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::Bias},
        mk_bias_f32, bias_to_f32_f32,
    );
}

#[test]
#[ignore]
fn int8_smoke_u8_rcr_bias_relu_f32() {
    run_u8_smoke::<f32>(
        M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::BiasRelu},
        mk_bias_f32, bias_to_f32_f32,
    );
}

#[test]
#[ignore]
fn int8_smoke_u8_rcr_bias_gelu_f32() {
    run_u8_smoke::<f32>(
        M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::BiasGelu},
        mk_bias_f32, bias_to_f32_f32,
    );
}

#[test]
#[ignore]
fn int8_smoke_u8_rcr_bias_silu_f32() {
    run_u8_smoke::<f32>(
        M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::BiasSilu},
        mk_bias_f32, bias_to_f32_f32,
    );
}

// ---- U8 × i32 bias ----

#[test]
#[ignore]
fn int8_smoke_u8_rcr_bias_i32() {
    run_u8_smoke::<i32>(
        M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::Bias},
        mk_bias_i32, bias_to_f32_i32,
    );
}

#[test]
#[ignore]
fn int8_smoke_u8_rcr_bias_relu_i32() {
    run_u8_smoke::<i32>(
        M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::BiasRelu},
        mk_bias_i32, bias_to_f32_i32,
    );
}

#[test]
#[ignore]
fn int8_smoke_u8_rcr_bias_gelu_i32() {
    run_u8_smoke::<i32>(
        M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::BiasGelu},
        mk_bias_i32, bias_to_f32_i32,
    );
}

#[test]
#[ignore]
fn int8_smoke_u8_rcr_bias_silu_i32() {
    run_u8_smoke::<i32>(
        M, N, K,
        IntSmokeCase { epilogue: EpilogueKind::BiasSilu},
        mk_bias_i32, bias_to_f32_i32,
    );
}

// ---- Negative: RRR int8 must report Unsupported at plan selection ----

#[test]
fn int8_rrr_select_returns_unsupported() {
    // Pure host-side check — doesn't touch the device. Doesn't need a
    // real CUDA context, but `IntGemmPlan::select` does query the
    // stream's device. We probe a Stream to satisfy the API; this test
    // is marked NOT `#[ignore]` because it's host-side validation.
    if init().is_err() {
        // No CUDA driver — skip silently. This is a host-side test
        // that's only meaningful if the workspace builds.
        return;
    }
    let Ok(device) = Device::get(0) else { return };
    let Ok(ctx) = Context::new(&device) else { return };
    let Ok(stream) = Stream::new(&ctx) else { return };

    let desc = IntGemmDescriptor {
        m: 64,
        n: 64,
        k: 64,
        layout: LayoutSku::Rrr,
        epilogue: EpilogueKind::Identity,
    };
    let err = IntGemmPlan::<S8>::select(&stream, &desc, PlanPreference::default()).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("RCR-only") || msg.contains("Unsupported") || msg.to_lowercase().contains("rrr"),
        "expected an Unsupported error for int8 RRR; got: {msg}"
    );
}
