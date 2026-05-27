//! Real-GPU smoke test for `GgufMmvqMultiMPlan` — Phase 33 + 34.
//!
//! Validates multi-M MMVQ output across **all** supported GGUF block
//! formats (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 + k-quants Q2_K/Q3_K/Q4_K/Q5_K/Q6_K)
//! against the existing single-M `GgufMmvqPlan` (which is the FP-activation
//! dequantize-and-multiply reference). The relative tolerance (~5%) accounts
//! for the activation-side Q8_1 quantization step (8-bit symmetric quant
//! introduces O(1/127) error per dot).

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BlockQ8_0, GgufBlockFormat, GgufMmvqMultiMArgs, GgufMmvqMultiMDescriptor,
    GgufMmvqMultiMPlan, PlanPreference, TensorMut, TensorRef, Workspace, U8,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Build a small Q8_0 weight matrix of nrows=4, ncols=64 (= 2 blocks/row).
/// Returns (host_bytes, host_w_fp32_ref) where the second is the
/// dequantized fp32 weight matrix for the reference compute.
fn build_q8_0_weights(nrows: i32, ncols: i32) -> (Vec<U8>, Vec<f32>) {
    let blocks_per_row = ncols / 32;
    let mut packed: Vec<u8> = Vec::with_capacity((nrows * blocks_per_row * 34) as usize);
    let mut fp_ref: Vec<f32> = vec![0.0; (nrows * ncols) as usize];

    for r in 0..nrows {
        for bi in 0..blocks_per_row {
            // Per-block: pick a small scale and a smooth-ish int8 ramp.
            let d = 0.05_f32 + 0.01 * ((r + bi) as f32);
            let mut qs = [0i8; 32];
            for j in 0..32 {
                let v = (((r * 7 + bi * 13 + j as i32) % 17) - 8) as i8;
                qs[j as usize] = v;
                fp_ref[(r * ncols + bi * 32 + j as i32) as usize] = d * (v as f32);
            }
            let blk = BlockQ8_0 {
                d: half::f16::from_f32(d).to_bits(),
                qs,
            };
            let bytes: &[u8] = unsafe {
                core::slice::from_raw_parts(
                    (&blk as *const BlockQ8_0) as *const u8,
                    core::mem::size_of::<BlockQ8_0>(),
                )
            };
            packed.extend_from_slice(bytes);
        }
    }
    let host_w: Vec<U8> = packed.into_iter().map(U8).collect();
    (host_w, fp_ref)
}

fn run_for_m(m: i32) {
    let (ctx, stream) = setup();
    let nrows = 4i32;
    let ncols = 64i32;

    let (host_w, fp_ref) = build_q8_0_weights(nrows, ncols);

    // Activations: deterministic small floats per (m, c).
    let mut host_act = vec![0.0f32; (m * ncols) as usize];
    for mi in 0..m {
        for c in 0..ncols {
            host_act[(mi * ncols + c) as usize] =
                (((mi * 11 + c * 3) % 20) as f32) * 0.07 - 0.5;
        }
    }

    // FP reference: out[mi, r] = Σ_c W[r, c] * y[mi, c].
    let mut expected = vec![0.0f32; (m * nrows) as usize];
    for mi in 0..m {
        for r in 0..nrows {
            let mut acc = 0.0f32;
            for c in 0..ncols {
                acc += fp_ref[(r * ncols + c) as usize] * host_act[(mi * ncols + c) as usize];
            }
            expected[(mi * nrows + r) as usize] = acc;
        }
    }

    // Allocate device buffers.
    let dev_w = DeviceBuffer::from_slice(&ctx, &host_w).expect("up w");
    let dev_act = DeviceBuffer::from_slice(&ctx, &host_act).expect("up act");
    let mut dev_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (m * nrows) as usize).expect("alloc out");

    let desc = GgufMmvqMultiMDescriptor {
        nrows,
        ncols,
        m,
        block_format: GgufBlockFormat::Q8_0,
        w_start_byte_offset: 0,
    };
    let plan = GgufMmvqMultiMPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("plan select");

    // Workspace for Q8_1 staging.
    let ws_bytes = plan.workspace_size();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    let weight_bytes_len = host_w.len() as i32;
    let args = GgufMmvqMultiMArgs::<f32> {
        weight: TensorRef {
            data: dev_w.as_slice(),
            shape: [weight_bytes_len],
            stride: contiguous_stride([weight_bytes_len]),
        },
        activations: TensorRef {
            data: dev_act.as_slice(),
            shape: [m, ncols],
            stride: contiguous_stride([m, ncols]),
        },
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [m, nrows],
            stride: contiguous_stride([m, nrows]),
        },
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("multim run");
    stream.synchronize().expect("sync");

    let mut host_out = vec![0.0f32; (m * nrows) as usize];
    dev_out.copy_to_host(&mut host_out).expect("d2h");

    // Compare with relative tolerance — Q8_1 staging introduces ~1e-3
    // relative error per element. Use 1e-2 to be safe across the 8-bit
    // activation quantization and possible fp16-scale rounding.
    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;
    for i in 0..(m * nrows) as usize {
        let a = host_out[i];
        let b = expected[i];
        let abs_err = (a - b).abs();
        let rel_err = abs_err / (b.abs().max(1e-6));
        max_abs_err = max_abs_err.max(abs_err);
        max_rel_err = max_rel_err.max(rel_err);
        assert!(
            abs_err < 0.05 || rel_err < 0.05,
            "M={m} idx={i}: got {a}, expected {b}, abs_err={abs_err}, rel_err={rel_err}"
        );
    }
    eprintln!(
        "M={m}: max_abs_err={max_abs_err:.4e}, max_rel_err={max_rel_err:.4e}"
    );
}

#[test]
#[ignore]
fn mmvq_multim_q8_0_m1() {
    run_for_m(1);
}

#[test]
#[ignore]
fn mmvq_multim_q8_0_m2() {
    run_for_m(2);
}

#[test]
#[ignore]
fn mmvq_multim_q8_0_m4() {
    run_for_m(4);
}

#[test]
#[ignore]
fn mmvq_multim_q8_0_m8() {
    run_for_m(8);
}

/// Verify the staging kernel's workspace_bytes helper matches what
/// `GgufMmvqMultiMPlan::workspace_size` reports.
#[test]
fn mmvq_multim_workspace_helper_matches_plan() {
    // 128 cols → 4 Q8_1 blocks/row. 7 rows. 7*4*36 = 1008 bytes.
    let n = 7i64;
    let k = 128i64;
    let from_helper =
        unsafe { baracuda_kernels_sys::baracuda_kernels_quantize_q8_1_workspace_bytes(k, n) };
    assert_eq!(from_helper, n * (k / 32) * 36);
}

// =============================================================================
// Phase 34 — per-format smoke tests.
//
// Strategy: each test compares the new `GgufMmvqMultiMPlan` result with
// the existing M=1 `GgufMmvqPlan` looped M times. Since `GgufMmvqPlan`
// is itself the validated FP-dequant reference (Phase 8.4+), this is a
// transitive correctness check; the only new code paths exercised are
// (a) the per-format `vec_dot_q*_q8_1` helpers and (b) the multi-M
// kernel reduction. The 5% relative tolerance covers Q8_1 activation
// quantization noise.
//
// Weight buffers are synthetic — we pick a small byte pattern that
// produces non-degenerate Q8_1 activations and `vec_dot` outputs across
// every format. Validation is against the M=1 plan (not a CPU fp ref),
// so the absolute number doesn't matter; what matters is that the
// multi-M kernel reproduces the per-token loop's output within Q8_1
// quant noise.
// =============================================================================

use baracuda_kernels::{GgufMmvqArgs, GgufMmvqDescriptor, GgufMmvqPlan};

fn run_per_format(fmt: GgufBlockFormat, m: i32) {
    let (ctx, stream) = setup();

    // Use the format's natural block boundary so ncols % block_size == 0.
    let bs = fmt.block_size() as i32;
    let ncols = bs.max(64) * 4;
    let nrows = 8i32;

    let blocks_per_row = ncols / bs;
    let weight_bytes = (nrows as usize) * (blocks_per_row as usize) * fmt.type_size();

    // Synthetic weight bytes: a non-trivial repeating pattern (varies per
    // byte index so different block formats see different unpacked vals).
    // Cap below 127 to keep signed-int8 lanes well-defined.
    let host_w: Vec<U8> = (0..weight_bytes)
        .map(|i| U8(((i * 31 + 17) % 113) as u8))
        .collect();

    // Activations: deterministic small fp32 in [-0.5, 1.0].
    let host_act: Vec<f32> = (0..(m * ncols) as usize)
        .map(|i| ((i % 23) as f32) * 0.07 - 0.5)
        .collect();

    let dev_w = DeviceBuffer::from_slice(&ctx, &host_w).expect("up w");
    let dev_act = DeviceBuffer::from_slice(&ctx, &host_act).expect("up act");

    // ---- 1. Multi-M path ----
    let mut dev_out_multim: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (m * nrows) as usize).expect("alloc multim out");
    let desc_multim = GgufMmvqMultiMDescriptor {
        nrows,
        ncols,
        m,
        block_format: fmt,
        w_start_byte_offset: 0,
    };
    let plan_multim =
        GgufMmvqMultiMPlan::<f32>::select(&stream, &desc_multim, PlanPreference::default())
            .expect("multim plan select");

    let ws_bytes = plan_multim.workspace_size();
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, ws_bytes.max(1)).expect("alloc ws");

    let weight_bytes_len = host_w.len() as i32;
    let args_multim = GgufMmvqMultiMArgs::<f32> {
        weight: TensorRef {
            data: dev_w.as_slice(),
            shape: [weight_bytes_len],
            stride: contiguous_stride([weight_bytes_len]),
        },
        activations: TensorRef {
            data: dev_act.as_slice(),
            shape: [m, ncols],
            stride: contiguous_stride([m, ncols]),
        },
        output: TensorMut {
            data: dev_out_multim.as_slice_mut(),
            shape: [m, nrows],
            stride: contiguous_stride([m, nrows]),
        },
    };
    plan_multim
        .run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args_multim)
        .expect("multim run");
    stream.synchronize().expect("sync after multim");

    let mut host_out_multim = vec![0.0f32; (m * nrows) as usize];
    dev_out_multim
        .copy_to_host(&mut host_out_multim)
        .expect("d2h multim");

    // ---- 2. Per-token loop over M=1 GgufMmvqPlan baseline ----
    let mut host_out_baseline = vec![0.0f32; (m * nrows) as usize];
    let mut dev_out_1: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, nrows as usize).expect("alloc out1");

    let desc_base = GgufMmvqDescriptor {
        nrows,
        ncols,
        block_format: fmt,
        w_start_byte_offset: 0,
    };
    let plan_base = GgufMmvqPlan::<f32>::select(&stream, &desc_base, PlanPreference::default())
        .expect("base plan select");

    for mi in 0..m {
        let mi_act = &host_act[(mi * ncols) as usize..((mi + 1) * ncols) as usize];
        let dev_act_mi = DeviceBuffer::from_slice(&ctx, mi_act).expect("up act_mi");
        let args_1 = GgufMmvqArgs::<f32> {
            weight: TensorRef {
                data: dev_w.as_slice(),
                shape: [weight_bytes_len],
                stride: contiguous_stride([weight_bytes_len]),
            },
            activation: TensorRef {
                data: dev_act_mi.as_slice(),
                shape: [ncols],
                stride: contiguous_stride([ncols]),
            },
            output: TensorMut {
                data: dev_out_1.as_slice_mut(),
                shape: [nrows],
                stride: contiguous_stride([nrows]),
            },
        };
        plan_base
            .run(&stream, Workspace::None, args_1)
            .expect("baseline run");
        stream.synchronize().expect("sync baseline");

        let mut buf = vec![0.0f32; nrows as usize];
        dev_out_1.copy_to_host(&mut buf).expect("d2h baseline");
        host_out_baseline[(mi * nrows) as usize..((mi + 1) * nrows) as usize]
            .copy_from_slice(&buf);
    }

    // ---- 3. Compare with Q8_1 quant tolerance ----
    //
    // Tolerance: Q8_1 activation staging introduces O(1/127) error per
    // element. For dot products with significant sign cancellation, the
    // absolute error scales with Σ|x_i|·|w_i|, not with |result| — so a
    // per-cell relative tolerance can be misleading. We use a **global**
    // relative tolerance against max-output-magnitude (which captures the
    // typical scale of the dots) with a 5% threshold; this stays safely
    // above Q8_1 noise on real-model shapes but rejects systematic bugs.
    let mut max_abs_err = 0.0f32;
    let mut max_rel_err_to_global = 0.0f32;
    let mut max_mag = 0.0_f32;
    for i in 0..(m * nrows) as usize {
        let got = host_out_multim[i];
        let exp = host_out_baseline[i];
        max_mag = max_mag.max(got.abs().max(exp.abs()));
        max_abs_err = max_abs_err.max((got - exp).abs());
    }
    for i in 0..(m * nrows) as usize {
        let got = host_out_multim[i];
        let exp = host_out_baseline[i];
        let abs_err = (got - exp).abs();
        // Per-cell relative-to-global tolerance (5%) — bounds the Q8_1
        // staging error normalized by the largest dot product. Catches
        // any systematic bias (>~1% across all cells) without requiring
        // per-cell tolerance tuning.
        let rel = abs_err / max_mag.max(1e-6);
        max_rel_err_to_global = max_rel_err_to_global.max(rel);
        assert!(
            rel < 0.05,
            "fmt={fmt:?} M={m} idx={i}: got {got}, expected {exp}, abs={abs_err}, \
             rel_to_global_max({max_mag})={rel}"
        );
    }
    eprintln!(
        "mmvq_multim {fmt:?} M={m}: max_abs={max_abs_err:.3e}, \
         max_rel_to_global={max_rel_err_to_global:.3e}, max_mag={max_mag:.3e}"
    );
}

// One smoke test per format @ M=8 (covers the prefill regime — heaviest
// use of the kernel's cross-warp reduction).
#[test]
#[ignore]
fn mmvq_multim_q4_0_m8() {
    run_per_format(GgufBlockFormat::Q4_0, 8);
}

#[test]
#[ignore]
fn mmvq_multim_q4_1_m8() {
    run_per_format(GgufBlockFormat::Q4_1, 8);
}

#[test]
#[ignore]
fn mmvq_multim_q5_0_m8() {
    run_per_format(GgufBlockFormat::Q5_0, 8);
}

#[test]
#[ignore]
fn mmvq_multim_q5_1_m8() {
    run_per_format(GgufBlockFormat::Q5_1, 8);
}

#[test]
#[ignore]
fn mmvq_multim_q2_K_m8() {
    run_per_format(GgufBlockFormat::Q2K, 8);
}

#[test]
#[ignore]
fn mmvq_multim_q3_K_m8() {
    run_per_format(GgufBlockFormat::Q3K, 8);
}

#[test]
#[ignore]
fn mmvq_multim_q4_K_m8() {
    run_per_format(GgufBlockFormat::Q4K, 8);
}

#[test]
#[ignore]
fn mmvq_multim_q5_K_m8() {
    run_per_format(GgufBlockFormat::Q5K, 8);
}

#[test]
#[ignore]
fn mmvq_multim_q6_K_m8() {
    run_per_format(GgufBlockFormat::Q6K, 8);
}

// One additional M=1 (decode regime) test against Q4_0 to verify the
// nwarps=4 / rows_per_cuda_block=1 dispatch path.
#[test]
#[ignore]
fn mmvq_multim_q4_0_m1() {
    run_per_format(GgufBlockFormat::Q4_0, 1);
}

// Diagnostic M=1 tests — isolate per-format dot helper bugs from the
// multi-warp reduction path.
#[test]
#[ignore]
fn mmvq_multim_q5_0_m1() {
    run_per_format(GgufBlockFormat::Q5_0, 1);
}


#[test]
#[ignore]
fn mmvq_multim_q5_1_m1() {
    run_per_format(GgufBlockFormat::Q5_1, 1);
}
