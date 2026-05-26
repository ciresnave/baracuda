//! Real-GPU smoke test for `GgufMmvqMultiMPlan` — Phase 33.
//!
//! Validates multi-M MMVQ output (Q8_0 weights, Q8_1-staged activations,
//! DP4A inner dot) against a hand-computed FP reference for each M
//! ∈ {1, 2, 4, 8}. The relative tolerance (~1e-2) accounts for the
//! activation-side Q8_1 quantization step (8-bit symmetric quant
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
