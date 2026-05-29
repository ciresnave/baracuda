//! Real-GPU smoke test for `Nf4MmvqMultiMPlan` — Phase 53
//! (bitsandbytes NF4 vendor, M ∈ {1, 2, 4, 8} batched-decode GEMV).
//!
//! Computes `M` decode steps against one shared NF4 weight matrix in a
//! single kernel launch. The reference is the M=1 plan run M times
//! sequentially — they must produce equivalent output (small fp16
//! rounding tolerance only; the actual codebook + absmax path is
//! identical).
//!
//! `#[ignore]` by default — requires a real CUDA device + the
//! `bnb_nf4` cargo feature.

#![cfg(feature = "bnb_nf4")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::quantize::nf4::nf4_pack_weight;
use baracuda_kernels::{
    contiguous_stride, Nf4Descriptor, Nf4MmvqArgs, Nf4MmvqMultiMArgs, Nf4MmvqMultiMDescriptor,
    Nf4MmvqMultiMPlan, Nf4MmvqPlan, PlanPreference, TensorMut, TensorRef, Workspace, U8,
};
use half::f16;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn run_for_m(m_test: i32) {
    let (ctx, stream) = setup();
    let n: usize = 16;
    let k: usize = 128;
    let block_size: usize = 64;

    // Build a small NF4 weight matrix.
    let mut host_w = vec![0.0f32; n * k];
    for r in 0..n {
        for c in 0..k {
            let scale = 0.08 + (r as f32) * 0.03;
            host_w[r * k + c] = scale * (((c as f32) - (k as f32) * 0.5) / (k as f32) * 2.0);
        }
    }
    let (packed, absmax) = nf4_pack_weight(&host_w, n, k, block_size);
    let packed_u8: Vec<U8> = packed.iter().map(|b| U8(*b)).collect();
    let weight_bytes_len = packed_u8.len() as i32;
    let absmax_len = absmax.len() as i32;

    // Build M activation rows.
    let mut host_y_f16 = vec![f16::ZERO; (m_test as usize) * k];
    for mi in 0..m_test as usize {
        for c in 0..k {
            let v = (((mi * 11 + c * 3) % 41) as f32) * 0.02 - 0.4;
            host_y_f16[mi * k + c] = f16::from_f32(v);
        }
    }

    // Upload weight + absmax once (shared between paths).
    let dev_w = DeviceBuffer::from_slice(&ctx, &packed_u8).expect("up w");
    let dev_amax = DeviceBuffer::from_slice(&ctx, &absmax).expect("up amax");

    // ----- Reference: M=1 plan looped M times. -----
    let dev_y_ref = DeviceBuffer::from_slice(&ctx, &host_y_f16).expect("up y ref");
    let mut dev_out_ref: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (m_test as usize) * n).expect("alloc out ref");

    let desc_m1 = Nf4Descriptor {
        n: n as i32,
        k: k as i32,
        block_size: block_size as i32,
    };
    let plan_m1: Nf4MmvqPlan<f16> = Nf4MmvqPlan::select(
        &stream, &desc_m1, PlanPreference::default(),
    )
    .expect("plan m1 select");

    for mi in 0..m_test as usize {
        // Each row of activations is contiguous; reference path reads
        // them as a length-K vector and writes a length-N output row.
        let act_slice = dev_y_ref.slice((mi * k)..((mi + 1) * k));
        let out_slice_mut = dev_out_ref.slice_mut((mi * n)..((mi + 1) * n));
        let args = Nf4MmvqArgs::<f16> {
            weight: TensorRef {
                data: dev_w.as_slice(),
                shape: [weight_bytes_len],
                stride: contiguous_stride([weight_bytes_len]),
            },
            absmax: TensorRef {
                data: dev_amax.as_slice(),
                shape: [absmax_len],
                stride: contiguous_stride([absmax_len]),
            },
            activation: TensorRef {
                data: act_slice,
                shape: [k as i32],
                stride: contiguous_stride([k as i32]),
            },
            output: TensorMut {
                data: out_slice_mut,
                shape: [n as i32],
                stride: contiguous_stride([n as i32]),
            },
        };
        plan_m1
            .run(&stream, Workspace::None, args)
            .expect("m1 ref run");
    }
    stream.synchronize().expect("sync ref");
    let mut host_out_ref_f16 = vec![f16::ZERO; (m_test as usize) * n];
    dev_out_ref
        .copy_to_host(&mut host_out_ref_f16)
        .expect("d2h ref");

    // ----- Multi-M plan. -----
    let dev_y_multi = DeviceBuffer::from_slice(&ctx, &host_y_f16).expect("up y multi");
    let mut dev_out_multi: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (m_test as usize) * n).expect("alloc out multi");

    let desc_multi = Nf4MmvqMultiMDescriptor {
        base: desc_m1,
        m: m_test,
    };
    let plan_multi: Nf4MmvqMultiMPlan<f16> = Nf4MmvqMultiMPlan::select(
        &stream,
        &desc_multi,
        PlanPreference::default(),
    )
    .expect("plan multi select");

    let args = Nf4MmvqMultiMArgs::<f16> {
        weight: TensorRef {
            data: dev_w.as_slice(),
            shape: [weight_bytes_len],
            stride: contiguous_stride([weight_bytes_len]),
        },
        absmax: TensorRef {
            data: dev_amax.as_slice(),
            shape: [absmax_len],
            stride: contiguous_stride([absmax_len]),
        },
        activations: TensorRef {
            data: dev_y_multi.as_slice(),
            shape: [m_test, k as i32],
            stride: contiguous_stride([m_test, k as i32]),
        },
        output: TensorMut {
            data: dev_out_multi.as_slice_mut(),
            shape: [m_test, n as i32],
            stride: contiguous_stride([m_test, n as i32]),
        },
    };
    plan_multi
        .run(&stream, Workspace::None, args)
        .expect("multi run");
    stream.synchronize().expect("sync multi");

    let mut host_out_multi_f16 = vec![f16::ZERO; (m_test as usize) * n];
    dev_out_multi
        .copy_to_host(&mut host_out_multi_f16)
        .expect("d2h multi");

    // ----- Compare. -----
    // Both paths apply the same codebook + same absmax + same fp16
    // activations. The only allowable delta is a 1-ulp rounding
    // difference in the final f16 cast. Use a small relative tolerance.
    let mut max_abs_err = 0.0f32;
    let mut max_ref = 0.0f32;
    for i in 0..(m_test as usize) * n {
        let r = host_out_ref_f16[i].to_f32();
        let g = host_out_multi_f16[i].to_f32();
        let err = (g - r).abs();
        max_abs_err = max_abs_err.max(err);
        max_ref = max_ref.max(r.abs());
        assert!(
            err < 0.01 * (r.abs().max(1e-3)),
            "M={m_test} idx={i}: multi={g}, ref={r}, abs_err={err}"
        );
    }
    eprintln!(
        "nf4 multi-M={m_test}: max_ref={max_ref:.4e}, max_abs_err={max_abs_err:.4e}"
    );
}

#[test]
#[ignore]
fn nf4_multim_m1_f16() {
    run_for_m(1);
}

#[test]
#[ignore]
fn nf4_multim_m2_f16() {
    run_for_m(2);
}

#[test]
#[ignore]
fn nf4_multim_m4_f16() {
    run_for_m(4);
}

#[test]
#[ignore]
fn nf4_multim_m8_f16() {
    run_for_m(8);
}

/// Verify the plan rejects M values outside {1, 2, 4, 8}.
#[test]
fn nf4_multim_rejects_unsupported_m() {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    let desc = Nf4MmvqMultiMDescriptor {
        base: Nf4Descriptor {
            n: 16,
            k: 64,
            block_size: 64,
        },
        m: 3, // not in {1, 2, 4, 8}
    };
    let r = Nf4MmvqMultiMPlan::<f16>::select(&stream, &desc, PlanPreference::default());
    assert!(r.is_err(), "M=3 should be rejected; got Ok");
    let _ = ctx;
}
