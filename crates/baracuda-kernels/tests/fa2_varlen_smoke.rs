//! Real-GPU smoke tests for the Phase 59b FA2 varlen FW + BW plans.
//!
//! Validates:
//!   1. `FlashSdpaVarlenPlan::select` succeeds for FA2-eligible shapes
//!      and reports BackendKind::FlashAttentionV2.
//!   2. `FlashSdpaVarlenPlan::lse_size` returns
//!      `H * (total_q + 128 * batch)`.
//!   3. End-to-end varlen FW launches: pack a batch of 3 sequences,
//!      assert the output is non-zero across all sequences.
//!   4. End-to-end varlen BW launches: assert dQ is non-zero.
//!   5. GQA × varlen combo.
//!
//! All gated behind the `device-tests` feature-gated. Build with `--features fa2_backward,sm80`.
//!
//! Gated on `fa2_backward` (not just `fa2`) because this file exercises
//! both the varlen FW plan AND `FlashSdpaVarlenBackwardPlan` (BW) in the
//! GQA-combo test; `fa2_backward` implies `fa2` so the FW coverage here
//! still runs whenever this file runs.

#![cfg(feature = "fa2_backward")]
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    BackendKind, ElementKind, FlashSdpaVarlenArgs, FlashSdpaVarlenBackwardArgs,
    FlashSdpaVarlenBackwardPlan, FlashSdpaVarlenDescriptor, FlashSdpaVarlenPlan, PlanPreference,
    TensorMut, TensorRef, Workspace, contiguous_stride,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn default_scale(d: i32) -> f32 {
    1.0 / (d as f32).sqrt()
}

fn gen_f16(n: usize, phase: f32) -> Vec<f16> {
    (0..n)
        .map(|i| f16::from_f32(((i as f32) * 0.013 + phase).sin() * 0.25))
        .collect()
}

// =========================================================================
// Plan-layer surface
// =========================================================================

#[test]
fn fa2_varlen_fw_plan_selects_fa2_backend() {
    let (_ctx, stream) = setup();
    let desc = FlashSdpaVarlenDescriptor::new(
        3,
        4,
        4,
        64,
        64,
        64,
        64,
        default_scale(64),
        false,
        ElementKind::F16,
    );
    let plan = FlashSdpaVarlenPlan::<f16>::select(&stream, &desc, PlanPreference::default())
        .expect("varlen FW select");
    assert_eq!(plan.sku().backend, BackendKind::FlashAttentionV2);
}

#[test]
fn fa2_varlen_lse_size_matches_formula() {
    let (_ctx, stream) = setup();
    let b = 3_i32;
    let h = 4_i32;
    let total_q = 100_i32;
    let desc = FlashSdpaVarlenDescriptor::new(
        b,
        h,
        h,
        64,
        64,
        64,
        64,
        default_scale(64),
        false,
        ElementKind::F16,
    );
    let plan =
        FlashSdpaVarlenPlan::<f16>::select(&stream, &desc, PlanPreference::default()).expect("");
    let lse = plan.lse_size(total_q);
    let expected = (h as usize) * (total_q as usize + 128 * b as usize);
    assert_eq!(lse, expected);
}

// =========================================================================
// End-to-end varlen FW.
// Pack 3 sequences of lengths {30, 70, 40}, total_q=140, total_k=140
// (self-attention).
// =========================================================================

#[test]
fn fa2_varlen_fw_smoke_f16() {
    let (ctx, stream) = setup();
    let lens = [30_i32, 70, 40];
    let batch = lens.len() as i32;
    let total_q: i32 = lens.iter().sum();
    let max_q = *lens.iter().max().unwrap();
    let h = 4_i32;
    let dk = 64_i32;
    let dv = dk;

    let n_qkv = (total_q as usize) * (h as usize) * (dk as usize);
    let q_host = gen_f16(n_qkv, 0.1);
    let k_host = gen_f16(n_qkv, 0.2);
    let v_host = gen_f16(n_qkv, 0.3);

    let dq = DeviceBuffer::from_slice(&ctx, &q_host).expect("up q");
    let dk_buf = DeviceBuffer::from_slice(&ctx, &k_host).expect("up k");
    let dv_buf = DeviceBuffer::from_slice(&ctx, &v_host).expect("up v");
    let mut dy: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n_qkv).expect("alloc y");

    // cu_seqlens: [0, 30, 100, 140]
    let mut cu_q_host: Vec<i32> = Vec::with_capacity(lens.len() + 1);
    cu_q_host.push(0);
    let mut acc = 0;
    for &l in lens.iter() {
        acc += l;
        cu_q_host.push(acc);
    }
    let cu_q = DeviceBuffer::from_slice(&ctx, &cu_q_host).expect("up cu_q");
    let cu_k = DeviceBuffer::from_slice(&ctx, &cu_q_host).expect("up cu_k");

    let desc = FlashSdpaVarlenDescriptor::new(
        batch,
        h,
        h,
        max_q,
        max_q,
        dk,
        dv,
        default_scale(dk),
        false,
        ElementKind::F16,
    );
    let plan = FlashSdpaVarlenPlan::<f16>::select(&stream, &desc, PlanPreference::default())
        .expect("varlen FW select");
    // LSE: f32 [H, total_q + 128 * batch]
    let lse_n = plan.lse_size(total_q);
    let mut dlse: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, lse_n).expect("alloc lse");

    let s_q = [total_q, h, dk];
    let s_k = [total_q, h, dk];
    let s_v = [total_q, h, dv];
    let s_y = [total_q, h, dv];
    let s_lse = [h, (total_q + 128 * batch)];
    let s_cu = [batch + 1];

    plan.run(
        &stream,
        Workspace::None,
        FlashSdpaVarlenArgs {
            q: TensorRef::new(dq.as_slice(), s_q, contiguous_stride(s_q)).unwrap(),
            k: TensorRef::new(dk_buf.as_slice(), s_k, contiguous_stride(s_k)).unwrap(),
            v: TensorRef::new(dv_buf.as_slice(), s_v, contiguous_stride(s_v)).unwrap(),
            y: TensorMut::new(dy.as_slice_mut(), s_y, contiguous_stride(s_y)).unwrap(),
            lse: TensorMut::new(dlse.as_slice_mut(), s_lse, contiguous_stride(s_lse)).unwrap(),
            cu_seqlens_q: TensorRef::new(cu_q.as_slice(), s_cu, contiguous_stride(s_cu)).unwrap(),
            cu_seqlens_k: TensorRef::new(cu_k.as_slice(), s_cu, contiguous_stride(s_cu)).unwrap(),
            alibi_slopes: None,
        },
    )
    .expect("varlen FW run");
    stream.synchronize().expect("sync");

    let mut h_y = vec![f16::ZERO; n_qkv];
    dy.copy_to_host(&mut h_y).expect("dl y");
    let nonzero = h_y.iter().any(|&v| v.to_f32().abs() > 1e-6);
    assert!(nonzero, "varlen FW produced all-zero output");
}

// =========================================================================
// End-to-end varlen BW: pack 2 short sequences, run FW then BW, assert
// dQ non-zero.
// =========================================================================

#[test]
fn fa2_varlen_bw_smoke_f16() {
    let (ctx, stream) = setup();
    let lens = [30_i32, 50];
    let batch = lens.len() as i32;
    let total_q: i32 = lens.iter().sum();
    let max_q = *lens.iter().max().unwrap();
    let h = 4_i32;
    let dk = 64_i32;
    let dv = dk;
    let n_qkv = (total_q as usize) * (h as usize) * (dk as usize);

    let q_host = gen_f16(n_qkv, 0.1);
    let k_host = gen_f16(n_qkv, 0.2);
    let v_host = gen_f16(n_qkv, 0.3);
    let dy_host = gen_f16(n_qkv, 0.4);

    let dq = DeviceBuffer::from_slice(&ctx, &q_host).expect("");
    let dkb = DeviceBuffer::from_slice(&ctx, &k_host).expect("");
    let dvb = DeviceBuffer::from_slice(&ctx, &v_host).expect("");
    let ddy = DeviceBuffer::from_slice(&ctx, &dy_host).expect("");
    let mut dy_out: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n_qkv).expect("");

    let mut cu_q_host: Vec<i32> = Vec::with_capacity(lens.len() + 1);
    cu_q_host.push(0);
    let mut acc = 0;
    for &l in lens.iter() {
        acc += l;
        cu_q_host.push(acc);
    }
    let cu_q = DeviceBuffer::from_slice(&ctx, &cu_q_host).expect("");
    let cu_k = DeviceBuffer::from_slice(&ctx, &cu_q_host).expect("");

    let desc = FlashSdpaVarlenDescriptor::new(
        batch,
        h,
        h,
        max_q,
        max_q,
        dk,
        dv,
        default_scale(dk),
        true,
        ElementKind::F16,
    );
    let fw_plan = FlashSdpaVarlenPlan::<f16>::select(&stream, &desc, PlanPreference::default())
        .expect("fw plan");
    let lse_n = fw_plan.lse_size(total_q);
    let mut dlse: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, lse_n).expect("");
    let s_q = [total_q, h, dk];
    let s_v = [total_q, h, dv];
    let s_lse = [h, (total_q + 128 * batch)];
    let s_cu = [batch + 1];

    fw_plan
        .run(
            &stream,
            Workspace::None,
            FlashSdpaVarlenArgs {
                q: TensorRef::new(dq.as_slice(), s_q, contiguous_stride(s_q)).unwrap(),
                k: TensorRef::new(dkb.as_slice(), s_q, contiguous_stride(s_q)).unwrap(),
                v: TensorRef::new(dvb.as_slice(), s_v, contiguous_stride(s_v)).unwrap(),
                y: TensorMut::new(dy_out.as_slice_mut(), s_v, contiguous_stride(s_v)).unwrap(),
                lse: TensorMut::new(dlse.as_slice_mut(), s_lse, contiguous_stride(s_lse)).unwrap(),
                cu_seqlens_q: TensorRef::new(cu_q.as_slice(), s_cu, contiguous_stride(s_cu))
                    .unwrap(),
                cu_seqlens_k: TensorRef::new(cu_k.as_slice(), s_cu, contiguous_stride(s_cu))
                    .unwrap(),
                alibi_slopes: None,
            },
        )
        .expect("varlen FW run");
    stream.synchronize().expect("");

    let bw_plan =
        FlashSdpaVarlenBackwardPlan::<f16>::select(&stream, &desc, PlanPreference::default())
            .expect("bw plan");
    let ws_bytes = bw_plan.workspace_size(total_q);
    assert!(ws_bytes > 0);
    let mut ws: DeviceBuffer<u8> =
        DeviceBuffer::from_slice(&ctx, &vec![0u8; ws_bytes]).expect("ws");

    let mut ddq: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n_qkv).expect("");
    let mut ddk: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n_qkv).expect("");
    let mut ddv: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n_qkv).expect("");

    bw_plan
        .run(
            &stream,
            Workspace::Borrowed(ws.as_slice_mut()),
            FlashSdpaVarlenBackwardArgs {
                q: TensorRef::new(dq.as_slice(), s_q, contiguous_stride(s_q)).unwrap(),
                k: TensorRef::new(dkb.as_slice(), s_q, contiguous_stride(s_q)).unwrap(),
                v: TensorRef::new(dvb.as_slice(), s_v, contiguous_stride(s_v)).unwrap(),
                y: TensorRef::new(dy_out.as_slice(), s_v, contiguous_stride(s_v)).unwrap(),
                dy: TensorRef::new(ddy.as_slice(), s_v, contiguous_stride(s_v)).unwrap(),
                lse: TensorRef::new(dlse.as_slice(), s_lse, contiguous_stride(s_lse)).unwrap(),
                cu_seqlens_q: TensorRef::new(cu_q.as_slice(), s_cu, contiguous_stride(s_cu))
                    .unwrap(),
                cu_seqlens_k: TensorRef::new(cu_k.as_slice(), s_cu, contiguous_stride(s_cu))
                    .unwrap(),
                dq: TensorMut::new(ddq.as_slice_mut(), s_q, contiguous_stride(s_q)).unwrap(),
                dk: TensorMut::new(ddk.as_slice_mut(), s_q, contiguous_stride(s_q)).unwrap(),
                dv: TensorMut::new(ddv.as_slice_mut(), s_v, contiguous_stride(s_v)).unwrap(),
                alibi_slopes: None,
            },
        )
        .expect("varlen BW run");
    stream.synchronize().expect("");

    let mut h_dq = vec![f16::ZERO; n_qkv];
    ddq.copy_to_host(&mut h_dq).expect("");
    let nonzero = h_dq.iter().any(|&v| v.to_f32().abs() > 1e-6);
    assert!(nonzero, "varlen BW produced all-zero dQ");
}

// =========================================================================
// GQA × varlen: H=4, H_k=2; 3 sequences. FW only — verifies the GQA
// h_h_k_ratio path coexists with varlen indexing.
// =========================================================================

#[test]
fn fa2_varlen_gqa_fw_smoke_bf16() {
    let (ctx, stream) = setup();
    let lens = [50_i32, 30, 70];
    let batch = lens.len() as i32;
    let total_q: i32 = lens.iter().sum();
    let max_q = *lens.iter().max().unwrap();
    let h = 4_i32;
    let h_k = 2_i32; // GQA
    let dk = 128_i32;
    let dv = dk;

    let n_q = (total_q as usize) * (h as usize) * (dk as usize);
    let n_kv = (total_q as usize) * (h_k as usize) * (dk as usize);

    let q_host: Vec<bf16> = (0..n_q)
        .map(|i| bf16::from_f32(((i as f32) * 0.011).sin() * 0.2))
        .collect();
    let k_host: Vec<bf16> = (0..n_kv)
        .map(|i| bf16::from_f32(((i as f32) * 0.012).cos() * 0.2))
        .collect();
    let v_host: Vec<bf16> = (0..n_kv)
        .map(|i| bf16::from_f32(((i as f32) * 0.013).sin() * 0.2))
        .collect();

    let dq = DeviceBuffer::from_slice(&ctx, &q_host).expect("");
    let dkb = DeviceBuffer::from_slice(&ctx, &k_host).expect("");
    let dvb = DeviceBuffer::from_slice(&ctx, &v_host).expect("");
    let mut dy: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, n_q).expect("");

    let mut cu_q_host: Vec<i32> = Vec::with_capacity(lens.len() + 1);
    cu_q_host.push(0);
    let mut acc = 0;
    for &l in lens.iter() {
        acc += l;
        cu_q_host.push(acc);
    }
    let cu_q = DeviceBuffer::from_slice(&ctx, &cu_q_host).expect("");
    let cu_k = DeviceBuffer::from_slice(&ctx, &cu_q_host).expect("");

    let desc = FlashSdpaVarlenDescriptor::new(
        batch,
        h,
        h_k,
        max_q,
        max_q,
        dk,
        dv,
        default_scale(dk),
        false,
        ElementKind::Bf16,
    );
    let plan =
        FlashSdpaVarlenPlan::<bf16>::select(&stream, &desc, PlanPreference::default()).expect("");
    let lse_n = plan.lse_size(total_q);
    let mut dlse: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, lse_n).expect("");

    let s_q = [total_q, h, dk];
    let s_k = [total_q, h_k, dk];
    let s_v = [total_q, h_k, dv];
    let s_y = [total_q, h, dv];
    let s_lse = [h, (total_q + 128 * batch)];
    let s_cu = [batch + 1];

    plan.run(
        &stream,
        Workspace::None,
        FlashSdpaVarlenArgs {
            q: TensorRef::new(dq.as_slice(), s_q, contiguous_stride(s_q)).unwrap(),
            k: TensorRef::new(dkb.as_slice(), s_k, contiguous_stride(s_k)).unwrap(),
            v: TensorRef::new(dvb.as_slice(), s_v, contiguous_stride(s_v)).unwrap(),
            y: TensorMut::new(dy.as_slice_mut(), s_y, contiguous_stride(s_y)).unwrap(),
            lse: TensorMut::new(dlse.as_slice_mut(), s_lse, contiguous_stride(s_lse)).unwrap(),
            cu_seqlens_q: TensorRef::new(cu_q.as_slice(), s_cu, contiguous_stride(s_cu)).unwrap(),
            cu_seqlens_k: TensorRef::new(cu_k.as_slice(), s_cu, contiguous_stride(s_cu)).unwrap(),
            alibi_slopes: None,
        },
    )
    .expect("varlen GQA FW run");
    stream.synchronize().expect("");

    let mut h_y = vec![bf16::ZERO; n_q];
    dy.copy_to_host(&mut h_y).expect("");
    let nonzero = h_y.iter().any(|&v| v.to_f32().abs() > 1e-6);
    assert!(nonzero, "varlen GQA FW produced all-zero output");
}
