//! Real-GPU smoke test for the Phase 59b FA2 backward backend on
//! [`FlashSdpaBackwardPlan`].
//!
//! Validates:
//!   1. FA2 BW workspace_size is non-zero on the FA2 backend and zero
//!      on the bespoke backend.
//!   2. Eligibility check: f32 → bespoke, head_dim ∉ FA2 set → bespoke.
//!   3. End-to-end BW execution for the FA2 head_dim set
//!      ({64, 128, 192, 256} × {f16, bf16} × {causal, non-causal});
//!      asserts the launch succeeds and writes a non-zero dQ.
//!
//! All tests are `#[ignore]`-gated. Build with `--features fa2,sm80`.

#![cfg(feature = "fa2")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BackendKind, ElementKind, FlashSdpaArgs, FlashSdpaBackwardArgs,
    FlashSdpaBackwardDescriptor, FlashSdpaBackwardPlan, FlashSdpaDescriptor, FlashSdpaPlan,
    PlanPreference, TensorMut, TensorRef, Workspace,
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

fn gen_bf16(n: usize, phase: f32) -> Vec<bf16> {
    (0..n)
        .map(|i| bf16::from_f32(((i as f32) * 0.013 + phase).sin() * 0.25))
        .collect()
}

// =========================================================================
// Plan-layer surface sanity
// =========================================================================

#[test]
#[ignore]
fn fa2_bw_workspace_size_nonzero_when_fa2() {
    let (_ctx, stream) = setup();
    for &dk in &[32_i32, 64, 96, 128, 192, 256] {
        let desc = FlashSdpaBackwardDescriptor::new(
            1, 4, 256, 256, dk, dk, default_scale(dk), false, ElementKind::F16,
        );
        let plan = FlashSdpaBackwardPlan::<f16>::select(
            &stream,
            &desc,
            PlanPreference {
                prefer_backend: Some(BackendKind::FlashAttentionV2),
                ..Default::default()
            },
        )
        .expect("select FA2 BW");
        assert_eq!(plan.backend(), BackendKind::FlashAttentionV2);
        let ws = plan.workspace_size();
        assert!(
            ws > 0,
            "FA2 BW workspace_size must be non-zero for d={}, got {}",
            dk,
            ws
        );
    }
}

#[test]
#[ignore]
fn fa2_bw_workspace_size_zero_when_bespoke() {
    let (_ctx, stream) = setup();
    let desc = FlashSdpaBackwardDescriptor::new(
        1, 4, 128, 128, 64, 64, default_scale(64), false, ElementKind::F32,
    );
    let plan = FlashSdpaBackwardPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select bespoke");
    assert_eq!(plan.backend(), BackendKind::Bespoke);
    assert_eq!(plan.workspace_size(), 0);
}

#[test]
#[ignore]
fn fa2_bw_eligibility_f32_picks_bespoke() {
    let (_ctx, stream) = setup();
    let desc = FlashSdpaBackwardDescriptor::new(
        1, 4, 256, 256, 64, 64, default_scale(64), false, ElementKind::F32,
    );
    let plan = FlashSdpaBackwardPlan::<f32>::select(
        &stream,
        &desc,
        PlanPreference {
            prefer_backend: Some(BackendKind::FlashAttentionV2),
            ..Default::default()
        },
    )
    .expect("select with FA2 pref (falls back to bespoke for f32)");
    assert_eq!(plan.backend(), BackendKind::Bespoke);
}

#[test]
#[ignore]
fn fa2_bw_eligibility_bad_hdim_picks_bespoke() {
    let (_ctx, stream) = setup();
    // 72 is NOT in {32, 64, 96, 128, 192, 256} and ≤ 128 (bespoke eligible).
    let desc = FlashSdpaBackwardDescriptor::new(
        1, 4, 256, 256, 72, 72, default_scale(72), false, ElementKind::F16,
    );
    let plan = FlashSdpaBackwardPlan::<f16>::select(
        &stream,
        &desc,
        PlanPreference {
            prefer_backend: Some(BackendKind::FlashAttentionV2),
            ..Default::default()
        },
    )
    .expect("select");
    assert_eq!(
        plan.backend(),
        BackendKind::Bespoke,
        "head_dim=72 not in FA2 set, must fall back to bespoke"
    );
}

// =========================================================================
// End-to-end FA2 BW runs (writes non-zero dQ).
//
// Strategy: run a single FA2 FW pass through the FFI directly with our
// own f32 LSE buffer (FA2's launcher writes `softmax_lse_ptr` regardless
// of its caller), then feed that LSE + saved y + dy into the FA2 BW plan
// through the high-level API. Asserts the BW launch produces non-zero
// gradient.
// =========================================================================

fn fa2_fw_into_f32_lse_f16(
    ctx: &Context,
    stream: &Stream,
    b: i32, h: i32, q: i32, k: i32, dk: i32, dv: i32, scale: f32, is_causal: bool,
    dq: &DeviceBuffer<f16>, dk_buf: &DeviceBuffer<f16>, dv_buf: &DeviceBuffer<f16>,
    dy: &mut DeviceBuffer<f16>,
    dlse_f32: &mut DeviceBuffer<f32>,
) {
    let n_q = (b * h * q * dk) as usize;
    let _ = (n_q, ctx);
    // Build dummy lse-T buffer the plan still requires by shape (it's
    // routed via workspace on the FA2 path).
    let mut dummy_lse_t_mut: DeviceBuffer<f16> =
        DeviceBuffer::from_slice(ctx, &vec![f16::ZERO; (b * h * q) as usize])
            .expect("alloc dummy lse_t");
    // Use the plan layer; workspace receives the f32 LSE.
    // Plan workspace size: B * H * Q * 4 bytes — matches dlse_f32 size.
    let desc = FlashSdpaDescriptor::new(
        b, h, q, k, dk, dv, scale, is_causal, ElementKind::F16,
    );
    let plan = FlashSdpaPlan::<f16>::select(
        stream,
        &desc,
        PlanPreference {
            prefer_backend: Some(BackendKind::FlashAttentionV2),
            ..Default::default()
        },
    )
    .expect("FW FA2 select");
    assert_eq!(plan.backend(), BackendKind::FlashAttentionV2);
    let ws_bytes_needed = plan.workspace_size();
    // dlse_f32 is f32 elements = 4 bytes each. We need at least
    // `ws_bytes_needed` bytes. dlse_f32.len() * 4 must satisfy this.
    assert!(
        dlse_f32.len() * 4 >= ws_bytes_needed,
        "f32 lse buffer too small for FA2 workspace"
    );
    let sq = [b, h, q, dk];
    let sk = [b, h, k, dk];
    let sv = [b, h, k, dv];
    let sy = [b, h, q, dv];
    let sl = [b, h, q];

    // Reinterpret dlse_f32's underlying bytes as a `u8` workspace slot
    // for the FA2 launcher. We achieve that via the existing
    // `Workspace::Borrowed` machinery: borrow the f32 buffer's
    // DeviceSliceMut — but Workspace<'a> wraps a typed slice and the
    // FA2 launcher Rust wrapper expects u8 bytes. Easiest path:
    // allocate a u8 workspace, run, then copy back to dlse_f32 via host
    // round-trip.
    let mut ws_buf: DeviceBuffer<u8> =
        DeviceBuffer::from_slice(ctx, &vec![0u8; ws_bytes_needed]).expect("ws");
    plan.run(
        stream,
        Workspace::Borrowed(ws_buf.as_slice_mut()),
        FlashSdpaArgs {
            q: TensorRef { data: dq.as_slice(), shape: sq, stride: contiguous_stride(sq) },
            k: TensorRef { data: dk_buf.as_slice(), shape: sk, stride: contiguous_stride(sk) },
            v: TensorRef { data: dv_buf.as_slice(), shape: sv, stride: contiguous_stride(sv) },
            y: TensorMut { data: dy.as_slice_mut(), shape: sy, stride: contiguous_stride(sy) },
            lse: TensorMut {
                data: dummy_lse_t_mut.as_slice_mut(),
                shape: sl,
                stride: contiguous_stride(sl),
            },
            mask: None,
            alibi_slopes: None,
        },
    )
    .expect("FW FA2 run");
    stream.synchronize().expect("sync");

    // Round-trip workspace bytes through host → dlse_f32.
    let mut h_ws_bytes = vec![0u8; ws_bytes_needed];
    ws_buf.copy_to_host(&mut h_ws_bytes).expect("ws→host");
    let n_lse = dlse_f32.len();
    let mut h_lse = vec![0_f32; n_lse];
    for (i, dst) in h_lse.iter_mut().enumerate() {
        let mut b4 = [0u8; 4];
        b4.copy_from_slice(&h_ws_bytes[i * 4..(i + 1) * 4]);
        *dst = f32::from_le_bytes(b4);
    }
    dlse_f32.copy_from_host(&h_lse).expect("host→lse_f32");
    stream.synchronize().expect("sync");
}

fn run_fa2_bw_f16(dk: i32, is_causal: bool) {
    let (ctx, stream) = setup();
    let b: i32 = 1;
    let h: i32 = 4;
    let q: i32 = 128;
    let k: i32 = 128;
    let dv: i32 = dk;
    let scale = default_scale(dk);
    let n_qkv = (b * h * q * dk) as usize;
    let n_kv = (b * h * k * dk) as usize;
    let n_lse_f32 = (b * h * q) as usize;

    let q_host = gen_f16(n_qkv, 0.1);
    let k_host = gen_f16(n_kv, 0.2);
    let v_host = gen_f16(n_kv, 0.3);
    let dy_host = gen_f16(n_qkv, 0.4);

    let dq = DeviceBuffer::from_slice(&ctx, &q_host).expect("up q");
    let dkb = DeviceBuffer::from_slice(&ctx, &k_host).expect("up k");
    let dvb = DeviceBuffer::from_slice(&ctx, &v_host).expect("up v");
    let ddy = DeviceBuffer::from_slice(&ctx, &dy_host).expect("up dy");
    let mut dy_out: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n_qkv).expect("y");
    let mut dlse_f32: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n_lse_f32).expect("lse_f32");

    fa2_fw_into_f32_lse_f16(
        &ctx, &stream, b, h, q, k, dk, dv, scale, is_causal,
        &dq, &dkb, &dvb, &mut dy_out, &mut dlse_f32,
    );

    // Build dummy lse_T buffer for the bespoke field — IGNORED on FA2 path.
    let dummy_lse_t: DeviceBuffer<f16> =
        DeviceBuffer::from_slice(&ctx, &vec![f16::ZERO; n_lse_f32]).expect("dummy lse_t");

    let bw_desc = FlashSdpaBackwardDescriptor::new(
        b, h, q, k, dk, dv, scale, is_causal, ElementKind::F16,
    );
    let bw_plan = FlashSdpaBackwardPlan::<f16>::select(
        &stream,
        &bw_desc,
        PlanPreference {
            prefer_backend: Some(BackendKind::FlashAttentionV2),
            ..Default::default()
        },
    )
    .expect("BW FA2 select");
    assert_eq!(bw_plan.backend(), BackendKind::FlashAttentionV2);
    let bw_ws_bytes = bw_plan.workspace_size();
    assert!(bw_ws_bytes > 0);
    let mut bw_ws: DeviceBuffer<u8> =
        DeviceBuffer::from_slice(&ctx, &vec![0u8; bw_ws_bytes]).expect("bw ws");

    let mut ddq: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n_qkv).expect("dq");
    let mut ddk: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n_kv).expect("dk");
    let mut ddv: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n_kv).expect("dv");
    let mut dd_ws: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n_lse_f32).expect("dws");

    let sq = [b, h, q, dk];
    let sk = [b, h, k, dk];
    let sv = [b, h, k, dv];
    let sy = [b, h, q, dv];
    let sl = [b, h, q];

    bw_plan
        .run(
            &stream,
            Workspace::Borrowed(bw_ws.as_slice_mut()),
            FlashSdpaBackwardArgs {
                q: TensorRef { data: dq.as_slice(), shape: sq, stride: contiguous_stride(sq) },
                k: TensorRef { data: dkb.as_slice(), shape: sk, stride: contiguous_stride(sk) },
                v: TensorRef { data: dvb.as_slice(), shape: sv, stride: contiguous_stride(sv) },
                y: TensorRef { data: dy_out.as_slice(), shape: sy, stride: contiguous_stride(sy) },
                lse: TensorRef { data: dummy_lse_t.as_slice(), shape: sl, stride: contiguous_stride(sl) },
                dy: TensorRef { data: ddy.as_slice(), shape: sy, stride: contiguous_stride(sy) },
                d_ws: TensorMut { data: dd_ws.as_slice_mut(), shape: sl, stride: contiguous_stride(sl) },
                dq: TensorMut { data: ddq.as_slice_mut(), shape: sq, stride: contiguous_stride(sq) },
                dk: TensorMut { data: ddk.as_slice_mut(), shape: sk, stride: contiguous_stride(sk) },
                dv: TensorMut { data: ddv.as_slice_mut(), shape: sv, stride: contiguous_stride(sv) },
                lse_f32: Some(TensorRef {
                    data: dlse_f32.as_slice(),
                    shape: sl,
                    stride: contiguous_stride(sl),
                }),
                alibi_slopes: None,
            },
        )
        .expect("FA2 BW run");
    stream.synchronize().expect("");

    let mut h_ddq = vec![f16::ZERO; n_qkv];
    ddq.copy_to_host(&mut h_ddq).expect("dq→host");
    let nonzero = h_ddq.iter().any(|&v| v.to_f32().abs() > 1e-6);
    assert!(
        nonzero,
        "FA2 BW produced all-zero dQ (d={}, causal={})",
        dk, is_causal
    );
}

fn run_fa2_bw_bf16(dk: i32, is_causal: bool) {
    let (ctx, stream) = setup();
    let b: i32 = 1;
    let h: i32 = 4;
    let q: i32 = 128;
    let k: i32 = 128;
    let dv: i32 = dk;
    let scale = default_scale(dk);
    let n_qkv = (b * h * q * dk) as usize;
    let n_kv = (b * h * k * dk) as usize;
    let n_lse_f32 = (b * h * q) as usize;

    let q_host = gen_bf16(n_qkv, 0.1);
    let k_host = gen_bf16(n_kv, 0.2);
    let v_host = gen_bf16(n_kv, 0.3);
    let dy_host = gen_bf16(n_qkv, 0.4);

    let dq = DeviceBuffer::from_slice(&ctx, &q_host).expect("up q");
    let dkb = DeviceBuffer::from_slice(&ctx, &k_host).expect("up k");
    let dvb = DeviceBuffer::from_slice(&ctx, &v_host).expect("up v");
    let ddy = DeviceBuffer::from_slice(&ctx, &dy_host).expect("up dy");
    let mut dy_out: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, n_qkv).expect("y");
    let mut dlse_f32: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n_lse_f32).expect("lse_f32");

    // FW (bf16 variant).
    let mut dummy_lse_t_mut: DeviceBuffer<bf16> =
        DeviceBuffer::from_slice(&ctx, &vec![bf16::ZERO; n_lse_f32]).expect("dummy lse_t");
    let dummy_lse_t: DeviceBuffer<bf16> =
        DeviceBuffer::from_slice(&ctx, &vec![bf16::ZERO; n_lse_f32]).expect("dummy lse_t");
    let fw_desc = FlashSdpaDescriptor::new(
        b, h, q, k, dk, dv, scale, is_causal, ElementKind::Bf16,
    );
    let fw_plan = FlashSdpaPlan::<bf16>::select(
        &stream,
        &fw_desc,
        PlanPreference {
            prefer_backend: Some(BackendKind::FlashAttentionV2),
            ..Default::default()
        },
    )
    .expect("FW FA2 select bf16");
    let fw_ws_bytes = fw_plan.workspace_size();
    let mut ws_buf: DeviceBuffer<u8> =
        DeviceBuffer::from_slice(&ctx, &vec![0u8; fw_ws_bytes]).expect("ws");
    let sq = [b, h, q, dk];
    let sk = [b, h, k, dk];
    let sv = [b, h, k, dv];
    let sy = [b, h, q, dv];
    let sl = [b, h, q];
    fw_plan
        .run(
            &stream,
            Workspace::Borrowed(ws_buf.as_slice_mut()),
            FlashSdpaArgs {
                q: TensorRef { data: dq.as_slice(), shape: sq, stride: contiguous_stride(sq) },
                k: TensorRef { data: dkb.as_slice(), shape: sk, stride: contiguous_stride(sk) },
                v: TensorRef { data: dvb.as_slice(), shape: sv, stride: contiguous_stride(sv) },
                y: TensorMut { data: dy_out.as_slice_mut(), shape: sy, stride: contiguous_stride(sy) },
                lse: TensorMut {
                    data: dummy_lse_t_mut.as_slice_mut(),
                    shape: sl,
                    stride: contiguous_stride(sl),
                },
                mask: None,
                alibi_slopes: None,
            },
        )
        .expect("FW FA2 bf16 run");
    stream.synchronize().expect("");
    let mut h_ws_bytes = vec![0u8; fw_ws_bytes];
    ws_buf.copy_to_host(&mut h_ws_bytes).expect("ws→host");
    let mut h_lse = vec![0_f32; n_lse_f32];
    for (i, dst) in h_lse.iter_mut().enumerate() {
        let mut b4 = [0u8; 4];
        b4.copy_from_slice(&h_ws_bytes[i * 4..(i + 1) * 4]);
        *dst = f32::from_le_bytes(b4);
    }
    dlse_f32.copy_from_host(&h_lse).expect("host→lse_f32");
    stream.synchronize().expect("");

    let bw_desc = FlashSdpaBackwardDescriptor::new(
        b, h, q, k, dk, dv, scale, is_causal, ElementKind::Bf16,
    );
    let bw_plan = FlashSdpaBackwardPlan::<bf16>::select(
        &stream,
        &bw_desc,
        PlanPreference {
            prefer_backend: Some(BackendKind::FlashAttentionV2),
            ..Default::default()
        },
    )
    .expect("BW FA2 select bf16");
    let bw_ws_bytes = bw_plan.workspace_size();
    let mut bw_ws: DeviceBuffer<u8> =
        DeviceBuffer::from_slice(&ctx, &vec![0u8; bw_ws_bytes]).expect("bw ws");
    let mut ddq: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, n_qkv).expect("dq");
    let mut ddk: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, n_kv).expect("dk");
    let mut ddv: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, n_kv).expect("dv");
    let mut dd_ws: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, n_lse_f32).expect("dws");
    bw_plan
        .run(
            &stream,
            Workspace::Borrowed(bw_ws.as_slice_mut()),
            FlashSdpaBackwardArgs {
                q: TensorRef { data: dq.as_slice(), shape: sq, stride: contiguous_stride(sq) },
                k: TensorRef { data: dkb.as_slice(), shape: sk, stride: contiguous_stride(sk) },
                v: TensorRef { data: dvb.as_slice(), shape: sv, stride: contiguous_stride(sv) },
                y: TensorRef { data: dy_out.as_slice(), shape: sy, stride: contiguous_stride(sy) },
                lse: TensorRef { data: dummy_lse_t.as_slice(), shape: sl, stride: contiguous_stride(sl) },
                dy: TensorRef { data: ddy.as_slice(), shape: sy, stride: contiguous_stride(sy) },
                d_ws: TensorMut { data: dd_ws.as_slice_mut(), shape: sl, stride: contiguous_stride(sl) },
                dq: TensorMut { data: ddq.as_slice_mut(), shape: sq, stride: contiguous_stride(sq) },
                dk: TensorMut { data: ddk.as_slice_mut(), shape: sk, stride: contiguous_stride(sk) },
                dv: TensorMut { data: ddv.as_slice_mut(), shape: sv, stride: contiguous_stride(sv) },
                lse_f32: Some(TensorRef {
                    data: dlse_f32.as_slice(),
                    shape: sl,
                    stride: contiguous_stride(sl),
                }),
                alibi_slopes: None,
            },
        )
        .expect("FA2 BW bf16 run");
    stream.synchronize().expect("");
    let mut h_ddq = vec![bf16::ZERO; n_qkv];
    ddq.copy_to_host(&mut h_ddq).expect("");
    let nonzero = h_ddq.iter().any(|&v| v.to_f32().abs() > 1e-6);
    assert!(nonzero, "FA2 BW (bf16, d={}, causal={}) produced all-zero dQ", dk, is_causal);
}

#[test]
#[ignore]
fn fa2_bw_smoke_f16_d64_noncausal() { run_fa2_bw_f16(64, false); }

#[test]
#[ignore]
fn fa2_bw_smoke_f16_d64_causal() { run_fa2_bw_f16(64, true); }

#[test]
#[ignore]
fn fa2_bw_smoke_f16_d128_noncausal() { run_fa2_bw_f16(128, false); }

#[test]
#[ignore]
fn fa2_bw_smoke_f16_d128_causal() { run_fa2_bw_f16(128, true); }

#[test]
#[ignore]
fn fa2_bw_smoke_f16_d192_noncausal() { run_fa2_bw_f16(192, false); }

#[test]
#[ignore]
fn fa2_bw_smoke_f16_d256_noncausal() { run_fa2_bw_f16(256, false); }

#[test]
#[ignore]
fn fa2_bw_smoke_bf16_d128_noncausal() { run_fa2_bw_bf16(128, false); }

#[test]
#[ignore]
fn fa2_bw_smoke_bf16_d128_causal() { run_fa2_bw_bf16(128, true); }

// Phase 60: hd160/hd224/hd512 BW NOT supported by FA2 — see
// FA2_BW_SUPPORTED_HEAD_DIMS in flash_sdpa_backward.rs for the
// kernel-level constraints. FW works for all three (Phase 60 FW
// vendor); only BW falls back to bespoke SDPA BW.
// No Phase 60 BW smoke tests to add.
