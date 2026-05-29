//! Real-GPU smoke test for `FlashSdpaBackwardPlan + AttentionKind::FlashAttention` BW.
//!
//! Validates dQ / dK / dV from the Flash BW pipeline against the naive
//! [`SdpaBackwardPlan`] (Milestone 6.2). Both implement the same math —
//! Flash re-derives `P` from saved `lse` instead of consuming a saved
//! `attn`, but the produced gradients match within the documented
//! tile-by-tile rounding envelope.
//!
//! Covers BW × 4 FP dtypes. `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BackendKind, ElementKind, FlashSdpaArgs, FlashSdpaBackwardArgs,
    FlashSdpaBackwardDescriptor, FlashSdpaBackwardPlan, FlashSdpaDescriptor, FlashSdpaPlan,
    PlanPreference, SdpaArgs, SdpaBackwardArgs, SdpaBackwardDescriptor, SdpaBackwardPlan,
    SdpaDescriptor, SdpaPlan, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

const F16_EPS: f32 = 9.77e-4;
const BF16_EPS: f32 = 7.81e-3;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

const B: i32 = 2;
const H: i32 = 4;
const Q: i32 = 128;
const K: i32 = 128;
const DK: i32 = 32;
const DV: i32 = 32;

fn default_scale() -> f32 {
    1.0 / (DK as f32).sqrt()
}

fn gen_q_f64() -> Vec<f64> {
    let n = (B * H * Q * DK) as usize;
    (0..n)
        .map(|i| ((i as f64) * 0.013 - 0.5).sin() * 0.5)
        .collect()
}
fn gen_k_f64() -> Vec<f64> {
    let n = (B * H * K * DK) as usize;
    (0..n)
        .map(|i| ((i as f64) * 0.017 + 0.2).cos() * 0.5)
        .collect()
}
fn gen_v_f64() -> Vec<f64> {
    let n = (B * H * K * DV) as usize;
    (0..n)
        .map(|i| ((i as f64) * 0.011 - 0.1).sin() * 0.7)
        .collect()
}
fn gen_dy_f64() -> Vec<f64> {
    let n = (B * H * Q * DV) as usize;
    (0..n)
        .map(|i| ((i as f64) * 0.019 + 0.3).cos() * 0.4)
        .collect()
}

// ----------------------------------------------------------------------------
// f32
// ----------------------------------------------------------------------------

#[test]
#[ignore]
fn flash_sdpa_backward_f32_basic() {
    let (ctx, stream) = setup();
    let q_f64 = gen_q_f64();
    let k_f64 = gen_k_f64();
    let v_f64 = gen_v_f64();
    let dy_f64 = gen_dy_f64();
    let scale = default_scale();

    let q_h: Vec<f32> = q_f64.iter().map(|&v| v as f32).collect();
    let k_h: Vec<f32> = k_f64.iter().map(|&v| v as f32).collect();
    let v_h: Vec<f32> = v_f64.iter().map(|&v| v as f32).collect();
    let dy_h: Vec<f32> = dy_f64.iter().map(|&v| v as f32).collect();

    let dq_dev = DeviceBuffer::from_slice(&ctx, &q_h).expect("up q");
    let dk_dev = DeviceBuffer::from_slice(&ctx, &k_h).expect("up k");
    let dv_dev = DeviceBuffer::from_slice(&ctx, &v_h).expect("up v");
    let ddy_dev = DeviceBuffer::from_slice(&ctx, &dy_h).expect("up dy");

    let sq = [B, H, Q, DK];
    let sk = [B, H, K, DK];
    let sv = [B, H, K, DV];
    let sa = [B, H, Q, K];
    let sy = [B, H, Q, DV];
    let sl = [B, H, Q];

    // -------- Reference: naive Sdpa FW+BW --------
    let mut dattn_ref: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * K) as usize).expect("ref attn");
    let mut dy_out_ref: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DV) as usize).expect("ref y");
    let mut dws_ref: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * K) as usize).expect("ref ws");
    let mut ddq_ref: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DK) as usize).expect("ref dq");
    let mut ddk_ref: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (B * H * K * DK) as usize).expect("ref dk");
    let mut ddv_ref: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (B * H * K * DV) as usize).expect("ref dv");

    let fw_desc = SdpaDescriptor {
        batch_size: B,
        num_heads: H,
        query_len: Q,
        key_len: K,
        d_k: DK,
        d_v: DV,
        scale,
        is_causal: false,
        has_mask: false,
        element: ElementKind::F32,
    };
    let fw_plan = SdpaPlan::<f32>::select(&stream, &fw_desc, PlanPreference::default()).expect("");
    fw_plan
        .run(
            &stream,
            Workspace::None,
            SdpaArgs {
                q: TensorRef { data: dq_dev.as_slice(), shape: sq, stride: contiguous_stride(sq) },
                k: TensorRef { data: dk_dev.as_slice(), shape: sk, stride: contiguous_stride(sk) },
                v: TensorRef { data: dv_dev.as_slice(), shape: sv, stride: contiguous_stride(sv) },
                mask: None,
                y: TensorMut { data: dy_out_ref.as_slice_mut(), shape: sy, stride: contiguous_stride(sy) },
                attn: TensorMut { data: dattn_ref.as_slice_mut(), shape: sa, stride: contiguous_stride(sa) },
            },
        )
        .expect("ref fw");
    stream.synchronize().expect("");

    let bw_desc = SdpaBackwardDescriptor {
        batch_size: B,
        num_heads: H,
        query_len: Q,
        key_len: K,
        d_k: DK,
        d_v: DV,
        scale,
        element: ElementKind::F32,
    };
    let bw_plan =
        SdpaBackwardPlan::<f32>::select(&stream, &bw_desc, PlanPreference::default()).expect("");
    bw_plan
        .run(
            &stream,
            Workspace::None,
            SdpaBackwardArgs {
                q: TensorRef { data: dq_dev.as_slice(), shape: sq, stride: contiguous_stride(sq) },
                k: TensorRef { data: dk_dev.as_slice(), shape: sk, stride: contiguous_stride(sk) },
                v: TensorRef { data: dv_dev.as_slice(), shape: sv, stride: contiguous_stride(sv) },
                attn: TensorRef { data: dattn_ref.as_slice(), shape: sa, stride: contiguous_stride(sa) },
                dy: TensorRef { data: ddy_dev.as_slice(), shape: sy, stride: contiguous_stride(sy) },
                dscores_ws: TensorMut { data: dws_ref.as_slice_mut(), shape: sa, stride: contiguous_stride(sa) },
                dq: TensorMut { data: ddq_ref.as_slice_mut(), shape: sq, stride: contiguous_stride(sq) },
                dk: TensorMut { data: ddk_ref.as_slice_mut(), shape: sk, stride: contiguous_stride(sk) },
                dv: TensorMut { data: ddv_ref.as_slice_mut(), shape: sv, stride: contiguous_stride(sv) },
            },
        )
        .expect("ref bw");
    stream.synchronize().expect("");

    // -------- Flash FW + BW --------
    let mut dy_flash: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DV) as usize).expect("flash y");
    let mut dlse: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (B * H * Q) as usize).expect("flash lse");
    let mut dd_ws: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (B * H * Q) as usize).expect("flash d_ws");
    let mut ddq_flash: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DK) as usize).expect("flash dq");
    let mut ddk_flash: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (B * H * K * DK) as usize).expect("flash dk");
    let mut ddv_flash: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (B * H * K * DV) as usize).expect("flash dv");

    let f_fw_desc = FlashSdpaDescriptor::new(
        B,
        H,
        Q,
        K,
        DK,
        DV,
        scale,
        false,
        ElementKind::F32,
    );
    let f_fw_plan =
        FlashSdpaPlan::<f32>::select(&stream, &f_fw_desc, PlanPreference::default()).expect("");
    f_fw_plan
        .run(
            &stream,
            Workspace::None,
            FlashSdpaArgs {
                q: TensorRef { data: dq_dev.as_slice(), shape: sq, stride: contiguous_stride(sq) },
                k: TensorRef { data: dk_dev.as_slice(), shape: sk, stride: contiguous_stride(sk) },
                v: TensorRef { data: dv_dev.as_slice(), shape: sv, stride: contiguous_stride(sv) },
                y: TensorMut { data: dy_flash.as_slice_mut(), shape: sy, stride: contiguous_stride(sy) },
                lse: TensorMut { data: dlse.as_slice_mut(), shape: sl, stride: contiguous_stride(sl) },
                mask: None,
                            alibi_slopes: None,
            },
        )
        .expect("flash fw");
    stream.synchronize().expect("");

    // Phase 59b: descriptor is #[non_exhaustive]; use ::new constructor.
    let f_bw_desc = FlashSdpaBackwardDescriptor::new(
        B, H, Q, K, DK, DV, scale, false, ElementKind::F32,
    );
    let f_bw_plan =
        FlashSdpaBackwardPlan::<f32>::select(&stream, &f_bw_desc, PlanPreference::default())
            .expect("");
    f_bw_plan
        .run(
            &stream,
            Workspace::None,
            FlashSdpaBackwardArgs {
                q: TensorRef { data: dq_dev.as_slice(), shape: sq, stride: contiguous_stride(sq) },
                k: TensorRef { data: dk_dev.as_slice(), shape: sk, stride: contiguous_stride(sk) },
                v: TensorRef { data: dv_dev.as_slice(), shape: sv, stride: contiguous_stride(sv) },
                y: TensorRef { data: dy_flash.as_slice(), shape: sy, stride: contiguous_stride(sy) },
                lse: TensorRef { data: dlse.as_slice(), shape: sl, stride: contiguous_stride(sl) },
                dy: TensorRef { data: ddy_dev.as_slice(), shape: sy, stride: contiguous_stride(sy) },
                d_ws: TensorMut { data: dd_ws.as_slice_mut(), shape: sl, stride: contiguous_stride(sl) },
                dq: TensorMut { data: ddq_flash.as_slice_mut(), shape: sq, stride: contiguous_stride(sq) },
                dk: TensorMut { data: ddk_flash.as_slice_mut(), shape: sk, stride: contiguous_stride(sk) },
                dv: TensorMut { data: ddv_flash.as_slice_mut(), shape: sv, stride: contiguous_stride(sv) },
                lse_f32: None,
                alibi_slopes: None,
            },
        )
        .expect("flash bw");
    stream.synchronize().expect("");

    let mut g_dq = vec![0f32; (B * H * Q * DK) as usize];
    let mut g_dk = vec![0f32; (B * H * K * DK) as usize];
    let mut g_dv = vec![0f32; (B * H * K * DV) as usize];
    let mut r_dq = vec![0f32; (B * H * Q * DK) as usize];
    let mut r_dk = vec![0f32; (B * H * K * DK) as usize];
    let mut r_dv = vec![0f32; (B * H * K * DV) as usize];
    ddq_flash.copy_to_host(&mut g_dq).expect("");
    ddk_flash.copy_to_host(&mut g_dk).expect("");
    ddv_flash.copy_to_host(&mut g_dv).expect("");
    ddq_ref.copy_to_host(&mut r_dq).expect("");
    ddk_ref.copy_to_host(&mut r_dk).expect("");
    ddv_ref.copy_to_host(&mut r_dv).expect("");

    // BW: 64 * eps. Slightly looser than FW because dQ / dK / dV are
    // each built from two-stage online accumulations (P → dS → dQ).
    let tol = 64.0 * f32::EPSILON;
    for (label, got, refv) in [
        ("dQ", &g_dq[..], &r_dq[..]),
        ("dK", &g_dk[..], &r_dk[..]),
        ("dV", &g_dv[..], &r_dv[..]),
    ] {
        for i in 0..got.len() {
            let r = refv[i];
            let diff = (got[i] - r).abs();
            let t = (r.abs() * tol).max(tol);
            assert!(
                diff <= t,
                "f32 flash-sdpa-bw {label} @ {i}: diff={diff} flash={f} ref={r}",
                f = got[i]
            );
        }
    }
}

// ----------------------------------------------------------------------------
// f64
// ----------------------------------------------------------------------------

// f64 BW. Tile shape is templated on dtype (`TileShape<T>` in the CUH):
// f64 uses kBr=kBc=32 so the dQ BW SMEM stays under sm_89's 99 KiB
// per-block cap (~56 KiB at d_k=32). Q=128 / Br=32 => 4 q-blocks, still
// tiles cleanly. Compares against the naive Milestone 6.2 reference
// kernel at 64·EPS tolerance, matching the f32 BW test pattern.
#[test]
#[ignore]
fn flash_sdpa_backward_f64_basic() {
    let (ctx, stream) = setup();
    let q_h = gen_q_f64();
    let k_h = gen_k_f64();
    let v_h = gen_v_f64();
    let dy_h = gen_dy_f64();
    let scale = default_scale();

    let dq_dev = DeviceBuffer::from_slice(&ctx, &q_h).expect("up q");
    let dk_dev = DeviceBuffer::from_slice(&ctx, &k_h).expect("up k");
    let dv_dev = DeviceBuffer::from_slice(&ctx, &v_h).expect("up v");
    let ddy_dev = DeviceBuffer::from_slice(&ctx, &dy_h).expect("up dy");

    let sq = [B, H, Q, DK];
    let sk = [B, H, K, DK];
    let sv = [B, H, K, DV];
    let sa = [B, H, Q, K];
    let sy = [B, H, Q, DV];
    let sl = [B, H, Q];

    // -------- Reference: naive Sdpa FW+BW --------
    let mut dattn_ref: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * K) as usize).expect("ref attn");
    let mut dy_out_ref: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DV) as usize).expect("ref y");
    let mut dws_ref: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * K) as usize).expect("ref ws");
    let mut ddq_ref: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DK) as usize).expect("ref dq");
    let mut ddk_ref: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (B * H * K * DK) as usize).expect("ref dk");
    let mut ddv_ref: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (B * H * K * DV) as usize).expect("ref dv");

    let fw_desc = SdpaDescriptor {
        batch_size: B,
        num_heads: H,
        query_len: Q,
        key_len: K,
        d_k: DK,
        d_v: DV,
        scale,
        is_causal: false,
        has_mask: false,
        element: ElementKind::F64,
    };
    let fw_plan = SdpaPlan::<f64>::select(&stream, &fw_desc, PlanPreference::default()).expect("");
    fw_plan
        .run(
            &stream,
            Workspace::None,
            SdpaArgs {
                q: TensorRef { data: dq_dev.as_slice(), shape: sq, stride: contiguous_stride(sq) },
                k: TensorRef { data: dk_dev.as_slice(), shape: sk, stride: contiguous_stride(sk) },
                v: TensorRef { data: dv_dev.as_slice(), shape: sv, stride: contiguous_stride(sv) },
                mask: None,
                y: TensorMut { data: dy_out_ref.as_slice_mut(), shape: sy, stride: contiguous_stride(sy) },
                attn: TensorMut { data: dattn_ref.as_slice_mut(), shape: sa, stride: contiguous_stride(sa) },
            },
        )
        .expect("ref fw");
    stream.synchronize().expect("");

    let bw_desc = SdpaBackwardDescriptor {
        batch_size: B,
        num_heads: H,
        query_len: Q,
        key_len: K,
        d_k: DK,
        d_v: DV,
        scale,
        element: ElementKind::F64,
    };
    let bw_plan =
        SdpaBackwardPlan::<f64>::select(&stream, &bw_desc, PlanPreference::default()).expect("");
    bw_plan
        .run(
            &stream,
            Workspace::None,
            SdpaBackwardArgs {
                q: TensorRef { data: dq_dev.as_slice(), shape: sq, stride: contiguous_stride(sq) },
                k: TensorRef { data: dk_dev.as_slice(), shape: sk, stride: contiguous_stride(sk) },
                v: TensorRef { data: dv_dev.as_slice(), shape: sv, stride: contiguous_stride(sv) },
                attn: TensorRef { data: dattn_ref.as_slice(), shape: sa, stride: contiguous_stride(sa) },
                dy: TensorRef { data: ddy_dev.as_slice(), shape: sy, stride: contiguous_stride(sy) },
                dscores_ws: TensorMut { data: dws_ref.as_slice_mut(), shape: sa, stride: contiguous_stride(sa) },
                dq: TensorMut { data: ddq_ref.as_slice_mut(), shape: sq, stride: contiguous_stride(sq) },
                dk: TensorMut { data: ddk_ref.as_slice_mut(), shape: sk, stride: contiguous_stride(sk) },
                dv: TensorMut { data: ddv_ref.as_slice_mut(), shape: sv, stride: contiguous_stride(sv) },
            },
        )
        .expect("ref bw");
    stream.synchronize().expect("");

    // -------- Flash FW + BW --------
    let mut dy_flash: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DV) as usize).expect("flash y");
    let mut dlse: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (B * H * Q) as usize).expect("flash lse");
    let mut dd_ws: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (B * H * Q) as usize).expect("flash d_ws");
    let mut ddq_flash: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DK) as usize).expect("flash dq");
    let mut ddk_flash: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (B * H * K * DK) as usize).expect("flash dk");
    let mut ddv_flash: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (B * H * K * DV) as usize).expect("flash dv");

    let f_fw_desc = FlashSdpaDescriptor::new(
        B,
        H,
        Q,
        K,
        DK,
        DV,
        scale,
        false,
        ElementKind::F64,
    );
    let f_fw_plan =
        FlashSdpaPlan::<f64>::select(&stream, &f_fw_desc, PlanPreference::default()).expect("");
    f_fw_plan
        .run(
            &stream,
            Workspace::None,
            FlashSdpaArgs {
                q: TensorRef { data: dq_dev.as_slice(), shape: sq, stride: contiguous_stride(sq) },
                k: TensorRef { data: dk_dev.as_slice(), shape: sk, stride: contiguous_stride(sk) },
                v: TensorRef { data: dv_dev.as_slice(), shape: sv, stride: contiguous_stride(sv) },
                y: TensorMut { data: dy_flash.as_slice_mut(), shape: sy, stride: contiguous_stride(sy) },
                lse: TensorMut { data: dlse.as_slice_mut(), shape: sl, stride: contiguous_stride(sl) },
                mask: None,
                            alibi_slopes: None,
            },
        )
        .expect("flash fw");
    stream.synchronize().expect("");

    let f_bw_desc = FlashSdpaBackwardDescriptor::new(
        B, H, Q, K, DK, DV, scale, false, ElementKind::F64,
    );
    let f_bw_plan =
        FlashSdpaBackwardPlan::<f64>::select(&stream, &f_bw_desc, PlanPreference::default())
            .expect("");
    f_bw_plan
        .run(
            &stream,
            Workspace::None,
            FlashSdpaBackwardArgs {
                q: TensorRef { data: dq_dev.as_slice(), shape: sq, stride: contiguous_stride(sq) },
                k: TensorRef { data: dk_dev.as_slice(), shape: sk, stride: contiguous_stride(sk) },
                v: TensorRef { data: dv_dev.as_slice(), shape: sv, stride: contiguous_stride(sv) },
                y: TensorRef { data: dy_flash.as_slice(), shape: sy, stride: contiguous_stride(sy) },
                lse: TensorRef { data: dlse.as_slice(), shape: sl, stride: contiguous_stride(sl) },
                dy: TensorRef { data: ddy_dev.as_slice(), shape: sy, stride: contiguous_stride(sy) },
                d_ws: TensorMut { data: dd_ws.as_slice_mut(), shape: sl, stride: contiguous_stride(sl) },
                dq: TensorMut { data: ddq_flash.as_slice_mut(), shape: sq, stride: contiguous_stride(sq) },
                dk: TensorMut { data: ddk_flash.as_slice_mut(), shape: sk, stride: contiguous_stride(sk) },
                dv: TensorMut { data: ddv_flash.as_slice_mut(), shape: sv, stride: contiguous_stride(sv) },
                lse_f32: None,
                alibi_slopes: None,
            },
        )
        .expect("flash bw");
    stream.synchronize().expect("");

    let mut g_dq = vec![0f64; (B * H * Q * DK) as usize];
    let mut g_dk = vec![0f64; (B * H * K * DK) as usize];
    let mut g_dv = vec![0f64; (B * H * K * DV) as usize];
    let mut r_dq = vec![0f64; (B * H * Q * DK) as usize];
    let mut r_dk = vec![0f64; (B * H * K * DK) as usize];
    let mut r_dv = vec![0f64; (B * H * K * DV) as usize];
    ddq_flash.copy_to_host(&mut g_dq).expect("");
    ddk_flash.copy_to_host(&mut g_dk).expect("");
    ddv_flash.copy_to_host(&mut g_dv).expect("");
    ddq_ref.copy_to_host(&mut r_dq).expect("");
    ddk_ref.copy_to_host(&mut r_dk).expect("");
    ddv_ref.copy_to_host(&mut r_dv).expect("");

    let tol = 64.0 * f64::EPSILON;
    for (label, got, refv) in [
        ("dQ", &g_dq[..], &r_dq[..]),
        ("dK", &g_dk[..], &r_dk[..]),
        ("dV", &g_dv[..], &r_dv[..]),
    ] {
        for i in 0..got.len() {
            let r = refv[i];
            let diff = (got[i] - r).abs();
            let t = (r.abs() * tol).max(tol);
            assert!(
                diff <= t,
                "f64 flash-sdpa-bw {label} @ {i}: diff={diff} flash={f} ref={r}",
                f = got[i]
            );
        }
    }
}

// ----------------------------------------------------------------------------
// f16 and bf16 — share helper to keep things compact.
// ----------------------------------------------------------------------------

#[test]
#[ignore]
fn flash_sdpa_backward_f16_basic() {
    let (ctx, stream) = setup();
    let q_f64 = gen_q_f64();
    let k_f64 = gen_k_f64();
    let v_f64 = gen_v_f64();
    let dy_f64 = gen_dy_f64();
    let scale = default_scale();

    let q_h: Vec<f16> = q_f64.iter().map(|&v| f16::from_f64(v)).collect();
    let k_h: Vec<f16> = k_f64.iter().map(|&v| f16::from_f64(v)).collect();
    let v_h: Vec<f16> = v_f64.iter().map(|&v| f16::from_f64(v)).collect();
    let dy_h: Vec<f16> = dy_f64.iter().map(|&v| f16::from_f64(v)).collect();

    let dq_dev = DeviceBuffer::from_slice(&ctx, &q_h).expect("");
    let dk_dev = DeviceBuffer::from_slice(&ctx, &k_h).expect("");
    let dv_dev = DeviceBuffer::from_slice(&ctx, &v_h).expect("");
    let ddy_dev = DeviceBuffer::from_slice(&ctx, &dy_h).expect("");

    let sq = [B, H, Q, DK];
    let sk = [B, H, K, DK];
    let sv = [B, H, K, DV];
    let sa = [B, H, Q, K];
    let sy = [B, H, Q, DV];
    let sl = [B, H, Q];

    let mut dattn_ref: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, (B * H * Q * K) as usize).expect("");
    let mut dy_out_ref: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, (B * H * Q * DV) as usize).expect("");
    let mut dws_ref: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, (B * H * Q * K) as usize).expect("");
    let mut ddq_ref: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, (B * H * Q * DK) as usize).expect("");
    let mut ddk_ref: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, (B * H * K * DK) as usize).expect("");
    let mut ddv_ref: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, (B * H * K * DV) as usize).expect("");

    let fw_desc = SdpaDescriptor {
        batch_size: B, num_heads: H, query_len: Q, key_len: K,
        d_k: DK, d_v: DV, scale, is_causal: false, has_mask: false,
        element: ElementKind::F16,
    };
    let fw_plan = SdpaPlan::<f16>::select(&stream, &fw_desc, PlanPreference::default()).expect("");
    fw_plan.run(&stream, Workspace::None, SdpaArgs {
        q: TensorRef { data: dq_dev.as_slice(), shape: sq, stride: contiguous_stride(sq) },
        k: TensorRef { data: dk_dev.as_slice(), shape: sk, stride: contiguous_stride(sk) },
        v: TensorRef { data: dv_dev.as_slice(), shape: sv, stride: contiguous_stride(sv) },
        mask: None,
        y: TensorMut { data: dy_out_ref.as_slice_mut(), shape: sy, stride: contiguous_stride(sy) },
        attn: TensorMut { data: dattn_ref.as_slice_mut(), shape: sa, stride: contiguous_stride(sa) },
    }).expect("");
    stream.synchronize().expect("");

    let bw_desc = SdpaBackwardDescriptor {
        batch_size: B, num_heads: H, query_len: Q, key_len: K,
        d_k: DK, d_v: DV, scale, element: ElementKind::F16,
    };
    let bw_plan = SdpaBackwardPlan::<f16>::select(&stream, &bw_desc, PlanPreference::default()).expect("");
    bw_plan.run(&stream, Workspace::None, SdpaBackwardArgs {
        q: TensorRef { data: dq_dev.as_slice(), shape: sq, stride: contiguous_stride(sq) },
        k: TensorRef { data: dk_dev.as_slice(), shape: sk, stride: contiguous_stride(sk) },
        v: TensorRef { data: dv_dev.as_slice(), shape: sv, stride: contiguous_stride(sv) },
        attn: TensorRef { data: dattn_ref.as_slice(), shape: sa, stride: contiguous_stride(sa) },
        dy: TensorRef { data: ddy_dev.as_slice(), shape: sy, stride: contiguous_stride(sy) },
        dscores_ws: TensorMut { data: dws_ref.as_slice_mut(), shape: sa, stride: contiguous_stride(sa) },
        dq: TensorMut { data: ddq_ref.as_slice_mut(), shape: sq, stride: contiguous_stride(sq) },
        dk: TensorMut { data: ddk_ref.as_slice_mut(), shape: sk, stride: contiguous_stride(sk) },
        dv: TensorMut { data: ddv_ref.as_slice_mut(), shape: sv, stride: contiguous_stride(sv) },
    }).expect("");
    stream.synchronize().expect("");

    let mut dy_flash: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, (B * H * Q * DV) as usize).expect("");
    let mut dlse: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, (B * H * Q) as usize).expect("");
    let mut dd_ws: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, (B * H * Q) as usize).expect("");
    let mut ddq_flash: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, (B * H * Q * DK) as usize).expect("");
    let mut ddk_flash: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, (B * H * K * DK) as usize).expect("");
    let mut ddv_flash: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, (B * H * K * DV) as usize).expect("");

    let f_fw_desc = FlashSdpaDescriptor::new(
        B, H, Q, K, DK, DV, scale, false, ElementKind::F16,
    );
    let f_fw_plan = FlashSdpaPlan::<f16>::select(&stream, &f_fw_desc, PlanPreference::default()).expect("");
    f_fw_plan.run(&stream, Workspace::None, FlashSdpaArgs {
        q: TensorRef { data: dq_dev.as_slice(), shape: sq, stride: contiguous_stride(sq) },
        k: TensorRef { data: dk_dev.as_slice(), shape: sk, stride: contiguous_stride(sk) },
        v: TensorRef { data: dv_dev.as_slice(), shape: sv, stride: contiguous_stride(sv) },
        y: TensorMut { data: dy_flash.as_slice_mut(), shape: sy, stride: contiguous_stride(sy) },
        lse: TensorMut { data: dlse.as_slice_mut(), shape: sl, stride: contiguous_stride(sl) },
                mask: None,
            alibi_slopes: None,
    }).expect("");
    stream.synchronize().expect("");

    let f_bw_desc = FlashSdpaBackwardDescriptor::new(
        B, H, Q, K, DK, DV, scale, false, ElementKind::F16,
    );
    // Phase 59b made FA2 the default BW backend for f16/bf16. This test
    // validates the BESPOKE BW pipeline (deterministic, three-kernel,
    // f16 lse), so force the bespoke backend.
    let f_bw_pref = PlanPreference {
        prefer_backend: Some(BackendKind::Bespoke),
        ..Default::default()
    };
    let f_bw_plan = FlashSdpaBackwardPlan::<f16>::select(&stream, &f_bw_desc, f_bw_pref).expect("");
    f_bw_plan.run(&stream, Workspace::None, FlashSdpaBackwardArgs {
        q: TensorRef { data: dq_dev.as_slice(), shape: sq, stride: contiguous_stride(sq) },
        k: TensorRef { data: dk_dev.as_slice(), shape: sk, stride: contiguous_stride(sk) },
        v: TensorRef { data: dv_dev.as_slice(), shape: sv, stride: contiguous_stride(sv) },
        y: TensorRef { data: dy_flash.as_slice(), shape: sy, stride: contiguous_stride(sy) },
        lse: TensorRef { data: dlse.as_slice(), shape: sl, stride: contiguous_stride(sl) },
        dy: TensorRef { data: ddy_dev.as_slice(), shape: sy, stride: contiguous_stride(sy) },
        d_ws: TensorMut { data: dd_ws.as_slice_mut(), shape: sl, stride: contiguous_stride(sl) },
        dq: TensorMut { data: ddq_flash.as_slice_mut(), shape: sq, stride: contiguous_stride(sq) },
        dk: TensorMut { data: ddk_flash.as_slice_mut(), shape: sk, stride: contiguous_stride(sk) },
        dv: TensorMut { data: ddv_flash.as_slice_mut(), shape: sv, stride: contiguous_stride(sv) },
        lse_f32: None,
        alibi_slopes: None,
    }).expect("");
    stream.synchronize().expect("");

    let mut g_dq = vec![f16::ZERO; (B * H * Q * DK) as usize];
    let mut g_dk = vec![f16::ZERO; (B * H * K * DK) as usize];
    let mut g_dv = vec![f16::ZERO; (B * H * K * DV) as usize];
    let mut r_dq = vec![f16::ZERO; (B * H * Q * DK) as usize];
    let mut r_dk = vec![f16::ZERO; (B * H * K * DK) as usize];
    let mut r_dv = vec![f16::ZERO; (B * H * K * DV) as usize];
    ddq_flash.copy_to_host(&mut g_dq).expect("");
    ddk_flash.copy_to_host(&mut g_dk).expect("");
    ddv_flash.copy_to_host(&mut g_dv).expect("");
    ddq_ref.copy_to_host(&mut r_dq).expect("");
    ddk_ref.copy_to_host(&mut r_dk).expect("");
    ddv_ref.copy_to_host(&mut r_dv).expect("");

    let tol = 64.0 * F16_EPS;
    for (label, got, refv) in [
        ("dQ", &g_dq[..], &r_dq[..]),
        ("dK", &g_dk[..], &r_dk[..]),
        ("dV", &g_dv[..], &r_dv[..]),
    ] {
        for i in 0..got.len() {
            let r = refv[i].to_f32();
            let g = got[i].to_f32();
            let diff = (g - r).abs();
            let t = (r.abs() * tol).max(tol);
            assert!(
                diff <= t,
                "f16 flash-sdpa-bw {label} @ {i}: diff={diff} flash={g} ref={r}"
            );
        }
    }
}

#[test]
#[ignore]
fn flash_sdpa_backward_bf16_basic() {
    let (ctx, stream) = setup();
    let q_f64 = gen_q_f64();
    let k_f64 = gen_k_f64();
    let v_f64 = gen_v_f64();
    let dy_f64 = gen_dy_f64();
    let scale = default_scale();

    let q_h: Vec<bf16> = q_f64.iter().map(|&v| bf16::from_f64(v)).collect();
    let k_h: Vec<bf16> = k_f64.iter().map(|&v| bf16::from_f64(v)).collect();
    let v_h: Vec<bf16> = v_f64.iter().map(|&v| bf16::from_f64(v)).collect();
    let dy_h: Vec<bf16> = dy_f64.iter().map(|&v| bf16::from_f64(v)).collect();

    let dq_dev = DeviceBuffer::from_slice(&ctx, &q_h).expect("");
    let dk_dev = DeviceBuffer::from_slice(&ctx, &k_h).expect("");
    let dv_dev = DeviceBuffer::from_slice(&ctx, &v_h).expect("");
    let ddy_dev = DeviceBuffer::from_slice(&ctx, &dy_h).expect("");

    let sq = [B, H, Q, DK];
    let sk = [B, H, K, DK];
    let sv = [B, H, K, DV];
    let sa = [B, H, Q, K];
    let sy = [B, H, Q, DV];
    let sl = [B, H, Q];

    let mut dattn_ref: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, (B * H * Q * K) as usize).expect("");
    let mut dy_out_ref: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, (B * H * Q * DV) as usize).expect("");
    let mut dws_ref: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, (B * H * Q * K) as usize).expect("");
    let mut ddq_ref: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, (B * H * Q * DK) as usize).expect("");
    let mut ddk_ref: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, (B * H * K * DK) as usize).expect("");
    let mut ddv_ref: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, (B * H * K * DV) as usize).expect("");

    let fw_desc = SdpaDescriptor {
        batch_size: B, num_heads: H, query_len: Q, key_len: K,
        d_k: DK, d_v: DV, scale, is_causal: false, has_mask: false,
        element: ElementKind::Bf16,
    };
    let fw_plan = SdpaPlan::<bf16>::select(&stream, &fw_desc, PlanPreference::default()).expect("");
    fw_plan.run(&stream, Workspace::None, SdpaArgs {
        q: TensorRef { data: dq_dev.as_slice(), shape: sq, stride: contiguous_stride(sq) },
        k: TensorRef { data: dk_dev.as_slice(), shape: sk, stride: contiguous_stride(sk) },
        v: TensorRef { data: dv_dev.as_slice(), shape: sv, stride: contiguous_stride(sv) },
        mask: None,
        y: TensorMut { data: dy_out_ref.as_slice_mut(), shape: sy, stride: contiguous_stride(sy) },
        attn: TensorMut { data: dattn_ref.as_slice_mut(), shape: sa, stride: contiguous_stride(sa) },
    }).expect("");
    stream.synchronize().expect("");

    let bw_desc = SdpaBackwardDescriptor {
        batch_size: B, num_heads: H, query_len: Q, key_len: K,
        d_k: DK, d_v: DV, scale, element: ElementKind::Bf16,
    };
    let bw_plan = SdpaBackwardPlan::<bf16>::select(&stream, &bw_desc, PlanPreference::default()).expect("");
    bw_plan.run(&stream, Workspace::None, SdpaBackwardArgs {
        q: TensorRef { data: dq_dev.as_slice(), shape: sq, stride: contiguous_stride(sq) },
        k: TensorRef { data: dk_dev.as_slice(), shape: sk, stride: contiguous_stride(sk) },
        v: TensorRef { data: dv_dev.as_slice(), shape: sv, stride: contiguous_stride(sv) },
        attn: TensorRef { data: dattn_ref.as_slice(), shape: sa, stride: contiguous_stride(sa) },
        dy: TensorRef { data: ddy_dev.as_slice(), shape: sy, stride: contiguous_stride(sy) },
        dscores_ws: TensorMut { data: dws_ref.as_slice_mut(), shape: sa, stride: contiguous_stride(sa) },
        dq: TensorMut { data: ddq_ref.as_slice_mut(), shape: sq, stride: contiguous_stride(sq) },
        dk: TensorMut { data: ddk_ref.as_slice_mut(), shape: sk, stride: contiguous_stride(sk) },
        dv: TensorMut { data: ddv_ref.as_slice_mut(), shape: sv, stride: contiguous_stride(sv) },
    }).expect("");
    stream.synchronize().expect("");

    let mut dy_flash: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, (B * H * Q * DV) as usize).expect("");
    let mut dlse: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, (B * H * Q) as usize).expect("");
    let mut dd_ws: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, (B * H * Q) as usize).expect("");
    let mut ddq_flash: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, (B * H * Q * DK) as usize).expect("");
    let mut ddk_flash: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, (B * H * K * DK) as usize).expect("");
    let mut ddv_flash: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, (B * H * K * DV) as usize).expect("");

    let f_fw_desc = FlashSdpaDescriptor::new(
        B, H, Q, K, DK, DV, scale, false, ElementKind::Bf16,
    );
    let f_fw_plan = FlashSdpaPlan::<bf16>::select(&stream, &f_fw_desc, PlanPreference::default()).expect("");
    f_fw_plan.run(&stream, Workspace::None, FlashSdpaArgs {
        q: TensorRef { data: dq_dev.as_slice(), shape: sq, stride: contiguous_stride(sq) },
        k: TensorRef { data: dk_dev.as_slice(), shape: sk, stride: contiguous_stride(sk) },
        v: TensorRef { data: dv_dev.as_slice(), shape: sv, stride: contiguous_stride(sv) },
        y: TensorMut { data: dy_flash.as_slice_mut(), shape: sy, stride: contiguous_stride(sy) },
        lse: TensorMut { data: dlse.as_slice_mut(), shape: sl, stride: contiguous_stride(sl) },
                mask: None,
            alibi_slopes: None,
    }).expect("");
    stream.synchronize().expect("");

    let f_bw_desc = FlashSdpaBackwardDescriptor::new(
        B, H, Q, K, DK, DV, scale, false, ElementKind::Bf16,
    );
    // Phase 59b made FA2 the default BW backend for f16/bf16. This test
    // validates the BESPOKE BW pipeline (deterministic, three-kernel,
    // bf16 lse), so force the bespoke backend.
    let f_bw_pref = PlanPreference {
        prefer_backend: Some(BackendKind::Bespoke),
        ..Default::default()
    };
    let f_bw_plan = FlashSdpaBackwardPlan::<bf16>::select(&stream, &f_bw_desc, f_bw_pref).expect("");
    f_bw_plan.run(&stream, Workspace::None, FlashSdpaBackwardArgs {
        q: TensorRef { data: dq_dev.as_slice(), shape: sq, stride: contiguous_stride(sq) },
        k: TensorRef { data: dk_dev.as_slice(), shape: sk, stride: contiguous_stride(sk) },
        v: TensorRef { data: dv_dev.as_slice(), shape: sv, stride: contiguous_stride(sv) },
        y: TensorRef { data: dy_flash.as_slice(), shape: sy, stride: contiguous_stride(sy) },
        lse: TensorRef { data: dlse.as_slice(), shape: sl, stride: contiguous_stride(sl) },
        dy: TensorRef { data: ddy_dev.as_slice(), shape: sy, stride: contiguous_stride(sy) },
        d_ws: TensorMut { data: dd_ws.as_slice_mut(), shape: sl, stride: contiguous_stride(sl) },
        dq: TensorMut { data: ddq_flash.as_slice_mut(), shape: sq, stride: contiguous_stride(sq) },
        dk: TensorMut { data: ddk_flash.as_slice_mut(), shape: sk, stride: contiguous_stride(sk) },
        dv: TensorMut { data: ddv_flash.as_slice_mut(), shape: sv, stride: contiguous_stride(sv) },
        lse_f32: None,
        alibi_slopes: None,
    }).expect("");
    stream.synchronize().expect("");

    let mut g_dq = vec![bf16::ZERO; (B * H * Q * DK) as usize];
    let mut g_dk = vec![bf16::ZERO; (B * H * K * DK) as usize];
    let mut g_dv = vec![bf16::ZERO; (B * H * K * DV) as usize];
    let mut r_dq = vec![bf16::ZERO; (B * H * Q * DK) as usize];
    let mut r_dk = vec![bf16::ZERO; (B * H * K * DK) as usize];
    let mut r_dv = vec![bf16::ZERO; (B * H * K * DV) as usize];
    ddq_flash.copy_to_host(&mut g_dq).expect("");
    ddk_flash.copy_to_host(&mut g_dk).expect("");
    ddv_flash.copy_to_host(&mut g_dv).expect("");
    ddq_ref.copy_to_host(&mut r_dq).expect("");
    ddk_ref.copy_to_host(&mut r_dk).expect("");
    ddv_ref.copy_to_host(&mut r_dv).expect("");

    let tol = 64.0 * BF16_EPS;
    for (label, got, refv) in [
        ("dQ", &g_dq[..], &r_dq[..]),
        ("dK", &g_dk[..], &r_dk[..]),
        ("dV", &g_dv[..], &r_dv[..]),
    ] {
        for i in 0..got.len() {
            let r = refv[i].to_f32();
            let g = got[i].to_f32();
            let diff = (g - r).abs();
            let t = (r.abs() * tol).max(tol);
            assert!(
                diff <= t,
                "bf16 flash-sdpa-bw {label} @ {i}: diff={diff} flash={g} ref={r}"
            );
        }
    }
}
