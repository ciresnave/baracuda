//! Real-GPU smoke test for `RingAttentionPlan` (Phase 56).
//!
//! Validates the single-rank degenerate case (`world_size == 1`)
//! against [`FlashSdpaPlan`] as ground truth. Multi-rank validation
//! requires 2+ GPUs and is deferred — a scaffold test is included
//! but marked `requires 2+ GPUs`.
//!
//! `#[ignore]` by default — requires a real CUDA device + NCCL.

#![cfg(feature = "ring_attention")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, FlashSdpaArgs, FlashSdpaDescriptor, FlashSdpaPlan,
    PlanPreference, RingAttentionArgs, RingAttentionDescriptor, RingAttentionPlan,
    TensorMut, TensorRef, Workspace,
};
use baracuda_nccl::Communicator;
use half::{bf16, f16};

// Tier 1: head_dim = 128.
const B: i32 = 1;
const H: i32 = 2;
const Q_LOCAL: i32 = 128; // 2 q-blocks of 64
const K_CHUNK: i32 = 128; // 2 k-blocks of 64
const HEAD_DIM: i32 = 128;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn default_scale() -> f32 {
    1.0 / (HEAD_DIM as f32).sqrt()
}

// Synthetic Q / K / V inputs in f64 so we can convert losslessly to
// either f16 or bf16. Slightly smaller scale (0.3) to keep the
// softmax exponents in a numerically friendly range for f16.
fn gen_qkv_f64(seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let nq = (B * H * Q_LOCAL * HEAD_DIM) as usize;
    let nk = (B * H * K_CHUNK * HEAD_DIM) as usize;
    let q = (0..nq)
        .map(|i| (((i as u64 + seed) as f64) * 0.0137 - 0.5).sin() * 0.3)
        .collect();
    let k = (0..nk)
        .map(|i| (((i as u64 + seed) as f64) * 0.0171 + 0.2).cos() * 0.3)
        .collect();
    let v = (0..nk)
        .map(|i| (((i as u64 + seed) as f64) * 0.0113 - 0.1).sin() * 0.3)
        .collect();
    (q, k, v)
}

// f16 single-rank degenerate case — should match FlashSdpaPlan output
// within streaming-softmax tolerance.
#[test]
#[ignore]
fn ring_attention_f16_single_rank_matches_flash_sdpa() {
    let (ctx, stream) = setup();
    let scale = default_scale();
    let (q_f64, k_f64, v_f64) = gen_qkv_f64(0);

    let q_h: Vec<f16> = q_f64.iter().map(|&v| f16::from_f64(v)).collect();
    let k_h: Vec<f16> = k_f64.iter().map(|&v| f16::from_f64(v)).collect();
    let v_h: Vec<f16> = v_f64.iter().map(|&v| f16::from_f64(v)).collect();

    let dq = DeviceBuffer::from_slice(&ctx, &q_h).expect("up q");
    let dk = DeviceBuffer::from_slice(&ctx, &k_h).expect("up k");
    let dv = DeviceBuffer::from_slice(&ctx, &v_h).expect("up v");

    let sq = [B, H, Q_LOCAL, HEAD_DIM];
    let sk = [B, H, K_CHUNK, HEAD_DIM];
    let sv = [B, H, K_CHUNK, HEAD_DIM];
    let sy = [B, H, Q_LOCAL, HEAD_DIM];
    let sl = [B, H, Q_LOCAL];

    // Reference: FlashSdpaPlan (this is the world_size=1 degenerate
    // target — Ring Attention math reduces to FlashAttention when
    // there's only one rank and hence one K/V chunk).
    let mut dy_ref: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q_LOCAL * HEAD_DIM) as usize).expect("alloc y_ref");
    let mut dlse_ref: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q_LOCAL) as usize).expect("alloc lse_ref");
    let flash_desc = FlashSdpaDescriptor::new(
        B,
        H,
        Q_LOCAL,
        K_CHUNK,
        HEAD_DIM,
        HEAD_DIM,
        scale,
        false,
        ElementKind::F16,
    );
    let flash_plan = FlashSdpaPlan::<f16>::select(
        &stream,
        &flash_desc,
        PlanPreference::default(),
    )
    .expect("flash sel");
    flash_plan
        .run(
            &stream,
            Workspace::None,
            FlashSdpaArgs {
                q: TensorRef {
                    data: dq.as_slice(),
                    shape: sq,
                    stride: contiguous_stride(sq),
                },
                k: TensorRef {
                    data: dk.as_slice(),
                    shape: sk,
                    stride: contiguous_stride(sk),
                },
                v: TensorRef {
                    data: dv.as_slice(),
                    shape: sv,
                    stride: contiguous_stride(sv),
                },
                y: TensorMut {
                    data: dy_ref.as_slice_mut(),
                    shape: sy,
                    stride: contiguous_stride(sy),
                },
                lse: TensorMut {
                    data: dlse_ref.as_slice_mut(),
                    shape: sl,
                    stride: contiguous_stride(sl),
                },
                mask: None,
                            alibi_slopes: None,
            },
        )
        .expect("flash run");
    stream.synchronize().expect("sync flash");

    // Ring Attention — single-rank communicator.
    let comm = match Communicator::new_single_gpu(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("NCCL not available on this host; skipping ({:?})", e);
            return;
        }
    };
    assert_eq!(comm.world_size(), 1);
    assert_eq!(comm.rank(), 0);

    let ring_desc = RingAttentionDescriptor {
        batch_size: B,
        num_heads: H,
        query_len: Q_LOCAL,
        key_len: K_CHUNK,
        head_dim: HEAD_DIM,
        scale,
        is_causal: false,
        element: ElementKind::F16,
    };
    let ring_plan = RingAttentionPlan::<f16>::select(
        &stream,
        &ring_desc,
        PlanPreference::default(),
    )
    .expect("ring sel");

    let kv_elems = ring_plan.kv_scratch_elements();
    let acc_bytes = ring_plan.accumulator_scratch_bytes();
    // Stage K then V into kv_scratch_a (the caller-stage contract).
    let mut staged: Vec<f16> = Vec::with_capacity(kv_elems);
    staged.extend_from_slice(&k_h);
    staged.extend_from_slice(&v_h);
    assert_eq!(staged.len(), kv_elems);
    let mut dkv_a: DeviceBuffer<f16> =
        DeviceBuffer::from_slice(&ctx, &staged).expect("alloc kv_a");
    let mut dkv_b: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, kv_elems).expect("alloc kv_b");
    let mut dacc: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, acc_bytes).expect("alloc acc");

    let mut dy_ring: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q_LOCAL * HEAD_DIM) as usize).expect("alloc y_ring");
    let mut dlse_ring: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q_LOCAL) as usize).expect("alloc lse_ring");

    ring_plan
        .run(
            &stream,
            &comm,
            RingAttentionArgs {
                q: TensorRef {
                    data: dq.as_slice(),
                    shape: sq,
                    stride: contiguous_stride(sq),
                },
                y: TensorMut {
                    data: dy_ring.as_slice_mut(),
                    shape: sy,
                    stride: contiguous_stride(sy),
                },
                lse: Some(TensorMut {
                    data: dlse_ring.as_slice_mut(),
                    shape: sl,
                    stride: contiguous_stride(sl),
                }),
                kv_scratch_a: dkv_a.as_slice_mut(),
                kv_scratch_b: dkv_b.as_slice_mut(),
                accumulator_scratch: dacc.as_slice_mut(),
            },
        )
        .expect("ring run");
    stream.synchronize().expect("sync ring");

    let mut got = vec![f16::ZERO; (B * H * Q_LOCAL * HEAD_DIM) as usize];
    let mut refv = vec![f16::ZERO; (B * H * Q_LOCAL * HEAD_DIM) as usize];
    dy_ring.copy_to_host(&mut got).expect("dl ring");
    dy_ref.copy_to_host(&mut refv).expect("dl ref");

    // Tolerance: f16 streaming-softmax. Both kernels do f32 accum,
    // but the float-summation order differs (Ring's per-tile fold has
    // an extra alpha-rescale path even at single-rank because the
    // accumulator starts at -INF rather than initialized in-tile).
    // ~5e-3 absolute is a tight but realistic bound for the chosen
    // input scale (0.3); raise modestly if hardware variance bites.
    let tol_abs: f32 = 5e-3;
    let mut max_diff = 0.0f32;
    for i in 0..got.len() {
        let g = got[i].to_f32();
        let r = refv[i].to_f32();
        let diff = (g - r).abs();
        max_diff = max_diff.max(diff);
        assert!(
            diff <= tol_abs,
            "f16 ring vs flash @ {i}: diff={diff} ring={g} flash={r}",
        );
    }
    eprintln!("ring_attention_f16_single_rank: max abs diff = {:.6e}", max_diff);
}

// bf16 single-rank degenerate case.
#[test]
#[ignore]
fn ring_attention_bf16_single_rank_matches_flash_sdpa() {
    let (ctx, stream) = setup();
    let scale = default_scale();
    let (q_f64, k_f64, v_f64) = gen_qkv_f64(7);

    let q_h: Vec<bf16> = q_f64.iter().map(|&v| bf16::from_f64(v)).collect();
    let k_h: Vec<bf16> = k_f64.iter().map(|&v| bf16::from_f64(v)).collect();
    let v_h: Vec<bf16> = v_f64.iter().map(|&v| bf16::from_f64(v)).collect();

    let dq = DeviceBuffer::from_slice(&ctx, &q_h).expect("up q");
    let dk = DeviceBuffer::from_slice(&ctx, &k_h).expect("up k");
    let dv = DeviceBuffer::from_slice(&ctx, &v_h).expect("up v");

    let sq = [B, H, Q_LOCAL, HEAD_DIM];
    let sk = [B, H, K_CHUNK, HEAD_DIM];
    let sv = [B, H, K_CHUNK, HEAD_DIM];
    let sy = [B, H, Q_LOCAL, HEAD_DIM];
    let sl = [B, H, Q_LOCAL];

    let mut dy_ref: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q_LOCAL * HEAD_DIM) as usize).expect("alloc y_ref");
    let mut dlse_ref: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q_LOCAL) as usize).expect("alloc lse_ref");
    let flash_desc = FlashSdpaDescriptor::new(
        B,
        H,
        Q_LOCAL,
        K_CHUNK,
        HEAD_DIM,
        HEAD_DIM,
        scale,
        false,
        ElementKind::Bf16,
    );
    let flash_plan = FlashSdpaPlan::<bf16>::select(
        &stream,
        &flash_desc,
        PlanPreference::default(),
    )
    .expect("flash sel");
    flash_plan
        .run(
            &stream,
            Workspace::None,
            FlashSdpaArgs {
                q: TensorRef {
                    data: dq.as_slice(),
                    shape: sq,
                    stride: contiguous_stride(sq),
                },
                k: TensorRef {
                    data: dk.as_slice(),
                    shape: sk,
                    stride: contiguous_stride(sk),
                },
                v: TensorRef {
                    data: dv.as_slice(),
                    shape: sv,
                    stride: contiguous_stride(sv),
                },
                y: TensorMut {
                    data: dy_ref.as_slice_mut(),
                    shape: sy,
                    stride: contiguous_stride(sy),
                },
                lse: TensorMut {
                    data: dlse_ref.as_slice_mut(),
                    shape: sl,
                    stride: contiguous_stride(sl),
                },
                mask: None,
                            alibi_slopes: None,
            },
        )
        .expect("flash run");
    stream.synchronize().expect("sync flash");

    let comm = match Communicator::new_single_gpu(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("NCCL not available on this host; skipping ({:?})", e);
            return;
        }
    };

    let ring_desc = RingAttentionDescriptor {
        batch_size: B,
        num_heads: H,
        query_len: Q_LOCAL,
        key_len: K_CHUNK,
        head_dim: HEAD_DIM,
        scale,
        is_causal: false,
        element: ElementKind::Bf16,
    };
    let ring_plan = RingAttentionPlan::<bf16>::select(
        &stream,
        &ring_desc,
        PlanPreference::default(),
    )
    .expect("ring sel");

    let kv_elems = ring_plan.kv_scratch_elements();
    let acc_bytes = ring_plan.accumulator_scratch_bytes();
    let mut staged: Vec<bf16> = Vec::with_capacity(kv_elems);
    staged.extend_from_slice(&k_h);
    staged.extend_from_slice(&v_h);
    let mut dkv_a: DeviceBuffer<bf16> =
        DeviceBuffer::from_slice(&ctx, &staged).expect("alloc kv_a");
    let mut dkv_b: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, kv_elems).expect("alloc kv_b");
    let mut dacc: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, acc_bytes).expect("alloc acc");

    let mut dy_ring: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q_LOCAL * HEAD_DIM) as usize).expect("alloc y_ring");

    ring_plan
        .run(
            &stream,
            &comm,
            RingAttentionArgs {
                q: TensorRef {
                    data: dq.as_slice(),
                    shape: sq,
                    stride: contiguous_stride(sq),
                },
                y: TensorMut {
                    data: dy_ring.as_slice_mut(),
                    shape: sy,
                    stride: contiguous_stride(sy),
                },
                lse: None,
                kv_scratch_a: dkv_a.as_slice_mut(),
                kv_scratch_b: dkv_b.as_slice_mut(),
                accumulator_scratch: dacc.as_slice_mut(),
            },
        )
        .expect("ring run");
    stream.synchronize().expect("sync ring");

    let mut got = vec![bf16::ZERO; (B * H * Q_LOCAL * HEAD_DIM) as usize];
    let mut refv = vec![bf16::ZERO; (B * H * Q_LOCAL * HEAD_DIM) as usize];
    dy_ring.copy_to_host(&mut got).expect("dl ring");
    dy_ref.copy_to_host(&mut refv).expect("dl ref");

    // bf16 has lower precision than f16 in the mantissa (~7 bits vs
    // 10) so the absolute tolerance is looser.
    let tol_abs: f32 = 2e-2;
    let mut max_diff = 0.0f32;
    for i in 0..got.len() {
        let g = got[i].to_f32();
        let r = refv[i].to_f32();
        let diff = (g - r).abs();
        max_diff = max_diff.max(diff);
        assert!(
            diff <= tol_abs,
            "bf16 ring vs flash @ {i}: diff={diff} ring={g} flash={r}",
        );
    }
    eprintln!("ring_attention_bf16_single_rank: max abs diff = {:.6e}", max_diff);
}

// Single-rank causal-mask check — also validates that the
// q_global_base + k_global_base parameters thread through the kernel
// correctly (in single-rank both are zero, so this is a baseline
// causal check rather than a cross-rank index check).
#[test]
#[ignore]
fn ring_attention_f16_single_rank_causal() {
    let (ctx, stream) = setup();
    let scale = default_scale();
    let (q_f64, k_f64, v_f64) = gen_qkv_f64(42);

    let q_h: Vec<f16> = q_f64.iter().map(|&v| f16::from_f64(v)).collect();
    let k_h: Vec<f16> = k_f64.iter().map(|&v| f16::from_f64(v)).collect();
    let v_h: Vec<f16> = v_f64.iter().map(|&v| f16::from_f64(v)).collect();

    let dq = DeviceBuffer::from_slice(&ctx, &q_h).expect("up q");
    let dk = DeviceBuffer::from_slice(&ctx, &k_h).expect("up k");
    let dv = DeviceBuffer::from_slice(&ctx, &v_h).expect("up v");

    let sq = [B, H, Q_LOCAL, HEAD_DIM];
    let sk = [B, H, K_CHUNK, HEAD_DIM];
    let sv = [B, H, K_CHUNK, HEAD_DIM];
    let sy = [B, H, Q_LOCAL, HEAD_DIM];
    let sl = [B, H, Q_LOCAL];

    let mut dy_ref: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q_LOCAL * HEAD_DIM) as usize).expect("alloc y_ref");
    let mut dlse_ref: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q_LOCAL) as usize).expect("alloc lse_ref");
    let flash_desc = FlashSdpaDescriptor::new(
        B,
        H,
        Q_LOCAL,
        K_CHUNK,
        HEAD_DIM,
        HEAD_DIM,
        scale,
        true,
        ElementKind::F16,
    );
    let flash_plan = FlashSdpaPlan::<f16>::select(
        &stream,
        &flash_desc,
        PlanPreference::default(),
    )
    .expect("flash sel");
    flash_plan
        .run(
            &stream,
            Workspace::None,
            FlashSdpaArgs {
                q: TensorRef {
                    data: dq.as_slice(),
                    shape: sq,
                    stride: contiguous_stride(sq),
                },
                k: TensorRef {
                    data: dk.as_slice(),
                    shape: sk,
                    stride: contiguous_stride(sk),
                },
                v: TensorRef {
                    data: dv.as_slice(),
                    shape: sv,
                    stride: contiguous_stride(sv),
                },
                y: TensorMut {
                    data: dy_ref.as_slice_mut(),
                    shape: sy,
                    stride: contiguous_stride(sy),
                },
                lse: TensorMut {
                    data: dlse_ref.as_slice_mut(),
                    shape: sl,
                    stride: contiguous_stride(sl),
                },
                mask: None,
                            alibi_slopes: None,
            },
        )
        .expect("flash run");
    stream.synchronize().expect("sync flash");

    let comm = match Communicator::new_single_gpu(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("NCCL not available on this host; skipping ({:?})", e);
            return;
        }
    };

    let ring_desc = RingAttentionDescriptor {
        batch_size: B,
        num_heads: H,
        query_len: Q_LOCAL,
        key_len: K_CHUNK,
        head_dim: HEAD_DIM,
        scale,
        is_causal: true,
        element: ElementKind::F16,
    };
    let ring_plan = RingAttentionPlan::<f16>::select(
        &stream,
        &ring_desc,
        PlanPreference::default(),
    )
    .expect("ring sel");

    let kv_elems = ring_plan.kv_scratch_elements();
    let acc_bytes = ring_plan.accumulator_scratch_bytes();
    let mut staged: Vec<f16> = Vec::with_capacity(kv_elems);
    staged.extend_from_slice(&k_h);
    staged.extend_from_slice(&v_h);
    let mut dkv_a: DeviceBuffer<f16> =
        DeviceBuffer::from_slice(&ctx, &staged).expect("alloc kv_a");
    let mut dkv_b: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, kv_elems).expect("alloc kv_b");
    let mut dacc: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, acc_bytes).expect("alloc acc");

    let mut dy_ring: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q_LOCAL * HEAD_DIM) as usize).expect("alloc y_ring");

    ring_plan
        .run(
            &stream,
            &comm,
            RingAttentionArgs {
                q: TensorRef {
                    data: dq.as_slice(),
                    shape: sq,
                    stride: contiguous_stride(sq),
                },
                y: TensorMut {
                    data: dy_ring.as_slice_mut(),
                    shape: sy,
                    stride: contiguous_stride(sy),
                },
                lse: None,
                kv_scratch_a: dkv_a.as_slice_mut(),
                kv_scratch_b: dkv_b.as_slice_mut(),
                accumulator_scratch: dacc.as_slice_mut(),
            },
        )
        .expect("ring run");
    stream.synchronize().expect("sync ring");

    let mut got = vec![f16::ZERO; (B * H * Q_LOCAL * HEAD_DIM) as usize];
    let mut refv = vec![f16::ZERO; (B * H * Q_LOCAL * HEAD_DIM) as usize];
    dy_ring.copy_to_host(&mut got).expect("dl ring");
    dy_ref.copy_to_host(&mut refv).expect("dl ref");

    let tol_abs: f32 = 5e-3;
    for i in 0..got.len() {
        let g = got[i].to_f32();
        let r = refv[i].to_f32();
        let diff = (g - r).abs();
        assert!(
            diff <= tol_abs,
            "f16 ring causal vs flash causal @ {i}: diff={diff}",
        );
    }
}

// Multi-rank scaffold — requires 2+ GPUs and is deferred. The body
// builds the right shapes and exercises the descriptor / args API
// but does not actually launch a multi-rank communicator. Multi-rank
// validation lands in a future hardware-enabled session.
#[test]
#[ignore = "requires 2+ GPUs and a multi-process NCCL bringup"]
fn ring_attention_multi_rank_scaffold() {
    // This test exists for API discoverability. The actual multi-rank
    // run would:
    //  1. Spawn world_size processes (or threads with separate ctxs).
    //  2. Generate a UniqueId on rank 0; broadcast to others.
    //  3. Each process: Communicator::new_with_id(id, world_size, rank).
    //  4. Each rank: stage its Q-slice + K-chunk + V-chunk into device
    //     buffers, allocate kv_scratch + accumulator scratch.
    //  5. Each rank: call RingAttentionPlan::run with its comm.
    //  6. Gather y across ranks and compare against a reference
    //     `FlashSdpaPlan` run on the FULL global K/V on rank 0.
    //
    // The plan itself is identical to the single-rank smoke test;
    // only the test harness changes. Tracking as `requires 2+ GPUs`
    // and gating until the hardware is available.
}
