//! Real-GPU smoke test for `MoePlan::ScalarGguf` — Phase 8 Milestone 8.5.
//!
//! Tiny fixture (T=4 tokens, num_experts=2, top_k=1, D_model=32 — must
//! be a multiple of the Q8_0 block size, D_expert=32) with Q8_0-packed
//! expert weights. Verifies the kernel output against a CPU reference
//! that dequantizes each block and dispatches manually.
//!
//! ## Routing convention (Phase 15.3 re-derivation)
//!
//! The vendored Fuel/llama.cpp kernel writes `out[token_id, row] = v`
//! *without* accumulation — the same address gets overwritten if
//! multiple `m_idx` entries share a `token_id`. CUDA block execution
//! order is non-deterministic, so when `top_k > 1` and several experts
//! route to the same token, the result of any cell is the value from
//! whichever expert block happened to write last. To get a
//! deterministic, race-free comparison against a CPU reference, this
//! fixture uses `top_k = 1` with each token routed to a single,
//! distinct expert (token `t` → expert `t % num_experts`). Production
//! callers that need per-token mixing across `top_k > 1` experts must
//! pre-allocate an [`MoePlan::Wmma`]-style scratch + accumulate
//! upstream of the kernel.
//!
//! Marked `#[ignore]` per project convention.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BlockQ8_0, ElementKind, GgufBlockFormat, MoeArgs, MoeDescriptor, MoePlan,
    MoeVariant, PlanPreference, TensorMut, TensorRef, Workspace, U8,
};
use half::f16;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn moe_scalar_gguf_q8_0_small_fixture() {
    let (ctx, stream) = setup();

    const T: i32 = 4;
    const NE: i32 = 2;
    const K: i32 = 1; // top_k = 1: each token routes to exactly one expert (no race).
    const DM: i32 = 32; // = QK8_0 block size, so D_model = 1 block per row
    const DE: i32 = 32;

    // Activations: f32 ramp.
    let mut acts_host = vec![0.0f32; (T * DM) as usize];
    for i in 0..T {
        for j in 0..DM {
            acts_host[(i * DM + j) as usize] = 0.1 * (i as f32) + 0.01 * (j as f32);
        }
    }

    // Per-expert weights as Q8_0 blocks. One block per (expert, n) row.
    let mut blocks_host = Vec::<BlockQ8_0>::with_capacity((NE * DE) as usize);
    // To keep the reference exact, encode weights such that
    //   w[e, n, k] = scale_e * q[e, n, k]
    // with a known per-row scale.
    let mut weights_dequant = vec![0.0f32; (NE * DE * DM) as usize];
    for e in 0..NE {
        for n in 0..DE {
            let scale = 0.001 * (e as f32 + 1.0);
            let mut qs = [0i8; 32];
            for k in 0..DM {
                let q = (((n - k) as i32) % 64 - 32) as i8;
                qs[k as usize] = q;
                weights_dequant[((e * DE + n) * DM + k) as usize] = scale * (q as f32);
            }
            blocks_host.push(BlockQ8_0 {
                d: f16::from_f32(scale).to_bits(),
                qs,
            });
        }
    }
    // Pack to bytes.
    let mut weights_bytes = Vec::<u8>::with_capacity(blocks_host.len() * 34);
    for b in &blocks_host {
        let bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(
                (b as *const BlockQ8_0) as *const u8,
                core::mem::size_of::<BlockQ8_0>(),
            )
        };
        weights_bytes.extend_from_slice(bytes);
    }
    let weights_u8: Vec<U8> = weights_bytes.iter().copied().map(U8).collect();
    let weights_bytes_len = weights_u8.len() as i32;

    // Routing: token t -> expert (t % NE), top_k=1. One m_idx per token,
    // so every output cell is written exactly once (no last-write-wins
    // race). `topk_weights` is per-token (length T) and read by the
    // kernel via `topk_weights[token_id]`.
    let mut sorted_token_ids = Vec::<i32>::with_capacity(T as usize);
    let mut flat_expert_ids = Vec::<i32>::with_capacity(T as usize);
    let mut topk_weight_flat = Vec::<f32>::with_capacity(T as usize);
    for t in 0..T {
        sorted_token_ids.push(t);
        flat_expert_ids.push(t % NE);
        topk_weight_flat.push(0.6); // constant mixing factor
    }
    let m_total = sorted_token_ids.len() as i32;

    // Reference: scalar GGUF kernel writes
    //   out[token_id, n] = topk_weights[token_id] * Σ_k act[token_id, k] * w[expert, n, k]
    // for each (m_idx -> (token_id, expert)) pair. With top_k=1
    // routing, every cell is written exactly once.
    let mut expected = vec![0.0f32; (T * DE) as usize];
    for m in 0..m_total {
        let token_id = sorted_token_ids[m as usize];
        let expert = flat_expert_ids[m as usize];
        let scale = topk_weight_flat[token_id as usize];
        for n in 0..DE {
            let mut acc = 0.0f32;
            for k in 0..DM {
                let a = acts_host[(token_id * DM + k) as usize];
                let w = weights_dequant[((expert * DE + n) * DM + k) as usize];
                acc += a * w;
            }
            expected[(token_id * DE + n) as usize] = scale * acc;
        }
    }

    // Upload device buffers.
    let acts_dev: DeviceBuffer<f32> = DeviceBuffer::from_slice(&ctx, &acts_host).expect("up acts");
    let weights_dev: DeviceBuffer<U8> = DeviceBuffer::from_slice(&ctx, &weights_u8).expect("up w");
    let sorted_dev: DeviceBuffer<i32> = DeviceBuffer::from_slice(&ctx, &sorted_token_ids).expect("up s");
    let eids_dev: DeviceBuffer<i32> = DeviceBuffer::from_slice(&ctx, &flat_expert_ids).expect("up eids");
    let tk_dev: DeviceBuffer<f32> = DeviceBuffer::from_slice(&ctx, &topk_weight_flat).expect("up tk");
    let mut out_dev: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, (T * DE) as usize).expect("alloc out");

    let desc = MoeDescriptor {
        num_tokens: T,
        num_experts: NE,
        top_k: K,
        d_model: DM,
        d_expert: DE,
        variant: MoeVariant::ScalarGguf,
        block_format: Some(GgufBlockFormat::Q8_0),
        element: ElementKind::F32,
        is_prefill: true,
    };
    let plan = MoePlan::select(&stream, &desc, PlanPreference::default()).expect("select");

    // Placeholders for unused fields.
    let expert_weights_dev: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (T * K) as usize).expect("ew");

    let args = MoeArgs::<f32> {
        activations: TensorRef {
            data: acts_dev.as_slice(),
            shape: [T, DM],
            stride: contiguous_stride([T, DM]),
        },
        expert_indices: TensorRef {
            data: eids_dev.as_slice(),
            shape: [T, K],
            stride: contiguous_stride([T, K]),
        },
        expert_weights: TensorRef {
            data: expert_weights_dev.as_slice(),
            shape: [T, K],
            stride: contiguous_stride([T, K]),
        },
        sorted_token_ids: TensorRef {
            data: sorted_dev.as_slice(),
            shape: [m_total],
            stride: contiguous_stride([m_total]),
        },
        flat_expert_ids: TensorRef {
            data: eids_dev.as_slice(),
            shape: [m_total],
            stride: contiguous_stride([m_total]),
        },
        topk_weight_flat: Some(TensorRef {
            data: tk_dev.as_slice(),
            shape: [T],
            stride: contiguous_stride([T]),
        }),
        expert_matrices: TensorRef {
            data: weights_dev.as_slice(),
            shape: [weights_bytes_len],
            stride: contiguous_stride([weights_bytes_len]),
        },
        output: TensorMut {
            data: out_dev.as_slice_mut(),
            shape: [T, DE],
            stride: contiguous_stride([T, DE]),
        },
        expert_counts_scratch: None,
        expert_offsets_scratch: None,
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; (T * DE) as usize];
    out_dev.copy_to_host(&mut got).expect("dl");

    // Tolerance: kernel stages f32 activations through q8_1 (per-token
    // amax / 127 quantization), introducing ~1/127 ≈ 0.8% per-element
    // rounding error in the activation factor. Weights are exact Q8_0
    // (test-built). Per-cell relative tolerance ~1.5% to absorb that
    // plus warp-reduce ordering.
    for i in 0..(T * DE) as usize {
        let g = got[i];
        let e = expected[i];
        let abs_tol = 0.015 * e.abs().max(0.01);
        assert!(
            (g - e).abs() <= abs_tol,
            "moe_scalar_gguf: cell {i}: got={g:.6} expected={e:.6} diff={:.6} tol={abs_tol:.6}",
            (g - e).abs(),
        );
    }
}
