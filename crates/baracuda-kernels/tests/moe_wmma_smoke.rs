//! Real-GPU smoke test for `MoePlan::Wmma` — Phase 8 Milestone 8.5.
//!
//! Tiny fixture (T=4 tokens, num_experts=2, top_k=1, D_model=16,
//! D_expert=32) with f16 expert weights. Verifies the WMMA MoE output
//! against a CPU reference that mirrors the per-token routing manually.
//!
//! ## Routing convention (Phase 15.3 re-derivation)
//!
//! See `moe_gguf_smoke.rs` for the full rationale. The vendored
//! Fuel/llama.cpp WMMA kernel writes `output[token_index, n_global] = val`
//! (assignment, not accumulation) at line ~942 of `baracuda_moe.cuh`.
//! When `top_k > 1` and multiple experts share a `token_id`, the
//! result is the value from whichever block wrote last, with
//! non-deterministic CUDA block scheduling. To get a deterministic
//! comparison this fixture uses `top_k = 1`.
//!
//! Marked `#[ignore]` per project convention.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, MoeArgs, MoeDescriptor, MoePlan, MoeVariant,
    PlanPreference, TensorMut, TensorRef, Workspace, U8,
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
fn moe_wmma_f16_small_fixture() {
    let (ctx, stream) = setup();

    const T: i32 = 4;
    const NE: i32 = 2;
    const K: i32 = 1;
    const DM: i32 = 16;
    const DE: i32 = 32;

    // Activations: deterministic ramp (token i, dim j) = (i * 0.1 + j * 0.01).
    let mut acts_host = vec![f16::ZERO; (T * DM) as usize];
    for i in 0..T {
        for j in 0..DM {
            acts_host[(i * DM + j) as usize] =
                f16::from_f32(0.1 * (i as f32) + 0.01 * (j as f32));
        }
    }
    // Weights per-expert: small deterministic pattern.
    // w[e, n, k] = (e + 1) * 0.001 * (n - k).
    let mut weights_host = vec![f16::ZERO; (NE * DE * DM) as usize];
    for e in 0..NE {
        for n in 0..DE {
            for k in 0..DM {
                weights_host[((e * DE + n) * DM + k) as usize] =
                    f16::from_f32((e as f32 + 1.0) * 0.001 * (n as f32 - k as f32));
            }
        }
    }
    // Routing: token t -> expert (t % NE), top_k=1. One m_idx per token,
    // so every output cell is written exactly once.
    let mut sorted_token_ids = Vec::<i32>::with_capacity(T as usize);
    let mut flat_expert_ids = Vec::<i32>::with_capacity(T as usize);
    let mut topk_weight_flat = Vec::<f32>::with_capacity(T as usize);
    for t in 0..T {
        sorted_token_ids.push(t);
        flat_expert_ids.push(t % NE);
        topk_weight_flat.push(0.6);
    }
    // Sort by expert (the kernel assumes this layout).
    let mut zipped: Vec<(i32, i32, f32)> = (0..T as usize)
        .map(|i| (flat_expert_ids[i], sorted_token_ids[i], topk_weight_flat[i]))
        .collect();
    zipped.sort_by_key(|x| x.0);
    let flat_expert_ids: Vec<i32> = zipped.iter().map(|x| x.0).collect();
    let sorted_token_ids: Vec<i32> = zipped.iter().map(|x| x.1).collect();
    let topk_weight_per_m: Vec<f32> = zipped.iter().map(|x| x.2).collect();
    let m_total = sorted_token_ids.len() as i32;
    assert_eq!(m_total, T * K);

    // CPU reference: WMMA kernel writes
    //   output[token_id, n] = topk_weights[token_id] * Σ_k acts[token_id, k] * w[expert, n, k]
    // (last-write-wins assignment; with top_k=1 each cell written once.)
    let mut expected = vec![0.0f32; (T * DE) as usize];
    for m in 0..m_total as usize {
        let token_id = sorted_token_ids[m] as usize;
        let expert = flat_expert_ids[m] as usize;
        let scale = topk_weight_per_m[m];
        for n in 0..DE as usize {
            let mut acc = 0.0f32;
            for k in 0..DM as usize {
                let a = acts_host[token_id * DM as usize + k].to_f32();
                let w = weights_host[(expert * DE as usize + n) * DM as usize + k].to_f32();
                acc += a * w;
            }
            expected[token_id * DE as usize + n] = scale * acc;
        }
    }

    // Upload buffers.
    let weights_u16: Vec<u16> = weights_host.iter().map(|f| f.to_bits()).collect();
    let weights_bytes_len = (weights_u16.len() * 2) as i32;
    let dev_weights_bytes_view: DeviceBuffer<U8> = {
        let raw_bytes: Vec<U8> = weights_u16
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .map(U8)
            .collect();
        DeviceBuffer::from_slice(&ctx, &raw_bytes).expect("up weights bytes")
    };

    let dev_sorted: DeviceBuffer<i32> =
        DeviceBuffer::from_slice(&ctx, &sorted_token_ids).expect("up sorted");
    let dev_expert_ids: DeviceBuffer<i32> =
        DeviceBuffer::from_slice(&ctx, &flat_expert_ids).expect("up eids");
    let dev_topk: DeviceBuffer<f32> =
        DeviceBuffer::from_slice(&ctx, &topk_weight_per_m).expect("up tk");

    let mut dev_ec: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, NE as usize).expect("ec");
    let mut dev_eo: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, (NE + 1) as usize).expect("eo");

    let desc = MoeDescriptor {
        num_tokens: T,
        num_experts: NE,
        top_k: K,
        d_model: DM,
        d_expert: DE,
        variant: MoeVariant::Wmma,
        block_format: None,
        element: ElementKind::F16,
        is_prefill: true,
    };
    let plan = MoePlan::select(&stream, &desc, PlanPreference::default()).expect("select");

    let acts_dev: DeviceBuffer<f16> =
        DeviceBuffer::from_slice(&ctx, &acts_host).expect("up acts as f16");
    let topk_weights_dev_f16: DeviceBuffer<f16> = {
        DeviceBuffer::zeros(&ctx, (T * K) as usize).expect("up tk f16")
    };
    let mut output_dev_f16: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (T * DE) as usize).expect("alloc out f16");

    let args = MoeArgs::<f16> {
        activations: TensorRef {
            data: acts_dev.as_slice(),
            shape: [T, DM],
            stride: contiguous_stride([T, DM]),
        },
        expert_indices: TensorRef {
            data: dev_expert_ids.as_slice(),
            shape: [T, K],
            stride: contiguous_stride([T, K]),
        },
        expert_weights: TensorRef {
            data: topk_weights_dev_f16.as_slice(),
            shape: [T, K],
            stride: contiguous_stride([T, K]),
        },
        sorted_token_ids: TensorRef {
            data: dev_sorted.as_slice(),
            shape: [m_total],
            stride: contiguous_stride([m_total]),
        },
        flat_expert_ids: TensorRef {
            data: dev_expert_ids.as_slice(),
            shape: [m_total],
            stride: contiguous_stride([m_total]),
        },
        topk_weight_flat: Some(TensorRef {
            data: dev_topk.as_slice(),
            shape: [m_total],
            stride: contiguous_stride([m_total]),
        }),
        expert_matrices: TensorRef {
            data: dev_weights_bytes_view.as_slice(),
            shape: [weights_bytes_len],
            stride: contiguous_stride([weights_bytes_len]),
        },
        output: TensorMut {
            data: output_dev_f16.as_slice_mut(),
            shape: [T, DE],
            stride: contiguous_stride([T, DE]),
        },
        expert_counts_scratch: Some(TensorMut {
            data: dev_ec.as_slice_mut(),
            shape: [NE],
            stride: contiguous_stride([NE]),
        }),
        expert_offsets_scratch: Some(TensorMut {
            data: dev_eo.as_slice_mut(),
            shape: [NE + 1],
            stride: contiguous_stride([NE + 1]),
        }),
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got_bits = vec![0u16; (T * DE) as usize];
    output_dev_f16
        .copy_to_host(unsafe {
            core::slice::from_raw_parts_mut(got_bits.as_mut_ptr() as *mut f16, got_bits.len())
        })
        .expect("dl");

    // Tolerance: WMMA tensor cores accumulate in f32 internally but
    // round to f16 on output (`moe_from_float`). Weights are already
    // f16 (rounded once at upload). Per-cell relative tolerance ~3%
    // absorbs f16 round-trip + tensor-core ULPs.
    for i in 0..(T * DE) as usize {
        let g = f16::from_bits(got_bits[i]).to_f32();
        let e = expected[i];
        let abs_tol = 0.03 * e.abs().max(1e-3);
        assert!(
            (g - e).abs() <= abs_tol,
            "moe_wmma: cell {i}: got={g:.6} expected={e:.6} diff={:.6} tol={abs_tol:.6}",
            (g - e).abs(),
        );
    }
}
