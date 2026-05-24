//! Real-GPU smoke test for `MoePlan::WmmaGguf` — Phase 8 Milestone 8.5.
//!
//! Same fixture style as `moe_gguf_smoke.rs` (T=4, NE=2, K=1, DM=32,
//! DE=32) with Q8_0-packed expert weights but f16 activations and
//! the WMMA + GGUF combined path. Compares against a CPU reference
//! that dequantizes the Q8_0 blocks then dispatches manually.
//!
//! ## Routing convention (Phase 15.3 re-derivation)
//!
//! See `moe_gguf_smoke.rs` for the full rationale. The vendored
//! Fuel/llama.cpp WMMA+GGUF prefill kernel writes
//! `output[token_index * size_n + n_global] = val` (assignment, not
//! accumulation, line ~1106 of `baracuda_moe.cuh`). When `top_k > 1`
//! and multiple experts share a `token_id`, the result is the value
//! from whichever block wrote last, with non-deterministic CUDA
//! block scheduling. This fixture uses `top_k = 1` for a
//! deterministic comparison.
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
fn moe_wmma_gguf_q8_0_small_fixture() {
    let (ctx, stream) = setup();

    const T: i32 = 4;
    const NE: i32 = 2;
    const K: i32 = 1;
    const DM: i32 = 32;
    const DE: i32 = 32;

    // Activations: f16 ramp.
    let mut acts_host = vec![f16::ZERO; (T * DM) as usize];
    for i in 0..T {
        for j in 0..DM {
            acts_host[(i * DM + j) as usize] =
                f16::from_f32(0.1 * (i as f32) + 0.01 * (j as f32));
        }
    }
    // Per-expert Q8_0-packed weights (one block per (expert, n) row).
    let mut blocks_host = Vec::<BlockQ8_0>::with_capacity((NE * DE) as usize);
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

    // Routing: token t -> expert (t % NE), top_k=1. Pre-sorted by expert.
    let mut routing: Vec<(i32, i32)> = (0..T).map(|t| (t % NE, t)).collect();
    routing.sort_by_key(|x| x.0);
    let flat_expert_ids: Vec<i32> = routing.iter().map(|x| x.0).collect();
    let sorted_token_ids: Vec<i32> = routing.iter().map(|x| x.1).collect();
    let topk_weight_per_m: Vec<f32> = vec![0.6; T as usize];
    let m_total = sorted_token_ids.len() as i32;

    // Reference: WMMA+GGUF kernel writes
    //   out[token_id, n] = topk_weights[token_id] * Σ_k acts[token_id, k] * w[expert, n, k]
    // (assignment; with top_k=1 each cell written once.)
    let mut expected = vec![0.0f32; (T * DE) as usize];
    for m in 0..m_total as usize {
        let token_id = sorted_token_ids[m] as usize;
        let expert = flat_expert_ids[m] as usize;
        let scale = topk_weight_per_m[m];
        for n in 0..DE as usize {
            let mut acc = 0.0f32;
            for k in 0..DM as usize {
                let a = acts_host[token_id * DM as usize + k].to_f32();
                let w = weights_dequant[(expert * DE as usize + n) * DM as usize + k];
                acc += a * w;
            }
            expected[token_id * DE as usize + n] = scale * acc;
        }
    }

    // Upload.
    let acts_dev: DeviceBuffer<f16> = DeviceBuffer::from_slice(&ctx, &acts_host).expect("up acts");
    let weights_dev: DeviceBuffer<U8> = DeviceBuffer::from_slice(&ctx, &weights_u8).expect("up w");
    let sorted_dev: DeviceBuffer<i32> = DeviceBuffer::from_slice(&ctx, &sorted_token_ids).expect("up s");
    let eids_dev: DeviceBuffer<i32> = DeviceBuffer::from_slice(&ctx, &flat_expert_ids).expect("up eids");
    let tk_dev: DeviceBuffer<f32> = DeviceBuffer::from_slice(&ctx, &topk_weight_per_m).expect("up tk");
    let mut ec_dev: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, NE as usize).expect("ec");
    let mut eo_dev: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, (NE + 1) as usize).expect("eo");
    let ew_dev: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, (T * K) as usize).expect("ew");

    let desc = MoeDescriptor {
        num_tokens: T,
        num_experts: NE,
        top_k: K,
        d_model: DM,
        d_expert: DE,
        variant: MoeVariant::WmmaGguf,
        block_format: Some(GgufBlockFormat::Q8_0),
        element: ElementKind::F16,
        is_prefill: true,
    };
    let plan = MoePlan::select(&stream, &desc, PlanPreference::default()).expect("select");

    // WmmaGguf writes f32 output. The MoeArgs::output is generic over
    // T = activation element (f16). We allocate an f16-typed buffer
    // with 2× the slot count so the byte storage holds T*DE f32
    // values. The kernel writes f32 directly through the raw pointer.
    let mut out_dev_typed: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (T * DE * 2) as usize).expect("alloc out typed");

    let args = MoeArgs::<f16> {
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
            data: ew_dev.as_slice(),
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
            shape: [m_total],
            stride: contiguous_stride([m_total]),
        }),
        expert_matrices: TensorRef {
            data: weights_dev.as_slice(),
            shape: [weights_bytes_len],
            stride: contiguous_stride([weights_bytes_len]),
        },
        output: TensorMut {
            data: out_dev_typed.as_slice_mut(),
            shape: [T, DE * 2], // 2x f16 slots == T*DE f32 slots
            stride: contiguous_stride([T, DE * 2]),
        },
        expert_counts_scratch: Some(TensorMut {
            data: ec_dev.as_slice_mut(),
            shape: [NE],
            stride: contiguous_stride([NE]),
        }),
        expert_offsets_scratch: Some(TensorMut {
            data: eo_dev.as_slice_mut(),
            shape: [NE + 1],
            stride: contiguous_stride([NE + 1]),
        }),
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    // Read back as f32 — interpret the f16-typed storage as f32 (2 f16 slots
    // = 1 f32). We must download via the same buffer-typed pointer.
    let mut got_f16_bits = vec![0u16; (T * DE * 2) as usize];
    out_dev_typed
        .copy_to_host(unsafe {
            core::slice::from_raw_parts_mut(
                got_f16_bits.as_mut_ptr() as *mut f16,
                got_f16_bits.len(),
            )
        })
        .expect("dl");
    let got_f32: Vec<f32> = got_f16_bits
        .chunks_exact(2)
        .map(|c| {
            let lo = c[0].to_le_bytes();
            let hi = c[1].to_le_bytes();
            f32::from_le_bytes([lo[0], lo[1], hi[0], hi[1]])
        })
        .collect();

    // Tolerance: f16 acts + f16 dequant (via half-precision Q8_0
    // dequant), WMMA tensor-core MMA with f32 accumulator, then
    // assignment to f32 output. Per-cell relative tolerance ~3%
    // absorbs the f16 dequant + ramp rounding.
    for i in 0..(T * DE) as usize {
        let g = got_f32[i];
        let e = expected[i];
        let abs_tol = 0.03 * e.abs().max(1e-3);
        assert!(
            (g - e).abs() <= abs_tol,
            "moe_wmma_gguf: cell {i}: got={g:.6} expected={e:.6} diff={:.6} tol={abs_tol:.6}",
            (g - e).abs(),
        );
    }
}
