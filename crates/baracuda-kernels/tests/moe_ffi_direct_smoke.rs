//! Real-GPU smoke test for the direct `baracuda-kernels-sys` MoE FFI
//! surface — Phase 20.2.
//!
//! This test demonstrates the **Fuel-replacement contract**: downstream
//! callers can invoke `baracuda_kernels_moe_*_run` C-ABI symbols
//! directly without going through the Rust `MoePlan` layer. Phase 20.2
//! retires `fuel-cuda-kernels/src/moe/` in favour of this surface; the
//! Fuel call sites switch their link target from `moe_gemm_wmma` /
//! `moe_gemm_gguf` / `moe_gemm_gguf_prefill` to the baracuda symbols
//! defined below.
//!
//! ## Symbol-vs-Fuel-shape comparison
//!
//! Fuel's pre-20.2 FFI surface (3 catch-all symbols):
//! ```text
//! moe_gemm_wmma        (input, weights, ..., dtype, is_prefill, stream)
//! moe_gemm_gguf        (input, weights, ..., gguf_dtype, stream)
//! moe_gemm_gguf_prefill(input, weights, ..., input_dtype, gguf_dtype, stream)
//! ```
//!
//! Baracuda's Phase 8.5 / 20.2 FFI surface (5 typed symbols):
//! ```text
//! baracuda_kernels_moe_wmma_f16_run   (..., is_prefill, ..., stream)
//! baracuda_kernels_moe_wmma_bf16_run  (..., is_prefill, ..., stream)
//! baracuda_kernels_moe_scalar_gguf_run(..., gguf_dtype, ..., stream)
//! baracuda_kernels_moe_wmma_gguf_f16_run (..., gguf_dtype, ..., stream)
//! baracuda_kernels_moe_wmma_gguf_bf16_run(..., gguf_dtype, ..., stream)
//! ```
//!
//! The collapse-into-typed-symbols pattern is the baracuda convention
//! across the entire FFI surface; activation dtype lives in the symbol
//! name, GGUF block format remains a runtime `i32` arg matching Fuel's
//! discriminant numbering (`0=Q8_0, 1=Q4_K, 2=Q2_K, 3=Q3_K, 4=Q5_K,
//! 5=Q6_K`).
//!
//! Routing convention identical to `moe_wmma_smoke.rs` — top_k=1 to
//! avoid the kernel's no-synchronization contract surface.
//!
//! Marked `#[ignore]` per project convention.

use core::ffi::c_void;

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::U8;
use half::f16;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Direct-FFI smoke test for `baracuda_kernels_moe_wmma_f16_run`.
///
/// Tiny fixture (T=4 tokens, num_experts=2, top_k=1, D_model=16,
/// D_expert=32) with f16 expert weights. Calls the FFI symbol
/// directly with raw pointers, mirroring how a Fuel-style consumer
/// will invoke the kernel post-20.2.
#[test]
#[ignore]
fn moe_wmma_f16_via_ffi_direct() {
    let (ctx, stream) = setup();

    const T: i32 = 4;
    const NE: i32 = 2;
    const K: i32 = 1;
    const DM: i32 = 16;
    const DE: i32 = 32;

    // Activations + weights — same fixture as moe_wmma_smoke.rs for
    // direct equivalence comparison.
    let mut acts_host = vec![f16::ZERO; (T * DM) as usize];
    for i in 0..T {
        for j in 0..DM {
            acts_host[(i * DM + j) as usize] =
                f16::from_f32(0.1 * (i as f32) + 0.01 * (j as f32));
        }
    }
    let mut weights_host = vec![f16::ZERO; (NE * DE * DM) as usize];
    for e in 0..NE {
        for n in 0..DE {
            for k in 0..DM {
                weights_host[((e * DE + n) * DM + k) as usize] =
                    f16::from_f32((e as f32 + 1.0) * 0.001 * (n as f32 - k as f32));
            }
        }
    }
    // Routing — token t -> expert (t % NE), top_k=1, sorted by expert.
    let mut zipped: Vec<(i32, i32, f32)> = (0..T as usize)
        .map(|t| (t as i32 % NE, t as i32, 0.6))
        .collect();
    zipped.sort_by_key(|x| x.0);
    let flat_expert_ids: Vec<i32> = zipped.iter().map(|x| x.0).collect();
    let sorted_token_ids: Vec<i32> = zipped.iter().map(|x| x.1).collect();
    let topk_weight_per_m: Vec<f32> = zipped.iter().map(|x| x.2).collect();
    let m_total = sorted_token_ids.len() as i32;

    // CPU reference.
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
    let dev_acts: DeviceBuffer<f16> =
        DeviceBuffer::from_slice(&ctx, &acts_host).expect("up acts");
    let weights_u16: Vec<u16> = weights_host.iter().map(|f| f.to_bits()).collect();
    let weights_bytes: Vec<U8> = weights_u16.iter().flat_map(|w| w.to_le_bytes()).map(U8).collect();
    let dev_weights: DeviceBuffer<U8> =
        DeviceBuffer::from_slice(&ctx, &weights_bytes).expect("up weights");
    let dev_sorted: DeviceBuffer<i32> =
        DeviceBuffer::from_slice(&ctx, &sorted_token_ids).expect("up sorted");
    let dev_eids: DeviceBuffer<i32> =
        DeviceBuffer::from_slice(&ctx, &flat_expert_ids).expect("up eids");
    let dev_topk: DeviceBuffer<f32> =
        DeviceBuffer::from_slice(&ctx, &topk_weight_per_m).expect("up tk");
    let mut dev_ec: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, NE as usize).expect("alloc ec");
    let mut dev_eo: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, (NE + 1) as usize).expect("alloc eo");
    let mut dev_out: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (T * DE) as usize).expect("alloc out");

    // Direct FFI call — bypassing MoePlan entirely.
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_moe_wmma_f16_run(
            dev_acts.as_slice().as_raw().0 as *const c_void,
            dev_weights.as_slice().as_raw().0 as *const c_void,
            dev_sorted.as_slice().as_raw().0 as *const i32,
            dev_eids.as_slice().as_raw().0 as *const i32,
            dev_topk.as_slice().as_raw().0 as *const f32,
            dev_out.as_slice_mut().as_raw().0 as *mut c_void,
            dev_ec.as_slice_mut().as_raw().0 as *mut i32,
            dev_eo.as_slice_mut().as_raw().0 as *mut i32,
            NE, K, m_total, DE, DM,
            /* is_prefill */ 1,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "FFI status: {status}");
    stream.synchronize().expect("sync");

    // Download + compare.
    let mut got_bits = vec![0u16; (T * DE) as usize];
    dev_out
        .copy_to_host(unsafe {
            core::slice::from_raw_parts_mut(got_bits.as_mut_ptr() as *mut f16, got_bits.len())
        })
        .expect("dl");

    for i in 0..(T * DE) as usize {
        let g = f16::from_bits(got_bits[i]).to_f32();
        let e = expected[i];
        let abs_tol = 0.03 * e.abs().max(1e-3);
        assert!(
            (g - e).abs() <= abs_tol,
            "moe_wmma_ffi_direct: cell {i}: got={g:.6} expected={e:.6} tol={abs_tol:.6}",
        );
    }
}

/// Direct-FFI smoke test for `baracuda_kernels_moe_scalar_gguf_run`.
///
/// Uses Q8_0 (block size 32) so the fixture is keeps a single block per
/// expert-row. Exercises the scalar GGUF path through the FFI symbol
/// directly. Caller is responsible for `gguf_dtype` numbering: Q8_0=0.
#[test]
#[ignore]
fn moe_scalar_gguf_q8_0_via_ffi_direct() {
    use baracuda_kernels::BlockQ8_0;

    let (ctx, stream) = setup();

    const T: i32 = 4;
    const NE: i32 = 2;
    const K: i32 = 1;
    const DM: i32 = 32; // = QK8_0 block size
    const DE: i32 = 32;

    // Activations (f32 dense).
    let mut acts_host = vec![0.0f32; (T * DM) as usize];
    for i in 0..T {
        for j in 0..DM {
            acts_host[(i * DM + j) as usize] = 0.1 * (i as f32) + 0.01 * (j as f32);
        }
    }
    // Weights as Q8_0 blocks (1 block per expert-row).
    let mut blocks_host = Vec::<BlockQ8_0>::with_capacity((NE * DE) as usize);
    let mut weights_dequant = vec![0.0f32; (NE * DE * DM) as usize];
    for e in 0..NE as usize {
        for n in 0..DE as usize {
            let mut row = [0.0f32; 32];
            for k in 0..32usize {
                row[k] = (e as f32 + 1.0) * 0.001 * (n as f32 - k as f32);
            }
            let amax = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let d = amax / 127.0;
            let mut qs = [0i8; 32];
            for k in 0..32usize {
                let q = (if d == 0.0 { 0.0 } else { (row[k] / d).round() }) as i8;
                qs[k] = q;
                weights_dequant[(e * DE as usize + n) * DM as usize + k] = (q as f32) * d;
            }
            blocks_host.push(BlockQ8_0 { d: f16::from_f32(d).to_bits(), qs });
        }
    }

    // Routing.
    let mut zipped: Vec<(i32, i32, f32)> = (0..T as usize)
        .map(|t| (t as i32 % NE, t as i32, 0.5))
        .collect();
    zipped.sort_by_key(|x| x.0);
    let flat_expert_ids: Vec<i32> = zipped.iter().map(|x| x.0).collect();
    let sorted_token_ids: Vec<i32> = zipped.iter().map(|x| x.1).collect();
    let topk_weight_per_m: Vec<f32> = zipped.iter().map(|x| x.2).collect();
    let m_total = sorted_token_ids.len() as i32;

    // CPU reference (use the dequantized weights so we compare apples-to-apples
    // — the kernel itself dequantizes Q8_0 on-the-fly).
    let mut expected = vec![0.0f32; (T * DE) as usize];
    for m in 0..m_total as usize {
        let token_id = sorted_token_ids[m] as usize;
        let expert = flat_expert_ids[m] as usize;
        let scale = topk_weight_per_m[m];
        for n in 0..DE as usize {
            let mut acc = 0.0f32;
            for k in 0..DM as usize {
                let a = acts_host[token_id * DM as usize + k];
                let w = weights_dequant[(expert * DE as usize + n) * DM as usize + k];
                acc += a * w;
            }
            expected[token_id * DE as usize + n] = scale * acc;
        }
    }

    // Upload.
    let dev_acts: DeviceBuffer<f32> =
        DeviceBuffer::from_slice(&ctx, &acts_host).expect("up acts");
    let weight_bytes: Vec<U8> = unsafe {
        let len = blocks_host.len() * core::mem::size_of::<BlockQ8_0>();
        let ptr = blocks_host.as_ptr() as *const u8;
        core::slice::from_raw_parts(ptr, len).iter().copied().map(U8).collect()
    };
    let dev_weights: DeviceBuffer<U8> =
        DeviceBuffer::from_slice(&ctx, &weight_bytes).expect("up weights");
    let dev_sorted: DeviceBuffer<i32> =
        DeviceBuffer::from_slice(&ctx, &sorted_token_ids).expect("up sorted");
    let dev_eids: DeviceBuffer<i32> =
        DeviceBuffer::from_slice(&ctx, &flat_expert_ids).expect("up eids");
    let dev_topk: DeviceBuffer<f32> =
        DeviceBuffer::from_slice(&ctx, &topk_weight_per_m).expect("up tk");
    let mut dev_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (T * DE) as usize).expect("alloc out");

    // Direct FFI call — Fuel-equivalent: gguf_dtype=0 means Q8_0.
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_moe_scalar_gguf_run(
            dev_acts.as_slice().as_raw().0 as *const c_void,
            dev_weights.as_slice().as_raw().0 as *const c_void,
            dev_sorted.as_slice().as_raw().0 as *const i32,
            dev_eids.as_slice().as_raw().0 as *const i32,
            dev_topk.as_slice().as_raw().0 as *const f32,
            dev_out.as_slice_mut().as_raw().0 as *mut c_void,
            NE, K, m_total, DE, DM,
            /* gguf_dtype */ 0,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "FFI status: {status}");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0f32; (T * DE) as usize];
    dev_out.copy_to_host(&mut got).expect("dl");

    for i in 0..(T * DE) as usize {
        let g = got[i];
        let e = expected[i];
        // Phase 20.2 tolerance — match the Phase 15.3 MoE Q8_0 floor
        // (`0.015 * max(|expected|, 0.01)`). Q8_0's f16 scale gives a
        // per-element quant noise floor of ~1/127 ≈ 0.78%; summing many
        // such products amplifies the relative bound but the absolute
        // floor on small-magnitude cells is what matters here.
        let abs_tol = 0.015 * e.abs().max(0.01);
        assert!(
            (g - e).abs() <= abs_tol,
            "moe_scalar_gguf_ffi: cell {i}: got={g:.6} expected={e:.6} tol={abs_tol:.6}",
        );
    }
}
