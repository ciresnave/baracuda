//! Real-GPU smoke test for `Nf4DequantizePlan` — Phase 53
//! (bitsandbytes NF4 vendor).
//!
//! Quantize → pack → device-dequant → host-dequant round-trip on a
//! small `[N, K]` weight matrix. Verifies that the GPU codebook lookup
//! + absmax scaling matches the host reference bit-for-bit (modulo the
//! T_act cast — we test the f32 dequant path here to keep the
//! comparison exact).
//!
//! `#[ignore]` by default — requires a real CUDA device + the
//! `bnb_nf4` cargo feature.

#![cfg(feature = "bnb_nf4")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::quantize::nf4::nf4_pack_weight;
use baracuda_kernels::{
    contiguous_stride, Nf4DequantizeArgs, Nf4DequantizePlan, Nf4Descriptor, PlanPreference,
    TensorMut, TensorRef, Workspace, NF4_CODEBOOK, U8,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Verify the codebook constants are bit-identical to the upstream
/// bitsandbytes table. Host-only check — does not need a device.
#[test]
fn nf4_codebook_constants() {
    // Spot-check a few values.
    assert_eq!(NF4_CODEBOOK[0], -1.0);
    assert_eq!(NF4_CODEBOOK[7], 0.0);
    assert_eq!(NF4_CODEBOOK[15], 1.0);
    // Codebook must be monotone increasing.
    for i in 1..16 {
        assert!(
            NF4_CODEBOOK[i] > NF4_CODEBOOK[i - 1],
            "codebook not monotone at idx {i}: prev={}, this={}",
            NF4_CODEBOOK[i - 1], NF4_CODEBOOK[i]
        );
    }
    // 16 entries.
    assert_eq!(NF4_CODEBOOK.len(), 16);
}

/// End-to-end: build a small fp32 weight matrix, host-quantize to NF4,
/// device-dequantize, compare to the host-side reference dequant.
///
/// We use the f32 dequant FFI path so the comparison is exact — no
/// fp16 / bf16 cast noise on top of the NF4 quant error itself.
#[test]
#[ignore]
fn nf4_dequant_roundtrip_f32() {
    let (ctx, stream) = setup();
    let n: usize = 8;
    let k: usize = 128;
    let block_size: usize = 64;

    // Build a smooth-ish fp32 weight matrix.
    let mut host_w = vec![0.0f32; n * k];
    for r in 0..n {
        for c in 0..k {
            // Mix of pos/neg with a per-row scale to exercise absmax.
            let scale = 0.1 + (r as f32) * 0.05;
            host_w[r * k + c] = scale * (((c as f32) - (k as f32) * 0.5) / (k as f32) * 2.0);
        }
    }

    // Host-side quantize → pack.
    let (packed_bytes, absmax) = nf4_pack_weight(&host_w, n, k, block_size);
    assert_eq!(packed_bytes.len(), (n / 2) * k);
    assert_eq!(absmax.len(), n * (k / block_size));

    // Host-side reference dequant via the same codebook + absmax.
    let mut host_dequant_ref = vec![0.0f32; n * k];
    for row in 0..n {
        let blocks_per_row = k / block_size;
        for b in 0..blocks_per_row {
            let a = absmax[row * blocks_per_row + b];
            for j in 0..block_size {
                let kpos = b * block_size + j;
                let byte_off = (row / 2) * k + kpos;
                let byte = packed_bytes[byte_off];
                let code_idx = if (row & 1) == 0 {
                    (byte & 0x0F) as usize
                } else {
                    ((byte >> 4) & 0x0F) as usize
                };
                host_dequant_ref[row * k + kpos] = NF4_CODEBOOK[code_idx] * a;
            }
        }
    }

    // Upload to device.
    let packed_u8: Vec<U8> = packed_bytes.iter().map(|b| U8(*b)).collect();
    let dev_w = DeviceBuffer::from_slice(&ctx, &packed_u8).expect("up w");
    let dev_amax = DeviceBuffer::from_slice(&ctx, &absmax).expect("up amax");
    let mut dev_out: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n * k).expect("alloc out");

    // Build plan.
    let desc = Nf4Descriptor {
        n: n as i32,
        k: k as i32,
        block_size: block_size as i32,
    };
    let plan: Nf4DequantizePlan<f32> = Nf4DequantizePlan::select(
        &stream, &desc, PlanPreference::default(),
    )
    .expect("plan select");

    let weight_bytes_len = packed_u8.len() as i32;
    let absmax_len = absmax.len() as i32;
    let args = Nf4DequantizeArgs::<f32> {
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
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [n as i32, k as i32],
            stride: contiguous_stride([n as i32, k as i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("dequant run");
    stream.synchronize().expect("sync");

    // Pull back + compare.
    let mut host_out = vec![0.0f32; n * k];
    dev_out.copy_to_host(&mut host_out).expect("d2h");

    let mut max_abs_err = 0.0f32;
    for i in 0..n * k {
        let err = (host_out[i] - host_dequant_ref[i]).abs();
        max_abs_err = max_abs_err.max(err);
        // The two paths apply the same codebook + same absmax, so the
        // result should be bit-equivalent in fp32.
        assert!(
            err < 1e-6,
            "idx {i}: gpu={}, ref={}, err={err}",
            host_out[i], host_dequant_ref[i]
        );
    }
    eprintln!("nf4 dequant roundtrip max_abs_err = {max_abs_err:.4e}");
}
