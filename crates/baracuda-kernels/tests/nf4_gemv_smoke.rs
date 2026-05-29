//! Real-GPU smoke test for `Nf4MmvqPlan` — Phase 53
//! (bitsandbytes NF4 vendor, M=1 single-vector decode).
//!
//! Build a small NF4-packed weight matrix + an f16 activation vector,
//! run the kernel, compare against a host-side `[N, K]` × `[K]` fp32
//! matmul on the de-quantized weights. The NF4 quantization itself is
//! lossy (relative error class ~1e-2 vs the original fp32 weight) but
//! the kernel must match the *dequantize-then-matmul* path tightly —
//! the only delta is the fp16 activation cast inside the kernel.
//!
//! `#[ignore]` by default — requires a real CUDA device + the
//! `bnb_nf4` cargo feature.

#![cfg(feature = "bnb_nf4")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::quantize::nf4::nf4_pack_weight;
use baracuda_kernels::{
    contiguous_stride, Nf4Descriptor, Nf4MmvqArgs, Nf4MmvqPlan, PlanPreference, TensorMut,
    TensorRef, Workspace, NF4_CODEBOOK, U8,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn build_nf4_weight(n: usize, k: usize, block_size: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let mut host_w = vec![0.0f32; n * k];
    for r in 0..n {
        for c in 0..k {
            let scale = 0.08 + (r as f32) * 0.03;
            host_w[r * k + c] = scale * (((c as f32) - (k as f32) * 0.5) / (k as f32) * 2.0);
        }
    }
    let (packed, absmax) = nf4_pack_weight(&host_w, n, k, block_size);
    // Build the *dequantized* fp32 weight matrix as the reference —
    // matmul against this is what the kernel computes internally.
    let mut deq = vec![0.0f32; n * k];
    let blocks_per_row = k / block_size;
    for row in 0..n {
        for b in 0..blocks_per_row {
            let a = absmax[row * blocks_per_row + b];
            for j in 0..block_size {
                let kpos = b * block_size + j;
                let byte_off = (row / 2) * k + kpos;
                let byte = packed[byte_off];
                let code_idx = if (row & 1) == 0 {
                    (byte & 0x0F) as usize
                } else {
                    ((byte >> 4) & 0x0F) as usize
                };
                deq[row * k + kpos] = NF4_CODEBOOK[code_idx] * a;
            }
        }
    }
    (packed, absmax, deq)
}

#[test]
#[ignore]
fn nf4_gemv_m1_f16() {
    let (ctx, stream) = setup();
    let n: usize = 16;
    let k: usize = 128;
    let block_size: usize = 64;

    let (packed, absmax, w_deq) = build_nf4_weight(n, k, block_size);

    // Activations in f16.
    let host_y_f32: Vec<f32> = (0..k)
        .map(|i| ((i as f32) - (k as f32) * 0.5) * 0.03)
        .collect();
    let host_y_f16: Vec<f16> = host_y_f32.iter().map(|x| f16::from_f32(*x)).collect();

    // Reference: deq @ y, computed in fp32 with f16-quantized activations.
    let host_y_f32_from_f16: Vec<f32> = host_y_f16.iter().map(|h| h.to_f32()).collect();
    let mut expected = vec![0.0f32; n];
    for row in 0..n {
        let mut acc = 0.0f32;
        for j in 0..k {
            acc += w_deq[row * k + j] * host_y_f32_from_f16[j];
        }
        expected[row] = acc;
    }

    // Upload.
    let packed_u8: Vec<U8> = packed.iter().map(|b| U8(*b)).collect();
    let dev_w = DeviceBuffer::from_slice(&ctx, &packed_u8).expect("up w");
    let dev_amax = DeviceBuffer::from_slice(&ctx, &absmax).expect("up amax");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y_f16).expect("up y");
    let mut dev_out: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n).expect("alloc out");

    let desc = Nf4Descriptor {
        n: n as i32,
        k: k as i32,
        block_size: block_size as i32,
    };
    let plan: Nf4MmvqPlan<f16> = Nf4MmvqPlan::select(
        &stream, &desc, PlanPreference::default(),
    )
    .expect("plan select");

    let weight_bytes_len = packed_u8.len() as i32;
    let absmax_len = absmax.len() as i32;
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
            data: dev_y.as_slice(),
            shape: [k as i32],
            stride: contiguous_stride([k as i32]),
        },
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [n as i32],
            stride: contiguous_stride([n as i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("gemv run");
    stream.synchronize().expect("sync");

    let mut host_out_f16 = vec![f16::ZERO; n];
    dev_out.copy_to_host(&mut host_out_f16).expect("d2h");
    let host_out: Vec<f32> = host_out_f16.iter().map(|h| h.to_f32()).collect();

    // f16 accumulator-store + fp32 reference computed in single precision.
    // Tolerance: ~1e-2 relative error class accounts for the f16 store
    // truncation and accumulator rounding inside the kernel.
    let max_ref = expected.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let mut max_abs_err = 0.0f32;
    for i in 0..n {
        let err = (host_out[i] - expected[i]).abs();
        max_abs_err = max_abs_err.max(err);
        assert!(
            err < 0.02 * (max_ref.max(1e-3)),
            "row {i}: got {}, expected {}, abs_err={err}",
            host_out[i], expected[i]
        );
    }
    eprintln!(
        "nf4 gemv M=1 f16: max_ref={max_ref:.4e}, max_abs_err={max_abs_err:.4e}"
    );
}

#[test]
#[ignore]
fn nf4_gemv_m1_bf16() {
    let (ctx, stream) = setup();
    let n: usize = 16;
    let k: usize = 128;
    let block_size: usize = 64;

    let (packed, absmax, w_deq) = build_nf4_weight(n, k, block_size);

    let host_y_f32: Vec<f32> = (0..k)
        .map(|i| ((i as f32) - (k as f32) * 0.5) * 0.03)
        .collect();
    let host_y_bf16: Vec<bf16> = host_y_f32.iter().map(|x| bf16::from_f32(*x)).collect();
    let host_y_f32_from_bf16: Vec<f32> = host_y_bf16.iter().map(|b| b.to_f32()).collect();

    let mut expected = vec![0.0f32; n];
    for row in 0..n {
        let mut acc = 0.0f32;
        for j in 0..k {
            acc += w_deq[row * k + j] * host_y_f32_from_bf16[j];
        }
        expected[row] = acc;
    }

    let packed_u8: Vec<U8> = packed.iter().map(|b| U8(*b)).collect();
    let dev_w = DeviceBuffer::from_slice(&ctx, &packed_u8).expect("up w");
    let dev_amax = DeviceBuffer::from_slice(&ctx, &absmax).expect("up amax");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y_bf16).expect("up y");
    let mut dev_out: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, n).expect("alloc out");

    let desc = Nf4Descriptor {
        n: n as i32,
        k: k as i32,
        block_size: block_size as i32,
    };
    let plan: Nf4MmvqPlan<bf16> = Nf4MmvqPlan::select(
        &stream, &desc, PlanPreference::default(),
    )
    .expect("plan select");

    let weight_bytes_len = packed_u8.len() as i32;
    let absmax_len = absmax.len() as i32;
    let args = Nf4MmvqArgs::<bf16> {
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
            data: dev_y.as_slice(),
            shape: [k as i32],
            stride: contiguous_stride([k as i32]),
        },
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [n as i32],
            stride: contiguous_stride([n as i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("gemv run");
    stream.synchronize().expect("sync");

    let mut host_out_bf16 = vec![bf16::ZERO; n];
    dev_out.copy_to_host(&mut host_out_bf16).expect("d2h");
    let host_out: Vec<f32> = host_out_bf16.iter().map(|b| b.to_f32()).collect();

    // bf16's reduced precision (8-bit mantissa vs f16's 11-bit) widens
    // the tolerance vs the f16 path.
    let max_ref = expected.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let mut max_abs_err = 0.0f32;
    for i in 0..n {
        let err = (host_out[i] - expected[i]).abs();
        max_abs_err = max_abs_err.max(err);
        assert!(
            err < 0.04 * (max_ref.max(1e-3)),
            "row {i}: got {}, expected {}, abs_err={err}",
            host_out[i], expected[i]
        );
    }
    eprintln!(
        "nf4 gemv M=1 bf16: max_ref={max_ref:.4e}, max_abs_err={max_abs_err:.4e}"
    );
}
