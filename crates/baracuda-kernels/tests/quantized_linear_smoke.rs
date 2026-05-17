//! Real-GPU smoke test for `QuantizedLinearPlan<f32, S8>` (Phase 8.3).
//!
//! Small `M=2`, `C_out=3`, `K=8` W8A8 quantized matmul. The test:
//!
//! 1. Generates an FP activation `[M, K]`.
//! 2. Generates an FP weight `[C_out, K]`, then offline-quantizes it
//!    per-output-channel symmetrically to int8 (this mirrors what an
//!    offline weight-quantization tool would do — the matching scale
//!    is the input to `QuantizedLinearPlan`).
//! 3. Runs `QuantizedLinearPlan` on the GPU (fuses per-token activation
//!    DRQ + int8 GEMM + dequant FP store).
//! 4. Runs the full-precision FP matmul on the CPU as ground truth.
//! 5. Verifies the GPU output is within ~2% relative error of the FP
//!    reference (int8 quantization adds ~1% per-row + ~1% per-column
//!    noise — 2% is the LLM W8A8 convention).
//!
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, QuantizedLinearArgs,
    QuantizedLinearDescriptor, QuantizedLinearPlan, TensorMut, TensorRef, Workspace, S8,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Offline weight quantization, per-output-channel symmetric.
/// Returns `(weight_q[C_out, K], weight_scale[C_out])`.
fn quantize_weight_per_channel_symmetric(
    c_out: usize,
    k: usize,
    w: &[f32],
) -> (Vec<i8>, Vec<f32>) {
    let qmax = 127i32;
    let mut scale = vec![0f32; c_out];
    let mut wq = vec![0i8; c_out * k];
    for r in 0..c_out {
        let mut max_abs = 0f32;
        for c in 0..k {
            let v = w[r * k + c].abs();
            if v > max_abs {
                max_abs = v;
            }
        }
        let s = if max_abs > 0.0 {
            max_abs / qmax as f32
        } else {
            1.0
        };
        scale[r] = s;
        let inv_s = 1.0 / s;
        for c in 0..k {
            let v = w[r * k + c];
            let q = (v * inv_s).round_ties_even() as i32;
            wq[r * k + c] = q.clamp(-127, 127) as i8;
        }
    }
    (wq, scale)
}

/// CPU full-precision reference: out[m, n] = sum_k a[m, k] * w[n, k].
fn cpu_fp_matmul(
    m: usize,
    c_out: usize,
    k: usize,
    a: &[f32],
    w: &[f32],
) -> Vec<f32> {
    let mut out = vec![0f32; m * c_out];
    for mm in 0..m {
        for nn in 0..c_out {
            let mut acc = 0f32;
            for kk in 0..k {
                acc += a[mm * k + kk] * w[nn * k + kk];
            }
            out[mm * c_out + nn] = acc;
        }
    }
    out
}

#[test]
#[ignore]
fn quantized_linear_w8a8_f32_basic() {
    let (ctx, stream) = setup();
    let m: i32 = 2;
    let c_out: i32 = 3;
    let k: i32 = 8;

    // Activation `[M, K]`. Mix of small + large magnitudes per row to
    // exercise per-token scaling.
    let host_a: Vec<f32> = vec![
        // row 0 — magnitudes around 1.0
        0.5, -0.3, 0.8, 1.2, -0.4, 0.6, 0.9, -0.7,
        // row 1 — one row with much larger range (forces a larger scale_a)
        4.0, -2.5, 1.8, -3.2, 2.1, -1.4, 0.9, 3.6,
    ];

    // Weight `[C_out, K]`. Distinct magnitudes per channel.
    let host_w: Vec<f32> = vec![
        // chan 0 — small magnitude
        0.1, -0.2, 0.15, 0.05, -0.1, 0.08, -0.12, 0.18,
        // chan 1 — mid magnitude
        0.4, 0.3, -0.5, 0.6, -0.45, 0.55, -0.35, 0.4,
        // chan 2 — large magnitude
        1.2, -0.8, 1.0, 0.9, -1.1, 0.7, 1.3, -0.95,
    ];

    let (host_wq, host_w_scale) =
        quantize_weight_per_channel_symmetric(c_out as usize, k as usize, &host_w);

    let expected_fp = cpu_fp_matmul(m as usize, c_out as usize, k as usize, &host_a, &host_w);

    // Device buffers.
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("up a");
    let host_wq_u: &[u8] =
        unsafe { core::slice::from_raw_parts(host_wq.as_ptr() as *const u8, host_wq.len()) };
    let dev_wq_bytes = DeviceBuffer::from_slice(&ctx, host_wq_u).expect("up wq");
    let dev_wq = dev_wq_bytes.view_as::<S8>();
    let dev_w_scale = DeviceBuffer::from_slice(&ctx, &host_w_scale).expect("up w_scale");
    let mut dev_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (m * c_out) as usize).expect("alloc out");
    let mut dev_act_q: DeviceBuffer<S8> =
        DeviceBuffer::zeros(&ctx, (m * k) as usize).expect("alloc act_q");
    let mut dev_act_scale: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, m as usize).expect("alloc act_scale");

    let desc = QuantizedLinearDescriptor {
        m,
        c_out,
        k,
        q_min: -127,
        q_max: 127,
        activation_element: ElementKind::F32,
        weight_element: ElementKind::S8,
    };
    let plan = QuantizedLinearPlan::<f32, S8>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = QuantizedLinearArgs::<f32, S8> {
        activation: TensorRef {
            data: dev_a.as_slice(),
            shape: [m, k],
            stride: contiguous_stride([m, k]),
        },
        weight_q: TensorRef {
            data: dev_wq,
            shape: [c_out, k],
            stride: contiguous_stride([c_out, k]),
        },
        weight_scale: TensorRef {
            data: dev_w_scale.as_slice(),
            shape: [c_out],
            stride: contiguous_stride([c_out]),
        },
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [m, c_out],
            stride: contiguous_stride([m, c_out]),
        },
        act_q_scratch: TensorMut {
            data: dev_act_q.as_slice_mut(),
            shape: [m, k],
            stride: contiguous_stride([m, k]),
        },
        act_scale_scratch: TensorMut {
            data: dev_act_scale.as_slice_mut(),
            shape: [m],
            stride: contiguous_stride([m]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; (m * c_out) as usize];
    dev_out.copy_to_host(&mut got).expect("dl");

    // Compare against full-precision FP matmul. ~2% relative tolerance
    // is the LLM W8A8 convention (~1% per quant axis, applied to both
    // activation and weight); we use 3% here for the small-K case
    // (K=8 means quantization noise dominates the dot-product's
    // averaging-out effect).
    for (i, (g, e)) in got.iter().zip(expected_fp.iter()).enumerate() {
        let abs_err = (g - e).abs();
        let denom = e.abs().max(1e-3);
        let rel = abs_err / denom;
        assert!(
            rel < 0.03,
            "cell {i}: got {g} expected {e} (abs_err {abs_err}, rel {rel})"
        );
    }
}
