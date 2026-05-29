//! Phase 45 — Real-GPU smoke test for `SmoothQuantLinearPlan<f32, S8>`.
//!
//! Small `M=2`, `N=3`, `K=8` SmoothQuant W8A8 matmul. The test:
//!
//! 1. Generates an FP activation `[M, K]` with an injected outlier
//!    channel (one column with ~10× the magnitude of the others —
//!    canonical SmoothQuant-relevant case).
//! 2. Generates an FP weight `[N, K]` with balanced per-channel
//!    magnitudes.
//! 3. Computes a SmoothQuant smoothing vector `s[K]` from the
//!    activation max-abs per column with `α = 0.5` (the recipe from
//!    §3.1 of the SmoothQuant paper).
//! 4. Pre-smooths activation `A_smooth = A / diag(s)` + quantizes
//!    per-tensor symmetrically to int8.
//! 5. Pre-smooths weight `W_smooth = diag(s) · W` + quantizes
//!    per-output-channel symmetrically to int8.
//! 6. Runs `SmoothQuantLinearPlan` on the GPU.
//! 7. Runs the full-precision FP matmul on the CPU as ground truth.
//! 8. Verifies the GPU output is within ~3% relative error of the
//!    FP reference (slightly looser than `QuantizedLinearPlan`'s
//!    2% — SmoothQuant on a tiny `K=8` problem has more per-cell
//!    rounding noise than typical LLM-sized matrices, but the
//!    contract is "comparable to FP linear within W8A8 noise").
//!
//! `#[ignore]` by default; needs a real CUDA device. Run with
//! `cargo test -p baracuda-kernels --release -- --ignored \
//!  smoothquant_smoke`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    ElementKind, PlanPreference, SmoothQuantLinearArgs, SmoothQuantLinearDescriptor,
    SmoothQuantLinearPlan, TensorMut, TensorRef, Workspace, S8,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// SmoothQuant smoothing vector: `s[k] = max(|A[:, k]|)^α / max(|W[:, k]|)^(1-α)`.
/// (Equation 5 in the SmoothQuant paper, with `α = 0.5` as the
/// canonical default for OPT / LLaMA-class models.)
fn compute_smoothing_vector(m: usize, n: usize, k: usize, a: &[f32], w: &[f32], alpha: f32) -> Vec<f32> {
    let mut a_max = vec![0f32; k];
    for r in 0..m {
        for c in 0..k {
            let v = a[r * k + c].abs();
            if v > a_max[c] {
                a_max[c] = v;
            }
        }
    }
    let mut w_max = vec![0f32; k];
    for r in 0..n {
        for c in 0..k {
            let v = w[r * k + c].abs();
            if v > w_max[c] {
                w_max[c] = v;
            }
        }
    }
    let mut s = vec![0f32; k];
    for c in 0..k {
        let a_term = a_max[c].max(1e-5).powf(alpha);
        let w_term = w_max[c].max(1e-5).powf(1.0 - alpha);
        s[c] = (a_term / w_term).max(1e-5);
    }
    s
}

/// Per-tensor symmetric int8 quantization. Returns `(q[M*K], scale)`.
fn quantize_per_tensor_symmetric(buf: &[f32]) -> (Vec<i8>, f32) {
    let qmax = 127i32;
    let max_abs = buf.iter().fold(0f32, |acc, &v| acc.max(v.abs()));
    let scale = if max_abs > 0.0 { max_abs / qmax as f32 } else { 1.0 };
    let inv_s = 1.0 / scale;
    let q: Vec<i8> = buf
        .iter()
        .map(|&v| ((v * inv_s).round_ties_even() as i32).clamp(-127, 127) as i8)
        .collect();
    (q, scale)
}

/// Per-output-channel symmetric int8 quantization, returning the
/// `[N]` scale vector + the row-major `[N, K]` int8 buffer.
fn quantize_weight_per_channel_symmetric(
    n: usize,
    k: usize,
    w: &[f32],
) -> (Vec<i8>, Vec<f32>) {
    let qmax = 127i32;
    let mut scale = vec![0f32; n];
    let mut wq = vec![0i8; n * k];
    for r in 0..n {
        let mut max_abs = 0f32;
        for c in 0..k {
            let v = w[r * k + c].abs();
            if v > max_abs {
                max_abs = v;
            }
        }
        let s = if max_abs > 0.0 { max_abs / qmax as f32 } else { 1.0 };
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
fn cpu_fp_matmul(m: usize, n: usize, k: usize, a: &[f32], w: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for mm in 0..m {
        for nn in 0..n {
            let mut acc = 0f32;
            for kk in 0..k {
                acc += a[mm * k + kk] * w[nn * k + kk];
            }
            out[mm * n + nn] = acc;
        }
    }
    out
}

#[test]
#[ignore]
fn smoothquant_linear_f32_basic() {
    let (ctx, stream) = setup();
    let m: i32 = 2;
    let n: i32 = 3;
    let k: i32 = 8;

    // Activation `[M, K]` — column 3 carries a large outlier (canonical
    // SmoothQuant-relevant case; this is exactly the per-channel
    // pattern Xiao et al. report for the LLM activation in §2).
    let host_a: Vec<f32> = vec![
        // row 0
        0.5, -0.3, 0.8,  12.0, -0.4, 0.6, 0.9, -0.7,
        // row 1
        0.4, -0.5,  0.6, -15.0, -0.2, 0.8, 0.3,  0.7,
    ];

    // Weight `[N, K]` — balanced per-channel magnitudes (channel 3
    // here is small relative to the activation's column 3 outlier;
    // SmoothQuant's whole point is to redistribute that imbalance).
    let host_w: Vec<f32> = vec![
        // chan 0
        0.4, -0.3, 0.5,  0.1, -0.2,  0.4, -0.3,  0.4,
        // chan 1
        0.2,  0.3, -0.4, 0.2,  0.4, -0.5,  0.3, -0.4,
        // chan 2
        -0.5, 0.4,  0.3, 0.2, -0.4,  0.5, -0.4,  0.3,
    ];

    let expected_fp = cpu_fp_matmul(m as usize, n as usize, k as usize, &host_a, &host_w);

    // ---- Step 1: compute the SmoothQuant smoothing vector. ----------
    let smooth = compute_smoothing_vector(
        m as usize, n as usize, k as usize, &host_a, &host_w, /*alpha=*/ 0.5,
    );

    // ---- Step 2: pre-smooth activation `A / diag(s)`. ---------------
    let mut host_a_smooth = vec![0f32; (m * k) as usize];
    for r in 0..m as usize {
        for c in 0..k as usize {
            host_a_smooth[r * k as usize + c] = host_a[r * k as usize + c] / smooth[c];
        }
    }

    // ---- Step 3: pre-smooth weight `diag(s) · W`. -------------------
    // For weight stored row-major as `[N, K]` (PyTorch convention), the
    // smoothing folds into column `c` across all output channels.
    let mut host_w_smooth = vec![0f32; (n * k) as usize];
    for r in 0..n as usize {
        for c in 0..k as usize {
            host_w_smooth[r * k as usize + c] = host_w[r * k as usize + c] * smooth[c];
        }
    }

    // ---- Step 4: quantize the smoothed activation per-tensor. -------
    let (host_a_q, act_scale) = quantize_per_tensor_symmetric(&host_a_smooth);

    // ---- Step 5: quantize the smoothed weight per-output-channel. ---
    let (host_w_q, host_w_scale) =
        quantize_weight_per_channel_symmetric(n as usize, k as usize, &host_w_smooth);

    // Device buffers.
    let host_a_u: &[u8] = unsafe {
        core::slice::from_raw_parts(host_a_q.as_ptr() as *const u8, host_a_q.len())
    };
    let host_w_u: &[u8] = unsafe {
        core::slice::from_raw_parts(host_w_q.as_ptr() as *const u8, host_w_q.len())
    };
    let dev_a_bytes = DeviceBuffer::from_slice(&ctx, host_a_u).expect("upload a_q");
    let dev_w_bytes = DeviceBuffer::from_slice(&ctx, host_w_u).expect("upload w_q");
    let dev_a_q = dev_a_bytes.view_as::<S8>();
    let dev_w_q = dev_w_bytes.view_as::<S8>();
    let dev_w_scale = DeviceBuffer::from_slice(&ctx, &host_w_scale).expect("upload w_scale");
    let mut dev_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (m * n) as usize).expect("alloc out");
    let mut dev_act_scratch: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, m as usize).expect("alloc act_scale_scratch");

    let desc = SmoothQuantLinearDescriptor {
        m, n, k,
        act_scale,
        activation_element: ElementKind::S8,
        weight_element: ElementKind::S8,
        output_element: ElementKind::F32,
    };
    let plan = SmoothQuantLinearPlan::<f32, S8>::select(&stream, &desc, PlanPreference::default())
        .expect("select SmoothQuantLinearPlan");

    let args = SmoothQuantLinearArgs::<f32, S8> {
        act_q: TensorRef {
            data: dev_a_q,
            shape: [m, k],
            stride: [k as i64, 1],
        },
        weight_q: TensorRef {
            data: dev_w_q,
            shape: [n, k],
            stride: [k as i64, 1],
        },
        weight_scale: TensorRef {
            data: dev_w_scale.as_slice(),
            shape: [n],
            stride: [1],
        },
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [m, n],
            stride: [n as i64, 1],
        },
        act_scale_scratch: TensorMut {
            data: dev_act_scratch.as_slice_mut(),
            shape: [m],
            stride: [1],
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; (m * n) as usize];
    dev_out.copy_to_host(&mut got).expect("download out");

    // ---- Step 8: validate against the FP reference. -----------------
    //
    // SmoothQuant W8A8 should match FP linear to within ~3% relative
    // error on this tiny K=8 problem. The injected outlier in the
    // activation (×10 magnitude) is precisely the case SmoothQuant
    // was designed for — without smoothing, per-tensor activation
    // quantization would lose ~10× precision on the other columns;
    // with smoothing, the per-column scale is normalized and the
    // per-tensor scale only has to cover the smoothed range.
    let max_abs_expected = expected_fp
        .iter()
        .fold(0f32, |acc, &v| acc.max(v.abs()))
        .max(1e-3);
    let tol = 0.03 * max_abs_expected;
    for (i, (g, &e)) in got.iter().zip(expected_fp.iter()).enumerate() {
        let diff = (g - e).abs();
        assert!(
            diff <= tol,
            "SmoothQuant mismatch @ {i}: got={g} expected={e} diff={diff} tol={tol}\n\
             full got = {got:?}\nfull expected = {expected_fp:?}\n\
             act_scale = {act_scale}, w_scale = {host_w_scale:?}, smooth = {smooth:?}"
        );
    }
}

#[test]
#[ignore]
fn smoothquant_select_rejects_unsupported_dtypes() {
    // Compile-time: smoke that select rejects unsupported dtype combos
    // without needing a GPU. Uses a stub stream — select() only
    // touches the descriptor.
    use half::f16;

    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    // f16 TIn currently unsupported (matches QuantizedLinearPlan).
    let desc_f16 = SmoothQuantLinearDescriptor {
        m: 2, n: 3, k: 8, act_scale: 0.5,
        activation_element: ElementKind::S8,
        weight_element: ElementKind::S8,
        output_element: ElementKind::F16,
    };
    let r = SmoothQuantLinearPlan::<f16, S8>::select(&stream, &desc_f16, PlanPreference::default());
    assert!(r.is_err(), "f16 should be rejected");
    drop(ctx);
}

#[test]
#[ignore]
fn smoothquant_select_rejects_negative_dims() {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let desc = SmoothQuantLinearDescriptor {
        m: -1, n: 3, k: 8, act_scale: 0.5,
        activation_element: ElementKind::S8,
        weight_element: ElementKind::S8,
        output_element: ElementKind::F32,
    };
    let r = SmoothQuantLinearPlan::<f32, S8>::select(&stream, &desc, PlanPreference::default());
    assert!(r.is_err(), "negative M should be rejected");
    drop(ctx);
}

#[test]
#[ignore]
fn smoothquant_select_rejects_non_finite_scale() {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let desc = SmoothQuantLinearDescriptor {
        m: 2, n: 3, k: 8, act_scale: f32::INFINITY,
        activation_element: ElementKind::S8,
        weight_element: ElementKind::S8,
        output_element: ElementKind::F32,
    };
    let r = SmoothQuantLinearPlan::<f32, S8>::select(&stream, &desc, PlanPreference::default());
    assert!(r.is_err(), "non-finite act_scale should be rejected");
    drop(ctx);
}
