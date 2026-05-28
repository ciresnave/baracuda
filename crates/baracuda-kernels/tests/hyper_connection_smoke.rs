//! Real-GPU smoke test for `HyperConnectionPlan` (Phase 43, Tier 1).
//!
//! mHC.cu's static-H FW path is a multi-step pipeline (RMSNorm +
//! sigmoid-gated stream-aggregate + Sinkhorn-Knopp iteration +
//! sigmoid-gated mix-and-add). Replicating the math bit-for-bit in
//! Rust is non-trivial — we don't, in Tier 1. Instead we assert:
//!
//!   1. The kernel returns Ok and writes finite, non-NaN output.
//!   2. Output is non-zero for non-zero input (sanity that we're not
//!      reading uninitialized memory or short-circuiting).
//!   3. Doubling `H_post` changes the output meaningfully (the
//!      post-mixing path is exercised).
//!   4. Output is element-wise reproducible across two back-to-back
//!      launches with the same inputs (matches the
//!      `bit_stable_on_same_hardware = true` claim).
//!
//! `#[ignore]` by default — requires a real CUDA device + the `mhc`
//! cargo feature.

#![cfg(feature = "mhc")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, HyperConnectionArgs, HyperConnectionDescriptor,
    HyperConnectionPlan, PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::bf16;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

const B: i32 = 2;
const N: i32 = 4;
const C: i32 = 16;
const SINKHORN_ITERS: i32 = 20;
const EPS: f32 = 1e-5;

fn gen_x() -> Vec<f32> {
    // Modestly varied inputs in (-0.5, 0.5).
    let len = (B * N * C) as usize;
    (0..len)
        .map(|i| ((i as f32) * 0.013 - 0.5).sin() * 0.4)
        .collect()
}

fn gen_gamma_bf16() -> Vec<bf16> {
    // RMSNorm gamma centered around 1.0 (the upstream test fills with
    // bf16 random in (0.75, 1.25) — we use a deterministic pattern in
    // the same range).
    (0..C as usize)
        .map(|i| bf16::from_f32(0.9 + 0.2 * ((i as f32) * 0.31).sin()))
        .collect()
}

fn gen_h_pre() -> Vec<f32> {
    // Pre-sigmoid logits centered around 0 (sigmoid → 0.5).
    (0..N as usize).map(|i| 0.1 * (i as f32) - 0.15).collect()
}

fn gen_h_post() -> Vec<f32> {
    // Pre-sigmoid logits centered around 0; kernel applies 2*sigmoid
    // → values around 1.0.
    (0..N as usize).map(|i| 0.2 * (i as f32) - 0.3).collect()
}

fn gen_h_res() -> Vec<f32> {
    // Small uniform-ish values that, after exp+Sinkhorn, become a
    // close-to-uniform doubly-stochastic matrix.
    let mut out = Vec::with_capacity((N * N) as usize);
    for i in 0..N {
        for j in 0..N {
            // Slightly favor the diagonal so the resulting M isn't
            // exactly uniform.
            let v = if i == j { 0.05 } else { 0.0 };
            out.push(v);
        }
    }
    out
}

fn run_once(
    ctx: &Context,
    stream: &Stream,
    plan: &HyperConnectionPlan<f32>,
    x: &[f32],
    gamma: &[bf16],
    h_pre: &[f32],
    h_post: &[f32],
    h_res: &[f32],
) -> Vec<f32> {
    let dx = DeviceBuffer::from_slice(ctx, x).expect("up x");
    let dgamma = DeviceBuffer::from_slice(ctx, gamma).expect("up gamma");
    let dh_pre = DeviceBuffer::from_slice(ctx, h_pre).expect("up h_pre");
    let dh_post = DeviceBuffer::from_slice(ctx, h_post).expect("up h_post");
    let dh_res = DeviceBuffer::from_slice(ctx, h_res).expect("up h_res");
    let mut dout: DeviceBuffer<f32> =
        DeviceBuffer::zeros(ctx, (B * N * C) as usize).expect("alloc out");

    let sx = [B, N, C];
    let sg = [C];
    let sh = [N];
    let sr = [N, N];

    plan.run(
        stream,
        Workspace::None,
        HyperConnectionArgs {
            x_expanded: TensorRef {
                data: dx.as_slice(),
                shape: sx,
                stride: contiguous_stride(sx),
            },
            rmsnorm_weight: TensorRef {
                data: dgamma.as_slice(),
                shape: sg,
                stride: contiguous_stride(sg),
            },
            h_pre: TensorRef {
                data: dh_pre.as_slice(),
                shape: sh,
                stride: contiguous_stride(sh),
            },
            h_post: TensorRef {
                data: dh_post.as_slice(),
                shape: sh,
                stride: contiguous_stride(sh),
            },
            h_res: TensorRef {
                data: dh_res.as_slice(),
                shape: sr,
                stride: contiguous_stride(sr),
            },
            out: TensorMut {
                data: dout.as_slice_mut(),
                shape: sx,
                stride: contiguous_stride(sx),
            },
        },
    )
    .expect("plan run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; (B * N * C) as usize];
    dout.copy_to_host(&mut got).expect("dl");
    got
}

#[test]
#[ignore]
fn hyper_connection_f32_static_h_basic() {
    let (ctx, stream) = setup();
    let x = gen_x();
    let gamma = gen_gamma_bf16();
    let h_pre = gen_h_pre();
    let h_post = gen_h_post();
    let h_res = gen_h_res();

    let desc = HyperConnectionDescriptor {
        batch: B,
        hidden_dim: C,
        n_streams: N,
        sinkhorn_iters: SINKHORN_ITERS,
        eps: EPS,
        element: ElementKind::F32,
    };
    let plan = HyperConnectionPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("plan select");

    let got = run_once(&ctx, &stream, &plan, &x, &gamma, &h_pre, &h_post, &h_res);

    // (1) Output is finite and non-NaN.
    let mut max_abs = 0f32;
    for (i, &v) in got.iter().enumerate() {
        assert!(
            v.is_finite(),
            "non-finite output at i={i}: {v} (NaN/Inf is a kernel bug)"
        );
        max_abs = max_abs.max(v.abs());
    }
    // (2) Non-zero output for non-zero input.
    assert!(
        max_abs > 1e-6,
        "all-zero output (max_abs={max_abs}) — kernel likely short-circuited"
    );

    // (4) Bit-stable: two back-to-back runs must produce identical
    // bytes (the `bit_stable_on_same_hardware = true` claim).
    let got2 = run_once(&ctx, &stream, &plan, &x, &gamma, &h_pre, &h_post, &h_res);
    for i in 0..got.len() {
        assert_eq!(
            got[i].to_bits(),
            got2[i].to_bits(),
            "bit-stability broken at i={i}: run1={a} run2={b}",
            a = got[i],
            b = got2[i]
        );
    }
}

#[test]
#[ignore]
fn hyper_connection_f32_static_h_h_post_changes_output() {
    // Verify that the H_post code path is exercised: doubling
    // (in logit space → bigger post-sigmoid scale) H_post should
    // change the output meaningfully.
    let (ctx, stream) = setup();
    let x = gen_x();
    let gamma = gen_gamma_bf16();
    let h_pre = gen_h_pre();
    let h_post_base = gen_h_post();
    let h_post_scaled: Vec<f32> = h_post_base.iter().map(|&v| v + 1.0).collect();
    let h_res = gen_h_res();

    let desc = HyperConnectionDescriptor {
        batch: B,
        hidden_dim: C,
        n_streams: N,
        sinkhorn_iters: SINKHORN_ITERS,
        eps: EPS,
        element: ElementKind::F32,
    };
    let plan = HyperConnectionPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("plan select");

    let base = run_once(
        &ctx,
        &stream,
        &plan,
        &x,
        &gamma,
        &h_pre,
        &h_post_base,
        &h_res,
    );
    let scaled = run_once(
        &ctx,
        &stream,
        &plan,
        &x,
        &gamma,
        &h_pre,
        &h_post_scaled,
        &h_res,
    );

    let mut max_diff = 0f32;
    for i in 0..base.len() {
        max_diff = max_diff.max((base[i] - scaled[i]).abs());
    }
    // sigmoid(0) = 0.5; sigmoid(1) ≈ 0.731 → post-scale changes by
    // factor ~0.46. The post contribution is `2 * sigmoid(.) * y_norm`,
    // and `y_norm` is RMSNorm(aggregate) — magnitude ~ O(1). Expect a
    // change > 1e-3 in at least one cell.
    assert!(
        max_diff > 1e-3,
        "doubling H_post had no visible effect (max_diff={max_diff}); \
         the post-mixing code path may not be wired correctly"
    );
}
