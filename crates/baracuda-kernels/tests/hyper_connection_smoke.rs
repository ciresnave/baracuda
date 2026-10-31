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
//! Gated behind the `device-tests` feature — requires a real CUDA device + the `mhc`
//! cargo feature.

#![cfg(feature = "mhc")]
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    ElementKind, HyperConnectionArgs, HyperConnectionBackwardArgs, HyperConnectionDescriptor,
    HyperConnectionPlan, PlanPreference, TensorMut, TensorRef, Workspace, contiguous_stride,
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
            x_expanded: TensorRef::new(dx.as_slice(), sx, contiguous_stride(sx)).unwrap(),
            rmsnorm_weight: TensorRef::new(dgamma.as_slice(), sg, contiguous_stride(sg)).unwrap(),
            h_pre: TensorRef::new(dh_pre.as_slice(), sh, contiguous_stride(sh)).unwrap(),
            h_post: TensorRef::new(dh_post.as_slice(), sh, contiguous_stride(sh)).unwrap(),
            h_res: TensorRef::new(dh_res.as_slice(), sr, contiguous_stride(sr)).unwrap(),
            out: TensorMut::new(dout.as_slice_mut(), sx, contiguous_stride(sx)).unwrap(),
        },
    )
    .expect("plan run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; (B * N * C) as usize];
    dout.copy_to_host(&mut got).expect("dl");
    got
}

#[test]
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

/// f64 reference of the static-H forward: `M = Sinkhorn(exp(h_res))` (row then column
/// normalisation, `iters` times), `y_agg = sum_i sigmoid(h_pre[i]) x[:, i]`,
/// `y_norm = y_agg / sqrt(mean(y_agg^2) + eps) * gamma`,
/// `out[b, i] = sum_j M[i, j] x[b, j] + 2 * sigmoid(h_post[i]) * y_norm[b]`.
struct HcRef {
    x: Vec<f64>,
    gamma: Vec<f64>,
    h_pre: Vec<f64>,
    h_post: Vec<f64>,
    h_res: Vec<f64>,
}

fn sigmoid(v: f64) -> f64 {
    1.0 / (1.0 + (-v).exp())
}

fn sinkhorn_ref(h_res: &[f64], n: usize, iters: i32, eps: f64) -> Vec<f64> {
    let mut m: Vec<f64> = h_res.iter().map(|v| v.exp()).collect();
    for _ in 0..iters {
        for r in 0..n {
            let s: f64 = (0..n).map(|c| m[r * n + c]).sum();
            if s > eps {
                (0..n).for_each(|c| m[r * n + c] /= s);
            }
        }
        for c in 0..n {
            let s: f64 = (0..n).map(|r| m[r * n + c]).sum();
            if s > eps {
                (0..n).for_each(|r| m[r * n + c] /= s);
            }
        }
    }
    m
}

impl HcRef {
    fn forward(&self) -> Vec<f64> {
        let (b_n, n, c_n) = (B as usize, N as usize, C as usize);
        let m = sinkhorn_ref(&self.h_res, n, SINKHORN_ITERS, f64::from(EPS));
        let mut out = vec![0.0; b_n * n * c_n];
        for b in 0..b_n {
            let agg: Vec<f64> = (0..c_n)
                .map(|c| {
                    (0..n)
                        .map(|i| sigmoid(self.h_pre[i]) * self.x[(b * n + i) * c_n + c])
                        .sum()
                })
                .collect();
            let ms = agg.iter().map(|v| v * v).sum::<f64>() / c_n as f64;
            let inv = 1.0 / (ms + f64::from(EPS)).sqrt();
            for i in 0..n {
                for c in 0..c_n {
                    let mix: f64 = (0..n)
                        .map(|j| m[i * n + j] * self.x[(b * n + j) * c_n + c])
                        .sum();
                    let post = 2.0 * sigmoid(self.h_post[i]) * agg[c] * inv * self.gamma[c];
                    out[(b * n + i) * c_n + c] = mix + post;
                }
            }
        }
        out
    }
}

fn widen(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

#[test]
fn hyper_connection_f32_static_h_matches_f64_reference() {
    let (ctx, stream) = setup();
    let gamma_bf = gen_gamma_bf16();
    let r = HcRef {
        x: widen(&gen_x()),
        gamma: gamma_bf.iter().map(|g| f64::from(g.to_f32())).collect(),
        h_pre: widen(&gen_h_pre()),
        h_post: widen(&gen_h_post()),
        h_res: widen(&gen_h_res()),
    };
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
    let got = run_once(
        &ctx,
        &stream,
        &plan,
        &gen_x(),
        &gamma_bf,
        &gen_h_pre(),
        &gen_h_post(),
        &gen_h_res(),
    );
    let want = r.forward();
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (f64::from(*g) - w).abs() < 2e-4 * w.abs().max(1.0),
            "forward mismatch @ {i}: got {g}, want {w}"
        );
    }
}

// ---- Tier 2: backward, checked against central differences of the f64 reference ----

/// Parameter groups of `HcRef`, in the order the gradients are compared.
const GROUPS: [&str; 5] = ["x", "gamma", "h_pre", "h_post", "h_res"];

impl HcRef {
    fn group_mut(&mut self, g: usize) -> &mut Vec<f64> {
        match g {
            0 => &mut self.x,
            1 => &mut self.gamma,
            2 => &mut self.h_pre,
            3 => &mut self.h_post,
            _ => &mut self.h_res,
        }
    }

    /// `L = sum(d_out * forward())`.
    fn loss(&self, d_out: &[f64]) -> f64 {
        self.forward().iter().zip(d_out).map(|(o, d)| o * d).sum()
    }

    /// Central-difference gradient of `loss` for every element of group `g`.
    fn numeric_grad(&mut self, g: usize, d_out: &[f64]) -> Vec<f64> {
        let h = 1e-6;
        (0..self.group_mut(g).len())
            .map(|i| {
                let orig = self.group_mut(g)[i];
                self.group_mut(g)[i] = orig + h;
                let up = self.loss(d_out);
                self.group_mut(g)[i] = orig - h;
                let down = self.loss(d_out);
                self.group_mut(g)[i] = orig;
                (up - down) / (2.0 * h)
            })
            .collect()
    }
}

fn gen_d_out() -> Vec<f32> {
    (0..(B * N * C) as usize)
        .map(|i| ((i as f32) * 0.021 + 0.3).cos() * 0.5)
        .collect()
}

/// Runs forward then backward on a training plan; returns the five gradients in `GROUPS` order.
fn run_backward_once(ctx: &Context, stream: &Stream, r: &HcRef) -> Vec<Vec<f32>> {
    let desc = HyperConnectionDescriptor {
        batch: B,
        hidden_dim: C,
        n_streams: N,
        sinkhorn_iters: SINKHORN_ITERS,
        eps: EPS,
        element: ElementKind::F32,
    };
    let plan =
        HyperConnectionPlan::<f32>::select_training(stream, &desc, PlanPreference::default())
            .expect("select_training");
    let narrow = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
    let gamma_bf: Vec<bf16> = r.gamma.iter().map(|&g| bf16::from_f64(g)).collect();
    run_once(
        ctx,
        stream,
        &plan,
        &narrow(&r.x),
        &gamma_bf,
        &narrow(&r.h_pre),
        &narrow(&r.h_post),
        &narrow(&r.h_res),
    );
    let d_out = DeviceBuffer::from_slice(ctx, &gen_d_out()).expect("up d_out");
    let mut dx: DeviceBuffer<f32> = DeviceBuffer::zeros(ctx, (B * N * C) as usize).expect("dx");
    let mut dg: DeviceBuffer<f32> = DeviceBuffer::zeros(ctx, C as usize).expect("dg");
    let mut dpre: DeviceBuffer<f32> = DeviceBuffer::zeros(ctx, N as usize).expect("dpre");
    let mut dpost: DeviceBuffer<f32> = DeviceBuffer::zeros(ctx, N as usize).expect("dpost");
    let mut dres: DeviceBuffer<f32> = DeviceBuffer::zeros(ctx, (N * N) as usize).expect("dres");
    let (sx, sg, sh, sr) = ([B, N, C], [C], [N], [N, N]);
    plan.run_backward(
        stream,
        HyperConnectionBackwardArgs {
            d_out: TensorRef::new(d_out.as_slice(), sx, contiguous_stride(sx)).unwrap(),
            d_x_expanded: TensorMut::new(dx.as_slice_mut(), sx, contiguous_stride(sx)).unwrap(),
            d_rmsnorm_weight: TensorMut::new(dg.as_slice_mut(), sg, contiguous_stride(sg)).unwrap(),
            d_h_pre: TensorMut::new(dpre.as_slice_mut(), sh, contiguous_stride(sh)).unwrap(),
            d_h_post: TensorMut::new(dpost.as_slice_mut(), sh, contiguous_stride(sh)).unwrap(),
            d_h_res: TensorMut::new(dres.as_slice_mut(), sr, contiguous_stride(sr)).unwrap(),
        },
    )
    .expect("run_backward");
    stream.synchronize().expect("sync");
    let host = |b: &DeviceBuffer<f32>, n: usize| {
        let mut v = vec![0f32; n];
        b.copy_to_host(&mut v).expect("dl");
        v
    };
    vec![
        host(&dx, (B * N * C) as usize),
        host(&dg, C as usize),
        host(&dpre, N as usize),
        host(&dpost, N as usize),
        host(&dres, (N * N) as usize),
    ]
}

#[test]
fn hyper_connection_f32_static_h_backward_matches_finite_differences() {
    let (ctx, stream) = setup();
    let gamma_bf = gen_gamma_bf16();
    let mut r = HcRef {
        x: widen(&gen_x()),
        gamma: gamma_bf.iter().map(|g| f64::from(g.to_f32())).collect(),
        h_pre: widen(&gen_h_pre()),
        h_post: widen(&gen_h_post()),
        h_res: widen(&gen_h_res()),
    };
    let d_out = widen(&gen_d_out());
    let got = run_backward_once(&ctx, &stream, &r);
    let mut bad = Vec::new();
    for (g, name) in GROUPS.iter().enumerate() {
        let want = r.numeric_grad(g, &d_out);
        let scale = want.iter().fold(1e-3f64, |m, v| m.max(v.abs()));
        let worst = got[g]
            .iter()
            .zip(&want)
            .map(|(a, w)| (f64::from(*a) - w).abs() / scale)
            .fold(0.0f64, f64::max);
        eprintln!("d_{name}: worst |got - fd| / scale = {worst:.3e} (scale {scale:.3e})");
        if worst >= 1e-2 {
            bad.push(*name);
        }
    }
    assert!(
        bad.is_empty(),
        "gradient groups off from finite differences: {bad:?}"
    );
}

#[test]
fn hyper_connection_backward_needs_a_training_plan() {
    let (ctx, stream) = setup();
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
    let d_out = DeviceBuffer::from_slice(&ctx, &gen_d_out()).expect("up");
    let mut dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, (B * N * C) as usize).expect("dx");
    let mut dg: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, C as usize).expect("dg");
    let mut dpre: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, N as usize).expect("dpre");
    let mut dpost: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, N as usize).expect("dpost");
    let mut dres: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, (N * N) as usize).expect("dres");
    let (sx, sg, sh, sr) = ([B, N, C], [C], [N], [N, N]);
    let res = plan.run_backward(
        &stream,
        HyperConnectionBackwardArgs {
            d_out: TensorRef::new(d_out.as_slice(), sx, contiguous_stride(sx)).unwrap(),
            d_x_expanded: TensorMut::new(dx.as_slice_mut(), sx, contiguous_stride(sx)).unwrap(),
            d_rmsnorm_weight: TensorMut::new(dg.as_slice_mut(), sg, contiguous_stride(sg)).unwrap(),
            d_h_pre: TensorMut::new(dpre.as_slice_mut(), sh, contiguous_stride(sh)).unwrap(),
            d_h_post: TensorMut::new(dpost.as_slice_mut(), sh, contiguous_stride(sh)).unwrap(),
            d_h_res: TensorMut::new(dres.as_slice_mut(), sr, contiguous_stride(sr)).unwrap(),
        },
    );
    assert!(
        res.is_err(),
        "run_backward on an inference plan must be refused"
    );
}
