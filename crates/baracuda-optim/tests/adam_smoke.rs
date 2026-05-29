//! Adam — single-step smoke test against a CPU reference.
//!
//! Builds a small synthetic parameter set, runs one AdamStepPlan
//! launch, and compares against a numpy-equivalent reference computed
//! on the host. Validates:
//!
//! 1. The multi-tensor apply launch produces correct results across
//!    multiple tensors of varying shapes.
//! 2. Bias correction is wired correctly (`1 - beta^t` term).
//! 3. The classic Adam vs AdamW mode flag is honored.
//! 4. Weight decay is applied in both modes.

use baracuda_driver::{Context, Device, DeviceBuffer};
use baracuda_optim::{AdamConfig, AdamMode, AdamStepPlan, TensorList};

/// CPU reference Adam update — exactly mirrors the kernel formula.
fn adam_step_cpu(
    params: &mut [f32],
    grads: &[f32],
    exp_avg: &mut [f32],
    exp_avg_sq: &mut [f32],
    cfg: &AdamConfig,
    step: i32,
) {
    let beta1_corr = if cfg.bias_correction {
        1.0 - cfg.beta1.powi(step)
    } else {
        1.0
    };
    let beta2_corr = if cfg.bias_correction {
        1.0 - cfg.beta2.powi(step)
    } else {
        1.0
    };

    for i in 0..params.len() {
        let mut g = grads[i];
        let p = params[i];

        if matches!(cfg.mode, AdamMode::Classic) && cfg.weight_decay != 0.0 {
            g = g + cfg.weight_decay * p;
        }

        let m = cfg.beta1 * exp_avg[i] + (1.0 - cfg.beta1) * g;
        let v = cfg.beta2 * exp_avg_sq[i] + (1.0 - cfg.beta2) * g * g;

        let m_hat = m / beta1_corr;
        let v_hat = v / beta2_corr;
        let update = m_hat / (v_hat.sqrt() + cfg.epsilon);

        if matches!(cfg.mode, AdamMode::Decoupled) && cfg.weight_decay != 0.0 {
            params[i] = p - cfg.lr * (update + cfg.weight_decay * p);
        } else {
            params[i] = p - cfg.lr * update;
        }

        exp_avg[i] = m;
        exp_avg_sq[i] = v;
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn adam_f32_two_tensors_step_1() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = baracuda_driver::Stream::new(&ctx).unwrap();

    // Two tensors of different sizes — exercises the per-tensor
    // metadata pack + the chunk-clamp path on the second one.
    let p1_host: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.01).collect();
    let p2_host: Vec<f32> = (0..3000).map(|i| (i as f32) * -0.005).collect();

    let g1_host: Vec<f32> = (0..1024).map(|i| ((i % 17) as f32) * 0.03 - 0.2).collect();
    let g2_host: Vec<f32> = (0..3000).map(|i| ((i % 7) as f32) * -0.01 + 0.1).collect();

    let m1_init: Vec<f32> = vec![0.0; 1024];
    let m2_init: Vec<f32> = vec![0.0; 3000];
    let v1_init: Vec<f32> = vec![0.0; 1024];
    let v2_init: Vec<f32> = vec![0.0; 3000];

    // NOTE: although params/moments/vels are mutated by the kernel,
    // TensorList borrows are immutable (the buffer itself moves no
    // bytes — only the kernel touches it via the device pointer).
    let p1 = DeviceBuffer::from_slice(&ctx, &p1_host).unwrap();
    let p2 = DeviceBuffer::from_slice(&ctx, &p2_host).unwrap();
    let g1 = DeviceBuffer::from_slice(&ctx, &g1_host).unwrap();
    let g2 = DeviceBuffer::from_slice(&ctx, &g2_host).unwrap();
    let m1 = DeviceBuffer::from_slice(&ctx, &m1_init).unwrap();
    let m2 = DeviceBuffer::from_slice(&ctx, &m2_init).unwrap();
    let v1 = DeviceBuffer::from_slice(&ctx, &v1_init).unwrap();
    let v2 = DeviceBuffer::from_slice(&ctx, &v2_init).unwrap();

    let cfg = AdamConfig {
        lr: 1e-2,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        weight_decay: 0.01,
        bias_correction: true,
        mode: AdamMode::Classic,
    };

    // Build TensorLists. The four lists pack pointers as immutable
    // borrows; even though Adam writes to them, the kernel's contract
    // requires a stable pointer for the launch duration which the
    // borrow checker can't express directly.
    let params = TensorList::new(&[&p1, &p2]).unwrap();
    let grads = TensorList::new(&[&g1, &g2]).unwrap();
    let mom = TensorList::new(&[&m1, &m2]).unwrap();
    let vel = TensorList::new(&[&v1, &v2]).unwrap();

    let plan = AdamStepPlan::<f32>::new(cfg);
    plan.step(&params, &grads, &mom, &vel, /*step_index=*/ 1, &stream)
        .expect("Adam step");

    stream.synchronize().unwrap();

    // Reference computation on host.
    let mut p1_ref = p1_host.clone();
    let mut p2_ref = p2_host.clone();
    let mut m1_ref = m1_init.clone();
    let mut m2_ref = m2_init.clone();
    let mut v1_ref = v1_init.clone();
    let mut v2_ref = v2_init.clone();
    adam_step_cpu(&mut p1_ref, &g1_host, &mut m1_ref, &mut v1_ref, &cfg, 1);
    adam_step_cpu(&mut p2_ref, &g2_host, &mut m2_ref, &mut v2_ref, &cfg, 1);

    // Read back & verify.
    let mut p1_got = vec![0.0f32; p1_host.len()];
    let mut p2_got = vec![0.0f32; p2_host.len()];
    let mut m1_got = vec![0.0f32; p1_host.len()];
    let mut m2_got = vec![0.0f32; p2_host.len()];
    let mut v1_got = vec![0.0f32; p1_host.len()];
    let mut v2_got = vec![0.0f32; p2_host.len()];
    p1.copy_to_host(&mut p1_got).unwrap();
    p2.copy_to_host(&mut p2_got).unwrap();
    m1.copy_to_host(&mut m1_got).unwrap();
    m2.copy_to_host(&mut m2_got).unwrap();
    v1.copy_to_host(&mut v1_got).unwrap();
    v2.copy_to_host(&mut v2_got).unwrap();

    // Tolerance: 1e-4 relative — the Adam BFP-vs-FP32 reference can differ
    // by a few ulp per chained float op (m_hat / (sqrt(v_hat) + eps) is
    // ~6 ops deep), accumulating to ~1e-4 at the magnitudes we test.
    let tol = 1e-4f32;
    for (i, (g, e)) in p1_got.iter().zip(&p1_ref).enumerate() {
        assert!(
            (g - e).abs() <= tol * (1.0 + e.abs()),
            "p1[{i}] = {g}, want {e}",
        );
    }
    for (i, (g, e)) in p2_got.iter().zip(&p2_ref).enumerate() {
        assert!(
            (g - e).abs() <= tol * (1.0 + e.abs()),
            "p2[{i}] = {g}, want {e}",
        );
    }
    for (i, (g, e)) in m1_got.iter().zip(&m1_ref).enumerate() {
        assert!(
            (g - e).abs() <= tol * (1.0 + e.abs()),
            "m1[{i}] = {g}, want {e}",
        );
    }
    for (i, (g, e)) in v1_got.iter().zip(&v1_ref).enumerate() {
        assert!(
            (g - e).abs() <= tol * (1.0 + e.abs()),
            "v1[{i}] = {g}, want {e}",
        );
    }
    for (i, (g, e)) in m2_got.iter().zip(&m2_ref).enumerate() {
        assert!(
            (g - e).abs() <= tol * (1.0 + e.abs()),
            "m2[{i}] = {g}, want {e}",
        );
    }
    for (i, (g, e)) in v2_got.iter().zip(&v2_ref).enumerate() {
        assert!(
            (g - e).abs() <= tol * (1.0 + e.abs()),
            "v2[{i}] = {g}, want {e}",
        );
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn adamw_f32_decoupled_decay() {
    // Same shapes, but mode = Decoupled — ensures the AdamW branch
    // takes the alternate `pi = pi - lr*(update + decay*pi)` path.
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = baracuda_driver::Stream::new(&ctx).unwrap();

    let n = 2048;
    let p_host: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
    let g_host: Vec<f32> = (0..n).map(|i| (i as f32).cos() * 0.1).collect();
    let m_init: Vec<f32> = vec![0.0; n];
    let v_init: Vec<f32> = vec![0.0; n];

    let p = DeviceBuffer::from_slice(&ctx, &p_host).unwrap();
    let g = DeviceBuffer::from_slice(&ctx, &g_host).unwrap();
    let m = DeviceBuffer::from_slice(&ctx, &m_init).unwrap();
    let v = DeviceBuffer::from_slice(&ctx, &v_init).unwrap();

    let cfg = AdamConfig {
        lr: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        weight_decay: 0.1,
        bias_correction: true,
        mode: AdamMode::Decoupled,
    };

    let params = TensorList::new(&[&p]).unwrap();
    let grads = TensorList::new(&[&g]).unwrap();
    let mom = TensorList::new(&[&m]).unwrap();
    let vel = TensorList::new(&[&v]).unwrap();

    AdamStepPlan::<f32>::new(cfg)
        .step(&params, &grads, &mom, &vel, 1, &stream)
        .expect("AdamW step");
    stream.synchronize().unwrap();

    let mut p_ref = p_host.clone();
    let mut m_ref = m_init.clone();
    let mut v_ref = v_init.clone();
    adam_step_cpu(&mut p_ref, &g_host, &mut m_ref, &mut v_ref, &cfg, 1);

    let mut p_got = vec![0.0f32; n];
    p.copy_to_host(&mut p_got).unwrap();
    m.copy_to_host(&mut vec![0.0f32; n]).unwrap(); // discard - check p only
    v.copy_to_host(&mut vec![0.0f32; n]).unwrap();

    // Tolerance: 1e-4 relative — the Adam BFP-vs-FP32 reference can differ
    // by a few ulp per chained float op (m_hat / (sqrt(v_hat) + eps) is
    // ~6 ops deep), accumulating to ~1e-4 at the magnitudes we test.
    let tol = 1e-4f32;
    for (i, (g, e)) in p_got.iter().zip(&p_ref).enumerate() {
        assert!(
            (g - e).abs() <= tol * (1.0 + e.abs()),
            "AdamW p[{i}] = {g}, want {e}",
        );
    }
}
