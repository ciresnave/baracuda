//! SGD — single-step smoke test against a PyTorch-equivalent reference.
//!
//! Validates momentum + dampening + weight_decay + Nesterov combos
//! against a host-side reference matching `torch.optim.SGD`.

use baracuda_driver::{Context, Device, DeviceBuffer};
use baracuda_optim::{SgdConfig, SgdStepPlan, TensorList};

fn sgd_step_cpu(
    params: &mut [f32],
    grads: &[f32],
    momentum: &mut [f32],
    cfg: &SgdConfig,
    first_step: bool,
) {
    for i in 0..params.len() {
        let mut g = grads[i] * cfg.grad_scale;
        let p = params[i];

        if !cfg.weight_decay_after_momentum && cfg.weight_decay != 0.0 {
            g = g + cfg.weight_decay * p;
        }

        let update = if cfg.momentum != 0.0 {
            let mut v = if first_step {
                g
            } else {
                cfg.momentum * momentum[i] + (1.0 - cfg.dampening) * g
            };
            momentum[i] = v;
            if cfg.weight_decay_after_momentum && cfg.weight_decay != 0.0 {
                v = v + cfg.weight_decay * p;
            }
            if cfg.nesterov {
                g + cfg.momentum * v
            } else {
                v
            }
        } else {
            g
        };

        params[i] = p - cfg.lr * update;
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn sgd_f32_momentum_nesterov_first_step() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = baracuda_driver::Stream::new(&ctx).unwrap();

    let n1 = 1024usize;
    let n2 = 5000usize;
    let p1_host: Vec<f32> = (0..n1).map(|i| (i as f32) * 0.005).collect();
    let p2_host: Vec<f32> = (0..n2).map(|i| ((i % 11) as f32) - 5.0).collect();
    let g1_host: Vec<f32> = (0..n1).map(|i| ((i as f32) * 0.01).sin()).collect();
    let g2_host: Vec<f32> = (0..n2).map(|i| ((i % 31) as f32) * 0.02).collect();
    // Momentum is uninitialized on first step — pass zeros, but the
    // kernel's first_run=true means it won't read them.
    let buf_init1: Vec<f32> = vec![0.0; n1];
    let buf_init2: Vec<f32> = vec![0.0; n2];

    let p1 = DeviceBuffer::from_slice(&ctx, &p1_host).unwrap();
    let p2 = DeviceBuffer::from_slice(&ctx, &p2_host).unwrap();
    let g1 = DeviceBuffer::from_slice(&ctx, &g1_host).unwrap();
    let g2 = DeviceBuffer::from_slice(&ctx, &g2_host).unwrap();
    let m1 = DeviceBuffer::from_slice(&ctx, &buf_init1).unwrap();
    let m2 = DeviceBuffer::from_slice(&ctx, &buf_init2).unwrap();

    let cfg = SgdConfig {
        lr: 1e-2,
        momentum: 0.9,
        dampening: 0.0,
        weight_decay: 1e-4,
        nesterov: true,
        weight_decay_after_momentum: false,
        grad_scale: 1.0,
    };

    let plan = SgdStepPlan::<f32>::new(cfg);
    plan.step(
        &TensorList::new(&[&p1, &p2]).unwrap(),
        &TensorList::new(&[&g1, &g2]).unwrap(),
        &TensorList::new(&[&m1, &m2]).unwrap(),
        /*first_step=*/ true,
        &stream,
    )
    .expect("SGD step");
    stream.synchronize().unwrap();

    let mut p1_ref = p1_host.clone();
    let mut p2_ref = p2_host.clone();
    let mut m1_ref = buf_init1.clone();
    let mut m2_ref = buf_init2.clone();
    sgd_step_cpu(&mut p1_ref, &g1_host, &mut m1_ref, &cfg, true);
    sgd_step_cpu(&mut p2_ref, &g2_host, &mut m2_ref, &cfg, true);

    let mut p1_got = vec![0.0f32; n1];
    let mut p2_got = vec![0.0f32; n2];
    let mut m1_got = vec![0.0f32; n1];
    let mut m2_got = vec![0.0f32; n2];
    p1.copy_to_host(&mut p1_got).unwrap();
    p2.copy_to_host(&mut p2_got).unwrap();
    m1.copy_to_host(&mut m1_got).unwrap();
    m2.copy_to_host(&mut m2_got).unwrap();

    let tol = 1e-5f32;
    for (i, (g, e)) in p1_got.iter().zip(&p1_ref).enumerate() {
        assert!(
            (g - e).abs() <= tol * (1.0 + e.abs()),
            "p1[{i}] = {g}, want {e}"
        );
    }
    for (i, (g, e)) in p2_got.iter().zip(&p2_ref).enumerate() {
        assert!(
            (g - e).abs() <= tol * (1.0 + e.abs()),
            "p2[{i}] = {g}, want {e}"
        );
    }
    for (i, (g, e)) in m1_got.iter().zip(&m1_ref).enumerate() {
        assert!(
            (g - e).abs() <= tol * (1.0 + e.abs()),
            "m1[{i}] = {g}, want {e}"
        );
    }
    for (i, (g, e)) in m2_got.iter().zip(&m2_ref).enumerate() {
        assert!(
            (g - e).abs() <= tol * (1.0 + e.abs()),
            "m2[{i}] = {g}, want {e}"
        );
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn sgd_f32_vanilla_no_momentum() {
    // momentum=0 disables the velocity buffer path — verify the
    // zero-momentum branch.
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = baracuda_driver::Stream::new(&ctx).unwrap();

    let n = 2048usize;
    let p_host: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
    let g_host: Vec<f32> = (0..n).map(|i| ((i % 19) as f32) - 9.0).collect();

    let p = DeviceBuffer::from_slice(&ctx, &p_host).unwrap();
    let g = DeviceBuffer::from_slice(&ctx, &g_host).unwrap();
    // momentum buffer required by the FFI even when momentum=0 (the
    // pointer must be non-null; we pass a small zero buffer).
    let m_init = vec![0.0f32; n];
    let m = DeviceBuffer::from_slice(&ctx, &m_init).unwrap();

    let cfg = SgdConfig {
        lr: 1e-3,
        momentum: 0.0,
        dampening: 0.0,
        weight_decay: 0.0,
        nesterov: false,
        weight_decay_after_momentum: false,
        grad_scale: 1.0,
    };
    SgdStepPlan::<f32>::new(cfg)
        .step(
            &TensorList::new(&[&p]).unwrap(),
            &TensorList::new(&[&g]).unwrap(),
            &TensorList::new(&[&m]).unwrap(),
            false,
            &stream,
        )
        .expect("Vanilla SGD step");
    stream.synchronize().unwrap();

    let mut p_got = vec![0.0f32; n];
    p.copy_to_host(&mut p_got).unwrap();

    for (i, (g_val, p_val)) in g_host.iter().zip(p_host.iter()).enumerate() {
        let expected = p_val - cfg.lr * g_val;
        assert!(
            (p_got[i] - expected).abs() <= 1e-6 * (1.0 + expected.abs()),
            "vanilla SGD p[{i}] = {}, want {expected}",
            p_got[i]
        );
    }
}
