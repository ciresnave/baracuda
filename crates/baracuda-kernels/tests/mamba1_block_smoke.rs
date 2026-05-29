//! End-to-end smoke for a Phase 50b Mamba-1 block (FW only).
//!
//! Composes the baracuda primitives that a Mamba-1 block needs:
//! `causal_conv1d` (SiLU) over the projected input, then
//! `selective_scan` with optional `D` skip and `z` gating, then a
//! residual add against the original input. Sanity-checks finiteness
//! and that the output is non-trivial (not collapsed to zero,
//! magnitude reasonable).
//!
//! This is the inference-time block shape from Mamba-7B /
//! Falcon-Mamba / Codestral-Mamba (modulo the input/output linear
//! projections that the caller orchestrates via baracuda's GEMM).

#![cfg(feature = "mamba")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, CausalConv1dArgs, CausalConv1dDescriptor, CausalConv1dPlan,
    ElementKind, PlanPreference, SelectiveScanArgs, SelectiveScanDescriptor,
    SelectiveScanPlan, TensorMut, TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn mamba1_block_tiny_end_to_end() {
    let (ctx, stream) = setup();

    let bsz: usize = 1;
    let l: usize = 8;
    let dim: usize = 8;       // channel dim (Mamba "dim")
    let dstate: usize = 4;    // state dim (Mamba "dstate")
    let w_conv: usize = 4;

    // Step 0: caller-supplied input x [B, L, D] (stand-in for an
    // input-projection from a hidden state).
    let x_host: Vec<f32> = (0..bsz * l * dim)
        .map(|i| ((i as f32) * 0.013).sin() * 0.5)
        .collect();

    // ---- Step 1: causal_conv1d. Conv expects [B, C, L]; permute
    // [B, L, D] -> [B, C=D, L] on the host (a "channels-last to
    // channels-second" transpose).
    let mut x_for_conv = vec![0.0f32; bsz * dim * l];
    for bi in 0..bsz {
        for di in 0..dim {
            for t in 0..l {
                let src = bi * l * dim + t * dim + di;
                let dst = bi * dim * l + di * l + t;
                x_for_conv[dst] = x_host[src];
            }
        }
    }
    let w_conv_host: Vec<f32> = (0..dim * w_conv)
        .map(|i| ((i as f32) * 0.07).cos() * 0.3).collect();
    let b_conv_host: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.02).collect();

    let x_conv_dev = DeviceBuffer::from_slice(&ctx, &x_for_conv).expect("x_conv");
    let w_conv_dev = DeviceBuffer::from_slice(&ctx, &w_conv_host).expect("w_conv");
    let b_conv_dev = DeviceBuffer::from_slice(&ctx, &b_conv_host).expect("b_conv");
    let mut u_dev = DeviceBuffer::<f32>::zeros(&ctx, bsz * dim * l).expect("u");

    let conv_desc = CausalConv1dDescriptor {
        batch_size: bsz as i32, channels: dim as i32, seq_len: l as i32,
        width: w_conv as i32, use_silu: true, element: ElementKind::F32,
    };
    let conv_plan = CausalConv1dPlan::<f32>::select(&stream, &conv_desc, PlanPreference::default())
        .expect("conv select");
    let s_x_conv: [i32; 3] = [bsz as i32, dim as i32, l as i32];
    let s_w_conv: [i32; 2] = [dim as i32, w_conv as i32];
    let s_b_conv: [i32; 1] = [dim as i32];
    conv_plan.run(&stream, Workspace::None, CausalConv1dArgs {
        x: TensorRef { data: x_conv_dev.as_slice(), shape: s_x_conv, stride: contiguous_stride(s_x_conv) },
        weight: TensorRef { data: w_conv_dev.as_slice(), shape: s_w_conv, stride: contiguous_stride(s_w_conv) },
        bias: Some(TensorRef { data: b_conv_dev.as_slice(), shape: s_b_conv, stride: contiguous_stride(s_b_conv) }),
        y: TensorMut { data: u_dev.as_slice_mut(), shape: s_x_conv, stride: contiguous_stride(s_x_conv) },
    }).expect("conv run");
    stream.synchronize().expect("sync");

    // Permute [B, C, L] -> [B, L, D] for the SSM.
    let mut u_host = vec![0.0f32; bsz * dim * l];
    u_dev.copy_to_host(&mut u_host).expect("u dl");
    let mut u_ssm_host = vec![0.0f32; bsz * l * dim];
    for bi in 0..bsz {
        for di in 0..dim {
            for t in 0..l {
                let src = bi * dim * l + di * l + t;
                let dst = bi * l * dim + t * dim + di;
                u_ssm_host[dst] = u_host[src];
            }
        }
    }

    // ---- Step 2: selective_scan with full options (delta_bias +
    // softplus + D skip + z gating). These are all-paths Mamba-1.
    let delta_host: Vec<f32> = (0..bsz * l * dim).map(|i| -1.0 + (i as f32) * 0.01).collect();
    let a_host: Vec<f32> = (0..dim * dstate).map(|i| -0.5 - (i as f32) * 0.02).collect();
    let b_host: Vec<f32> = (0..bsz * l * dstate).map(|i| ((i as f32) * 0.09).sin() * 0.3).collect();
    let c_host: Vec<f32> = (0..bsz * l * dstate).map(|i| ((i as f32) * 0.11).cos() * 0.3).collect();
    let d_host: Vec<f32> = (0..dim).map(|i| 0.1 + (i as f32) * 0.02).collect();
    let z_host: Vec<f32> = (0..bsz * l * dim).map(|i| ((i as f32) * 0.05).cos() * 0.5).collect();
    let db_host: Vec<f32> = (0..dim).map(|i| -0.5 + (i as f32) * 0.1).collect();

    let u_ssm_dev = DeviceBuffer::from_slice(&ctx, &u_ssm_host).expect("u_ssm");
    let delta_dev = DeviceBuffer::from_slice(&ctx, &delta_host).expect("delta");
    let a_dev = DeviceBuffer::from_slice(&ctx, &a_host).expect("a");
    let b_dev = DeviceBuffer::from_slice(&ctx, &b_host).expect("b");
    let c_dev = DeviceBuffer::from_slice(&ctx, &c_host).expect("c");
    let d_dev = DeviceBuffer::from_slice(&ctx, &d_host).expect("d");
    let z_dev = DeviceBuffer::from_slice(&ctx, &z_host).expect("z");
    let db_dev = DeviceBuffer::from_slice(&ctx, &db_host).expect("db");
    let mut y_ssm_dev = DeviceBuffer::<f32>::zeros(&ctx, bsz * l * dim).expect("y_ssm");

    let ssm_desc = SelectiveScanDescriptor {
        batch_size: bsz as i32, seq_len: l as i32, dim: dim as i32, dstate: dstate as i32,
        delta_softplus: true, element: ElementKind::F32,
    };
    let ssm_plan = SelectiveScanPlan::<f32>::select(&stream, &ssm_desc, PlanPreference::default())
        .expect("ssm select");
    let s_ud: [i32; 3] = [bsz as i32, l as i32, dim as i32];
    let s_a: [i32; 2] = [dim as i32, dstate as i32];
    let s_bc: [i32; 3] = [bsz as i32, l as i32, dstate as i32];
    let s_d: [i32; 1] = [dim as i32];
    ssm_plan.run(&stream, Workspace::None, SelectiveScanArgs {
        u: TensorRef { data: u_ssm_dev.as_slice(), shape: s_ud, stride: contiguous_stride(s_ud) },
        delta: TensorRef { data: delta_dev.as_slice(), shape: s_ud, stride: contiguous_stride(s_ud) },
        a: TensorRef { data: a_dev.as_slice(), shape: s_a, stride: contiguous_stride(s_a) },
        b: TensorRef { data: b_dev.as_slice(), shape: s_bc, stride: contiguous_stride(s_bc) },
        c: TensorRef { data: c_dev.as_slice(), shape: s_bc, stride: contiguous_stride(s_bc) },
        d_skip: Some(TensorRef { data: d_dev.as_slice(), shape: s_d, stride: contiguous_stride(s_d) }),
        z: Some(TensorRef { data: z_dev.as_slice(), shape: s_ud, stride: contiguous_stride(s_ud) }),
        delta_bias: Some(TensorRef { data: db_dev.as_slice(), shape: s_d, stride: contiguous_stride(s_d) }),
        y: TensorMut { data: y_ssm_dev.as_slice_mut(), shape: s_ud, stride: contiguous_stride(s_ud) },
        last_state: None,
    }).expect("ssm run");
    stream.synchronize().expect("sync");

    // ---- Step 3: residual add against original x.
    let mut y_ssm = vec![0.0f32; bsz * l * dim];
    y_ssm_dev.copy_to_host(&mut y_ssm).expect("y dl");
    let out: Vec<f32> = y_ssm.iter().zip(x_host.iter()).map(|(a, b)| a + b).collect();

    let mut max_abs = 0.0f32;
    for v in &out {
        assert!(v.is_finite(), "non-finite Mamba-1 block output: {}", v);
        max_abs = max_abs.max(v.abs());
    }
    assert!(max_abs < 100.0, "block output magnitude too large: {}", max_abs);
    let mean_abs: f32 = out.iter().map(|v| v.abs()).sum::<f32>() / (out.len() as f32);
    assert!(mean_abs > 1e-4,
        "block output mean magnitude suspiciously small: {}", mean_abs);
}
