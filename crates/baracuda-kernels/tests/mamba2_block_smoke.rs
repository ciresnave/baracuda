//! End-to-end smoke for the Phase 50 Mamba-2 block (FW only).
//!
//! Assembles a tiny Mamba-2 block: causal_conv1d (SiLU) + SSD chunk-
//! scan + residual add. Sanity-checks finiteness + non-collapse.

#![cfg(feature = "mamba")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, CausalConv1dArgs, CausalConv1dDescriptor, CausalConv1dPlan,
    ElementKind, PlanPreference, SsdChunkScanArgs, SsdChunkScanDescriptor,
    SsdChunkScanPlan, TensorMut, TensorRef, Workspace,
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
fn mamba2_block_tiny_end_to_end() {
    let (ctx, stream) = setup();

    let bsz: usize = 1;
    let l: usize = 8;
    let h: usize = 2;
    let d: usize = 4;
    let n: usize = 4;
    let chunk: i32 = 4;
    let w_conv: usize = 4;
    let channels = h * d;

    let x_host: Vec<f32> = (0..bsz * l * h * d)
        .map(|i| ((i as f32) * 0.013).sin() * 0.5)
        .collect();

    // Reshape [B, L, H, D] → [B, C=H*D, L] for the conv (host-side
    // permute — a "linear projection" stand-in for the smoke).
    let mut x_for_conv = vec![0.0f32; bsz * channels * l];
    for bi in 0..bsz {
        for hi in 0..h {
            for di in 0..d {
                let ci = hi * d + di;
                for t in 0..l {
                    let src = bi * l * h * d + t * h * d + hi * d + di;
                    let dst = bi * channels * l + ci * l + t;
                    x_for_conv[dst] = x_host[src];
                }
            }
        }
    }
    let w_conv_host: Vec<f32> = (0..channels * w_conv).map(|i| ((i as f32) * 0.07).cos() * 0.3).collect();
    let b_conv_host: Vec<f32> = (0..channels).map(|i| (i as f32) * 0.02).collect();
    let dt_host: Vec<f32> = (0..bsz * l * h).map(|i| 0.05 + (i as f32) * 0.005).collect();
    let a_host: Vec<f32> = (0..h).map(|i| -0.5 - (i as f32) * 0.1).collect();
    let b_proj_host: Vec<f32> = (0..bsz * l * h * n).map(|i| ((i as f32) * 0.09).sin() * 0.3).collect();
    let c_proj_host: Vec<f32> = (0..bsz * l * h * n).map(|i| ((i as f32) * 0.11).cos() * 0.3).collect();

    let x_conv_dev = DeviceBuffer::from_slice(&ctx, &x_for_conv).expect("x_conv");
    let w_conv_dev = DeviceBuffer::from_slice(&ctx, &w_conv_host).expect("w_conv");
    let b_conv_dev = DeviceBuffer::from_slice(&ctx, &b_conv_host).expect("b_conv");
    let mut u_dev = DeviceBuffer::<f32>::zeros(&ctx, bsz * channels * l).expect("u");

    let conv_desc = CausalConv1dDescriptor {
        batch_size: bsz as i32, channels: channels as i32, seq_len: l as i32,
        width: w_conv as i32, use_silu: true, element: ElementKind::F32,
    };
    let conv_plan = CausalConv1dPlan::<f32>::select(&stream, &conv_desc, PlanPreference::default())
        .expect("conv select");
    let shape_xconv: [i32; 3] = [bsz as i32, channels as i32, l as i32];
    let shape_wconv: [i32; 2] = [channels as i32, w_conv as i32];
    let shape_bconv: [i32; 1] = [channels as i32];
    conv_plan.run(&stream, Workspace::None, CausalConv1dArgs {
        x: TensorRef { data: x_conv_dev.as_slice(), shape: shape_xconv, stride: contiguous_stride(shape_xconv) },
        weight: TensorRef { data: w_conv_dev.as_slice(), shape: shape_wconv, stride: contiguous_stride(shape_wconv) },
        bias: Some(TensorRef { data: b_conv_dev.as_slice(), shape: shape_bconv, stride: contiguous_stride(shape_bconv) }),
        y: TensorMut { data: u_dev.as_slice_mut(), shape: shape_xconv, stride: contiguous_stride(shape_xconv) },
    }).expect("conv run");
    stream.synchronize().expect("sync");

    let mut u_host = vec![0.0f32; bsz * channels * l];
    u_dev.copy_to_host(&mut u_host).expect("u dl");
    let mut u_ssd_host = vec![0.0f32; bsz * l * h * d];
    for bi in 0..bsz {
        for hi in 0..h {
            for di in 0..d {
                let ci = hi * d + di;
                for t in 0..l {
                    let src = bi * channels * l + ci * l + t;
                    let dst = bi * l * h * d + t * h * d + hi * d + di;
                    u_ssd_host[dst] = u_host[src];
                }
            }
        }
    }
    let u_ssd_dev = DeviceBuffer::from_slice(&ctx, &u_ssd_host).expect("u_ssd");

    let dt_dev = DeviceBuffer::from_slice(&ctx, &dt_host).expect("dt");
    let a_dev = DeviceBuffer::from_slice(&ctx, &a_host).expect("a");
    let b_proj_dev = DeviceBuffer::from_slice(&ctx, &b_proj_host).expect("b_proj");
    let c_proj_dev = DeviceBuffer::from_slice(&ctx, &c_proj_host).expect("c_proj");
    let mut y_ssm_dev = DeviceBuffer::<f32>::zeros(&ctx, bsz * l * h * d).expect("y_ssm");

    let ssd_desc = SsdChunkScanDescriptor {
        batch_size: bsz as i32, seq_len: l as i32, num_heads: h as i32,
        head_dim: d as i32, state_dim: n as i32, chunk_size: chunk,
        element: ElementKind::F32,
    };
    let ssd_plan = SsdChunkScanPlan::<f32>::select(&stream, &ssd_desc, PlanPreference::default())
        .expect("ssd select");

    let shape_x: [i32; 4] = [bsz as i32, l as i32, h as i32, d as i32];
    let shape_dt: [i32; 3] = [bsz as i32, l as i32, h as i32];
    let shape_a: [i32; 1] = [h as i32];
    let shape_bn: [i32; 4] = [bsz as i32, l as i32, h as i32, n as i32];
    ssd_plan.run(&stream, Workspace::None, SsdChunkScanArgs {
        x: TensorRef { data: u_ssd_dev.as_slice(), shape: shape_x, stride: contiguous_stride(shape_x) },
        dt: TensorRef { data: dt_dev.as_slice(), shape: shape_dt, stride: contiguous_stride(shape_dt) },
        a: TensorRef { data: a_dev.as_slice(), shape: shape_a, stride: contiguous_stride(shape_a) },
        b: TensorRef { data: b_proj_dev.as_slice(), shape: shape_bn, stride: contiguous_stride(shape_bn) },
        c: TensorRef { data: c_proj_dev.as_slice(), shape: shape_bn, stride: contiguous_stride(shape_bn) },
        y: TensorMut { data: y_ssm_dev.as_slice_mut(), shape: shape_x, stride: contiguous_stride(shape_x) },
    }).expect("ssd run");
    stream.synchronize().expect("sync");

    let mut y_ssm = vec![0.0f32; bsz * l * h * d];
    y_ssm_dev.copy_to_host(&mut y_ssm).expect("y_ssm dl");
    let out: Vec<f32> = y_ssm.iter().zip(x_host.iter()).map(|(a, b)| a + b).collect();

    let mut max_abs = 0.0f32;
    for v in &out {
        assert!(v.is_finite(), "non-finite block output: {}", v);
        max_abs = max_abs.max(v.abs());
    }
    assert!(max_abs < 100.0, "block output magnitude too large: {}", max_abs);
    let mean_abs: f32 = out.iter().map(|v| v.abs()).sum::<f32>() / (out.len() as f32);
    assert!(mean_abs > 1e-4, "block output mean magnitude suspiciously small: {}", mean_abs);
}
