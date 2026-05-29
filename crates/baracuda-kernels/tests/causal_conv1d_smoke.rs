//! Real-GPU smoke test for Phase 50 [`CausalConv1dPlan`] (FW).
//!
//! Validates the bespoke causal-conv1d FW kernel against a CPU
//! reference implementation that follows the same depthwise-causal
//! cross-correlation contract.
//!
//! All tests are `#[ignore]` by default — requires a real CUDA device
//! and a build with `--features mamba`.

#![cfg(feature = "mamba")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, CausalConv1dArgs, CausalConv1dDescriptor, CausalConv1dPlan, ElementKind,
    PlanPreference, TensorMut, TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// CPU reference for depthwise causal conv1d.
fn cpu_ref_causal_conv1d_f32(
    x: &[f32], weight: &[f32], bias: Option<&[f32]>,
    b: usize, c: usize, l: usize, w: usize, use_silu: bool,
) -> Vec<f32> {
    let mut y = vec![0.0f32; b * c * l];
    for bi in 0..b {
        for ci in 0..c {
            for t in 0..l {
                let mut acc = 0.0f32;
                for k in 0..w {
                    let xi = t as isize - (w as isize - 1 - k as isize);
                    if xi >= 0 {
                        acc += weight[ci * w + k] * x[bi * c * l + ci * l + xi as usize];
                    }
                }
                if let Some(b_arr) = bias {
                    acc += b_arr[ci];
                }
                if use_silu {
                    acc = acc / (1.0 + (-acc).exp());
                }
                y[bi * c * l + ci * l + t] = acc;
            }
        }
    }
    y
}

fn check_close(a: &[f32], b: &[f32], tol: f32, tag: &str) {
    assert_eq!(a.len(), b.len(), "{}: length mismatch", tag);
    for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (av - bv).abs();
        let scale = av.abs().max(bv.abs()).max(1e-3);
        if diff > tol * scale {
            panic!(
                "{}: mismatch at index {} — got {}, expected {}, diff {} (scale {})",
                tag, i, av, bv, diff, scale
            );
        }
    }
}

#[test]
#[ignore]
fn causal_conv1d_f32_width4_silu_matches_cpu_ref() {
    let (ctx, stream) = setup();

    let b = 2;
    let c = 4;
    let l = 16;
    let w = 4;
    let use_silu = true;

    let x_host: Vec<f32> = (0..b * c * l).map(|i| (i as f32) * 0.05 - 0.5).collect();
    let weight_host: Vec<f32> = (0..c * w).map(|i| ((i as f32) * 0.13).sin() * 0.5).collect();
    let bias_host: Vec<f32> = (0..c).map(|i| (i as f32) * 0.1).collect();

    let x_dev = DeviceBuffer::from_slice(&ctx, &x_host).expect("x alloc");
    let weight_dev = DeviceBuffer::from_slice(&ctx, &weight_host).expect("w alloc");
    let bias_dev = DeviceBuffer::from_slice(&ctx, &bias_host).expect("b alloc");
    let mut y_dev = DeviceBuffer::<f32>::zeros(&ctx, b * c * l).expect("y alloc");

    let desc = CausalConv1dDescriptor {
        batch_size: b as i32,
        channels: c as i32,
        seq_len: l as i32,
        width: w as i32,
        use_silu,
        element: ElementKind::F32,
    };

    let plan = CausalConv1dPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");

    let shape_x: [i32; 3] = [b as i32, c as i32, l as i32];
    let shape_w: [i32; 2] = [c as i32, w as i32];
    let shape_b: [i32; 1] = [c as i32];

    plan.run(
        &stream,
        Workspace::None,
        CausalConv1dArgs {
            x: TensorRef { data: x_dev.as_slice(), shape: shape_x, stride: contiguous_stride(shape_x) },
            weight: TensorRef { data: weight_dev.as_slice(), shape: shape_w, stride: contiguous_stride(shape_w) },
            bias: Some(TensorRef { data: bias_dev.as_slice(), shape: shape_b, stride: contiguous_stride(shape_b) }),
            y: TensorMut { data: y_dev.as_slice_mut(), shape: shape_x, stride: contiguous_stride(shape_x) },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut y_got = vec![0.0f32; b * c * l];
    y_dev.copy_to_host(&mut y_got).expect("download");
    let y_expected = cpu_ref_causal_conv1d_f32(
        &x_host, &weight_host, Some(&bias_host), b, c, l, w, use_silu,
    );
    check_close(&y_got, &y_expected, 1e-5, "causal_conv1d_f32_width4_silu");
}

#[test]
#[ignore]
fn causal_conv1d_f32_width2_no_silu_no_bias() {
    let (ctx, stream) = setup();
    let b = 1;
    let c = 3;
    let l = 8;
    let w = 2;

    let x_host: Vec<f32> = (0..b * c * l).map(|i| i as f32 * 0.1).collect();
    let weight_host: Vec<f32> = (0..c * w).map(|i| (i as f32) * 0.2 + 0.1).collect();

    let x_dev = DeviceBuffer::from_slice(&ctx, &x_host).expect("x");
    let weight_dev = DeviceBuffer::from_slice(&ctx, &weight_host).expect("w");
    let mut y_dev = DeviceBuffer::<f32>::zeros(&ctx, b * c * l).expect("y");

    let desc = CausalConv1dDescriptor {
        batch_size: b as i32, channels: c as i32, seq_len: l as i32,
        width: w as i32, use_silu: false, element: ElementKind::F32,
    };
    let plan = CausalConv1dPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");

    let shape_x: [i32; 3] = [b as i32, c as i32, l as i32];
    let shape_w: [i32; 2] = [c as i32, w as i32];
    plan.run(&stream, Workspace::None, CausalConv1dArgs {
        x: TensorRef { data: x_dev.as_slice(), shape: shape_x, stride: contiguous_stride(shape_x) },
        weight: TensorRef { data: weight_dev.as_slice(), shape: shape_w, stride: contiguous_stride(shape_w) },
        bias: None,
        y: TensorMut { data: y_dev.as_slice_mut(), shape: shape_x, stride: contiguous_stride(shape_x) },
    }).expect("run");
    stream.synchronize().expect("sync");

    let mut y_got = vec![0.0f32; b * c * l];
    y_dev.copy_to_host(&mut y_got).expect("download");
    let y_expected = cpu_ref_causal_conv1d_f32(
        &x_host, &weight_host, None, b, c, l, w, false,
    );
    check_close(&y_got, &y_expected, 1e-5, "causal_conv1d_w2_no_silu");
}

#[test]
#[ignore]
fn causal_conv1d_width_5_rejected() {
    let (_ctx, stream) = setup();
    let desc = CausalConv1dDescriptor {
        batch_size: 1, channels: 1, seq_len: 4, width: 5,
        use_silu: false, element: ElementKind::F32,
    };
    let res = CausalConv1dPlan::<f32>::select(&stream, &desc, PlanPreference::default());
    assert!(res.is_err(), "width=5 should be rejected");
}
