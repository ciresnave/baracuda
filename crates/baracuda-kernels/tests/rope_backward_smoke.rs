//! Real-GPU smoke test for `RopeBackwardPlan` (Phase 6.1).
//!
//! BW rotates by `-θ`:
//!   dx[2i]   = dy[2i]   · cos(θ) + dy[2i+1] · sin(θ)
//!   dx[2i+1] = dy[2i+1] · cos(θ) - dy[2i]   · sin(θ)
//!
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, RopeBackwardArgs, RopeBackwardDescriptor,
    RopeBackwardPlan, TensorMut, TensorRef, Workspace, ROPE_DEFAULT_BASE,
};
use half::{bf16, f16};

const F16_EPS: f32 = 9.77e-4;
const BF16_EPS: f32 = 7.81e-3;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn host_rope_backward_f32(
    batch: usize,
    heads: usize,
    seq: usize,
    head_dim: usize,
    base: f32,
    dy: &[f32],
    positions: Option<&[i64]>,
) -> Vec<f32> {
    assert_eq!(head_dim % 2, 0);
    let total = batch * heads * seq * head_dim;
    let mut dx = vec![0f32; total];
    let inv_d = 1.0 / head_dim as f32;
    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq {
                let pos = positions.map(|p| p[s]).unwrap_or(s as i64) as f32;
                for pair in 0..(head_dim / 2) {
                    let de = pair * 2;
                    let d_o = de + 1;
                    let exponent = -((de as f32) * inv_d);
                    let freq = base.powf(exponent);
                    let theta = pos * freq;
                    let c = theta.cos();
                    let si = theta.sin();
                    let off = ((b * heads + h) * seq + s) * head_dim;
                    let dy_e = dy[off + de];
                    let dy_o = dy[off + d_o];
                    dx[off + de] = dy_e * c + dy_o * si;
                    dx[off + d_o] = dy_o * c - dy_e * si;
                }
            }
        }
    }
    dx
}

#[test]
#[ignore]
fn rope_backward_f32_default_positions() {
    let (ctx, stream) = setup();
    let batch = 2i32;
    let heads = 4i32;
    let seq = 8i32;
    let head_dim = 16i32;
    let numel = (batch * heads * seq * head_dim) as usize;
    let host_dy: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.09 - 0.3).cos() * 0.8)
        .collect();
    let base = ROPE_DEFAULT_BASE;
    let expected = host_rope_backward_f32(
        batch as usize,
        heads as usize,
        seq as usize,
        head_dim as usize,
        base,
        &host_dy,
        None,
    );

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");

    let desc = RopeBackwardDescriptor {
        batch_size: batch,
        num_heads: heads,
        seq_len: seq,
        head_dim,
        base,
        element: ElementKind::F32,
    };
    let plan =
        RopeBackwardPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let shape = [batch, heads, seq, head_dim];
    plan.run(
        &stream,
        Workspace::None,
        RopeBackwardArgs {
            dy: TensorRef {
                data: dev_dy.as_slice(),
                shape,
                stride: contiguous_stride(shape),
            },
            positions: None,
            dx: TensorMut {
                data: dev_dx.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_dx.copy_to_host(&mut got).expect("dl");

    // 16 eps relative — kernel uses __sinf / __cosf fast intrinsics,
    // which sacrifice 1-2 ULPs vs the libm reference. Plus there are
    // two multiplies and an add per pair, so 16 eps is the
    // round-after-each-op bound.
    let tol = 16.0 * f32::EPSILON;
    for i in 0..numel {
        let t = (expected[i].abs() * tol).max(tol);
        assert!(
            (got[i] - expected[i]).abs() <= t,
            "f32 rope_bw dx @ {i}: got={} want={}",
            got[i],
            expected[i]
        );
    }
}

#[test]
#[ignore]
fn rope_backward_f64_default_positions() {
    let (ctx, stream) = setup();
    let batch = 2i32;
    let heads = 3i32;
    let seq = 6i32;
    let head_dim = 8i32;
    let numel = (batch * heads * seq * head_dim) as usize;
    let host_dy: Vec<f64> = (0..numel)
        .map(|i| ((i as f64) * 0.13 - 0.5).sin() * 0.7)
        .collect();
    let base = ROPE_DEFAULT_BASE;
    let mut expected = vec![0f64; numel];
    let inv_d = 1.0 / head_dim as f64;
    for b in 0..batch as usize {
        for h in 0..heads as usize {
            for s in 0..seq as usize {
                let pos = s as f64;
                for pair in 0..(head_dim as usize / 2) {
                    let de = pair * 2;
                    let d_o = de + 1;
                    let exponent = -((de as f64) * inv_d);
                    let freq = (base as f64).powf(exponent);
                    let theta = pos * freq;
                    let c = theta.cos();
                    let si = theta.sin();
                    let off =
                        ((b * heads as usize + h) * seq as usize + s) * head_dim as usize;
                    let dy_e = host_dy[off + de];
                    let dy_o = host_dy[off + d_o];
                    expected[off + de] = dy_e * c + dy_o * si;
                    expected[off + d_o] = dy_o * c - dy_e * si;
                }
            }
        }
    }

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");

    let desc = RopeBackwardDescriptor {
        batch_size: batch,
        num_heads: heads,
        seq_len: seq,
        head_dim,
        base,
        element: ElementKind::F64,
    };
    let plan =
        RopeBackwardPlan::<f64>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let shape = [batch, heads, seq, head_dim];
    plan.run(
        &stream,
        Workspace::None,
        RopeBackwardArgs {
            dy: TensorRef {
                data: dev_dy.as_slice(),
                shape,
                stride: contiguous_stride(shape),
            },
            positions: None,
            dx: TensorMut {
                data: dev_dx.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; numel];
    dev_dx.copy_to_host(&mut got).expect("dl");
    let tol = 4.0 * f64::EPSILON;
    for i in 0..numel {
        let t = (expected[i].abs() * tol).max(tol);
        assert!(
            (got[i] - expected[i]).abs() <= t,
            "f64 rope_bw dx @ {i}: got={} want={}",
            got[i],
            expected[i]
        );
    }
}

#[test]
#[ignore]
fn rope_backward_f16_default_positions() {
    let (ctx, stream) = setup();
    let batch = 2i32;
    let heads = 2i32;
    let seq = 8i32;
    let head_dim = 16i32;
    let numel = (batch * heads * seq * head_dim) as usize;
    let host_dy_f32: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.07 - 0.2).cos() * 0.6)
        .collect();
    let base = ROPE_DEFAULT_BASE;
    let expected = host_rope_backward_f32(
        batch as usize,
        heads as usize,
        seq as usize,
        head_dim as usize,
        base,
        &host_dy_f32,
        None,
    );
    let host_dy: Vec<f16> = host_dy_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");

    let desc = RopeBackwardDescriptor {
        batch_size: batch,
        num_heads: heads,
        seq_len: seq,
        head_dim,
        base,
        element: ElementKind::F16,
    };
    let plan =
        RopeBackwardPlan::<f16>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let shape = [batch, heads, seq, head_dim];
    plan.run(
        &stream,
        Workspace::None,
        RopeBackwardArgs {
            dy: TensorRef {
                data: dev_dy.as_slice(),
                shape,
                stride: contiguous_stride(shape),
            },
            positions: None,
            dx: TensorMut {
                data: dev_dx.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![f16::ZERO; numel];
    dev_dx.copy_to_host(&mut got).expect("dl");

    let tol = 8.0 * F16_EPS;
    for i in 0..numel {
        let t = (expected[i].abs() * tol).max(tol);
        let diff = (got[i].to_f32() - expected[i]).abs();
        assert!(
            diff <= t,
            "f16 rope_bw dx @ {i}: diff={diff} want={}",
            expected[i]
        );
    }
}

#[test]
#[ignore]
fn rope_backward_bf16_default_positions() {
    let (ctx, stream) = setup();
    let batch = 2i32;
    let heads = 2i32;
    let seq = 6i32;
    let head_dim = 12i32;
    let numel = (batch * heads * seq * head_dim) as usize;
    let host_dy_f32: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.15 - 0.6).sin() * 0.5)
        .collect();
    let base = ROPE_DEFAULT_BASE;
    let expected = host_rope_backward_f32(
        batch as usize,
        heads as usize,
        seq as usize,
        head_dim as usize,
        base,
        &host_dy_f32,
        None,
    );
    let host_dy: Vec<bf16> = host_dy_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");

    let desc = RopeBackwardDescriptor {
        batch_size: batch,
        num_heads: heads,
        seq_len: seq,
        head_dim,
        base,
        element: ElementKind::Bf16,
    };
    let plan =
        RopeBackwardPlan::<bf16>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let shape = [batch, heads, seq, head_dim];
    plan.run(
        &stream,
        Workspace::None,
        RopeBackwardArgs {
            dy: TensorRef {
                data: dev_dy.as_slice(),
                shape,
                stride: contiguous_stride(shape),
            },
            positions: None,
            dx: TensorMut {
                data: dev_dx.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![bf16::ZERO; numel];
    dev_dx.copy_to_host(&mut got).expect("dl");

    let tol = 8.0 * BF16_EPS;
    for i in 0..numel {
        let t = (expected[i].abs() * tol).max(tol);
        let diff = (got[i].to_f32() - expected[i]).abs();
        assert!(
            diff <= t,
            "bf16 rope_bw dx @ {i}: diff={diff} want={}",
            expected[i]
        );
    }
}
