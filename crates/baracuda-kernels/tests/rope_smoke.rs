//! Real-GPU smoke test for `RopePlan + AttentionKind::Rope` FW.
//!
//! For Q/K of shape `[B, H, S, D]` (D even), rotates pairs (2i, 2i+1)
//! by per-position angles `θ_i = pos · base^(-2i/D)`:
//!     y[2i]   = x[2i]   · cos(θ) - x[2i+1] · sin(θ)
//!     y[2i+1] = x[2i+1] · cos(θ) + x[2i]   · sin(θ)
//!
//! Each test exercises a non-trivial (B, H, S, D) and either the default
//! `positions = [0, 1, ..., S-1]` (None case) or an explicit positions
//! tensor.
//!
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, RopeArgs, RopeDescriptor, RopePlan, TensorMut,
    TensorRef, Workspace, ROPE_DEFAULT_BASE,
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

/// CPU reference, f32 — emits per-cell expected output for shape
/// `[B, H, S, D]`. `positions` may be None to default to `s`.
fn host_rope_f32(
    batch: usize,
    heads: usize,
    seq: usize,
    head_dim: usize,
    base: f32,
    x: &[f32],
    positions: Option<&[i64]>,
) -> Vec<f32> {
    assert_eq!(head_dim % 2, 0, "head_dim must be even");
    let total = batch * heads * seq * head_dim;
    let mut y = vec![0f32; total];
    let inv_d = 1.0 / (head_dim as f32);
    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq {
                let pos = positions.map(|p| p[s]).unwrap_or(s as i64) as f32;
                for pair in 0..(head_dim / 2) {
                    let d_even = pair * 2;
                    let d_odd = d_even + 1;
                    let exponent = -((d_even as f32) * inv_d);
                    let freq = base.powf(exponent);
                    let theta = pos * freq;
                    let c = theta.cos();
                    let si = theta.sin();
                    let base_off = ((b * heads + h) * seq + s) * head_dim;
                    let x_e = x[base_off + d_even];
                    let x_o = x[base_off + d_odd];
                    y[base_off + d_even] = x_e * c - x_o * si;
                    y[base_off + d_odd] = x_o * c + x_e * si;
                }
            }
        }
    }
    y
}

#[test]
#[ignore]
fn rope_f32_default_positions() {
    let (ctx, stream) = setup();
    let batch = 2i32;
    let heads = 4i32;
    let seq = 8i32;
    let head_dim = 16i32;
    let numel = (batch * heads * seq * head_dim) as usize;
    let host_x: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.13 - 0.7).sin() * 1.2)
        .collect();
    let base = ROPE_DEFAULT_BASE;
    let expected = host_rope_f32(
        batch as usize,
        heads as usize,
        seq as usize,
        head_dim as usize,
        base,
        &host_x,
        None,
    );

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let desc = RopeDescriptor {
        batch_size: batch,
        num_heads: heads,
        seq_len: seq,
        head_dim,
        base,
        element: ElementKind::F32,
    };
    let plan = RopePlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let shape = [batch, heads, seq, head_dim];
    plan.run(
        &stream,
        Workspace::None,
        RopeArgs {
            x: TensorRef {
                data: dev_x.as_slice(),
                shape,
                stride: contiguous_stride(shape),
            },
            positions: None,
            y: TensorMut {
                data: dev_y.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("dl");

    // 16 eps relative — kernel uses __sinf / __cosf fast intrinsics,
    // which sacrifice 1-2 ULPs vs the libm reference.
    let tol = 16.0 * f32::EPSILON;
    for i in 0..numel {
        let t = (expected[i].abs() * tol).max(tol);
        assert!(
            (got[i] - expected[i]).abs() <= t,
            "f32 rope y @ {i}: got={} want={}",
            got[i],
            expected[i]
        );
    }
}

#[test]
#[ignore]
fn rope_f64_explicit_positions() {
    let (ctx, stream) = setup();
    let batch = 2i32;
    let heads = 3i32;
    let seq = 6i32;
    let head_dim = 8i32;
    let numel = (batch * heads * seq * head_dim) as usize;
    let host_x: Vec<f64> = (0..numel)
        .map(|i| ((i as f64) * 0.21 - 1.1).cos() * 0.9)
        .collect();
    // shifted positions — start at 5 to test that the kernel actually
    // reads positions[s] rather than defaulting to s.
    let host_positions: Vec<i64> = (0..seq as usize).map(|s| (s as i64) + 5).collect();
    let base = ROPE_DEFAULT_BASE;
    // CPU ref in f64
    let mut expected = vec![0f64; numel];
    let inv_d = 1.0 / head_dim as f64;
    for b in 0..batch as usize {
        for h in 0..heads as usize {
            for s in 0..seq as usize {
                let pos = host_positions[s] as f64;
                for pair in 0..(head_dim as usize / 2) {
                    let d_even = pair * 2;
                    let d_odd = d_even + 1;
                    let exponent = -((d_even as f64) * inv_d);
                    let freq = (base as f64).powf(exponent);
                    let theta = pos * freq;
                    let c = theta.cos();
                    let si = theta.sin();
                    let off =
                        ((b * heads as usize + h) * seq as usize + s) * head_dim as usize;
                    let x_e = host_x[off + d_even];
                    let x_o = host_x[off + d_odd];
                    expected[off + d_even] = x_e * c - x_o * si;
                    expected[off + d_odd] = x_o * c + x_e * si;
                }
            }
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_pos = DeviceBuffer::from_slice(&ctx, &host_positions).expect("up pos");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let desc = RopeDescriptor {
        batch_size: batch,
        num_heads: heads,
        seq_len: seq,
        head_dim,
        base,
        element: ElementKind::F64,
    };
    let plan = RopePlan::<f64>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let shape = [batch, heads, seq, head_dim];
    plan.run(
        &stream,
        Workspace::None,
        RopeArgs {
            x: TensorRef {
                data: dev_x.as_slice(),
                shape,
                stride: contiguous_stride(shape),
            },
            positions: Some(TensorRef {
                data: dev_pos.as_slice(),
                shape: [seq],
                stride: [1],
            }),
            y: TensorMut {
                data: dev_y.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; numel];
    dev_y.copy_to_host(&mut got).expect("dl");

    let tol = 4.0 * f64::EPSILON;
    for i in 0..numel {
        let t = (expected[i].abs() * tol).max(tol);
        assert!(
            (got[i] - expected[i]).abs() <= t,
            "f64 rope y @ {i}: got={} want={}",
            got[i],
            expected[i]
        );
    }
}

#[test]
#[ignore]
fn rope_f16_default_positions() {
    let (ctx, stream) = setup();
    let batch = 2i32;
    let heads = 2i32;
    let seq = 8i32;
    let head_dim = 16i32;
    let numel = (batch * heads * seq * head_dim) as usize;
    let host_x_f32: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.11 - 0.4).sin() * 0.8)
        .collect();
    let base = ROPE_DEFAULT_BASE;
    let expected = host_rope_f32(
        batch as usize,
        heads as usize,
        seq as usize,
        head_dim as usize,
        base,
        &host_x_f32,
        None,
    );

    let host_x: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let desc = RopeDescriptor {
        batch_size: batch,
        num_heads: heads,
        seq_len: seq,
        head_dim,
        base,
        element: ElementKind::F16,
    };
    let plan = RopePlan::<f16>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let shape = [batch, heads, seq, head_dim];
    plan.run(
        &stream,
        Workspace::None,
        RopeArgs {
            x: TensorRef {
                data: dev_x.as_slice(),
                shape,
                stride: contiguous_stride(shape),
            },
            positions: None,
            y: TensorMut {
                data: dev_y.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![f16::ZERO; numel];
    dev_y.copy_to_host(&mut got).expect("dl");

    let tol = 8.0 * F16_EPS;
    for i in 0..numel {
        let t = (expected[i].abs() * tol).max(tol);
        let diff = (got[i].to_f32() - expected[i]).abs();
        assert!(diff <= t, "f16 rope y @ {i}: diff={diff} want={}", expected[i]);
    }
}

#[test]
#[ignore]
fn rope_bf16_default_positions() {
    let (ctx, stream) = setup();
    let batch = 2i32;
    let heads = 2i32;
    let seq = 6i32;
    let head_dim = 12i32;
    let numel = (batch * heads * seq * head_dim) as usize;
    let host_x_f32: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.17 - 0.9).cos() * 0.7)
        .collect();
    let base = ROPE_DEFAULT_BASE;
    let expected = host_rope_f32(
        batch as usize,
        heads as usize,
        seq as usize,
        head_dim as usize,
        base,
        &host_x_f32,
        None,
    );

    let host_x: Vec<bf16> = host_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let desc = RopeDescriptor {
        batch_size: batch,
        num_heads: heads,
        seq_len: seq,
        head_dim,
        base,
        element: ElementKind::Bf16,
    };
    let plan = RopePlan::<bf16>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let shape = [batch, heads, seq, head_dim];
    plan.run(
        &stream,
        Workspace::None,
        RopeArgs {
            x: TensorRef {
                data: dev_x.as_slice(),
                shape,
                stride: contiguous_stride(shape),
            },
            positions: None,
            y: TensorMut {
                data: dev_y.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![bf16::ZERO; numel];
    dev_y.copy_to_host(&mut got).expect("dl");

    let tol = 8.0 * BF16_EPS;
    for i in 0..numel {
        let t = (expected[i].abs() * tol).max(tol);
        let diff = (got[i].to_f32() - expected[i]).abs();
        assert!(
            diff <= t,
            "bf16 rope y @ {i}: diff={diff} want={}",
            expected[i]
        );
    }
}
