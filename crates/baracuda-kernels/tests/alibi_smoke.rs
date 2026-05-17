//! Real-GPU smoke test for `AlibiPlan + AttentionKind::Alibi` FW.
//!
//! `y[b, h, i, j] = scores[b, h, i, j] + slope[h] · (j - i)`.
//!
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, AlibiArgs, AlibiDescriptor, AlibiPlan, ElementKind, PlanPreference,
    TensorMut, TensorRef, Workspace,
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

fn host_alibi_f32(
    batch: usize,
    heads: usize,
    q: usize,
    k: usize,
    scores: &[f32],
    slopes: &[f32],
) -> Vec<f32> {
    let total = batch * heads * q * k;
    let mut y = vec![0f32; total];
    for b in 0..batch {
        for h in 0..heads {
            for i in 0..q {
                for j in 0..k {
                    let off = ((b * heads + h) * q + i) * k + j;
                    let delta = (j as f32) - (i as f32);
                    y[off] = scores[off] + slopes[h] * delta;
                }
            }
        }
    }
    y
}

#[test]
#[ignore]
fn alibi_f32_basic() {
    let (ctx, stream) = setup();
    let batch = 2i32;
    let heads = 4i32;
    let q = 6i32;
    let k = 8i32;
    let numel = (batch * heads * q * k) as usize;
    let host_scores: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.03 - 0.4).sin())
        .collect();
    let host_slopes: Vec<f32> = (0..heads as usize)
        .map(|h| 0.1 + 0.05 * (h as f32))
        .collect();
    let expected = host_alibi_f32(
        batch as usize,
        heads as usize,
        q as usize,
        k as usize,
        &host_scores,
        &host_slopes,
    );

    let dev_scores = DeviceBuffer::from_slice(&ctx, &host_scores).expect("up s");
    let dev_slopes = DeviceBuffer::from_slice(&ctx, &host_slopes).expect("up sl");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let desc = AlibiDescriptor {
        batch_size: batch,
        num_heads: heads,
        query_len: q,
        key_len: k,
        element: ElementKind::F32,
    };
    let plan = AlibiPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let shape = [batch, heads, q, k];
    plan.run(
        &stream,
        Workspace::None,
        AlibiArgs {
            scores: TensorRef {
                data: dev_scores.as_slice(),
                shape,
                stride: contiguous_stride(shape),
            },
            slopes: TensorRef {
                data: dev_slopes.as_slice(),
                shape: [heads],
                stride: [1],
            },
            out: TensorMut {
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

    // Single add per cell — bit-exact at f32.
    for i in 0..numel {
        let diff = (got[i] - expected[i]).abs();
        let tol = (expected[i].abs() * f32::EPSILON).max(f32::EPSILON);
        assert!(diff <= tol, "f32 alibi y @ {i}: diff={diff}");
    }
}

#[test]
#[ignore]
fn alibi_f64_basic() {
    let (ctx, stream) = setup();
    let batch = 2i32;
    let heads = 3i32;
    let q = 5i32;
    let k = 7i32;
    let numel = (batch * heads * q * k) as usize;
    let host_scores: Vec<f64> = (0..numel)
        .map(|i| ((i as f64) * 0.04 - 0.5).cos())
        .collect();
    let host_slopes: Vec<f64> = (0..heads as usize)
        .map(|h| 0.05 + 0.02 * (h as f64))
        .collect();
    let mut expected = vec![0f64; numel];
    for b in 0..batch as usize {
        for h in 0..heads as usize {
            for i in 0..q as usize {
                for j in 0..k as usize {
                    let off = ((b * heads as usize + h) * q as usize + i) * k as usize + j;
                    let delta = (j as f64) - (i as f64);
                    expected[off] = host_scores[off] + host_slopes[h] * delta;
                }
            }
        }
    }

    let dev_scores = DeviceBuffer::from_slice(&ctx, &host_scores).expect("up s");
    let dev_slopes = DeviceBuffer::from_slice(&ctx, &host_slopes).expect("up sl");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let desc = AlibiDescriptor {
        batch_size: batch,
        num_heads: heads,
        query_len: q,
        key_len: k,
        element: ElementKind::F64,
    };
    let plan = AlibiPlan::<f64>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let shape = [batch, heads, q, k];
    plan.run(
        &stream,
        Workspace::None,
        AlibiArgs {
            scores: TensorRef {
                data: dev_scores.as_slice(),
                shape,
                stride: contiguous_stride(shape),
            },
            slopes: TensorRef {
                data: dev_slopes.as_slice(),
                shape: [heads],
                stride: [1],
            },
            out: TensorMut {
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
    for i in 0..numel {
        let tol = (expected[i].abs() * f64::EPSILON).max(f64::EPSILON);
        assert!((got[i] - expected[i]).abs() <= tol, "f64 alibi y @ {i}");
    }
}

#[test]
#[ignore]
fn alibi_f16_basic() {
    let (ctx, stream) = setup();
    let batch = 2i32;
    let heads = 4i32;
    let q = 6i32;
    let k = 8i32;
    let numel = (batch * heads * q * k) as usize;
    let host_scores_f32: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.025 - 0.3).sin())
        .collect();
    let host_slopes_f32: Vec<f32> = (0..heads as usize)
        .map(|h| 0.1 + 0.05 * (h as f32))
        .collect();
    let expected = host_alibi_f32(
        batch as usize,
        heads as usize,
        q as usize,
        k as usize,
        &host_scores_f32,
        &host_slopes_f32,
    );

    let host_scores: Vec<f16> = host_scores_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_slopes: Vec<f16> = host_slopes_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let dev_scores = DeviceBuffer::from_slice(&ctx, &host_scores).expect("up s");
    let dev_slopes = DeviceBuffer::from_slice(&ctx, &host_slopes).expect("up sl");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let desc = AlibiDescriptor {
        batch_size: batch,
        num_heads: heads,
        query_len: q,
        key_len: k,
        element: ElementKind::F16,
    };
    let plan = AlibiPlan::<f16>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let shape = [batch, heads, q, k];
    plan.run(
        &stream,
        Workspace::None,
        AlibiArgs {
            scores: TensorRef {
                data: dev_scores.as_slice(),
                shape,
                stride: contiguous_stride(shape),
            },
            slopes: TensorRef {
                data: dev_slopes.as_slice(),
                shape: [heads],
                stride: [1],
            },
            out: TensorMut {
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

    let tol = 4.0 * F16_EPS;
    for i in 0..numel {
        let t = (expected[i].abs() * tol).max(tol);
        let diff = (got[i].to_f32() - expected[i]).abs();
        assert!(diff <= t, "f16 alibi y @ {i}: diff={diff}");
    }
}

#[test]
#[ignore]
fn alibi_bf16_basic() {
    let (ctx, stream) = setup();
    let batch = 2i32;
    let heads = 3i32;
    let q = 5i32;
    let k = 7i32;
    let numel = (batch * heads * q * k) as usize;
    let host_scores_f32: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.03 - 0.4).cos())
        .collect();
    let host_slopes_f32: Vec<f32> = (0..heads as usize)
        .map(|h| 0.05 + 0.02 * (h as f32))
        .collect();
    let expected = host_alibi_f32(
        batch as usize,
        heads as usize,
        q as usize,
        k as usize,
        &host_scores_f32,
        &host_slopes_f32,
    );

    let host_scores: Vec<bf16> = host_scores_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_slopes: Vec<bf16> = host_slopes_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let dev_scores = DeviceBuffer::from_slice(&ctx, &host_scores).expect("up s");
    let dev_slopes = DeviceBuffer::from_slice(&ctx, &host_slopes).expect("up sl");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let desc = AlibiDescriptor {
        batch_size: batch,
        num_heads: heads,
        query_len: q,
        key_len: k,
        element: ElementKind::Bf16,
    };
    let plan = AlibiPlan::<bf16>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let shape = [batch, heads, q, k];
    plan.run(
        &stream,
        Workspace::None,
        AlibiArgs {
            scores: TensorRef {
                data: dev_scores.as_slice(),
                shape,
                stride: contiguous_stride(shape),
            },
            slopes: TensorRef {
                data: dev_slopes.as_slice(),
                shape: [heads],
                stride: [1],
            },
            out: TensorMut {
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

    let tol = 4.0 * BF16_EPS;
    for i in 0..numel {
        let t = (expected[i].abs() * tol).max(tol);
        let diff = (got[i].to_f32() - expected[i]).abs();
        assert!(diff <= t, "bf16 alibi y @ {i}: diff={diff}");
    }
}
