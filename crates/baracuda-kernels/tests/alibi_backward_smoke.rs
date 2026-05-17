//! Real-GPU smoke test for `AlibiBackwardPlan` (Phase 6.1).
//!
//! BW:  dA[b, h, i, j] = dy[b, h, i, j]                (pass-through copy)
//!      dslope[h]     = Σ_{b, i, j} dy[b, h, i, j] · (j - i)
//!
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, AlibiBackwardArgs, AlibiBackwardDescriptor, AlibiBackwardPlan, ElementKind,
    PlanPreference, TensorMut, TensorRef, Workspace,
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

fn host_alibi_backward_f32(
    batch: usize,
    heads: usize,
    q: usize,
    k: usize,
    dy: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let total = batch * heads * q * k;
    let da = dy.to_vec();
    let mut dslope = vec![0f32; heads];
    for b in 0..batch {
        for h in 0..heads {
            let mut acc = 0.0f64;
            for i in 0..q {
                for j in 0..k {
                    let off = ((b * heads + h) * q + i) * k + j;
                    let delta = (j as f64) - (i as f64);
                    acc += dy[off] as f64 * delta;
                }
            }
            dslope[h] += acc as f32;
        }
    }
    let _ = total;
    (da, dslope)
}

#[test]
#[ignore]
fn alibi_backward_f32_basic() {
    let (ctx, stream) = setup();
    let batch = 2i32;
    let heads = 4i32;
    let q = 5i32;
    let k = 7i32;
    let numel = (batch * heads * q * k) as usize;
    let host_dy: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.07 - 0.3).sin() * 0.5)
        .collect();
    let (expected_da, expected_dslope) = host_alibi_backward_f32(
        batch as usize,
        heads as usize,
        q as usize,
        k as usize,
        &host_dy,
    );

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_da: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc da");
    let mut dev_dslope: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, heads as usize).expect("alloc dslope");

    let desc = AlibiBackwardDescriptor {
        batch_size: batch,
        num_heads: heads,
        query_len: q,
        key_len: k,
        element: ElementKind::F32,
    };
    let plan = AlibiBackwardPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let shape = [batch, heads, q, k];
    plan.run(
        &stream,
        Workspace::None,
        AlibiBackwardArgs {
            dy: TensorRef {
                data: dev_dy.as_slice(),
                shape,
                stride: contiguous_stride(shape),
            },
            dscores: Some(TensorMut {
                data: dev_da.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            }),
            dslopes: Some(TensorMut {
                data: dev_dslope.as_slice_mut(),
                shape: [heads],
                stride: [1],
            }),
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got_da = vec![0f32; numel];
    let mut got_dslope = vec![0f32; heads as usize];
    dev_da.copy_to_host(&mut got_da).expect("dl da");
    dev_dslope.copy_to_host(&mut got_dslope).expect("dl dslope");

    // dA is a bit-exact copy
    for i in 0..numel {
        assert_eq!(got_da[i], expected_da[i], "f32 alibi_bw da @ {i}");
    }
    // dslope reduction — 16 eps relative tolerance
    let tol_factor = 16.0 * f32::EPSILON;
    for h in 0..heads as usize {
        let tol = (expected_dslope[h].abs() * tol_factor).max(tol_factor);
        let diff = (got_dslope[h] - expected_dslope[h]).abs();
        assert!(
            diff <= tol * (numel as f32),
            "f32 alibi_bw dslope @ {h}: diff={diff} want={} got={}",
            expected_dslope[h],
            got_dslope[h]
        );
    }
}

#[test]
#[ignore]
fn alibi_backward_f64_basic() {
    let (ctx, stream) = setup();
    let batch = 2i32;
    let heads = 3i32;
    let q = 5i32;
    let k = 6i32;
    let numel = (batch * heads * q * k) as usize;
    let host_dy: Vec<f64> = (0..numel)
        .map(|i| ((i as f64) * 0.09 - 0.4).cos() * 0.6)
        .collect();
    let mut expected_dslope = vec![0f64; heads as usize];
    for b in 0..batch as usize {
        for h in 0..heads as usize {
            let mut acc = 0.0f64;
            for i in 0..q as usize {
                for j in 0..k as usize {
                    let off = ((b * heads as usize + h) * q as usize + i) * k as usize + j;
                    let delta = (j as f64) - (i as f64);
                    acc += host_dy[off] * delta;
                }
            }
            expected_dslope[h] += acc;
        }
    }
    let expected_da = host_dy.clone();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_da: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc da");
    let mut dev_dslope: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, heads as usize).expect("alloc dslope");

    let desc = AlibiBackwardDescriptor {
        batch_size: batch,
        num_heads: heads,
        query_len: q,
        key_len: k,
        element: ElementKind::F64,
    };
    let plan = AlibiBackwardPlan::<f64>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let shape = [batch, heads, q, k];
    plan.run(
        &stream,
        Workspace::None,
        AlibiBackwardArgs {
            dy: TensorRef {
                data: dev_dy.as_slice(),
                shape,
                stride: contiguous_stride(shape),
            },
            dscores: Some(TensorMut {
                data: dev_da.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            }),
            dslopes: Some(TensorMut {
                data: dev_dslope.as_slice_mut(),
                shape: [heads],
                stride: [1],
            }),
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got_da = vec![0f64; numel];
    let mut got_dslope = vec![0f64; heads as usize];
    dev_da.copy_to_host(&mut got_da).expect("dl da");
    dev_dslope.copy_to_host(&mut got_dslope).expect("dl dslope");

    for i in 0..numel {
        assert_eq!(got_da[i], expected_da[i], "f64 alibi_bw da @ {i}");
    }
    let tol_factor = 16.0 * f64::EPSILON;
    for h in 0..heads as usize {
        let tol = (expected_dslope[h].abs() * tol_factor).max(tol_factor);
        let diff = (got_dslope[h] - expected_dslope[h]).abs();
        assert!(
            diff <= tol * (numel as f64),
            "f64 alibi_bw dslope @ {h}: diff={diff} want={} got={}",
            expected_dslope[h],
            got_dslope[h]
        );
    }
}

#[test]
#[ignore]
fn alibi_backward_f16_basic() {
    let (ctx, stream) = setup();
    let batch = 2i32;
    let heads = 4i32;
    let q = 5i32;
    let k = 7i32;
    let numel = (batch * heads * q * k) as usize;
    let host_dy_f32: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.05 - 0.2).sin() * 0.4)
        .collect();
    let (_expected_da_f32, expected_dslope_f32) = host_alibi_backward_f32(
        batch as usize,
        heads as usize,
        q as usize,
        k as usize,
        &host_dy_f32,
    );
    let host_dy: Vec<f16> = host_dy_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let expected_da: Vec<f16> = host_dy.clone();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_da: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc da");
    let mut dev_dslope: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, heads as usize).expect("alloc dslope");

    let desc = AlibiBackwardDescriptor {
        batch_size: batch,
        num_heads: heads,
        query_len: q,
        key_len: k,
        element: ElementKind::F16,
    };
    let plan = AlibiBackwardPlan::<f16>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let shape = [batch, heads, q, k];
    plan.run(
        &stream,
        Workspace::None,
        AlibiBackwardArgs {
            dy: TensorRef {
                data: dev_dy.as_slice(),
                shape,
                stride: contiguous_stride(shape),
            },
            dscores: Some(TensorMut {
                data: dev_da.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            }),
            dslopes: Some(TensorMut {
                data: dev_dslope.as_slice_mut(),
                shape: [heads],
                stride: [1],
            }),
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got_da = vec![f16::ZERO; numel];
    let mut got_dslope = vec![f16::ZERO; heads as usize];
    dev_da.copy_to_host(&mut got_da).expect("dl da");
    dev_dslope.copy_to_host(&mut got_dslope).expect("dl dslope");

    // dA is bit-exact copy at f16
    for i in 0..numel {
        assert_eq!(
            got_da[i].to_bits(),
            expected_da[i].to_bits(),
            "f16 alibi_bw da @ {i}"
        );
    }
    let tol = (numel as f32) * 4.0 * F16_EPS;
    for h in 0..heads as usize {
        let t = (expected_dslope_f32[h].abs() * tol).max(tol);
        let diff = (got_dslope[h].to_f32() - expected_dslope_f32[h]).abs();
        assert!(
            diff <= t,
            "f16 alibi_bw dslope @ {h}: diff={diff} want={}",
            expected_dslope_f32[h]
        );
    }
}

#[test]
#[ignore]
fn alibi_backward_bf16_basic() {
    let (ctx, stream) = setup();
    let batch = 2i32;
    let heads = 3i32;
    let q = 5i32;
    let k = 6i32;
    let numel = (batch * heads * q * k) as usize;
    let host_dy_f32: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.08 - 0.5).cos() * 0.3)
        .collect();
    let (_expected_da_f32, expected_dslope_f32) = host_alibi_backward_f32(
        batch as usize,
        heads as usize,
        q as usize,
        k as usize,
        &host_dy_f32,
    );
    let host_dy: Vec<bf16> = host_dy_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let expected_da: Vec<bf16> = host_dy.clone();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_da: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc da");
    let mut dev_dslope: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, heads as usize).expect("alloc dslope");

    let desc = AlibiBackwardDescriptor {
        batch_size: batch,
        num_heads: heads,
        query_len: q,
        key_len: k,
        element: ElementKind::Bf16,
    };
    let plan = AlibiBackwardPlan::<bf16>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let shape = [batch, heads, q, k];
    plan.run(
        &stream,
        Workspace::None,
        AlibiBackwardArgs {
            dy: TensorRef {
                data: dev_dy.as_slice(),
                shape,
                stride: contiguous_stride(shape),
            },
            dscores: Some(TensorMut {
                data: dev_da.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            }),
            dslopes: Some(TensorMut {
                data: dev_dslope.as_slice_mut(),
                shape: [heads],
                stride: [1],
            }),
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got_da = vec![bf16::ZERO; numel];
    let mut got_dslope = vec![bf16::ZERO; heads as usize];
    dev_da.copy_to_host(&mut got_da).expect("dl da");
    dev_dslope.copy_to_host(&mut got_dslope).expect("dl dslope");

    for i in 0..numel {
        assert_eq!(
            got_da[i].to_bits(),
            expected_da[i].to_bits(),
            "bf16 alibi_bw da @ {i}"
        );
    }
    let tol = (numel as f32) * 4.0 * BF16_EPS;
    for h in 0..heads as usize {
        let t = (expected_dslope_f32[h].abs() * tol).max(tol);
        let diff = (got_dslope[h].to_f32() - expected_dslope_f32[h]).abs();
        assert!(
            diff <= t,
            "bf16 alibi_bw dslope @ {h}: diff={diff} want={}",
            expected_dslope_f32[h]
        );
    }
}
