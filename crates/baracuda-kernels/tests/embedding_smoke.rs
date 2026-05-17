//! Real-GPU smoke test for `EmbeddingPlan<T>` (Phase 7 Milestone 7.5).
//!
//! `out[n, :] = weight[indices[n], :]` with optional `padding_idx`
//! zeroing matching rows. Pure copy → bit-exact across every wired
//! dtype.
//!
//! Coverage:
//! - f32 / f64 — base lookup with no padding.
//! - f32 with `padding_idx` — row-zero on match, OOB skip.
//! - f16 / bf16 — pure-copy parity (FW only).
//!
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, EmbeddingArgs, EmbeddingDescriptor, EmbeddingPlan,
    PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn embedding_f32_basic() {
    let (ctx, stream) = setup();
    let v: usize = 5;
    let d: usize = 4;
    let n: usize = 3;
    let host_w: Vec<f32> = (0..(v * d)).map(|i| i as f32 * 0.25 + 1.0).collect();
    let host_idx: Vec<i32> = vec![1, 3, 0];
    let mut expected = vec![0f32; n * d];
    for (i, &idx) in host_idx.iter().enumerate() {
        for j in 0..d {
            expected[i * d + j] = host_w[idx as usize * d + j];
        }
    }
    let dev_w = DeviceBuffer::from_slice(&ctx, &host_w).expect("up weight");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let mut dev_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, n * d).expect("alloc out");

    let desc = EmbeddingDescriptor {
        num_embeddings: v as i32,
        embedding_dim: d as i32,
        num_indices: n as i32,
        padding_idx: None,
        element: ElementKind::F32,
    };
    let plan = EmbeddingPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = EmbeddingArgs::<f32> {
        weight: TensorRef {
            data: dev_w.as_slice(),
            shape: [v as i32, d as i32],
            stride: contiguous_stride([v as i32, d as i32]),
        },
        indices: TensorRef {
            data: dev_idx.as_slice(),
            shape: [n as i32],
            stride: contiguous_stride([n as i32]),
        },
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [n as i32, d as i32],
            stride: contiguous_stride([n as i32, d as i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; n * d];
    dev_out.copy_to_host(&mut got).expect("dl");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "embedding f32 mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn embedding_f64_basic() {
    let (ctx, stream) = setup();
    let v: usize = 5;
    let d: usize = 3;
    let n: usize = 4;
    let host_w: Vec<f64> = (0..(v * d)).map(|i| i as f64 * -0.5 + 2.0).collect();
    let host_idx: Vec<i32> = vec![4, 2, 0, 1];
    let mut expected = vec![0f64; n * d];
    for (i, &idx) in host_idx.iter().enumerate() {
        for j in 0..d {
            expected[i * d + j] = host_w[idx as usize * d + j];
        }
    }
    let dev_w = DeviceBuffer::from_slice(&ctx, &host_w).expect("up weight");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let mut dev_out: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, n * d).expect("alloc out");

    let desc = EmbeddingDescriptor {
        num_embeddings: v as i32,
        embedding_dim: d as i32,
        num_indices: n as i32,
        padding_idx: None,
        element: ElementKind::F64,
    };
    let plan = EmbeddingPlan::<f64>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = EmbeddingArgs::<f64> {
        weight: TensorRef {
            data: dev_w.as_slice(),
            shape: [v as i32, d as i32],
            stride: contiguous_stride([v as i32, d as i32]),
        },
        indices: TensorRef {
            data: dev_idx.as_slice(),
            shape: [n as i32],
            stride: contiguous_stride([n as i32]),
        },
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [n as i32, d as i32],
            stride: contiguous_stride([n as i32, d as i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; n * d];
    dev_out.copy_to_host(&mut got).expect("dl");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "embedding f64 mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn embedding_f32_padding_idx_zeros_row() {
    let (ctx, stream) = setup();
    let v: usize = 5;
    let d: usize = 4;
    let n: usize = 4;
    let host_w: Vec<f32> = (0..(v * d)).map(|i| i as f32 + 1.0).collect();
    // padding_idx = 0; indices contain 0 (-> zero row), 2, 4, -1 (OOB -> zero row).
    let host_idx: Vec<i32> = vec![0, 2, 4, -1];
    let padding = 0i32;
    let mut expected = vec![0f32; n * d];
    for (i, &idx) in host_idx.iter().enumerate() {
        if idx == padding || idx < 0 || idx >= v as i32 {
            continue; // already zeros
        }
        for j in 0..d {
            expected[i * d + j] = host_w[idx as usize * d + j];
        }
    }
    let dev_w = DeviceBuffer::from_slice(&ctx, &host_w).expect("up weight");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let mut dev_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, n * d).expect("alloc out");

    let desc = EmbeddingDescriptor {
        num_embeddings: v as i32,
        embedding_dim: d as i32,
        num_indices: n as i32,
        padding_idx: Some(padding),
        element: ElementKind::F32,
    };
    let plan = EmbeddingPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = EmbeddingArgs::<f32> {
        weight: TensorRef {
            data: dev_w.as_slice(),
            shape: [v as i32, d as i32],
            stride: contiguous_stride([v as i32, d as i32]),
        },
        indices: TensorRef {
            data: dev_idx.as_slice(),
            shape: [n as i32],
            stride: contiguous_stride([n as i32]),
        },
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [n as i32, d as i32],
            stride: contiguous_stride([n as i32, d as i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; n * d];
    dev_out.copy_to_host(&mut got).expect("dl");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "embedding f32 padding mismatch @ {i}"
        );
    }
}

#[test]
#[ignore]
fn embedding_f16_basic() {
    let (ctx, stream) = setup();
    let v: usize = 4;
    let d: usize = 3;
    let n: usize = 3;
    let host_w_f32: Vec<f32> = (0..(v * d)).map(|i| (i as f32) * 0.5 - 1.0).collect();
    let host_w: Vec<f16> = host_w_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let host_idx: Vec<i32> = vec![3, 0, 1];
    let mut expected = vec![f16::from_f32(0.0); n * d];
    for (i, &idx) in host_idx.iter().enumerate() {
        for j in 0..d {
            expected[i * d + j] = host_w[idx as usize * d + j];
        }
    }
    let dev_w = DeviceBuffer::from_slice(&ctx, &host_w).expect("up weight");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let mut dev_out: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, n * d).expect("alloc out");

    let desc = EmbeddingDescriptor {
        num_embeddings: v as i32,
        embedding_dim: d as i32,
        num_indices: n as i32,
        padding_idx: None,
        element: ElementKind::F16,
    };
    let plan = EmbeddingPlan::<f16>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = EmbeddingArgs::<f16> {
        weight: TensorRef {
            data: dev_w.as_slice(),
            shape: [v as i32, d as i32],
            stride: contiguous_stride([v as i32, d as i32]),
        },
        indices: TensorRef {
            data: dev_idx.as_slice(),
            shape: [n as i32],
            stride: contiguous_stride([n as i32]),
        },
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [n as i32, d as i32],
            stride: contiguous_stride([n as i32, d as i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![f16::from_f32(0.0); n * d];
    dev_out.copy_to_host(&mut got).expect("dl");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "embedding f16 mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn embedding_bf16_basic() {
    let (ctx, stream) = setup();
    let v: usize = 4;
    let d: usize = 3;
    let n: usize = 3;
    let host_w_f32: Vec<f32> = (0..(v * d)).map(|i| (i as f32) * 0.25 + 0.5).collect();
    let host_w: Vec<bf16> = host_w_f32.iter().map(|&x| bf16::from_f32(x)).collect();
    let host_idx: Vec<i32> = vec![2, 0, 3];
    let mut expected = vec![bf16::from_f32(0.0); n * d];
    for (i, &idx) in host_idx.iter().enumerate() {
        for j in 0..d {
            expected[i * d + j] = host_w[idx as usize * d + j];
        }
    }
    let dev_w = DeviceBuffer::from_slice(&ctx, &host_w).expect("up weight");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let mut dev_out: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, n * d).expect("alloc out");

    let desc = EmbeddingDescriptor {
        num_embeddings: v as i32,
        embedding_dim: d as i32,
        num_indices: n as i32,
        padding_idx: None,
        element: ElementKind::Bf16,
    };
    let plan = EmbeddingPlan::<bf16>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = EmbeddingArgs::<bf16> {
        weight: TensorRef {
            data: dev_w.as_slice(),
            shape: [v as i32, d as i32],
            stride: contiguous_stride([v as i32, d as i32]),
        },
        indices: TensorRef {
            data: dev_idx.as_slice(),
            shape: [n as i32],
            stride: contiguous_stride([n as i32]),
        },
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [n as i32, d as i32],
            stride: contiguous_stride([n as i32, d as i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![bf16::from_f32(0.0); n * d];
    dev_out.copy_to_host(&mut got).expect("dl");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "embedding bf16 mismatch @ {i}");
    }
}
