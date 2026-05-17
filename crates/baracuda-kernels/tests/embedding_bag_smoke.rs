//! Real-GPU smoke test for `EmbeddingBagPlan<T>` (Phase 7 Milestone 7.5).
//!
//! Per-bag reduction over a flat indices buffer partitioned by an
//! offsets table. Sum + Mean modes; empty-bag and padding_idx behavior.
//!
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, EmbeddingBagArgs, EmbeddingBagDescriptor, EmbeddingBagMode,
    EmbeddingBagPlan, PlanPreference, TensorMut, TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn cpu_embedding_bag_f32(
    v: usize,
    d: usize,
    weight: &[f32],
    indices: &[i32],
    offsets: &[i32],
    num_bags: usize,
    total_indices: usize,
    mode: EmbeddingBagMode,
    padding_idx: Option<i32>,
) -> Vec<f32> {
    let mut out = vec![0f32; num_bags * d];
    for b in 0..num_bags {
        let start = offsets[b] as usize;
        let end = if b + 1 < num_bags {
            offsets[b + 1] as usize
        } else {
            total_indices
        };
        let mut counted = 0usize;
        let mut acc = vec![0f64; d];
        for k in start..end {
            let idx = indices[k];
            if Some(idx) == padding_idx || idx < 0 || idx >= v as i32 {
                continue;
            }
            counted += 1;
            for j in 0..d {
                acc[j] += weight[idx as usize * d + j] as f64;
            }
        }
        let divisor = match mode {
            EmbeddingBagMode::Sum => 1f64,
            EmbeddingBagMode::Mean => {
                if counted == 0 {
                    // Empty / all-padded: output zero.
                    for j in 0..d {
                        out[b * d + j] = 0f32;
                    }
                    continue;
                }
                counted as f64
            }
        };
        for j in 0..d {
            out[b * d + j] = (acc[j] / divisor) as f32;
        }
    }
    out
}

#[test]
#[ignore]
fn embedding_bag_f32_sum_basic() {
    let (ctx, stream) = setup();
    let v: usize = 5;
    let d: usize = 4;
    let host_w: Vec<f32> = (0..(v * d)).map(|i| (i as f32) * 0.25 + 1.0).collect();
    // 3 bags: [1, 3], [0, 4, 2], [].
    let host_idx: Vec<i32> = vec![1, 3, 0, 4, 2];
    let host_off: Vec<i32> = vec![0, 2, 5];
    let num_bags = 3usize;
    let total_indices = host_idx.len();
    let expected = cpu_embedding_bag_f32(
        v,
        d,
        &host_w,
        &host_idx,
        &host_off,
        num_bags,
        total_indices,
        EmbeddingBagMode::Sum,
        None,
    );

    let dev_w = DeviceBuffer::from_slice(&ctx, &host_w).expect("up weight");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let dev_off = DeviceBuffer::from_slice(&ctx, &host_off).expect("up offsets");
    let mut dev_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, num_bags * d).expect("alloc out");

    let desc = EmbeddingBagDescriptor {
        num_embeddings: v as i32,
        embedding_dim: d as i32,
        num_bags: num_bags as i32,
        total_indices: total_indices as i32,
        mode: EmbeddingBagMode::Sum,
        padding_idx: None,
        element: ElementKind::F32,
    };
    let plan = EmbeddingBagPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = EmbeddingBagArgs::<f32> {
        weight: TensorRef {
            data: dev_w.as_slice(),
            shape: [v as i32, d as i32],
            stride: contiguous_stride([v as i32, d as i32]),
        },
        indices: TensorRef {
            data: dev_idx.as_slice(),
            shape: [total_indices as i32],
            stride: contiguous_stride([total_indices as i32]),
        },
        offsets: TensorRef {
            data: dev_off.as_slice(),
            shape: [num_bags as i32],
            stride: contiguous_stride([num_bags as i32]),
        },
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [num_bags as i32, d as i32],
            stride: contiguous_stride([num_bags as i32, d as i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; num_bags * d];
    dev_out.copy_to_host(&mut got).expect("dl");
    let eps = f32::EPSILON;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let tol = 16.0 * eps * e.abs().max(1.0);
        assert!(
            (g - e).abs() <= tol,
            "embedding_bag f32 sum mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn embedding_bag_f32_mean_basic() {
    let (ctx, stream) = setup();
    let v: usize = 5;
    let d: usize = 3;
    let host_w: Vec<f32> = (0..(v * d)).map(|i| (i as f32) - 2.0).collect();
    let host_idx: Vec<i32> = vec![0, 2, 4, 1, 3];
    let host_off: Vec<i32> = vec![0, 3, 5]; // bags: [0,2,4], [1,3]
    let num_bags = host_off.len();
    let total_indices = host_idx.len();
    let expected = cpu_embedding_bag_f32(
        v,
        d,
        &host_w,
        &host_idx,
        &host_off,
        num_bags,
        total_indices,
        EmbeddingBagMode::Mean,
        None,
    );

    let dev_w = DeviceBuffer::from_slice(&ctx, &host_w).expect("up weight");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let dev_off = DeviceBuffer::from_slice(&ctx, &host_off).expect("up offsets");
    let mut dev_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, num_bags * d).expect("alloc out");

    let desc = EmbeddingBagDescriptor {
        num_embeddings: v as i32,
        embedding_dim: d as i32,
        num_bags: num_bags as i32,
        total_indices: total_indices as i32,
        mode: EmbeddingBagMode::Mean,
        padding_idx: None,
        element: ElementKind::F32,
    };
    let plan = EmbeddingBagPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = EmbeddingBagArgs::<f32> {
        weight: TensorRef {
            data: dev_w.as_slice(),
            shape: [v as i32, d as i32],
            stride: contiguous_stride([v as i32, d as i32]),
        },
        indices: TensorRef {
            data: dev_idx.as_slice(),
            shape: [total_indices as i32],
            stride: contiguous_stride([total_indices as i32]),
        },
        offsets: TensorRef {
            data: dev_off.as_slice(),
            shape: [num_bags as i32],
            stride: contiguous_stride([num_bags as i32]),
        },
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [num_bags as i32, d as i32],
            stride: contiguous_stride([num_bags as i32, d as i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; num_bags * d];
    dev_out.copy_to_host(&mut got).expect("dl");
    let eps = f32::EPSILON;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let tol = 32.0 * eps * e.abs().max(1.0);
        assert!(
            (g - e).abs() <= tol,
            "embedding_bag f32 mean mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn embedding_bag_f32_padding_idx_and_empty_bag() {
    let (ctx, stream) = setup();
    let v: usize = 5;
    let d: usize = 3;
    let host_w: Vec<f32> = (0..(v * d)).map(|i| (i as f32) * 0.5 + 1.0).collect();
    let padding = 4i32;
    // 3 bags: [1, 4, 2] (4 padded → mean uses 2), [] (empty → zero row),
    // [4, 4] (all padded → mean → zero), final implicit bag uses end=total.
    let host_idx: Vec<i32> = vec![1, 4, 2, 4, 4];
    let host_off: Vec<i32> = vec![0, 3, 3];
    let num_bags = host_off.len();
    let total_indices = host_idx.len();
    let expected = cpu_embedding_bag_f32(
        v,
        d,
        &host_w,
        &host_idx,
        &host_off,
        num_bags,
        total_indices,
        EmbeddingBagMode::Mean,
        Some(padding),
    );

    let dev_w = DeviceBuffer::from_slice(&ctx, &host_w).expect("up weight");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let dev_off = DeviceBuffer::from_slice(&ctx, &host_off).expect("up offsets");
    let mut dev_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, num_bags * d).expect("alloc out");

    let desc = EmbeddingBagDescriptor {
        num_embeddings: v as i32,
        embedding_dim: d as i32,
        num_bags: num_bags as i32,
        total_indices: total_indices as i32,
        mode: EmbeddingBagMode::Mean,
        padding_idx: Some(padding),
        element: ElementKind::F32,
    };
    let plan = EmbeddingBagPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = EmbeddingBagArgs::<f32> {
        weight: TensorRef {
            data: dev_w.as_slice(),
            shape: [v as i32, d as i32],
            stride: contiguous_stride([v as i32, d as i32]),
        },
        indices: TensorRef {
            data: dev_idx.as_slice(),
            shape: [total_indices as i32],
            stride: contiguous_stride([total_indices as i32]),
        },
        offsets: TensorRef {
            data: dev_off.as_slice(),
            shape: [num_bags as i32],
            stride: contiguous_stride([num_bags as i32]),
        },
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [num_bags as i32, d as i32],
            stride: contiguous_stride([num_bags as i32, d as i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; num_bags * d];
    dev_out.copy_to_host(&mut got).expect("dl");
    let eps = f32::EPSILON;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let tol = 32.0 * eps * e.abs().max(1.0);
        assert!(
            (g - e).abs() <= tol,
            "embedding_bag f32 padding/empty mismatch @ {i}: got {g} expected {e}"
        );
    }
    // Bag 1 (empty) must be exact zero.
    for j in 0..d {
        assert_eq!(
            got[1 * d + j].to_bits(),
            0u32,
            "empty bag must produce zero @ feature {j}"
        );
    }
}

#[test]
#[ignore]
fn embedding_bag_f64_sum() {
    let (ctx, stream) = setup();
    let v: usize = 4;
    let d: usize = 3;
    let host_w: Vec<f64> = (0..(v * d)).map(|i| (i as f64) * 0.125 - 1.0).collect();
    let host_idx: Vec<i32> = vec![0, 2, 3, 1];
    let host_off: Vec<i32> = vec![0, 2];
    let num_bags = host_off.len();
    let total_indices = host_idx.len();
    let mut expected = vec![0f64; num_bags * d];
    for b in 0..num_bags {
        let start = host_off[b] as usize;
        let end = if b + 1 < num_bags {
            host_off[b + 1] as usize
        } else {
            total_indices
        };
        for k in start..end {
            let idx = host_idx[k] as usize;
            for j in 0..d {
                expected[b * d + j] += host_w[idx * d + j];
            }
        }
    }

    let dev_w = DeviceBuffer::from_slice(&ctx, &host_w).expect("up weight");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let dev_off = DeviceBuffer::from_slice(&ctx, &host_off).expect("up offsets");
    let mut dev_out: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, num_bags * d).expect("alloc out");

    let desc = EmbeddingBagDescriptor {
        num_embeddings: v as i32,
        embedding_dim: d as i32,
        num_bags: num_bags as i32,
        total_indices: total_indices as i32,
        mode: EmbeddingBagMode::Sum,
        padding_idx: None,
        element: ElementKind::F64,
    };
    let plan = EmbeddingBagPlan::<f64>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = EmbeddingBagArgs::<f64> {
        weight: TensorRef {
            data: dev_w.as_slice(),
            shape: [v as i32, d as i32],
            stride: contiguous_stride([v as i32, d as i32]),
        },
        indices: TensorRef {
            data: dev_idx.as_slice(),
            shape: [total_indices as i32],
            stride: contiguous_stride([total_indices as i32]),
        },
        offsets: TensorRef {
            data: dev_off.as_slice(),
            shape: [num_bags as i32],
            stride: contiguous_stride([num_bags as i32]),
        },
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [num_bags as i32, d as i32],
            stride: contiguous_stride([num_bags as i32, d as i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; num_bags * d];
    dev_out.copy_to_host(&mut got).expect("dl");
    let eps = f64::EPSILON;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let tol = 16.0 * eps * e.abs().max(1.0);
        assert!(
            (g - e).abs() <= tol,
            "embedding_bag f64 sum mismatch @ {i}: got {g} expected {e}"
        );
    }
}
