//! Real-GPU smoke test for `EmbeddingBagBackwardPlan<T>` (Phase 7 7.5).
//!
//! Sum-mode: `dweight[indices[k], :] += dout[b, :]` for k in bag b.
//! Mean-mode: same, divided by post-skip bag size.
//!
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, EmbeddingBagBackwardArgs, EmbeddingBagBackwardDescriptor,
    EmbeddingBagBackwardPlan, EmbeddingBagMode, PlanPreference, TensorMut, TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn cpu_embedding_bag_backward_f32(
    v: usize,
    d: usize,
    dout: &[f32],
    indices: &[i32],
    offsets: &[i32],
    num_bags: usize,
    total_indices: usize,
    mode: EmbeddingBagMode,
    padding_idx: Option<i32>,
) -> Vec<f32> {
    let mut dw = vec![0f32; v * d];
    for b in 0..num_bags {
        let start = offsets[b] as usize;
        let end = if b + 1 < num_bags {
            offsets[b + 1] as usize
        } else {
            total_indices
        };
        if end <= start {
            continue;
        }
        // Count non-padded / in-bounds indices for the divisor.
        let mut counted = 0usize;
        for k in start..end {
            let idx = indices[k];
            if Some(idx) == padding_idx || idx < 0 || idx >= v as i32 {
                continue;
            }
            counted += 1;
        }
        if counted == 0 {
            continue;
        }
        let divisor = match mode {
            EmbeddingBagMode::Sum => 1f32,
            EmbeddingBagMode::Mean => counted as f32,
        };
        for k in start..end {
            let idx = indices[k];
            if Some(idx) == padding_idx || idx < 0 || idx >= v as i32 {
                continue;
            }
            for j in 0..d {
                dw[idx as usize * d + j] += dout[b * d + j] / divisor;
            }
        }
    }
    dw
}

#[test]
#[ignore]
fn embedding_bag_backward_f32_sum() {
    let (ctx, stream) = setup();
    let v: usize = 5;
    let d: usize = 4;
    let host_idx: Vec<i32> = vec![1, 3, 0, 4, 2];
    let host_off: Vec<i32> = vec![0, 2, 5]; // bags [1,3], [0,4,2]
    let num_bags = host_off.len();
    let total_indices = host_idx.len();
    let host_dout: Vec<f32> = (0..(num_bags * d)).map(|i| (i as f32) * 0.5 + 1.0).collect();
    let expected = cpu_embedding_bag_backward_f32(
        v,
        d,
        &host_dout,
        &host_idx,
        &host_off,
        num_bags,
        total_indices,
        EmbeddingBagMode::Sum,
        None,
    );
    let dev_dout = DeviceBuffer::from_slice(&ctx, &host_dout).expect("up dout");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let dev_off = DeviceBuffer::from_slice(&ctx, &host_off).expect("up offsets");
    let mut dev_dw: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, v * d).expect("alloc dweight");

    let desc = EmbeddingBagBackwardDescriptor {
        num_embeddings: v as i32,
        embedding_dim: d as i32,
        num_bags: num_bags as i32,
        total_indices: total_indices as i32,
        mode: EmbeddingBagMode::Sum,
        padding_idx: None,
        element: ElementKind::F32,
    };
    let plan = EmbeddingBagBackwardPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = EmbeddingBagBackwardArgs::<f32> {
        dout: TensorRef {
            data: dev_dout.as_slice(),
            shape: [num_bags as i32, d as i32],
            stride: contiguous_stride([num_bags as i32, d as i32]),
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
        dweight: TensorMut {
            data: dev_dw.as_slice_mut(),
            shape: [v as i32, d as i32],
            stride: contiguous_stride([v as i32, d as i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; v * d];
    dev_dw.copy_to_host(&mut got).expect("dl");
    let eps = f32::EPSILON;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let tol = 16.0 * eps * e.abs().max(1.0);
        assert!(
            (g - e).abs() <= tol,
            "embedding_bag_backward f32 sum mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn embedding_bag_backward_f32_mean_with_padding() {
    let (ctx, stream) = setup();
    let v: usize = 5;
    let d: usize = 3;
    let padding = 4i32;
    // Bag 0 = [1, 4, 2] -> counted=2 -> divisor=2.
    // Bag 1 = [4]       -> counted=0 -> skip.
    let host_idx: Vec<i32> = vec![1, 4, 2, 4];
    let host_off: Vec<i32> = vec![0, 3];
    let num_bags = host_off.len();
    let total_indices = host_idx.len();
    let host_dout: Vec<f32> = (0..(num_bags * d)).map(|i| (i as f32) + 1.0).collect();
    let expected = cpu_embedding_bag_backward_f32(
        v,
        d,
        &host_dout,
        &host_idx,
        &host_off,
        num_bags,
        total_indices,
        EmbeddingBagMode::Mean,
        Some(padding),
    );
    let dev_dout = DeviceBuffer::from_slice(&ctx, &host_dout).expect("up dout");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let dev_off = DeviceBuffer::from_slice(&ctx, &host_off).expect("up offsets");
    let mut dev_dw: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, v * d).expect("alloc dweight");

    let desc = EmbeddingBagBackwardDescriptor {
        num_embeddings: v as i32,
        embedding_dim: d as i32,
        num_bags: num_bags as i32,
        total_indices: total_indices as i32,
        mode: EmbeddingBagMode::Mean,
        padding_idx: Some(padding),
        element: ElementKind::F32,
    };
    let plan = EmbeddingBagBackwardPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = EmbeddingBagBackwardArgs::<f32> {
        dout: TensorRef {
            data: dev_dout.as_slice(),
            shape: [num_bags as i32, d as i32],
            stride: contiguous_stride([num_bags as i32, d as i32]),
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
        dweight: TensorMut {
            data: dev_dw.as_slice_mut(),
            shape: [v as i32, d as i32],
            stride: contiguous_stride([v as i32, d as i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; v * d];
    dev_dw.copy_to_host(&mut got).expect("dl");
    let eps = f32::EPSILON;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let tol = 32.0 * eps * e.abs().max(1.0);
        assert!(
            (g - e).abs() <= tol,
            "embedding_bag_backward f32 mean mismatch @ {i}: got {g} expected {e}"
        );
    }
    // Padding row (4) must remain zero.
    for j in 0..d {
        assert_eq!(
            got[padding as usize * d + j].to_bits(),
            0u32,
            "padding row must remain zero @ feature {j}"
        );
    }
}

#[test]
#[ignore]
fn embedding_bag_backward_f64_sum() {
    let (ctx, stream) = setup();
    let v: usize = 4;
    let d: usize = 3;
    let host_idx: Vec<i32> = vec![0, 2, 3, 1, 0];
    let host_off: Vec<i32> = vec![0, 2, 4];
    let num_bags = host_off.len();
    let total_indices = host_idx.len();
    let host_dout: Vec<f64> = (0..(num_bags * d)).map(|i| (i as f64) * 0.25 + 1.0).collect();
    let mut expected = vec![0f64; v * d];
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
                expected[idx * d + j] += host_dout[b * d + j];
            }
        }
    }
    let dev_dout = DeviceBuffer::from_slice(&ctx, &host_dout).expect("up dout");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let dev_off = DeviceBuffer::from_slice(&ctx, &host_off).expect("up offsets");
    let mut dev_dw: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, v * d).expect("alloc dweight");

    let desc = EmbeddingBagBackwardDescriptor {
        num_embeddings: v as i32,
        embedding_dim: d as i32,
        num_bags: num_bags as i32,
        total_indices: total_indices as i32,
        mode: EmbeddingBagMode::Sum,
        padding_idx: None,
        element: ElementKind::F64,
    };
    let plan = EmbeddingBagBackwardPlan::<f64>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = EmbeddingBagBackwardArgs::<f64> {
        dout: TensorRef {
            data: dev_dout.as_slice(),
            shape: [num_bags as i32, d as i32],
            stride: contiguous_stride([num_bags as i32, d as i32]),
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
        dweight: TensorMut {
            data: dev_dw.as_slice_mut(),
            shape: [v as i32, d as i32],
            stride: contiguous_stride([v as i32, d as i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; v * d];
    dev_dw.copy_to_host(&mut got).expect("dl");
    let eps = f64::EPSILON;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let tol = 16.0 * eps * e.abs().max(1.0);
        assert!(
            (g - e).abs() <= tol,
            "embedding_bag_backward f64 sum mismatch @ {i}: got {g} expected {e}"
        );
    }
}
