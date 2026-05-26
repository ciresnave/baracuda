//! Real-GPU smoke test for `EmbeddingBagMaxPlan<T>` (Phase 25).
//!
//! Per-bag Max-mode FW with per-feature argmax tracking.
//!
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, EmbeddingBagMaxArgs, EmbeddingBagMaxDescriptor,
    EmbeddingBagMaxPlan, PlanPreference, TensorMut, TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// CPU reference. Uses first-occurrence argmax tie-break (matches kernel).
fn cpu_bag_max_f32(
    v: usize,
    d: usize,
    weight: &[f32],
    indices: &[i32],
    offsets: &[i32],
    num_bags: usize,
    total_indices: usize,
    padding_idx: Option<i32>,
) -> (Vec<f32>, Vec<i32>) {
    let mut out = vec![0f32; num_bags * d];
    let mut out_idx = vec![-1i32; num_bags * d];
    for b in 0..num_bags {
        let start = offsets[b] as usize;
        let end = if b + 1 < num_bags {
            offsets[b + 1] as usize
        } else {
            total_indices
        };
        for col in 0..d {
            let mut best: Option<(f32, i32)> = None;
            for k in start..end {
                let idx = indices[k];
                if Some(idx) == padding_idx || idx < 0 || idx >= v as i32 {
                    continue;
                }
                let val = weight[idx as usize * d + col];
                match best {
                    None => best = Some((val, idx)),
                    Some((bv, _)) if val > bv => best = Some((val, idx)),
                    _ => {}
                }
            }
            if let Some((val, idx)) = best {
                out[b * d + col] = val;
                out_idx[b * d + col] = idx;
            } else {
                out[b * d + col] = 0.0;
                out_idx[b * d + col] = -1;
            }
        }
    }
    (out, out_idx)
}

#[test]
#[ignore]
fn embedding_bag_max_f32_basic() {
    let (ctx, stream) = setup();
    let v: usize = 5;
    let d: usize = 4;
    let host_w: Vec<f32> = (0..(v * d)).map(|i| (i as f32) * 0.25 - 1.0).collect();
    // 3 bags: [1, 3], [0, 4, 2], implicit last bag extends to total_indices.
    let host_idx: Vec<i32> = vec![1, 3, 0, 4, 2];
    let host_off: Vec<i32> = vec![0, 2, 5];
    let num_bags = 3usize;
    let total_indices = host_idx.len();
    let (expected, expected_idx) = cpu_bag_max_f32(
        v, d, &host_w, &host_idx, &host_off, num_bags, total_indices, None,
    );

    let dev_w = DeviceBuffer::from_slice(&ctx, &host_w).expect("up weight");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let dev_off = DeviceBuffer::from_slice(&ctx, &host_off).expect("up offsets");
    let mut dev_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, num_bags * d).expect("alloc out");
    let mut dev_oidx: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, num_bags * d).expect("alloc oidx");

    let desc = EmbeddingBagMaxDescriptor {
        num_embeddings: v as i32,
        embedding_dim: d as i32,
        num_bags: num_bags as i32,
        total_indices: total_indices as i32,
        padding_idx: None,
        element: ElementKind::F32,
    };
    let plan = EmbeddingBagMaxPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = EmbeddingBagMaxArgs::<f32> {
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
        output_index: TensorMut {
            data: dev_oidx.as_slice_mut(),
            shape: [num_bags as i32, d as i32],
            stride: contiguous_stride([num_bags as i32, d as i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; num_bags * d];
    let mut got_idx = vec![0i32; num_bags * d];
    dev_out.copy_to_host(&mut got).expect("dl");
    dev_oidx.copy_to_host(&mut got_idx).expect("dl idx");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "embedding_bag_max f32 mismatch @ {i}: got {g} expected {e}"
        );
    }
    for (i, (g, e)) in got_idx.iter().zip(expected_idx.iter()).enumerate() {
        assert_eq!(g, e, "embedding_bag_max f32 out_index mismatch @ {i}: got {g} expected {e}");
    }
}

#[test]
#[ignore]
fn embedding_bag_max_f32_padding_and_empty() {
    let (ctx, stream) = setup();
    let v: usize = 5;
    let d: usize = 3;
    let host_w: Vec<f32> = (0..(v * d)).map(|i| (i as f32) * 0.5 - 2.0).collect();
    let padding = 4i32;
    // 3 bags: [1, 4, 2] (4 padded), [] (empty → all -1), final bag = [4, 4] (all padded).
    let host_idx: Vec<i32> = vec![1, 4, 2, 4, 4];
    let host_off: Vec<i32> = vec![0, 3, 3];
    let num_bags = host_off.len();
    let total_indices = host_idx.len();
    let (expected, expected_idx) = cpu_bag_max_f32(
        v, d, &host_w, &host_idx, &host_off, num_bags, total_indices, Some(padding),
    );

    let dev_w = DeviceBuffer::from_slice(&ctx, &host_w).expect("up weight");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let dev_off = DeviceBuffer::from_slice(&ctx, &host_off).expect("up offsets");
    let mut dev_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, num_bags * d).expect("alloc out");
    let mut dev_oidx: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, num_bags * d).expect("alloc oidx");

    let desc = EmbeddingBagMaxDescriptor {
        num_embeddings: v as i32,
        embedding_dim: d as i32,
        num_bags: num_bags as i32,
        total_indices: total_indices as i32,
        padding_idx: Some(padding),
        element: ElementKind::F32,
    };
    let plan = EmbeddingBagMaxPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = EmbeddingBagMaxArgs::<f32> {
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
        output_index: TensorMut {
            data: dev_oidx.as_slice_mut(),
            shape: [num_bags as i32, d as i32],
            stride: contiguous_stride([num_bags as i32, d as i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; num_bags * d];
    let mut got_idx = vec![0i32; num_bags * d];
    dev_out.copy_to_host(&mut got).expect("dl");
    dev_oidx.copy_to_host(&mut got_idx).expect("dl idx");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "embedding_bag_max f32 padding mismatch @ {i}: got {g} expected {e}"
        );
    }
    for (i, (g, e)) in got_idx.iter().zip(expected_idx.iter()).enumerate() {
        assert_eq!(g, e, "embedding_bag_max f32 padding out_index mismatch @ {i}: got {g} expected {e}");
    }
}

#[test]
#[ignore]
fn embedding_bag_max_f64() {
    let (ctx, stream) = setup();
    let v: usize = 4;
    let d: usize = 3;
    let host_w: Vec<f64> = (0..(v * d)).map(|i| (i as f64) * 0.125 - 1.0).collect();
    let host_idx: Vec<i32> = vec![0, 2, 3, 1];
    let host_off: Vec<i32> = vec![0, 2];
    let num_bags = host_off.len();
    let total_indices = host_idx.len();

    // CPU ref (f64 path).
    let mut expected = vec![0f64; num_bags * d];
    let mut expected_idx = vec![-1i32; num_bags * d];
    for b in 0..num_bags {
        let start = host_off[b] as usize;
        let end = if b + 1 < num_bags {
            host_off[b + 1] as usize
        } else {
            total_indices
        };
        for col in 0..d {
            let mut best: Option<(f64, i32)> = None;
            for k in start..end {
                let idx = host_idx[k];
                let val = host_w[idx as usize * d + col];
                match best {
                    None => best = Some((val, idx)),
                    Some((bv, _)) if val > bv => best = Some((val, idx)),
                    _ => {}
                }
            }
            if let Some((val, idx)) = best {
                expected[b * d + col] = val;
                expected_idx[b * d + col] = idx;
            }
        }
    }

    let dev_w = DeviceBuffer::from_slice(&ctx, &host_w).expect("up weight");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let dev_off = DeviceBuffer::from_slice(&ctx, &host_off).expect("up offsets");
    let mut dev_out: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, num_bags * d).expect("alloc out");
    let mut dev_oidx: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, num_bags * d).expect("alloc oidx");

    let desc = EmbeddingBagMaxDescriptor {
        num_embeddings: v as i32,
        embedding_dim: d as i32,
        num_bags: num_bags as i32,
        total_indices: total_indices as i32,
        padding_idx: None,
        element: ElementKind::F64,
    };
    let plan = EmbeddingBagMaxPlan::<f64>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = EmbeddingBagMaxArgs::<f64> {
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
        output_index: TensorMut {
            data: dev_oidx.as_slice_mut(),
            shape: [num_bags as i32, d as i32],
            stride: contiguous_stride([num_bags as i32, d as i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; num_bags * d];
    let mut got_idx = vec![0i32; num_bags * d];
    dev_out.copy_to_host(&mut got).expect("dl");
    dev_oidx.copy_to_host(&mut got_idx).expect("dl idx");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "embedding_bag_max f64 mismatch @ {i}: got {g} expected {e}");
    }
    for (i, (g, e)) in got_idx.iter().zip(expected_idx.iter()).enumerate() {
        assert_eq!(g, e, "embedding_bag_max f64 out_index mismatch @ {i}");
    }
}
