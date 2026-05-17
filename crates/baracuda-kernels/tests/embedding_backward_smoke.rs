//! Real-GPU smoke test for `EmbeddingBackwardPlan<T>` (Phase 7 7.5).
//!
//! `dweight[indices[n], :] += dout[n, :]` along axis 0 (atomicAdd),
//! skipping rows where `indices[n] == padding_idx` (or OOB).
//!
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, EmbeddingBackwardArgs, EmbeddingBackwardDescriptor,
    EmbeddingBackwardPlan, PlanPreference, TensorMut, TensorRef, Workspace,
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
fn embedding_backward_f32_duplicate_indices() {
    let (ctx, stream) = setup();
    let v: usize = 5;
    let d: usize = 4;
    let n: usize = 4;
    // 1 appears twice → atomicAdd accumulation; -1 is OOB → skip.
    let host_idx: Vec<i32> = vec![1, 3, 1, -1];
    let host_dout: Vec<f32> = (0..n * d).map(|i| (i as f32) * 0.5 + 1.0).collect();
    let mut expected = vec![0f32; v * d];
    for (i, &idx) in host_idx.iter().enumerate() {
        if idx < 0 || idx >= v as i32 {
            continue;
        }
        for j in 0..d {
            expected[idx as usize * d + j] += host_dout[i * d + j];
        }
    }
    let dev_dout = DeviceBuffer::from_slice(&ctx, &host_dout).expect("up dout");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let mut dev_dw: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, v * d).expect("alloc dweight");

    let desc = EmbeddingBackwardDescriptor {
        num_embeddings: v as i32,
        embedding_dim: d as i32,
        num_indices: n as i32,
        padding_idx: None,
        element: ElementKind::F32,
    };
    let plan = EmbeddingBackwardPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = EmbeddingBackwardArgs::<f32> {
        dout: TensorRef {
            data: dev_dout.as_slice(),
            shape: [n as i32, d as i32],
            stride: contiguous_stride([n as i32, d as i32]),
        },
        indices: TensorRef {
            data: dev_idx.as_slice(),
            shape: [n as i32],
            stride: contiguous_stride([n as i32]),
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
        let tol = 8.0 * eps * e.abs().max(1.0);
        assert!(
            (g - e).abs() <= tol,
            "embedding_backward f32 mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn embedding_backward_f32_padding_idx_skipped() {
    let (ctx, stream) = setup();
    let v: usize = 4;
    let d: usize = 3;
    let n: usize = 5;
    let padding = 2i32;
    // 2 (padding_idx) appears twice → no contribution.
    let host_idx: Vec<i32> = vec![0, 2, 3, 2, 1];
    let host_dout: Vec<f32> = (0..n * d).map(|i| (i as f32) * 0.25 + 0.5).collect();
    let mut expected = vec![0f32; v * d];
    for (i, &idx) in host_idx.iter().enumerate() {
        if idx == padding || idx < 0 || idx >= v as i32 {
            continue;
        }
        for j in 0..d {
            expected[idx as usize * d + j] += host_dout[i * d + j];
        }
    }
    let dev_dout = DeviceBuffer::from_slice(&ctx, &host_dout).expect("up dout");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let mut dev_dw: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, v * d).expect("alloc dweight");

    let desc = EmbeddingBackwardDescriptor {
        num_embeddings: v as i32,
        embedding_dim: d as i32,
        num_indices: n as i32,
        padding_idx: Some(padding),
        element: ElementKind::F32,
    };
    let plan = EmbeddingBackwardPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = EmbeddingBackwardArgs::<f32> {
        dout: TensorRef {
            data: dev_dout.as_slice(),
            shape: [n as i32, d as i32],
            stride: contiguous_stride([n as i32, d as i32]),
        },
        indices: TensorRef {
            data: dev_idx.as_slice(),
            shape: [n as i32],
            stride: contiguous_stride([n as i32]),
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
        let tol = 8.0 * eps * e.abs().max(1.0);
        assert!(
            (g - e).abs() <= tol,
            "embedding_backward f32 padding mismatch @ {i}: got {g} expected {e}"
        );
    }
    // Row 2 (padding_idx) must remain exact zero.
    for j in 0..d {
        assert_eq!(
            got[padding as usize * d + j].to_bits(),
            0u32,
            "padding row {j} should be untouched"
        );
    }
}

#[test]
#[ignore]
fn embedding_backward_f64_basic() {
    let (ctx, stream) = setup();
    let v: usize = 4;
    let d: usize = 3;
    let n: usize = 5;
    let host_idx: Vec<i32> = vec![0, 3, 0, 2, 3];
    let host_dout: Vec<f64> = (0..n * d).map(|i| (i as f64) * 0.125 - 0.5).collect();
    let mut expected = vec![0f64; v * d];
    for (i, &idx) in host_idx.iter().enumerate() {
        if idx < 0 || idx >= v as i32 {
            continue;
        }
        for j in 0..d {
            expected[idx as usize * d + j] += host_dout[i * d + j];
        }
    }
    let dev_dout = DeviceBuffer::from_slice(&ctx, &host_dout).expect("up dout");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let mut dev_dw: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, v * d).expect("alloc dweight");

    let desc = EmbeddingBackwardDescriptor {
        num_embeddings: v as i32,
        embedding_dim: d as i32,
        num_indices: n as i32,
        padding_idx: None,
        element: ElementKind::F64,
    };
    let plan = EmbeddingBackwardPlan::<f64>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = EmbeddingBackwardArgs::<f64> {
        dout: TensorRef {
            data: dev_dout.as_slice(),
            shape: [n as i32, d as i32],
            stride: contiguous_stride([n as i32, d as i32]),
        },
        indices: TensorRef {
            data: dev_idx.as_slice(),
            shape: [n as i32],
            stride: contiguous_stride([n as i32]),
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
        let tol = 8.0 * eps * e.abs().max(1.0);
        assert!(
            (g - e).abs() <= tol,
            "embedding_backward f64 mismatch @ {i}: got {g} expected {e}"
        );
    }
}
