//! Real-GPU smoke test for `ScatterPlan<T, N>` (Phase 39 / Fuel 6c.4 Gap 5).
//!
//! Verifies the pure-assign scatter (no accumulation; last writer wins
//! on duplicate-target races). Tests use **disjoint target indices** so
//! the output is deterministic.
//!
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, ScatterArgs, ScatterDescriptor, ScatterPlan,
    TensorMut, TensorRef, Workspace,
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
fn scatter_f32_2d_dim1_disjoint() {
    let (ctx, stream) = setup();
    let out_shape = [3i32, 6];
    let upd_shape = [3i32, 4];
    let out_numel: usize = 3 * 6;
    let upd_numel: usize = 3 * 4;
    // Disjoint per-row: each row's idx values are unique within the row
    // (and the scatter axis is `dim=1`, so cross-row collisions don't
    // matter — different rows write to different cells anyway).
    let host_idx: Vec<i32> = vec![0, 1, 2, 5,  3, 0, 4, 1,  5, 4, 0, 2];
    let host_upd: Vec<f32> = (0..upd_numel).map(|i| (i as f32) * 0.5 + 1.0).collect();
    let host_out_init: Vec<f32> = (0..out_numel).map(|i| (i as f32) * 0.1).collect();
    // Reference: copy init, then for each (i,j) overwrite expected[i, idx[i,j]] = upd[i,j].
    let mut expected = host_out_init.clone();
    for i in 0..3usize {
        for j in 0..4usize {
            let k = host_idx[i * 4 + j] as usize;
            expected[i * 6 + k] = host_upd[i * 4 + j];
        }
    }
    let dev_upd = DeviceBuffer::from_slice(&ctx, &host_upd).expect("up upd");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let mut dev_out = DeviceBuffer::from_slice(&ctx, &host_out_init).expect("up out");

    let desc = ScatterDescriptor {
        upd_shape,
        scatter_dim: 1,
        out_dim_size: 6,
        element: ElementKind::F32,
    };
    let plan = ScatterPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = ScatterArgs::<f32, 2> {
        updates: TensorRef {
            data: dev_upd.as_slice(),
            shape: upd_shape,
            stride: contiguous_stride(upd_shape),
        },
        index: TensorRef {
            data: dev_idx.as_slice(),
            shape: upd_shape,
            stride: contiguous_stride(upd_shape),
        },
        out: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: out_shape,
            stride: contiguous_stride(out_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; out_numel];
    dev_out.copy_to_host(&mut got).expect("dl");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        // Pure copy: bit-exact.
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "scatter f32 mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn scatter_f64_2d_dim0_disjoint() {
    let (ctx, stream) = setup();
    let out_shape = [6i32, 3];
    let upd_shape = [4i32, 3];
    let out_numel: usize = 6 * 3;
    let upd_numel: usize = 4 * 3;
    // Per-column disjoint along dim=0: each column's 4 index entries pick
    // 4 distinct rows from {0..6}.
    let host_idx: Vec<i32> = vec![0, 1, 2,  1, 2, 3,  3, 4, 4,  5, 0, 5];
    let host_upd: Vec<f64> = (0..upd_numel).map(|i| (i as f64) * 0.25 + 2.0).collect();
    let host_out_init: Vec<f64> = (0..out_numel).map(|i| -(i as f64)).collect();
    let mut expected = host_out_init.clone();
    for i in 0..4usize {
        for j in 0..3usize {
            let row = host_idx[i * 3 + j] as usize;
            expected[row * 3 + j] = host_upd[i * 3 + j];
        }
    }
    let dev_upd = DeviceBuffer::from_slice(&ctx, &host_upd).expect("up upd");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let mut dev_out = DeviceBuffer::from_slice(&ctx, &host_out_init).expect("up out");

    let desc = ScatterDescriptor {
        upd_shape,
        scatter_dim: 0,
        out_dim_size: 6,
        element: ElementKind::F64,
    };
    let plan = ScatterPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = ScatterArgs::<f64, 2> {
        updates: TensorRef {
            data: dev_upd.as_slice(),
            shape: upd_shape,
            stride: contiguous_stride(upd_shape),
        },
        index: TensorRef {
            data: dev_idx.as_slice(),
            shape: upd_shape,
            stride: contiguous_stride(upd_shape),
        },
        out: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: out_shape,
            stride: contiguous_stride(out_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; out_numel];
    dev_out.copy_to_host(&mut got).expect("dl");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "scatter f64 mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn scatter_f32_2d_dim1_i64idx() {
    let (ctx, stream) = setup();
    let out_shape = [2i32, 5];
    let upd_shape = [2i32, 3];
    let out_numel: usize = 2 * 5;
    let upd_numel: usize = 2 * 3;
    let host_idx: Vec<i64> = vec![0, 2, 4,  1, 3, 4];
    let host_upd: Vec<f32> = (0..upd_numel).map(|i| (i as f32) + 10.0).collect();
    let host_out_init: Vec<f32> = vec![0.0; out_numel];
    let mut expected = host_out_init.clone();
    for i in 0..2usize {
        for j in 0..3usize {
            let k = host_idx[i * 3 + j] as usize;
            expected[i * 5 + k] = host_upd[i * 3 + j];
        }
    }
    let dev_upd = DeviceBuffer::from_slice(&ctx, &host_upd).expect("up upd");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let mut dev_out = DeviceBuffer::from_slice(&ctx, &host_out_init).expect("up out");

    let desc = ScatterDescriptor {
        upd_shape,
        scatter_dim: 1,
        out_dim_size: 5,
        element: ElementKind::F32,
    };
    let plan = ScatterPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = ScatterArgs::<f32, 2, i64> {
        updates: TensorRef {
            data: dev_upd.as_slice(),
            shape: upd_shape,
            stride: contiguous_stride(upd_shape),
        },
        index: TensorRef {
            data: dev_idx.as_slice(),
            shape: upd_shape,
            stride: contiguous_stride(upd_shape),
        },
        out: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: out_shape,
            stride: contiguous_stride(out_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; out_numel];
    dev_out.copy_to_host(&mut got).expect("dl");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "scatter f32/i64idx mismatch @ {i}: got {g} expected {e}"
        );
    }
}
