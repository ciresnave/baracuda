//! Real-GPU smoke test for `IndexAddPlan<T, N>` (Phase 39 / Fuel 6c.4 Gap 5).
//!
//! Verifies `dst[idx[i], ...] += src[i, ...]` along an axis using
//! atomicAdd-Σ. Includes a **duplicate-target** test that exercises the
//! atomic-accumulation path (multiple `i` mapping to the same `dst`
//! row) — this is what makes the op valuable over a pure assign.
//!
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, IndexAddArgs, IndexAddDescriptor, IndexAddPlan,
    PlanPreference, TensorMut, TensorRef, Workspace,
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
fn index_add_f32_2d_dim0() {
    let (ctx, stream) = setup();
    // dst: [5, 4], src: [3, 4], idx 1-D length 3.
    let src_shape = [3i32, 4];
    let dst_shape = [5i32, 4];
    let src_numel: usize = 3 * 4;
    let dst_numel: usize = 5 * 4;
    // Index entries: row 1, row 3, row 1 again — duplicate target on dst[1].
    let host_idx: Vec<i32> = vec![1, 3, 1];
    let host_src: Vec<f32> = (0..src_numel).map(|i| (i as f32) + 1.0).collect();
    let host_dst_init: Vec<f32> = vec![0.0; dst_numel];

    // Reference: dst[idx[i], j] += src[i, j].
    let mut expected = host_dst_init.clone();
    for i in 0..3usize {
        let row = host_idx[i] as usize;
        for j in 0..4usize {
            expected[row * 4 + j] += host_src[i * 4 + j];
        }
    }

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("up src");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let mut dev_dst = DeviceBuffer::from_slice(&ctx, &host_dst_init).expect("up dst");

    let desc = IndexAddDescriptor {
        src_shape,
        add_dim: 0,
        dst_dim_size: 5,
        element: ElementKind::F32,
    };
    let plan = IndexAddPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = IndexAddArgs::<f32, 2> {
        src: TensorRef {
            data: dev_src.as_slice(),
            shape: src_shape,
            stride: contiguous_stride(src_shape),
        },
        idx: TensorRef {
            data: dev_idx.as_slice(),
            shape: [3i32],
            stride: contiguous_stride([3i32]),
        },
        dst: TensorMut {
            data: dev_dst.as_slice_mut(),
            shape: dst_shape,
            stride: contiguous_stride(dst_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; dst_numel];
    dev_dst.copy_to_host(&mut got).expect("dl");
    let eps = f32::EPSILON;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let tol = 8.0 * eps * e.abs().max(1.0);
        assert!(
            (g - e).abs() <= tol,
            "index_add f32 mismatch @ {i}: got {g} expected {e} tol {tol}"
        );
    }
}

#[test]
#[ignore]
fn index_add_f64_2d_dim0_duplicate_target() {
    let (ctx, stream) = setup();
    // Heavier duplicate-target test: 5 src rows, idx maps {0, 1, 0, 2, 0},
    // so dst[0] should accumulate src[0] + src[2] + src[4].
    let src_shape = [5i32, 3];
    let dst_shape = [3i32, 3];
    let src_numel: usize = 5 * 3;
    let dst_numel: usize = 3 * 3;
    let host_idx: Vec<i32> = vec![0, 1, 0, 2, 0];
    let host_src: Vec<f64> = (0..src_numel).map(|i| (i as f64) * 0.5 + 1.0).collect();
    let host_dst_init: Vec<f64> = (0..dst_numel).map(|i| (i as f64) * 0.1).collect();

    let mut expected = host_dst_init.clone();
    for i in 0..5usize {
        let row = host_idx[i] as usize;
        for j in 0..3usize {
            expected[row * 3 + j] += host_src[i * 3 + j];
        }
    }

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("up src");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let mut dev_dst = DeviceBuffer::from_slice(&ctx, &host_dst_init).expect("up dst");

    let desc = IndexAddDescriptor {
        src_shape,
        add_dim: 0,
        dst_dim_size: 3,
        element: ElementKind::F64,
    };
    let plan = IndexAddPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = IndexAddArgs::<f64, 2> {
        src: TensorRef {
            data: dev_src.as_slice(),
            shape: src_shape,
            stride: contiguous_stride(src_shape),
        },
        idx: TensorRef {
            data: dev_idx.as_slice(),
            shape: [5i32],
            stride: contiguous_stride([5i32]),
        },
        dst: TensorMut {
            data: dev_dst.as_slice_mut(),
            shape: dst_shape,
            stride: contiguous_stride(dst_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; dst_numel];
    dev_dst.copy_to_host(&mut got).expect("dl");
    let eps = f64::EPSILON;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let tol = 16.0 * eps * e.abs().max(1.0);
        assert!(
            (g - e).abs() <= tol,
            "index_add f64 dup-target mismatch @ {i}: got {g} expected {e} tol {tol}"
        );
    }
}

#[test]
#[ignore]
fn index_add_f32_i64idx_2d_dim0() {
    let (ctx, stream) = setup();
    let src_shape = [4i32, 3];
    let dst_shape = [4i32, 3];
    let src_numel: usize = 4 * 3;
    let dst_numel: usize = 4 * 3;
    let host_idx: Vec<i64> = vec![3, 2, 1, 0]; // reversed permutation
    let host_src: Vec<f32> = (0..src_numel).map(|i| (i as f32) + 1.0).collect();
    let host_dst_init: Vec<f32> = vec![0.0; dst_numel];

    let mut expected = host_dst_init.clone();
    for i in 0..4usize {
        let row = host_idx[i] as usize;
        for j in 0..3usize {
            expected[row * 3 + j] += host_src[i * 3 + j];
        }
    }

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("up src");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let mut dev_dst = DeviceBuffer::from_slice(&ctx, &host_dst_init).expect("up dst");

    let desc = IndexAddDescriptor {
        src_shape,
        add_dim: 0,
        dst_dim_size: 4,
        element: ElementKind::F32,
    };
    let plan = IndexAddPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = IndexAddArgs::<f32, 2, i64> {
        src: TensorRef {
            data: dev_src.as_slice(),
            shape: src_shape,
            stride: contiguous_stride(src_shape),
        },
        idx: TensorRef {
            data: dev_idx.as_slice(),
            shape: [4i32],
            stride: contiguous_stride([4i32]),
        },
        dst: TensorMut {
            data: dev_dst.as_slice_mut(),
            shape: dst_shape,
            stride: contiguous_stride(dst_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; dst_numel];
    dev_dst.copy_to_host(&mut got).expect("dl");
    let eps = f32::EPSILON;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let tol = 8.0 * eps * e.abs().max(1.0);
        assert!(
            (g - e).abs() <= tol,
            "index_add f32/i64idx mismatch @ {i}: got {g} expected {e} tol {tol}"
        );
    }
}
