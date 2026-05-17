//! Real-GPU smoke test for `ScatterAddPlan<T, N>` (Phase 7 7.3).
//!
//! Verifies atomic accumulation into the destination tensor along the
//! scatter dim. Duplicate indices in the update tensor exercise the
//! `atomicAdd` dup-safe path.
//!
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, ScatterAddArgs, ScatterAddDescriptor,
    ScatterAddPlan, TensorMut, TensorRef, Workspace,
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
fn scatter_add_f32_2d_dim1() {
    let (ctx, stream) = setup();
    let out_shape = [3i32, 6];
    let upd_shape = [3i32, 4];
    let out_numel: usize = 3 * 6;
    let upd_numel: usize = 3 * 4;
    let host_idx: Vec<i32> = vec![0, 1, 1, 5,  2, 0, 3, 4,  5, 5, 5, 0];
    let host_upd: Vec<f32> = (0..upd_numel).map(|i| (i as f32) * 0.5 + 1.0).collect();
    // Pre-populated `out` with non-zero initial state.
    let host_out_init: Vec<f32> = (0..out_numel).map(|i| (i as f32) * 0.1).collect();
    let mut expected = host_out_init.clone();
    for i in 0..3usize {
        for j in 0..4usize {
            let k = host_idx[i * 4 + j] as usize;
            expected[i * 6 + k] += host_upd[i * 4 + j];
        }
    }
    let dev_upd = DeviceBuffer::from_slice(&ctx, &host_upd).expect("up upd");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let mut dev_out = DeviceBuffer::from_slice(&ctx, &host_out_init).expect("up out");

    let desc = ScatterAddDescriptor {
        upd_shape,
        scatter_dim: 1,
        out_dim_size: 6,
        element: ElementKind::F32,
    };
    let plan = ScatterAddPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = ScatterAddArgs::<f32, 2> {
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
    let eps = f32::EPSILON;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let tol = 8.0 * eps * e.abs().max(1.0);
        assert!(
            (g - e).abs() <= tol,
            "scatter_add f32 mismatch @ {i}: got {g} expected {e} tol {tol}"
        );
    }
}

#[test]
#[ignore]
fn scatter_add_f64_2d_dim0() {
    let (ctx, stream) = setup();
    let out_shape = [5i32, 3];
    let upd_shape = [4i32, 3];
    let out_numel: usize = 5 * 3;
    let upd_numel: usize = 4 * 3;
    let host_idx: Vec<i32> = vec![0, 0, 0,  1, 2, 3,  4, 1, 2,  3, 3, 4];
    let host_upd: Vec<f64> = (0..upd_numel).map(|i| (i as f64) * 0.25 + 2.0).collect();
    let mut expected = vec![0f64; out_numel]; // pre-zero state
    for i in 0..4usize {
        for j in 0..3usize {
            let row = host_idx[i * 3 + j] as usize;
            expected[row * 3 + j] += host_upd[i * 3 + j];
        }
    }
    let dev_upd = DeviceBuffer::from_slice(&ctx, &host_upd).expect("up upd");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let mut dev_out: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc out");

    let desc = ScatterAddDescriptor {
        upd_shape,
        scatter_dim: 0,
        out_dim_size: 5,
        element: ElementKind::F64,
    };
    let plan = ScatterAddPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = ScatterAddArgs::<f64, 2> {
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
    let eps = f64::EPSILON;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let tol = 8.0 * eps * e.abs().max(1.0);
        assert!(
            (g - e).abs() <= tol,
            "scatter_add f64 mismatch @ {i}: got {g} expected {e} tol {tol}"
        );
    }
}
