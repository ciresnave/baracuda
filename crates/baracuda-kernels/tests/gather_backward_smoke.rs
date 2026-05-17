//! Real-GPU smoke test for `GatherBackwardPlan<T, N>` (Phase 7 7.3).
//!
//! Sweeps gradients via atomicAdd — verify the per-cell accumulation
//! lands at the right `(src-coord-along-gather-dim)` cells with the
//! right summed values. Uses `8 * eps`-relative tolerance.
//!
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, GatherBackwardArgs, GatherBackwardDescriptor,
    GatherBackwardPlan, PlanPreference, TensorMut, TensorRef, Workspace,
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
fn gather_backward_f32_2d_dim1() {
    let (ctx, stream) = setup();
    let dsrc_shape = [4i32, 5];
    let out_shape = [4i32, 3];
    let dsrc_numel: usize = 4 * 5;
    let out_numel: usize = 4 * 3;
    // Index includes duplicates so atomicAdd accumulates.
    let host_idx: Vec<i32> = vec![0, 0, 2,  4, 4, 4,  1, 3, 1,  2, 2, 0];
    let host_dout: Vec<f32> = (0..out_numel).map(|i| (i as f32) * 0.5 + 1.0).collect();
    let mut expected = vec![0f32; dsrc_numel];
    for i in 0..4usize {
        for j in 0..3usize {
            let k = host_idx[i * 3 + j] as usize;
            expected[i * 5 + k] += host_dout[i * 3 + j];
        }
    }
    let dev_dout = DeviceBuffer::from_slice(&ctx, &host_dout).expect("up dout");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let mut dev_dsrc: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, dsrc_numel).expect("alloc dsrc");

    let desc = GatherBackwardDescriptor {
        out_shape,
        gather_dim: 1,
        src_dim_size: 5,
        element: ElementKind::F32,
    };
    let plan = GatherBackwardPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = GatherBackwardArgs::<f32, 2> {
        dout: TensorRef {
            data: dev_dout.as_slice(),
            shape: out_shape,
            stride: contiguous_stride(out_shape),
        },
        index: TensorRef {
            data: dev_idx.as_slice(),
            shape: out_shape,
            stride: contiguous_stride(out_shape),
        },
        dsrc: TensorMut {
            data: dev_dsrc.as_slice_mut(),
            shape: dsrc_shape,
            stride: contiguous_stride(dsrc_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; dsrc_numel];
    dev_dsrc.copy_to_host(&mut got).expect("dl");
    let eps = f32::EPSILON;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let tol = 8.0 * eps * e.abs().max(1.0);
        assert!(
            (g - e).abs() <= tol,
            "gather_backward f32 mismatch @ {i}: got {g} expected {e} tol {tol}"
        );
    }
}

#[test]
#[ignore]
fn gather_backward_f64_2d_dim0() {
    let (ctx, stream) = setup();
    let dsrc_shape = [5i32, 3];
    let out_shape = [4i32, 3];
    let dsrc_numel: usize = 5 * 3;
    let out_numel: usize = 4 * 3;
    let host_idx: Vec<i32> = vec![0, 1, 2,  3, 3, 3,  0, 0, 4,  2, 4, 1];
    let host_dout: Vec<f64> = (0..out_numel).map(|i| (i as f64) * 0.125 - 1.5).collect();
    let mut expected = vec![0f64; dsrc_numel];
    for i in 0..4usize {
        for j in 0..3usize {
            let row = host_idx[i * 3 + j] as usize;
            expected[row * 3 + j] += host_dout[i * 3 + j];
        }
    }
    let dev_dout = DeviceBuffer::from_slice(&ctx, &host_dout).expect("up dout");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let mut dev_dsrc: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, dsrc_numel).expect("alloc dsrc");

    let desc = GatherBackwardDescriptor {
        out_shape,
        gather_dim: 0,
        src_dim_size: 5,
        element: ElementKind::F64,
    };
    let plan = GatherBackwardPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = GatherBackwardArgs::<f64, 2> {
        dout: TensorRef {
            data: dev_dout.as_slice(),
            shape: out_shape,
            stride: contiguous_stride(out_shape),
        },
        index: TensorRef {
            data: dev_idx.as_slice(),
            shape: out_shape,
            stride: contiguous_stride(out_shape),
        },
        dsrc: TensorMut {
            data: dev_dsrc.as_slice_mut(),
            shape: dsrc_shape,
            stride: contiguous_stride(dsrc_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; dsrc_numel];
    dev_dsrc.copy_to_host(&mut got).expect("dl");
    let eps = f64::EPSILON;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let tol = 8.0 * eps * e.abs().max(1.0);
        assert!(
            (g - e).abs() <= tol,
            "gather_backward f64 mismatch @ {i}: got {g} expected {e} tol {tol}"
        );
    }
}
