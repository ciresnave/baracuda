//! Real-GPU smoke test for `IndexSelectBackwardPlan<T, N>` (Phase 7 7.3).
//!
//! `dsrc[..., idx[j], ...] += dout[..., j, ...]` along `select_dim`
//! (atomicAdd). Verifies the scatter-add accumulation pattern with
//! 8·eps relative tolerance.
//!
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, IndexSelectBackwardArgs, IndexSelectBackwardDescriptor,
    IndexSelectBackwardPlan, PlanPreference, TensorMut, TensorRef, Workspace,
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
fn index_select_backward_f32_2d_dim0() {
    let (ctx, stream) = setup();
    let dsrc_shape = [5i32, 4];
    let out_shape = [3i32, 4];
    let dsrc_numel: usize = 5 * 4;
    let out_numel: usize = 3 * 4;
    let host_idx: Vec<i32> = vec![4, 1, 1]; // 1 appears twice → atomicAdd
    let host_dout: Vec<f32> = (0..out_numel).map(|i| (i as f32) * 0.5 + 1.0).collect();
    let mut expected = vec![0f32; dsrc_numel];
    for i in 0..3usize {
        for j in 0..4usize {
            let row = host_idx[i] as usize;
            expected[row * 4 + j] += host_dout[i * 4 + j];
        }
    }
    let dev_dout = DeviceBuffer::from_slice(&ctx, &host_dout).expect("up dout");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let mut dev_dsrc: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, dsrc_numel).expect("alloc dsrc");

    let desc = IndexSelectBackwardDescriptor {
        out_shape,
        select_dim: 0,
        src_dim_size: 5,
        element: ElementKind::F32,
    };
    let plan = IndexSelectBackwardPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = IndexSelectBackwardArgs::<f32, 2> {
        dout: TensorRef {
            data: dev_dout.as_slice(),
            shape: out_shape,
            stride: contiguous_stride(out_shape),
        },
        idx: TensorRef {
            data: dev_idx.as_slice(),
            shape: [3i32],
            stride: contiguous_stride([3i32]),
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
            "index_select_backward f32 mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn index_select_backward_f64_2d_dim1() {
    let (ctx, stream) = setup();
    let dsrc_shape = [3i32, 6];
    let out_shape = [3i32, 4];
    let dsrc_numel: usize = 3 * 6;
    let out_numel: usize = 3 * 4;
    let host_idx: Vec<i32> = vec![5, 2, 2, 0];
    let host_dout: Vec<f64> = (0..out_numel).map(|i| (i as f64) * 0.25 - 1.5).collect();
    let mut expected = vec![0f64; dsrc_numel];
    for i in 0..3usize {
        for j in 0..4usize {
            let col = host_idx[j] as usize;
            expected[i * 6 + col] += host_dout[i * 4 + j];
        }
    }
    let dev_dout = DeviceBuffer::from_slice(&ctx, &host_dout).expect("up dout");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let mut dev_dsrc: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, dsrc_numel).expect("alloc dsrc");

    let desc = IndexSelectBackwardDescriptor {
        out_shape,
        select_dim: 1,
        src_dim_size: 6,
        element: ElementKind::F64,
    };
    let plan = IndexSelectBackwardPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = IndexSelectBackwardArgs::<f64, 2> {
        dout: TensorRef {
            data: dev_dout.as_slice(),
            shape: out_shape,
            stride: contiguous_stride(out_shape),
        },
        idx: TensorRef {
            data: dev_idx.as_slice(),
            shape: [4i32],
            stride: contiguous_stride([4i32]),
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
            "index_select_backward f64 mismatch @ {i}: got {g} expected {e}"
        );
    }
}
