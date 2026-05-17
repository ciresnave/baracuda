//! Real-GPU smoke test for `IndexSelectPlan<T, N>` (Phase 7 7.3).
//!
//! `out[..., j, ...] = src[..., idx[j], ...]` along `select_dim` with
//! a 1-D i32 idx. f32, f64, i32 coverage. Bit-exact (no arithmetic).
//!
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, IndexSelectArgs, IndexSelectDescriptor, IndexSelectPlan,
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
fn index_select_f32_2d_dim0() {
    let (ctx, stream) = setup();
    let src_shape = [5i32, 4];
    let out_shape = [3i32, 4];
    let src_numel: usize = 5 * 4;
    let out_numel: usize = 3 * 4;
    let host_src: Vec<f32> = (0..src_numel).map(|i| i as f32 * 0.25 + 1.0).collect();
    let host_idx: Vec<i32> = vec![4, 1, 2];
    let mut expected = vec![0f32; out_numel];
    for i in 0..3usize {
        for j in 0..4usize {
            let row = host_idx[i] as usize;
            expected[i * 4 + j] = host_src[row * 4 + j];
        }
    }
    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("up src");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let mut dev_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, out_numel).expect("alloc out");

    let desc = IndexSelectDescriptor {
        out_shape,
        select_dim: 0,
        src_dim_size: 5,
        element: ElementKind::F32,
    };
    let plan = IndexSelectPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = IndexSelectArgs::<f32, 2> {
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
        assert_eq!(g.to_bits(), e.to_bits(), "index_select f32 mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn index_select_f64_3d_dim2() {
    let (ctx, stream) = setup();
    let src_shape = [2i32, 3, 6];
    let out_shape = [2i32, 3, 4];
    let src_numel: usize = 2 * 3 * 6;
    let out_numel: usize = 2 * 3 * 4;
    let host_src: Vec<f64> = (0..src_numel).map(|i| i as f64 * 0.125 - 2.0).collect();
    let host_idx: Vec<i32> = vec![1, 5, 2, 0];
    let mut expected = vec![0f64; out_numel];
    for b in 0..2usize {
        for c in 0..3usize {
            for j in 0..4usize {
                let k = host_idx[j] as usize;
                expected[(b * 3 + c) * 4 + j] = host_src[(b * 3 + c) * 6 + k];
            }
        }
    }
    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("up src");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let mut dev_out: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, out_numel).expect("alloc out");

    let desc = IndexSelectDescriptor {
        out_shape,
        select_dim: 2,
        src_dim_size: 6,
        element: ElementKind::F64,
    };
    let plan = IndexSelectPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = IndexSelectArgs::<f64, 3> {
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
        assert_eq!(g.to_bits(), e.to_bits(), "index_select f64 mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn index_select_i32_2d_dim1() {
    let (ctx, stream) = setup();
    let src_shape = [3i32, 5];
    let out_shape = [3i32, 4];
    let src_numel: usize = 3 * 5;
    let out_numel: usize = 3 * 4;
    let host_src: Vec<i32> = (0..src_numel as i32).map(|i| i.wrapping_mul(17) - 50).collect();
    let host_idx: Vec<i32> = vec![3, 0, 4, 1];
    let mut expected = vec![0i32; out_numel];
    for i in 0..3usize {
        for j in 0..4usize {
            let k = host_idx[j] as usize;
            expected[i * 4 + j] = host_src[i * 5 + k];
        }
    }
    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("up src");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let mut dev_out: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, out_numel).expect("alloc out");

    let desc = IndexSelectDescriptor {
        out_shape,
        select_dim: 1,
        src_dim_size: 5,
        element: ElementKind::I32,
    };
    let plan = IndexSelectPlan::<i32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = IndexSelectArgs::<i32, 2> {
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
        out: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: out_shape,
            stride: contiguous_stride(out_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0i32; out_numel];
    dev_out.copy_to_host(&mut got).expect("dl");
    assert_eq!(got, expected, "index_select i32 mismatch");
}
