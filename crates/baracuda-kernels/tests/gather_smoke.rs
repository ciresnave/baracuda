//! Real-GPU smoke test for `GatherPlan<T, N>` (Phase 7 Milestone 7.3).
//!
//! Covers contig 2-D and 3-D fixtures across `f32, f64, i32`. The
//! kernel does no arithmetic (pure load + store) so output is bit-exact
//! against the CPU reference.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test gather_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, GatherArgs, GatherDescriptor, GatherPlan, PlanPreference,
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
fn gather_f32_2d_dim1() {
    let (ctx, stream) = setup();
    let src_shape = [4i32, 5];
    let out_shape = [4i32, 3];
    let src_numel: usize = 4 * 5;
    let out_numel: usize = 4 * 3;
    let host_src: Vec<f32> = (0..src_numel).map(|i| i as f32 * 0.5 + 1.0).collect();
    let host_idx: Vec<i32> = vec![0, 4, 2,  1, 0, 3,  2, 4, 4,  3, 3, 1];
    // CPU reference: out[i, j] = src[i, idx[i, j]]
    let mut expected = vec![0f32; out_numel];
    for i in 0..4usize {
        for j in 0..3usize {
            let k = host_idx[i * 3 + j] as usize;
            expected[i * 3 + j] = host_src[i * 5 + k];
        }
    }
    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("up src");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let mut dev_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, out_numel).expect("alloc out");

    let desc = GatherDescriptor {
        out_shape,
        gather_dim: 1,
        src_dim_size: 5,
        element: ElementKind::F32,
    };
    let plan = GatherPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = GatherArgs::<f32, 2> {
        src: TensorRef {
            data: dev_src.as_slice(),
            shape: src_shape,
            stride: contiguous_stride(src_shape),
        },
        index: TensorRef {
            data: dev_idx.as_slice(),
            shape: out_shape,
            stride: contiguous_stride(out_shape),
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
        assert_eq!(g.to_bits(), e.to_bits(), "gather f32 mismatch @ {i}: got {g} expected {e}");
    }
}

#[test]
#[ignore]
fn gather_f64_3d_dim0() {
    let (ctx, stream) = setup();
    let src_shape = [6i32, 3, 4];
    let out_shape = [2i32, 3, 4];
    let src_numel: usize = 6 * 3 * 4;
    let out_numel: usize = 2 * 3 * 4;
    let host_src: Vec<f64> = (0..src_numel).map(|i| i as f64 * 0.25 - 5.0).collect();
    // Index along dim 0 — every output cell carries an integer in [0, 6).
    let host_idx: Vec<i32> = (0..out_numel).map(|i| (i % 6) as i32).collect();
    let mut expected = vec![0f64; out_numel];
    for i in 0..2usize {
        for j in 0..3usize {
            for k in 0..4usize {
                let out_off = (i * 3 + j) * 4 + k;
                let src_i = host_idx[out_off] as usize;
                expected[out_off] = host_src[(src_i * 3 + j) * 4 + k];
            }
        }
    }
    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("up src");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let mut dev_out: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, out_numel).expect("alloc out");

    let desc = GatherDescriptor {
        out_shape,
        gather_dim: 0,
        src_dim_size: 6,
        element: ElementKind::F64,
    };
    let plan = GatherPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = GatherArgs::<f64, 3> {
        src: TensorRef {
            data: dev_src.as_slice(),
            shape: src_shape,
            stride: contiguous_stride(src_shape),
        },
        index: TensorRef {
            data: dev_idx.as_slice(),
            shape: out_shape,
            stride: contiguous_stride(out_shape),
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
        assert_eq!(g.to_bits(), e.to_bits(), "gather f64 mismatch @ {i}: got {g} expected {e}");
    }
}

#[test]
#[ignore]
fn gather_i32_2d_dim1() {
    let (ctx, stream) = setup();
    let src_shape = [3i32, 4];
    let out_shape = [3i32, 5];
    let src_numel: usize = 3 * 4;
    let out_numel: usize = 3 * 5;
    let host_src: Vec<i32> = (0..src_numel as i32).map(|i| i.wrapping_mul(31) - 100).collect();
    let host_idx: Vec<i32> = vec![0, 1, 2, 3, 0,  3, 2, 1, 0, 2,  1, 1, 1, 2, 3];
    let mut expected = vec![0i32; out_numel];
    for i in 0..3usize {
        for j in 0..5usize {
            let k = host_idx[i * 5 + j] as usize;
            expected[i * 5 + j] = host_src[i * 4 + k];
        }
    }
    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("up src");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let mut dev_out: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, out_numel).expect("alloc out");

    let desc = GatherDescriptor {
        out_shape,
        gather_dim: 1,
        src_dim_size: 4,
        element: ElementKind::I32,
    };
    let plan = GatherPlan::<i32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = GatherArgs::<i32, 2> {
        src: TensorRef {
            data: dev_src.as_slice(),
            shape: src_shape,
            stride: contiguous_stride(src_shape),
        },
        index: TensorRef {
            data: dev_idx.as_slice(),
            shape: out_shape,
            stride: contiguous_stride(out_shape),
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
    assert_eq!(got, expected, "gather i32 mismatch");
}
