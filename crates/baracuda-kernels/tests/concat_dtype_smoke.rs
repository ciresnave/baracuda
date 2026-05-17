//! Real-GPU smoke tests for `ConcatPlan<{f16,bf16,f64}, N>` — dtype
//! fanout of the f32 trailblazer in `concat_smoke.rs`.
//!
//! Concat does no math — bit-exact compare via `to_bits()`. Each test
//! mirrors `concat_3d_axis_1` from the f32 file (concat along the
//! middle axis of a rank-3 tensor).
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test concat_dtype_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ConcatArgs, ConcatDescriptor, ConcatPlan, ElementKind, PlanPreference,
    TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Generic CPU concat (rank-3, contig inputs / output). T: Copy + Default.
fn cpu_concat_3d<T: Copy + Default>(
    a: &[T],
    a_shape: [i32; 3],
    b: &[T],
    b_shape: [i32; 3],
    concat_dim: usize,
) -> (Vec<T>, [i32; 3]) {
    let mut output_shape = a_shape;
    output_shape[concat_dim] = a_shape[concat_dim] + b_shape[concat_dim];
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let mut out = vec![T::default(); out_numel];

    let a_strides = contiguous_stride(a_shape);
    let b_strides = contiguous_stride(b_shape);
    let out_strides = contiguous_stride(output_shape);
    let split = a_shape[concat_dim];

    for i in 0..out_numel {
        let mut linear = i as i64;
        let mut coord = [0i64; 3];
        for d in (0..3).rev() {
            let s = output_shape[d] as i64;
            coord[d] = linear % s;
            linear /= s;
        }
        let mut out_off: i64 = 0;
        for d in 0..3 {
            out_off += coord[d] * out_strides[d];
        }
        if coord[concat_dim] < split as i64 {
            let mut a_off: i64 = 0;
            for d in 0..3 {
                a_off += coord[d] * a_strides[d];
            }
            out[out_off as usize] = a[a_off as usize];
        } else {
            let mut b_coord = coord;
            b_coord[concat_dim] -= split as i64;
            let mut b_off: i64 = 0;
            for d in 0..3 {
                b_off += b_coord[d] * b_strides[d];
            }
            out[out_off as usize] = b[b_off as usize];
        }
    }
    (out, output_shape)
}

#[test]
#[ignore]
fn concat_3d_axis_1_f16() {
    let (ctx, stream) = setup();
    let a_shape = [4i32, 5, 12];
    let b_shape = [4i32, 7, 12];
    let concat_dim: usize = 1;

    let a_numel: usize = a_shape.iter().map(|&d| d as usize).product();
    let b_numel: usize = b_shape.iter().map(|&d| d as usize).product();
    // Inputs in [-10, 10] to stay safely representable in f16.
    let host_a: Vec<f16> = (0..a_numel)
        .map(|i| f16::from_f32(((i % 41) as f32) * 0.5 - 10.0))
        .collect();
    let host_b: Vec<f16> = (0..b_numel)
        .map(|i| f16::from_f32(((i % 37) as f32) * 0.25 - 4.5))
        .collect();
    let (expected, output_shape) =
        cpu_concat_3d::<f16>(&host_a, a_shape, &host_b, b_shape, concat_dim);

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc y");

    let desc = ConcatDescriptor {
        a_shape,
        b_shape,
        concat_dim: concat_dim as u8,
        element: ElementKind::F16,
    };
    let plan = ConcatPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select f16");
    let args = ConcatArgs::<f16, 3> {
        a: TensorRef {
            data: dev_a.as_slice(),
            shape: a_shape,
            stride: contiguous_stride(a_shape),
        },
        b: TensorRef {
            data: dev_b.as_slice(),
            shape: b_shape,
            stride: contiguous_stride(b_shape),
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: output_shape,
            stride: contiguous_stride(output_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run f16");
    stream.synchronize().expect("sync");

    let mut got = vec![f16::from_f32(0.0); out_numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "concat f16 mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn concat_3d_axis_1_bf16() {
    let (ctx, stream) = setup();
    let a_shape = [4i32, 5, 12];
    let b_shape = [4i32, 7, 12];
    let concat_dim: usize = 1;

    let a_numel: usize = a_shape.iter().map(|&d| d as usize).product();
    let b_numel: usize = b_shape.iter().map(|&d| d as usize).product();
    let host_a: Vec<bf16> = (0..a_numel)
        .map(|i| bf16::from_f32(((i % 41) as f32) * 0.5 - 10.0))
        .collect();
    let host_b: Vec<bf16> = (0..b_numel)
        .map(|i| bf16::from_f32(((i % 37) as f32) * 0.25 - 4.5))
        .collect();
    let (expected, output_shape) =
        cpu_concat_3d::<bf16>(&host_a, a_shape, &host_b, b_shape, concat_dim);

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc y");

    let desc = ConcatDescriptor {
        a_shape,
        b_shape,
        concat_dim: concat_dim as u8,
        element: ElementKind::Bf16,
    };
    let plan = ConcatPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select bf16");
    let args = ConcatArgs::<bf16, 3> {
        a: TensorRef {
            data: dev_a.as_slice(),
            shape: a_shape,
            stride: contiguous_stride(a_shape),
        },
        b: TensorRef {
            data: dev_b.as_slice(),
            shape: b_shape,
            stride: contiguous_stride(b_shape),
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: output_shape,
            stride: contiguous_stride(output_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run bf16");
    stream.synchronize().expect("sync");

    let mut got = vec![bf16::from_f32(0.0); out_numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "concat bf16 mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn concat_3d_axis_1_f64() {
    let (ctx, stream) = setup();
    let a_shape = [4i32, 5, 12];
    let b_shape = [4i32, 7, 12];
    let concat_dim: usize = 1;

    let a_numel: usize = a_shape.iter().map(|&d| d as usize).product();
    let b_numel: usize = b_shape.iter().map(|&d| d as usize).product();
    let host_a: Vec<f64> = (0..a_numel).map(|i| (i as f64) * 0.125 - 1.0).collect();
    let host_b: Vec<f64> = (0..b_numel).map(|i| (i as f64) * 0.0625 + 100.0).collect();
    let (expected, output_shape) =
        cpu_concat_3d::<f64>(&host_a, a_shape, &host_b, b_shape, concat_dim);

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc y");

    let desc = ConcatDescriptor {
        a_shape,
        b_shape,
        concat_dim: concat_dim as u8,
        element: ElementKind::F64,
    };
    let plan = ConcatPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select f64");
    let args = ConcatArgs::<f64, 3> {
        a: TensorRef {
            data: dev_a.as_slice(),
            shape: a_shape,
            stride: contiguous_stride(a_shape),
        },
        b: TensorRef {
            data: dev_b.as_slice(),
            shape: b_shape,
            stride: contiguous_stride(b_shape),
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: output_shape,
            stride: contiguous_stride(output_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run f64");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; out_numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "concat f64 mismatch @ {i}: got {g} expected {e}"
        );
    }
}
