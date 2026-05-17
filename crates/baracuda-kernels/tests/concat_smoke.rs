//! Real-GPU smoke test for `ConcatPlan<f32, N>` — 2-input concat.
//!
//! Covers concat along each axis of a 3D tensor (rank-3) plus a 1D
//! edge case. Bit-exact compare — concat does no math, just element
//! copy.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test concat_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ConcatArgs, ConcatDescriptor, ConcatPlan, ElementKind, PlanPreference,
    TensorMut, TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// CPU reference: walk every output cell, branch on concat-axis coord
/// to pick a or b, copy the appropriate cell.
fn cpu_concat_3d(
    a: &[f32],
    a_shape: [i32; 3],
    b: &[f32],
    b_shape: [i32; 3],
    concat_dim: usize,
) -> (Vec<f32>, [i32; 3]) {
    let mut output_shape = a_shape;
    output_shape[concat_dim] = a_shape[concat_dim] + b_shape[concat_dim];
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let mut out = vec![0f32; out_numel];

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

fn run_case_3d(concat_dim: usize) {
    let (ctx, stream) = setup();
    // a and b differ in the concat dim.
    let mut a_shape = [4i32, 8, 12];
    let mut b_shape = [4i32, 8, 12];
    a_shape[concat_dim] = 5;
    b_shape[concat_dim] = 7;

    let a_numel: usize = a_shape.iter().map(|&d| d as usize).product();
    let b_numel: usize = b_shape.iter().map(|&d| d as usize).product();

    let host_a: Vec<f32> = (0..a_numel).map(|i| (i as f32) * 0.125 - 1.0).collect();
    let host_b: Vec<f32> = (0..b_numel).map(|i| (i as f32) * 0.0625 + 100.0).collect();
    let (expected, output_shape) = cpu_concat_3d(&host_a, a_shape, &host_b, b_shape, concat_dim);

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc y");

    let desc = ConcatDescriptor {
        a_shape,
        b_shape,
        concat_dim: concat_dim as u8,
        element: ElementKind::F32,
    };
    let plan = ConcatPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = ConcatArgs::<f32, 3> {
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
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; out_numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "concat dim={concat_dim} mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn concat_3d_axis_0() {
    run_case_3d(0);
}

#[test]
#[ignore]
fn concat_3d_axis_1() {
    run_case_3d(1);
}

#[test]
#[ignore]
fn concat_3d_axis_2() {
    run_case_3d(2);
}

/// 1D concat — simplest case, just appends b after a.
#[test]
#[ignore]
fn concat_1d() {
    let (ctx, stream) = setup();
    let a_shape = [128i32];
    let b_shape = [192i32];

    let host_a: Vec<f32> = (0..128).map(|i| (i as f32) * 0.5).collect();
    let host_b: Vec<f32> = (0..192).map(|i| (i as f32) * 0.25 + 1000.0).collect();
    let mut expected: Vec<f32> = Vec::with_capacity(320);
    expected.extend(&host_a);
    expected.extend(&host_b);
    let output_shape = [320i32];

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 320).expect("alloc y");

    let desc = ConcatDescriptor {
        a_shape,
        b_shape,
        concat_dim: 0,
        element: ElementKind::F32,
    };
    let plan = ConcatPlan::<f32, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = ConcatArgs::<f32, 1> {
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
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; 320];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "concat 1d mismatch @ {i}");
    }
}

/// Residual-style 2D concat — `[B, H_a] cat [B, H_b]` along the H axis.
/// Most common pattern in real ML code.
#[test]
#[ignore]
fn concat_2d_feature_dim() {
    let (ctx, stream) = setup();
    let a_shape = [16i32, 64];
    let b_shape = [16i32, 32];
    let output_shape = [16i32, 96];

    let a_numel = (16 * 64) as usize;
    let b_numel = (16 * 32) as usize;
    let host_a: Vec<f32> = (0..a_numel).map(|i| (i as f32) * 0.5).collect();
    let host_b: Vec<f32> = (0..b_numel).map(|i| (i as f32) * 0.25 + 5000.0).collect();

    // Reference: for row i, output[i, 0..64] = a[i, :]; output[i, 64..96] = b[i, :]
    let mut expected = vec![0f32; 16 * 96];
    for i in 0..16 {
        for j in 0..64 {
            expected[i * 96 + j] = host_a[i * 64 + j];
        }
        for j in 0..32 {
            expected[i * 96 + 64 + j] = host_b[i * 32 + j];
        }
    }

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 16 * 96).expect("alloc y");

    let desc = ConcatDescriptor {
        a_shape,
        b_shape,
        concat_dim: 1,
        element: ElementKind::F32,
    };
    let plan = ConcatPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = ConcatArgs::<f32, 2> {
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
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; 16 * 96];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "concat 2d feature mismatch @ {i}");
    }
}
