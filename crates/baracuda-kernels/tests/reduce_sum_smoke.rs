//! Real-GPU smoke test for the Phase 4 reduction trailblazer
//! (`ReducePlan<f32, N> + ReduceKind::Sum`).
//!
//! Covers reduction along each axis of a 3D tensor (rank-3) at a
//! moderate shape `[4, 16, 32]`. The naive trailblazer kernel sums
//! sequentially per output cell (one thread, loop over reduce axis),
//! producing deterministic in-order f32 addition. Host reference does
//! the same in-order addition, so compare is bit-exact.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test reduce_sum_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, ReduceArgs, ReduceDescriptor, ReduceKind,
    ReducePlan, TensorMut, TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// CPU reference: sum-along-axis with keepdim convention. Output coord
/// has [reduce_axis] = 0; for each, sum over the input's reduce-axis
/// extent in input order.
fn cpu_reduce_sum_3d(
    x: &[f32],
    input_shape: [i32; 3],
    reduce_axis: usize,
) -> (Vec<f32>, [i32; 3]) {
    let mut output_shape = input_shape;
    output_shape[reduce_axis] = 1;
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let mut out = vec![0f32; out_numel];

    let in_strides = contiguous_stride(input_shape);
    let out_strides = contiguous_stride(output_shape);

    // Walk every output cell.
    for i in 0..out_numel {
        let mut linear = i as i64;
        let mut coord = [0i64; 3];
        for d in (0..3).rev() {
            let s = output_shape[d] as i64;
            coord[d] = linear % s;
            linear /= s;
        }
        // Sum over the reduce axis of the input (other coords match).
        let mut acc = 0f32;
        let reduce_extent = input_shape[reduce_axis];
        for k in 0..reduce_extent {
            let mut in_coord = coord;
            in_coord[reduce_axis] = k as i64;
            let mut in_off: i64 = 0;
            for d in 0..3 {
                in_off += in_coord[d] * in_strides[d];
            }
            acc += x[in_off as usize];
        }
        let mut out_off: i64 = 0;
        for d in 0..3 {
            out_off += coord[d] * out_strides[d];
        }
        out[out_off as usize] = acc;
    }
    (out, output_shape)
}

fn run_case(reduce_axis: usize) {
    let (ctx, stream) = setup();
    let input_shape = [4i32, 16, 32];
    let in_numel: usize = input_shape.iter().map(|&d| d as usize).product();

    // Deterministic input.
    let host_x: Vec<f32> = (0..in_numel)
        .map(|i| (i as f32) * 0.0625 - 50.0)
        .collect();
    let (expected, output_shape) = cpu_reduce_sum_3d(&host_x, input_shape, reduce_axis);
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc y");

    let desc = ReduceDescriptor {
        kind: ReduceKind::Sum,
        input_shape,
        reduce_axis: reduce_axis as u8,
        element: ElementKind::F32,
        correction: 1,
    };
    let plan = ReducePlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = ReduceArgs::<f32, 3> {
        x: TensorRef {
            data: dev_x.as_slice(),
            shape: input_shape,
            stride: contiguous_stride(input_shape),
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
            "reduce sum axis={reduce_axis} mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn reduce_sum_axis_0() {
    run_case(0);
}

#[test]
#[ignore]
fn reduce_sum_axis_1() {
    run_case(1);
}

#[test]
#[ignore]
fn reduce_sum_axis_2() {
    run_case(2);
}

/// Trivial 1D reduction → output is rank-1 with shape [1].
#[test]
#[ignore]
fn reduce_sum_1d() {
    let (ctx, stream) = setup();
    let input_shape = [256i32];
    let host_x: Vec<f32> = (0..256).map(|i| (i as f32) * 0.5 - 50.0).collect();
    let mut expected = 0f32;
    for v in &host_x {
        expected += v;
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc y");

    let output_shape = [1i32];
    let desc = ReduceDescriptor {
        kind: ReduceKind::Sum,
        input_shape,
        reduce_axis: 0,
        element: ElementKind::F32,
        correction: 1,
    };
    let plan = ReducePlan::<f32, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = ReduceArgs::<f32, 1> {
        x: TensorRef {
            data: dev_x.as_slice(),
            shape: input_shape,
            stride: contiguous_stride(input_shape),
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: output_shape,
            stride: contiguous_stride(output_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; 1];
    dev_y.copy_to_host(&mut got).expect("download");
    assert_eq!(got[0].to_bits(), expected.to_bits(),
        "reduce sum 1d mismatch: got {} expected {}", got[0], expected);
}
