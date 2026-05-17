//! Real-GPU dtype-fanout smoke tests for `RepeatPlan<T, 2>` across
//! `{f16, bf16, f64}`. The `f32` cell is covered by the trailblazer
//! file `repeat_smoke.rs`.
//!
//! Repeat is a pure copy + modular coord transform — no arithmetic — so
//! the comparison is **bit-exact** via `to_bits()` against a CPU
//! reference that does the same coord walk.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test repeat_dtype_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, RepeatArgs, RepeatDescriptor, RepeatPlan,
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

const INPUT_SHAPE_2D: [i32; 2] = [2, 3];
const REPEATS_2D: [i32; 2] = [3, 2];

fn output_shape_2d() -> [i32; 2] {
    [
        INPUT_SHAPE_2D[0] * REPEATS_2D[0],
        INPUT_SHAPE_2D[1] * REPEATS_2D[1],
    ]
}

/// Generic CPU repeat reference: for each output coord, input coord is
/// `c[d] % input_shape[d]`. `T: Copy`.
fn cpu_repeat_2d<T: Copy>(x: &[T], input_shape: [i32; 2], output_shape: [i32; 2]) -> Vec<T> {
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let mut out = Vec::with_capacity(out_numel);
    for i in 0..output_shape[0] {
        for j in 0..output_shape[1] {
            let in_i = i % input_shape[0];
            let in_j = j % input_shape[1];
            out.push(x[(in_i * input_shape[1] + in_j) as usize]);
        }
    }
    out
}

#[test]
#[ignore]
fn repeat_2d_f16() {
    let (ctx, stream) = setup();
    let input_shape = INPUT_SHAPE_2D;
    let output_shape = output_shape_2d();
    let in_numel = (input_shape[0] * input_shape[1]) as usize;
    let host_x: Vec<f16> = (0..in_numel)
        .map(|i| f16::from_f32((i as f32) * 0.5 - 1.5))
        .collect();
    let expected = cpu_repeat_2d::<f16>(&host_x, input_shape, output_shape);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc y");

    let desc = RepeatDescriptor {
        input_shape,
        repeats: REPEATS_2D,
        element: ElementKind::F16,
    };
    let plan =
        RepeatPlan::<f16, 2>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = RepeatArgs::<f16, 2> {
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

    let mut got = vec![f16::from_f32(0.0); out_numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "repeat 2d f16 mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn repeat_2d_bf16() {
    let (ctx, stream) = setup();
    let input_shape = INPUT_SHAPE_2D;
    let output_shape = output_shape_2d();
    let in_numel = (input_shape[0] * input_shape[1]) as usize;
    let host_x: Vec<bf16> = (0..in_numel)
        .map(|i| bf16::from_f32((i as f32) * 0.25 - 0.75))
        .collect();
    let expected = cpu_repeat_2d::<bf16>(&host_x, input_shape, output_shape);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc y");

    let desc = RepeatDescriptor {
        input_shape,
        repeats: REPEATS_2D,
        element: ElementKind::Bf16,
    };
    let plan =
        RepeatPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = RepeatArgs::<bf16, 2> {
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

    let mut got = vec![bf16::from_f32(0.0); out_numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "repeat 2d bf16 mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn repeat_2d_f64() {
    let (ctx, stream) = setup();
    let input_shape = INPUT_SHAPE_2D;
    let output_shape = output_shape_2d();
    let in_numel = (input_shape[0] * input_shape[1]) as usize;
    let host_x: Vec<f64> = (0..in_numel).map(|i| (i as f64) * 0.125 - 0.5).collect();
    let expected = cpu_repeat_2d::<f64>(&host_x, input_shape, output_shape);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc y");

    let desc = RepeatDescriptor {
        input_shape,
        repeats: REPEATS_2D,
        element: ElementKind::F64,
    };
    let plan =
        RepeatPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = RepeatArgs::<f64, 2> {
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

    let mut got = vec![0f64; out_numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "repeat 2d f64 mismatch @ {i}: got {g} expected {e}"
        );
    }
}
