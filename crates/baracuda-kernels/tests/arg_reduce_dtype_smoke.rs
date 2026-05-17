//! Real-GPU smoke tests for `ArgReducePlan<{f16,bf16,f64}, N>` — dtype
//! fanout of the f32 trailblazer in `arg_reduce_smoke.rs`.
//!
//! ArgReduce returns deterministic i64 indices (PyTorch convention,
//! ties broken by first occurrence). Input values are kept in
//! `[-10, 10]` (integers) so they are exactly representable in
//! f16/bf16; that way the argmax/argmin index is rounding-independent.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test arg_reduce_dtype_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ArgReduceArgs, ArgReduceDescriptor, ArgReduceKind, ArgReducePlan,
    ElementKind, PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// Build a length-16 vector with a unique max at `max_idx` (value +9) and
// a unique min at `min_idx` (value -9). Other entries vary in [-7, 7]
// deterministically.
fn build_pattern_f32() -> ([f32; 16], i64, i64) {
    // max at 7, min at 11.
    let max_idx: usize = 7;
    let min_idx: usize = 11;
    let mut v = [0f32; 16];
    for i in 0..16 {
        // values from -7..7 by index, but overridden at max/min.
        v[i] = ((i as i32) - 8) as f32; // -8..7 — keep in [-10, 10]
    }
    v[max_idx] = 9.0;
    v[min_idx] = -9.0;
    (v, max_idx as i64, min_idx as i64)
}

fn run_arg_reduce_f16(kind: ArgReduceKind, host_x_f32: &[f32; 16]) -> i64 {
    let (ctx, stream) = setup();
    let input_shape = [16i32];
    let host_x: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<i64> = DeviceBuffer::zeros(&ctx, 1).expect("alloc y");

    let desc = ArgReduceDescriptor {
        kind,
        input_shape,
        reduce_axis: 0,
        element: ElementKind::F16,
    };
    let plan = ArgReducePlan::<f16, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("select f16");
    let output_shape = [1i32];
    let args = ArgReduceArgs::<f16, 1> {
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
    plan.run(&stream, Workspace::None, args).expect("run f16");
    stream.synchronize().expect("sync");

    let mut got = vec![0i64; 1];
    dev_y.copy_to_host(&mut got).expect("download");
    got[0]
}

fn run_arg_reduce_bf16(kind: ArgReduceKind, host_x_f32: &[f32; 16]) -> i64 {
    let (ctx, stream) = setup();
    let input_shape = [16i32];
    let host_x: Vec<bf16> = host_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<i64> = DeviceBuffer::zeros(&ctx, 1).expect("alloc y");

    let desc = ArgReduceDescriptor {
        kind,
        input_shape,
        reduce_axis: 0,
        element: ElementKind::Bf16,
    };
    let plan = ArgReducePlan::<bf16, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("select bf16");
    let output_shape = [1i32];
    let args = ArgReduceArgs::<bf16, 1> {
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
    plan.run(&stream, Workspace::None, args).expect("run bf16");
    stream.synchronize().expect("sync");

    let mut got = vec![0i64; 1];
    dev_y.copy_to_host(&mut got).expect("download");
    got[0]
}

fn run_arg_reduce_f64(kind: ArgReduceKind, host_x_f32: &[f32; 16]) -> i64 {
    let (ctx, stream) = setup();
    let input_shape = [16i32];
    let host_x: Vec<f64> = host_x_f32.iter().map(|&v| v as f64).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<i64> = DeviceBuffer::zeros(&ctx, 1).expect("alloc y");

    let desc = ArgReduceDescriptor {
        kind,
        input_shape,
        reduce_axis: 0,
        element: ElementKind::F64,
    };
    let plan = ArgReducePlan::<f64, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("select f64");
    let output_shape = [1i32];
    let args = ArgReduceArgs::<f64, 1> {
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
    plan.run(&stream, Workspace::None, args).expect("run f64");
    stream.synchronize().expect("sync");

    let mut got = vec![0i64; 1];
    dev_y.copy_to_host(&mut got).expect("download");
    got[0]
}

#[test]
#[ignore]
fn argmax_1d_f16() {
    let (host_x, max_idx, _min_idx) = build_pattern_f32();
    let got = run_arg_reduce_f16(ArgReduceKind::Argmax, &host_x);
    assert_eq!(got, max_idx, "argmax f16: expected {}, got {}", max_idx, got);
}

#[test]
#[ignore]
fn argmin_1d_f16() {
    let (host_x, _max_idx, min_idx) = build_pattern_f32();
    let got = run_arg_reduce_f16(ArgReduceKind::Argmin, &host_x);
    assert_eq!(got, min_idx, "argmin f16: expected {}, got {}", min_idx, got);
}

#[test]
#[ignore]
fn argmax_1d_bf16() {
    let (host_x, max_idx, _min_idx) = build_pattern_f32();
    let got = run_arg_reduce_bf16(ArgReduceKind::Argmax, &host_x);
    assert_eq!(got, max_idx, "argmax bf16: expected {}, got {}", max_idx, got);
}

#[test]
#[ignore]
fn argmin_1d_bf16() {
    let (host_x, _max_idx, min_idx) = build_pattern_f32();
    let got = run_arg_reduce_bf16(ArgReduceKind::Argmin, &host_x);
    assert_eq!(got, min_idx, "argmin bf16: expected {}, got {}", min_idx, got);
}

#[test]
#[ignore]
fn argmax_1d_f64() {
    let (host_x, max_idx, _min_idx) = build_pattern_f32();
    let got = run_arg_reduce_f64(ArgReduceKind::Argmax, &host_x);
    assert_eq!(got, max_idx, "argmax f64: expected {}, got {}", max_idx, got);
}

#[test]
#[ignore]
fn argmin_1d_f64() {
    let (host_x, _max_idx, min_idx) = build_pattern_f32();
    let got = run_arg_reduce_f64(ArgReduceKind::Argmin, &host_x);
    assert_eq!(got, min_idx, "argmin f64: expected {}, got {}", min_idx, got);
}
