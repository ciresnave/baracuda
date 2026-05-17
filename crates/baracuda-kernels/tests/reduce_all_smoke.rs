//! Real-GPU smoke tests for `BoolReducePlan<T, N>` — `all` along one
//! axis. Output dtype is always `Bool` (u8 storage, 0 = false, 1 =
//! true). One test per input dtype:
//! `{f32, f16, bf16, f64, i32, i64, Bool}`.
//!
//! Bit-exact comparison — All is integer-style AND of `(x != 0)`
//! predicates, no FP math.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, Bool, BoolReduceArgs, BoolReduceDescriptor, BoolReducePlan, ElementKind,
    PlanPreference, ReduceKind, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// Build a `(rows=3, cols=32)` layout:
//   row 0: all non-zero (all → true)
//   row 1: one zero hole at col 5 (all → false)
//   row 2: all zeros (all → false)
fn all_pattern_expected() -> [Bool; 3] {
    [Bool::new(true), Bool::new(false), Bool::new(false)]
}

fn fill_pattern_f32(host: &mut Vec<f32>) {
    let rows = 3usize;
    let cols = 32usize;
    host.clear();
    host.resize(rows * cols, 0.0);
    // row 0: all non-zero.
    for c in 0..cols {
        host[0 * cols + c] = (c as f32) + 1.0;
    }
    // row 1: all non-zero except col 5.
    for c in 0..cols {
        host[1 * cols + c] = (c as f32) + 1.0;
    }
    host[1 * cols + 5] = 0.0;
    // row 2: all zeros (default).
}

fn run_all<T>(
    ctx: &Context,
    stream: &Stream,
    host_x: &[T],
    element: ElementKind,
) -> Vec<u8>
where
    T: baracuda_kernels::Element,
{
    let input_shape = [3i32, 32];
    let output_shape = [3i32, 1];
    let dev_x = DeviceBuffer::from_slice(ctx, host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<Bool> = DeviceBuffer::zeros(ctx, 3).expect("alloc y");

    let desc = BoolReduceDescriptor {
        kind: ReduceKind::All,
        input_shape,
        reduce_axis: 1,
        element,
    };
    let plan = BoolReducePlan::<T, 2>::select(stream, &desc, PlanPreference::default())
        .expect("select All");
    let args = BoolReduceArgs::<T, 2> {
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
    plan.run(stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got_bool = vec![Bool::new(false); 3];
    dev_y.copy_to_host(&mut got_bool).expect("download");
    got_bool.iter().map(|b| b.0).collect()
}

fn assert_expected(got: &[u8]) {
    let expected = all_pattern_expected();
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(*g, e.0, "row {i}: expected {} got {}", e.0, g);
    }
}

#[test]
#[ignore]
fn all_f32() {
    let (ctx, stream) = setup();
    let mut host = Vec::<f32>::new();
    fill_pattern_f32(&mut host);
    let got = run_all::<f32>(&ctx, &stream, &host, ElementKind::F32);
    assert_expected(&got);
}

#[test]
#[ignore]
fn all_f16() {
    let (ctx, stream) = setup();
    let mut host_f32 = Vec::<f32>::new();
    fill_pattern_f32(&mut host_f32);
    let host: Vec<f16> = host_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let got = run_all::<f16>(&ctx, &stream, &host, ElementKind::F16);
    assert_expected(&got);
}

#[test]
#[ignore]
fn all_bf16() {
    let (ctx, stream) = setup();
    let mut host_f32 = Vec::<f32>::new();
    fill_pattern_f32(&mut host_f32);
    let host: Vec<bf16> = host_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let got = run_all::<bf16>(&ctx, &stream, &host, ElementKind::Bf16);
    assert_expected(&got);
}

#[test]
#[ignore]
fn all_f64() {
    let (ctx, stream) = setup();
    let mut host_f32 = Vec::<f32>::new();
    fill_pattern_f32(&mut host_f32);
    let host: Vec<f64> = host_f32.iter().map(|&v| v as f64).collect();
    let got = run_all::<f64>(&ctx, &stream, &host, ElementKind::F64);
    assert_expected(&got);
}

#[test]
#[ignore]
fn all_i32() {
    let (ctx, stream) = setup();
    let mut host_f32 = Vec::<f32>::new();
    fill_pattern_f32(&mut host_f32);
    let host: Vec<i32> = host_f32.iter().map(|&v| v as i32).collect();
    let got = run_all::<i32>(&ctx, &stream, &host, ElementKind::I32);
    assert_expected(&got);
}

#[test]
#[ignore]
fn all_i64() {
    let (ctx, stream) = setup();
    let mut host_f32 = Vec::<f32>::new();
    fill_pattern_f32(&mut host_f32);
    let host: Vec<i64> = host_f32.iter().map(|&v| v as i64).collect();
    let got = run_all::<i64>(&ctx, &stream, &host, ElementKind::I64);
    assert_expected(&got);
}

#[test]
#[ignore]
fn all_bool() {
    let (ctx, stream) = setup();
    let mut host_f32 = Vec::<f32>::new();
    fill_pattern_f32(&mut host_f32);
    let host: Vec<Bool> = host_f32
        .iter()
        .map(|&v| Bool::new(v != 0.0))
        .collect();
    let got = run_all::<Bool>(&ctx, &stream, &host, ElementKind::Bool);
    assert_expected(&got);
}
