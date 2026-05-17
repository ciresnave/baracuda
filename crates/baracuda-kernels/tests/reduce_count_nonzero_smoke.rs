//! Real-GPU smoke tests for `CountReducePlan<T, N>` — `count_nonzero`
//! along one axis. Output dtype is always `i64` (PyTorch convention).
//! One test per input dtype: `{f32, f16, bf16, f64, i32, i64, Bool}`.
//!
//! Bit-exact comparison — pure integer accumulation of `(x != 0 ? 1 :
//! 0)`, no FP math.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, Bool, CountReduceArgs, CountReduceDescriptor, CountReducePlan, ElementKind,
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

// Build a `(rows=3, cols=32)` layout with known non-zero counts:
//   row 0: all zeros → count 0.
//   row 1: 7 non-zero values at columns {0, 3, 7, 11, 19, 25, 30} → count 7.
//   row 2: all 32 non-zero (col+1) → count 32.
fn count_pattern_expected() -> [i64; 3] {
    [0, 7, 32]
}

fn fill_pattern_f32(host: &mut Vec<f32>) {
    let rows = 3usize;
    let cols = 32usize;
    host.clear();
    host.resize(rows * cols, 0.0);
    // row 0: all zeros.
    // row 1: 7 non-zero.
    for &c in &[0usize, 3, 7, 11, 19, 25, 30] {
        host[1 * cols + c] = (c as f32) + 1.0;
    }
    // row 2: all non-zero.
    for c in 0..cols {
        host[2 * cols + c] = (c as f32) + 1.0;
    }
}

fn run_count<T>(
    ctx: &Context,
    stream: &Stream,
    host_x: &[T],
    element: ElementKind,
) -> Vec<i64>
where
    T: baracuda_kernels::Element,
{
    let input_shape = [3i32, 32];
    let output_shape = [3i32, 1];
    let dev_x = DeviceBuffer::from_slice(ctx, host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<i64> = DeviceBuffer::zeros(ctx, 3).expect("alloc y");

    let desc = CountReduceDescriptor {
        kind: ReduceKind::CountNonzero,
        input_shape,
        reduce_axis: 1,
        element,
    };
    let plan = CountReducePlan::<T, 2>::select(stream, &desc, PlanPreference::default())
        .expect("select CountNonzero");
    let args = CountReduceArgs::<T, 2> {
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

    let mut got = vec![0i64; 3];
    dev_y.copy_to_host(&mut got).expect("download");
    got
}

fn assert_expected(got: &[i64]) {
    let expected = count_pattern_expected();
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(*g, *e, "row {i}: expected {} got {}", e, g);
    }
}

#[test]
#[ignore]
fn count_nonzero_f32() {
    let (ctx, stream) = setup();
    let mut host = Vec::<f32>::new();
    fill_pattern_f32(&mut host);
    let got = run_count::<f32>(&ctx, &stream, &host, ElementKind::F32);
    assert_expected(&got);
}

#[test]
#[ignore]
fn count_nonzero_f16() {
    let (ctx, stream) = setup();
    let mut host_f32 = Vec::<f32>::new();
    fill_pattern_f32(&mut host_f32);
    let host: Vec<f16> = host_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let got = run_count::<f16>(&ctx, &stream, &host, ElementKind::F16);
    assert_expected(&got);
}

#[test]
#[ignore]
fn count_nonzero_bf16() {
    let (ctx, stream) = setup();
    let mut host_f32 = Vec::<f32>::new();
    fill_pattern_f32(&mut host_f32);
    let host: Vec<bf16> = host_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let got = run_count::<bf16>(&ctx, &stream, &host, ElementKind::Bf16);
    assert_expected(&got);
}

#[test]
#[ignore]
fn count_nonzero_f64() {
    let (ctx, stream) = setup();
    let mut host_f32 = Vec::<f32>::new();
    fill_pattern_f32(&mut host_f32);
    let host: Vec<f64> = host_f32.iter().map(|&v| v as f64).collect();
    let got = run_count::<f64>(&ctx, &stream, &host, ElementKind::F64);
    assert_expected(&got);
}

#[test]
#[ignore]
fn count_nonzero_i32() {
    let (ctx, stream) = setup();
    let mut host_f32 = Vec::<f32>::new();
    fill_pattern_f32(&mut host_f32);
    let host: Vec<i32> = host_f32.iter().map(|&v| v as i32).collect();
    let got = run_count::<i32>(&ctx, &stream, &host, ElementKind::I32);
    assert_expected(&got);
}

#[test]
#[ignore]
fn count_nonzero_i64() {
    let (ctx, stream) = setup();
    let mut host_f32 = Vec::<f32>::new();
    fill_pattern_f32(&mut host_f32);
    let host: Vec<i64> = host_f32.iter().map(|&v| v as i64).collect();
    let got = run_count::<i64>(&ctx, &stream, &host, ElementKind::I64);
    assert_expected(&got);
}

#[test]
#[ignore]
fn count_nonzero_bool() {
    let (ctx, stream) = setup();
    let mut host_f32 = Vec::<f32>::new();
    fill_pattern_f32(&mut host_f32);
    let host: Vec<Bool> = host_f32
        .iter()
        .map(|&v| Bool::new(v != 0.0))
        .collect();
    let got = run_count::<Bool>(&ctx, &stream, &host, ElementKind::Bool);
    assert_expected(&got);
}
