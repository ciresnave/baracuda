//! Real-GPU smoke tests for `TracePlan<T>` across
//! `{f32, f16, bf16, f64}`. Matrix shape `[D, D]` with `D = 32` —
//! single-thread diagonal walk, output is a rank-0 scalar.
//!
//! Tolerance: bit-exact for f64 (small D, single-pass sum), `4 * eps`
//! for f32, and weighted `(4 * eps * D)` for f16/bf16 (32 sequential
//! adds each carry a half-precision rounding).
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test trace_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef, TraceArgs,
    TraceDescriptor, TracePlan, Workspace,
};
use half::{bf16, f16};

const D: i32 = 32;

const F32_EPS: f32 = f32::EPSILON;
const F16_EPS: f32 = 9.77e-4;
const BF16_EPS: f32 = 7.81e-3;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn shape() -> [i32; 2] {
    [D, D]
}

#[test]
#[ignore]
fn trace_f32() {
    let (ctx, stream) = setup();
    let numel = (D as usize) * (D as usize);
    let host_x: Vec<f32> = (0..numel)
        .map(|i| ((i % 17) as f32) * 0.125 - 1.0)
        .collect();

    // Expected: sum of M[i, i] for i in 0..D.
    let strides = contiguous_stride(shape());
    let mut expected = 0f32;
    for i in 0..(D as i64) {
        let off = i * strides[0] + i * strides[1];
        expected += host_x[off as usize];
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc y");

    let desc = TraceDescriptor {
        n: D,
        element: ElementKind::F32,
    };
    let plan =
        TracePlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = TraceArgs::<f32> {
        x: TensorRef {
            data: dev_x.as_slice(),
            shape: shape(),
            stride: strides,
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: [],
            stride: [],
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = [0f32; 1];
    dev_y.copy_to_host(&mut got).expect("download");
    let diff = (got[0] - expected).abs();
    let allow = expected.abs().max(1.0) * (4.0 * F32_EPS * D as f32);
    assert!(
        diff <= allow,
        "trace f32 mismatch: got {} expected {} diff {} allow {}",
        got[0],
        expected,
        diff,
        allow
    );
}

#[test]
#[ignore]
fn trace_f16() {
    let (ctx, stream) = setup();
    let numel = (D as usize) * (D as usize);
    let host_x: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32(((i % 17) as f32) * 0.125 - 1.0))
        .collect();

    // f32-detour accumulator to match the kernel.
    let strides = contiguous_stride(shape());
    let mut expected = 0f32;
    for i in 0..(D as i64) {
        let off = i * strides[0] + i * strides[1];
        expected += host_x[off as usize].to_f32();
    }
    let expected_f16 = f16::from_f32(expected);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, 1).expect("alloc y");

    let desc = TraceDescriptor {
        n: D,
        element: ElementKind::F16,
    };
    let plan =
        TracePlan::<f16>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = TraceArgs::<f16> {
        x: TensorRef {
            data: dev_x.as_slice(),
            shape: shape(),
            stride: strides,
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: [],
            stride: [],
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = [f16::from_f32(0.0); 1];
    dev_y.copy_to_host(&mut got).expect("download");
    let gf = got[0].to_f32();
    let ef = expected_f16.to_f32();
    let diff = (gf - ef).abs();
    let allow = ef.abs().max(1.0) * (4.0 * F16_EPS);
    assert!(
        diff <= allow,
        "trace f16 mismatch: got {gf} expected {ef} diff {diff} allow {allow}"
    );
}

#[test]
#[ignore]
fn trace_bf16() {
    let (ctx, stream) = setup();
    let numel = (D as usize) * (D as usize);
    let host_x: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32(((i % 17) as f32) * 0.125 - 1.0))
        .collect();

    let strides = contiguous_stride(shape());
    let mut expected = 0f32;
    for i in 0..(D as i64) {
        let off = i * strides[0] + i * strides[1];
        expected += host_x[off as usize].to_f32();
    }
    let expected_bf16 = bf16::from_f32(expected);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, 1).expect("alloc y");

    let desc = TraceDescriptor {
        n: D,
        element: ElementKind::Bf16,
    };
    let plan =
        TracePlan::<bf16>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = TraceArgs::<bf16> {
        x: TensorRef {
            data: dev_x.as_slice(),
            shape: shape(),
            stride: strides,
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: [],
            stride: [],
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = [bf16::from_f32(0.0); 1];
    dev_y.copy_to_host(&mut got).expect("download");
    let gf = got[0].to_f32();
    let ef = expected_bf16.to_f32();
    let diff = (gf - ef).abs();
    let allow = ef.abs().max(1.0) * (4.0 * BF16_EPS);
    assert!(
        diff <= allow,
        "trace bf16 mismatch: got {gf} expected {ef} diff {diff} allow {allow}"
    );
}

#[test]
#[ignore]
fn trace_f64() {
    let (ctx, stream) = setup();
    let numel = (D as usize) * (D as usize);
    let host_x: Vec<f64> = (0..numel)
        .map(|i| ((i % 17) as f64) * 0.125 - 1.0)
        .collect();

    // f64 single-pass sum in input order — kernel walks the diagonal
    // exactly the same way, so the result should be bit-exact.
    let strides = contiguous_stride(shape());
    let mut expected = 0f64;
    for i in 0..(D as i64) {
        let off = i * strides[0] + i * strides[1];
        expected += host_x[off as usize];
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, 1).expect("alloc y");

    let desc = TraceDescriptor {
        n: D,
        element: ElementKind::F64,
    };
    let plan =
        TracePlan::<f64>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = TraceArgs::<f64> {
        x: TensorRef {
            data: dev_x.as_slice(),
            shape: shape(),
            stride: strides,
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: [],
            stride: [],
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = [0f64; 1];
    dev_y.copy_to_host(&mut got).expect("download");
    assert_eq!(
        got[0].to_bits(),
        expected.to_bits(),
        "trace f64 mismatch: got {} expected {}",
        got[0],
        expected
    );
}
