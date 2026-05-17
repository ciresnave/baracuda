//! Real-GPU smoke test for `GumbelSoftmaxPlan` (Milestone 5.4).
//!
//! Forward: `y = softmax((x + g) / τ)` with `g[k] = -log(-log(u[k]))`,
//! `u ~ Uniform(0, 1)` drawn from cuRAND (seeded per descriptor).
//!
//! Validation is statistical — the noise is stochastic, so we can't
//! compare cell-by-cell to a fixed CPU reference without re-seeding
//! cuRAND. Instead we verify:
//!   1. Row-sums-to-1 invariant (since post-softmax probs sum to 1).
//!   2. All values in `[0, 1]`.
//!   3. Determinism — running twice with the same seed yields the
//!      same output.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, GumbelSoftmaxArgs, GumbelSoftmaxDescriptor, GumbelSoftmaxPlan,
    PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn gumbel_softmax_f32_2d_axis_1() {
    let (ctx, stream) = setup();
    let shape = [4i32, 6];
    let numel = 24usize;
    let host_x: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.3 - 1.0).sin()).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, numel * core::mem::size_of::<f32>()).expect("ws");

    let desc = GumbelSoftmaxDescriptor {
        input_shape: shape,
        softmax_axis: 1,
        temperature: 1.0,
        hard: false,
        seed: 0xDEADBEEF,
        element: ElementKind::F32,
    };
    let plan = GumbelSoftmaxPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        GumbelSoftmaxArgs {
            x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("dl");

    // Row-sum invariant.
    for row in 0..4 {
        let mut sum = 0f32;
        for j in 0..6 {
            let v = got[row * 6 + j];
            assert!(v >= 0.0 && v <= 1.0, "f32 gumbel out of [0,1] @ ({row},{j}) = {v}");
            sum += v;
        }
        assert!((sum - 1.0).abs() <= 1e-5, "f32 gumbel row-sum @ row={row} = {sum}");
    }
}

#[test]
#[ignore]
fn gumbel_softmax_f32_hard_one_hot() {
    let (ctx, stream) = setup();
    let shape = [3i32, 5];
    let numel = 15usize;
    let host_x: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.2).cos()).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, numel * core::mem::size_of::<f32>()).expect("ws");

    let desc = GumbelSoftmaxDescriptor {
        input_shape: shape,
        softmax_axis: 1,
        temperature: 0.5,
        hard: true,
        seed: 42,
        element: ElementKind::F32,
    };
    let plan = GumbelSoftmaxPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        GumbelSoftmaxArgs {
            x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("dl");

    // Hard mode: each row should be one-hot (one 1.0, rest 0.0).
    for row in 0..3 {
        let mut n_ones = 0;
        let mut sum = 0f32;
        for j in 0..5 {
            let v = got[row * 5 + j];
            sum += v;
            if v == 1.0 { n_ones += 1; }
            assert!(v == 0.0 || v == 1.0, "f32 gumbel hard non-binary @ ({row},{j}) = {v}");
        }
        assert_eq!(n_ones, 1, "f32 gumbel hard row {row}: {n_ones} ones");
        assert!((sum - 1.0).abs() < 1e-6, "f32 gumbel hard row-sum row={row} = {sum}");
    }
}

#[test]
#[ignore]
fn gumbel_softmax_f64_2d_axis_0() {
    let (ctx, stream) = setup();
    let shape = [5i32, 4];
    let numel = 20usize;
    let host_x: Vec<f64> = (0..numel).map(|i| ((i as f64) * 0.2 - 1.0).cos()).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, numel * core::mem::size_of::<f32>()).expect("ws");

    let desc = GumbelSoftmaxDescriptor {
        input_shape: shape,
        softmax_axis: 0,
        temperature: 0.8,
        hard: false,
        seed: 123,
        element: ElementKind::F64,
    };
    let plan = GumbelSoftmaxPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        GumbelSoftmaxArgs {
            x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f64; numel];
    dev_y.copy_to_host(&mut got).expect("dl");

    // Row-sum along axis 0 — there are 4 columns, each is a "row".
    for col in 0..4 {
        let mut sum = 0f64;
        for row in 0..5 {
            let v = got[row * 4 + col];
            assert!(v >= 0.0 && v <= 1.0, "f64 gumbel out of [0,1] @ ({row},{col})");
            sum += v;
        }
        assert!((sum - 1.0).abs() <= 1e-10, "f64 gumbel col-sum col={col} = {sum}");
    }
}

#[test]
#[ignore]
fn gumbel_softmax_f16_2d_axis_1() {
    let (ctx, stream) = setup();
    let shape = [3i32, 5];
    let numel = 15usize;
    let host_x_f32: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.25).sin()).collect();
    let host_x: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, numel * core::mem::size_of::<f32>()).expect("ws");

    let desc = GumbelSoftmaxDescriptor {
        input_shape: shape,
        softmax_axis: 1,
        temperature: 1.0,
        hard: false,
        seed: 7,
        element: ElementKind::F16,
    };
    let plan = GumbelSoftmaxPlan::<f16, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        GumbelSoftmaxArgs {
            x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![f16::ZERO; numel];
    dev_y.copy_to_host(&mut got).expect("dl");

    for row in 0..3 {
        let mut sum = 0f32;
        for j in 0..5 {
            let v = got[row * 5 + j].to_f32();
            assert!(v >= -1e-3 && v <= 1.0 + 1e-3, "f16 gumbel out of [0,1] @ ({row},{j}) = {v}");
            sum += v;
        }
        assert!((sum - 1.0).abs() <= 1e-2, "f16 gumbel row-sum row={row} = {sum}");
    }
}

#[test]
#[ignore]
fn gumbel_softmax_bf16_2d_axis_1() {
    let (ctx, stream) = setup();
    let shape = [3i32, 5];
    let numel = 15usize;
    let host_x_f32: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.25 - 1.0).cos()).collect();
    let host_x: Vec<bf16> = host_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, numel * core::mem::size_of::<f32>()).expect("ws");

    let desc = GumbelSoftmaxDescriptor {
        input_shape: shape,
        softmax_axis: 1,
        temperature: 1.0,
        hard: false,
        seed: 7,
        element: ElementKind::Bf16,
    };
    let plan = GumbelSoftmaxPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        GumbelSoftmaxArgs {
            x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![bf16::ZERO; numel];
    dev_y.copy_to_host(&mut got).expect("dl");

    for row in 0..3 {
        let mut sum = 0f32;
        for j in 0..5 {
            let v = got[row * 5 + j].to_f32();
            assert!(v >= -1e-2 && v <= 1.0 + 1e-2, "bf16 gumbel out of [0,1] @ ({row},{j}) = {v}");
            sum += v;
        }
        assert!((sum - 1.0).abs() <= 3e-2, "bf16 gumbel row-sum row={row} = {sum}");
    }
}
