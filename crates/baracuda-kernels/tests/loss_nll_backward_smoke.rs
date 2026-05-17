//! Real-GPU smoke test for `NllLossBackwardPlan`. BW × 4 dtypes × Mean.
//! `dinput[i, c] = -dy/N if c == target[i] else 0`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, LossReduction, NllLossBackwardArgs,
    NllLossBackwardDescriptor, NllLossBackwardPlan, PlanPreference, TensorMut, TensorRef,
    Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn host_nll_bw_f64(
    n_rows: usize,
    classes: usize,
    target: &[i64],
    dy: f64,
) -> Vec<f64> {
    let mut out = vec![0.0; n_rows * classes];
    let scale = -dy / (n_rows as f64);
    for i in 0..n_rows {
        out[i * classes + target[i] as usize] = scale;
    }
    out
}

#[test]
#[ignore]
fn loss_nll_backward_f32_mean() {
    let (ctx, stream) = setup();
    let n_rows = 4i32;
    let class_extent = 5i32;
    let host_t: Vec<i64> = vec![0, 2, 4, 1];
    let dy_host = [1.0f32];
    let expected: Vec<f32> = host_nll_bw_f64(
        n_rows as usize,
        class_extent as usize,
        &host_t,
        1.0,
    )
    .into_iter()
    .map(|v| v as f32)
    .collect();

    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_di: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (n_rows * class_extent) as usize).unwrap();

    let desc = NllLossBackwardDescriptor {
        n_rows,
        class_extent,
        reduction: LossReduction::Mean,
        element: ElementKind::F32,
    };
    let plan =
        NllLossBackwardPlan::<f32>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        NllLossBackwardArgs {
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1], stride: [1] },
            target: TensorRef {
                data: dev_t.as_slice(),
                shape: [n_rows],
                stride: contiguous_stride([n_rows]),
            },
            dinput: TensorMut {
                data: dev_di.as_slice_mut(),
                shape: [n_rows, class_extent],
                stride: contiguous_stride([n_rows, class_extent]),
            },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![0f32; (n_rows * class_extent) as usize];
    dev_di.copy_to_host(&mut got).unwrap();
    for i in 0..got.len() {
        let tol = expected[i].abs() * 16.0 * f32::EPSILON + 1e-6;
        assert!((got[i] - expected[i]).abs() <= tol);
    }
}

#[test]
#[ignore]
fn loss_nll_backward_f64_mean() {
    let (ctx, stream) = setup();
    let n_rows = 3i32;
    let class_extent = 4i32;
    let host_t: Vec<i64> = vec![1, 3, 0];
    let dy_host = [1.0f64];
    let expected = host_nll_bw_f64(
        n_rows as usize,
        class_extent as usize,
        &host_t,
        1.0,
    );

    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_di: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (n_rows * class_extent) as usize).unwrap();

    let desc = NllLossBackwardDescriptor {
        n_rows,
        class_extent,
        reduction: LossReduction::Mean,
        element: ElementKind::F64,
    };
    let plan =
        NllLossBackwardPlan::<f64>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        NllLossBackwardArgs {
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1], stride: [1] },
            target: TensorRef {
                data: dev_t.as_slice(),
                shape: [n_rows],
                stride: contiguous_stride([n_rows]),
            },
            dinput: TensorMut {
                data: dev_di.as_slice_mut(),
                shape: [n_rows, class_extent],
                stride: contiguous_stride([n_rows, class_extent]),
            },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![0f64; (n_rows * class_extent) as usize];
    dev_di.copy_to_host(&mut got).unwrap();
    for i in 0..got.len() {
        let tol = expected[i].abs() * 16.0 * f64::EPSILON + 1e-13;
        assert!((got[i] - expected[i]).abs() <= tol);
    }
}

#[test]
#[ignore]
fn loss_nll_backward_f16_mean() {
    let (ctx, stream) = setup();
    let n_rows = 4i32;
    let class_extent = 4i32;
    let host_t: Vec<i64> = vec![0, 1, 2, 3];
    let dy_host = [f16::from_f32(1.0)];
    let expected: Vec<f32> = host_nll_bw_f64(
        n_rows as usize,
        class_extent as usize,
        &host_t,
        1.0,
    )
    .into_iter()
    .map(|v| v as f32)
    .collect();

    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_di: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (n_rows * class_extent) as usize).unwrap();

    let desc = NllLossBackwardDescriptor {
        n_rows,
        class_extent,
        reduction: LossReduction::Mean,
        element: ElementKind::F16,
    };
    let plan =
        NllLossBackwardPlan::<f16>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        NllLossBackwardArgs {
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1], stride: [1] },
            target: TensorRef {
                data: dev_t.as_slice(),
                shape: [n_rows],
                stride: contiguous_stride([n_rows]),
            },
            dinput: TensorMut {
                data: dev_di.as_slice_mut(),
                shape: [n_rows, class_extent],
                stride: contiguous_stride([n_rows, class_extent]),
            },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![f16::ZERO; (n_rows * class_extent) as usize];
    dev_di.copy_to_host(&mut got).unwrap();
    for i in 0..got.len() {
        let tol = expected[i].abs() * 16.0 * 9.77e-4_f32 + 5e-3;
        let g = got[i].to_f32();
        assert!((g - expected[i]).abs() <= tol);
    }
}

#[test]
#[ignore]
fn loss_nll_backward_bf16_mean() {
    let (ctx, stream) = setup();
    let n_rows = 3i32;
    let class_extent = 4i32;
    let host_t: Vec<i64> = vec![1, 3, 0];
    let dy_host = [bf16::from_f32(1.0)];
    let expected: Vec<f32> = host_nll_bw_f64(
        n_rows as usize,
        class_extent as usize,
        &host_t,
        1.0,
    )
    .into_iter()
    .map(|v| v as f32)
    .collect();

    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_di: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, (n_rows * class_extent) as usize).unwrap();

    let desc = NllLossBackwardDescriptor {
        n_rows,
        class_extent,
        reduction: LossReduction::Mean,
        element: ElementKind::Bf16,
    };
    let plan =
        NllLossBackwardPlan::<bf16>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        NllLossBackwardArgs {
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1], stride: [1] },
            target: TensorRef {
                data: dev_t.as_slice(),
                shape: [n_rows],
                stride: contiguous_stride([n_rows]),
            },
            dinput: TensorMut {
                data: dev_di.as_slice_mut(),
                shape: [n_rows, class_extent],
                stride: contiguous_stride([n_rows, class_extent]),
            },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![bf16::ZERO; (n_rows * class_extent) as usize];
    dev_di.copy_to_host(&mut got).unwrap();
    for i in 0..got.len() {
        let tol = expected[i].abs() * 16.0 * 7.81e-3_f32 + 3e-2;
        let g = got[i].to_f32();
        assert!((g - expected[i]).abs() <= tol);
    }
}
