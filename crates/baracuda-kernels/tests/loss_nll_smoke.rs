//! Real-GPU smoke test for `NllLossPlan`. FW × 4 dtypes × Mean reduction.
//! `y = -mean(input[i, target[i]])`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, LossReduction, NllLossArgs, NllLossDescriptor, NllLossPlan,
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

fn host_nll_mean_f64(input: &[f64], target: &[i64], n_rows: usize, classes: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..n_rows {
        let t = target[i] as usize;
        s += -input[i * classes + t];
    }
    s / (n_rows as f64)
}

#[test]
#[ignore]
fn loss_nll_f32_mean() {
    let (ctx, stream) = setup();
    let n_rows = 4i32;
    let class_extent = 5i32;
    let host_inp: Vec<f32> = (0..(n_rows * class_extent) as usize)
        .map(|i| -1.0 - (i as f32) * 0.1)
        .collect();
    let host_t: Vec<i64> = vec![0, 2, 4, 1];
    let expected = host_nll_mean_f64(
        &host_inp.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &host_t,
        n_rows as usize,
        class_extent as usize,
    ) as f32;

    let dev_inp = DeviceBuffer::from_slice(&ctx, &host_inp).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, (n_rows as usize) * 4).unwrap();

    let desc = NllLossDescriptor {
        n_rows,
        class_extent,
        reduction: LossReduction::Mean,
        element: ElementKind::F32,
    };
    let plan = NllLossPlan::<f32>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        NllLossArgs {
            input: TensorRef {
                data: dev_inp.as_slice(),
                shape: [n_rows, class_extent],
                stride: contiguous_stride([n_rows, class_extent]),
            },
            target: TensorRef {
                data: dev_t.as_slice(),
                shape: [n_rows],
                stride: contiguous_stride([n_rows]),
            },
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1], stride: [1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [0f32; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let tol = expected.abs() * 8.0 * f32::EPSILON + 1e-6;
    assert!((got[0] - expected).abs() <= tol, "f32 NLL: got={} want={}", got[0], expected);
}

#[test]
#[ignore]
fn loss_nll_f64_mean() {
    let (ctx, stream) = setup();
    let n_rows = 3i32;
    let class_extent = 6i32;
    let host_inp: Vec<f64> = (0..(n_rows * class_extent) as usize)
        .map(|i| -0.7 - (i as f64) * 0.08)
        .collect();
    let host_t: Vec<i64> = vec![1, 3, 5];
    let expected = host_nll_mean_f64(
        &host_inp,
        &host_t,
        n_rows as usize,
        class_extent as usize,
    );

    let dev_inp = DeviceBuffer::from_slice(&ctx, &host_inp).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, (n_rows as usize) * 8).unwrap();

    let desc = NllLossDescriptor {
        n_rows,
        class_extent,
        reduction: LossReduction::Mean,
        element: ElementKind::F64,
    };
    let plan = NllLossPlan::<f64>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        NllLossArgs {
            input: TensorRef {
                data: dev_inp.as_slice(),
                shape: [n_rows, class_extent],
                stride: contiguous_stride([n_rows, class_extent]),
            },
            target: TensorRef {
                data: dev_t.as_slice(),
                shape: [n_rows],
                stride: contiguous_stride([n_rows]),
            },
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1], stride: [1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [0f64; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let tol = expected.abs() * 8.0 * f64::EPSILON + 1e-12;
    assert!((got[0] - expected).abs() <= tol);
}

#[test]
#[ignore]
fn loss_nll_f16_mean() {
    let (ctx, stream) = setup();
    let n_rows = 4i32;
    let class_extent = 4i32;
    let host_inp_f32: Vec<f32> = (0..(n_rows * class_extent) as usize)
        .map(|i| -0.5 - (i as f32) * 0.05)
        .collect();
    let host_t: Vec<i64> = vec![0, 1, 2, 3];
    let expected = host_nll_mean_f64(
        &host_inp_f32.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &host_t,
        n_rows as usize,
        class_extent as usize,
    ) as f32;
    let host_inp: Vec<f16> = host_inp_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let dev_inp = DeviceBuffer::from_slice(&ctx, &host_inp).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, (n_rows as usize) * 2).unwrap();

    let desc = NllLossDescriptor {
        n_rows,
        class_extent,
        reduction: LossReduction::Mean,
        element: ElementKind::F16,
    };
    let plan = NllLossPlan::<f16>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        NllLossArgs {
            input: TensorRef {
                data: dev_inp.as_slice(),
                shape: [n_rows, class_extent],
                stride: contiguous_stride([n_rows, class_extent]),
            },
            target: TensorRef {
                data: dev_t.as_slice(),
                shape: [n_rows],
                stride: contiguous_stride([n_rows]),
            },
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1], stride: [1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [f16::ZERO; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let g = got[0].to_f32();
    let tol = expected.abs() * 8.0 * 9.77e-4_f32 + 5e-3;
    assert!((g - expected).abs() <= tol, "f16 NLL: got={} want={}", g, expected);
}

#[test]
#[ignore]
fn loss_nll_bf16_mean() {
    let (ctx, stream) = setup();
    let n_rows = 3i32;
    let class_extent = 5i32;
    let host_inp_f32: Vec<f32> = (0..(n_rows * class_extent) as usize)
        .map(|i| -0.6 - (i as f32) * 0.07)
        .collect();
    let host_t: Vec<i64> = vec![2, 4, 0];
    let expected = host_nll_mean_f64(
        &host_inp_f32.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &host_t,
        n_rows as usize,
        class_extent as usize,
    ) as f32;
    let host_inp: Vec<bf16> = host_inp_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let dev_inp = DeviceBuffer::from_slice(&ctx, &host_inp).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, (n_rows as usize) * 2).unwrap();

    let desc = NllLossDescriptor {
        n_rows,
        class_extent,
        reduction: LossReduction::Mean,
        element: ElementKind::Bf16,
    };
    let plan = NllLossPlan::<bf16>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        NllLossArgs {
            input: TensorRef {
                data: dev_inp.as_slice(),
                shape: [n_rows, class_extent],
                stride: contiguous_stride([n_rows, class_extent]),
            },
            target: TensorRef {
                data: dev_t.as_slice(),
                shape: [n_rows],
                stride: contiguous_stride([n_rows]),
            },
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1], stride: [1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [bf16::ZERO; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let g = got[0].to_f32();
    let tol = expected.abs() * 8.0 * 7.81e-3_f32 + 3e-2;
    assert!((g - expected).abs() <= tol, "bf16 NLL: got={} want={}", g, expected);
}
