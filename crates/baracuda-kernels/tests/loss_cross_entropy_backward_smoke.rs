//! Real-GPU smoke test for `CrossEntropyLossBackwardPlan`. BW × 4 dtypes × Mean.
//! `dinput[i, c] = (softmax(input)[i, c] - 1{c == t[i]}) · dy / N`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, CrossEntropyLossBackwardArgs, CrossEntropyLossBackwardDescriptor,
    CrossEntropyLossBackwardPlan, CrossEntropyTargetKind, ElementKind, LossReduction,
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

fn host_ce_bw_f64(
    input: &[f64],
    target: &[i64],
    n_rows: usize,
    classes: usize,
    dy: f64,
) -> Vec<f64> {
    let mut out = vec![0.0; n_rows * classes];
    let scale = dy / (n_rows as f64);
    for i in 0..n_rows {
        let row = &input[i * classes..(i + 1) * classes];
        let mut max = f64::NEG_INFINITY;
        for &v in row {
            if v > max {
                max = v;
            }
        }
        let mut se = 0.0;
        for &v in row {
            se += (v - max).exp();
        }
        let t = target[i] as usize;
        for c in 0..classes {
            let p = (row[c] - max).exp() / se;
            let one_hot = if c == t { 1.0 } else { 0.0 };
            out[i * classes + c] = (p - one_hot) * scale;
        }
    }
    out
}

#[test]
#[ignore]
fn loss_cross_entropy_backward_f32_mean() {
    let (ctx, stream) = setup();
    let n_rows = 4i32;
    let class_extent = 5i32;
    let host_inp: Vec<f32> = (0..(n_rows * class_extent) as usize)
        .map(|i| ((i as f32) * 0.3 - 1.5).sin())
        .collect();
    let host_t: Vec<i64> = vec![0, 2, 4, 1];
    let dy_host = [1.0f32];
    let expected: Vec<f32> = host_ce_bw_f64(
        &host_inp.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &host_t,
        n_rows as usize,
        class_extent as usize,
        1.0,
    )
    .into_iter()
    .map(|v| v as f32)
    .collect();

    let dev_inp = DeviceBuffer::from_slice(&ctx, &host_inp).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_di: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (n_rows * class_extent) as usize).unwrap();

    let desc = CrossEntropyLossBackwardDescriptor {
        n_rows,
        class_extent,
        reduction: LossReduction::Mean,
        target_kind: CrossEntropyTargetKind::ClassIndex,
        element: ElementKind::F32,
    };
    let plan = CrossEntropyLossBackwardPlan::<f32>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .unwrap();
    plan.run(
        &stream,
        Workspace::None,
        CrossEntropyLossBackwardArgs {
            input: TensorRef {
                data: dev_inp.as_slice(),
                shape: [n_rows, class_extent],
                stride: contiguous_stride([n_rows, class_extent]),
            },
            target: Some(TensorRef {
                data: dev_t.as_slice(),
                shape: [n_rows],
                stride: contiguous_stride([n_rows]),
            }),
            soft_target: None,
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1], stride: [1] },
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
        let tol = expected[i].abs() * 32.0 * f32::EPSILON + 1e-5;
        assert!((got[i] - expected[i]).abs() <= tol, "f32 CE BW @{i}: got={} want={}",
            got[i], expected[i]);
    }
}

#[test]
#[ignore]
fn loss_cross_entropy_backward_f64_mean() {
    let (ctx, stream) = setup();
    let n_rows = 3i32;
    let class_extent = 5i32;
    let host_inp: Vec<f64> = (0..(n_rows * class_extent) as usize)
        .map(|i| ((i as f64) * 0.25 - 1.0).sin())
        .collect();
    let host_t: Vec<i64> = vec![2, 4, 0];
    let dy_host = [1.0f64];
    let expected = host_ce_bw_f64(
        &host_inp,
        &host_t,
        n_rows as usize,
        class_extent as usize,
        1.0,
    );

    let dev_inp = DeviceBuffer::from_slice(&ctx, &host_inp).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_di: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (n_rows * class_extent) as usize).unwrap();

    let desc = CrossEntropyLossBackwardDescriptor {
        n_rows,
        class_extent,
        reduction: LossReduction::Mean,
        target_kind: CrossEntropyTargetKind::ClassIndex,
        element: ElementKind::F64,
    };
    let plan = CrossEntropyLossBackwardPlan::<f64>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .unwrap();
    plan.run(
        &stream,
        Workspace::None,
        CrossEntropyLossBackwardArgs {
            input: TensorRef {
                data: dev_inp.as_slice(),
                shape: [n_rows, class_extent],
                stride: contiguous_stride([n_rows, class_extent]),
            },
            target: Some(TensorRef {
                data: dev_t.as_slice(),
                shape: [n_rows],
                stride: contiguous_stride([n_rows]),
            }),
            soft_target: None,
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1], stride: [1] },
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
        let tol = expected[i].abs() * 32.0 * f64::EPSILON + 1e-13;
        assert!((got[i] - expected[i]).abs() <= tol);
    }
}

#[test]
#[ignore]
fn loss_cross_entropy_backward_f16_mean() {
    let (ctx, stream) = setup();
    let n_rows = 4i32;
    let class_extent = 4i32;
    let host_inp_f32: Vec<f32> = (0..(n_rows * class_extent) as usize)
        .map(|i| ((i as f32) * 0.3 - 1.5).sin())
        .collect();
    let host_t: Vec<i64> = vec![0, 1, 2, 3];
    let dy_host = [f16::from_f32(1.0)];
    let expected: Vec<f32> = host_ce_bw_f64(
        &host_inp_f32.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &host_t,
        n_rows as usize,
        class_extent as usize,
        1.0,
    )
    .into_iter()
    .map(|v| v as f32)
    .collect();
    let host_inp: Vec<f16> = host_inp_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let dev_inp = DeviceBuffer::from_slice(&ctx, &host_inp).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_di: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (n_rows * class_extent) as usize).unwrap();

    let desc = CrossEntropyLossBackwardDescriptor {
        n_rows,
        class_extent,
        reduction: LossReduction::Mean,
        target_kind: CrossEntropyTargetKind::ClassIndex,
        element: ElementKind::F16,
    };
    let plan = CrossEntropyLossBackwardPlan::<f16>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .unwrap();
    plan.run(
        &stream,
        Workspace::None,
        CrossEntropyLossBackwardArgs {
            input: TensorRef {
                data: dev_inp.as_slice(),
                shape: [n_rows, class_extent],
                stride: contiguous_stride([n_rows, class_extent]),
            },
            target: Some(TensorRef {
                data: dev_t.as_slice(),
                shape: [n_rows],
                stride: contiguous_stride([n_rows]),
            }),
            soft_target: None,
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1], stride: [1] },
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
        let tol = expected[i].abs() * 32.0 * 9.77e-4_f32 + 5e-3;
        let g = got[i].to_f32();
        assert!((g - expected[i]).abs() <= tol, "f16 CE BW @{i}: got={} want={}",
            g, expected[i]);
    }
}

#[test]
#[ignore]
fn loss_cross_entropy_backward_bf16_mean() {
    let (ctx, stream) = setup();
    let n_rows = 3i32;
    let class_extent = 5i32;
    let host_inp_f32: Vec<f32> = (0..(n_rows * class_extent) as usize)
        .map(|i| ((i as f32) * 0.25 - 1.0).cos())
        .collect();
    let host_t: Vec<i64> = vec![2, 4, 0];
    let dy_host = [bf16::from_f32(1.0)];
    let expected: Vec<f32> = host_ce_bw_f64(
        &host_inp_f32.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &host_t,
        n_rows as usize,
        class_extent as usize,
        1.0,
    )
    .into_iter()
    .map(|v| v as f32)
    .collect();
    let host_inp: Vec<bf16> = host_inp_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let dev_inp = DeviceBuffer::from_slice(&ctx, &host_inp).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_di: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, (n_rows * class_extent) as usize).unwrap();

    let desc = CrossEntropyLossBackwardDescriptor {
        n_rows,
        class_extent,
        reduction: LossReduction::Mean,
        target_kind: CrossEntropyTargetKind::ClassIndex,
        element: ElementKind::Bf16,
    };
    let plan = CrossEntropyLossBackwardPlan::<bf16>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .unwrap();
    plan.run(
        &stream,
        Workspace::None,
        CrossEntropyLossBackwardArgs {
            input: TensorRef {
                data: dev_inp.as_slice(),
                shape: [n_rows, class_extent],
                stride: contiguous_stride([n_rows, class_extent]),
            },
            target: Some(TensorRef {
                data: dev_t.as_slice(),
                shape: [n_rows],
                stride: contiguous_stride([n_rows]),
            }),
            soft_target: None,
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1], stride: [1] },
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
        let tol = expected[i].abs() * 32.0 * 7.81e-3_f32 + 3e-2;
        let g = got[i].to_f32();
        assert!((g - expected[i]).abs() <= tol);
    }
}
