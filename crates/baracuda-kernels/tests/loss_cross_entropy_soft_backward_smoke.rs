//! Real-GPU smoke test for `CrossEntropyLossBackwardPlan` with soft targets.
//! BW × 4 dtypes × Mean.
//!
//! `dinput[n, c] = (softmax(input)[n, c] - target[n, c]) · dy / N`.

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

fn host_soft_ce_bw_f64(
    input: &[f64],
    target: &[f64],
    n_rows: usize,
    classes: usize,
    dy: f64,
) -> Vec<f64> {
    let mut out = vec![0.0; n_rows * classes];
    let scale = dy / (n_rows as f64);
    for i in 0..n_rows {
        let row = &input[i * classes..(i + 1) * classes];
        let trow = &target[i * classes..(i + 1) * classes];
        let mut m = f64::NEG_INFINITY;
        for &v in row {
            if v > m {
                m = v;
            }
        }
        let mut se = 0.0;
        for &v in row {
            se += (v - m).exp();
        }
        for c in 0..classes {
            let p = (row[c] - m).exp() / se;
            out[i * classes + c] = (p - trow[c]) * scale;
        }
    }
    out
}

fn make_soft_target_f32(n_rows: usize, classes: usize) -> Vec<f32> {
    let mut t = vec![0.1f32 / ((classes - 1) as f32); n_rows * classes];
    for i in 0..n_rows {
        t[i * classes + (i % classes)] = 0.9;
    }
    t
}

#[test]
#[ignore]
fn loss_cross_entropy_soft_backward_f32_mean() {
    let (ctx, stream) = setup();
    let n_rows = 4i32;
    let class_extent = 5i32;
    let host_inp: Vec<f32> = (0..(n_rows * class_extent) as usize)
        .map(|i| ((i as f32) * 0.3 - 1.5).sin())
        .collect();
    let host_t = make_soft_target_f32(n_rows as usize, class_extent as usize);
    let dy_host = [1.0f32];
    let expected: Vec<f32> = host_soft_ce_bw_f64(
        &host_inp.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &host_t.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        n_rows as usize, class_extent as usize, 1.0,
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
        target_kind: CrossEntropyTargetKind::SoftProb,
        element: ElementKind::F32,
    };
    let plan = CrossEntropyLossBackwardPlan::<f32>::select(
        &stream, &desc, PlanPreference::default()
    ).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        CrossEntropyLossBackwardArgs {
            input: TensorRef {
                data: dev_inp.as_slice(),
                shape: [n_rows, class_extent],
                stride: contiguous_stride([n_rows, class_extent]),
            },
            target: None,
            soft_target: Some(TensorRef {
                data: dev_t.as_slice(),
                shape: [n_rows, class_extent],
                stride: contiguous_stride([n_rows, class_extent]),
            }),
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
        assert!((got[i] - expected[i]).abs() <= tol,
            "f32 soft CE BW @{i}: got={} want={}", got[i], expected[i]);
    }
}

#[test]
#[ignore]
fn loss_cross_entropy_soft_backward_f64_mean() {
    let (ctx, stream) = setup();
    let n_rows = 3i32;
    let class_extent = 6i32;
    let host_inp: Vec<f64> = (0..(n_rows * class_extent) as usize)
        .map(|i| ((i as f64) * 0.3 - 1.5).sin())
        .collect();
    let host_t_f32 = make_soft_target_f32(n_rows as usize, class_extent as usize);
    let host_t: Vec<f64> = host_t_f32.iter().map(|&v| v as f64).collect();
    let dy_host = [2.0f64];
    let expected = host_soft_ce_bw_f64(
        &host_inp, &host_t, n_rows as usize, class_extent as usize, 2.0,
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
        target_kind: CrossEntropyTargetKind::SoftProb,
        element: ElementKind::F64,
    };
    let plan = CrossEntropyLossBackwardPlan::<f64>::select(
        &stream, &desc, PlanPreference::default()
    ).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        CrossEntropyLossBackwardArgs {
            input: TensorRef {
                data: dev_inp.as_slice(),
                shape: [n_rows, class_extent],
                stride: contiguous_stride([n_rows, class_extent]),
            },
            target: None,
            soft_target: Some(TensorRef {
                data: dev_t.as_slice(),
                shape: [n_rows, class_extent],
                stride: contiguous_stride([n_rows, class_extent]),
            }),
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
        let tol = expected[i].abs() * 32.0 * f64::EPSILON + 1e-11;
        assert!((got[i] - expected[i]).abs() <= tol);
    }
}

#[test]
#[ignore]
fn loss_cross_entropy_soft_backward_f16_mean() {
    let (ctx, stream) = setup();
    let n_rows = 4i32;
    let class_extent = 5i32;
    let host_inp_f32: Vec<f32> = (0..(n_rows * class_extent) as usize)
        .map(|i| ((i as f32) * 0.3 - 1.5).sin() * 0.5)
        .collect();
    let host_t_f32 = make_soft_target_f32(n_rows as usize, class_extent as usize);
    let host_inp: Vec<f16> = host_inp_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_t: Vec<f16> = host_t_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let dy_host: Vec<f16> = [1.0f32].iter().map(|&v| f16::from_f32(v)).collect();
    let inp64: Vec<f64> = host_inp.iter().map(|&v| v.to_f32() as f64).collect();
    let t64: Vec<f64> = host_t.iter().map(|&v| v.to_f32() as f64).collect();
    let expected_f64 = host_soft_ce_bw_f64(
        &inp64, &t64, n_rows as usize, class_extent as usize, 1.0,
    );

    let dev_inp = DeviceBuffer::from_slice(&ctx, &host_inp).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_di: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (n_rows * class_extent) as usize).unwrap();

    let desc = CrossEntropyLossBackwardDescriptor {
        n_rows,
        class_extent,
        reduction: LossReduction::Mean,
        target_kind: CrossEntropyTargetKind::SoftProb,
        element: ElementKind::F16,
    };
    let plan = CrossEntropyLossBackwardPlan::<f16>::select(
        &stream, &desc, PlanPreference::default()
    ).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        CrossEntropyLossBackwardArgs {
            input: TensorRef {
                data: dev_inp.as_slice(),
                shape: [n_rows, class_extent],
                stride: contiguous_stride([n_rows, class_extent]),
            },
            target: None,
            soft_target: Some(TensorRef {
                data: dev_t.as_slice(),
                shape: [n_rows, class_extent],
                stride: contiguous_stride([n_rows, class_extent]),
            }),
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
        let got_f32 = got[i].to_f32();
        let want = expected_f64[i] as f32;
        let tol = want.abs() * 16.0 * 9.77e-4_f32 + 5e-3;
        assert!((got_f32 - want).abs() <= tol,
            "f16 soft CE BW @{i}: got={} want={}", got_f32, want);
    }
}

#[test]
#[ignore]
fn loss_cross_entropy_soft_backward_bf16_mean() {
    let (ctx, stream) = setup();
    let n_rows = 3i32;
    let class_extent = 6i32;
    let host_inp_f32: Vec<f32> = (0..(n_rows * class_extent) as usize)
        .map(|i| ((i as f32) * 0.3 - 1.5).sin() * 0.5)
        .collect();
    let host_t_f32 = make_soft_target_f32(n_rows as usize, class_extent as usize);
    let host_inp: Vec<bf16> = host_inp_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_t: Vec<bf16> = host_t_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let dy_host: Vec<bf16> = [1.0f32].iter().map(|&v| bf16::from_f32(v)).collect();
    let inp64: Vec<f64> = host_inp.iter().map(|&v| v.to_f32() as f64).collect();
    let t64: Vec<f64> = host_t.iter().map(|&v| v.to_f32() as f64).collect();
    let expected_f64 = host_soft_ce_bw_f64(
        &inp64, &t64, n_rows as usize, class_extent as usize, 1.0,
    );

    let dev_inp = DeviceBuffer::from_slice(&ctx, &host_inp).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_di: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, (n_rows * class_extent) as usize).unwrap();

    let desc = CrossEntropyLossBackwardDescriptor {
        n_rows,
        class_extent,
        reduction: LossReduction::Mean,
        target_kind: CrossEntropyTargetKind::SoftProb,
        element: ElementKind::Bf16,
    };
    let plan = CrossEntropyLossBackwardPlan::<bf16>::select(
        &stream, &desc, PlanPreference::default()
    ).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        CrossEntropyLossBackwardArgs {
            input: TensorRef {
                data: dev_inp.as_slice(),
                shape: [n_rows, class_extent],
                stride: contiguous_stride([n_rows, class_extent]),
            },
            target: None,
            soft_target: Some(TensorRef {
                data: dev_t.as_slice(),
                shape: [n_rows, class_extent],
                stride: contiguous_stride([n_rows, class_extent]),
            }),
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
        let got_f32 = got[i].to_f32();
        let want = expected_f64[i] as f32;
        let tol = want.abs() * 16.0 * 7.81e-3_f32 + 2e-2;
        assert!((got_f32 - want).abs() <= tol,
            "bf16 soft CE BW @{i}: got={} want={}", got_f32, want);
    }
}
