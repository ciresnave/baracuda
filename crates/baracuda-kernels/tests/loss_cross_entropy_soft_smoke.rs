//! Real-GPU smoke test for `CrossEntropyLossPlan` with `target_kind=SoftProb`.
//! FW × 4 dtypes × Mean.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, CrossEntropyLossArgs, CrossEntropyLossDescriptor, CrossEntropyLossPlan,
    CrossEntropyTargetKind, ElementKind, LossReduction, PlanPreference, TensorMut, TensorRef,
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

fn host_soft_ce_mean_f64(
    input: &[f64],
    target: &[f64],
    n_rows: usize,
    classes: usize,
) -> f64 {
    let mut s = 0.0;
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
        let lse = m + se.ln();
        let mut acc = 0.0;
        for c in 0..classes {
            acc += trow[c] * (row[c] - lse);
        }
        s += -acc;
    }
    s / (n_rows as f64)
}

fn make_soft_target(n_rows: usize, classes: usize) -> Vec<f32> {
    // Simple deterministic soft targets: smoothed one-hot
    let mut t = vec![0.1f32 / ((classes - 1) as f32); n_rows * classes];
    for i in 0..n_rows {
        t[i * classes + (i % classes)] = 0.9;
    }
    t
}

#[test]
#[ignore]
fn loss_cross_entropy_soft_f32_mean() {
    let (ctx, stream) = setup();
    let n_rows = 4i32;
    let class_extent = 5i32;
    let host_inp: Vec<f32> = (0..(n_rows * class_extent) as usize)
        .map(|i| ((i as f32) * 0.3 - 1.5).sin())
        .collect();
    let host_t = make_soft_target(n_rows as usize, class_extent as usize);
    let expected = host_soft_ce_mean_f64(
        &host_inp.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &host_t.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        n_rows as usize,
        class_extent as usize,
    ) as f32;

    let dev_inp = DeviceBuffer::from_slice(&ctx, &host_inp).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, (n_rows as usize) * 4).unwrap();

    let desc = CrossEntropyLossDescriptor {
        n_rows,
        class_extent,
        reduction: LossReduction::Mean,
        target_kind: CrossEntropyTargetKind::SoftProb,
        element: ElementKind::F32,
    };
    let plan = CrossEntropyLossPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        CrossEntropyLossArgs {
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
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1], stride: [1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [0f32; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let tol = expected.abs() * 16.0 * f32::EPSILON + 1e-5;
    assert!((got[0] - expected).abs() <= tol, "f32 soft CE: got={} want={}", got[0], expected);
}

#[test]
#[ignore]
fn loss_cross_entropy_soft_f64_mean() {
    let (ctx, stream) = setup();
    let n_rows = 3i32;
    let class_extent = 6i32;
    let host_inp: Vec<f64> = (0..(n_rows * class_extent) as usize)
        .map(|i| ((i as f64) * 0.3 - 1.5).sin())
        .collect();
    let host_t_f32 = make_soft_target(n_rows as usize, class_extent as usize);
    let host_t: Vec<f64> = host_t_f32.iter().map(|&v| v as f64).collect();
    let expected = host_soft_ce_mean_f64(
        &host_inp, &host_t, n_rows as usize, class_extent as usize,
    );

    let dev_inp = DeviceBuffer::from_slice(&ctx, &host_inp).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, (n_rows as usize) * 8).unwrap();

    let desc = CrossEntropyLossDescriptor {
        n_rows,
        class_extent,
        reduction: LossReduction::Mean,
        target_kind: CrossEntropyTargetKind::SoftProb,
        element: ElementKind::F64,
    };
    let plan = CrossEntropyLossPlan::<f64>::select(&stream, &desc, PlanPreference::default())
        .unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        CrossEntropyLossArgs {
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
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1], stride: [1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [0f64; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let tol = expected.abs() * 16.0 * f64::EPSILON + 1e-11;
    assert!((got[0] - expected).abs() <= tol);
}

#[test]
#[ignore]
fn loss_cross_entropy_soft_f16_mean() {
    let (ctx, stream) = setup();
    let n_rows = 4i32;
    let class_extent = 5i32;
    let host_inp_f32: Vec<f32> = (0..(n_rows * class_extent) as usize)
        .map(|i| ((i as f32) * 0.3 - 1.5).sin() * 0.5)
        .collect();
    let host_t_f32 = make_soft_target(n_rows as usize, class_extent as usize);
    let host_inp: Vec<f16> = host_inp_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_t: Vec<f16> = host_t_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let inp64: Vec<f64> = host_inp.iter().map(|&v| v.to_f32() as f64).collect();
    let t64: Vec<f64> = host_t.iter().map(|&v| v.to_f32() as f64).collect();
    let expected = host_soft_ce_mean_f64(
        &inp64, &t64, n_rows as usize, class_extent as usize,
    ) as f32;

    let dev_inp = DeviceBuffer::from_slice(&ctx, &host_inp).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, (n_rows as usize) * 2).unwrap();

    let desc = CrossEntropyLossDescriptor {
        n_rows,
        class_extent,
        reduction: LossReduction::Mean,
        target_kind: CrossEntropyTargetKind::SoftProb,
        element: ElementKind::F16,
    };
    let plan = CrossEntropyLossPlan::<f16>::select(&stream, &desc, PlanPreference::default())
        .unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        CrossEntropyLossArgs {
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
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1], stride: [1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [f16::ZERO; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let got_f32 = got[0].to_f32();
    let tol = expected.abs() * 32.0 * 9.77e-4_f32 + 1e-2;
    assert!((got_f32 - expected).abs() <= tol, "f16 soft CE: got={} want={}", got_f32, expected);
}

#[test]
#[ignore]
fn loss_cross_entropy_soft_bf16_mean() {
    let (ctx, stream) = setup();
    let n_rows = 3i32;
    let class_extent = 6i32;
    let host_inp_f32: Vec<f32> = (0..(n_rows * class_extent) as usize)
        .map(|i| ((i as f32) * 0.3 - 1.5).sin() * 0.5)
        .collect();
    let host_t_f32 = make_soft_target(n_rows as usize, class_extent as usize);
    let host_inp: Vec<bf16> = host_inp_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_t: Vec<bf16> = host_t_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let inp64: Vec<f64> = host_inp.iter().map(|&v| v.to_f32() as f64).collect();
    let t64: Vec<f64> = host_t.iter().map(|&v| v.to_f32() as f64).collect();
    let expected = host_soft_ce_mean_f64(
        &inp64, &t64, n_rows as usize, class_extent as usize,
    ) as f32;

    let dev_inp = DeviceBuffer::from_slice(&ctx, &host_inp).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, (n_rows as usize) * 2).unwrap();

    let desc = CrossEntropyLossDescriptor {
        n_rows,
        class_extent,
        reduction: LossReduction::Mean,
        target_kind: CrossEntropyTargetKind::SoftProb,
        element: ElementKind::Bf16,
    };
    let plan = CrossEntropyLossPlan::<bf16>::select(&stream, &desc, PlanPreference::default())
        .unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        CrossEntropyLossArgs {
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
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1], stride: [1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [bf16::ZERO; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let got_f32 = got[0].to_f32();
    let tol = expected.abs() * 32.0 * 7.81e-3_f32 + 3e-2;
    assert!((got_f32 - expected).abs() <= tol, "bf16 soft CE: got={} want={}", got_f32, expected);
}
