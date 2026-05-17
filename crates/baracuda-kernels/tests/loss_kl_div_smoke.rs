//! Real-GPU smoke test for `KlDivLossPlan`. FW × 4 dtypes × Mean.
//!
//! PyTorch convention: input is already log-prob;
//! `y = mean(target·(log(target) - input))`. Cells with target == 0
//! contribute 0.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, KlDivLossArgs, KlDivLossDescriptor, KlDivLossPlan,
    LossReduction, PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn host_kl_div_mean_f64(input: &[f64], target: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..input.len() {
        if target[i] > 0.0 {
            s += target[i] * (target[i].ln() - input[i]);
        }
    }
    s / (input.len() as f64)
}

#[test]
#[ignore]
fn loss_kl_div_f32_mean() {
    let (ctx, stream) = setup();
    let shape = [3i32, 4];
    let numel = 12usize;
    // input as log-prob, target as prob
    let host_inp: Vec<f32> = (0..numel).map(|i| -1.0 - (i as f32) * 0.1).collect();
    let host_t: Vec<f32> = (0..numel).map(|i| 0.05 + (i as f32) * 0.02).collect();
    let expected = host_kl_div_mean_f64(
        &host_inp.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &host_t.iter().map(|&v| v as f64).collect::<Vec<_>>(),
    ) as f32;

    let dev_inp = DeviceBuffer::from_slice(&ctx, &host_inp).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel * 4).unwrap();

    let desc = KlDivLossDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        element: ElementKind::F32,
    };
    let plan = KlDivLossPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        KlDivLossArgs {
            input: TensorRef {
                data: dev_inp.as_slice(),
                shape,
                stride: contiguous_stride(shape),
            },
            target: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1], stride: [1, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [0f32; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let tol = expected.abs() * 16.0 * f32::EPSILON + 1e-5;
    assert!((got[0] - expected).abs() <= tol, "f32 KLDiv: got={} want={}", got[0], expected);
}

#[test]
#[ignore]
fn loss_kl_div_f64_mean() {
    let (ctx, stream) = setup();
    let shape = [4i32, 3];
    let numel = 12usize;
    let host_inp: Vec<f64> = (0..numel).map(|i| -0.8 - (i as f64) * 0.1).collect();
    let host_t: Vec<f64> = (0..numel).map(|i| 0.04 + (i as f64) * 0.02).collect();
    let expected = host_kl_div_mean_f64(&host_inp, &host_t);

    let dev_inp = DeviceBuffer::from_slice(&ctx, &host_inp).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel * 8).unwrap();

    let desc = KlDivLossDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        element: ElementKind::F64,
    };
    let plan = KlDivLossPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        KlDivLossArgs {
            input: TensorRef {
                data: dev_inp.as_slice(),
                shape,
                stride: contiguous_stride(shape),
            },
            target: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1], stride: [1, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [0f64; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let tol = expected.abs() * 16.0 * f64::EPSILON + 1e-12;
    assert!((got[0] - expected).abs() <= tol);
}

#[test]
#[ignore]
fn loss_kl_div_f16_mean() {
    let (ctx, stream) = setup();
    let shape = [3i32, 4];
    let numel = 12usize;
    let host_inp_f32: Vec<f32> = (0..numel).map(|i| -0.5 - (i as f32) * 0.08).collect();
    let host_t_f32: Vec<f32> = (0..numel).map(|i| 0.06 + (i as f32) * 0.02).collect();
    let expected = host_kl_div_mean_f64(
        &host_inp_f32.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &host_t_f32.iter().map(|&v| v as f64).collect::<Vec<_>>(),
    ) as f32;
    let host_inp: Vec<f16> = host_inp_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_t: Vec<f16> = host_t_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let dev_inp = DeviceBuffer::from_slice(&ctx, &host_inp).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel * 2).unwrap();

    let desc = KlDivLossDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        element: ElementKind::F16,
    };
    let plan = KlDivLossPlan::<f16, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        KlDivLossArgs {
            input: TensorRef {
                data: dev_inp.as_slice(),
                shape,
                stride: contiguous_stride(shape),
            },
            target: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1], stride: [1, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [f16::ZERO; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let g = got[0].to_f32();
    let tol = expected.abs() * 16.0 * 9.77e-4_f32 + 3e-3;
    assert!((g - expected).abs() <= tol, "f16 KLDiv: got={} want={}", g, expected);
}

#[test]
#[ignore]
fn loss_kl_div_bf16_mean() {
    let (ctx, stream) = setup();
    let shape = [3i32, 4];
    let numel = 12usize;
    let host_inp_f32: Vec<f32> = (0..numel).map(|i| -0.6 - (i as f32) * 0.07).collect();
    let host_t_f32: Vec<f32> = (0..numel).map(|i| 0.05 + (i as f32) * 0.02).collect();
    let expected = host_kl_div_mean_f64(
        &host_inp_f32.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &host_t_f32.iter().map(|&v| v as f64).collect::<Vec<_>>(),
    ) as f32;
    let host_inp: Vec<bf16> = host_inp_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_t: Vec<bf16> = host_t_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let dev_inp = DeviceBuffer::from_slice(&ctx, &host_inp).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel * 2).unwrap();

    let desc = KlDivLossDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        element: ElementKind::Bf16,
    };
    let plan = KlDivLossPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        KlDivLossArgs {
            input: TensorRef {
                data: dev_inp.as_slice(),
                shape,
                stride: contiguous_stride(shape),
            },
            target: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1], stride: [1, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [bf16::ZERO; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let g = got[0].to_f32();
    let tol = expected.abs() * 16.0 * 7.81e-3_f32 + 2e-2;
    assert!((g - expected).abs() <= tol, "bf16 KLDiv: got={} want={}", g, expected);
}
