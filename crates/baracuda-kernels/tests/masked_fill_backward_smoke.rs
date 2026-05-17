//! Real-GPU smoke test for `MaskedFillBackwardPlan<T, N>` (Phase 7 7.3).
//!
//! `dsrc[i] = mask[i] ? 0 : dout[i]`. f32, f64, i32, bool coverage.
//!
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, MaskedFillBackwardArgs, MaskedFillBackwardDescriptor,
    MaskedFillBackwardPlan, PlanPreference, TensorMut, TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn masked_fill_backward_f32_2d() {
    let (ctx, stream) = setup();
    let shape = [4i32, 6];
    let numel: usize = 4 * 6;
    let host_dout: Vec<f32> = (0..numel).map(|i| i as f32 * 0.5 + 1.0).collect();
    let host_mask: Vec<u8> = (0..numel).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();
    let expected: Vec<f32> = host_dout
        .iter()
        .zip(host_mask.iter())
        .map(|(d, m)| if *m != 0 { 0.0 } else { *d })
        .collect();
    let dev_dout = DeviceBuffer::from_slice(&ctx, &host_dout).expect("up dout");
    let dev_mask = DeviceBuffer::from_slice(&ctx, &host_mask).expect("up mask");
    let mut dev_dsrc: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dsrc");

    let desc = MaskedFillBackwardDescriptor {
        shape,
        element: ElementKind::F32,
    };
    let plan = MaskedFillBackwardPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = MaskedFillBackwardArgs::<f32, 2> {
        dout: TensorRef {
            data: dev_dout.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        mask: TensorRef {
            data: dev_mask.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        dsrc: TensorMut {
            data: dev_dsrc.as_slice_mut(),
            shape,
            stride: contiguous_stride(shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_dsrc.copy_to_host(&mut got).expect("dl");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "masked_fill_backward f32 mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn masked_fill_backward_f64_1d() {
    let (ctx, stream) = setup();
    let shape = [20i32];
    let numel: usize = 20;
    let host_dout: Vec<f64> = (0..numel).map(|i| i as f64 * 0.125 - 1.5).collect();
    let host_mask: Vec<u8> = (0..numel).map(|i| if i % 5 == 0 { 1 } else { 0 }).collect();
    let expected: Vec<f64> = host_dout
        .iter()
        .zip(host_mask.iter())
        .map(|(d, m)| if *m != 0 { 0.0 } else { *d })
        .collect();
    let dev_dout = DeviceBuffer::from_slice(&ctx, &host_dout).expect("up dout");
    let dev_mask = DeviceBuffer::from_slice(&ctx, &host_mask).expect("up mask");
    let mut dev_dsrc: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dsrc");

    let desc = MaskedFillBackwardDescriptor {
        shape,
        element: ElementKind::F64,
    };
    let plan = MaskedFillBackwardPlan::<f64, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = MaskedFillBackwardArgs::<f64, 1> {
        dout: TensorRef {
            data: dev_dout.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        mask: TensorRef {
            data: dev_mask.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        dsrc: TensorMut {
            data: dev_dsrc.as_slice_mut(),
            shape,
            stride: contiguous_stride(shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; numel];
    dev_dsrc.copy_to_host(&mut got).expect("dl");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "masked_fill_backward f64 mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn masked_fill_backward_i32_1d() {
    let (ctx, stream) = setup();
    let shape = [16i32];
    let numel: usize = 16;
    let host_dout: Vec<i32> = (0..numel as i32).map(|i| i.wrapping_mul(7) - 30).collect();
    let host_mask: Vec<u8> = (0..numel).map(|i| if i % 4 == 0 { 1 } else { 0 }).collect();
    let expected: Vec<i32> = host_dout
        .iter()
        .zip(host_mask.iter())
        .map(|(d, m)| if *m != 0 { 0 } else { *d })
        .collect();
    let dev_dout = DeviceBuffer::from_slice(&ctx, &host_dout).expect("up dout");
    let dev_mask = DeviceBuffer::from_slice(&ctx, &host_mask).expect("up mask");
    let mut dev_dsrc: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dsrc");

    let desc = MaskedFillBackwardDescriptor {
        shape,
        element: ElementKind::I32,
    };
    let plan = MaskedFillBackwardPlan::<i32, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = MaskedFillBackwardArgs::<i32, 1> {
        dout: TensorRef {
            data: dev_dout.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        mask: TensorRef {
            data: dev_mask.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        dsrc: TensorMut {
            data: dev_dsrc.as_slice_mut(),
            shape,
            stride: contiguous_stride(shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0i32; numel];
    dev_dsrc.copy_to_host(&mut got).expect("dl");
    assert_eq!(got, expected, "masked_fill_backward i32 mismatch");
}
