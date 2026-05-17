//! Real-GPU smoke test for `MaskedFillPlan<T, N>` (Phase 7 7.3).
//!
//! `out[i] = mask[i] ? value : src[i]`. f32, f64, i32, bool coverage.
//! Bit-exact (pure element-select).
//!
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, Bool, MaskedFillArgs, MaskedFillDescriptor, MaskedFillPlan,
    PlanPreference, TensorMut, TensorRef, Workspace,
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
fn masked_fill_f32_2d() {
    let (ctx, stream) = setup();
    let shape = [4i32, 6];
    let numel: usize = 4 * 6;
    let value: f32 = -123.456;
    let host_src: Vec<f32> = (0..numel).map(|i| i as f32 * 0.5 + 1.0).collect();
    let host_mask: Vec<u8> = (0..numel).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();
    let expected: Vec<f32> = host_src
        .iter()
        .zip(host_mask.iter())
        .map(|(s, m)| if *m != 0 { value } else { *s })
        .collect();
    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("up src");
    let dev_mask = DeviceBuffer::from_slice(&ctx, &host_mask).expect("up mask");
    let mut dev_out: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc out");

    let desc = MaskedFillDescriptor::new_f32(shape, value);
    let plan = MaskedFillPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = MaskedFillArgs::<f32, 2> {
        src: TensorRef {
            data: dev_src.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        mask: TensorRef {
            data: dev_mask.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        out: TensorMut {
            data: dev_out.as_slice_mut(),
            shape,
            stride: contiguous_stride(shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_out.copy_to_host(&mut got).expect("dl");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "masked_fill f32 mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn masked_fill_f64_2d() {
    let (ctx, stream) = setup();
    let shape = [3i32, 5];
    let numel: usize = 3 * 5;
    let value: f64 = 99.875;
    let host_src: Vec<f64> = (0..numel).map(|i| i as f64 * 0.25 - 2.0).collect();
    let host_mask: Vec<u8> = (0..numel).map(|i| if i % 2 == 1 { 1 } else { 0 }).collect();
    let expected: Vec<f64> = host_src
        .iter()
        .zip(host_mask.iter())
        .map(|(s, m)| if *m != 0 { value } else { *s })
        .collect();
    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("up src");
    let dev_mask = DeviceBuffer::from_slice(&ctx, &host_mask).expect("up mask");
    let mut dev_out: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc out");

    let desc = MaskedFillDescriptor::new_f64(shape, value);
    let plan = MaskedFillPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = MaskedFillArgs::<f64, 2> {
        src: TensorRef {
            data: dev_src.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        mask: TensorRef {
            data: dev_mask.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        out: TensorMut {
            data: dev_out.as_slice_mut(),
            shape,
            stride: contiguous_stride(shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; numel];
    dev_out.copy_to_host(&mut got).expect("dl");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "masked_fill f64 mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn masked_fill_i32_2d() {
    let (ctx, stream) = setup();
    let shape = [4i32, 4];
    let numel: usize = 4 * 4;
    let value: i32 = -777;
    let host_src: Vec<i32> = (0..numel as i32).map(|i| i.wrapping_mul(13) - 50).collect();
    let host_mask: Vec<u8> = (0..numel).map(|i| if i % 4 == 2 { 1 } else { 0 }).collect();
    let expected: Vec<i32> = host_src
        .iter()
        .zip(host_mask.iter())
        .map(|(s, m)| if *m != 0 { value } else { *s })
        .collect();
    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("up src");
    let dev_mask = DeviceBuffer::from_slice(&ctx, &host_mask).expect("up mask");
    let mut dev_out: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc out");

    let desc = MaskedFillDescriptor::new_i32(shape, value);
    let plan = MaskedFillPlan::<i32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = MaskedFillArgs::<i32, 2> {
        src: TensorRef {
            data: dev_src.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        mask: TensorRef {
            data: dev_mask.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        out: TensorMut {
            data: dev_out.as_slice_mut(),
            shape,
            stride: contiguous_stride(shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0i32; numel];
    dev_out.copy_to_host(&mut got).expect("dl");
    assert_eq!(got, expected, "masked_fill i32 mismatch");
}

#[test]
#[ignore]
fn masked_fill_bool_1d() {
    let (ctx, stream) = setup();
    let shape = [16i32];
    let numel: usize = 16;
    // Set value to `true` (= 1).
    let host_src: Vec<Bool> = (0..numel).map(|i| Bool((i % 2) as u8)).collect();
    let host_mask: Vec<u8> = (0..numel).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();
    let expected: Vec<Bool> = host_src
        .iter()
        .zip(host_mask.iter())
        .map(|(s, m)| if *m != 0 { Bool(1) } else { *s })
        .collect();
    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("up src");
    let dev_mask = DeviceBuffer::from_slice(&ctx, &host_mask).expect("up mask");
    let mut dev_out: DeviceBuffer<Bool> = DeviceBuffer::zeros(&ctx, numel).expect("alloc out");

    let desc = MaskedFillDescriptor::new_bool(shape, true);
    let plan = MaskedFillPlan::<Bool, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = MaskedFillArgs::<Bool, 1> {
        src: TensorRef {
            data: dev_src.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        mask: TensorRef {
            data: dev_mask.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        out: TensorMut {
            data: dev_out.as_slice_mut(),
            shape,
            stride: contiguous_stride(shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![Bool(0); numel];
    dev_out.copy_to_host(&mut got).expect("dl");
    assert_eq!(got, expected, "masked_fill bool mismatch");
}
