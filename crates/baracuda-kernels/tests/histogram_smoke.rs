//! Real-GPU smoke test for `HistogramPlan<T>` + `BincountPlan<T>`
//! (Phase 9 Category O). gated behind the `device-tests` feature.
#![cfg(feature = "device-tests")]
#![allow(
    clippy::needless_range_loop,
    reason = "these host reference models index several parallel buffers by one i; iterator rewrites would hide the element-wise correspondence with the kernel under test."
)]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    BincountArgs, BincountDescriptor, BincountPlan, ElementKind, HistogramArgs,
    HistogramDescriptor, HistogramPlan, PlanPreference, TensorMut, TensorRef, Workspace,
    contiguous_stride,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
fn histogram_f32_basic() {
    let (ctx, stream) = setup();
    let numel: i64 = 1000;
    let num_bins: i32 = 10;
    let lo: f64 = 0.0;
    let hi: f64 = 1.0;
    let input: Vec<f32> = (0..numel as usize)
        .map(|i| (i as f32) / (numel as f32))
        .collect();
    let dev_in = DeviceBuffer::from_slice(&ctx, &input).expect("up");
    let mut dev_out: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, num_bins as usize).expect("alloc");

    let desc = HistogramDescriptor {
        numel,
        num_bins,
        lo,
        hi,
        element: ElementKind::F32,
    };
    let plan =
        HistogramPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("select");
    plan.run(
        &stream,
        Workspace::None,
        HistogramArgs::<f32> {
            input: TensorRef::new(
                dev_in.as_slice(),
                [numel as i32],
                contiguous_stride([numel as i32]),
            )
            .unwrap(),
            output: TensorMut::new(
                dev_out.as_slice_mut(),
                [num_bins],
                contiguous_stride([num_bins]),
            )
            .unwrap(),
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0i32; num_bins as usize];
    dev_out.copy_to_host(&mut got).expect("dl");

    // Reference: uniform inputs in [0, 1) → ~100 per bin (with last
    // bin getting the i = numel-1 == 0.999 cell).
    let total: i32 = got.iter().sum();
    assert_eq!(total, numel as i32, "histogram total != numel");
    // Inputs are exactly i/1000 over 10 bins, so every bin holds 100 cells; f32 rounding
    // of i/1000 at a bin edge moves at most one cell across it. Measured on sm_89:
    // [100, 100, 100, 100, 100, 100, 101, 99, 101, 99]. Allow +-1 (was +-5, which a
    // five-cell bin-boundary shift would have passed).
    for b in 0..num_bins as usize {
        assert!(
            (got[b] - 100).abs() <= 1,
            "histogram bin {b} count {} too far from 100",
            got[b]
        );
    }
}

#[test]
fn bincount_i32_basic() {
    let (ctx, stream) = setup();
    let input: Vec<i32> = vec![0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 5];
    let num_bins: i32 = 6;
    let dev_in = DeviceBuffer::from_slice(&ctx, &input).expect("up");
    let mut dev_out: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, num_bins as usize).expect("alloc");

    let desc = BincountDescriptor {
        numel: input.len() as i64,
        num_bins,
        element: ElementKind::I32,
    };
    let plan =
        BincountPlan::<i32>::select(&stream, &desc, PlanPreference::default()).expect("select");
    plan.run(
        &stream,
        Workspace::None,
        BincountArgs::<i32> {
            input: TensorRef::new(
                dev_in.as_slice(),
                [input.len() as i32],
                contiguous_stride([input.len() as i32]),
            )
            .unwrap(),
            output: TensorMut::new(
                dev_out.as_slice_mut(),
                [num_bins],
                contiguous_stride([num_bins]),
            )
            .unwrap(),
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0i32; num_bins as usize];
    dev_out.copy_to_host(&mut got).expect("dl");
    let expected: Vec<i32> = vec![1, 2, 3, 4, 0, 1];
    assert_eq!(got, expected);
}
