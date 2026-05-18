//! Real-GPU smoke test for `HistogramPlan<T>` + `BincountPlan<T>`
//! (Phase 9 Category O). `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BincountArgs, BincountDescriptor, BincountPlan, ElementKind, HistogramArgs,
    HistogramDescriptor, HistogramPlan, PlanPreference, TensorMut, TensorRef, Workspace,
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
            input: TensorRef {
                data: dev_in.as_slice(),
                shape: [numel as i32],
                stride: contiguous_stride([numel as i32]),
            },
            output: TensorMut {
                data: dev_out.as_slice_mut(),
                shape: [num_bins],
                stride: contiguous_stride([num_bins]),
            },
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
    // Each bin should hold ~100 cells (allow ±5).
    for b in 0..num_bins as usize {
        assert!(
            (got[b] - 100).abs() <= 5,
            "histogram bin {b} count {} too far from 100",
            got[b]
        );
    }
}

#[test]
#[ignore]
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
            input: TensorRef {
                data: dev_in.as_slice(),
                shape: [input.len() as i32],
                stride: contiguous_stride([input.len() as i32]),
            },
            output: TensorMut {
                data: dev_out.as_slice_mut(),
                shape: [num_bins],
                stride: contiguous_stride([num_bins]),
            },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0i32; num_bins as usize];
    dev_out.copy_to_host(&mut got).expect("dl");
    let expected: Vec<i32> = vec![1, 2, 3, 4, 0, 1];
    assert_eq!(got, expected);
}
