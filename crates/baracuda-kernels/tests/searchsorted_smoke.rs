//! Real-GPU smoke test for `SearchsortedPlan<T>` (Phase 9 Category O).
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, SearchsortedArgs, SearchsortedDescriptor,
    SearchsortedPlan, TensorMut, TensorRef, Workspace,
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
fn searchsorted_f32_lower_bound() {
    let (ctx, stream) = setup();
    let sorted: Vec<f32> = vec![1.0, 3.0, 5.0, 7.0, 9.0];
    let queries: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 9.0, 10.0];
    let dev_seq = DeviceBuffer::from_slice(&ctx, &sorted).expect("up seq");
    let dev_q = DeviceBuffer::from_slice(&ctx, &queries).expect("up q");
    let mut dev_out: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, queries.len()).expect("alloc out");

    let desc = SearchsortedDescriptor {
        num_queries: queries.len() as i64,
        len_sorted: sorted.len() as i32,
        right: false,
        element: ElementKind::F32,
    };
    let plan = SearchsortedPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    plan.run(
        &stream,
        Workspace::None,
        SearchsortedArgs::<f32> {
            sorted_seq: TensorRef {
                data: dev_seq.as_slice(),
                shape: [sorted.len() as i32],
                stride: contiguous_stride([sorted.len() as i32]),
            },
            values: TensorRef {
                data: dev_q.as_slice(),
                shape: [queries.len() as i32],
                stride: contiguous_stride([queries.len() as i32]),
            },
            output: TensorMut {
                data: dev_out.as_slice_mut(),
                shape: [queries.len() as i32],
                stride: contiguous_stride([queries.len() as i32]),
            },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0i32; queries.len()];
    dev_out.copy_to_host(&mut got).expect("dl");
    // Reference lower-bounds: positions of each query in `sorted`.
    let expected: Vec<i32> = vec![0, 0, 1, 1, 2, 4, 5];
    assert_eq!(got, expected);
}
