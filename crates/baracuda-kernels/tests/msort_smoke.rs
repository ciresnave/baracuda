//! Real-GPU smoke test for `MsortPlan<T>` — stable sort with tie-break
//! on input index (Phase 9 Category O). `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, MsortArgs, MsortDescriptor, MsortPlan, PlanPreference,
    TensorMut, TensorRef, Workspace,
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
fn msort_f32_ties_stable() {
    let (ctx, stream) = setup();
    let batch: i32 = 1;
    let row_len: i32 = 8;
    // 4 distinct values, each duplicated. Stable ascending sort must
    // emit indices in original-position order for ties.
    let input: Vec<f32> = vec![2.0, 1.0, 2.0, 1.0, 3.0, 3.0, 1.0, 2.0];
    let dev_in = DeviceBuffer::from_slice(&ctx, &input).expect("up");
    let mut dev_v: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, input.len()).expect("alloc v");
    let mut dev_i: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, input.len()).expect("alloc i");

    let desc = MsortDescriptor {
        batch,
        row_len,
        descending: false,
        element: ElementKind::F32,
    };
    let plan = MsortPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("select");
    plan.run(
        &stream,
        Workspace::None,
        MsortArgs::<f32> {
            input: TensorRef {
                data: dev_in.as_slice(),
                shape: [batch, row_len],
                stride: contiguous_stride([batch, row_len]),
            },
            values: TensorMut {
                data: dev_v.as_slice_mut(),
                shape: [batch, row_len],
                stride: contiguous_stride([batch, row_len]),
            },
            indices: TensorMut {
                data: dev_i.as_slice_mut(),
                shape: [batch, row_len],
                stride: contiguous_stride([batch, row_len]),
            },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got_v = vec![0f32; input.len()];
    let mut got_i = vec![0i32; input.len()];
    dev_v.copy_to_host(&mut got_v).expect("dl v");
    dev_i.copy_to_host(&mut got_i).expect("dl i");

    // Stable ascending: 1@1, 1@3, 1@6, 2@0, 2@2, 2@7, 3@4, 3@5.
    let expected_v: Vec<f32> = vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0];
    let expected_i: Vec<i32> = vec![1, 3, 6, 0, 2, 7, 4, 5];
    for i in 0..input.len() {
        assert!(
            (got_v[i] - expected_v[i]).abs() < 1e-6,
            "msort v[{i}]: got {} expected {}",
            got_v[i],
            expected_v[i]
        );
        assert_eq!(
            got_i[i], expected_i[i],
            "msort i[{i}]: got {} expected {}",
            got_i[i], expected_i[i]
        );
    }
}
