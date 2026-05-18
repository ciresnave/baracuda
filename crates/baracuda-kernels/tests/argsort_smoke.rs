//! Real-GPU smoke test for `ArgsortPlan<T>` (Phase 9 Category O).
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ArgsortArgs, ArgsortDescriptor, ArgsortPlan, ElementKind, PlanPreference,
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
fn argsort_f32_basic() {
    let (ctx, stream) = setup();
    let batch: i32 = 1;
    let row_len: i32 = 6;
    let input: Vec<f32> = vec![3.0, 1.0, 4.0, 1.5, 5.0, 9.0];
    let dev_in = DeviceBuffer::from_slice(&ctx, &input).expect("up");
    let mut dev_i: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, input.len()).expect("alloc i");

    let desc = ArgsortDescriptor {
        batch,
        row_len,
        descending: false,
        element: ElementKind::F32,
    };
    let plan =
        ArgsortPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("select");
    plan.run(
        &stream,
        Workspace::None,
        ArgsortArgs::<f32> {
            input: TensorRef {
                data: dev_in.as_slice(),
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

    let mut got_i = vec![0i32; input.len()];
    dev_i.copy_to_host(&mut got_i).expect("dl");
    // For 3, 1, 4, 1.5, 5, 9 ascending the index permutation is
    // 1, 3, 0, 2, 4, 5 (1.0 < 1.5 < 3 < 4 < 5 < 9).
    let expected: Vec<i32> = vec![1, 3, 0, 2, 4, 5];
    assert_eq!(got_i, expected);
}
