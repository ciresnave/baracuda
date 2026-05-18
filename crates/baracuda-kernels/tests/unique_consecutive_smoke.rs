//! Real-GPU smoke test for `UniqueConsecutivePlan<T>` (Phase 9
//! Category O). `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef,
    UniqueConsecutiveArgs, UniqueConsecutiveDescriptor, UniqueConsecutivePlan, Workspace,
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
fn unique_consecutive_i32_basic() {
    let (ctx, stream) = setup();
    let batch: i32 = 2;
    let row_len: i32 = 8;
    let max_unique: i32 = 8;
    // Row 0 has 3 runs (1, 2, 3); row 1 has 4 runs (5, 6, 7, 8).
    let input: Vec<i32> = vec![1, 1, 1, 2, 2, 3, 3, 3, 5, 5, 6, 6, 7, 7, 8, 8];
    let dev_in = DeviceBuffer::from_slice(&ctx, &input).expect("up");
    let mut dev_v: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, (batch * max_unique) as usize).expect("alloc v");
    let mut dev_c: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, (batch * max_unique) as usize).expect("alloc c");
    let mut dev_n: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, batch as usize).expect("alloc n");

    let desc = UniqueConsecutiveDescriptor {
        batch,
        row_len,
        max_unique,
        return_counts: true,
        element: ElementKind::I32,
    };
    let plan = UniqueConsecutivePlan::<i32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    plan.run(
        &stream,
        Workspace::None,
        UniqueConsecutiveArgs::<i32> {
            input: TensorRef {
                data: dev_in.as_slice(),
                shape: [batch, row_len],
                stride: contiguous_stride([batch, row_len]),
            },
            values: TensorMut {
                data: dev_v.as_slice_mut(),
                shape: [batch, max_unique],
                stride: contiguous_stride([batch, max_unique]),
            },
            counts: TensorMut {
                data: dev_c.as_slice_mut(),
                shape: [batch, max_unique],
                stride: contiguous_stride([batch, max_unique]),
            },
            counter: TensorMut {
                data: dev_n.as_slice_mut(),
                shape: [batch],
                stride: contiguous_stride([batch]),
            },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got_v = vec![0i32; (batch * max_unique) as usize];
    let mut got_n = vec![0i32; batch as usize];
    dev_v.copy_to_host(&mut got_v).expect("dl v");
    dev_n.copy_to_host(&mut got_n).expect("dl n");

    // Row 0: 3 distinct values (1, 2, 3). Row 1: 4 distinct values.
    assert_eq!(got_n[0], 3, "row 0 unique count");
    assert_eq!(got_n[1], 4, "row 1 unique count");

    // The slot order is atomic-race (not input-order), so just check
    // SET membership.
    let mut set0: Vec<i32> = got_v[..got_n[0] as usize].to_vec();
    set0.sort();
    assert_eq!(set0, vec![1, 2, 3]);
    let row1_off = max_unique as usize;
    let mut set1: Vec<i32> = got_v[row1_off..row1_off + got_n[1] as usize].to_vec();
    set1.sort();
    assert_eq!(set1, vec![5, 6, 7, 8]);
}
