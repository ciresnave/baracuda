//! Real-GPU smoke for `NmsPlan<T>` (Phase 9 Category T).
//! Three boxes pre-sorted by score; box 1 overlaps box 0 above
//! threshold and box 2 is disjoint. Expected: keep_mask = [1, 0, 1].

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, NmsArgs, NmsDescriptor, NmsPlan, PlanPreference, TensorMut,
    TensorRef, Workspace,
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
fn nms_basic_overlap_f32() {
    let (ctx, stream) = setup();
    // 3 boxes in (x1, y1, x2, y2). All in decreasing-score order.
    //   box 0: (0, 0, 10, 10)   area = 100
    //   box 1: (1, 1, 11, 11)   area = 100, IoU(0, 1) = 81 / 119 ≈ 0.68
    //   box 2: (20, 20, 30, 30) disjoint with both → kept.
    let boxes: Vec<f32> = vec![
        0.0, 0.0, 10.0, 10.0,
        1.0, 1.0, 11.0, 11.0,
        20.0, 20.0, 30.0, 30.0,
    ];
    let num_boxes = 3;
    let dev_boxes = DeviceBuffer::from_slice(&ctx, &boxes).expect("");
    let mut dev_mask: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, num_boxes as usize).expect("");
    let mut dev_count: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("");
    let desc = NmsDescriptor { num_boxes, iou_threshold: 0.5, element: ElementKind::F32 };
    let plan = NmsPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("");
    plan.run(&stream, Workspace::None, NmsArgs {
        boxes: TensorRef {
            data: dev_boxes.as_slice(),
            shape: [num_boxes, 4],
            stride: contiguous_stride([num_boxes, 4]),
        },
        keep_mask: TensorMut {
            data: dev_mask.as_slice_mut(),
            shape: [num_boxes],
            stride: contiguous_stride([num_boxes]),
        },
        count: TensorMut {
            data: dev_count.as_slice_mut(),
            shape: [1],
            stride: contiguous_stride([1]),
        },
    }).expect("");
    stream.synchronize().expect("sync");
    let mut got = vec![0u8; 3];
    let mut got_count = vec![0i32; 1];
    dev_mask.copy_to_host(&mut got).expect("");
    dev_count.copy_to_host(&mut got_count).expect("");
    assert_eq!(got, vec![1, 0, 1], "nms mask mismatch: {got:?}");
    assert_eq!(got_count[0], 2);
}

#[test]
#[ignore]
fn nms_all_disjoint_f64() {
    let (ctx, stream) = setup();
    let boxes: Vec<f64> = vec![
        0.0, 0.0, 5.0, 5.0,
        10.0, 10.0, 15.0, 15.0,
        20.0, 20.0, 25.0, 25.0,
    ];
    let num_boxes = 3;
    let dev_boxes = DeviceBuffer::from_slice(&ctx, &boxes).expect("");
    let mut dev_mask: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, 3).expect("");
    let mut dev_count: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("");
    let desc = NmsDescriptor { num_boxes, iou_threshold: 0.5, element: ElementKind::F64 };
    let plan = NmsPlan::<f64>::select(&stream, &desc, PlanPreference::default()).expect("");
    plan.run(&stream, Workspace::None, NmsArgs {
        boxes: TensorRef {
            data: dev_boxes.as_slice(),
            shape: [num_boxes, 4],
            stride: contiguous_stride([num_boxes, 4]),
        },
        keep_mask: TensorMut {
            data: dev_mask.as_slice_mut(),
            shape: [num_boxes],
            stride: contiguous_stride([num_boxes]),
        },
        count: TensorMut {
            data: dev_count.as_slice_mut(),
            shape: [1],
            stride: contiguous_stride([1]),
        },
    }).expect("");
    stream.synchronize().expect("sync");
    let mut got = vec![0u8; 3];
    let mut got_count = vec![0i32; 1];
    dev_mask.copy_to_host(&mut got).expect("");
    dev_count.copy_to_host(&mut got_count).expect("");
    assert_eq!(got, vec![1, 1, 1]);
    assert_eq!(got_count[0], 3);
}
