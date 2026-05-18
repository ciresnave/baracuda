//! Real-GPU smoke test for `SortBackwardPlan<T>` (Phase 9 Category O).
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, SortArgs, SortBackwardArgs,
    SortBackwardDescriptor, SortBackwardPlan, SortDescriptor, SortPlan, TensorMut, TensorRef,
    Workspace,
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
fn sort_backward_f32_basic() {
    let (ctx, stream) = setup();
    let batch: i32 = 2;
    let row_len: i32 = 6;
    let input: Vec<f32> = vec![3.0, 1.0, 4.0, 1.5, 5.0, 9.0,
                                2.0, 6.0, 5.0, 3.5, 5.5, 1.0];
    let dev_in = DeviceBuffer::from_slice(&ctx, &input).expect("up");
    let mut dev_vals: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, input.len()).expect("alloc v");
    let mut dev_idx: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, input.len()).expect("alloc i");

    let desc = SortDescriptor {
        batch,
        row_len,
        descending: false,
        element: ElementKind::F32,
    };
    let plan = SortPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("select");
    plan.run(
        &stream,
        Workspace::None,
        SortArgs::<f32> {
            input: TensorRef {
                data: dev_in.as_slice(),
                shape: [batch, row_len],
                stride: contiguous_stride([batch, row_len]),
            },
            values: TensorMut {
                data: dev_vals.as_slice_mut(),
                shape: [batch, row_len],
                stride: contiguous_stride([batch, row_len]),
            },
            indices: TensorMut {
                data: dev_idx.as_slice_mut(),
                shape: [batch, row_len],
                stride: contiguous_stride([batch, row_len]),
            },
        },
    )
    .expect("fw");
    stream.synchronize().expect("sync");

    // BW: dy[i] = i + 1 so the test detects misrouting.
    let dy_host: Vec<f32> = (0..(batch * row_len) as i32)
        .map(|i| (i + 1) as f32)
        .collect();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).expect("up dy");
    let mut dev_dx: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, dy_host.len()).expect("alloc dx");

    let bw_desc = SortBackwardDescriptor {
        batch,
        row_len,
        element: ElementKind::F32,
    };
    let bw_plan = SortBackwardPlan::<f32>::select(&stream, &bw_desc, PlanPreference::default())
        .expect("select bw");
    bw_plan
        .run(
            &stream,
            Workspace::None,
            SortBackwardArgs::<f32> {
                dy: TensorRef {
                    data: dev_dy.as_slice(),
                    shape: [batch, row_len],
                    stride: contiguous_stride([batch, row_len]),
                },
                indices: TensorRef {
                    data: dev_idx.as_slice(),
                    shape: [batch, row_len],
                    stride: contiguous_stride([batch, row_len]),
                },
                dx: TensorMut {
                    data: dev_dx.as_slice_mut(),
                    shape: [batch, row_len],
                    stride: contiguous_stride([batch, row_len]),
                },
            },
        )
        .expect("bw");
    stream.synchronize().expect("sync2");

    let mut got_idx = vec![0i32; (batch * row_len) as usize];
    dev_idx.copy_to_host(&mut got_idx).expect("dl idx");
    let mut got_dx = vec![0f32; (batch * row_len) as usize];
    dev_dx.copy_to_host(&mut got_dx).expect("dl dx");

    // Reference: for each row, dx[indices[i]] = dy[i].
    let mut expected = vec![0f32; (batch * row_len) as usize];
    for row in 0..batch as usize {
        let off = row * row_len as usize;
        for i in 0..row_len as usize {
            let src = got_idx[off + i] as usize;
            expected[off + src] = dy_host[off + i];
        }
    }
    for i in 0..expected.len() {
        assert!(
            (got_dx[i] - expected[i]).abs() < 1e-6,
            "sort BW mismatch @ {i}: got {} expected {}",
            got_dx[i],
            expected[i]
        );
    }
}
