//! Real-GPU smoke test for `SortPlan<T>` (Phase 9 Category O).
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, SortArgs, SortDescriptor, SortPlan, TensorMut,
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
fn sort_f32_ascending_basic() {
    let (ctx, stream) = setup();
    let batch: i32 = 3;
    let row_len: i32 = 8;
    // Row 0: simple shuffled. Row 1: descending input. Row 2: ties.
    let input: Vec<f32> = vec![
        5.0, 1.0, 4.0, 2.0, 8.0, 3.0, 7.0, 6.0,
        8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        2.0, 2.0, 1.0, 3.0, 1.0, 3.0, 2.0, 1.0,
    ];
    let dev_in = DeviceBuffer::from_slice(&ctx, &input).expect("up");
    let mut dev_vals: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (batch * row_len) as usize).expect("alloc vals");
    let mut dev_idx: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, (batch * row_len) as usize).expect("alloc idx");

    let desc = SortDescriptor {
        batch,
        row_len,
        descending: false,
        element: ElementKind::F32,
    };
    let plan = SortPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = SortArgs::<f32> {
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
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got_v = vec![0f32; input.len()];
    let mut got_i = vec![0i32; input.len()];
    dev_vals.copy_to_host(&mut got_v).expect("dl v");
    dev_idx.copy_to_host(&mut got_i).expect("dl i");

    for row in 0..batch as usize {
        let off = row * row_len as usize;
        let mut ref_pairs: Vec<(f32, i32)> = (0..row_len as usize)
            .map(|i| (input[off + i], i as i32))
            .collect();
        ref_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for i in 0..row_len as usize {
            // Values must match exactly (sorted permutation).
            assert!(
                (got_v[off + i] - ref_pairs[i].0).abs() < 1e-6,
                "sort row {row} val[{i}] got {} expected {}",
                got_v[off + i],
                ref_pairs[i].0
            );
            // Index must point to an input cell whose value equals
            // ref_pairs[i].0 (ties allow multiple valid indices in
            // non-stable sort).
            let idx = got_i[off + i] as usize;
            assert!(idx < row_len as usize, "idx OOB");
            assert!(
                (input[off + idx] - ref_pairs[i].0).abs() < 1e-6,
                "sort row {row} idx[{i}]={idx} points to {} but ref val is {}",
                input[off + idx],
                ref_pairs[i].0
            );
        }
    }
}

#[test]
#[ignore]
fn sort_i32_descending_basic() {
    let (ctx, stream) = setup();
    let batch: i32 = 2;
    let row_len: i32 = 5;
    let input: Vec<i32> = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3];
    let dev_in = DeviceBuffer::from_slice(&ctx, &input).expect("up");
    let mut dev_vals: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, (batch * row_len) as usize).expect("alloc v");
    let mut dev_idx: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, (batch * row_len) as usize).expect("alloc i");

    let desc = SortDescriptor {
        batch,
        row_len,
        descending: true,
        element: ElementKind::I32,
    };
    let plan = SortPlan::<i32>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = SortArgs::<i32> {
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
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got_v = vec![0i32; input.len()];
    dev_vals.copy_to_host(&mut got_v).expect("dl v");
    for row in 0..batch as usize {
        let off = row * row_len as usize;
        let mut ref_vals: Vec<i32> = input[off..off + row_len as usize].to_vec();
        ref_vals.sort_by(|a, b| b.cmp(a));
        for i in 0..row_len as usize {
            assert_eq!(
                got_v[off + i], ref_vals[i],
                "sort i32 descending row {row} val[{i}]"
            );
        }
    }
}
