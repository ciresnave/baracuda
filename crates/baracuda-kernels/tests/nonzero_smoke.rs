//! Real-GPU smoke test for `NonzeroPlan<T, N>` (Phase 7 7.3).
//!
//! Verifies the *set* of nonzero coordinates is correct; output
//! ordering is NOT row-major (atomic-counter races), so the test sorts
//! the returned coords before comparing.
//!
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, NonzeroArgs, NonzeroDescriptor, NonzeroPlan, PlanPreference,
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
fn nonzero_f32_2d() {
    let (ctx, stream) = setup();
    let shape = [3i32, 4];
    let numel: usize = 3 * 4;
    let host_x: Vec<f32> = vec![
        0.0, 1.5, 0.0, -3.0,
        2.5, 0.0, 0.0, 4.0,
        0.0, 0.0, 7.25, 0.0,
    ];
    // Expected nonzero coords (row-major-sorted reference).
    let mut expected: Vec<(i32, i32)> = Vec::new();
    for i in 0..3i32 {
        for j in 0..4i32 {
            if host_x[(i as usize) * 4 + j as usize] != 0.0 {
                expected.push((i, j));
            }
        }
    }
    expected.sort();
    let max_nz = numel as i32;
    let rank: i32 = 2;
    let coords_len: usize = (max_nz as usize) * (rank as usize);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_coords: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, coords_len).expect("alloc coords");
    let mut dev_counter: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc counter");

    let desc = NonzeroDescriptor {
        shape,
        max_nz,
        element: ElementKind::F32,
    };
    let plan = NonzeroPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = NonzeroArgs::<f32, 2> {
        x: TensorRef {
            data: dev_x.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        out_coords: TensorMut {
            data: dev_coords.as_slice_mut(),
            shape: [coords_len as i32],
            stride: contiguous_stride([coords_len as i32]),
        },
        counter: TensorMut {
            data: dev_counter.as_slice_mut(),
            shape: [1i32],
            stride: contiguous_stride([1i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got_counter = vec![0i32; 1];
    dev_counter.copy_to_host(&mut got_counter).expect("dl ctr");
    let nnz = got_counter[0] as usize;
    assert_eq!(nnz, expected.len(), "nonzero count mismatch");

    let mut got_coords_flat = vec![0i32; coords_len];
    dev_coords.copy_to_host(&mut got_coords_flat).expect("dl coords");

    let mut got: Vec<(i32, i32)> = (0..nnz)
        .map(|s| (got_coords_flat[s * 2], got_coords_flat[s * 2 + 1]))
        .collect();
    got.sort();
    assert_eq!(got, expected, "nonzero coords mismatch");
}

#[test]
#[ignore]
fn nonzero_i32_1d() {
    let (ctx, stream) = setup();
    let shape = [16i32];
    let host_x: Vec<i32> = (0..16i32).map(|i| if i % 3 == 1 { i * 7 } else { 0 }).collect();
    let mut expected: Vec<i32> = Vec::new();
    for (i, v) in host_x.iter().enumerate() {
        if *v != 0 {
            expected.push(i as i32);
        }
    }
    expected.sort();

    let max_nz = host_x.len() as i32;
    let coords_len: usize = max_nz as usize;
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_coords: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, coords_len).expect("alloc coords");
    let mut dev_counter: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc counter");

    let desc = NonzeroDescriptor {
        shape,
        max_nz,
        element: ElementKind::I32,
    };
    let plan = NonzeroPlan::<i32, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = NonzeroArgs::<i32, 1> {
        x: TensorRef {
            data: dev_x.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        out_coords: TensorMut {
            data: dev_coords.as_slice_mut(),
            shape: [coords_len as i32],
            stride: contiguous_stride([coords_len as i32]),
        },
        counter: TensorMut {
            data: dev_counter.as_slice_mut(),
            shape: [1i32],
            stride: contiguous_stride([1i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got_counter = vec![0i32; 1];
    dev_counter.copy_to_host(&mut got_counter).expect("dl ctr");
    let nnz = got_counter[0] as usize;
    assert_eq!(nnz, expected.len(), "nonzero count mismatch");

    let mut got_coords_flat = vec![0i32; coords_len];
    dev_coords.copy_to_host(&mut got_coords_flat).expect("dl coords");
    let mut got: Vec<i32> = got_coords_flat[..nnz].to_vec();
    got.sort();
    assert_eq!(got, expected, "nonzero i32 1d mismatch");
}
