//! Real-GPU smoke test for the i64 output-coords variant of
//! `NonzeroPlan<T, N>` (Phase 15.2).
//!
//! Phase 11.5 shipped the `_i64idx_` FFI symbols (i64 output coords +
//! counter); Phase 15.2 wired the Rust plan wrapper to be generic over
//! `I: IndexElement` (default `i32` for source-compat). This file
//! covers the new i64 output path.
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
fn nonzero_f32_i64idx_2d() {
    let (ctx, stream) = setup();
    let shape = [3i32, 4];
    let numel: usize = 3 * 4;
    let host_x: Vec<f32> = vec![
        0.0, 1.5, 0.0, -3.0,
        2.5, 0.0, 0.0, 4.0,
        0.0, 0.0, 7.25, 0.0,
    ];
    let mut expected: Vec<(i64, i64)> = Vec::new();
    for i in 0..3i64 {
        for j in 0..4i64 {
            if host_x[(i as usize) * 4 + (j as usize)] != 0.0 {
                expected.push((i, j));
            }
        }
    }
    expected.sort();
    let max_nz = numel as i32;
    let rank: i32 = 2;
    let coords_len: usize = (max_nz as usize) * (rank as usize);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    // i64 output coord table + i64 counter.
    let mut dev_coords: DeviceBuffer<i64> =
        DeviceBuffer::zeros(&ctx, coords_len).expect("alloc coords");
    let mut dev_counter: DeviceBuffer<i64> = DeviceBuffer::zeros(&ctx, 1).expect("alloc counter");

    let desc = NonzeroDescriptor {
        shape,
        max_nz,
        element: ElementKind::F32,
    };
    let plan = NonzeroPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    // Explicit `I = i64` opts into the new path.
    let args = NonzeroArgs::<f32, 2, i64> {
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

    let mut got_counter = vec![0i64; 1];
    dev_counter.copy_to_host(&mut got_counter).expect("dl ctr");
    let nnz = got_counter[0] as usize;
    assert_eq!(nnz, expected.len(), "nonzero i64idx count mismatch");

    let mut got_coords_flat = vec![0i64; coords_len];
    dev_coords.copy_to_host(&mut got_coords_flat).expect("dl coords");

    let mut got: Vec<(i64, i64)> = (0..nnz)
        .map(|s| (got_coords_flat[s * 2], got_coords_flat[s * 2 + 1]))
        .collect();
    got.sort();
    assert_eq!(got, expected, "nonzero i64idx coords mismatch");
}

#[test]
#[ignore]
fn nonzero_bool_i64idx_1d() {
    let (ctx, stream) = setup();
    let shape = [20i32];
    use baracuda_kernels::Bool;
    let host_x: Vec<Bool> = (0..20i32)
        .map(|i| Bool::new(i % 4 == 2 || i == 13))
        .collect();
    let mut expected: Vec<i64> = Vec::new();
    for (i, v) in host_x.iter().enumerate() {
        if v.to_bool() {
            expected.push(i as i64);
        }
    }
    expected.sort();

    let max_nz = host_x.len() as i32;
    let coords_len: usize = max_nz as usize;
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_coords: DeviceBuffer<i64> =
        DeviceBuffer::zeros(&ctx, coords_len).expect("alloc coords");
    let mut dev_counter: DeviceBuffer<i64> = DeviceBuffer::zeros(&ctx, 1).expect("alloc counter");

    let desc = NonzeroDescriptor {
        shape,
        max_nz,
        element: ElementKind::Bool,
    };
    let plan = NonzeroPlan::<Bool, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = NonzeroArgs::<Bool, 1, i64> {
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

    let mut got_counter = vec![0i64; 1];
    dev_counter.copy_to_host(&mut got_counter).expect("dl ctr");
    let nnz = got_counter[0] as usize;
    assert_eq!(nnz, expected.len(), "nonzero bool i64idx count mismatch");

    let mut got_coords_flat = vec![0i64; coords_len];
    dev_coords.copy_to_host(&mut got_coords_flat).expect("dl coords");
    let mut got: Vec<i64> = got_coords_flat[..nnz].to_vec();
    got.sort();
    assert_eq!(got, expected, "nonzero bool i64idx 1d mismatch");
}

/// Regression guard: explicit `I = i32` (or the default) still picks
/// the legacy FFI symbol — Phase 15.2 must not break pre-existing
/// call sites.
#[test]
#[ignore]
fn nonzero_f32_i32idx_default_path() {
    let (ctx, stream) = setup();
    let shape = [3i32, 3];
    let host_x: Vec<f32> = vec![
        0.0, 1.0, 0.0,
        2.0, 0.0, 3.0,
        0.0, 4.0, 0.0,
    ];
    let mut expected: Vec<(i32, i32)> = Vec::new();
    for i in 0..3i32 {
        for j in 0..3i32 {
            if host_x[(i as usize) * 3 + (j as usize)] != 0.0 {
                expected.push((i, j));
            }
        }
    }
    expected.sort();
    let max_nz = 9i32;
    let coords_len: usize = (max_nz as usize) * 2;

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
    // No explicit `I` — relies on the `I = i32` default. This is what
    // every pre-Phase-15.2 caller looks like.
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
    assert_eq!(nnz, expected.len(), "nonzero default-i32 path count mismatch");

    let mut got_coords_flat = vec![0i32; coords_len];
    dev_coords.copy_to_host(&mut got_coords_flat).expect("dl coords");
    let mut got: Vec<(i32, i32)> = (0..nnz)
        .map(|s| (got_coords_flat[s * 2], got_coords_flat[s * 2 + 1]))
        .collect();
    got.sort();
    assert_eq!(got, expected, "nonzero default-i32 path coords mismatch");
}
