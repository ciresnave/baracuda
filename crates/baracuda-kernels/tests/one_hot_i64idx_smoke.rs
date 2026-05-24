//! Real-GPU smoke test for the i64 class-index variant of
//! `OneHotPlan<T, N>` (Phase 15.2).
//!
//! Phase 11.5 shipped the `_i64idx_` FFI symbols; Phase 15.2 wired the
//! Rust plan wrapper to be generic over `I: IndexElement` (default
//! `i32` for source-compat). This file covers the new i64 path.
//!
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, OneHotArgs, OneHotDescriptor, OneHotPlan, PlanPreference,
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
fn one_hot_f32_i64idx_2d() {
    let (ctx, stream) = setup();
    let num_classes = 5i32;
    // 4 batch items; output shape is [4, 5]. Class indices are i64
    // (PyTorch's default integer dtype).
    let host_src: Vec<i64> = vec![0, 4, 2, 3];
    let out_shape = [4i32, num_classes];
    let out_numel: usize = 4 * num_classes as usize;
    let mut expected = vec![0f32; out_numel];
    for (i, &c) in host_src.iter().enumerate() {
        if c >= 0 && c < num_classes as i64 {
            expected[i * num_classes as usize + c as usize] = 1.0;
        }
    }
    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("up src");
    let mut dev_out: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc out");

    let desc = OneHotDescriptor {
        out_shape,
        num_classes,
        element: ElementKind::F32,
    };
    let plan = OneHotPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    // Explicit `I = i64` opts into the new path.
    let args = OneHotArgs::<f32, 2, i64> {
        src: TensorRef {
            data: dev_src.as_slice(),
            shape: [4i32],
            stride: contiguous_stride([4i32]),
        },
        out: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: out_shape,
            stride: contiguous_stride(out_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; out_numel];
    dev_out.copy_to_host(&mut got).expect("dl");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "one_hot f32 i64idx mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn one_hot_i32_i64idx_2d_with_oob() {
    let (ctx, stream) = setup();
    let num_classes = 4i32;
    // Negative + out-of-range src values yield all-zero rows. Negative
    // values exercise the i64 sign-extension path.
    let host_src: Vec<i64> = vec![2, -1, 0, 3, 7];
    let out_shape = [5i32, num_classes];
    let out_numel: usize = 5 * num_classes as usize;
    let mut expected = vec![0i32; out_numel];
    for (i, &c) in host_src.iter().enumerate() {
        if c >= 0 && c < num_classes as i64 {
            expected[i * num_classes as usize + c as usize] = 1;
        }
    }
    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("up src");
    let mut dev_out: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc out");

    let desc = OneHotDescriptor {
        out_shape,
        num_classes,
        element: ElementKind::I32,
    };
    let plan = OneHotPlan::<i32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = OneHotArgs::<i32, 2, i64> {
        src: TensorRef {
            data: dev_src.as_slice(),
            shape: [5i32],
            stride: contiguous_stride([5i32]),
        },
        out: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: out_shape,
            stride: contiguous_stride(out_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0i32; out_numel];
    dev_out.copy_to_host(&mut got).expect("dl");
    assert_eq!(got, expected, "one_hot i32 i64idx mismatch");
}

/// Regression guard: explicit `I = i32` (or the default) still picks
/// the legacy FFI symbol — Phase 15.2 must not break pre-existing
/// call sites.
#[test]
#[ignore]
fn one_hot_f32_i32idx_default_path() {
    let (ctx, stream) = setup();
    let num_classes = 3i32;
    let host_src: Vec<i32> = vec![1, 0, 2];
    let out_shape = [3i32, num_classes];
    let out_numel: usize = 3 * num_classes as usize;
    let mut expected = vec![0f32; out_numel];
    for (i, &c) in host_src.iter().enumerate() {
        if c >= 0 && c < num_classes {
            expected[i * num_classes as usize + c as usize] = 1.0;
        }
    }
    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("up src");
    let mut dev_out: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc out");

    let desc = OneHotDescriptor {
        out_shape,
        num_classes,
        element: ElementKind::F32,
    };
    let plan = OneHotPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    // No explicit `I` — relies on the `I = i32` default. This is what
    // every pre-Phase-15.2 caller looks like.
    let args = OneHotArgs::<f32, 2> {
        src: TensorRef {
            data: dev_src.as_slice(),
            shape: [3i32],
            stride: contiguous_stride([3i32]),
        },
        out: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: out_shape,
            stride: contiguous_stride(out_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; out_numel];
    dev_out.copy_to_host(&mut got).expect("dl");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "one_hot f32 default-i32 path mismatch @ {i}");
    }
}
